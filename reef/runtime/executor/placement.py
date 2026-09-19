"""Reef-owned GPU reservations for one model deployment on Ray.

A deployment reserves every model GPU in one placement group so training and
inference cannot be scheduled apart, then hands each component the ordered
bundles it may use. Bundles are ordered by node address and device id, so a
tensor-parallel engine or a Megatron group always lands on contiguous devices
of one node. Colocated deployments give both components the same bundles;
the runtime's memory handoffs make that safe.

Ray is imported lazily so the base installation stays CPU-only.
"""

from __future__ import annotations

import logging
import socket
import time
from dataclasses import dataclass
from typing import Any, NamedTuple

logger = logging.getLogger(__name__)

#: How often a reservation that cannot be placed yet logs the cluster's GPU counts.
WAIT_LOG_INTERVAL_S = 30.0


class GpuBundles(NamedTuple):
    """A placement group with the ordered bundles one component schedules on.

    Unpacks as ``(group, bundle_indices, gpu_ids)``: the shape Reef's SGLang
    engine groups and Slime's training launcher schedule their actors with.
    """

    group: Any
    bundle_indices: list[int]
    gpu_ids: list[int]

    def slice(self, start: int, stop: int | None = None) -> GpuBundles:
        return GpuBundles(self.group, self.bundle_indices[start:stop], self.gpu_ids[start:stop])


@dataclass(frozen=True)
class ModelGpuLayout:
    """How many GPUs one deployment reserves and where inference's share starts."""

    training_gpus: int
    inference_gpus: int
    colocate: bool = False
    external_inference: bool = False

    def __post_init__(self) -> None:
        # Zero training GPUs is a hosted trainer: the reservation then holds inference alone.
        if isinstance(self.training_gpus, bool) or not isinstance(self.training_gpus, int) or self.training_gpus < 0:
            raise ValueError("a model deployment needs a non-negative number of training GPUs")
        if self.training_gpus == 0 and (self.colocate or self.external_inference):
            raise ValueError("a hosted trainer has no training GPUs to colocate or to pair with external engines")
        if (
            isinstance(self.inference_gpus, bool)
            or not isinstance(self.inference_gpus, int)
            or self.inference_gpus < 0
            or (self.inference_gpus == 0 and not self.external_inference)
        ):
            raise ValueError(
                "a model deployment needs a positive number of inference GPUs unless engines are external"
            )

    @property
    def total(self) -> int:
        """GPUs in the shared placement group."""
        if self.external_inference:
            return self.training_gpus
        if self.colocate:
            return max(self.training_gpus, self.inference_gpus)
        return self.training_gpus + self.inference_gpus

    @property
    def inference_offset(self) -> int:
        """First bundle inference may use; equal to ``total`` when it gets none."""
        if self.external_inference:
            return self.training_gpus
        return 0 if self.colocate else self.training_gpus


class ModelGpuReservation:
    """One deployment's placement group, sliced for its training and inference components."""

    def __init__(self, bundles: GpuBundles, layout: ModelGpuLayout) -> None:
        if len(bundles.bundle_indices) != layout.total:
            raise ValueError("reserved bundles do not match the model GPU layout")
        self._bundles = bundles
        self.layout = layout
        self._released = False

    @property
    def training(self) -> GpuBundles:
        return self._bundles.slice(0, self.layout.training_gpus)

    @property
    def inference(self) -> GpuBundles:
        return self._bundles.slice(self.layout.inference_offset)

    def release(self) -> None:
        """Return the placement group to the cluster; safe to call more than once."""
        if self._released:
            return
        self._released = True
        from ray.util.placement_group import remove_placement_group

        remove_placement_group(self._bundles.group)


def reserve_model_gpus(
    layout: ModelGpuLayout, *, wait_log_interval_s: float = WAIT_LOG_INTERVAL_S
) -> ModelGpuReservation:
    """Reserve ``layout.total`` GPUs in one placement group and order its bundles.

    The wait for placement is unbounded: on an autoscaling cluster the
    pending group is what drives scale-up. Progress is logged so a group that
    can never be placed is visible rather than a silent hang.
    """
    import ray
    from ray.util.placement_group import placement_group

    group = placement_group([{"GPU": 1, "CPU": 1} for _ in range(layout.total)], strategy="PACK")
    _wait_until_placed(ray, group, layout.total, wait_log_interval_s)
    identities = _bundle_identities(ray, group, layout.total)
    order = sorted(range(layout.total), key=lambda index: _bundle_sort_key(*identities[index]))
    for position, index in enumerate(order):
        node, gpu = identities[index]
        logger.info("bundle %4d: placement bundle %4d, node %s, gpu %s", position, index, node, gpu)
    bundles = GpuBundles(group, order, [identities[index][1] for index in order])
    return ModelGpuReservation(bundles, layout)


def _wait_until_placed(ray: Any, group: Any, count: int, log_interval_s: float) -> None:
    ready = group.ready()
    started = time.monotonic()
    while not ray.wait([ready], timeout=log_interval_s)[0]:
        logger.info(
            "waiting for a placement group of %d GPUs (%.0fs): %g GPUs registered with Ray, %g available",
            count,
            time.monotonic() - started,
            ray.cluster_resources().get("GPU", 0),
            ray.available_resources().get("GPU", 0),
        )


class _BundleProbe:
    """Runs inside each bundle to report which node and device it received."""

    def identity(self) -> tuple[str, int]:
        import ray

        return ray.util.get_node_ip_address(), int(ray.get_gpu_ids()[0])


def _bundle_identities(ray: Any, group: Any, count: int) -> list[tuple[str, int]]:
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    probe = ray.remote(num_gpus=1)(_BundleProbe)
    actors = [
        probe.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=group, placement_group_bundle_index=index
            )
        ).remote()
        for index in range(count)
    ]
    try:
        return ray.get([actor.identity.remote() for actor in actors])
    finally:
        for actor in actors:
            ray.kill(actor)


def _bundle_sort_key(node: str, gpu: int) -> tuple[list[int], int]:
    """Order bundles by node address, then device id, so neighbours share a node."""
    try:
        return [int(part) for part in node.split(".")], gpu
    except ValueError:
        pass
    try:
        return [int(part) for part in socket.gethostbyname(node).split(".")], gpu
    except (socket.gaierror, TypeError, ValueError):
        # Not an IPv4 address; any stable total order keeps one node's bundles together.
        return [ord(character) for character in node], gpu
