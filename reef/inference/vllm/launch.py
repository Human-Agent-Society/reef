"""Ray placement and native vLLM launch, independent of any trainer."""

from __future__ import annotations

from typing import Any

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from reef.inference.process import EngineGroup, EngineStartup
from reef.inference.vllm.config import VLLMConfig
from reef.inference.vllm.engine import ReefVLLMEngine


def engine_environment(config: VLLMConfig) -> dict[str, str]:
    """Ray must leave device visibility alone: every engine names its own GPUs."""
    return {
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
        "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES": "1",
        "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES": "1",
        **config.env_vars,
    }


class VLLMEngineGroup(EngineGroup):
    """One model's engines: identical single-node replicas, one actor each, on the reserved GPUs.

    ``placement`` is the deployment's inference reservation: the placement
    group, one bundle index per GPU and the physical GPU ids in bundle order.
    """

    def __init__(self, config: VLLMConfig, placement: Any) -> None:
        self.config = config
        self.placement = placement
        self.all_engines: list[Any] = [None] * config.engine_count
        self.port_ranges: dict[int, tuple[str, range]] = {}

    @property
    def engines(self) -> list[Any]:
        """Live engine actors; a ``None`` slot is an engine awaiting recovery."""
        return [engine for engine in self.all_engines if engine is not None]

    def gpu_ids(self, index: int) -> tuple[int, ...]:
        _, _, devices = self.placement
        width = self.config.gpus_per_engine
        selected = devices[index * width : (index + 1) * width]
        if len(selected) != width:
            raise ValueError("inference placement is smaller than its vLLM engines")
        return tuple(int(device) for device in selected)

    def start_engines(self, cursors: dict[str, int]) -> dict[int, Any]:
        """Launch an actor for every empty slot; return each new slot's pending ``init`` call.

        ``cursors`` tracks the next free port per host so engines of this
        deployment never race for the same port; probing starts at the
        configured ``engine_port_base``.
        """
        created = [index for index, engine in enumerate(self.all_engines) if engine is None]
        for index in created:
            self.all_engines[index] = self._launch_actor(index)
        return self.init_engines(created, cursors)

    def init_engines(self, slots: list[int], cursors: dict[str, int]) -> dict[int, Any]:
        pending = {}
        for index in slots:
            actor = self.all_engines[index]
            host, _ = ray.get(actor.node_address_and_port.remote())
            _, port = ray.get(
                actor.node_address_and_port.remote(start_port=cursors.get(host, self.config.engine_port_base))
            )
            cursors[host] = port + 1
            self.port_ranges[index] = (host, range(port, port + 1))
            pending[index] = actor.init.remote(host, port)
        return pending

    def _launch_actor(self, index: int) -> Any:
        pg, bundles, _ = self.placement
        strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_bundle_index=bundles[index * self.config.gpus_per_engine],
            placement_group_capture_child_tasks=True,
        )
        return (
            ray.remote(ReefVLLMEngine)
            .options(
                num_cpus=0.2,
                num_gpus=0.2,
                runtime_env={"env_vars": engine_environment(self.config)},
                scheduling_strategy=strategy,
            )
            .remote(self.config, rank=index, gpu_ids=self.gpu_ids(index))
        )

    def recover(self) -> None:
        """Relaunch dead engines."""
        startup = EngineStartup()
        startup.start(self)
        startup.wait()
