"""Ray placement, native vLLM launch and engine health, independent of any trainer."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import suppress
from typing import Any

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from reef.inference.vllm.config import VLLMConfig
from reef.inference.vllm.engine import ReefVLLMEngine
from reef.runtime.recovery import EngineHealthChecks, EngineHealthTarget


def engine_environment(config: VLLMConfig) -> dict[str, str]:
    """Ray must leave device visibility alone: every engine names its own GPUs."""
    return {
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
        "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES": "1",
        "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES": "1",
        **config.env_vars,
    }


class VLLMEngineGroup:
    """One model's engines: identical single-node replicas, one actor each, on the reserved GPUs.

    ``placement`` is the deployment's inference reservation: the placement
    group, one bundle index per GPU and the physical GPU ids in bundle order.
    """

    def __init__(self, config: VLLMConfig, placement: Any) -> None:
        self.config = config
        self.placement = placement
        self.all_engines: list[Any] = [None] * config.engine_count
        self.num_new_engines = 0
        self.needs_offload = bool(config.offload and config.shared_gpus > 0)

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

    def parallel_config(self) -> dict[str, int]:
        return {"tp_size": self.config.gpus_per_engine, "pp_size": 1, "ep_size": 1, "moe_dp_size": 1}

    def start_engines(self, cursors: dict[str, int]) -> list[Any]:
        """Launch an actor for every empty slot; return the pending ``init`` calls.

        ``cursors`` tracks the next free port per host so engines on one node
        never race for the same port.
        """
        created = [index for index, engine in enumerate(self.all_engines) if engine is None]
        for index in created:
            self.all_engines[index] = self._launch_actor(index)
        self.num_new_engines = len(created)
        pending = []
        for index in created:
            actor = self.all_engines[index]
            host, _ = ray.get(actor.node_address_and_port.remote())
            _, port = ray.get(actor.node_address_and_port.remote(start_port=cursors.get(host, 15000)))
            cursors[host] = port + 1
            pending.append(actor.init.remote(host, port))
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
        """Relaunch dead engines; a colocated replacement releases memory like the originals did."""
        missing = [index for index, engine in enumerate(self.all_engines) if engine is None]
        pending = self.start_engines({})
        if pending:
            ray.get(pending)
        replaced = [self.all_engines[index] for index in missing]
        if self.needs_offload and replaced:
            ray.get([engine.release_memory_occupation.remote() for engine in replaced])
            ray.get([engine.resume_memory_occupation.remote(tags=["weights"]) for engine in replaced])


class VLLMEngineHealthChecks(EngineHealthChecks):
    """Snapshot the live engine actors of one group."""

    def __init__(self, group: VLLMEngineGroup) -> None:
        self._group = group

    def targets(self) -> Sequence[EngineHealthTarget]:
        return [
            _VLLMEngineHealthTarget(self._group, index, engine)
            for index, engine in enumerate(self._group.all_engines)
            if engine is not None
        ]


class _VLLMEngineHealthTarget(EngineHealthTarget):
    def __init__(self, group: VLLMEngineGroup, index: int, engine: Any) -> None:
        self._group = group
        self._index = index
        self._engine = engine

    def check(self, timeout: float) -> None:
        result = ray.get(self._engine.health_generate.remote(timeout=timeout), timeout=timeout)
        if result is not True:
            raise RuntimeError("inference health probe did not report success")

    def retire(self, timeout: float) -> None:
        current = self._group.all_engines
        if self._index >= len(current) or current[self._index] is not self._engine:
            return
        with suppress(Exception):
            ray.get(self._engine.shutdown.remote(), timeout=timeout)
        # Kill the captured handle even if shutdown failed. Never replace it
        # with the current occupant after waiting for an RPC.
        ray.kill(self._engine, no_restart=True)
        if current[self._index] is self._engine:
            current[self._index] = None
