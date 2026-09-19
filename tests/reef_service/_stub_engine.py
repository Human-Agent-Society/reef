"""A CPU stand-in for the SGLang engines: the control surface Reef's coordinator drives, as Ray actors.

Selected with ``inference.backend: reef_service._stub_engine:create_inference``.
It borrows the deployment's reservation like SGLang does, accepts Reef's
adapter-file transfer, and records every adapter it was asked to load so a
test can check what a hosted trainer published.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from reef.inference.sglang.backend import SGLangInferenceBackend
from reef.runtime.deployment import (
    ADAPTER_FILES_PROTOCOL,
    DeploymentResources,
    InferenceConnection,
    InferenceResources,
    InferenceService,
)
from reef.runtime.executor import ExecutorConfig, WorkerSpec
from reef.runtime.executor.ray import RayExecutor
from reef.runtime.interfaces import InferenceBackend

PROTOCOL = "reef-test-engine-v1"
CONTROL_NAME = "reef-test-engine"


class StubEngine:
    """One engine: a served version and the adapters it holds."""

    def __init__(self) -> None:
        self.version = "engine:0"
        self.adapters: dict[str, str] = {}

    def get_runtime_load_id(self) -> str:
        return self.version

    def set_runtime_load_id(self, runtime_load_id: str) -> dict[str, Any]:
        self.version = str(runtime_load_id)
        return {"success": True}

    def load_lora_adapter_from_disk(self, lora_name: str, lora_path: str) -> dict[str, Any]:
        if not os.path.isfile(os.path.join(lora_path, "adapter_config.json")):
            return {"success": False, "message": f"no adapter_config.json under {lora_path}"}
        self.adapters[lora_name] = lora_path
        return {"success": True}

    def unload_lora_adapter(self, lora_name: str) -> dict[str, Any]:
        self.adapters.pop(lora_name, None)
        return {"success": True}


class StubControl:
    """The control actor: what ``SGLangControl`` offers, over stub engines."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        import ray

        self.config = dict(config)
        self.engines = [ray.remote(StubEngine).remote() for _ in range(max(1, int(config.get("num_gpus", 1))))]
        self.loaded: list[tuple[str, str, str | None]] = []
        self.paused = False
        self.terminated = 0

    def check_health(self) -> None:
        return

    def inference_url(self) -> str:
        return "http://127.0.0.1:1/stub-engine"

    def get_runtime_load_ids(self) -> list[str]:
        import ray

        return ray.get([engine.get_runtime_load_id.remote() for engine in self.engines])

    def get_updatable_engines_and_lock(self) -> tuple[Any, ...]:
        return self.engines, None, 0, [], [], []

    def pause_generation_for_update(self) -> None:
        self.paused = True

    def continue_generation_after_update(self) -> None:
        self.paused = False

    def terminate_updatable_engines(self) -> int:
        self.terminated += 1
        return 0

    def recover_updatable_engines(self) -> tuple[Any, ...]:
        return self.get_updatable_engines_and_lock()

    def prepare_training_connection(self) -> None:
        return

    def offload(self, tags: Any = None) -> None:
        raise AssertionError("a hosted trainer never offloads the engines")

    def onload(self, tags: Any = None) -> None:
        raise AssertionError("a hosted trainer never onloads")

    def onload_weights(self) -> None:
        raise AssertionError("a hosted trainer never onloads")

    def onload_kv(self) -> None:
        raise AssertionError("a hosted trainer never onloads")

    def load_adapter_from_disk(self, lora_name: str, lora_path: str, runtime_load_id: str | None = None) -> None:
        import ray

        results = ray.get(
            [
                engine.load_lora_adapter_from_disk.remote(lora_name=lora_name, lora_path=lora_path)
                for engine in self.engines
            ]
        )
        for result in results:
            if result.get("success") is not True:
                raise RuntimeError(f"stub engine refused adapter {lora_name!r}: {result}")
        if runtime_load_id is not None:
            ray.get([engine.set_runtime_load_id.remote(runtime_load_id) for engine in self.engines])
        self.loaded.append((lora_name, lora_path, runtime_load_id))

    def loads(self) -> list[tuple[str, str, str | None]]:
        return list(self.loaded)

    def state(self) -> dict[str, Any]:
        return {"paused": self.paused, "terminated": self.terminated, "versions": self.get_runtime_load_ids()}

    def shutdown(self) -> None:
        return


class StubEngineService(InferenceService):
    """Start the control actor in the deployment's Ray session on the reserved inference bundles."""

    connection_protocol = PROTOCOL
    supported_transfer_protocols = (PROTOCOL, ADAPTER_FILES_PROTOCOL)

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self._control: RayExecutor | None = None
        self._closed = False

    def start(self, resources: DeploymentResources) -> InferenceConnection:
        if not isinstance(resources, InferenceResources) or resources.inference_placement is None:
            raise ValueError("the stub engine borrows the deployment's inference reservation")
        placement = resources.inference_placement
        if len(placement.bundle_indices) != int(self.config["num_gpus"]):
            raise ValueError("the stub engine expects one reserved bundle per inference GPU")
        self._control = RayExecutor(
            ExecutorConfig(
                backend=RayExecutor,
                workers=(WorkerSpec(worker_cls="reef_service._stub_engine:StubControl", args=(self.config,)),),
                # Ray workers import this module by name; give them the driver's import path.
                options={
                    "num_cpus": 1,
                    "num_gpus": 0,
                    "name": CONTROL_NAME,
                    "runtime_env": {"env_vars": {"PYTHONPATH": os.environ.get("PYTHONPATH", "")}},
                },
                launch_timeout_s=120,
            )
        )
        return InferenceConnection(self.connection_protocol, RayExecutor.from_workers(self._control.workers))

    def prepare_weight_transfer(self, connection: InferenceConnection) -> None:
        connection.control.rpc(0, "prepare_training_connection", timeout=30)

    def backend(self, connection: InferenceConnection) -> InferenceBackend:
        return SGLangInferenceBackend(connection.control)

    def check_health(self) -> None:
        if self._control is None or self._closed:
            raise RuntimeError("stub engine is not running")
        self._control.rpc(0, "check_health", timeout=30)

    def poll(self) -> None:
        self.check_health()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._control is not None:
            self._control.shutdown()


def create_inference(config: Mapping[str, Any]) -> StubEngineService:
    return StubEngineService(config)
