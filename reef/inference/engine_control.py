"""Shared control of the engine groups that Reef launches on Ray.

SGLang and vLLM use the same control vocabulary. Each role here has one base
class: the inference backend, the inference service, the control actor and
its default serving executor. The engine subpackages subclass these bases and
set the differences as class attributes. This module imports no engine library.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, Generic, TypeVar

from reef.inference.process import RayHealthProbe
from reef.runtime.deployment import DeploymentResources, InferenceConnection, InferenceResources, InferenceService
from reef.runtime.executor import Executor, ExecutorConfig, WorkerSpec
from reef.runtime.executor.config import ExecutorSettings, select_executor
from reef.runtime.executor.delegating import DelegatingExecutor
from reef.runtime.executor.ray import RayExecutor
from reef.runtime.executor.uniproc import UniProcExecutor
from reef.runtime.interfaces import InferenceBackend, WeightFiles

#: Bound on one control RPC or engine launch; a weight update legitimately takes hours.
CONTROL_TIMEOUT_S = 14_400


class EngineConfig:
    """Settings that the shared control reads from every engine configuration."""

    env_vars: dict[str, str]
    executor: str | type[Executor]
    executor_options: dict[str, Any]


ConfigT = TypeVar("ConfigT", bound=EngineConfig)


class EngineInferenceBackend(InferenceBackend):
    """Borrow a control connection without owning engine lifetime.

    The inference component decides operation order and whether a loaded
    revision may serve requests. This adapter translates individual operations
    to the engine control vocabulary and validates receiver results.
    """

    #: The engine name in error messages.
    engine_name: str
    #: Whether an engine may answer an adapter unload with no result.
    unload_result_optional = False

    def __init__(self, executor: Executor) -> None:
        self._executor = executor

    def inference_url(self) -> str:
        value = self._call("inference_url")
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"{self.engine_name} inference did not provide a serving endpoint")
        return value

    def runtime_load_ids(self) -> Sequence[str]:
        values = self._call("get_runtime_load_ids")
        if (
            not isinstance(values, Sequence)
            or isinstance(values, (str, bytes))
            or any(not isinstance(value, str) or not value for value in values)
        ):
            raise RuntimeError(f"{self.engine_name} inference returned invalid runtime load IDs")
        return tuple(values)

    def initialize_version(self, runtime_load_id: str) -> None:
        """Stamp Reef's initial version while serving stays paused."""
        if not isinstance(runtime_load_id, str) or not runtime_load_id:
            raise ValueError("initial runtime load ID must be a nonempty string")
        self._engines().collective_rpc("set_runtime_load_id", args=(runtime_load_id,), timeout=CONTROL_TIMEOUT_S)

    def pause(self) -> None:
        self._call("pause_generation_for_update")

    def resume(self) -> None:
        self._call("continue_generation_after_update")

    def recover(self) -> int:
        replaced = self._call("recover_updatable_engines")
        if isinstance(replaced, bool) or not isinstance(replaced, int) or replaced < 0:
            raise RuntimeError(f"{self.engine_name} inference returned an invalid replaced-engine count: {replaced!r}")
        return replaced

    def abort(self) -> None:
        self._call("terminate_updatable_engines")

    def load_weight_files(self, files: WeightFiles, *, adapter_name: str | None, runtime_load_id: str | None) -> None:
        if (adapter_name is None) != (files.layout == "full"):
            raise ValueError("an adapter name is given exactly for adapter weight files")
        if adapter_name is None:
            self._call("load_weights_from_disk", str(files.path), runtime_load_id)
        else:
            self._call("load_adapter_from_disk", adapter_name, str(files.path), runtime_load_id)

    def unload_adapter(self, name: str) -> None:
        results = self._engines().collective_rpc(
            "unload_lora_adapter", kwargs={"lora_name": name}, timeout=CONTROL_TIMEOUT_S
        )
        for result in results:
            if result is None and self.unload_result_optional:
                continue
            if not isinstance(result, Mapping) or result.get("success") is not True:
                raise RuntimeError(f"engine kept adapter {name!r}: {result!r}")

    def _engines(self) -> RayExecutor:
        return RayExecutor.from_workers(self._call("updatable_engines"))

    def _call(self, method: str, *args: Any) -> Any:
        return self._executor.rpc(0, method, args=args, timeout=CONTROL_TIMEOUT_S)


class EngineInferenceService(InferenceService, Generic[ConfigT]):
    """Own engines and the control actor; borrow the deployment allocation."""

    #: The engine name in error messages.
    engine_name: str
    #: Import reference of the control actor class, resolved inside its Ray worker.
    control_worker_cls: str
    backend_type: type[EngineInferenceBackend]

    def __init__(self, config: ConfigT) -> None:
        self.config = config
        self._inference: RayExecutor | None = None
        self._started = False
        self._closed = False
        self._probe = RayHealthProbe()

    @abstractmethod
    def engine_environment(self) -> dict[str, str]:
        """Return the environment variables of the control actor and its engines."""

    def start(self, resources: DeploymentResources) -> InferenceConnection:
        if self._started or self._closed:
            raise RuntimeError("inference service can only be started once")
        if not isinstance(resources, InferenceResources) or resources.inference_placement is None:
            raise ValueError(f"{self.engine_name} inference requires its supplied model reservations")
        self._started = True
        self._inference = RayExecutor(
            ExecutorConfig(
                backend=RayExecutor,
                workers=(
                    WorkerSpec(worker_cls=self.control_worker_cls, args=(self.config, resources.inference_placement)),
                ),
                options={
                    "num_cpus": 1,
                    "num_gpus": 0,
                    "runtime_env": {"env_vars": self.engine_environment()},
                },
                launch_timeout_s=CONTROL_TIMEOUT_S,
            )
        )
        return InferenceConnection(self.connection_protocol, RayExecutor.from_workers(self._inference.workers))

    def _control(self, connection: InferenceConnection) -> Executor:
        if connection.protocol != self.connection_protocol or connection.control is None:
            raise ValueError(f"incompatible {self.engine_name} inference connection: {connection.protocol!r}")
        return connection.control

    def backend(self, connection: InferenceConnection) -> EngineInferenceBackend:
        """Adapt a compatible borrowed connection for the serving runtime that loads weights into the engines."""
        return self.backend_type(self._control(connection))

    def _running(self) -> RayExecutor:
        if self._inference is None or self._closed:
            raise RuntimeError("inference service is not running")
        return self._inference

    def check_health(self) -> None:
        self._running().rpc(0, "check_health", timeout=30)

    def poll(self) -> None:
        self._probe.poll(self._running().workers[0], "check_health")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._inference is not None:
            try:
                self._inference.rpc(0, "shutdown", timeout=90)
            except Exception:
                # The resource owner confirms retirement of native children.
                logging.getLogger(__name__).exception("Inference shutdown failed; retiring its owned process groups")
            finally:
                self._inference.shutdown()


class EngineControl(Generic[ConfigT]):
    """Own the selected serving executor inside Reef's inference control actor.

    Every method forwards one control operation to the serving worker.
    """

    def __init__(self, config: ConfigT, pg: Any) -> None:
        self._serving = Executor.create(
            ExecutorConfig(
                backend=config.executor,
                options={**config.executor_options, "config": config, "pg": pg},
            )
        )
        self._closed = False

    def _call(self, method: str, *args: Any) -> Any:
        return self._serving.rpc(0, method, args=args, timeout=CONTROL_TIMEOUT_S)

    def check_health(self) -> None:
        self._serving.check_health(timeout=30)

    def inference_url(self) -> Any:
        return self._call("inference_url")

    def get_runtime_load_ids(self) -> Any:
        return self._call("get_runtime_load_ids")

    def load_adapter_from_disk(self, lora_name: str, lora_path: str, runtime_load_id: str | None = None) -> Any:
        return self._call("load_adapter_from_disk", lora_name, lora_path, runtime_load_id)

    def load_weights_from_disk(self, model_path: str, runtime_load_id: str | None = None) -> Any:
        return self._call("load_weights_from_disk", model_path, runtime_load_id)

    def pause_generation_for_update(self) -> Any:
        return self._call("pause_generation_for_update")

    def continue_generation_after_update(self) -> Any:
        return self._call("continue_generation_after_update")

    def terminate_updatable_engines(self) -> Any:
        return self._call("terminate_updatable_engines")

    def updatable_engines(self) -> Any:
        return self._call("updatable_engines")

    def recover_updatable_engines(self) -> Any:
        return self._call("recover_updatable_engines")

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._serving.shutdown()


class EngineExecutor(DelegatingExecutor):
    """One control rank manages the engine groups of one inference service.

    The serving worker lives in the inference control actor; engine processes
    are Ray actors allocated from supplied reservations. Alternative executors
    implement the same control RPC vocabulary without inheriting these classes.
    """

    @abstractmethod
    def create_worker(self) -> object:
        """Build the serving worker from ``self.config.options``; import the engine library here."""

    def _init_executor(self) -> None:
        self._rpc = UniProcExecutor.from_workers([self.create_worker()], owned=True)

    def check_health(self, timeout: float | None = None) -> None:
        self._rpc.rpc(0, "check_health", timeout=timeout)


def engine_executor_class(value: str | type[Executor], default: type[Executor], engine_name: str) -> type[Executor]:
    """Resolve the configured serving executor; ``ray`` selects the engine's ``default`` executor."""
    if isinstance(value, type):
        return Executor.get_class(value)
    selected = select_executor(ExecutorSettings(value or "auto"), role="rollout").settings.backend
    if selected in ("mp", "uni"):
        raise ValueError(f"{engine_name} inference requires ray or a native inference executor; got {selected!r}")
    return default if selected == "ray" else Executor.get_class(selected)
