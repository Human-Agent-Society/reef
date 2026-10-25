"""Serve saved weights on inference engines that the control plane controls directly.

Reef's serving engines are their own service, separate from every trainer.
The runtimes here are the control plane's ``InferenceRuntime`` for such a
service. They reach the engines through an ``InferenceBackend`` (for an
inference component, ``reef.inference.remote.service.RemoteInferenceBackend``)
and load only weights that a training job saved and Reef published: a
release's files, when a scenario starts, when a release is committed, or
when one is rolled back. No trainer holds a serving engine.

``AdapterServingRuntime`` serves each scenario its own PEFT adapter on one
base model; ``FullWeightServingRuntime`` serves one scenario's full weights.
"""

from __future__ import annotations

import logging
import threading
from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reef.artifact.artifact import REPOSITORY_FILES, Artifact
from reef.core.errors import ReefError
from reef.runtime.deployment import InferenceService
from reef.runtime.interfaces import (
    AdapterEngine,
    InferenceBackend,
    InferenceHandler,
    InferenceRuntime,
    RuntimeContractError,
    RuntimeLoadId,
    WeightFiles,
    new_runtime_load_id_incarnation,
)
from reef.runtime.residency import AdapterResidencyManager
from reef.surface.adapter import parse_adapter_name

logger = logging.getLogger(__name__)

#: The directory of a published release that holds its PEFT adapter.
ADAPTER_DIR = "adapter"
#: Seconds between two health checks of the serving engines' service.
HEALTH_INTERVAL_S = 30.0


def holds_weights(path: Path) -> bool:
    """Whether a materialized release holds weights; one with only the repository's files is the base model."""
    return bool({entry.name for entry in path.iterdir()} - REPOSITORY_FILES)


class SavedWeightServingRuntime(InferenceRuntime):
    """The engines, the request handler and the runtime load IDs that both serving runtimes share.

    ``backend`` controls the engines, which start paused with the base
    model, and ``handler`` executes requests at ``backend.inference_url()``.
    ``service``, when given, owns the engines and is closed on shutdown.
    Runtime load IDs belong to one fresh incarnation, so a restart never
    reuses an ID for other weights. The base model serves under the first one,
    ``base_version``, and each load under the next unused one. Each load
    first replaces the engines that died since the last one
    (``InferenceBackend.recover``), while the engines are paused.

    With a ``service``, a thread checks its health every
    ``health_interval_s`` seconds. When the check fails, for example because
    the inference component restarted, ``recover_engines`` replaces the
    engines and loads what the runtime serves into them again, so the
    engines serve the same versions as before.
    """

    def __init__(
        self,
        backend: InferenceBackend,
        handler: InferenceHandler,
        *,
        service: InferenceService | None = None,
        model_path: str = "",
        inference_timeout_s: float = 300.0,
        health_interval_s: float = HEALTH_INTERVAL_S,
    ) -> None:
        super().__init__(base_url=backend.inference_url(), inference_timeout_s=inference_timeout_s)
        self._backend = backend
        self._handler = handler
        self._service = service
        self._model_path = model_path
        self._lock = threading.RLock()
        self._incarnation = new_runtime_load_id_incarnation()
        self._next_sequence = 1
        self.base_version = str(RuntimeLoadId(self._incarnation, 0))
        self._version = self.base_version
        backend.initialize_version(self._version)
        backend.resume()
        self.mark_published()
        self._stopping = threading.Event()
        self._watcher: threading.Thread | None = None
        if service is not None:
            self._health_interval_s = health_interval_s
            self._watcher = threading.Thread(target=self.watch_engines, name="reef-serving-health", daemon=True)
            self._watcher.start()

    @property
    def inference_handler(self) -> InferenceHandler:
        return self._handler

    @property
    def model_path(self) -> str:
        return self._model_path

    def serving_runtime_load_id(self) -> str | None:
        versions = set(self._backend.runtime_load_ids())
        return versions.pop() if len(versions) == 1 else None

    def next_version(self) -> str:
        version = str(RuntimeLoadId(self._incarnation, self._next_sequence))
        self._next_sequence += 1
        return version

    def check_engines_serve(self, version: str) -> None:
        reported = set(self._backend.runtime_load_ids())
        if reported != {version}:
            raise RuntimeContractError(f"serving engines report {sorted(reported)!r} after loading {version!r}")

    def watch_engines(self) -> None:
        """The health thread: recover the engines after each failed check, until the runtime shuts down."""
        if self._service is None:
            return
        while not self._stopping.wait(self._health_interval_s):
            try:
                self._service.check_health()
                continue
            except Exception:
                logger.warning("the serving engines failed a health check; recovering them", exc_info=True)
            try:
                self.recover_engines()
            except Exception:
                logger.exception("recovering the serving engines failed; the next health check tries again")

    def recover_engines(self) -> None:
        """Replace lost or dead engines and load the served releases into them again; generation pauses meanwhile."""
        with self._lock:
            self._backend.pause()
            try:
                if self._backend.recover():
                    self.reload_served()
            finally:
                self._backend.resume()

    @abstractmethod
    def reload_served(self) -> None:
        """Load what the runtime serves into the engines again, after recovery replaced some; the engines are paused."""

    def shutdown(self) -> None:
        self.pause_admission()
        self._stopping.set()
        if self._watcher is not None:
            self._watcher.join()
        if self._service is not None:
            self._service.close()


@dataclass(frozen=True)
class ServedAdapter:
    """The adapter the engines serve for one scenario.

    ``runtime_load_id`` is the non-empty version it was loaded under, in this
    runtime's incarnation; ``content_id`` is the non-empty content ID of the
    release that holds it; ``files`` is the release's adapter directory, from
    which replacement engines load it again. ``AdapterServingRuntime`` makes
    the value; the version lookup, the same-content check and the engine
    restart read it.
    """

    runtime_load_id: str
    content_id: str
    files: WeightFiles

    def __post_init__(self) -> None:
        for name, value in (("runtime_load_id", self.runtime_load_id), ("content_id", self.content_id)):
            if not isinstance(value, str) or not value:
                raise ValueError(f"ServedAdapter.{name} must be a non-empty string")
        if not isinstance(self.files, WeightFiles) or self.files.layout != "adapter":
            raise ValueError("ServedAdapter.files must be adapter WeightFiles")


class AdapterServingRuntime(SavedWeightServingRuntime, AdapterEngine):
    """Load each scenario's published PEFT adapter into the serving engines and serve requests with it.

    A release holds its adapter in ``adapter/``. Each load names the adapter
    ``adapter_name(scenario, version)`` with a new runtime load ID
    ``version``, and every engine then reports that ID. At most
    ``adapter_capacity`` adapters stay resident; a scenario that serves no
    adapter samples the base model. ``current_runtime_load_id`` is the
    version of the last load, of any scenario.
    """

    def __init__(
        self,
        backend: InferenceBackend,
        handler: InferenceHandler,
        *,
        adapter_capacity: int,
        service: InferenceService | None = None,
        model_path: str = "",
        inference_timeout_s: float = 300.0,
        health_interval_s: float = HEALTH_INTERVAL_S,
    ) -> None:
        self._residency = AdapterResidencyManager(adapter_capacity)
        self._served: dict[str, ServedAdapter] = {}
        super().__init__(
            backend,
            handler,
            service=service,
            model_path=model_path,
            inference_timeout_s=inference_timeout_s,
            health_interval_s=health_interval_s,
        )

    @property
    def per_scenario_adapters(self) -> bool:
        return True

    @property
    def shared_by_training_scenarios(self) -> bool:
        return True

    def serving_adapter_runtime_load_id(self, scenario: str) -> str | None:
        with self._lock:
            served = self._served.get(scenario)
        return None if served is None else served.runtime_load_id

    def adapter_residency_status(self) -> Mapping[str, Any]:
        return self._residency.status()

    def restore_checkpoint(self, artifact: Artifact, *, scenario: str | None = None) -> str:
        """Serve a rolled-back release of ``scenario``; return the version that serves it."""
        return self.activate_checkpoint(artifact, scenario=scenario)

    def activate_checkpoint(self, artifact: Artifact, *, scenario: str | None = None) -> str:
        """Serve the release ``artifact`` of ``scenario``; return the version that serves it.

        A release whose content the engines already serve for the scenario
        loads nothing. A release without weights unloads the scenario's
        adapter and returns ``base_version``. Any other release loads its
        adapter under a new version. Raise ``ReefError`` without ``scenario``.

        When the recovery before a load replaced an engine, the replacement
        holds no adapter. The runtime then restarts every engine and loads
        each scenario's current adapter again (``restart_engines``), so the
        engines hold exactly the adapters the residency manager records.
        """
        if scenario is None:
            raise ReefError("per-scenario adapters are served for a named scenario")
        path = artifact.materialize().local_path
        if path is None:
            raise ReefError("serving a release requires a materialized artifact")
        content_id = artifact.ref.content_id
        with self._lock:
            served = self._served.get(scenario)
            if served is not None and served.content_id == content_id:
                return served.runtime_load_id
            if not holds_weights(path):
                # Requests for a release without weights sample the base model.
                # A refused unload raises before the scenario stops serving its adapter.
                self._residency.release(scenario, self)
                self._served.pop(scenario, None)
                return self.base_version
            adapter = path / ADAPTER_DIR
            if not (adapter / "adapter_config.json").is_file():
                raise ReefError(f"release {artifact.ref.release_id} holds no PEFT adapter in {ADAPTER_DIR}/")
            version = self.next_version()
            files = WeightFiles(adapter, "adapter")
            self._backend.pause()
            try:
                if self._backend.recover():
                    self.restart_engines()
                # Generation is paused, so the scenario's own adapter may make room for its new one.
                self._residency.activate(scenario, version, self, payload=files, supersede=True)
            finally:
                self._backend.resume()
            self._served[scenario] = ServedAdapter(version, content_id, files)
            self._version = version
            self.mark_published()
            return version

    def reload_served(self) -> None:
        self.restart_engines()

    def restart_engines(self) -> None:
        """Retire every engine, start them all again and load each scenario's current adapter; engines are paused."""
        self._backend.abort()
        self._backend.recover()
        self._residency = AdapterResidencyManager(self._residency.capacity)
        for scenario, served in self._served.items():
            self._residency.activate(scenario, served.runtime_load_id, self, payload=served.files)

    # -- Adapter engine, for the residency manager

    def load_adapter(self, name: str, payload: Any) -> None:
        if not isinstance(payload, WeightFiles):
            raise TypeError("an adapter load needs the adapter's WeightFiles")
        _, version = parse_adapter_name(name)
        self._backend.load_weight_files(payload, adapter_name=name, runtime_load_id=version)
        self.check_engines_serve(version)

    def unload_adapter(self, name: str) -> None:
        self._backend.unload_adapter(name)


class FullWeightServingRuntime(SavedWeightServingRuntime):
    """Load one scenario's published full weights, a Hugging Face checkpoint, into the serving engines.

    Each load replaces the served weights under a new runtime load ID, which
    every engine then reports, and becomes the version Reef records for the
    requests that follow. The runtime serves one scenario, so it ignores
    ``scenario``; the registry refuses a second training scenario on it
    (``shared_by_training_scenarios`` is False). A release without weights is
    the base model, which the engines serve when they start. Returning to it
    after another release needs the base model's files, which the engines do
    not keep, so it raises ``ReefError``.
    """

    def __init__(
        self,
        backend: InferenceBackend,
        handler: InferenceHandler,
        *,
        service: InferenceService | None = None,
        model_path: str = "",
        inference_timeout_s: float = 300.0,
        health_interval_s: float = HEALTH_INTERVAL_S,
    ) -> None:
        self._served_content: str | None = None
        self._served_files: WeightFiles | None = None
        super().__init__(
            backend,
            handler,
            service=service,
            model_path=model_path,
            inference_timeout_s=inference_timeout_s,
            health_interval_s=health_interval_s,
        )

    def restore_checkpoint(self, artifact: Artifact, *, scenario: str | None = None) -> str:
        return self.activate_checkpoint(artifact, scenario=scenario)

    def activate_checkpoint(self, artifact: Artifact, *, scenario: str | None = None) -> str:
        path = artifact.materialize().local_path
        if path is None:
            raise ReefError("serving a release requires a materialized artifact")
        content_id = artifact.ref.content_id
        with self._lock:
            if self._served_content == content_id:
                return self._version
            if not holds_weights(path):
                if self._served_content is not None:
                    raise ReefError(
                        "the serving engines cannot return to the base model; roll back to a saved release"
                    )
                return self._version
            version = self.next_version()
            files = WeightFiles(path, "full")
            self._backend.pause()
            try:
                # The load writes every engine, so an engine that recovery replaced serves the release too.
                self._backend.recover()
                self._backend.load_weight_files(files, adapter_name=None, runtime_load_id=version)
                self.check_engines_serve(version)
            finally:
                self._backend.resume()
            self._served_content = content_id
            self._served_files = files
            self._version = version
            self.mark_published()
            return version

    def reload_served(self) -> None:
        """Load the served release into every engine again under its version; the base model needs no load."""
        if self._served_files is None:
            return
        self._backend.load_weight_files(self._served_files, adapter_name=None, runtime_load_id=self._version)
        self.check_engines_serve(self._version)
