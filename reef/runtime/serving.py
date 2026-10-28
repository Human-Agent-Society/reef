"""Serve saved weights on inference engines that the control plane controls directly.

Reef's serving engines are their own service, separate from every trainer.
The runtimes here are the control plane's ``InferenceRuntime`` for such a
service. They reach the engines through an ``InferenceBackend`` (for an
inference component, ``reef.inference.remote.service.RemoteInferenceBackend``)
and load only weights that a training job saved and Reef published: a
release's files, when a scenario starts, when a release is committed, or
when one is activated. No trainer holds a serving engine.

``AdapterServingRuntime`` serves each scenario its own PEFT adapter on one
base model; ``FullWeightServingRuntime`` serves one scenario's full weights.
"""

from __future__ import annotations

import logging
import threading
from abc import abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from reef.artifact.artifact import REPOSITORY_FILES, Artifact
from reef.core.errors import ReefError
from reef.core.evaluation import ArtifactKind
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
from reef.runtime.residency import AdapterResidencyError, AdapterResidencyManager
from reef.surface.adapter import parse_adapter_name

logger = logging.getLogger(__name__)

#: The directory of a published release that holds its PEFT adapter.
ADAPTER_DIR = "adapter"
#: Seconds between two health checks of the serving engines' service.
HEALTH_INTERVAL_S = 30.0


def holds_weights(path: Path) -> bool:
    """Whether a materialized release holds weights; one with only the repository's files is the base model."""
    return bool({entry.name for entry in path.iterdir()} - REPOSITORY_FILES)


def check_engines_serve(backend: InferenceBackend, version: str) -> None:
    """Raise ``RuntimeContractError`` unless every engine of ``backend`` reports exactly ``version``."""
    reported = set(backend.runtime_load_ids())
    if reported != {version}:
        raise RuntimeContractError(f"serving engines report {sorted(reported)!r} after loading {version!r}")


class SavedWeightServingRuntime(InferenceRuntime):
    """The engines, the request handler and the runtime load IDs that both serving runtimes share.

    ``backend`` controls the engines, which start paused with the base
    model, and ``handler`` executes requests at ``backend.inference_url()``.
    ``service``, when given, owns the engines and is closed on shutdown.
    Runtime load IDs belong to one fresh incarnation, so a restart never
    reuses an ID for other weights. The base model serves under the first one,
    ``base_version``, and each load under the next unused one.
    ``current_runtime_load_id`` is the version of the last release activated,
    of any scenario. Each load
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

    def current_runtime_load_id(self) -> str:
        with self._lock:
            return self._version

    def next_version(self) -> str:
        version = str(RuntimeLoadId(self._incarnation, self._next_sequence))
        self._next_sequence += 1
        return version

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
        self.close_admission()
        self._stopping.set()
        if self._watcher is not None:
            self._watcher.join()
        if self._service is not None:
            self._service.close()


class BackendAdapterEngine(AdapterEngine):
    """The serving engines of ``backend`` as the ``AdapterEngine`` that the residency manager drives.

    A load loads the adapter files into every engine under the name, with the
    runtime load ID that the name carries, and checks that every engine then
    reports it. An unload removes the name from every engine.
    ``AdapterServingRuntime`` makes the value and hands it to its residency
    manager, which alone calls it.
    """

    def __init__(self, backend: InferenceBackend) -> None:
        self.backend = backend

    def load_adapter(self, name: str, files: WeightFiles) -> None:
        _, version = parse_adapter_name(name)
        self.backend.load_weight_files(files, adapter_name=name, runtime_load_id=version)
        check_engines_serve(self.backend, version)

    def unload_adapter(self, name: str) -> None:
        self.backend.unload_adapter(name)


class AdapterServingRuntime(SavedWeightServingRuntime):
    """Load each scenario's published PEFT adapter into the serving engines and serve requests with it.

    A release holds its adapter in ``adapter/``. Each load names the adapter
    ``adapter_name(scenario, version)`` with a new runtime load ID
    ``version``, and every engine then reports that ID. At most
    ``adapter_capacity`` adapters stay resident. The residency manager is the
    one record of what each scenario serves: the version of its current
    adapter, or none, and then the scenario samples the base model. The
    runtime also keeps the content ID of the release that each current
    adapter was loaded from, so a release of the same content loads nothing.
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
        self._engine = BackendAdapterEngine(backend)
        self._residency = AdapterResidencyManager(adapter_capacity)
        #: The content ID of the release that each current adapter was loaded from, by its runtime load ID.
        self._content_ids: dict[str, str] = {}
        super().__init__(
            backend,
            handler,
            service=service,
            model_path=model_path,
            inference_timeout_s=inference_timeout_s,
            health_interval_s=health_interval_s,
        )

    @property
    def served_weight_kind(self) -> ArtifactKind:
        return "model_adapter"

    @property
    def per_scenario_adapters(self) -> bool:
        return True

    @property
    def shared_by_training_scenarios(self) -> bool:
        return True

    def serving_adapter_runtime_load_id(self, scenario: str) -> str | None:
        current = self._residency.current_adapter(scenario)
        return None if current is None else current.runtime_load_id

    def adapter_residency_status(self) -> Mapping[str, Any]:
        return self._residency.status()

    def activate_checkpoint(self, artifact: Artifact, *, scenario: str | None = None) -> str:
        """Serve the release ``artifact`` of ``scenario``; return the version that serves it.

        A release whose content the engines already serve for the scenario
        loads nothing. A release without weights unloads the scenario's
        adapter and returns ``base_version``. Any other release loads its
        adapter under a new version. Raise ``ReefError`` without ``scenario``.
        A failed load can leave the scenario serving no adapter, when the new
        adapter took the slot of its own previous one; the residency manager
        records that, so requests then sample the base model.

        When the recovery before a load replaced an engine, the replacement
        holds no adapter. The runtime then restarts every engine and loads
        each scenario's current adapter again (``restart_engines``).
        """
        if scenario is None:
            raise ReefError("per-scenario adapters are served for a named scenario")
        path = artifact.materialize().local_path
        if path is None:
            raise ReefError("serving a release requires a materialized artifact")
        content_id = artifact.ref.content_id
        with self._lock:
            current = self._residency.current_adapter(scenario)
            if current is not None and self._content_ids.get(current.runtime_load_id) == content_id:
                return current.runtime_load_id
            files: WeightFiles | None
            if holds_weights(path):
                adapter = path / ADAPTER_DIR
                if not (adapter / "adapter_config.json").is_file():
                    raise ReefError(f"release {artifact.ref.release_id} holds no PEFT adapter in {ADAPTER_DIR}/")
                version = self.next_version()
                files = WeightFiles(adapter, "adapter")
            else:
                # Requests for a release without weights sample the base model.
                version = self.base_version
                files = None
            self._backend.pause()
            try:
                if files is None:
                    # A refused unload raises before the scenario stops serving its adapter.
                    self._residency.release(scenario, self._engine)
                else:
                    if self._backend.recover():
                        self.restart_engines()
                    # Generation is paused, so the scenario's own adapter may make room for its new one.
                    self._residency.activate(scenario, version, self._engine, files=files, supersede=True)
                    self._content_ids[version] = content_id
            finally:
                self._backend.resume()
                resident = {adapter.runtime_load_id for adapter in self._residency.current_adapters()}
                self._content_ids = {key: value for key, value in self._content_ids.items() if key in resident}
            self._version = version
            return version

    def reload_served(self) -> None:
        self.restart_engines()

    def restart_engines(self) -> None:
        """Retire every engine, start them all again and load each scenario's current adapter; engines are paused.

        A scenario whose adapter fails to load serves no adapter afterwards; the other scenarios still load.
        Raise ``AdapterResidencyError`` after the loop when one failed.
        """
        current = self._residency.current_adapters()
        self._backend.abort()
        self._backend.recover()
        self._residency = AdapterResidencyManager(self._residency.capacity)
        failed: list[str] = []
        for adapter in current:
            try:
                self._residency.activate(adapter.scenario, adapter.runtime_load_id, self._engine, files=adapter.files)
            except AdapterResidencyError:
                logger.exception(
                    "scenario %r lost its adapter %r in an engine restart", adapter.scenario, adapter.name
                )
                failed.append(adapter.scenario)
        if failed:
            raise AdapterResidencyError(f"the engine restart could not load the adapters of scenarios {failed}")


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
                    raise ReefError("the serving engines cannot return to the base model; activate a saved release")
                return self._version
            version = self.next_version()
            files = WeightFiles(path, "full")
            self._backend.pause()
            try:
                # The load writes every engine, so an engine that recovery replaced serves the release too.
                self._backend.recover()
                self._backend.load_weight_files(files, adapter_name=None, runtime_load_id=version)
                check_engines_serve(self._backend, version)
            finally:
                self._backend.resume()
            self._served_content = content_id
            self._served_files = files
            self._version = version
            return version

    def reload_served(self) -> None:
        """Load the served release into every engine again under its version; the base model needs no load."""
        if self._served_files is None:
            return
        self._backend.load_weight_files(self._served_files, adapter_name=None, runtime_load_id=self._version)
        check_engines_serve(self._backend, self._version)
