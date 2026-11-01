"""Serve saved weights on inference engines that the control plane controls directly.

Reef's serving engines are their own service, separate from every trainer.
The runtimes here are the control plane's ``InferenceRuntime`` for such a
service. They reach the engines through an ``InferenceBackend`` (for an
inference component, ``reef.inference.remote.service.RemoteInferenceBackend``)
and load only weights that a training job saved and Reef published: a
release's files, when a scenario starts, when a release is committed, or
when one is activated. No trainer holds a serving engine. ``ReleaseWeights``
says where a release's files come from: the release itself
(``ReleaseFiles``), or a service that the release names.

``AdapterServingRuntime`` serves each scenario its own PEFT adapter on one
base model; ``FullWeightServingRuntime`` serves one scenario's full weights.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reef.artifact.artifact import Artifact
from reef.artifact.sources import HuggingFaceSource, download_huggingface_snapshot, read_reference
from reef.core.errors import ReefError
from reef.core.evaluation import ArtifactKind
from reef.core.plugins import ConfiguredPlugin
from reef.core.served_models import ServedArtifact
from reef.core.weight_layout import ADAPTER_DIR, WeightLayout
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
from reef.surface.adapter import adapter_name, parse_adapter_name

logger = logging.getLogger(__name__)

#: Seconds between two health checks of the serving engines' service.
HEALTH_INTERVAL_S = 30.0


@dataclass(frozen=True)
class OpenedEngines:
    """Engines that control took: ``backend`` drives them, and ``attached`` is whether they ran already, under the
    weights and versions that a former control instance gave them; false for engines that ``open`` started.

    Purpose: ``ServingEngines.open`` makes it; ``SavedWeightServingRuntime.open`` reads it.
    """

    backend: InferenceBackend
    attached: bool

    def __post_init__(self) -> None:
        if not isinstance(self.backend, InferenceBackend):
            raise TypeError("opened engines need an InferenceBackend")
        if not isinstance(self.attached, bool):
            raise TypeError("attached is a bool")


class ServingEngines(ABC):
    """The engines of a serving runtime, which only control opens.

    Values: ``base_url`` is the engines' HTTP(S) address, known without opening them. ``open`` attaches to the
    engines that run, or starts them when none run (``OpenedEngines``); a runtime calls it at most once.
    ``watched`` is whether the runtime checks their health (``check_health``, which raises when they failed).
    ``close`` releases what ``open`` took and stops no engine that serves requests.

    Purpose: a ``SavedWeightServingRuntime`` holds one and opens it in ``InferenceRuntime.open``. A process that
    runs no control never opens it, so it only sends requests to ``base_url``.
    """

    @property
    @abstractmethod
    def base_url(self) -> str: ...

    @property
    def watched(self) -> bool:
        return False

    @abstractmethod
    def open(self) -> OpenedEngines: ...

    def check_health(self) -> None:
        """Raise when the engines failed; engines that are not ``watched`` have no check."""
        return

    def close(self) -> None:
        """Release what ``open`` took; the default holds nothing."""
        return


class ReleaseWeights(ConfiguredPlugin, ABC):
    """Where a serving runtime gets the weight files of a published release.

    ``weight_files(release, layout)`` takes the local directory ``release`` of
    a materialized release and the ``layout`` that the engines load, and
    returns ``WeightFiles`` of that layout with the release's weights. Equal
    releases give files of equal content. It raises ``ReefError`` when the
    release gives no files of ``layout``.

    Purpose: the serving runtimes below call it at each load. ``ReleaseFiles``,
    the default, reads the release's own files. A training backend whose
    releases name weights on a service instead names the plugin that fetches
    them (``TrainingDeployment.engine_weights``).
    """

    @abstractmethod
    def weight_files(self, release: Path, layout: WeightLayout) -> WeightFiles: ...


class ReleaseFiles(ReleaseWeights):
    """The release's own files: the release directory for ``"full"``, its ``adapter/`` for ``"adapter"``.

    The files of a release that refers to a Hugging Face revision (``read_reference``, RFC #995) are that
    revision's snapshot, which Hugging Face downloads once into its cache.
    """

    def weight_files(self, release: Path, layout: WeightLayout) -> WeightFiles:
        try:
            revision = read_reference(release)
        except ValueError as exc:
            raise ReefError(f"the release in {release} holds no readable reference: {exc}") from exc
        if revision is not None:
            release = download_huggingface_snapshot(HuggingFaceSource(revision.repo, revision.commit)).local_path
        if layout == "full":
            return WeightFiles(release, "full")
        adapter = release / ADAPTER_DIR
        if not (adapter / "adapter_config.json").is_file():
            raise ReefError(f"the release in {release} holds no PEFT adapter in {ADAPTER_DIR}/")
        return WeightFiles(adapter, "adapter")


def check_engines_serve(backend: InferenceBackend, version: str) -> None:
    """Raise ``RuntimeContractError`` unless every engine of ``backend`` reports exactly ``version``."""
    reported = set(backend.runtime_load_ids())
    if reported != {version}:
        raise RuntimeContractError(f"serving engines report {sorted(reported)!r} after loading {version!r}")


class SavedWeightServingRuntime(InferenceRuntime):
    """The engines, the request handler and the runtime load IDs that both serving runtimes share.

    ``engines`` reaches the engines, and ``handler`` executes requests at ``engines.base_url``. Building the
    runtime touches no engine: control opens it (``open``), which attaches to the engines or starts them, and
    stamps the base model's version on them; before that, the runtime only sends requests, and an operation that
    changes the engines raises ``ReefError``.
    ``weights`` gives each release's weight files; the default is
    ``ReleaseFiles``.
    Runtime load IDs belong to one fresh incarnation, so a restart never
    reuses an ID for other weights. The base model serves under the first one,
    ``base_version``, and each load under the next unused one;
    ``activate_checkpoint`` returns it. Each load
    first replaces the engines that died since the last one
    (``InferenceBackend.recover``), while the engines are paused.

    When ``engines`` are ``watched``, a thread checks their health every
    ``health_interval_s`` seconds. When the check fails, for example because
    the inference component restarted, ``recover_engines`` replaces the
    engines and loads what the runtime serves into them again, so the
    engines serve the same versions as before.
    """

    def __init__(
        self,
        engines: ServingEngines,
        handler: InferenceHandler,
        *,
        model_path: str = "",
        inference_timeout_s: float = 300.0,
        health_interval_s: float = HEALTH_INTERVAL_S,
        weights: ReleaseWeights | None = None,
    ) -> None:
        super().__init__(base_url=engines.base_url, inference_timeout_s=inference_timeout_s)
        self._weights = ReleaseFiles() if weights is None else weights
        self._engines = engines
        self._opened: InferenceBackend | None = None
        self._handler = handler
        self._model_path = model_path
        self._lock = threading.RLock()
        self._incarnation = new_runtime_load_id_incarnation()
        self._next_sequence = 1
        self.base_version = str(RuntimeLoadId(self._incarnation, 0))
        self._version = self.base_version
        #: The versions the engines served when control opened this runtime: what they kept from before a
        #: control restart, which ``adopt`` takes over. Empty until ``open``.
        self.versions_at_start: frozenset[str] = frozenset()
        self._health_interval_s = health_interval_s
        self._stopping = threading.Event()
        self._watcher: threading.Thread | None = None

    def open(self) -> None:
        """Take control of the engines: attach to them or start them, note the versions that they serve, stamp the
        base model's version on engines that it started, and resume them. A second call does nothing.

        Attached engines keep their versions, so that ``adopt`` takes over what they still serve."""
        with self._lock:
            if self._opened is not None:
                return
            opened = self._engines.open()
            backend = opened.backend
            self.versions_at_start = frozenset(backend.runtime_load_ids())
            if not opened.attached:
                backend.initialize_version(self._version)
            backend.resume()
            self._opened = backend
        if self._engines.watched:
            self._watcher = threading.Thread(target=self.watch_engines, name="reef-serving-health", daemon=True)
            self._watcher.start()

    @property
    def _backend(self) -> InferenceBackend:
        if self._opened is None:
            raise ReefError("the serving runtime is not open: only control, which opens it, changes the engines")
        return self._opened

    @property
    def inference_handler(self) -> InferenceHandler:
        return self._handler

    @property
    def model_path(self) -> str:
        return self._model_path

    def next_version(self) -> str:
        version = str(RuntimeLoadId(self._incarnation, self._next_sequence))
        self._next_sequence += 1
        return version

    def watch_engines(self) -> None:
        """The health thread: recover the engines after each failed check, until the runtime shuts down."""
        while not self._stopping.wait(self._health_interval_s):
            try:
                self._engines.check_health()
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
        if self._opened is not None:
            self._engines.close()


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

    ``weights`` gives each release's adapter (by default the release's ``adapter/``). Each load names the adapter
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
        engines: ServingEngines,
        handler: InferenceHandler,
        *,
        adapter_capacity: int,
        model_path: str = "",
        inference_timeout_s: float = 300.0,
        health_interval_s: float = HEALTH_INTERVAL_S,
        weights: ReleaseWeights | None = None,
    ) -> None:
        self._residency = AdapterResidencyManager(adapter_capacity)
        #: The adapter that the engines still serve for each scenario from before this runtime (``adopt``), by
        #: scenario: its runtime load ID and the artifact it serves.
        self._adopted: dict[str, tuple[str, str]] = {}
        #: The content ID of the release that each current adapter was loaded from, by its runtime load ID.
        self._content_ids: dict[str, str] = {}
        super().__init__(
            engines,
            handler,
            model_path=model_path,
            inference_timeout_s=inference_timeout_s,
            health_interval_s=health_interval_s,
            weights=weights,
        )

    @property
    def _engine(self) -> BackendAdapterEngine:
        """The engines as the residency manager drives them; only an open runtime has them."""
        return BackendAdapterEngine(self._backend)

    @property
    def served_weight_kind(self) -> ArtifactKind:
        return "model_adapter"

    @property
    def per_scenario_adapters(self) -> bool:
        return True

    @property
    def shared_by_training_scenarios(self) -> bool:
        return True

    def served_model_name(self, scenario: str) -> str:
        """The adapter that serves ``scenario`` now, by its engine name; the base model before any adapter."""
        version = self.serving_adapter_runtime_load_id(scenario)
        return self.model_path if version is None else adapter_name(scenario, version)

    def serving_adapter_runtime_load_id(self, scenario: str) -> str | None:
        current = self._residency.current_adapter(scenario)
        return None if current is None else current.runtime_load_id

    def adapter_residency_status(self) -> Mapping[str, Any]:
        return self._residency.status()

    def adopt(self, served: ServedArtifact) -> None:
        """Take over the adapter that ``served`` names when the engines still report its version."""
        try:
            scenario, version = parse_adapter_name(served.model_name)
        except ValueError:
            return
        if version in self.versions_at_start:
            with self._lock:
                self._adopted[scenario] = (version, served.artifact_id)

    def stop_serving(self, served: ServedArtifact) -> None:
        """Unload the adapter that ``served`` names, unless a scenario serves it again."""
        try:
            parse_adapter_name(served.model_name)
        except ValueError:
            return
        with self._lock:
            self._residency.retire(served.model_name, self._engine)

    def activate_checkpoint(self, artifact: Artifact, *, scenario: str | None = None) -> str:
        """Serve the release ``artifact`` of ``scenario``; return the version that serves it.

        A release whose content the engines already serve for the scenario
        loads nothing. Any other release loads its adapter under a new
        version; a root release made from the default start holds an untrained adapter, which loads in
        the same way. Raise ``ReefError`` without ``scenario``.
        A failed load can leave the scenario serving no adapter, when the new
        adapter took the slot of its own previous one; the residency manager
        records that, so requests then sample the base model.

        When the recovery before a load replaced an engine, the replacement
        holds no adapter. The runtime then restarts every engine and loads
        each scenario's current adapter again (``restart_engines``).
        """
        if scenario is None:
            raise ReefError("per-scenario adapters are served for a named scenario")
        content_id = artifact.ref.content_id
        with self._lock:
            adopted = self._adopted.pop(scenario, None)
            if adopted is not None and adopted[1] == artifact.artifact_id:
                # The engines still serve this artifact's adapter from before a control restart: load nothing.
                path = artifact.materialize().local_path
                if path is None:
                    raise ReefError("serving a release requires a materialized artifact")
                version = adopted[0]
                self._residency.adopt(scenario, version, files=self._weights.weight_files(path, "adapter"))
                self._content_ids[version] = content_id
                return version
            current = self._residency.current_adapter(scenario)
            if current is not None and self._content_ids.get(current.runtime_load_id) == content_id:
                return current.runtime_load_id
            path = artifact.materialize().local_path
            if path is None:
                raise ReefError("serving a release requires a materialized artifact")
            files = self._weights.weight_files(path, "adapter")
            version = self.next_version()
            self._backend.pause()
            try:
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
    (``shared_by_training_scenarios`` is False). The engines start with the
    base model. Every release loads its files, a root release's included,
    unless the engines serve its content already.
    """

    def __init__(
        self,
        engines: ServingEngines,
        handler: InferenceHandler,
        *,
        model_path: str = "",
        inference_timeout_s: float = 300.0,
        health_interval_s: float = HEALTH_INTERVAL_S,
        weights: ReleaseWeights | None = None,
    ) -> None:
        self._served_content: str | None = None
        self._served_files: WeightFiles | None = None
        super().__init__(
            engines,
            handler,
            model_path=model_path,
            inference_timeout_s=inference_timeout_s,
            health_interval_s=health_interval_s,
            weights=weights,
        )

    def activate_checkpoint(self, artifact: Artifact, *, scenario: str | None = None) -> str:
        content_id = artifact.ref.content_id
        with self._lock:
            if self._served_content == content_id:
                return self._version
            path = artifact.materialize().local_path
            if path is None:
                raise ReefError("serving a release requires a materialized artifact")
            files = self._weights.weight_files(path, "full")
            version = self.next_version()
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
        """Load the served release into every engine again under its version; the base the engines start with needs none."""
        if self._served_files is None:
            return
        self._backend.load_weight_files(self._served_files, adapter_name=None, runtime_load_id=self._version)
        check_engines_serve(self._backend, self._version)
