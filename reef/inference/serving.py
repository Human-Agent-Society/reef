"""Serve saved weights on inference engines that the inference service controls directly.

Reef's serving engines are their own service, separate from every trainer.
The runtimes here are the inference service's ``InferenceRuntime`` for such a
service. They reach the engines through an ``InferenceBackend`` (for an
inference component, ``reef.inference.remote.service.RemoteInferenceBackend``)
and load only weights that a training job saved and Reef published: a
snapshot's files, when a scenario starts, when a snapshot is committed, or
when one is activated. No trainer holds a serving engine. ``ArtifactWeights``
says where a snapshot's files come from: the snapshot itself
(``ArtifactWeightFiles``), or a service that the snapshot names.

``AdapterServingRuntime`` serves each scenario its own PEFT adapter on one
base model; ``FullWeightServingRuntime`` serves one scenario's full weights.
A base root (``reef.artifact.base_root``) holds no weight files: both runtimes
answer it with the base model, and ask ``ArtifactWeights`` for nothing.
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
from reef.artifact.base_root import read_base_root
from reef.artifact.sources import HuggingFaceSource, download_huggingface_snapshot, read_reference
from reef.core.errors import ReefError
from reef.core.evaluation import ArtifactKind
from reef.core.served_models import ServedArtifact
from reef.core.weight_layout import ADAPTER_DIR, ArtifactWeights, WeightFiles, WeightLayout
from reef.inference.interfaces import (
    AdapterEngine,
    InferenceBackend,
    InferenceHandler,
    InferenceRuntime,
    RuntimeContractError,
    RuntimeLoadId,
    full_weight_name,
    new_runtime_load_id_incarnation,
)
from reef.inference.residency import (
    AdapterCapacityExhaustedError,
    AdapterEvictionFailedError,
    AdapterResidencyError,
    AdapterResidencyManager,
    ResidentAdapter,
    adapter_name,
    parse_adapter_name,
)

logger = logging.getLogger(__name__)

#: Seconds between two health checks of the serving engines' service.
HEALTH_INTERVAL_S = 30.0


@dataclass(frozen=True)
class OpenedEngines:
    """Engines that inference took: ``backend`` drives them, and ``attached`` is whether they ran already, under the
    weights and versions that a former inference worker gave them; false for engines that ``open`` started.

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
    """The engines of a serving runtime, which only the inference worker opens.

    Values: ``base_url`` is the engines' HTTP(S) address, known without opening them. ``open`` attaches to the
    engines that run, or starts them when none run (``OpenedEngines``). A runtime takes the engines of one
    successful ``open``. After an ``open`` that raised, the runtime may call it again, and that call must give
    engines in a known state: ``ComponentEngines`` then starts a new stack in place of a half-started one.
    ``check_health`` raises when the engines failed; the runtime calls it on a schedule after ``open``.
    ``request_pinning`` declares that the engines' receiver protects each request's weights.
    ``close`` releases what ``open`` took and stops no engine that serves requests.

    Purpose: a ``SavedWeightServingRuntime`` holds one and opens it in ``InferenceRuntime.open``. A process that
    runs no inference worker never opens it, so it only sends requests to ``base_url``.
    """

    @property
    @abstractmethod
    def base_url(self) -> str: ...

    @property
    def request_pinning(self) -> bool:
        """Whether the engines' receiver keeps each admitted request on the weights that the request selected.

        True declares all of the following; a ``SavedWeightServingRuntime`` refuses engines that do not declare it.
        The receiver admits a request only under a model name that the engines serve now, and refuses any other
        name before generation. Before the engines pause, load weights or unload an adapter, every admitted request
        has ended, also one whose client disconnected. When the receiver cannot confirm that a generation ended, it
        admits nothing until the engines are replaced. ``InferenceBackend.recover`` replaces every engine or none.
        The default declares nothing: an implementation returns True only for a receiver that enforces these rules.
        """
        return False

    def set_owner_sequence(self, sequence: int) -> None:
        """Set the durable serving-owner sequence for a receiver that fences control calls."""
        return

    @abstractmethod
    def open(self) -> OpenedEngines: ...

    @abstractmethod
    def check_health(self) -> None:
        """Raise when the engines failed."""

    def close(self) -> None:
        """Release what ``open`` took; the default holds nothing."""
        return


class ArtifactWeightFiles(ArtifactWeights):
    """The snapshot's own files: the snapshot directory for ``"full"``, its ``adapter/`` for ``"adapter"``.

    The files of a snapshot that refers to a Hugging Face revision (``read_reference``, RFC #995) are that
    revision's snapshot, which Hugging Face downloads once into its cache.
    """

    def weight_files(self, artifact_dir: Path, layout: WeightLayout) -> WeightFiles:
        try:
            revision = read_reference(artifact_dir)
        except ValueError as exc:
            raise ReefError(f"the snapshot in {artifact_dir} holds no readable reference: {exc}") from exc
        if revision is not None:
            artifact_dir = download_huggingface_snapshot(HuggingFaceSource(revision.repo, revision.commit)).local_path
        if layout == "full":
            return WeightFiles(artifact_dir, "full")
        adapter = artifact_dir / ADAPTER_DIR
        if not (adapter / "adapter_config.json").is_file():
            raise ReefError(f"the snapshot in {artifact_dir} holds no PEFT adapter in {ADAPTER_DIR}/")
        return WeightFiles(adapter, "adapter")


def holds_base_root(artifact_dir: Path) -> bool:
    """Whether the materialized snapshot in ``artifact_dir`` is a base root (``read_base_root``).

    Raise ``ReefError`` when its base file is not readable, so a broken base root is never served as the base.
    """
    try:
        return read_base_root(artifact_dir) is not None
    except ValueError as exc:
        raise ReefError(f"the snapshot in {artifact_dir} holds no readable base file: {exc}") from exc


def check_engines_serve(backend: InferenceBackend, version: str) -> None:
    """Raise ``RuntimeContractError`` unless every engine of ``backend`` reports exactly ``version``."""
    reported = set(backend.runtime_load_ids())
    if reported != {version}:
        raise RuntimeContractError(f"serving engines report {sorted(reported)!r} after loading {version!r}")


class SavedWeightServingRuntime(InferenceRuntime):
    """The engines, the request handler and the runtime load IDs that both serving runtimes share.

    ``engines`` reaches the engines, and ``handler`` executes requests at ``engines.base_url``. The runtimes replace
    weights in engines that keep running, so ``engines`` must declare ``ServingEngines.request_pinning``; building a
    runtime on other engines raises ``RuntimeContractError``. Building the runtime touches no engine: the inference
    worker opens it (``open``), which attaches to the engines or starts them, and stamps the base model's version on
    them. Before that, the runtime only sends requests, and an operation that changes the engines raises
    ``ReefError``. ``weights`` gives each snapshot's weight files; the default is ``ArtifactWeightFiles``.

    Runtime load IDs belong to one fresh incarnation, so a restart never reuses an ID for other weights. The base
    model serves under the first one, ``base_version``, and each load under the next unused one;
    ``activate_checkpoint`` returns it. ``loaded_version`` is the load ID of the last successful activation, which
    the runtime keeps for recovery, and ``base_version`` before any activation. A load that the engines
    acknowledged, followed by a resume that failed, leaves the earlier value. Each load first replaces the engines
    that died since the last one (``InferenceBackend.recover``), while the engines are paused.

    ``lock`` serializes every operation that changes the engines or the bindings: activation, recovery, release
    and the watcher's own recovery. ``served_artifact`` reads the bindings under it.

    After ``open``, a thread checks the engines' health every
    ``health_interval_s`` seconds, until ``shutdown`` sets ``stopping``. When the check fails, for example because
    the inference component restarted, ``recover_engines`` replaces the
    engines and loads what the runtime serves into them again, so the
    engines serve the same versions as before.

    ``recovery_pending`` is true while a recovery or an activation is unfinished, and while full-weight engines
    await their first verified activation. A failed operation leaves it true and generation paused, so the next
    health check retries the restoration even when every engine is alive. An adapter runtime also retains its
    restoration targets until every adapter is loaded again.

    A runtime that awaits its first verified activation (``awaits_first_activation``) has nothing to restore. Its
    health checks only replace failed engines, and generation stays paused until an activation succeeds.
    """

    def __init__(
        self,
        engines: ServingEngines,
        handler: InferenceHandler,
        *,
        model_path: str = "",
        inference_timeout_s: float = 300.0,
        health_interval_s: float = HEALTH_INTERVAL_S,
        weights: ArtifactWeights | None = None,
    ) -> None:
        if not engines.request_pinning:
            raise RuntimeContractError(
                "saved-weight serving replaces weights in running engines, so it needs engines whose receiver keeps "
                "each request on its weights (ServingEngines.request_pinning)"
            )
        super().__init__(base_url=engines.base_url, inference_timeout_s=inference_timeout_s)
        self._weights = ArtifactWeightFiles() if weights is None else weights
        self.engines = engines
        self._opened: InferenceBackend | None = None
        self._handler = handler
        self._model_path = model_path
        self.lock = threading.RLock()
        self._incarnation = new_runtime_load_id_incarnation()
        self._next_sequence = 1
        self.base_version = str(RuntimeLoadId(self._incarnation, 0))
        self.loaded_version = self.base_version
        self.recovery_pending = False
        self.health_interval_s = health_interval_s
        self.stopping = threading.Event()
        self._watcher: threading.Thread | None = None

    def set_owner_sequence(self, sequence: int) -> None:
        self.engines.set_owner_sequence(sequence)

    def open(self) -> None:
        """Take control of the engines: attach to them or start them, stamp the base model's version on engines that
        it started, and prepare generation. A call after a successful one does nothing; a call after one that raised
        opens the engines again (``ServingEngines.open``)."""
        with self.lock:
            if self._opened is not None:
                return
            opened = self.engines.open()
            backend = opened.backend
            if not opened.attached:
                backend.initialize_version(self.loaded_version)
            self.prepare_generation(opened)
            self._opened = backend
            self._watcher = threading.Thread(target=self.watch_engines, name="reef-serving-health", daemon=True)
            self._watcher.start()

    def prepare_generation(self, opened: OpenedEngines) -> None:
        """Resume the opened engines. Full-weight serving keeps them paused until its first verified activation."""
        opened.backend.resume()

    @property
    def backend(self) -> InferenceBackend:
        """The backend of the opened engines; ``ReefError`` before ``open``."""
        if self._opened is None:
            raise ReefError(
                "the serving runtime is not open: only the inference worker, which opens it, changes the engines"
            )
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

    @property
    def awaits_first_activation(self) -> bool:
        """Whether the runtime has no verified binding yet that ``recover_engines`` could restore.

        False, the default, for a runtime that can always restore what it serves. Full-weight serving returns True
        until its first verified activation, and again after it released its checkpoint. Only an activation can
        then resume generation.
        """
        return False

    def watch_engines(self) -> None:
        """Recover failed engines or retry an unfinished restoration, until the runtime shuts down.

        A runtime that awaits its first activation has nothing to restore. The watcher then only replaces failed
        engines, and the replacements stay paused for that activation.
        """
        while not self.stopping.wait(self.health_interval_s):
            try:
                self.engines.check_health()
                healthy = True
            except Exception:
                healthy = False
                logger.warning("the serving engines failed a health check; recovering them", exc_info=True)
            try:
                with self.lock:
                    if self.stopping.is_set():
                        return
                    if not self.awaits_first_activation:
                        if not healthy or self.recovery_pending:
                            self.recover_engines()
                    elif not healthy:
                        # Nothing loads into the replacements, so they stay paused and no route serves.
                        self.recovery_pending = True
                        self.backend.pause()
                        self.backend.recover()
            except Exception:
                logger.exception("recovering the serving engines failed; the next health check tries again")

    @abstractmethod
    def recover_engines(self) -> None:
        """Replace lost or dead engines and load the served artifacts into them again; generation pauses meanwhile.

        Restore the last verified binding after an unfinished activation or restoration, even when every engine
        is alive. A failure leaves generation paused and ``recovery_pending`` true."""

    def shutdown(self) -> None:
        self.close_admission()
        self.stopping.set()
        if self._watcher is not None:
            self._watcher.join()
        if self._opened is not None:
            self.engines.close()


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

    ``weights`` gives each snapshot's adapter (by default the snapshot's ``adapter/``). Each load names the adapter
    ``adapter_name(scenario, version)`` with a new runtime load ID
    ``version``, and every engine then reports that ID. At most
    ``adapter_capacity`` adapters stay resident. The residency manager records
    the version of each scenario's current adapter, or none. The
    runtime records the artifact ID of each current adapter. Selecting that
    artifact again loads nothing, even when its storage snapshot changes.

    A base root loads nothing. The scenario then serves no adapter, and ``base_artifact_ids`` binds the base root's
    artifact ID to the base model. Its route names the base model under ``base_version`` (``full_weight_name``):
    the engines' receiver admits that name while they serve the base model under that load ID. A scenario that has
    no adapter and no base root cannot serve.

    A new runtime takes over no adapter from earlier engines (``adopt`` is False), because the engines' load IDs
    do not prove which adapters are resident. It loads each selected artifact again.

    ``restore_adapters`` is None while no activation or engine restart is unfinished. An activation, or a recovery
    that restarts the engines, sets it to ``residency.current_adapters()`` of that moment: one adapter for each
    scenario that has one, in activation order, possibly none. ``restart_engines`` loads exactly these adapters into
    new engines and keeps the value after a partial restore, so a retry restores every scenario and not only those
    that loaded. The value is None again when generation resumes (``resume_restored``). ``recovery_pending`` is true
    whenever the value is set.
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
        weights: ArtifactWeights | None = None,
    ) -> None:
        self.residency = AdapterResidencyManager(adapter_capacity)
        #: The artifact ID of each resident adapter, indexed by its runtime load ID.
        self.loaded_artifact_ids: dict[str, str] = {}
        #: The artifact ID of the base root that each scenario serves, for the scenarios that serve no adapter.
        self.base_artifact_ids: dict[str, str] = {}
        self.restore_adapters: tuple[ResidentAdapter, ...] | None = None
        super().__init__(
            engines,
            handler,
            model_path=model_path,
            inference_timeout_s=inference_timeout_s,
            health_interval_s=health_interval_s,
            weights=weights,
        )

    @property
    def adapter_engine(self) -> BackendAdapterEngine:
        """The engines as the residency manager drives them; only an open runtime has them."""
        return BackendAdapterEngine(self.backend)

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
        """The adapter that serves ``scenario`` now, by its engine name. A scenario that serves a base root gets
        the base model under ``base_version`` (``full_weight_name``), and any other scenario the base model's path.
        ``ValueError`` for a base root when no model path names the base model."""
        version = self.serving_adapter_runtime_load_id(scenario)
        if version is not None:
            return adapter_name(scenario, version)
        if scenario in self.base_artifact_ids:
            return full_weight_name(self.model_path, self.base_version)
        return self.model_path

    def serving_adapter_runtime_load_id(self, scenario: str) -> str | None:
        current = self.residency.current_adapter(scenario)
        return None if current is None else current.runtime_load_id

    def served_artifact(self, artifact_id: str, scenario: str) -> ServedArtifact | None:
        """Publish a resident adapter bound to this artifact, or the base model for the base root that the scenario
        selected; a missing adapter is never a base-model route.

        None while recovery is pending, for any other artifact, and when ``base_url`` is no HTTP(S) URL: another
        process then has no route that it can call. None also for a base root when no model path names the base
        model."""
        with self.lock:
            if self.recovery_pending:
                return None
            version = self.serving_adapter_runtime_load_id(scenario)
            bound = self.base_artifact_ids.get(scenario) if version is None else self.loaded_artifact_ids.get(version)
            if bound != artifact_id:
                return None
            try:
                return ServedArtifact(artifact_id, self.base_url, self.served_model_name(scenario))
            except ValueError:
                # The engines' base URL is no HTTP(S) URL that another process can call, or no model path names the
                # base model.
                return None

    def adapter_residency_status(self) -> Mapping[str, Any]:
        return self.residency.status()

    def stop_serving(self, served: ServedArtifact) -> None:
        """Unload the adapter that ``served`` names, unless a scenario serves it again."""
        try:
            parse_adapter_name(served.model_name)
        except ValueError:
            return
        with self.lock:
            self.residency.retire(served.model_name, self.adapter_engine)

    def activate_checkpoint(self, artifact: Artifact, *, scenario: str | None = None) -> str:
        """Serve the snapshot ``artifact`` of ``scenario``; return the version that serves it.

        An artifact that the engines already serve for the scenario loads
        nothing. A base root (``holds_base_root``) loads nothing either: the scenario serves no adapter from
        then on, its route names the base model, and the result is ``base_version``. The adapter that the
        scenario served before stays resident without protection, for the requests that still name it, until its
        route stops (``stop_serving``) or another adapter needs its slot; ``served_artifact`` gives no route to its
        artifact. Any other artifact loads its adapter under a new version; a root snapshot of an older Reef, which
        holds an untrained adapter, loads in the same way. Raise ``ReefError`` without ``scenario``.

        A capacity refusal that changed no adapter resumes generation as before. Any other failure leaves generation
        paused and ``recovery_pending`` true, so no scenario runs on an adapter that the runtime cannot name.
        ``recover_engines``, which the next activation calls first, then restores the adapters that were current
        before the failed activation.

        When the recovery before a load replaced the engines, the replacements hold no adapter. The runtime then
        loads each scenario's current adapter again (``restart_engines``). A scenario keeps its base root through a
        failed load and a recovery, since no adapter load changes the base model.
        """
        if scenario is None:
            raise ReefError("per-scenario adapters are served for a named scenario")
        artifact_id = artifact.artifact_id
        with self.lock:
            if self.recovery_pending:
                self.recover_engines()
            current = self.residency.current_adapter(scenario)
            if current is not None and self.loaded_artifact_ids.get(current.runtime_load_id) == artifact_id:
                return current.runtime_load_id
            path = artifact.materialize().local_path
            if path is None:
                raise ReefError("serving a snapshot requires a materialized artifact")
            if holds_base_root(path):
                self.residency.serve_base(scenario)
                self.base_artifact_ids[scenario] = artifact_id
                return self.base_version
            files = self._weights.weight_files(path, "adapter")
            version = self.next_version()
            self.restore_adapters = self.residency.current_adapters()
            self.recovery_pending = True
            self.backend.pause()
            try:
                if self.backend.recover():
                    self.restart_engines(engines_replaced=True)
                # Generation is paused, so the scenario's own adapter may make room for its new one.
                self.residency.activate(scenario, version, self.adapter_engine, files=files, supersede=True)
                self.loaded_artifact_ids[version] = artifact_id
                self.base_artifact_ids.pop(scenario, None)
                self.resume_restored()
            except AdapterEvictionFailedError:
                # The engine kept an adapter that the runtime tried to unload, so recovery must restore the bindings.
                raise
            except AdapterCapacityExhaustedError:
                # A plain capacity refusal changed no binding.
                self.resume_restored()
                raise
            self.loaded_version = version
            return version

    def resume_restored(self) -> None:
        """Resume generation once every binding is known, and forget the artifacts of adapters that are gone."""
        self.backend.resume()
        self.recovery_pending = False
        self.restore_adapters = None
        resident = {adapter.runtime_load_id for adapter in self.residency.current_adapters()}
        self.loaded_artifact_ids = {key: value for key, value in self.loaded_artifact_ids.items() if key in resident}

    def recover_engines(self) -> None:
        """Restore the whole residency after lost engines or a partial restore; resume only when every adapter is
        known."""
        with self.lock:
            retry_reload = self.recovery_pending
            self.recovery_pending = True
            self.backend.pause()
            replaced = self.backend.recover()
            if replaced or retry_reload:
                self.restart_engines(engines_replaced=bool(replaced))
            self.resume_restored()

    def restart_engines(self, *, engines_replaced: bool = False) -> None:
        """Retire every engine, start them all again and load each scenario's current adapter; engines are paused.

        The adapters are those of ``restore_adapters`` when an earlier restore is unfinished.
        A scenario whose adapter fails to load serves no adapter afterwards; the other scenarios still load.
        Raise ``AdapterResidencyError`` after the loop when one failed. ``engines_replaced`` means that recovery
        already replaced every engine, so no second restart is needed.
        """
        if self.restore_adapters is None:
            self.restore_adapters = self.residency.current_adapters()
        if not engines_replaced:
            self.backend.abort()
            self.backend.recover()
        self.residency = AdapterResidencyManager(self.residency.capacity)
        failed: list[str] = []
        for adapter in self.restore_adapters:
            try:
                self.residency.activate(
                    adapter.scenario, adapter.runtime_load_id, self.adapter_engine, files=adapter.files
                )
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
    base model and stay paused until the first verified activation, so no request runs on weights that the runtime
    cannot name. Every snapshot loads its files, a root snapshot's included,
    unless the engines serve that artifact already. A base root has no files
    of its own: it loads the base model's files, the directory ``model_path``
    (``base_weight_files``), under a new version, and binds the base root's artifact as any activation does.

    ``served_artifact_id`` and ``served_files`` are both None before the first successful activation. Afterwards,
    they name the last successful activation's artifact and files, with its load ID in ``loaded_version``. A failed
    load or load-ID check leaves generation paused and ``recovery_pending`` true. The last successful artifact remains
    the recovery target, but it must load again before requests resume. With no successful activation yet, only an
    explicit activation can establish that target. Until then, the health watcher attempts no restoration.
    ``release_checkpoint`` returns the runtime to that first state: both values are None again, and the engines
    stay paused and allocated for a later activation.
    """

    def __init__(
        self,
        engines: ServingEngines,
        handler: InferenceHandler,
        *,
        model_path: str = "",
        inference_timeout_s: float = 300.0,
        health_interval_s: float = HEALTH_INTERVAL_S,
        weights: ArtifactWeights | None = None,
    ) -> None:
        self.served_artifact_id: str | None = None
        self.served_files: WeightFiles | None = None
        super().__init__(
            engines,
            handler,
            model_path=model_path,
            inference_timeout_s=inference_timeout_s,
            health_interval_s=health_interval_s,
            weights=weights,
        )

    def prepare_generation(self, opened: OpenedEngines) -> None:
        """Keep the opened engines paused until this runtime verifies an activation.

        New engines hold the base model, which is no published artifact. An earlier worker may have changed the
        weights of attached engines without updating their load IDs. In both cases this runtime has no verified
        artifact yet, so it resumes nothing based on the engines' load IDs alone.
        """
        self.recovery_pending = True
        opened.backend.pause()

    @property
    def awaits_first_activation(self) -> bool:
        """True until the first verified activation, and again after ``release_checkpoint``: no artifact is known,
        so recovery has nothing to restore."""
        return self.served_files is None

    def served_model_name(self, scenario: str) -> str:
        """The route name of the current load (``full_weight_name``), which the receiver admits only while the
        engines serve that load."""
        return full_weight_name(self.model_path, self.loaded_version)

    def served_artifact(self, artifact_id: str, scenario: str) -> ServedArtifact | None:
        """The route for the last verified activation, or None while its binding needs restoration.

        Also None for any other artifact, and when ``base_url`` is no HTTP(S) URL or the route has no model name:
        another process then has no route that it can call."""
        with self.lock:
            if self.recovery_pending or self.served_artifact_id != artifact_id:
                return None
            try:
                return ServedArtifact(artifact_id, self.base_url, self.served_model_name(scenario))
            except ValueError:
                # The engines' base URL is no HTTP(S) URL, or they serve no model name.
                return None

    def activate_checkpoint(self, artifact: Artifact, *, scenario: str | None = None) -> str:
        artifact_id = artifact.artifact_id
        with self.lock:
            if self.served_artifact_id == artifact_id and not self.recovery_pending:
                return self.loaded_version
            path = artifact.materialize().local_path
            if path is None:
                raise ReefError("serving a snapshot requires a materialized artifact")
            files = self.base_weight_files() if holds_base_root(path) else self._weights.weight_files(path, "full")
            version = self.next_version()
            backend = self.backend
            self.recovery_pending = True
            backend.pause()
            # The load writes every engine, so an engine that recovery replaced serves the snapshot too.
            backend.recover()
            backend.load_weight_files(files, adapter_name=None, runtime_load_id=version)
            check_engines_serve(backend, version)
            backend.resume()
            self.served_artifact_id = artifact_id
            self.served_files = files
            self.loaded_version = version
            self.recovery_pending = False
            return version

    def base_weight_files(self) -> WeightFiles:
        """The base model's files, which a base root loads: the directory ``model_path``.

        Raise ``ReefError`` when ``model_path`` is no directory on this host. The engines then cannot load the
        base model again after another artifact, since a load sends them a directory of this host.
        """
        directory = Path(self.model_path).expanduser()
        if not self.model_path or not directory.is_dir():
            raise ReefError(
                f"a base root serves the base model, but the model path {self.model_path!r} is no directory on "
                "this host, so the engines cannot load it"
            )
        return WeightFiles(directory, "full")

    def release_checkpoint(self) -> None:
        """Stop serving the bound checkpoint and bind none, as before the first verified activation.

        The engines pause first and keep running with the weights that they hold, so the call stops no engine and
        frees no GPU memory. After the pause, ``served_artifact`` returns None for every artifact, recovery has
        nothing to restore, and only an activation resumes generation; it loads under a new runtime load ID. A
        pause that raises keeps the checkpoint as the recovery target, with ``recovery_pending`` true until a
        recovery or an activation loads it again. A runtime that binds no checkpoint only pauses its engines
        again, so engines that an unfinished first activation left stay paused.
        """
        with self.lock:
            backend = self.backend
            # As in an activation, the binding is uncertain from here until the pause is confirmed.
            self.recovery_pending = True
            backend.pause()
            self.served_artifact_id = None
            self.served_files = None

    def reload_served(self) -> None:
        """Restore the last successfully activated snapshot under its version, or refuse when no snapshot is known."""
        if self.served_files is None:
            raise ReefError("no verified full-weight artifact to restore; activate a published artifact first")
        self.backend.load_weight_files(self.served_files, adapter_name=None, runtime_load_id=self.loaded_version)
        check_engines_serve(self.backend, self.loaded_version)

    def recover_engines(self) -> None:
        """Restore the last successful activation after lost engines or an unfinished load, then resume.

        An unfinished load requires restoration even when every engine is alive. A failure keeps recovery pending
        and does not resume generation; the next health check tries again. Before the first verified activation,
        the call replaces the lost engines and then raises ``ReefError``, because no artifact is known; the health
        watcher does not make this call.
        """
        with self.lock:
            backend = self.backend
            retry_reload = self.recovery_pending
            self.recovery_pending = True
            backend.pause()
            replaced = backend.recover()
            if retry_reload or (replaced and self.served_files is not None):
                self.reload_served()
            backend.resume()
            self.recovery_pending = False
