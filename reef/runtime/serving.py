"""Serve saved adapters on inference engines that the control plane controls directly.

Reef's serving engines are their own service, separate from every trainer.
``AdapterServingRuntime`` is the control plane's ``InferenceRuntime`` for such
a service. It reaches the engines through an ``InferenceBackend`` (for an
inference component, ``reef.inference.remote.service.RemoteInferenceBackend``)
and loads only adapters that a training job saved: a candidate's checkpoint
when Reef selects it, and a release's artifact when a scenario starts or rolls
back. No trainer holds a serving engine.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reef.artifact.artifact import REPOSITORY_FILES, Artifact
from reef.core.errors import ReefError
from reef.runtime.deployment import InferenceService
from reef.runtime.interfaces import (
    ActivatedModel,
    AdapterEngine,
    InferenceBackend,
    InferenceHandler,
    InferenceRuntime,
    ModelCandidate,
    RuntimeLoadId,
    TrainingRuntimeError,
    WeightFiles,
    new_runtime_load_id_incarnation,
)
from reef.runtime.publication import AdapterResidencyManager
from reef.surface.adapter import parse_adapter_name

#: The directory of a saved checkpoint that holds its PEFT adapter.
ADAPTER_DIR = "adapter"


@dataclass(frozen=True)
class ServedAdapter:
    """The adapter the engines serve for one scenario.

    ``runtime_load_id`` is the non-empty version it was loaded under, in this
    runtime's incarnation. ``content_id`` is the content ID of the release that
    holds it, or ``None`` while it is a candidate that no release holds yet.
    """

    runtime_load_id: str
    content_id: str | None


@dataclass(frozen=True)
class PendingCandidate:
    """A selected candidate that the engines serve and that Reef has not committed yet.

    ``training_job_id`` is the job that saved it, ``scenario`` the scenario it
    belongs to, and ``runtime_load_id`` the version it was loaded under.
    """

    training_job_id: str
    scenario: str
    runtime_load_id: str


class AdapterServingRuntime(InferenceRuntime, AdapterEngine):
    """Load each scenario's saved PEFT adapter into the serving engines and serve requests with it.

    A saved checkpoint holds its adapter in ``adapter/``. Each load names the
    adapter ``adapter_name(scenario, version)`` with a new runtime load ID
    ``version``, and every engine then reports that ID. At most
    ``adapter_capacity`` adapters stay resident; a scenario that has loaded
    nothing samples the base model.

    ``backend`` controls the engines and ``handler`` executes requests at
    ``backend.inference_url()``. ``service``, when given, owns the engines and
    is closed on shutdown. Runtime load IDs belong to one fresh incarnation,
    so a restart never reuses an ID for other weights.
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
    ) -> None:
        super().__init__(base_url=backend.inference_url(), inference_timeout_s=inference_timeout_s)
        self._backend = backend
        self._handler = handler
        self._service = service
        self._model_path = model_path
        self._lock = threading.RLock()
        self._incarnation = new_runtime_load_id_incarnation()
        self._next_sequence = 1
        self._residency = AdapterResidencyManager(adapter_capacity)
        self._served: dict[str, ServedAdapter] = {}
        self._pending: PendingCandidate | None = None
        self._version = str(RuntimeLoadId(self._incarnation, 0))
        # The engines start paused, with the base model.
        backend.initialize_version(self._version)
        backend.resume()

    @property
    def inference_handler(self) -> InferenceHandler:
        return self._handler

    @property
    def model_path(self) -> str:
        return self._model_path

    # -- Serving version

    def serving_runtime_load_id(self) -> str | None:
        versions = set(self._backend.runtime_load_ids())
        return versions.pop() if len(versions) == 1 else None

    def serving_adapter_runtime_load_id(self, scenario: str) -> str | None:
        with self._lock:
            served = self._served.get(scenario)
        return None if served is None else served.runtime_load_id

    def adapter_residency_status(self) -> Mapping[str, Any]:
        return self._residency.status()

    # -- Loading saved adapters

    def activate_candidate(self, candidate: ModelCandidate) -> ActivatedModel:
        """Load the selected candidate's adapter; ``candidate.metadata["scenario"]`` names its scenario."""
        scenario = candidate.metadata.get("scenario")
        if not isinstance(scenario, str) or not scenario:
            raise TrainingRuntimeError("a candidate for per-scenario adapters must name its scenario")
        if candidate.checkpoint_path is None:
            raise TrainingRuntimeError("only a candidate with a saved checkpoint can be served")
        with self._lock:
            pending = self._pending
            if pending is not None and pending.training_job_id == candidate.training_job_id:
                return ActivatedModel(candidate.candidate_id, pending.runtime_load_id)
            version = self.load_saved_adapter(scenario, Path(candidate.checkpoint_path), content_id=None)
            self._pending = PendingCandidate(candidate.training_job_id, scenario, version)
            return ActivatedModel(candidate.candidate_id, version)

    def acknowledge_publication(self, training_job_id: str) -> None:
        with self._lock:
            pending = self._pending
            if pending is None:
                return
            if pending.training_job_id != training_job_id:
                raise TrainingRuntimeError(
                    f"serving holds the candidate of job {pending.training_job_id!r}, "
                    f"not the committed job {training_job_id!r}"
                )
            self._pending = None

    def restore_checkpoint(self, artifact: Artifact, *, scenario: str | None = None) -> str:
        """Serve a rolled-back release of ``scenario``; return the version that serves it."""
        return self.activate_checkpoint(artifact, scenario=scenario)

    def activate_checkpoint(self, artifact: Artifact, *, scenario: str | None = None) -> str:
        """Serve the final release ``artifact`` of ``scenario``; return the version that serves it.

        A release that holds the adapter the engines already serve for the
        scenario, such as the release of the selected candidate, loads nothing.
        A release without weights serves the base model, and any other release
        loads its adapter under a new version.
        """
        if scenario is None:
            raise ReefError("per-scenario adapters are served for a named scenario")
        path = artifact.materialize().local_path
        if path is None:
            raise ReefError("serving a release requires a materialized artifact")
        content_id = artifact.ref.content_id
        with self._lock:
            served = self._served.get(scenario)
            if served is not None and (
                served.content_id == content_id or artifact.metadata.get("runtime_load_id") == served.runtime_load_id
            ):
                self._served[scenario] = ServedAdapter(served.runtime_load_id, content_id)
                return served.runtime_load_id
            if self._pending is not None and self._pending.scenario == scenario:
                # Another release replaces a candidate that Reef did not commit.
                self._pending = None
            if not {entry.name for entry in path.iterdir()} - REPOSITORY_FILES:
                # Requests for a release without weights sample the base model.
                self._served.pop(scenario, None)
                return self._version
            return self.load_saved_adapter(scenario, path, content_id=content_id)

    def load_saved_adapter(self, scenario: str, checkpoint: Path, *, content_id: str | None) -> str:
        adapter = checkpoint / ADAPTER_DIR
        if not (adapter / "adapter_config.json").is_file():
            raise ReefError(f"saved checkpoint {checkpoint} holds no PEFT adapter in {ADAPTER_DIR}/")
        version = str(RuntimeLoadId(self._incarnation, self._next_sequence))
        self._next_sequence += 1
        self._backend.pause()
        try:
            # Generation is paused, so the scenario's own adapter may make room for its new one.
            self._residency.activate(scenario, version, self, payload=WeightFiles(adapter, "adapter"), supersede=True)
        finally:
            self._backend.resume()
        self._served[scenario] = ServedAdapter(version, content_id)
        self._version = version
        return version

    # -- Adapter engine, for the residency manager

    def load_adapter(self, name: str, payload: Any) -> None:
        if not isinstance(payload, WeightFiles):
            raise TypeError("an adapter load needs the adapter's WeightFiles")
        _, version = parse_adapter_name(name)
        self._backend.load_weight_files(payload, adapter_name=name, runtime_load_id=version)
        reported = set(self._backend.runtime_load_ids())
        if reported != {version}:
            raise TrainingRuntimeError(f"serving engines report {sorted(reported)!r} after loading {version!r}")

    def unload_adapter(self, name: str) -> None:
        self._backend.unload_adapter(name)

    def shutdown(self) -> None:
        self.pause_admission()
        if self._service is not None:
            self._service.close()
