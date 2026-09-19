"""Shared training and inference contracts, values, and default request admission.

Who implements what:

- Recipes and the dispatcher *use* ``InferenceRuntime`` and ``TrainingRuntime``
  and the values they exchange (``ModelCandidate``, ``ActivatedModel``,
  ``PreparedTrainingStep``, ``TrainingJobResult``). These are the backbone.
- A training integration *implements* ``TrainingBackend`` (with
  ``PreparedTrainingJob``) so Reef's coordinator can drive it.
- An inference integration *implements* ``InferenceHandler`` for requests,
  ``InferenceBackend`` for weight receipt, and ``InferenceMemoryOperations``
  and ``AdapterEngine`` for the memory and adapter operations ``scheduler``
  and ``publication`` call back into. The engine supervision hooks live in
  ``recovery`` next to the objects that drive them.
- Reef itself implements the durable stores (``TrainingJobStore``,
  ``ScenarioHistoryStore``); they are abstract here only so ``publication``
  and ``recovery`` can share them without importing each other.

The module reads top-down in that order: identities and errors, training
values, request admission, request handling, runtime contracts, native
backend contracts, durable stores.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractContextManager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from threading import Condition
from typing import Any, Literal

from reef.artifact.artifact import Artifact
from reef.core.batches import StepScheduling, TrainingBatch
from reef.core.errors import ReefError
from reef.core.evaluation import SelectionDecision, UpdateCandidate
from reef.surface.base import AdapterWeightRuntime, InferenceLease

# -- Identities and errors ----------------------------------------------------


def new_runtime_load_id_incarnation() -> str:
    """Return a fresh namespace for Reef's serving-version sequence."""
    return uuid.uuid4().hex


@dataclass(frozen=True)
class RuntimeLoadId:
    """Globally unique identity for one serving-runtime weight load."""

    incarnation: str
    sequence: int

    def __post_init__(self) -> None:
        if not self.incarnation or ":" in self.incarnation:
            raise ValueError("runtime-load incarnation must be non-empty and contain no ':'")
        if not isinstance(self.sequence, int) or isinstance(self.sequence, bool) or self.sequence < 0:
            raise ValueError("runtime-load sequence must be a non-negative integer")

    def __str__(self) -> str:
        return f"{self.incarnation}:{self.sequence}"

    @classmethod
    def parse(cls, value: str) -> RuntimeLoadId:
        if not isinstance(value, str):
            raise TypeError("runtime load id must be a string")
        incarnation, separator, sequence = value.rpartition(":")
        if not separator or not sequence.isascii() or not sequence.isdecimal():
            raise ValueError("runtime load id must use '<incarnation>:<sequence>'")
        return cls(incarnation, int(sequence))


class RuntimeContractError(ReefError):
    """A runtime or backend violated its contract with Reef.

    Raised for malformed runtime results and missing capabilities that a
    correctly configured deployment would never produce — distinct from user
    input errors, which surface as more specific ``ReefError`` subclasses.
    """


class TrainingRuntimeError(ReefError):
    """Raised when a training backend violates the runtime contract."""


class UpstreamStatusError(ReefError):
    """An upstream service answered an inference request with an error status.

    Carries ``status`` so the service can hand the caller the upstream's own
    4xx and message rather than collapsing both into an opaque 500. That
    distinction is load-bearing: agents correct themselves from these bodies
    (shrinking ``max_tokens`` when the engine reports a context overflow, for
    instance), and an opaque 500 leaves them retrying the identical request
    until they give up.
    """

    def __init__(self, message: str, *, status: int) -> None:
        super().__init__(message)
        self.status = status


class CandidateTrainingDeferred(Exception):
    """A candidate was not produced, but retrying the same batch is safe."""

    def __init__(self, storage: Mapping[str, Any]) -> None:
        super().__init__("candidate training is waiting for checkpoint storage")
        self.storage = dict(storage)


class StaleCandidate(Exception):
    """The reserved batch became stale before it could produce a candidate."""

    def __init__(self, metrics: Mapping[str, Any] | None = None) -> None:
        super().__init__("candidate training rejected a stale batch")
        self.metrics = dict(metrics or {})


# -- Training values ----------------------------------------------------------


def _require_non_empty(value: object, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True, kw_only=True)
class ModelCandidate(UpdateCandidate):
    """A checkpointed model update that has not changed serving weights."""

    training_job_id: str
    checkpoint_path: str
    current_runtime_load_id: str | None
    training_metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        _require_non_empty(self.training_job_id, "training_job_id")
        _require_non_empty(self.checkpoint_path, "checkpoint_path")
        if self.current_runtime_load_id is not None and (
            not isinstance(self.current_runtime_load_id, str) or not self.current_runtime_load_id
        ):
            raise ValueError("current_runtime_load_id must be a non-empty string or None")


@dataclass(frozen=True)
class ActivatedModel:
    """Serving identity returned after a selected candidate is activated."""

    candidate_id: str
    runtime_load_id: str

    def __post_init__(self) -> None:
        _require_non_empty(self.candidate_id, "candidate_id")
        _require_non_empty(self.runtime_load_id, "runtime_load_id")


@dataclass(frozen=True, slots=True)
class TrainingJobResult:
    """Outcome of one restart-safe training job.

    ``metrics`` is backend telemetry carried opaquely, the same contract as
    ``TrainStepResult.metrics``: the backend that produced it owns the schema,
    reef never interprets it, and it reaches the commit record so per-step
    metrics remain available when the resulting version is served.
    """

    outcome: Literal["complete", "checkpoint", "stale", "storage_blocked"]
    runtime_load_id: str
    checkpoint_path: str | None = None
    storage: Mapping[str, Any] | None = None
    metrics: Mapping[str, Any] | None = None
    training_job_id: str | None = None

    def __post_init__(self) -> None:
        # Fail closed at the boundary: a completed job that cannot name the
        # checkpoint it exported would otherwise reach the committer and
        # be published as a durable version pointing at nothing.
        if self.outcome in {"complete", "checkpoint"} and not self.checkpoint_path:
            raise ValueError(f"a {self.outcome} training job must report the checkpoint path it exported")
        if not self.runtime_load_id:
            raise ValueError("a training job result must report a runtime load ID")
        if self.training_job_id is not None and (
            not isinstance(self.training_job_id, str) or not self.training_job_id
        ):
            raise ValueError("training_job_id must be a non-empty string or None")


@dataclass(frozen=True, slots=True)
class PreparedTrainingStep:
    """Backend-prepared work for one reserved Reef training batch.

    Algorithm selection, data-only signal construction, and backend payload
    shaping all happen before this value crosses back into Reef's dispatcher.
    Reef keeps ownership of the reserved batch and commits ``next_algorithm_state``
    only after the backend job succeeds. ``skip`` supports state-only algorithm
    transitions without handing a payload to the executor.
    """

    action: Literal["train", "skip"]
    next_algorithm_state: Mapping[str, Any]
    metrics: Mapping[str, Any]
    payload: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.action == "train" and self.payload is None:
            raise ValueError("a train step requires a backend payload")
        if self.action == "skip" and self.payload is not None:
            raise ValueError("a skipped step cannot carry a backend payload")


@dataclass(frozen=True)
class TrainingCheckpoint:
    """Backend-selected checkpoint index and path, optionally scoped to a scenario."""

    rollout_id: int
    path: Path
    scenario: str | None = None
    scenario_step: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.rollout_id, int) or isinstance(self.rollout_id, bool) or self.rollout_id < 0:
            raise ValueError("checkpoint rollout_id must be non-negative")
        if self.scenario is not None:
            if not isinstance(self.scenario, str) or not self.scenario:
                raise ValueError("checkpoint scenario must be non-empty")
            if (
                not isinstance(self.scenario_step, int)
                or isinstance(self.scenario_step, bool)
                or self.scenario_step < 0
            ):
                raise ValueError("checkpoint scenario_step must be non-negative")
        elif self.scenario_step is not None:
            raise ValueError("checkpoint scenario_step requires a scenario")


@dataclass(frozen=True)
class TrainingMetrics:
    """Worker/loss metrics and durable method telemetry from one optimizer step."""

    training: Mapping[str, Any] = field(default_factory=dict)
    durable: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class TrainingJobState:
    """Health phase; durable recovery decisions always come from the job marker."""

    phase: str = "serving"


# -- Request admission --------------------------------------------------------


class InferenceAdmissionHandle(InferenceLease):
    """A handle for one admitted inference, released after model execution."""

    def __init__(self, controller: InferenceAdmissionController) -> None:
        self._controller = controller
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._controller.release()


class InferenceAdmissionController:
    """Thread-safe inference admission shared by async requests and training.

    Requests await one event-loop-local event without occupying worker
    threads. Training closes admission from its serial worker thread; an
    integration may optionally drain admitted work before a destructive
    backend operation.
    """

    def __init__(self) -> None:
        self._condition = Condition()
        self._open = True
        self._active = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._open_event: asyncio.Event | None = None

    async def acquire(self) -> InferenceAdmissionHandle:
        """Wait until admission is open, then count one active request."""
        loop = asyncio.get_running_loop()
        while True:
            with self._condition:
                event = self._bind_loop(loop)
                if self._open:
                    self._active += 1
                    return InferenceAdmissionHandle(self)
            await event.wait()
            # ``close`` clears the asyncio event on its owning loop. If this
            # task raced the thread-safe callback, a still-set event would
            # otherwise make this loop spin without yielding and prevent the
            # clear callback from ever running.
            await asyncio.sleep(0)

    def _bind_loop(self, loop: asyncio.AbstractEventLoop) -> asyncio.Event:
        """Return the open event for ``loop``, creating it on first use. Caller holds the lock."""
        if self._loop is None or self._loop.is_closed():
            self._loop = loop
            self._open_event = asyncio.Event()
            if self._open:
                self._open_event.set()
        elif self._loop is not loop:
            raise RuntimeError("inference admission cannot span concurrent event loops")
        if self._open_event is None:
            raise RuntimeError("closed inference admission has no loop event")
        return self._open_event

    def close(self, *, wait: bool = False, timeout: float | None = None) -> None:
        """Reject new admissions and optionally drain already admitted work."""
        with self._condition:
            self._open = False
            self._signal_loop(is_open=False)
            if wait and not self._condition.wait_for(lambda: self._active == 0, timeout=timeout):
                raise TimeoutError("timed out waiting for admitted inference requests to drain")

    def open(self) -> None:
        """Admit queued and future requests."""
        with self._condition:
            self._open = True
            self._condition.notify_all()
            self._signal_loop(is_open=True)

    def _signal_loop(self, *, is_open: bool) -> None:
        """Mirror the open flag onto the loop's event, if a loop is still running."""
        loop, event = self._loop, self._open_event
        if loop is None or event is None or loop.is_closed():
            return
        # The loop can close between is_closed() and scheduling.
        with suppress(RuntimeError):
            loop.call_soon_threadsafe(event.set if is_open else event.clear)

    @property
    def status(self) -> Mapping[str, Any]:
        with self._condition:
            return {"open": self._open, "active": self._active}

    def release(self) -> None:
        """Count one admitted request as finished; wakes a draining ``close``."""
        with self._condition:
            if self._active <= 0:
                raise RuntimeError("inference admission handle released without an active request")
            self._active -= 1
            if self._active == 0:
                self._condition.notify_all()


# -- Request handling ---------------------------------------------------------


class InferenceStream:
    """One open provider response whose bytes can be forwarded incrementally."""

    def __init__(
        self,
        *,
        status: int,
        headers: Mapping[str, str],
        chunks: AsyncIterator[bytes],
        close: Callable[[], Awaitable[None]] | None = None,
        record_response: Mapping[str, Any] | None = None,
        record_response_pending: bool = False,
    ) -> None:
        self.status = status
        self.headers = dict(headers)
        self.chunks = chunks
        self._close = close
        self._closed = False
        self.record_response = None if record_response is None else dict(record_response)
        # Some custom handlers can only construct their exact, provider-neutral
        # recording response after the upstream stream reaches its terminal
        # event. RequestService keeps durable admission open for those streams
        # and validates the completed capture before accepting the record.
        self.record_response_pending = record_response_pending

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._close is not None:
            await self._close()


class InferenceHandler(ABC):
    """Execute inference for a selected artifact without implicitly materializing it."""

    @classmethod
    def from_config(
        cls,
        upstream_url: str,
        *,
        model_path: str,
        timeout_s: float,
        **config: Any,
    ) -> InferenceHandler:
        """Construct a configured handler; direct injection needs only inference()."""
        raise ValueError(f"{cls.__name__} does not support deployment configuration")

    def reconnect(self, upstream_url: str) -> None:
        """Retarget a managed endpoint, preserving handler-specific configuration."""
        raise RuntimeError(f"{type(self).__name__} does not support inference endpoint replacement")

    @abstractmethod
    async def inference(self, artifact: Artifact, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Return the provider response for one native request payload."""

    async def inference_stream(self, artifact: Artifact, path: str, payload: dict[str, Any]) -> InferenceStream:
        """Return a stream for one native request payload.

        Custom handlers that only implement buffered inference keep working: the
        default implementation exposes their JSON response as one chunk. HTTP
        handlers override this method to preserve provider-native streaming.
        """
        value = await self.inference(artifact, path, payload)

        async def chunks() -> AsyncIterator[bytes]:
            yield json.dumps(value, ensure_ascii=False).encode()

        return InferenceStream(status=200, headers={"Content-Type": "application/json"}, chunks=chunks())


#: The routes a client calls on Reef for a multimodal call; a recipe's relay decides which its provider serves.
MULTIMODAL_ROUTES: tuple[str, ...] = ("/v1/images", "/v1/embeddings", "/v1/audio/speech", "/v1/decisions")


class MultimodalRelay(ABC):
    """Relay a recipe's multimodal calls (images, embeddings, speech, decisions) to the provider it configured.

    A recipe that offers one keeps the provider's address and key; Reef only
    forwards the client's request body and streams the answer back unchanged,
    and records nothing.
    """

    @abstractmethod
    async def relay(self, path: str, payload: dict[str, Any]) -> InferenceStream:
        """Forward one request body to the provider and return its answer as a stream; raise
        :class:`NotImplementedError` for a route the provider does not serve."""


# -- Runtime contracts --------------------------------------------------------


class InferenceRuntime(AdapterWeightRuntime):
    """Own inference requests, serving weights and admission.

    An InferenceHandler executes individual requests. This runtime owns that
    handler and its endpoint, weight activation and serving version state;
    training and optimizer state belong to a separate TrainingRuntime.
    """

    def __init__(self, *, base_url: str, inference_timeout_s: float = 300.0) -> None:
        if not base_url:
            raise ValueError("base_url must be non-empty")
        if inference_timeout_s <= 0:
            raise ValueError("inference_timeout_s must be positive")
        self._base_url = base_url.rstrip("/")
        self._inference_timeout_s = inference_timeout_s
        self._inference_admission = InferenceAdmissionController()
        self._current_runtime_load_id: str | None = None

    # -- Requests

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def inference_timeout_s(self) -> float:
        return self._inference_timeout_s

    @property
    @abstractmethod
    def inference_handler(self) -> InferenceHandler:
        """The inference request handler owned by this runtime."""

    def reconnect(self, base_url: str) -> None:
        """Retarget the request handler after managed inference recovery."""
        if not base_url:
            raise ValueError("base_url must be non-empty")
        self.inference_handler.reconnect(base_url)
        self._base_url = base_url.rstrip("/")

    def shutdown(self) -> None:
        """Release owned resources after all users of this runtime have stopped.

        Shared or injected runtimes are closed by their owner, never by an
        individual scenario. Proxy-only runtimes have no local resources.
        """
        return

    # -- Admission

    async def acquire_inference(self) -> InferenceAdmissionHandle:
        """Wait until this runtime may freeze and execute a new inference."""
        return await self._inference_admission.acquire()

    @property
    def inference_admission_status(self) -> Mapping[str, Any]:
        return self._inference_admission.status

    def pause_admission(self, *, wait: bool = False, timeout: float | None = None) -> None:
        """Close request admission; optionally wait for admitted work to finish."""
        self._inference_admission.close(wait=wait, timeout=timeout)

    def resume_admission(self) -> None:
        """Reopen request admission after the coordinator permits serving."""
        self._inference_admission.open()

    # -- Serving version

    def serving_runtime_load_id(self) -> str | None:
        """The runtime load ID the serving engine currently reports, if knowable.

        A read-only probe used at recovery to detect an engine that disagrees
        with the recovered head. ``None`` means the runtime cannot tell;
        callers must treat that as "unverified", never as "matching".

        The returned value is an opaque engine-side token, not a durable
        artifact identity. Backends must namespace monotonic counters by a
        fresh serving incarnation so a restart cannot reuse an old token for
        different weights. Equality identifies the same published engine
        update; it does not make that update's bytes durable.
        """
        return None

    def current_runtime_load_id(self) -> str | None:
        """Return the version acknowledged as published by Reef."""
        return self._current_runtime_load_id

    def mark_published(self) -> None:
        """Record the loaded version after the durable publication handshake."""
        self._current_runtime_load_id = self.serving_runtime_load_id()

    # -- Adapters

    def serving_adapter_name(self) -> str | None:
        """Name of the one adapter the serving engine applies, if it serves one.

        A LoRA deployment trains an adapter over frozen base weights, and the
        engine applies it only when a request names it. Reporting the name
        here is what lets the weight surface address every request to it, so
        no harness can silently sample the frozen base. ``None`` means the
        runtime publishes full weights and requests need no adapter name, or
        that adapters are per scenario.
        """
        return None

    def serving_adapter_runtime_load_id(self, scenario: str) -> str | None:
        """The serving runtime load ID of ``scenario``'s resident adapter.

        ``None`` when the runtime does not serve per-scenario adapters or the
        scenario has published nothing yet (requests then sample the base).
        """
        return None

    def adapter_residency_status(self) -> Mapping[str, Any] | None:
        return None

    # -- Weight updates

    def restore_checkpoint(self, artifact: Artifact) -> str:
        """Restore served weights from a durable artifact.

        Runtimes that support weight rollback override this and return the new
        serving-engine version token. The default fails explicitly: silently
        moving Reef's artifact head while the engine keeps newer weights would
        corrupt serving-version records.
        """
        raise ReefError(f"{type(self).__name__} does not support checkpoint restore")

    def activate_candidate(self, candidate: ModelCandidate) -> ActivatedModel:
        """Load a selected checkpoint/adapter without authorizing new requests."""
        raise ReefError(f"{type(self).__name__} does not support candidate activation")

    def resume_weight_update(self, training_job_id: str) -> ActivatedModel:
        """Resume an interrupted receiver update using its durable identity."""
        raise ReefError(f"{type(self).__name__} does not support weight-update recovery")

    def acknowledge_publication(self, training_job_id: str) -> None:
        """Confirm the durable head to an engine with deferred publication."""
        return


class TrainingRuntime(ABC):
    """Training preparation and checkpoint production, independent of inference.

    Serving versions are input values, never an inference runtime dependency.
    The existing training backend coordinates training with inference and Reef
    publication. Implementations never receive an inference runtime object.
    """

    @property
    def max_staleness(self) -> int:
        """Largest producing-to-serving version lag this runtime admits.

        Exact-version admission is the default. Runtimes that support a
        positive bounded-staleness window override this property.
        """
        return 0

    @property
    def concurrent_training_scenarios(self) -> bool:
        """Whether several scenarios may train on this runtime at once.

        A per-scenario LoRA runtime keeps one frozen base and time-slices its
        adapter slot between scenarios, publishing each under a
        scenario-qualified name; the dispatcher then drives every training
        scenario instead of binding the process to one.
        """
        return False

    def training_job_status(self) -> Mapping[str, Any] | None:
        """Remote durable job state, or None for an in-process candidate trainer."""
        return None

    def reject_training_job(self, training_job_id: str) -> None:
        """Resume rejection of a durable job whose candidate was already exported."""
        raise ReefError(f"{type(self).__name__} does not support durable rejection recovery")

    @abstractmethod
    def prepare_training_step(
        self,
        batch: TrainingBatch,
        objective: str,
        algorithm_state: Mapping[str, Any],
        scheduling: StepScheduling,
        scenario_step: int,
        *,
        serving_runtime_load_id: str | None = None,
    ) -> PreparedTrainingStep:
        """Turn one reserved batch into backend work, or a state-only skip.

        ``objective`` names the recipe's training objective and ``scheduling``
        is the recipe's step schedule; the backend resolves the objective in
        its own process and cuts the batch into optimizer steps accordingly.
        """

    def execute_training_job(self, payload: Mapping[str, Any]) -> TrainingJobResult:
        """Execute a backend-native durable job through checkpoint export."""
        raise ReefError(f"{type(self).__name__} does not support native training jobs")

    @abstractmethod
    def train_candidate(self, payload: Mapping[str, Any]) -> ModelCandidate:
        """Train through durable checkpoint export without changing serving."""

    @abstractmethod
    def reject_candidate(self, candidate: ModelCandidate, decision: SelectionDecision) -> None:
        """Finish a rejected training candidate."""

    def restore_checkpoint(self, artifact: Artifact) -> None:
        """Restore training weights and optimizer state, without touching inference."""
        raise ReefError(f"{type(self).__name__} does not support training checkpoint restore")

    def commit_candidate(self, training_job_id: str) -> None:
        """Learn that Reef committed the candidate ``training_job_id`` trained here.

        Called after the durable commit, including once at startup for the
        head's job. The default does nothing: a trainer whose weights advance
        in place already holds them. A trainer that branches every candidate
        from an immutable incumbent makes that candidate the incumbent here.
        """
        return

    def shutdown(self) -> None:
        """Release only owned training resources."""
        return


# -- Native backend contracts -------------------------------------------------


class PreparedTrainingJob(ABC):
    """A prepared job whose reservation stays held through checkpoint recording.

    ``train`` may change model/optimizer state and returns all training metrics.
    ``save_checkpoint`` synchronously persists every required model/optimizer
    checkpoint and backend recovery metadata. Neither method may publish serving
    weights or advance Reef's job marker.
    """

    @property
    @abstractmethod
    def checkpoint(self) -> TrainingCheckpoint: ...

    @abstractmethod
    def train(self) -> TrainingMetrics: ...

    @abstractmethod
    def save_checkpoint(self) -> None: ...


@dataclass(frozen=True)
class TrainingCoordinationConfig:
    """Deployment policy interpreted only by Reef's coordinator."""

    save_hf_template: str | None
    colocate: bool = False
    lora: bool = False
    adapter_capacity: int | None = None
    keep_lora_base_resident: bool = False


def _initial_runtime_load_id() -> str:
    return str(RuntimeLoadId(new_runtime_load_id_incarnation(), 0))


@dataclass
class TrainingContext:
    """Scheduling state available to backend preparation without control handles."""

    next_rollout_id: int = 0
    runtime_load_id: str = field(default_factory=_initial_runtime_load_id)
    history: ScenarioHistoryStore | None = None


class TrainingBackend(ABC):
    """Training-only preparation, checkpoint I/O and native weight sending.

    Native training backends implement this interface for Reef's coordinator.
    The recipe-facing candidate lifecycle is ``reef.train.backend.CandidateBackend``.
    Sender methods must never pause, resume, offload or restart inference.
    Reef supplies the exact identity for each transfer. The sender must echo
    that identity; Reef independently verifies every receiver before commit.
    """

    @property
    @abstractmethod
    def config(self) -> TrainingCoordinationConfig:
        """Return this backend's deployment policy."""

    @property
    @abstractmethod
    def context(self) -> TrainingContext:
        """Return mutable scheduling state shared with Reef's coordinator."""

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def check_health(self) -> None: ...

    @abstractmethod
    def prepare_training_step(
        self,
        batch: TrainingBatch,
        objective: str,
        algorithm_state: Mapping[str, Any],
        scheduling: StepScheduling,
    ) -> PreparedTrainingStep: ...

    @abstractmethod
    def prepare(
        self,
        payload: Mapping[str, Any],
        *,
        job_id: str,
        rollout_id: int,
        prior_marker: Mapping[str, Any] | None,
    ) -> AbstractContextManager[PreparedTrainingJob | TrainingJobResult]:
        """Admit and prepare one job without changing model or optimizer state.

        Validation, scoring, data packing and storage admission finish before
        the prepared job is yielded. The context retains resource reservations
        until Reef records the checkpoint (including on failure); it must not
        suppress exceptions. An early result may only be stale or storage-blocked.
        """

    @abstractmethod
    def prepare_weights(self, runtime_load_id: str, *, force_full: bool) -> None:
        """Prepare a sender while colocated inference resources are released."""

    @abstractmethod
    def send_weights(self, runtime_load_id: str, *, force_full: bool) -> str: ...

    @abstractmethod
    def initialize_version(self, runtime_load_id: str) -> None: ...

    @abstractmethod
    def activate_scenario(self, scenario: str) -> None: ...

    @abstractmethod
    def send_adapter(self, scenario: str, name: str) -> None: ...

    def adapter_files(self, scenario: str, runtime_load_id: str) -> Path | None:
        """The adapter directory to serve as ``runtime_load_id`` when this trainer delivers files.

        A trainer that produces adapters as PEFT directories instead of
        sending tensors returns the directory; Reef then asks the receiver to
        load it and never calls the sender methods. The default, None,
        selects the native sender path.
        """
        return None

    @abstractmethod
    def close(self) -> None: ...


class InferenceBackend(ABC):
    """Receiver operations with acknowledged completion and no commit policy."""

    def load_adapter_files(self, name: str, path: Path, runtime_load_id: str | None) -> None:
        """Load a PEFT adapter directory under ``name``, serving it as ``runtime_load_id`` when given.

        Reef calls this for trainers that deliver adapter files; a receiver
        that can only accept native tensor transfers keeps the default.
        """
        raise ReefError(f"{type(self).__name__} does not load adapter files")

    @abstractmethod
    def initialize_version(self, runtime_load_id: str) -> None: ...

    @abstractmethod
    def inference_url(self) -> str: ...

    @abstractmethod
    def runtime_load_ids(self) -> Sequence[str]: ...

    @abstractmethod
    def pause(self) -> None: ...

    @abstractmethod
    def resume(self) -> None: ...

    @abstractmethod
    def recover(self) -> None: ...

    @abstractmethod
    def abort(self) -> None: ...

    @abstractmethod
    def offload(self, tags: tuple[str, ...] | None) -> None: ...

    @abstractmethod
    def onload_weights(self) -> None: ...

    @abstractmethod
    def onload_kv(self) -> None: ...

    @abstractmethod
    def unload_adapter(self, name: str) -> None: ...


class AdapterEngine(ABC):
    """Engine-side adapter operations the residency manager drives.

    ``load_adapter`` must return only once the engine can serve requests that
    name ``name``; raising means the adapter never became resident.
    ``payload`` is the caller's opaque description of the bytes to load.
    ``unload_adapter`` raising means the slot may still be occupied.
    """

    @abstractmethod
    def load_adapter(self, name: str, payload: Any) -> None: ...

    @abstractmethod
    def unload_adapter(self, name: str) -> None: ...


class InferenceMemoryOperations(ABC):
    """Synchronous engine operations; return only after every region is changed."""

    @abstractmethod
    def release(self, regions: Sequence[str]) -> None: ...

    @abstractmethod
    def resume(self, regions: Sequence[str]) -> None: ...


# -- Durable stores -----------------------------------------------------------


class ScenarioHistoryStore(ABC):
    """Committed scenario versions and checkpoints used by scheduling and recovery."""

    @property
    @abstractmethod
    def path(self) -> Path: ...

    @property
    @abstractmethod
    def scenarios(self) -> tuple[str, ...]: ...

    @abstractmethod
    def entry(self, scenario: str) -> Mapping[str, Any] | None: ...

    @abstractmethod
    def adapter(self, scenario: str) -> str | None: ...

    @abstractmethod
    def last_publication(self, scenario: str) -> str | None: ...

    @abstractmethod
    def lag(self, scenario: str, producing: RuntimeLoadId) -> int | None: ...

    @abstractmethod
    def protected_rollouts(self) -> set[int]: ...

    @abstractmethod
    def record_checkpoint(self, scenario: str, rollout_id: int) -> None: ...

    @abstractmethod
    def record_publication(self, scenario: str, runtime_load_id: str, adapter: str) -> None: ...

    @abstractmethod
    def status(self) -> dict[str, dict[str, Any]]: ...


MarkerStatus = Literal[
    "RUNNING",
    "CHECKPOINT",
    "UPDATING_WEIGHTS",
    "READY_TO_COMMIT",
    "HEAD_COMMITTED",
    "COMPLETE",
    "REJECTING",
    "REJECTED",
]


class TrainingJobStore(ABC):
    """Durable training-job transitions shared by execution and publication."""

    @property
    @abstractmethod
    def path(self) -> Path: ...

    @abstractmethod
    def read(self) -> dict[str, Any] | None: ...

    @abstractmethod
    def write(self, marker: Mapping[str, Any]) -> None: ...

    @abstractmethod
    def transition(self, marker: dict[str, Any], status: MarkerStatus, **updates: Any) -> dict[str, Any]: ...
