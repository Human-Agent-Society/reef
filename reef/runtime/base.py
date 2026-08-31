"""Execution capabilities shared by inference and training runtimes."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from threading import Condition
from typing import Any, Literal

from reef.artifact.artifact import Artifact
from reef.core.errors import ReefError
from reef.runtime.candidates import ActivatedModel, ModelCandidate
from reef.runtime.inference import InferenceBackend
from reef.train.evaluation.contracts import SelectionDecision
from reef.train.types import TrainingBatch


class RuntimeContractError(ReefError):
    """A runtime or backend violated its contract with Reef.

    Raised for malformed runtime results and missing capabilities that a
    correctly configured deployment would never produce — distinct from user
    input errors, which surface as more specific ``ReefError`` subclasses.
    """


@dataclass(frozen=True, slots=True)
class TrainingJobResult:
    """Outcome of one restart-safe training job.

    ``metrics`` is backend telemetry carried opaquely, the same contract as
    ``TrainStepResult.metrics``: the backend that produced it owns the schema,
    reef never interprets it, and it reaches the commit record so per-step
    provenance survives to serving.
    """

    outcome: Literal["complete", "checkpoint", "stale", "storage_blocked"]
    runtime_load_id: str
    checkpoint_path: str | None = None
    storage: Mapping[str, Any] | None = None
    metrics: Mapping[str, Any] | None = None
    training_job_id: str | None = None

    def __post_init__(self) -> None:
        # Fail closed at the boundary: a completed job that cannot name the
        # checkpoint it exported would otherwise reach the commit protocol and
        # be published as a durable version pointing at nothing.
        if self.outcome in {"complete", "checkpoint"} and not self.checkpoint_path:
            raise ValueError(f"a {self.outcome} training job must report the checkpoint path it exported")
        if not self.runtime_load_id:
            raise ValueError("a training job result must report a runtime load ID")
        if self.training_job_id is not None and (
            not isinstance(self.training_job_id, str) or not self.training_job_id
        ):
            raise ValueError("training_job_id must be a non-empty string or None")


class InferenceAdmissionHandle:
    """A handle for one admitted inference, released after model execution."""

    def __init__(self, controller: InferenceAdmissionController) -> None:
        self._controller = controller
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._controller._release()


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
        loop = asyncio.get_running_loop()
        while True:
            with self._condition:
                if self._loop is None or self._loop.is_closed():
                    self._loop = loop
                    self._open_event = asyncio.Event()
                    if self._open:
                        self._open_event.set()
                elif self._loop is not loop:
                    raise RuntimeError("inference admission cannot span concurrent event loops")
                if self._open:
                    self._active += 1
                    return InferenceAdmissionHandle(self)
                if self._open_event is None:
                    raise RuntimeError("closed inference admission has no loop event")
                event = self._open_event
            await event.wait()
            # ``close`` clears the asyncio event on its owning loop. If this
            # task raced the thread-safe callback, a still-set event would
            # otherwise make this loop spin without yielding and prevent the
            # clear callback from ever running.
            await asyncio.sleep(0)

    def close(self, *, wait: bool = False, timeout: float | None = None) -> None:
        """Reject new admissions and optionally drain already admitted work."""
        with self._condition:
            self._open = False
            loop, event = self._loop, self._open_event
            if loop is not None and event is not None and not loop.is_closed():
                # The loop can close between is_closed() and scheduling.
                with suppress(RuntimeError):
                    loop.call_soon_threadsafe(event.clear)
            if wait and not self._condition.wait_for(lambda: self._active == 0, timeout=timeout):
                raise TimeoutError("timed out waiting for admitted inference requests to drain")

    def open(self) -> None:
        """Admit queued and future requests."""
        with self._condition:
            self._open = True
            loop, event = self._loop, self._open_event
            self._condition.notify_all()
        if loop is not None and event is not None and not loop.is_closed():
            with suppress(RuntimeError):
                loop.call_soon_threadsafe(event.set)

    @property
    def status(self) -> Mapping[str, Any]:
        with self._condition:
            return {"open": self._open, "active": self._active}

    def _release(self) -> None:
        with self._condition:
            if self._active <= 0:
                raise RuntimeError("inference admission handle released without an active request")
            self._active -= 1
            if self._active == 0:
                self._condition.notify_all()


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


class InferenceRuntime(ABC):
    """A runtime owns at least an inference backend.

    The runtime itself is NOT an InferenceBackend: it composes one. This
    separates the 'lifecycle owner' role from the 'request executor' role
    and lets a runtime swap backends without subclassing.
    """

    def __init__(
        self,
        *,
        base_url: str,
        inference_timeout_s: float = 300.0,
    ) -> None:
        if not base_url:
            raise ValueError("base_url must be non-empty")
        if inference_timeout_s <= 0:
            raise ValueError("inference_timeout_s must be positive")
        self._base_url = base_url.rstrip("/")
        self._inference_timeout_s = inference_timeout_s
        self._inference_admission = InferenceAdmissionController()

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def inference_timeout_s(self) -> float:
        return self._inference_timeout_s

    async def acquire_inference(self) -> InferenceAdmissionHandle:
        """Wait until this runtime may freeze and execute a new inference."""
        return await self._inference_admission.acquire()

    @property
    def inference_admission_status(self) -> Mapping[str, Any]:
        return self._inference_admission.status

    @property
    @abstractmethod
    def inference_backend(self) -> InferenceBackend:
        """The inference backend owned by this runtime."""


class TrainingRuntime(InferenceRuntime, ABC):
    """A runtime that produces selectable model candidates."""

    @property
    def max_staleness(self) -> int:
        """Largest producing-to-serving version lag this runtime admits.

        Exact-version admission is the default. Runtimes that support a
        positive bounded-staleness window override this property.
        """
        return 0

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

    def serving_adapter_name(self) -> str | None:
        """Name of the one adapter the serving engine applies, if it serves one.

        A LoRA deployment trains an adapter over frozen base weights, and the
        engine applies it only when a request names it. Reporting the name
        here is what lets the weight surface address every request to it, so
        no harness can silently sample the frozen base. ``None`` means the
        runtime publishes full weights and requests need no adapter name, or
        that adapters are per scenario (see
        :attr:`concurrent_training_scenarios`).
        """
        return None

    @property
    def concurrent_training_scenarios(self) -> bool:
        """Whether several scenarios may train on this runtime at once.

        A per-scenario LoRA runtime keeps one frozen base and time-slices its
        adapter slot between scenarios, publishing each under a
        scenario-qualified name; the dispatcher then drives every training
        scenario instead of binding the process to one.
        """
        return False

    def serving_adapter_runtime_load_id(self, scenario: str) -> str | None:
        """The serving runtime load ID of ``scenario``'s resident adapter.

        ``None`` when the runtime does not serve per-scenario adapters or the
        scenario has published nothing yet (requests then sample the base).
        """
        return None

    def current_runtime_load_id(self) -> str | None:
        """Return the version Reef has made available to new inference.

        The default is suitable for runtimes whose serving update and Reef
        publication are one operation. Runtimes with a deferred commit
        handshake retain the previous value until that handshake completes.
        """
        return self.serving_runtime_load_id() if self.inference_admission_status.get("open") is True else None

    def restore_checkpoint(self, artifact: Artifact) -> str:
        """Restore training and serving weights from a durable artifact.

        Runtimes that support weight rollback override this and return the new
        serving-engine version token. The default fails explicitly: silently
        moving Reef's artifact head while the engine keeps newer weights would
        corrupt serving-version records.
        """
        raise ReefError(f"{type(self).__name__} does not support checkpoint restore")

    def reconcile_training_job(
        self,
        scenario_step: int,
        *,
        committed_training_job_id: str | None = None,
        committed_training_without_job_id: bool = False,
        scenario: str | None = None,
    ) -> None:
        """Reconcile a backend training job against Reef's durable commit.

        ``scenario`` is passed only by a backend bound to a runtime that
        trains several scenarios at once (see
        :attr:`concurrent_training_scenarios`).
        """

    @abstractmethod
    def prepare_training_step(
        self,
        batch: TrainingBatch,
        step_preparer: str,
        algorithm_state: Mapping[str, Any],
        scenario_step: int,
    ) -> PreparedTrainingStep: ...

    @abstractmethod
    def train_candidate(self, payload: Mapping[str, Any]) -> ModelCandidate:
        """Train through durable checkpoint export without changing serving."""
        ...

    @abstractmethod
    def activate_candidate(self, candidate: ModelCandidate) -> ActivatedModel:
        """Apply a selected candidate to serving."""
        ...

    @abstractmethod
    def reject_candidate(self, candidate: ModelCandidate, decision: SelectionDecision) -> None:
        """Finish a rejected candidate without changing serving."""
        ...
