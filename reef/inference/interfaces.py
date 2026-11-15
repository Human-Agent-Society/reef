"""Inference contracts and default request admission.

Who implements what:

- Recipes and the dispatcher *use* ``InferenceRuntime``.
- An inference integration *implements* ``InferenceHandler`` for requests,
  ``InferenceBackend`` for loading weight files into its engines,
  ``InferenceEngine`` for each engine, and ``AdapterEngine`` for the adapter
  operations of ``reef.inference.residency``. The engine supervision hooks live
  in ``recovery`` next to the objects that drive them.

The module reads top-down in that order: identities and errors, request
admission, request handling, runtime contracts, native backend contracts.
"""

from __future__ import annotations

import base64
import json
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Literal

from reef.artifact.artifact import Artifact
from reef.core.errors import InferenceUnavailableError, InternalError, ReefError
from reef.core.evaluation import ArtifactKind, PolicyModel
from reef.core.plugins import Plugin
from reef.core.served_models import ServedArtifact
from reef.core.values import JsonValue
from reef.core.weight_layout import WeightFiles

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


class RuntimeLoadMismatchError(RuntimeContractError):
    """The binding selected for a request does not hold.

    The engine refused the selected model name before it generated (HTTP 421). The inference replica retries such a
    request once while the client has received no response.
    """


class InferenceOwnerLostError(RuntimeContractError, InternalError, RuntimeError):
    """The receiver has accepted a later serving owner and refuses this owner's control operations.

    A remote component's control client raises it for HTTP 412. The serving worker then gives up its ownership
    and reports a lost work lease, so the selection stays for the later owner.
    """


#: The prefix of every name that ``full_weight_name`` makes; a name without it is an ordinary model name.
FULL_WEIGHT_NAME_PREFIX = "reef-weights-"


def full_weight_name(model_path: str, runtime_load_id: str) -> str:
    """The route model name of one full-weight load: the model at ``model_path`` under ``runtime_load_id``.

    The name is ``FULL_WEIGHT_NAME_PREFIX``, the unpadded base64url form of ``model_path``, a ``.`` and
    ``runtime_load_id``, so ``parse_full_weight_name`` returns both values. Raise ``ValueError`` for an empty
    ``model_path`` or a ``runtime_load_id`` that ``RuntimeLoadId.parse`` refuses.
    """
    if not model_path:
        raise ValueError("a full-weight route needs a model path")
    RuntimeLoadId.parse(runtime_load_id)
    encoded = base64.urlsafe_b64encode(model_path.encode()).decode().rstrip("=")
    return f"{FULL_WEIGHT_NAME_PREFIX}{encoded}.{runtime_load_id}"


def parse_full_weight_name(name: str) -> tuple[str, str] | None:
    """The ``(model_path, runtime_load_id)`` of a name that ``full_weight_name`` made.

    None for a name without ``FULL_WEIGHT_NAME_PREFIX``, which is an ordinary model name. Raise ``ValueError`` for
    any other name that ``full_weight_name`` does not make.
    """
    if not name.startswith(FULL_WEIGHT_NAME_PREFIX):
        return None
    encoded, separator, version = name[len(FULL_WEIGHT_NAME_PREFIX) :].partition(".")
    if not separator or not encoded:
        raise ValueError("invalid full-weight route name")
    model_path = base64.b64decode(encoded + "=" * (-len(encoded) % 4), altchars=b"-_", validate=True).decode()
    if full_weight_name(model_path, version) != name:
        raise ValueError("invalid full-weight route name")
    return model_path, version


# -- Request admission --------------------------------------------------------


class InferenceAdmissionHandle:
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
    """Thread-safe inference admission: it counts the admitted requests, and once closed it admits none.

    Values: ``open`` is True until ``close``, and never True again; ``active``
    is the non-negative count of admitted requests not yet released.
    ``status`` reports both. The runtime's ``acquire_inference`` makes the
    handles, and the runtime closes admission when it shuts down.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._open = True
        self._active = 0

    async def acquire(self) -> InferenceAdmissionHandle:
        """Count one active request; ``InferenceUnavailableError`` once admission is closed."""
        with self._lock:
            if not self._open:
                raise InferenceUnavailableError("the inference runtime is shutting down")
            self._active += 1
            return InferenceAdmissionHandle(self)

    def close(self) -> None:
        """Admit no new request from now on; admitted requests continue."""
        with self._lock:
            self._open = False

    @property
    def status(self) -> Mapping[str, Any]:
        with self._lock:
            return {"open": self._open, "active": self._active}

    def release(self) -> None:
        """Count one admitted request as finished."""
        with self._lock:
            if self._active <= 0:
                raise RuntimeError("inference admission handle released without an active request")
            self._active -= 1


# -- Request handling ---------------------------------------------------------


class InferenceStream:
    """One open provider response whose bytes can be forwarded incrementally.

    ``chunks`` must end only after capture finalization and transport validation. When
    ``record_response_pending`` is true, a successful stream supplies ``record_response`` before that end.
    A terminal frame alone is not successful completion. ``close`` releases transport resources.
    """

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
        # event. The inference replica exhausts the stream and checks that this
        # capture arrived before accepting the record as complete.
        self.record_response_pending = record_response_pending

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._close is not None:
            await self._close()


@dataclass(frozen=True)
class InferenceArtifact:
    """The selected component identity and request-time requirements for inference.

    ``artifact_id`` is a nonempty component ID. ``local_path`` is None or a
    Path supplied by local evaluation. ``served_model`` is None or the
    nonempty model name of the route that ``served_models`` selected for the
    request; a handler sends it to an engine receiver that keeps each request
    on the weights that the name selects. This value contains no storage locator.
    Serving and evaluation construct it; handlers read its fields without I/O.
    """

    artifact_id: str
    local_path: Path | None = None
    served_model: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_id, str) or not self.artifact_id:
            raise ValueError("inference requires a nonempty artifact_id")
        if self.local_path is not None and not isinstance(self.local_path, Path):
            raise TypeError("inference local_path must be a Path or None")
        if self.served_model is not None and (not isinstance(self.served_model, str) or not self.served_model):
            raise ValueError("served_model must be a nonempty string or None")

    @classmethod
    def from_artifact(cls, artifact: Artifact) -> InferenceArtifact:
        """Copy the artifact's identity and its available path."""
        return cls(artifact.artifact_id, artifact.local_path)


#: The state of the engine behind a route (RFC #1269): ``warm`` answers now, ``starting`` runs but does not answer
#: yet, ``idle`` runs no engine, and a request starts one.
EngineState = Literal["warm", "starting", "idle"]

#: The phases of an artifact's startup, in their order (RFC #1283): ``prepare`` (the serving work waits, deploys the
#: engine, and converts or quantizes the weights), ``start`` (a machine and the weights) and ``load`` (GPU memory,
#: kernels, CUDA graphs and warm-up). ``reef.inference.startup`` reports them.
StartupPhase = Literal["prepare", "start", "load"]
#: The phases, in the order of a startup.
PHASES: tuple[StartupPhase, ...] = ("prepare", "start", "load")


@dataclass(frozen=True)
class StartupStep:
    """One phase of an artifact's startup.

    Values: ``phase`` is one of ``PHASES``. ``started_at`` is when it began, and ``ended_at`` when it ended, in epoch
    seconds, at least ``started_at``; None while it runs. ``detail`` is a short note of what it does, such as
    "Quantizing to NVFP4", or None.

    Purpose: ``reported_steps`` and ``InferenceHandler.engine_steps`` make it, and the startup status gives it as JSON.
    """

    phase: StartupPhase
    started_at: float
    ended_at: float | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        if self.phase not in PHASES:
            raise ValueError(f"a startup phase is one of {PHASES}, not {self.phase!r}")
        if self.ended_at is not None and self.ended_at < self.started_at:
            raise ValueError("a startup step ends at or after its start")
        if self.detail is not None and not isinstance(self.detail, str):
            raise ValueError("a startup step's detail is text or None")

    def to_json(self) -> dict[str, JsonValue]:
        return {"phase": self.phase, "started_at": self.started_at, "ended_at": self.ended_at, "detail": self.detail}


@dataclass(frozen=True)
class StartupSteps:
    """The phases of an artifact's startup so far, and how long it may still take.

    Values: ``steps`` have distinct phases in the order of ``PHASES``, and start in that order; every step but the
    last has ``ended_at``. ``estimate_seconds`` is a guess of the seconds until the engine is warm, at least 0, or None.

    Purpose: the startup status gives it as its ``startup`` field (``to_json``).
    """

    steps: tuple[StartupStep, ...]
    estimate_seconds: float | None = None

    def __post_init__(self) -> None:
        order = [PHASES.index(step.phase) for step in self.steps]
        if order != sorted(set(order)):
            raise ValueError("startup steps have distinct phases, in the order of a startup")
        starts = [step.started_at for step in self.steps]
        if starts != sorted(starts):
            raise ValueError("startup steps start in the order of their phases")
        if any(step.ended_at is None for step in self.steps[:-1]):
            raise ValueError("only the last startup step can run")
        if self.estimate_seconds is not None and self.estimate_seconds < 0:
            raise ValueError("a startup's estimate is at least 0 seconds")

    def to_json(self) -> dict[str, JsonValue]:
        return {"steps": [step.to_json() for step in self.steps], "estimate_seconds": self.estimate_seconds}


class InferenceHandler(Plugin, ABC):
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

    @abstractmethod
    async def inference(self, artifact: InferenceArtifact, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Return the provider response for one native request payload."""

    def address_model(self, payload: Mapping[str, Any], model_name: str) -> dict[str, Any]:
        """``payload`` addressed to ``model_name``, the name under which the engine serves an artifact
        (``ServedArtifact.model_name``). The default names it as the request's ``model``."""
        return {**payload, "model": model_name}

    def serving_config(self, override: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        """The request fields that a scenario may set (RFC #1097), with their effective values under ``override``.

        ``override`` maps field names to the values that one scenario sets. The result maps each field that has a
        value to its value: the override's value when it sets the field, else the handler's own value. A field with
        neither value has no key. ``ValueError`` when ``override`` sets a field that the handler does not have, or a
        value that the field does not take. The default handler has no field: it gives ``{}`` for an empty
        ``override`` and refuses any other.
        """
        if override:
            raise ValueError(f"{type(self).__name__} has no serving fields; got {sorted(override)}")
        return {}

    def request_payload(
        self, payload: Mapping[str, Any], *, serving_config: Mapping[str, JsonValue]
    ) -> dict[str, Any]:
        """The provider request body that the engine receives for the client's body ``payload``, before
        ``address_model``: a new dict, with the values that the handler sets itself, such as fixed sampling values.

        ``serving_config`` is the override of the scenario that the request is for, which its runtime's
        ``serving_config`` accepted. The handler applies the fields of it that are its own and leaves the rest,
        which the runtime applies to the engine. The inference service sends the body and records it.
        ``ValueError`` for a body that the handler refuses. The default is ``payload`` unchanged."""
        return dict(payload)

    def engine_state(self, served: ServedArtifact) -> EngineState | None:
        """The state of the engine that answers at ``served`` (``EngineState``), or None when the handler cannot tell.

        It must not start an engine. It may block: the inference service calls it in a thread, for the status of a
        ready artifact. The default is None.
        """
        return None

    def engine_detail(self, served: ServedArtifact) -> str | None:
        """What the starting engine at ``served`` does now, as one short line for a person, such as "Loading weights
        · 0:42"; None when the handler cannot tell, the default. Reef asks it only while ``engine_state`` is
        ``starting``, under the same rules."""
        return None

    def engine_steps(self, served: ServedArtifact) -> StartupSteps | None:
        """The ``start`` and ``load`` phases of the starting engine at ``served`` (RFC #1283), when a ready artifact's
        engine starts again from zero; None when the handler cannot tell, the default. Reef asks it only while
        ``engine_state`` is ``starting``, under the same rules."""
        return None

    async def inference_stream(
        self, artifact: InferenceArtifact, path: str, payload: dict[str, Any]
    ) -> InferenceStream:
        """Return a stream for one native request payload.

        Custom handlers that only implement buffered inference keep working: the
        default implementation exposes their JSON response as one chunk. HTTP
        handlers override this method to preserve provider-native streaming.
        """
        value = await self.inference(artifact, path, payload)

        async def chunks() -> AsyncIterator[bytes]:
            yield json.dumps(value, ensure_ascii=False).encode()

        return InferenceStream(status=200, headers={"Content-Type": "application/json"}, chunks=chunks())


class EngineClients(ABC):
    """The handlers that reach the engines that ``served_models`` names.

    Purpose: an inference replica asks for the handler of each ``ServedArtifact`` it serves; the deployment's
    engine configuration, or the engines that an inference process runs, decide which one it is.
    """

    @abstractmethod
    def handler(self, served: ServedArtifact) -> InferenceHandler:
        """The handler that calls the engine at ``served.url``; ``LookupError`` when no engine answers there."""


class FixedEngineClients(EngineClients):
    """One handler for every ``ServedArtifact``: an embedding application's own handler.

    Values: ``fixed`` is that handler. Purpose: ``build_app`` gives it to the inference routes when the caller
    passes an ``inference_handler``.
    """

    def __init__(self, fixed: InferenceHandler) -> None:
        self.fixed = fixed

    def handler(self, served: ServedArtifact) -> InferenceHandler:
        return self.fixed


# -- Runtime contracts --------------------------------------------------------


class InferenceRuntime(ABC):
    """Own inference requests, serving weights and admission.

    An InferenceHandler executes individual requests. This runtime owns that
    handler and its endpoint, weight activation and serving version state;
    training and optimizer state belong to a training component. A concrete
    runtime gives ``inference_handler``, ``activate_checkpoint`` and
    ``served_artifact``. A runtime that serves snapshots without weights also
    overrides ``endpoint_model_name``, whose default refuses them.
    """

    def __init__(self, *, base_url: str, inference_timeout_s: float = 300.0) -> None:
        if not base_url:
            raise ValueError("base_url must be non-empty")
        if inference_timeout_s <= 0:
            raise ValueError("inference_timeout_s must be positive")
        self._base_url = base_url.rstrip("/")
        self._inference_timeout_s = inference_timeout_s
        self._inference_admission = InferenceAdmissionController()

    # -- Requests

    @property
    def serves_requests(self) -> bool:
        """Whether this runtime supports model requests. False keeps offline training deployments out of startup."""
        return True

    @property
    def serving_slots(self) -> int:
        """How many serving works the serving worker runs at once (RFC #1299): a positive int, 1 by default. With
        more than one, the runtime gets concurrent calls for different scenarios, and for the artifacts of one
        scenario, so it must lock what they share, such as an engine that scenarios share."""
        return 1

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def model_path(self) -> str:
        """The served model name, or empty when requests must select a model."""
        return ""

    @property
    def inference_timeout_s(self) -> float:
        return self._inference_timeout_s

    @property
    @abstractmethod
    def inference_handler(self) -> InferenceHandler:
        """The inference request handler owned by this runtime."""

    def handler_for(self, url: str) -> InferenceHandler | None:
        """The handler of the engine at ``url`` that this runtime serves, for a process that did not open the
        runtime, or None when the runtime serves nothing at ``url``.

        A separate inference service calls it for the URL of each ``served_models`` row, and reaches a URL with no
        handler as a plain proxy. The default gives ``inference_handler`` for ``base_url`` and None for any other
        URL. A runtime that serves artifacts at several URLs, or whose engines need credentials or retries that a
        plain proxy does not send, gives a handler for each of them.
        """
        return self.inference_handler if url == self.base_url else None

    @abstractmethod
    def activate_checkpoint(self, artifact: Artifact, *, scenario: str | None = None) -> str:
        """Serve the published snapshot ``artifact``; return the runtime load ID that serves it.

        Reef calls it for the weight component of a snapshot, at a commit that
        activates the snapshot, at an activation and at a start, before traffic
        routes to the snapshot; a snapshot without one gets no activation
        (``endpoint_model_name``). ``scenario`` names
        the snapshot's scenario exactly when the runtime serves a separate
        adapter per scenario (``per_scenario_adapters``). A raise reports a
        failed activation without promising rollback: callers cannot assume that
        the previous artifact is still loaded. Implementations document their
        recovery behavior; a cached activation alone cannot establish a binding
        after failure. A runtime that loads weights makes a new runtime load ID
        for each load. A runtime that serves requests but can neither load nor
        select the snapshot's weights, such as a generic proxy, raises, so that
        no route claims them; a runtime that serves no requests returns the
        artifact ID.
        """

    def served_model_name(self, scenario: str) -> str:
        """The model name that the runtime gives ``scenario``'s route now; ``model_path`` unless a runtime serves each
        scenario under its own name.

        A runtime that binds weights uses this name in ``served_artifact``. The built-in proxy returns this name
        through ``endpoint_model_name`` without claiming a weight binding.
        """
        return self.model_path

    @abstractmethod
    def served_artifact(self, artifact_id: str, scenario: str) -> ServedArtifact | None:
        """The route to ``artifact_id``'s weights as ``scenario`` serves them now, for a caller in another process.

        The result is ``ServedArtifact(artifact_id, url, model_name)``: at this call, the HTTP(S) ``url`` answers under
        ``model_name`` from a weight binding of this artifact that is already established. The
        runtime establishes it in ``activate_checkpoint``, with weights that the engines loaded and verified or with an
        immutable sampler or provider revision that the activation selected; or the runtime knows an immutable
        provider revision of the artifact. This call establishes nothing and loads no weights. A runtime with one
        binding for every scenario, such as a full-weight runtime, answers from that binding whatever ``scenario`` is;
        a runtime that binds each scenario separately (``per_scenario_adapters``) answers from ``scenario``'s binding.

        None when the runtime holds no such binding: the artifact is unknown, the binding is
        unavailable or uncertain (for example after a failed load or a restart), or the endpoint has no HTTP(S) address
        or model name. The inference worker asks only for the snapshot of a scenario with a weight component, publishes
        the result as the scenario's route, and fails the startup on None; a snapshot without one routes through
        ``endpoint_model_name``. Before an activation, the worker calls it to tell whether its earlier activation
        of the same artifact still serves.

        The result describes the binding at this call and reserves it for no request. A request that the route
        answers must use only this artifact's weights until its generation completes, and a request that selects a
        binding which is stale or unavailable must fail instead of running on other weights. The runtime's serving
        path keeps that guarantee in one of two ways. An immutable revision behind the route, such as a provider
        or sampler revision, never changes. A runtime that replaces weights in running engines needs a receiver
        that admits each request only under the name of weights that it serves now, and that changes no weights
        before every admitted request ended or its engine stopped. The built-in saved-weight runtimes refuse
        engines without such a receiver (``ServingEngines.request_pinning``).
        """

    def endpoint_model_name(self, scenario: str) -> str | None:
        """The non-empty model name at ``base_url`` for ``scenario``'s snapshot without weights, or None.

        A snapshot without a weight component (``ScenarioComponents.loaded_component`` is None), such as a record-only
        or harness snapshot, gets no activation. The inference worker routes it to this model under the snapshot's own
        artifact ID, and each artifact that the scenario serves beside it under that artifact's ID. Such a route
        associates the deployment's configured model with the artifact; it identifies no weights, and Reef does not
        check what the endpoint serves under the name. The default, None, refuses such snapshots, so their startup
        fails. The worker never calls it for a scenario with a weight component.
        """
        return None

    def serving_config(self, override: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        """The serving fields that a scenario may set (RFC #1097), with their effective values under ``override``.

        As ``InferenceHandler.serving_config``, over the fields of the engine and of the handler: ``override`` maps
        field names to one scenario's values, and the result maps each field that has a value to the override's
        value, else the deployment's. ``ValueError`` names a field that neither has, or a value that its field does
        not take. Control calls it without opening the runtime. The default has the handler's fields only.
        """
        return self.inference_handler.serving_config(override)

    def apply_serving_config(self, scenario: str, override: Mapping[str, JsonValue]) -> None:
        """Make the engine fields of ``override``, which ``serving_config`` accepted, hold for ``scenario``: for the
        engine that serves it now, and for each engine that starts for it later. A field that ``override`` does not
        set takes the deployment's value again. The inference worker calls it before each activation and each
        ``serve_artifact``, and when only the override changed, so a second call with the same values changes
        nothing. After it, the worker reads the scenario's route again (``served_artifact``), so a runtime may move
        the scenario to another engine. The default has no engine field and does nothing; the handler applies
        request fields to each request."""
        return

    def open(self) -> None:
        """Open the engines before the inference worker adopts or activates artifacts. A second call does nothing.

        Other service processes build the same runtime without opening it. The default has no engine to open.
        """
        return

    def set_owner_sequence(self, sequence: int) -> None:
        """Set the durable serving-owner take sequence before opening; runtimes without remote control ignore it."""
        return

    def adopt(self, served: ServedArtifact) -> bool:
        """Take over ``served``, which a ``served_models`` row names for this runtime's engine, from before this
        runtime existed (an inference worker restart), and return whether the engine still serves it. Then the route
        stays while the new owner serves the scenario again, and the next activation of the same artifact loads
        nothing. The default can tell nothing about the engine: it returns False, so the route goes until the
        scenario serves again, and the next activation loads."""
        return False

    def stop_serving(self, served: ServedArtifact) -> None:
        """Stop serving ``served``, which no ``served_models`` row names any more: the engine frees what it holds
        for it. A value the runtime does not know is no error. The default holds nothing per value: a runtime that
        loads full weights in place replaced them at the next activation."""
        return

    def release_unreferenced_artifacts(self, retained: Collection[ServedArtifact]) -> None:
        """Stop serving each artifact that this runtime holds and that no value of ``retained`` uses. The default
        holds nothing per value, so it releases nothing.

        ``retained`` holds every value that a ``served_models`` row names and the value of every scheduled stop,
        also of a stop that is not due yet. A value of ``retained`` uses what the runtime holds at its URL under
        its model name, whatever its artifact ID is (as ``ServedModels.is_referenced``). So the call releases only
        what a serving work loaded and did not publish, because the work failed or stopped, and what waits for its
        scheduled stop stays. Only the inference worker that owns the runtime calls it, between two serving works,
        while none runs. A release that fails raises, and the next call tries it again. The call adds no way to
        serve an artifact that is not deployed (``serves_any_artifact``).
        """
        return

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

    def close_admission(self) -> None:
        """Admit no new request, before shutdown; admitted requests continue."""
        self._inference_admission.close()

    # -- Scenarios

    @property
    def served_weight_kind(self) -> ArtifactKind:
        """The kind of the weights that this runtime serves from a snapshot.

        ``model_adapter`` is a LoRA adapter on the runtime's base model, and ``model`` is a whole model's weights,
        which is the default.
        """
        return "model"

    @property
    def api(self) -> str:
        """The API dialect in which Reef's training side calls the model itself: ``openai`` (Chat Completions),
        ``responses`` or ``anthropic``. The default is ``openai``: an engine serves Chat Completions."""
        return "openai"

    @property
    def provider_key(self) -> str | None:
        """The key with which the runtime calls a provider; the default is None, for a runtime that calls no
        provider."""
        return None

    @property
    def serves_any_artifact(self) -> bool:
        """Whether this runtime answers each call from the artifact that the call names, without loading it.

        It is the one statement of that capability. False, the default, for a runtime that serves only the artifact
        it has loaded. A runtime that returns True overrides ``artifact_model`` and ``serve_artifact``;
        the defaults of both refuse.
        """
        return False

    @property
    def served_artifact_capacity(self) -> int | None:
        """How many weight artifacts of one scenario this runtime keeps bound at once: a positive int, or None for
        unlimited capacity.

        The count holds the scenario's active artifact and each artifact that serves beside it
        (``serve_artifact``), until ``stop_serving`` releases it. It bounds deployed artifacts only: whether the
        runtime answers from an artifact that is not deployed, as a candidate evaluation needs, is
        ``serves_any_artifact`` alone. The default is None for a runtime that answers from any artifact without a
        load, and 1 for a runtime that serves only the artifact it has loaded.
        """
        return None if self.serves_any_artifact else 1

    def artifact_model(self, artifact: Artifact) -> PolicyModel:
        """The model that an evaluation rollout names to sample ``artifact`` on request, without a load.

        Raise ``ReefError`` when ``serves_any_artifact`` is False, and ``ValueError`` when ``artifact`` is
        not one that this runtime can serve.
        """
        raise ReefError("this runtime serves only the artifact it has loaded, so it serves no artifact on request")

    def serve_artifact(self, artifact: Artifact, *, scenario: str) -> ServedArtifact | None:
        """Make an engine answer for ``artifact`` of ``scenario``, and return the value that reaches it: the engine's
        URL and the model name that samples that artifact. None when the engine has no address that another process
        can call. The inference worker applies ``scenario``'s serving config (``apply_serving_config``) first.

        Return only when the engine is ready. A second call for a served artifact starts nothing. Only the inference
        owner calls it, one call at a time.

        The call deploys the artifact: it serves beside the scenario's active artifact until ``stop_serving``, and
        it takes one place of ``served_artifact_capacity``. That differs from ``artifact_model``, which names an
        artifact without a load and needs ``serves_any_artifact``. So a runtime with a capacity above 1 implements
        this method and can leave ``serves_any_artifact`` False: it serves deployed artifacts beside the active
        one, and it still evaluates no candidate. The default refuses, for a runtime with a capacity of 1.

        Raise ``ReefError`` when the runtime cannot deploy the artifact beside what it serves, as the default does
        and as a runtime does whose places all hold other artifacts. Raise ``ValueError`` when ``artifact`` is not
        one that this runtime can serve.
        """
        raise ReefError("this runtime serves only the artifact it has loaded, so it serves no artifact on request")

    @property
    def per_scenario_adapters(self) -> bool:
        """Whether the runtime serves each scenario its own adapter, which requests name per scenario."""
        return False

    @property
    def shared_by_training_scenarios(self) -> bool:
        """Whether several training scenarios can share this runtime, each served its own weights.

        A runtime that loads one model into its engines serves one training
        scenario: a second scenario's snapshot would replace the first one's
        weights.
        """
        return False

    # -- Adapters

    def serving_adapter_runtime_load_id(self, scenario: str) -> str | None:
        """The serving runtime load ID of ``scenario``'s resident adapter.

        ``None`` when the runtime does not serve per-scenario adapters or the
        scenario has published nothing yet (requests then sample the base).
        """
        return None

    def adapter_residency_status(self) -> Mapping[str, Any] | None:
        return None


class InferenceBackend(ABC):
    """Receiver operations with acknowledged completion and no commit policy.

    The serving runtimes of ``reef.inference.serving`` call these operations on
    one engine group. The engines start paused, serving the base model, and
    serve requests after ``resume``. Each operation returns once every engine
    has done it, and raises when one has not.
    """

    def load_weight_files(self, files: WeightFiles, *, adapter_name: str | None, runtime_load_id: str | None) -> None:
        """Load ``files`` into every engine and return once they serve them.

        An ``"adapter"`` layout is loaded under ``adapter_name``, which is then
        given; a ``"full"`` layout replaces the served weights, and
        ``adapter_name`` is None. When ``runtime_load_id`` is given, every
        engine reports it afterwards. A receiver that loads no files keeps the
        default, which refuses.
        A repeated runtime load ID does not skip loading: recovery can restore the files after a partial mutation
        that left the old ID on the engine. Successful return acknowledges the load, not an inspection of weights.
        """
        raise ReefError(f"{type(self).__name__} does not load weight files")

    @abstractmethod
    def initialize_version(self, runtime_load_id: str) -> None:
        """Make every engine report ``runtime_load_id`` for the weights it serves now; generation stays paused."""

    @abstractmethod
    def inference_url(self) -> str:
        """The non-empty base URL that inference requests for these engines go to."""

    @abstractmethod
    def runtime_load_ids(self) -> Sequence[str]:
        """The runtime load ID that each live engine reports, one per engine."""

    @abstractmethod
    def pause(self) -> None:
        """Stop every engine from scheduling generation, so that a weight load can follow.

        Repeated pauses are idempotent; one successful resume releases them.
        """

    @abstractmethod
    def resume(self) -> None:
        """Let every engine schedule generation again."""

    @abstractmethod
    def recover(self) -> int:
        """Replace the engines that died, keep the live ones, and return the number of engine slots refilled.

        A paused group stays paused. A replacement engine serves the base
        model and holds no adapter.
        """

    @abstractmethod
    def abort(self) -> None:
        """Retire the owned engines, which ``recover`` then replaces, such as to restart them all; borrowed engines stay."""

    @abstractmethod
    def unload_adapter(self, name: str) -> None:
        """Remove the adapter ``name`` from every engine; raise when an engine may still hold it."""


class AdapterEngine(ABC):
    """Engine-side adapter operations the residency manager drives.

    ``load_adapter`` must return only once the engine can serve requests that
    name ``name``; raising means the adapter never became resident.
    ``files`` is the adapter's PEFT directory.
    ``unload_adapter`` raising means the slot may still be occupied.
    """

    @abstractmethod
    def load_adapter(self, name: str, files: WeightFiles) -> None: ...

    @abstractmethod
    def unload_adapter(self, name: str) -> None: ...


class InferenceEngine(ABC):
    """One inference engine as its control actor, backend and weight updaters see it.

    Each native integration implements this on the object its Ray engine actor
    is built from (``ReefSGLangEngine``, ``ReefVLLMEngine``). The methods are
    the vocabulary those callers already use by name; engine launch and the
    engine's own transports stay on the implementation. Results that only
    report the engine's answer are returned as ``object``.
    """

    @abstractmethod
    def get_url(self) -> str: ...

    @abstractmethod
    def shutdown(self) -> None: ...

    @abstractmethod
    def get_runtime_load_id(self) -> str: ...

    @abstractmethod
    def set_runtime_load_id(self, runtime_load_id: str) -> object: ...

    @abstractmethod
    def pause_generation(self, mode: str) -> object:
        """Stop scheduling; ``in_place`` keeps in-flight requests' KV, ``retract`` frees it and resets the cache."""

    @abstractmethod
    def continue_generation(self) -> object: ...

    @abstractmethod
    def flush_cache(self) -> None:
        """Reset the shared prefix cache; raise when requests still hold KV."""

    @abstractmethod
    def update_weights_from_disk(self, model_path: str, *, runtime_load_id: str | None = None) -> object:
        """Load the weights under ``model_path`` and, when given, serve them as ``runtime_load_id``."""

    @abstractmethod
    def load_lora_adapter_from_disk(self, lora_name: str, lora_path: str) -> object:
        """Load a PEFT adapter directory the engine's host can read, under ``lora_name``."""

    @abstractmethod
    def unload_lora_adapter(self, lora_name: str) -> object: ...
