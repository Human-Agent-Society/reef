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

import json
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Literal

from reef.artifact.artifact import Artifact
from reef.core.errors import InferenceUnavailable, ReefError
from reef.core.evaluation import ArtifactKind, PolicyModel
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
        """Count one active request; ``InferenceUnavailable`` once admission is closed."""
        with self._lock:
            if not self._open:
                raise InferenceUnavailable("the inference runtime is shutting down")
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


@dataclass(frozen=True)
class InferenceArtifact:
    """The selected component identity and request-time requirements for inference.

    ``artifact_id`` is a nonempty component ID. ``local_path`` is None or a
    Path supplied by local evaluation. ``require_token_runtime_load_ids`` is
    a bool: true requires engine stamps for each generated token; false
    permits a fixed local model's stamp. This value contains no storage locator.
    Serving and evaluation construct it; handlers read its fields without I/O.
    """

    artifact_id: str
    local_path: Path | None = None
    require_token_runtime_load_ids: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_id, str) or not self.artifact_id:
            raise ValueError("inference requires a nonempty artifact_id")
        if self.local_path is not None and not isinstance(self.local_path, Path):
            raise TypeError("inference local_path must be a Path or None")
        if not isinstance(self.require_token_runtime_load_ids, bool):
            raise TypeError("require_token_runtime_load_ids must be a bool")

    @classmethod
    def from_artifact(cls, artifact: Artifact) -> InferenceArtifact:
        """Copy identity and available path; only unpublished local artifacts permit unstamped tokens."""
        return cls(artifact.artifact_id, artifact.local_path, not artifact.ref.local)


#: The state of the engine behind a route (RFC #1269): ``warm`` answers now, ``starting`` runs but does not answer
#: yet, ``idle`` runs no engine, and a request starts one.
EngineState = Literal["warm", "starting", "idle"]


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
    training and optimizer state belong to a training component.
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

        Reef calls it at a commit that activates a snapshot, at an activation
        and at a start, before traffic routes to the snapshot. ``scenario`` names
        the snapshot's scenario exactly when the runtime serves a separate
        adapter per scenario (``per_scenario_adapters``). A raise leaves the
        runtime serving what it served before. A runtime that loads weights
        makes a new runtime load ID for each load; a runtime that loads
        nothing, such as a proxy, returns the release ID.
        """

    def served_model_name(self, scenario: str) -> str:
        """The name under which the engine serves ``scenario``'s weights now; ``model_path`` unless a runtime serves
        each scenario under its own name."""
        return self.model_path

    def served_artifact(self, artifact_id: str, scenario: str) -> ServedArtifact | None:
        """The ``ServedArtifact`` that reaches ``artifact_id``, the artifact that serves ``scenario`` now, from
        outside this process: the engine's HTTP(S) ``base_url`` and ``served_model_name``.

        None when the engine has no such address or name, so an inference service cannot reach it without this
        runtime; startup then fails instead of publishing a route.
        """
        name = self.served_model_name(scenario)
        if not name:
            return None
        try:
            return ServedArtifact(artifact_id, self.base_url, name)
        except ValueError:
            # The base URL is no HTTP(S) URL that another process can call.
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

        Raise as ``artifact_model`` does.
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
        """Stop every engine from scheduling generation, so that a weight load can follow."""

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
    report the engine's answer are returned as ``object``. ``get_runtime_load_id``
    returns ``None`` on a node that is not the engine's node zero.
    """

    @abstractmethod
    def get_url(self) -> str: ...

    @abstractmethod
    def health_generate(self, timeout: float = 5) -> bool: ...

    @abstractmethod
    def shutdown(self) -> None: ...

    @abstractmethod
    def get_runtime_load_id(self) -> str | None: ...

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
