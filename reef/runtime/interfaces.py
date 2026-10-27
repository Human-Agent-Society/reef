"""Shared inference contracts, training values, and default request admission.

Who implements what:

- Recipes and the dispatcher *use* ``InferenceRuntime`` and the training
  values that jobs carry (``TrainingMethod``, ``LearningRateSchedule``).
- An inference integration *implements* ``InferenceHandler`` for requests,
  ``InferenceBackend`` for loading weight files into its engines,
  ``InferenceEngine`` for each engine, and ``AdapterEngine`` for the adapter
  operations of ``reef.runtime.residency``. The engine supervision hooks live
  in ``recovery`` next to the objects that drive them.

The module reads top-down in that order: identities and errors, training
values, request admission, request handling, runtime contracts, native
backend contracts.
"""

from __future__ import annotations

import json
import math
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from numbers import Integral, Real
from pathlib import Path
from threading import Lock
from typing import Any, Literal

from reef.artifact.artifact import Artifact
from reef.core.errors import InferenceUnavailable, ReefError
from reef.core.weight_layout import WEIGHT_LAYOUTS, WeightLayout
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


# -- Training values ----------------------------------------------------------


LearningRateDecayStyle = Literal["constant", "linear", "cosine"]
LEARNING_RATE_DECAY_STYLES: tuple[LearningRateDecayStyle, ...] = ("constant", "linear", "cosine")


def checked_learning_rate(value: object, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite number >= 0, got {value!r}")
    return float(value)


def checked_step_count(value: object, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return int(value)


@dataclass(frozen=True)
class LearningRateSchedule:
    """Linear warmup to ``peak_learning_rate``, then a constant, linear or cosine decay.

    Step ``s`` is the number of optimizer steps this schedule has already
    completed, so its first step uses ``s = 0``. During warmup
    (``s <= warmup_steps``) the rate rises linearly from
    ``initial_learning_rate`` to the peak. ``decay_style="constant"`` then holds
    the peak; ``"linear"`` and ``"cosine"`` decay it to ``min_learning_rate``
    over the next ``decay_steps`` steps and hold that minimum afterwards. This
    is Megatron's ``OptimizerParamScheduler`` curve counted in optimizer steps,
    so Slime and Tinker train a schedule the same way.

    ``name`` identifies the schedule together with its values. The backend
    keeps the active schedule and its progress with its training state
    (:class:`LearningRateScheduleState`): a job that selects the active
    schedule continues it, a job that selects another one starts that one at
    step 0, and a job that selects none keeps the active one. To start the same
    curve again, give it another ``name``.
    """

    name: str
    peak_learning_rate: float
    warmup_steps: int = 0
    decay_style: LearningRateDecayStyle = "constant"
    decay_steps: int = 0
    min_learning_rate: float = 0.0
    initial_learning_rate: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("LearningRateSchedule.name must be a non-empty string")
        peak_learning_rate = checked_learning_rate(self.peak_learning_rate, "LearningRateSchedule.peak_learning_rate")
        if peak_learning_rate <= 0:
            raise ValueError("LearningRateSchedule.peak_learning_rate must be positive")
        min_learning_rate = checked_learning_rate(self.min_learning_rate, "LearningRateSchedule.min_learning_rate")
        initial_learning_rate = checked_learning_rate(
            self.initial_learning_rate, "LearningRateSchedule.initial_learning_rate"
        )
        if min_learning_rate > peak_learning_rate or initial_learning_rate > peak_learning_rate:
            raise ValueError(
                "LearningRateSchedule min_learning_rate and initial_learning_rate must not exceed the peak"
            )
        warmup_steps = checked_step_count(self.warmup_steps, "LearningRateSchedule.warmup_steps")
        decay_steps = checked_step_count(self.decay_steps, "LearningRateSchedule.decay_steps")
        if self.decay_style not in LEARNING_RATE_DECAY_STYLES:
            raise ValueError(
                f"LearningRateSchedule.decay_style must be one of {', '.join(LEARNING_RATE_DECAY_STYLES)}, "
                f"got {self.decay_style!r}"
            )
        if self.decay_style == "constant" and decay_steps > 0:
            raise ValueError("a constant LearningRateSchedule takes no decay_steps")
        if self.decay_style != "constant" and decay_steps <= 0:
            raise ValueError(f"a {self.decay_style} LearningRateSchedule needs positive decay_steps")
        object.__setattr__(self, "peak_learning_rate", peak_learning_rate)
        object.__setattr__(self, "min_learning_rate", min_learning_rate)
        object.__setattr__(self, "initial_learning_rate", initial_learning_rate)
        object.__setattr__(self, "warmup_steps", warmup_steps)
        object.__setattr__(self, "decay_steps", decay_steps)

    def learning_rate(self, completed_steps: int) -> float:
        """The rate of the optimizer step that follows ``completed_steps`` steps of this schedule."""
        if self.warmup_steps and completed_steps <= self.warmup_steps:
            warmup_fraction = completed_steps / self.warmup_steps
            learning_rate_range = self.peak_learning_rate - self.initial_learning_rate
            return self.initial_learning_rate + learning_rate_range * warmup_fraction
        if self.decay_style == "constant":
            return self.peak_learning_rate
        steps_since_warmup = completed_steps - self.warmup_steps
        if steps_since_warmup >= self.decay_steps:
            return self.min_learning_rate
        decay_fraction = steps_since_warmup / self.decay_steps
        if self.decay_style == "linear":
            decay_factor = 1.0 - decay_fraction
        else:
            decay_factor = 0.5 * (math.cos(math.pi * decay_fraction) + 1.0)
        return self.min_learning_rate + decay_factor * (self.peak_learning_rate - self.min_learning_rate)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> LearningRateSchedule:
        if not isinstance(value, Mapping):
            raise ValueError("a learning-rate schedule must be an object")
        return cls(**dict(value))


@dataclass(frozen=True)
class TrainingMethod:
    """The objective one training job trains with, and the learning-rate schedule it selects.

    A recipe's job plan selects one for every job (``reef.train.jobs``); the
    job carries it to the training component beside the job data, and the job's payload
    records it, so it is part of the job's identity and stays fixed across the
    job's retries. ``objective`` is a registered objective name or a dotted
    ``package.module:Objective`` reference; the backend resolves it in its own
    process, so a name another package registers is only known where that
    package was imported. ``learning_rate_schedule`` ``None`` keeps the active
    schedule.
    """

    objective: str
    learning_rate_schedule: LearningRateSchedule | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.objective, str) or not self.objective.strip():
            raise ValueError("TrainingMethod.objective must be a non-empty objective reference")
        if self.learning_rate_schedule is not None and not isinstance(
            self.learning_rate_schedule, LearningRateSchedule
        ):
            raise TypeError("TrainingMethod.learning_rate_schedule must be a LearningRateSchedule or None")

    def to_dict(self) -> dict[str, Any]:
        """The job payload's record of this method; it is part of the job's identity."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TrainingMethod:
        if not isinstance(value, Mapping):
            raise ValueError("a training method must be an object")
        schedule = value.get("learning_rate_schedule")
        return cls(value["objective"], None if schedule is None else LearningRateSchedule.from_dict(schedule))


@dataclass(frozen=True)
class LearningRateScheduleState:
    """The active schedule and the optimizer steps it has completed, kept with the training state."""

    schedule: LearningRateSchedule
    completed_steps: int = 0

    def __post_init__(self) -> None:
        checked_step_count(self.completed_steps, "LearningRateScheduleState.completed_steps")

    def learning_rates(self, optimizer_steps: int) -> tuple[float, ...]:
        """The rates of the next ``optimizer_steps`` steps, in order."""
        return tuple(self.schedule.learning_rate(self.completed_steps + step) for step in range(optimizer_steps))

    def advanced(self, optimizer_steps: int) -> LearningRateScheduleState:
        """This state after ``optimizer_steps`` more steps."""
        return replace(self, completed_steps=self.completed_steps + optimizer_steps)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> LearningRateScheduleState:
        if not isinstance(value, Mapping):
            raise ValueError("a learning-rate schedule state must be an object")
        return cls(LearningRateSchedule.from_dict(value["schedule"]), value["completed_steps"])


def resolve_learning_rate_schedule_state(
    active: LearningRateScheduleState | None, requested: LearningRateSchedule | None
) -> LearningRateScheduleState | None:
    """The schedule state a job trains with, given the active state and the job's request.

    No request keeps the active state (``None``: the backend's configured
    rate); the active schedule continues; any other schedule starts at step 0.
    """
    if requested is None or (active is not None and active.schedule == requested):
        return active
    return LearningRateScheduleState(requested)


def learning_rate_metrics(
    learning_rates: Sequence[float], schedule_state: LearningRateScheduleState | None
) -> dict[str, Any]:
    """The metrics every backend reports for a job's rate: its last optimizer step's, and the schedule's progress."""
    metrics: dict[str, Any] = {"learning_rate": learning_rates[-1]}
    if schedule_state is not None:
        metrics["learning_rate_schedule"] = {
            "name": schedule_state.schedule.name,
            "completed_steps": schedule_state.completed_steps,
        }
    return metrics


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


#: Delivery diagnostic retained by older records that stripped the abort finish reason.
#: Stored records keep this text and ``inference_aborted`` compares it exactly. Do not change it.
INFERENCE_ABORT_ERROR = "upstream engine aborted the generation"


def inference_aborted(response: Mapping[str, object]) -> bool:
    """Whether a native, provider, or stored response reports an engine abort.

    Read finish reasons without requiring training tensors. Older stored
    Anthropic aborts kept only the stream-delivery error after stripping tensors.
    """
    delivery = response.get("stream_delivery")
    if isinstance(delivery, Mapping) and delivery.get("error") == INFERENCE_ABORT_ERROR:
        return True

    def aborted(value: object) -> bool:
        return value == "abort" or (isinstance(value, Mapping) and value.get("type") == "abort")

    meta = response.get("meta_info")
    if aborted(response.get("finish_reason")) or (isinstance(meta, Mapping) and aborted(meta.get("finish_reason"))):
        return True
    training = response.get("training")
    if isinstance(training, Mapping) and aborted(training.get("finish_reason")):
        return True
    choices = response.get("choices")
    return isinstance(choices, list) and any(
        isinstance(choice, Mapping)
        and (
            aborted(choice.get("finish_reason"))
            or (isinstance(choice.get("meta_info"), Mapping) and aborted(choice["meta_info"].get("finish_reason")))
        )
        for choice in choices
    )


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
    def per_scenario_adapters(self) -> bool:
        """Whether the runtime serves each scenario its own adapter, which requests name per scenario."""
        return False

    @property
    def shared_by_training_scenarios(self) -> bool:
        """Whether several training scenarios can share this runtime, each served its own weights.

        A runtime that loads one model into its engines serves one training
        scenario: a second scenario's release would replace the first one's
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


@dataclass(frozen=True)
class WeightFiles:
    """Saved weights as files, for the serving engines to load.

    ``path`` is a directory on the control plane's node. ``layout`` names what
    it holds: ``"full"`` is a Hugging Face checkpoint of every model weight,
    and ``"adapter"`` is a PEFT adapter directory. The serving runtimes in
    ``reef.runtime.serving`` make the values from a published release and pass
    them to ``InferenceBackend.load_weight_files``.
    """

    path: Path
    layout: WeightLayout

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise TypeError("weight files path must be a Path")
        if self.layout not in WEIGHT_LAYOUTS:
            raise ValueError(f"weight files layout must be full or adapter, got {self.layout!r}")


class InferenceBackend(ABC):
    """Receiver operations with acknowledged completion and no commit policy.

    The serving runtimes of ``reef.runtime.serving`` call these operations on
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
