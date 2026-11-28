"""Common error hierarchy for Reef.

All Reef-raised exceptions derive from :class:`ReefError` so callers can
catch any domain error with one exception type. Each class name ends in
``Error``. :class:`ConfigError` is the base of every configuration error.
:class:`InternalError` is the base of every failure that the caller's
request did not cause. ``inference_aborted`` reads how a response reports
an engine abort, for the service, the training pipeline and the training
components alike. ``UpstreamLimit`` names the provider limit that refused a
request, for the handler that reads the provider's reply and for the evaluation
that acts on it.
"""

import math
from collections.abc import Mapping
from enum import Enum


class ReefError(Exception):
    """Base class for all Reef-raised exceptions."""


class InternalError(ReefError):
    """A failure inside Reef or its environment that the caller's request did not cause.

    It names a failure of a process, an engine, a sandbox, a store or a deployment step. Its one purpose is to
    separate such a failure from a refused request: the HTTP layer does not translate it, so a route answers 500,
    and a worker handles it as any other unexpected error.
    """


class ScenarioNotFoundError(ReefError):
    """The scenario does not exist and the caller may not create it implicitly."""


class ScenarioDeletedError(ReefError):
    """The scenario ID belonged to a scenario that was deleted. A scenario ID is never used again after its delete,
    so a new scenario needs a new ID."""


class ScenarioLimitReachedError(ReefError):
    """The service holds its maximum number of registered scenarios. It refuses a new scenario."""


class ScenarioCapacityFullError(ReefError):
    """The service holds its maximum number of loaded scenarios, and too few of them are idle to unload."""

    def __init__(self, message: str, *, retry_after_seconds: int = 30) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class ConfigError(ReefError, ValueError):
    """A configuration cannot be loaded or is invalid.

    Each subclass names the configuration: a deployment, a ``runtime`` section or training component, a recipe, or
    a plugin setting. It is a ``ValueError``, so code that reads a config value can refuse it in one way.
    """


class DeployConfigError(ConfigError):
    """A ``reef serve`` deployment config cannot be loaded or is invalid."""


class InferenceUnavailableError(ReefError):
    """The deployment serves no inference requests, for example one with ``inference.backend: none``."""


class CommitPendingError(ReefError):
    """A training result waits for its scenario commit, so an activation or a new training start must wait too."""


class TrainingStartRefusedError(ReefError):
    """A trainer cannot start training from the artifact it was given; nothing changed."""


class UpstreamLimit(Enum):
    """The kind of limit that refused a request, as the provider's status, ``Retry-After`` and structured fields say.

    Values: ``RATE`` is a limit that ends with time, so the same request can pass later unchanged. ``PAYMENT`` is a
    payment the provider requires for this request, so the same request does not pass again on its own. Each value is
    also the error ``code`` with which the evaluation endpoint answers such a refusal.

    Purpose: ``reef.inference.http.upstream_limit`` makes it from a provider's reply. The evaluation endpoint sends a
    call that ``RATE`` refused again, and sends no call again that ``PAYMENT`` refused.
    """

    RATE = "rate_limit"
    PAYMENT = "payment_required"


class UpstreamStatusError(ReefError):
    """An upstream service answered an inference request with an error status.

    Carries ``status`` so the service can hand the caller the upstream's own
    4xx and message rather than collapsing both into an opaque 500. That
    distinction is load-bearing: agents correct themselves from these bodies
    (shrinking ``max_tokens`` when the engine reports a context overflow, for
    instance), and an opaque 500 leaves them retrying the identical request
    until they give up.

    ``retry_after_seconds`` is the wait that the upstream asked for before the next try, from its ``Retry-After``
    header: a finite number of seconds of at least 0, or None when the upstream named no valid wait. ``limit`` is the
    limit that refused the request (``UpstreamLimit``), or None when the reply names none. The constructor raises
    ``ValueError`` for another ``retry_after_seconds`` and ``TypeError`` for another ``limit``.
    """

    def __init__(
        self,
        message: str,
        *,
        status: int,
        retry_after_seconds: float | None = None,
        limit: UpstreamLimit | None = None,
    ) -> None:
        super().__init__(message)
        if retry_after_seconds is not None and not (math.isfinite(retry_after_seconds) and retry_after_seconds >= 0):
            raise ValueError("retry_after_seconds is a finite number of seconds of at least 0, or None")
        if limit is not None and not isinstance(limit, UpstreamLimit):
            raise TypeError("limit is an UpstreamLimit or None")
        self.status = status
        self.retry_after_seconds = retry_after_seconds
        self.limit = limit


class MalformedUpstreamResponseError(UpstreamStatusError):
    """An upstream service answered an inference request with a 2xx status whose body was not a valid JSON object.

    Values: an ``UpstreamStatusError`` with ``status=502``, ``retry_after_seconds=None``, and ``limit=None``.
    Purpose: ``HttpInferenceHandler.post_json`` raises it so downstream callers map the failure to HTTP 502 while
    knowing the upstream already processed (and may have billed) the request, so it must not be retried as a
    transient provider 502.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, status=502)


class RuntimeConfigError(ConfigError):
    """A ``runtime`` section or a training component config is invalid or misses a required key."""


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


def inference_incomplete(response: Mapping[str, object]) -> bool:
    """Whether a response explicitly reports an abort or incomplete stream delivery.

    Diagnostic records remain readable but cannot become training samples. Older buffered records without
    delivery metadata retain their existing eligibility; a provider's own ``complete`` field is not a receipt.
    """
    delivery = response.get("stream_delivery")
    return (
        inference_aborted(response)
        or (isinstance(delivery, Mapping) and delivery.get("complete") is False)
        or (response.get("stream") is True and response.get("complete") is False)
    )
