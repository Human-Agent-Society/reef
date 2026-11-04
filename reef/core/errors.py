"""Common error hierarchy for Reef.

All Reef-raised exceptions derive from :class:`ReefError` so callers can
catch any domain error with one exception type. ``inference_aborted`` reads
how a response reports an engine abort, for the service, the training
pipeline and the training components alike.
"""

from collections.abc import Mapping


class ReefError(Exception):
    """Base class for all Reef-raised exceptions."""


class UnknownScenario(ReefError):
    """The scenario does not exist and the caller may not create it implicitly."""


class ScenarioLimitReached(ReefError):
    """The service holds its maximum number of registered scenarios. It refuses a new scenario."""


class ScenarioCapacityFull(ReefError):
    """The service holds its maximum number of loaded scenarios, and too few of them are idle to unload."""

    def __init__(self, message: str, *, retry_after_seconds: int = 30) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class DeployConfigError(ReefError):
    """A ``reef serve`` deployment config cannot be loaded or is invalid."""


class InferenceUnavailable(ReefError):
    """The deployment serves no inference requests, for example one with ``inference.backend: none``."""


class CommitPending(ReefError):
    """A training result waits for its scenario commit, so an activation or a new training start must wait too."""


class TrainingStartRefused(ReefError):
    """A trainer cannot start training from the artifact it was given; nothing changed."""


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


class RuntimeConfigError(ReefError):
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
