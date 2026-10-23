"""Common error hierarchy for Reef.

All Reef-raised exceptions derive from :class:`ReefError` so callers can
catch any domain error with one exception type.
"""


class ReefError(Exception):
    """Base class for all Reef-raised exceptions."""


class UnknownScenario(ReefError):
    """The scenario does not exist and the caller may not create it implicitly."""


class ScenarioBusy(ReefError):
    """The scenario has a training job out at its backend; the operation waits for the job to land."""


class ScenarioLimitReached(ReefError):
    """The service holds its maximum number of registered scenarios. It refuses a new scenario."""


class ScenarioCapacityFull(ReefError):
    """The service holds its maximum number of loaded scenarios, and too few of them are idle to unload."""

    def __init__(self, message: str, *, retry_after_seconds: int = 30) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class DeployConfigError(ReefError):
    """A ``reef serve`` deployment config cannot be loaded or is invalid."""


class ImageContentNotSupported(ReefError, ValueError):
    """A recorded request or a record carries image content, and the scenario's recipe does not support images.

    Values: the error made from a scenario name ``scenario`` and an optional
    record ID ``agent_record_id``. The constructor raises ``TypeError``
    unless ``scenario`` is a str and ``agent_record_id`` is a str or
    ``None``. Its only state is its message. The message names the
    scenario, and the record, or "the request" when ``agent_record_id`` is
    ``None``.
    Purpose: ``validate_inference_content`` raises it when image content
    reaches a weight-training recipe without image support. Recorded requests
    are checked after request hooks; record ingress preserves identical
    retries of stored records. The service returns HTTP 400 with the message.
    It is also a ``ValueError``, so record ingress counts refusals in
    ``ingest/rejected_request_total``.
    """

    def __init__(self, scenario: str, *, agent_record_id: str | None = None) -> None:
        if not isinstance(scenario, str):
            raise TypeError("scenario must be a str")
        if agent_record_id is not None and not isinstance(agent_record_id, str):
            raise TypeError("agent_record_id must be a str or None")
        subject = "the request" if agent_record_id is None else f"record {agent_record_id!r}"
        super().__init__(
            f"{subject} carries image content, and the recipe of scenario {scenario!r} does not support images. "
            "Reef refuses image content for this recipe."
        )
