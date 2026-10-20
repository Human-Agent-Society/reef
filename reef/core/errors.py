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
    record ID ``agent_record_id``. Its only state is its message. The message
    names the scenario, and the record, or "the request" when
    ``agent_record_id`` is ``None``.
    Purpose: ``RequestService._prepare_request`` raises it, with no record
    ID, for a recorded request (not an evaluation call). ``validate_record``
    raises it, with the record ID, for an inference record that is not an
    identical retry of a stored one. Each does so when the scenario refuses
    image content and ``has_image_content`` is true. The service
    returns HTTP 400 with its message. It is also a ``ValueError``, so record
    ingest counts it in ``ingest/rejected_request_total``.
    """

    def __init__(self, scenario: str, *, agent_record_id: str | None = None) -> None:
        subject = "the request" if agent_record_id is None else f"record {agent_record_id!r}"
        super().__init__(
            f"{subject} carries image content, and the recipe of scenario {scenario!r} does not support images. "
            "Reef refuses image content for this recipe."
        )
