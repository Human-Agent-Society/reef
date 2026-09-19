"""Common error hierarchy for Reef.

All Reef-raised exceptions derive from :class:`ReefError` so callers can
catch any domain error with one exception type.
"""


class ReefError(Exception):
    """Base class for all Reef-raised exceptions."""


class UnknownScenario(ReefError):
    """The scenario does not exist and the caller may not create it implicitly."""


class DeployConfigError(ReefError):
    """A ``reef serve`` deployment config cannot be loaded or is invalid."""
