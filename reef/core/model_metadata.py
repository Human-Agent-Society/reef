"""Model capabilities used when configuring a harness for a served model."""

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelMetadata:
    """The context window in tokens and whether the endpoint accepts reasoning effort."""

    context_window: int
    reasoning: bool

    def __post_init__(self) -> None:
        if (
            isinstance(self.context_window, bool)
            or not isinstance(self.context_window, int)
            or self.context_window <= 0
        ):
            raise ValueError("model metadata context_window must be a positive integer")
        if not isinstance(self.reasoning, bool):
            raise ValueError("model metadata reasoning must be a boolean")

    @classmethod
    def from_config(cls, value: object) -> "ModelMetadata":
        """Read explicit capabilities without silently accepting misspelled fields."""
        if not isinstance(value, Mapping) or set(value) != {"context_window", "reasoning"}:
            raise ValueError("model metadata requires context_window and reasoning")
        return cls(context_window=value["context_window"], reasoning=value["reasoning"])
