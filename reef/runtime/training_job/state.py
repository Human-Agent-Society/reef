"""Live progress shared by training and publication within one coordinator."""

from dataclasses import dataclass


@dataclass
class TrainingJobState:
    """Health phase; durable recovery decisions always come from the job marker."""

    phase: str = "serving"
