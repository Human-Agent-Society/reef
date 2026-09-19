"""Abstract generation and validation hooks for processors that produce tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from reef.core.records_types import AgentRecord
from reef.core.tasks import HarborTask
from reef.train.processors.base import DataProcessor


@dataclass(frozen=True)
class TaskGenerationRequest:
    """One task's source records, requirements, and optional local assets.

    Assets are files or directories accessible to the generator, such as a
    repository snapshot or verifier fixtures. Construction does not read them.
    Method-specific settings belong to the processor's configuration. A method
    that writes tasks from the description alone, such as a designer prompted
    with a target, passes no source records.
    """

    source_records: tuple[AgentRecord, ...]
    description: str
    assets: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.source_records, tuple):
            raise ValueError("source_records must be a tuple of AgentRecord values")
        if any(not isinstance(record, AgentRecord) for record in self.source_records):
            raise TypeError("source_records must contain AgentRecord values")
        if len({record.scenario for record in self.source_records}) > 1:
            raise ValueError("source_records must belong to one scenario")
        record_ids = [record.agent_record_id for record in self.source_records]
        if any(not record_id for record_id in record_ids) or len(set(record_ids)) != len(record_ids):
            raise ValueError("source_records must have distinct non-empty record ids")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("description must be non-empty text")
        if not isinstance(self.assets, tuple) or any(not isinstance(path, Path) for path in self.assets):
            raise TypeError("assets must be a tuple of pathlib.Path values")


@dataclass(frozen=True)
class TaskValidationResult:
    """A completed validation: no errors means the task passed all required checks.

    Errors describe task defects. Failures to execute the checks (for example,
    an unavailable container runtime) raise exceptions instead of rejecting the
    task as invalid.
    """

    errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.errors, tuple):
            raise TypeError("errors must be a tuple of non-empty strings")
        if any(not isinstance(error, str) or not error.strip() for error in self.errors):
            raise ValueError("errors must contain non-empty strings")

    @property
    def is_valid(self) -> bool:
        """Whether every required check passed."""
        return not self.errors


class TaskGenerationProcessor(DataProcessor, ABC):
    """The method contract for turning records into validated Harbor tasks.

    This ABC declares the two asynchronous hooks only. It does not yet supply
    background execution, task publication, retries, or record-to-batch assembly.
    Implementing the hooks alone retains DataProcessor's no-update lifecycle.

    A lifecycle implementation must run these hooks outside the trainer lock,
    preserve source records until acknowledgement, and expose only validated,
    complete task directories through TaskItem. Reserved directories must remain
    accessible to the consumer until consumption finishes. Its synchronous
    ingest/ready/build_batch methods must not wait for generation or validation.
    """

    @abstractmethod
    async def generate(self, request: TaskGenerationRequest) -> HarborTask:
        """Generate one task from records belonging to this processor's scenario.

        Return the task specification without publishing a directory or starting
        training. Preserve the request's ordered source record ids in the task's
        source_agent_record_ids. Model calls and other slow work belong here;
        generation failures propagate to the lifecycle implementation.
        """

    @abstractmethod
    async def validate(self, task_path: Path) -> TaskValidationResult:
        """Check a materialized candidate before it becomes a ready TaskItem.

        Run the method's required structural and execution checks without
        modifying the candidate. Return task defects as validation errors;
        raise execution failures so the lifecycle can handle retries separately.
        Validation alone does not publish a task or consume its source records.
        """
