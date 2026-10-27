"""Backend-neutral algorithm objectives, before optimizer or worker partitioning."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from reef.core.job_data import JobData
from reef.train.algos.schedule import RolloutUnit, StepScheduling
from reef.train.algos.signals import TrainingSignal


class TrainingObjective(ABC):
    """Own a method's data preparation and choice of backend loss implementation.

    Declare ``name`` and ``loss_family`` and implement ``prepare``. Preparation
    receives the complete reserved job data, so group-relative statistics do not
    depend on optimizer or data-parallel partitioning. Model-dependent terms
    remain in the backend loss implementation. Return proposed algorithm state;
    the trainer owns its commit, including retries and recovery.

    A training component cuts the job's data into optimizer steps. The objective
    declares which rows one step keeps together (``rollout_unit``) and what
    its loss tolerates (``supports_multiple_epochs``), and it rejects a job's
    ``StepScheduling`` that it cannot train correctly
    (:meth:`validate_scheduling`).

    Keep this module and method implementations independent of torch, Slime,
    and Tinker. A recipe's loss family must be inspectable before workers start.
    """

    name: str = ""
    loss_family: str = ""
    #: Whether the loss stays valid when job data is trained for more than one
    #: pass (``StepScheduling.epochs > 1``). Old-policy log-probs are computed
    #: once before the first pass, so later passes are off-policy: a clipped
    #: PPO-style ratio tolerates that, an unclipped one does not.
    supports_multiple_epochs: bool = False
    #: The rows one optimizer step keeps together: ``"sample"`` (each sample is
    #: its own rollout) or ``"comparison_set"`` (a comparison set's samples
    #: stay in one step).
    rollout_unit: RolloutUnit = "sample"

    def validate_scheduling(self, scheduling: StepScheduling) -> None:
        """Raise ``ValueError`` for a job schedule this objective's loss cannot train correctly."""
        if scheduling.epochs > 1 and not self.supports_multiple_epochs:
            raise ValueError(
                f"objective {self.name!r} does not support {scheduling.epochs} epochs per job: "
                "its loss is not clipped for off-policy passes"
            )

    @abstractmethod
    def prepare(self, data: JobData, state: Mapping[str, Any]) -> TrainingSignal:
        """Prepare a job's full data without mutating it or committing algorithm state."""
