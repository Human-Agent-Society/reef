"""Backend-neutral algorithm objectives, before optimizer or worker partitioning."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from reef.core.batches import StepScheduling, TrainingBatch
from reef.train.algos.signals import StepSignal


class TrainingObjective(ABC):
    """Own a method's batch preparation and choice of backend loss implementation.

    Declare ``name`` and ``loss_family`` and implement ``prepare``. Preparation
    receives the complete reserved batch, so group-relative statistics do not
    depend on optimizer or data-parallel partitioning. Model-dependent terms
    remain in the backend loss implementation. Return proposed algorithm state;
    the trainer owns its commit, including retries and recovery.

    How the runtime cuts a batch into optimizer steps is the recipe's
    ``StepScheduling``, not the objective's: the objective only declares what
    its loss tolerates (``supports_multiple_epochs``) and rejects a schedule
    it cannot train correctly through :meth:`validate_scheduling`.

    Keep this module and method implementations independent of torch, Slime,
    and Tinker. A recipe's loss family must be inspectable before workers start.
    """

    name: str = ""
    loss_family: str = ""
    #: Whether the loss stays valid when a batch is trained for more than one
    #: pass (``StepScheduling.epochs > 1``). Old-policy log-probs are computed
    #: once before the first pass, so later passes are off-policy: a clipped
    #: PPO-style ratio tolerates that, an unclipped one does not.
    supports_multiple_epochs: bool = False

    def validate_scheduling(self, scheduling: StepScheduling) -> None:
        """Reject a step schedule this objective's loss cannot train correctly."""
        if scheduling.epochs > 1 and not self.supports_multiple_epochs:
            raise ValueError(
                f"objective {self.name!r} does not support StepScheduling(epochs={scheduling.epochs}): "
                "its loss is not clipped for off-policy passes"
            )

    @abstractmethod
    def prepare(self, batch: TrainingBatch, state: Mapping[str, Any]) -> StepSignal:
        """Prepare a full batch without mutating it or committing algorithm state."""
