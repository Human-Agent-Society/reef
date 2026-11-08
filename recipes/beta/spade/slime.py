"""SPADE's Slime objective: task jobs that center the plays by SPADE's rule and carry the Designer's replies.

Only Slime's training driver and the workers resolve this module, through the reference that the package registers.
It imports no torch, Ray or Slime.
"""

from collections.abc import Mapping
from typing import Any

from reef.core.job_data import JobData
from reef.train.algos.group_advantages import GroupAdvantageRule
from reef.train.algos.signals import TrainingSignal
from reef.train.slime_backend.objectives import SlimeObjective, register_slime_objective

#: SPADE's per-game z-score of the plays: the population deviation (``np.std``) plus 1e-8 (``game_utils.py``).
SPADE_ROLLOUT_ADVANTAGES = GroupAdvantageRule(divide_by_std=True, std="population", epsilon=1e-8)


@register_slime_objective
class SpadeObjective(SlimeObjective):
    """A SPADE task job: Slime's own loop over the generation's tasks, with the Designer's delayed replies.

    Slime rolls each task out and trains its loss, as for ``slime_dataset``. The plays of each task center by SPADE's
    rule, and the job data's delayed samples, the Designer replies of an earlier generation with the advantages that
    the generator gave them, train in the job's first optimizer step.
    """

    name = "spade"
    loss_family = "slime"
    rollout_advantages = SPADE_ROLLOUT_ADVANTAGES
    carries_samples = True

    def prepare(self, data: JobData, state: Mapping[str, Any]) -> TrainingSignal:
        raise RuntimeError("a SPADE task job trains on Slime's own rollouts; Reef prepares no training signal for it")


__all__ = ["SPADE_ROLLOUT_ADVANTAGES", "SpadeObjective"]
