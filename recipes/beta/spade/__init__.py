"""SPADE (arXiv:2608.19197): self play in adaptive synthetic executable environments, as a task generator.

Reef knows one task format, Harbor, and writes and checks Harbor tasks and runs their rollouts in ``reef.record2sample.generator``.
The built-in generated-task recipe runs the tasks and trains on them. What SPADE adds is the generator:
the Environment Designer's adversarial experience section, the hint arm each task asks for, and the hint
based regret as the Designer's signal.

- ``generation``: the experience section (results sorted by regret into the frontier, the mastered and the
  out of reach), the Designer's score of a task (``designer_score``) and the advantages of a generation's replies
  (``centered_advantages``), a generation's records and its report document.
- ``generator``: ``SpadeTaskGenerator``, the Designer's generations through the generator service.
- ``slime``: ``SpadeObjective``, the objective ``spade`` of Slime task jobs: SPADE's z-score of the plays, and the
  Designer's delayed replies in the job's first optimizer step.
"""

from recipes.beta.spade.generation import (
    GenerationRecord,
    PlayRecord,
    ProposalRecord,
    ScoreSettings,
    TaskMeasure,
    centered_advantages,
    designer_score,
    experience_for,
    experience_text,
    load_experience,
    plateau_reward,
)
from recipes.beta.spade.generator import SpadeTaskGenerator
from reef.train.algos.registry import register_objective_ref

register_objective_ref("spade", "recipes.beta.spade.slime:SpadeObjective")

__all__ = [
    "GenerationRecord",
    "PlayRecord",
    "ProposalRecord",
    "ScoreSettings",
    "SpadeTaskGenerator",
    "TaskMeasure",
    "centered_advantages",
    "designer_score",
    "experience_for",
    "experience_text",
    "load_experience",
    "plateau_reward",
]
