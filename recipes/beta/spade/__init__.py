"""SPADE (arXiv:2608.19197): self play in adaptive synthetic executable environments, one method, one package.

Reef knows one task format, Harbor, and writes, checks and plays Harbor tasks in ``reef.record2dataset``.
What SPADE adds is the method: the Environment Designer's adversarial experience section, the two arms
each task is played with (as it is, and with the hint), the hint based regret as the Designer's signal,
and the Reasoning Agent's group relative training on the plain arm's episodes.

- ``generation``: the experience section (results sorted by regret into the frontier, the mastered and the
  out of reach), a generation's records and its report file.
- ``processor``: the reported feedback half (episodes grouped by task) and the task generation half (the
  Designer's generations, run on a worker through the generator service).
- ``designer_processor``: the Designer's reports grouped by generation, its regret as the reward; skills
  compare through the group id.
- ``objective``: group relative advantages per group on Tinker's importance sampling loss, shared by both roles.
- ``recipe``: the two recipes, the Reasoning Agent's on its episodes and the Designer's on its regret.
- ``harness``: the Designer's prompt as a harness tree the evolution loop rewrites from the regret reports.
"""

from recipes.beta.spade.designer_processor import SpadeDesignerProcessor
from recipes.beta.spade.generation import (
    GenerationRecord,
    GenerationSummary,
    PlayRecord,
    ProposalRecord,
    TaskMeasure,
    experience_for,
    experience_text,
    load_experience,
)
from recipes.beta.spade.harness import (
    DESIGNER_SEED,
    ReportedRegretSelection,
    SpadeDesignerHarnessRecipe,
    never_scored,
    propose_prompt,
)
from recipes.beta.spade.objective import SpadeObjective
from recipes.beta.spade.processor import ProposalRefused, SpadeProcessor
from recipes.beta.spade.recipe import SpadeDesignerRecipe, SpadeRecipe

__all__ = [
    "DESIGNER_SEED",
    "GenerationRecord",
    "GenerationSummary",
    "PlayRecord",
    "ProposalRecord",
    "ProposalRefused",
    "ReportedRegretSelection",
    "SpadeDesignerHarnessRecipe",
    "SpadeDesignerProcessor",
    "SpadeDesignerRecipe",
    "SpadeObjective",
    "SpadeProcessor",
    "SpadeRecipe",
    "TaskMeasure",
    "experience_for",
    "experience_text",
    "load_experience",
    "never_scored",
    "propose_prompt",
]
