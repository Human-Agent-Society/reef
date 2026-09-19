"""SPADE (arXiv:2608.19197): self play in adaptive synthetic executable environments, one method, one package.

Reef knows one task format, Harbor, and writes, checks and plays Harbor tasks in ``reef.record2dataset``.
What SPADE adds is the method: the Environment Designer's adversarial experience section, the two arms
each task is played with (as it is, and with the hint), the hint based regret as the Designer's signal,
and the Reasoning Agent's group relative training on the plain arm's episodes.

- ``generation``: the experience section (results sorted by regret into the frontier, the mastered and the
  out of reach), a generation's records and its report file.
- ``processor``: the reported feedback half (episodes grouped by task) and the task generation half (the
  Designer's generations, run on a worker through the generator service).
- ``objective``: group relative advantages per task group on Tinker's importance sampling loss.
- ``recipe``: the configuration that binds them.

The Designer's own training follows.
"""

from recipes.beta.spade.generation import (
    GenerationRecord,
    PlayRecord,
    ProposalRecord,
    TaskMeasure,
    experience_for,
    experience_text,
    load_experience,
)
from recipes.beta.spade.objective import SpadeObjective
from recipes.beta.spade.processor import ProposalRefused, SpadeProcessor
from recipes.beta.spade.recipe import SpadeRecipe

__all__ = [
    "GenerationRecord",
    "PlayRecord",
    "ProposalRecord",
    "ProposalRefused",
    "SpadeObjective",
    "SpadeProcessor",
    "SpadeRecipe",
    "TaskMeasure",
    "experience_for",
    "experience_text",
    "load_experience",
]
