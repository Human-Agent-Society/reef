"""SPADE (arXiv:2608.19197): self play in adaptive synthetic executable environments, as a task generator.

Reef knows one task format, Harbor, and writes, checks and plays Harbor tasks in ``reef.record2task.generator``.
The built-in generated-task recipe plays the tasks and trains on them. What SPADE adds is the generator:
the Environment Designer's adversarial experience section, the hint arm each task asks for, and the hint
based regret as the Designer's signal.

- ``generation``: the experience section (results sorted by regret into the frontier, the mastered and the
  out of reach), a generation's records and its report file.
- ``generator``: ``SpadeTaskGenerator``, the Designer's generations through the generator service.

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
from recipes.beta.spade.generator import SpadeTaskGenerator

__all__ = [
    "GenerationRecord",
    "PlayRecord",
    "ProposalRecord",
    "SpadeTaskGenerator",
    "TaskMeasure",
    "experience_for",
    "experience_text",
    "load_experience",
]
