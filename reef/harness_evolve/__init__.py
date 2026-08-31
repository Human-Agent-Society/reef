"""Harness composition evolution: one method, one package.

- ``recipe`` — the harness-evolution recipe class; ``propose`` and
  ``evaluate`` bind as dotted callable references.
- ``processor`` — batches scored traces for the propose/evaluate/select
  loop.

The backend that runs the loop (``HarnessEvolveBackend``) is the harness
surface's training runtime, ``reef.train.harness_backend`` — shared
machinery, the counterpart of ``reef.train.slime_backend`` for weights.
"""

from reef.harness_evolve.processor import HarnessEvolveProcessor
from reef.harness_evolve.recipe import HarnessEvolveRecipe

__all__ = ["HarnessEvolveProcessor", "HarnessEvolveRecipe"]
