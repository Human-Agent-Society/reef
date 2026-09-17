"""Distillation on the Slime backend: one implementation that every distilling recipe's family shares.

- ``algorithm`` — the driver side, torch-free: the two axes, teacher source
  and divergence, with the representation (whole distribution or top-K),
  the EMA rate, the importance-sampling cap and the skipped tokens
  (:class:`DistillSettings`); the six-column wire row (the policy row plus
  ``teacher_tokens``); and :class:`DistillAlgorithm`, the base a recipe's
  family subclasses with its prefix and its defaults.
- ``objective`` — the worker side: forward and reverse KL and the
  generalized JSD over the vocabulary shards of tensor parallel, exact or
  top-K, with their gradients; the loss and the pre-train hook a family's
  ``objective.py`` forwards to.
- ``teacher`` — the teacher's weights (the student's current ones, a
  slow-moving copy of them, or a separate checkpoint, switched in and out
  through the actor's backups) and the forward-only pass over every
  sample's teacher sequence.
"""

from reef.train.slime_backend.distill.algorithm import (
    DIVERGENCES,
    TEACHER_BATCH_KEYS,
    TEACHER_SOURCES,
    DistillAlgorithm,
    DistillSettings,
    settings_from_args,
)

__all__ = [
    "DIVERGENCES",
    "TEACHER_BATCH_KEYS",
    "TEACHER_SOURCES",
    "DistillAlgorithm",
    "DistillSettings",
    "settings_from_args",
]
