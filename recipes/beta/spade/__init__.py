"""SPADE (arXiv:2608.19197): self play in adaptive synthetic executable environments, one method, one package.

- ``tasks``: a generated environment as a gym task (``reef.core.tasks.gym``) with SPADE's naming, metadata and
  hint, and a train/eval split per generation.

The Environment Designer call, the regeneration step and the training side follow.
"""

from recipes.beta.spade.tasks import GeneratedEnvironment, environment_task, split_generation

__all__ = ["GeneratedEnvironment", "environment_task", "split_generation"]
