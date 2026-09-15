"""SPADE (arXiv:2608.19197): self play in adaptive synthetic executable environments, one method, one package.

- ``environment_loader``: load and play a generated environment as the reference code does; ships into every task.
- ``tasks``: a generated environment as a Harbor task directory, with a replay verifier and a split per generation.

The Designer call, the driver that plays the environments and the training side follow.
"""

from recipes.beta.spade.environment_loader import (
    environment_class_name,
    episode_return,
    extract_boxed_answer,
    load_environment_class,
    make_environment,
    normalized_action,
    play,
)
from recipes.beta.spade.tasks import GeneratedEnvironment, environment_task, split_generation

__all__ = [
    "GeneratedEnvironment",
    "environment_class_name",
    "environment_task",
    "episode_return",
    "extract_boxed_answer",
    "load_environment_class",
    "make_environment",
    "normalized_action",
    "play",
    "split_generation",
]
