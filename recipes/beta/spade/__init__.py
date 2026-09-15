"""SPADE (arXiv:2608.19197): self play in adaptive synthetic executable environments, one method, one package.

Reef knows one task format, Harbor; everything SPADE needs beyond it lives here.

- ``environment_loader``: load and play a Gym style environment class as the reference code does; ships into
  every task's container and verifier.
- ``tasks``: a generated environment as a Harbor task any Harbor agent can play: the class behind root only
  ``observe`` and ``act`` commands inside the container, the verifier replaying the action log, SPADE's naming,
  metadata and hint, and a train/eval split per generation.
- ``process``: the class in a child interpreter on the host, for the Designer's smoke test.

The Environment Designer call, the regeneration step and the training side follow.
"""

from recipes.beta.spade.process import EnvironmentProcess, EnvironmentProcessError
from recipes.beta.spade.tasks import GeneratedEnvironment, environment_task, split_generation

__all__ = [
    "EnvironmentProcess",
    "EnvironmentProcessError",
    "GeneratedEnvironment",
    "environment_task",
    "split_generation",
]
