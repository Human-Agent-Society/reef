"""SPADE (arXiv:2608.19197): self play in adaptive synthetic executable environments, one method, one package.

Reef knows one task format, Harbor; everything SPADE needs beyond it lives here. The Designer writes an
environment in one of the three interfaces the community writes, ``harbor``, ``gym`` or ``openenv``, and
every kind becomes a Harbor task any Harbor agent can play.

- ``environment_loader``: load and play a Gym style environment class as the reference code does; ships into
  every gym task's container and verifier.
- ``tasks``: the gym kind as a Harbor task: the class behind root only ``observe`` and ``act`` commands inside
  the container, the verifier replaying the action log, SPADE's naming, metadata and hint, and a train/eval
  split per generation.
- ``process``: a class in a child interpreter on the host, for the Designer's smoke test.
- ``designer``: the Environment Designer's adversarial prompt for the three kinds, its replies parsed, and the
  smoke test of a ``gym`` class.
- ``harbor``: the harbor kind: a Harbor task written directly, the team's structural gate, and Harbor's oracle check.
- ``openenv``: the openenv kind: an OpenEnv environment package served inside the container behind ``serve``,
  and the check that it serves a reset and a step.

The regeneration step and the training side follow.
"""

from recipes.beta.spade.designer import (
    DesignerReplyError,
    DesignerRequest,
    GymReply,
    HarborReply,
    OpenEnvReply,
    PlayRecord,
    SmokeResult,
    designer_messages,
    designer_prompt,
    parse_gym_reply,
    parse_harbor_reply,
    parse_openenv_reply,
    smoke_test,
)
from recipes.beta.spade.harbor import GeneratedHarborTask, OracleResult, harbor_task, oracle_check
from recipes.beta.spade.openenv import GeneratedOpenEnvTask, OpenEnvCheck, openenv_check, openenv_task
from recipes.beta.spade.process import EnvironmentProcess, EnvironmentProcessError
from recipes.beta.spade.tasks import GeneratedEnvironment, environment_task, split_generation

__all__ = [
    "DesignerReplyError",
    "DesignerRequest",
    "EnvironmentProcess",
    "EnvironmentProcessError",
    "GeneratedEnvironment",
    "GeneratedHarborTask",
    "GeneratedOpenEnvTask",
    "GymReply",
    "HarborReply",
    "OpenEnvCheck",
    "OpenEnvReply",
    "OracleResult",
    "PlayRecord",
    "SmokeResult",
    "designer_messages",
    "designer_prompt",
    "environment_task",
    "harbor_task",
    "openenv_check",
    "openenv_task",
    "oracle_check",
    "parse_gym_reply",
    "parse_harbor_reply",
    "parse_openenv_reply",
    "smoke_test",
    "split_generation",
]
