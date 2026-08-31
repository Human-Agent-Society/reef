"""Suite-wide fixtures: two plain loss families and a plain step preparer.

Reef bundles no method-neutral objective, but many contracts want the
simplest possible family to drive the bridge, the runtime and the driver:
``sft`` (Slime's stock ``sft_loss``, advantages forbidden) and ``pg``
(Slime's stock ``policy_loss`` over one advantage per sample). They are
registered here as *external* families and the matching ``sft`` preparer as
an external preparer, through the same public extension points a cookbook
method would use, so every test that names them sees a registry shaped like
a deployment that brought its own plain objective.
"""

from __future__ import annotations

import importlib
from argparse import Namespace
from collections.abc import Mapping
from typing import Any

from reef.train.algos.base import StepPreparer
from reef.train.algos.helpers import next_steps
from reef.train.algos.registry import register_preparer
from reef.train.algos.signals import StepSignal
from reef.train.slime_backend.algorithm import SlimeAlgorithm
from reef.train.slime_backend.loss_families import register_loss_family
from reef.train.types import PolicyBatch, TrainingBatch

# The source suite exercises the repository cookbook as well as Reef core.
# Load those packages explicitly: production ``import reef`` deliberately does
# not, while importing a selected cookbook package registers its preparer and
# lazy loss-family reference in this test process.
for _cookbook_package in (
    "recipes.harness_evolve",
    "recipes.openclawrl",
    "recipes.sao",
    "recipes.tttd",
):
    importlib.import_module(_cookbook_package)

#: The plain families the suite registers alongside cookbook families.
TEST_FAMILIES = ("pg", "sft")


class SftAlgorithm(SlimeAlgorithm):
    loss_family = "sft"
    loss_type = "sft_loss"
    advantages = "forbidden"
    forbidden_advantages_message = (
        "sft ignores advantages: Slime's sft_loss_function trains every token unweighted, so the "
        "payload's advantages would be silently discarded. For a reward-weighted objective use the "
        "pg loss family (one advantage per sample), or register a custom loss family "
        "(register_loss_family) whose loss consumes them."
    )

    def validate_specific_args(self, args: Namespace, source: str) -> None:
        pass


class PgAlgorithm(SlimeAlgorithm):
    loss_family = "pg"
    loss_type = "policy_loss"
    requires_rollout_logprobs = True
    advantages = "required"

    def validate_specific_args(self, args: Namespace, source: str) -> None:
        pass


class SftPreparer(StepPreparer):
    """Train every sample in the batch, unweighted."""

    name = "sft"

    def __call__(self, batch: TrainingBatch, state: Mapping[str, Any]) -> StepSignal:
        if not isinstance(batch, PolicyBatch):
            raise TypeError(f"{self.name} requires PolicyBatch, got {type(batch).__name__}")
        steps = next_steps(state)
        return StepSignal("train", self.name, {"steps": steps}, {"samples": len(batch.samples), "steps": steps})


register_loss_family(SftAlgorithm())
register_loss_family(PgAlgorithm())
register_preparer(SftPreparer())
