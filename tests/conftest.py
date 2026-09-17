"""Suite-wide fixtures: one plain loss family beside the cookbook's.

Reef bundles no method-neutral policy-gradient objective, but several
contracts want the simplest possible family to drive the bridge, the runtime
and the driver: ``pg`` (Slime's stock ``policy_loss`` over one advantage per
sample). It is registered here as an *external* family through the same
public extension point a cookbook method would use, so every test that names
it sees a registry shaped like a deployment that brought its own plain
objective. The plain ``sft`` family and objective are the cookbook's
``recipes.sft`` package.
"""

from __future__ import annotations

import importlib
from argparse import Namespace

from reef.train.slime_backend.algorithm import SlimeAlgorithm
from reef.train.slime_backend.loss_families import register_loss_family

# The source suite exercises the repository cookbook as well as Reef core.
# Load those packages explicitly: production ``import reef`` deliberately does
# not, while importing a selected cookbook package registers its objective and
# lazy loss-family reference in this test process.
for _cookbook_package in (
    "reef.train.cordis_backend",
    "recipes.openclawrl",
    "recipes.sao",
    "recipes.sft",
    "recipes.tttd",
):
    importlib.import_module(_cookbook_package)

#: The plain family the suite registers alongside cookbook families.
TEST_FAMILIES = ("pg",)


class PgAlgorithm(SlimeAlgorithm):
    loss_family = "pg"
    loss_type = "policy_loss"
    requires_rollout_logprobs = True
    advantages = "required"

    def validate_specific_args(self, args: Namespace, source: str) -> None:
        pass


register_loss_family(PgAlgorithm())
