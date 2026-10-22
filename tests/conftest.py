"""Suite-wide fixtures: the built-in SFT recipe, the cookbook, and one plain loss family.

Many contracts use the simplest families to drive the bridge, the runtime,
and the driver:

- ``sft`` is the family of the built-in SFT recipe (``reef.recipe.sft``).
  It uses Slime's ``sft_loss`` and refuses advantages.
- ``pg`` uses Slime's ``policy_loss`` with one advantage for each sample.
  This module registers it as an *external* family, through the public
  extension point that a cookbook method uses.
"""

from __future__ import annotations

import importlib
from argparse import Namespace

from reef.train.slime_backend.algorithm import SlimeAlgorithm
from reef.train.slime_backend.loss_families import register_loss_family

# The source suite exercises the repository cookbook as well as Reef core.
# Load those packages explicitly: production ``import reef`` deliberately does
# not, while importing a selected cookbook package registers its objective and
# lazy loss-family reference in this test process. The built-in SFT recipe
# is loaded the same way: ``import reef`` does not import it either.
for _cookbook_package in (
    "reef.recipe.sft",
    "reef.train.cordis_backend",
    "recipes.openclawrl",
    "recipes.opd",
    "recipes.sao",
    "recipes.sdft",
    "recipes.sdpo",
    "recipes.tttd",
):
    importlib.import_module(_cookbook_package)


class PgAlgorithm(SlimeAlgorithm):
    loss_family = "pg"
    loss_type = "policy_loss"
    requires_rollout_logprobs = True
    advantages = "required"

    def validate_specific_args(self, args: Namespace, source: str) -> None:
        pass


register_loss_family(PgAlgorithm())
