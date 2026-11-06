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

import functools
import importlib
import socketserver
from argparse import Namespace

import pytest

from reef.dispatcher import Dispatcher
from reef.train.slime_backend.algorithm import SlimeAlgorithm
from reef.train.slime_backend.loss_families import register_loss_family

# The source suite exercises the repository cookbook as well as Reef core.
# Load those packages explicitly: production ``import reef`` deliberately does
# not, while importing a selected cookbook package registers its objective and
# lazy loss-family reference in this test process. The built-in SFT and
# dataset recipes are loaded the same way: ``import reef`` does not import them either.
for _cookbook_package in (
    "reef.recipe.sft",
    "reef.recipe.dataset",
    "reef.recipe.generated_tasks",
    "reef.train.harness_backend",
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
    advantages = "required"

    def validate_specific_args(self, args: Namespace, source: str) -> None:
        pass


register_loss_family(PgAlgorithm())


@pytest.fixture(scope="session", autouse=True)
def quick_server_shutdown():
    """Make every in-process server check for shutdown every 10 ms instead of every 0.5 s.

    ``socketserver`` servers poll for ``shutdown()`` at ``poll_interval``, so each
    stopped server (a capture proxy, a fake Reef, a serve socket) otherwise
    blocks its test for up to 0.5 s. Child processes keep the default."""
    with pytest.MonkeyPatch.context() as patch:
        quick = functools.partialmethod(socketserver.BaseServer.serve_forever, poll_interval=0.01)
        patch.setattr(socketserver.BaseServer, "serve_forever", quick)
        yield


#: The dispatchers that the running test built, in the order it built them.
BUILT_DISPATCHERS: list[Dispatcher] = []


@pytest.fixture(scope="session", autouse=True)
def record_built_dispatchers():
    """Record each ``Dispatcher`` that a test builds, so ``close_open_dispatchers`` can close it."""
    original = Dispatcher.__init__

    @functools.wraps(original)
    def record(self: Dispatcher, *args, **kwargs) -> None:
        original(self, *args, **kwargs)
        BUILT_DISPATCHERS.append(self)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Dispatcher, "__init__", record)
        yield


@pytest.fixture(autouse=True)
def close_open_dispatchers():
    """Close each dispatcher that the test built and left open.

    An open dispatcher keeps its service threads, which poll until the process ends. One xdist worker runs
    hundreds of tests, so the threads of unclosed dispatchers pile up and starve the timed waits of later tests.
    ``Dispatcher.close`` does nothing for a dispatcher that is closed already.
    """
    BUILT_DISPATCHERS.clear()
    yield
    while BUILT_DISPATCHERS:
        BUILT_DISPATCHERS.pop().close()
