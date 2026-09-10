"""CORAL test-time training through Reef: one method, one package.

- ``middleware`` — the ASGI layer that stamps Reef scenario/tag headers onto
  CORAL gateway traffic and captures Reef receipts.
- ``journal`` — the append-only call journal correlation reads.
- ``reporter`` — finalized CORAL attempts posted to ``/reef/report``.
- ``watcher`` — CORAL's on-disk attempt records, reported exactly once each.
- ``processor``/``recipe`` — sibling groups as grouped relative-reward
  training units, reusing the tttd preparer and loss family.
- ``bundle`` — the run's result bundle, derived from journal + reports.
- ``gateway_launcher`` — splices the middleware under CORAL's gateway (a
  ``GatewayManager`` the caller owns, or the one ``AgentManager`` builds).

The runnable example lives in ``examples/coral_demo``.
"""

from recipes.coral.journal import CallJournal, CallRecord
from recipes.coral.middleware import ReefGatewayMiddleware
from recipes.coral.recipe import CoralRecipe
from recipes.coral.reporter import AttemptReport, report_attempt
from recipes.coral.watcher import AttemptWatcher, FinalizedAttempt

__all__ = [
    "AttemptReport",
    "AttemptWatcher",
    "CallJournal",
    "CallRecord",
    "CoralRecipe",
    "FinalizedAttempt",
    "ReefGatewayMiddleware",
    "report_attempt",
]
