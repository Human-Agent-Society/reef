"""Scenario state, commit, and recovery contracts.

A scenario owns one runtime binding, trainer, artifact chain, and store session.
The modules follow these responsibilities:

- ``scenario`` exposes operations on one instance; ``binding`` freezes its
  deployment-selected admission, surface, runtime, and inference backend.
- ``registry`` owns loaded instances, per-scenario locks, model updates, and
  scenario archival coordination. It caches concrete ``runtime.model_config.ModelConfig``
  instances and calls ``storage.model_config`` functions for private JSON files.
  Recipes and the factory receive only the current scenario's configuration.
- ``factory`` registers the base artifact, validates release selectors,
  opens storage, reconciles committed state, synchronizes and
  activates the checkpoint, builds the trainer, and replays retained records.
  It returns a complete ``Scenario`` and closes owned resources on failure.
- ``committer`` orders commit, rollback, retries, and artifact publication
  around store settlement. Trainer state is exposed only after settlement.
  ``recipe.checkpoint_strategy`` selects steps that publish durable checkpoints.
- ``releases`` queries releases, artifact content, and committed training metadata.
  It shares the committer's publication lock. Writers take the operation lock
  before the publication lock; readers never take the operation lock, so long
  training preparation does not block serving. ``history`` pages retained
  records and commits for the dispatcher.
Persisted values and metadata encoding belong to ``reef.storage.commits``.
Recovery reads a ``CommitRecord`` (none at initial registration). Storage
contracts and implementations never import this package.

The aggregate never reaches back to its recipe; the factory and registry
compose recipes at construction. Application assembly supplies
its storage service; the dispatcher owns its lifecycle and each
scenario owns one opened session. Artifact publication and recovery ordering
remain in this package.
"""

from reef.scenario.binding import ScenarioBinding
from reef.scenario.committer import ScenarioCommitter
from reef.scenario.registry import ScenarioRegistry
from reef.scenario.scenario import ReleaseNotRestorable, Scenario

__all__ = [
    "ReleaseNotRestorable",
    "Scenario",
    "ScenarioBinding",
    "ScenarioCommitter",
    "ScenarioRegistry",
]
