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
  ``checkpoint_strategy`` selects steps that publish durable checkpoints.
- ``releases`` queries releases, artifact content, and committed training metadata.
  It shares the committer's publication lock. Writers take the operation lock
  before the publication lock; readers never take the operation lock, so long
  training preparation does not block serving. ``history`` pages retained
  records and commits for the dispatcher.
- ``commits`` defines persisted values and encodes registration and checkpoint
  metadata without storage or training dependencies. Recovery reads a
  ``CommitRecord`` (none at initial registration). ``store`` defines storage
  contracts; concrete record and commit storage belongs to ``reef.storage``.

The aggregate never reaches back to its recipe. Application assembly supplies
its storage service; the dispatcher owns its lifecycle and each
scenario owns one opened session. Artifact publication and recovery ordering
remain in this package.
"""

from reef.scenario.binding import AcceptAnyArtifact, ArtifactValidator, ScenarioBinding
from reef.scenario.checkpoint_strategy import CheckpointStrategy, EveryNVersions
from reef.scenario.commits import SCENARIO_METADATA_KEY, CommitRecord
from reef.scenario.committer import ScenarioCommitter
from reef.scenario.registry import ScenarioRegistry
from reef.scenario.scenario import ReleaseNotRestorable, Scenario
from reef.scenario.store import ScenarioStorage, ScenarioStore, ScenarioStoreConflict

__all__ = [
    "SCENARIO_METADATA_KEY",
    "AcceptAnyArtifact",
    "ArtifactValidator",
    "CheckpointStrategy",
    "CommitRecord",
    "EveryNVersions",
    "ReleaseNotRestorable",
    "Scenario",
    "ScenarioBinding",
    "ScenarioCommitter",
    "ScenarioRegistry",
    "ScenarioStorage",
    "ScenarioStore",
    "ScenarioStoreConflict",
]
