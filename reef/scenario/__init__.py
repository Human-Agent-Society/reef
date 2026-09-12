"""Scenario state, commit, and recovery contracts.

A scenario is one durable training aggregate. Its lifecycle, and the module
responsible for each piece:

- **values** — ``state`` defines persisted commits, snapshots, and record
  progress without importing storage, artifact operations, or training.
- **create** — ``factory`` forks a base artifact and persists a registration
  snapshot through the ``snapshot`` metadata adapter; ``binding`` freezes the deployment-selected
  admission, surface, runtime, and inference backend.
- **train** — ``scenario`` exposes the trainer through lock-guarded methods
  so every mutating path serializes against commit and rollback.
- **commit** — ``commit_protocol`` prepares trainer state and orders artifact
  publication around ``store`` settlement. The store validates the expected
  step, records the commit, and applies record compaction. Only then does the
  protocol expose trainer state. ``checkpoint_strategy`` decides when a step
  publishes a durable checkpoint carrying fresh snapshot metadata.
- **recover** — ``factory`` supplies the registration snapshot to ``store``,
  which reconciles committed history and repairs interrupted compaction. The
  factory restores artifact delivery and trainer memory from that result.
- **rollback** — ``commit_protocol`` republishes an older checkpointed
  version as a new fenced commit; history is never rewritten.
- **read** — ``commit_protocol`` serializes catalog and artifact snapshots
  with publication and rollback, but not with long-running step preparation.
  Writers acquire the operation lock before the publication lock; readers
  take only the publication lock and must not acquire the operation lock.

The scenario aggregate does not retain recipe identity: the deployment's
recipe configures its runtime binding through the factory, and the aggregate
never reaches back. The application supplies a ``ScenarioStoreFactory`` and
gives each scenario ownership of one session. Storage implementations are
selected outside this package. Artifact publication and recovery ordering remain owned here;
database connections, schemas, and journal files belong to ``reef.storage``.
"""

from reef.scenario.binding import AcceptAnyArtifact, ArtifactValidator, ScenarioBinding
from reef.scenario.checkpoint_strategy import CheckpointStrategy, EveryNVersions
from reef.scenario.commit_protocol import ScenarioCommitProtocol
from reef.scenario.registry import ScenarioRegistry
from reef.scenario.scenario import ReleaseNotRestorable, Scenario
from reef.scenario.snapshot import SCENARIO_SNAPSHOT_METADATA_KEY
from reef.scenario.state import CommitRecord
from reef.scenario.store import ScenarioStore, ScenarioStoreConflict, ScenarioStoreFactory

__all__ = [
    "SCENARIO_SNAPSHOT_METADATA_KEY",
    "AcceptAnyArtifact",
    "ArtifactValidator",
    "CheckpointStrategy",
    "CommitRecord",
    "EveryNVersions",
    "ReleaseNotRestorable",
    "Scenario",
    "ScenarioBinding",
    "ScenarioCommitProtocol",
    "ScenarioRegistry",
    "ScenarioStore",
    "ScenarioStoreConflict",
    "ScenarioStoreFactory",
]
