"""Scenario state, commit, and recovery contracts.

A scenario is one durable training aggregate. Its lifecycle, and the module
responsible for each piece:

- **create** — ``factory`` forks a base artifact and persists a registration
  snapshot (format in ``snapshot``); ``binding`` freezes the recipe-selected
  admission, surface, runtime, and inference backend.
- **train** — ``scenario`` exposes the trainer through lock-guarded methods
  so every mutating path serializes against commit and rollback.
- **commit** — ``commit_protocol`` orders one atomic step: commit trainer
  state, append the record to ``commit_log`` (the durable commit point),
  replay compaction, move the artifact head; ``checkpoint_strategy`` decides
  when a step also publishes a durable checkpoint carrying fresh snapshot
  metadata.
- **recover** — ``factory`` reads the snapshot, heals and replays the commit
  log (``ScenarioCommitProtocol.recover_head``), and resumes trainer and
  record consumption at the committed high-water mark.
- **rollback** — ``commit_protocol`` republishes an older checkpointed
  version as a new fenced commit; history is never rewritten.

The scenario domain does not import recipes: a recipe configures a scenario
through the factory, and the aggregate never reaches back. Every
step-advancing commit appends exactly one atomic record, and recovery
re-derives every other store from the log. Checkpoint cadence is the one
extension point (subclass ``CheckpointStrategy``); the commit ordering itself
is not pluggable, and that rigidity is the point.
"""

from reef.scenario.binding import AcceptAnyArtifact, ArtifactValidator, ScenarioBinding
from reef.scenario.checkpoint_strategy import CheckpointStrategy, EveryNVersions
from reef.scenario.commit_log import CommitLog, CommitRecord
from reef.scenario.commit_protocol import ScenarioCommitProtocol
from reef.scenario.registry import ScenarioRegistry
from reef.scenario.scenario import ReleaseNotRestorable, Scenario
from reef.scenario.snapshot import SCENARIO_SNAPSHOT_METADATA_KEY

__all__ = [
    "SCENARIO_SNAPSHOT_METADATA_KEY",
    "AcceptAnyArtifact",
    "ArtifactValidator",
    "CheckpointStrategy",
    "CommitLog",
    "CommitRecord",
    "EveryNVersions",
    "ReleaseNotRestorable",
    "Scenario",
    "ScenarioBinding",
    "ScenarioCommitProtocol",
    "ScenarioRegistry",
]
