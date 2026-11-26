State Model: Records, Artifacts, and Operations
==============================================

Use these contracts when implementing storage, publishing artifacts or debugging
recovery. For deployment commands and backups, see `Operate a deployment
<../user-guide/operate.rst>`__ and `Storage <../user-guide/storage.rst>`__.

Records
-------

An ``AgentRecord`` holds a recorded inference exchange or report, its ID and
its metadata. ``RecordStore`` provides append, replay and audit operations.
It does not own training, scenario lifecycle or artifact publication.
``ItemRecordStore`` exposes one scenario's records from its owner's data items.

Each trainer tracks its own consumed IDs. Consumption does not delete records.
Retention removes old bodies but preserves commit metadata; retrying an evicted
record can store it again. Recovery cannot reconstruct deleted bodies.

Persistent SQLite scenario databases use WAL and ``synchronous = FULL``.
See the `record contract <../../reef/storage/scenario/records.py>`__ and
`retention settings <../user-guide/storage.rst#record-retention>`__.

The snapshot chain
------------------

An **artifact** is an immutable output of one component. Its public identity
is an opaque ``artifact_id``. Creation, training, import and harness sync can
create artifacts. Changing one component preserves the other components' IDs.

A **selection** maps component names to artifacts, such as
``{weights: A, harness: B}``. An **operation** records a state change or training
outcome. Selecting an existing artifact creates an operation, not another artifact.
Rejected and skipped training results also remain in operation history.

Serving and training starts are independent. Activation changes serving;
train-from changes the next default start. Harness sync changes both.
Imports change neither. See `artifact operations
<../user-guide/operate.rst#read-the-artifacts>`__.
Traffic weights live in coordination storage outside commit history.

The artifact repository has a publication chain separate from the active
selection. ``SnapshotRef.revision`` and ``parent_revision`` locate stored bytes;
they are not public artifact IDs. The built-in backend uses Git and Git LFS.
Publication checks its expected parent, and remote Git updates use a push lease.
A stale writer conflicts instead of overwriting another publication.

A backend must persist each artifact ID's binding with its snapshot.
The binding never changes, including after restart. Rebinding an ID to different
files or metadata fails. Retain bindings needed by committed operations.
See `RepositoryBackend <../../reef/artifact/repository.py>`__.

Commit ordering
---------------

``ScenarioStore`` owns committed scenario state. SQLite and PostgreSQL store
commits in ``scenario_commit``. The transaction that appends that row is the
commit point. It also updates sample training counts, so retries do not count
samples twice.

``CommitRecord`` holds the operation, selection and trainer progress.
See its `field and operation contracts
<../../reef/storage/scenario/commits.py>`__.

``append_commit(expected_commit_number=..., commit=...)`` must:

- Accept only the expected successor for the correct scenario.
- Return the stored record for an identical retry, ignoring only ``recorded_at``.
- Reject different content at that number or a stale expected number.

For a durable store, ``ScenarioCommitter`` commits a training artifact in this order:

#. Stage and publish durable artifact content and its bindings.
#. Append the commit record, including its activation decision.
#. Install the committed selections and synchronize the repository pointer.
#. Apply the trainer's prepared state.

Inactive artifacts advance publication without changing serving. Operations
without artifacts publish no bytes. Activation commits before inference loads.

A pointer synchronization failure after the commit does not undo that commit.
Status reports ``artifact_head_sync`` as ``pending`` or ``conflict``.
Later commits must synchronize first. After an ambiguous append result, read
history or retry the same commit before publishing another artifact.
See `ScenarioCommitter <../../reef/control/scenario/committer.py>`__.

**Recovery.** Reef reconciles commit history with checkpoint metadata and repairs
repository pointers from committed state. A checkpoint ahead of history can be
adopted into it. Unsupported metadata formats fail explicitly; see
`Storage upgrades <../user-guide/storage.rst#upgrade-from-a-reef-that-wrote-older-formats>`__.

Each trainer restores its state and consumed IDs from its own commits and
stored samples. It rereads retained records from the beginning and skips those
IDs. The stored high-water mark is not the recovery cursor: older record stores
could assign different sequences. Unconsumed records can belong to incomplete jobs.

Training runs outside the commit transaction. A job ID hashes its backend,
start, stable data identity and parameters. Retrying the same job must reuse
its result or reconcile its progress. Saved checkpoints support recovery;
unsaved optimizer work can run again.

For checkpoint units and intervals, see `checkpoint cadence
<../user-guide/operate.rst#set-the-checkpoint-cadence>`__.

One scenario's prepare-and-commit cycles are serialized. This does not make
remote training atomic or prevent concurrent jobs on a backend that supports them.
A custom backend must make external effects idempotent or recoverable.
The expected commit number does not replace the deployment's single active
control writer. Multiple trainers require durable history; in-memory history
cannot provide recovery after restart.
