State Model: Records, Commits, and Releases
=============================================

Records
-------

Reef stores every exchange of inference and every report as ``AgentRecord``.
It includes a record id, a scenario, an inference payload (for inference
exchange) or feedback (for report), and necessary metadata (e.g. request type
or artifact identifier used for serving).

``RecordStore`` defines record append, replay, audit, and consumption receipt
operations. It does not know about scenario lifecycle, commits, artifact
publication, or training. ``SQLiteRecordStore`` and ``PostgresRecordStore``
share SQL record operations while supplying their own database connections,
schemas, and transactions.

Consumption does not delete a record: reads include every retained body,
whether a trainer consumed it or not. Each trainer keeps its own consumption
in its commits (see `Commit ordering`_). Retention evicts the oldest record
bodies only when the deployment-wide byte budget is full. It keeps the retry
hashes, so a retry with different content still conflicts, and it keeps the
commit metadata.

With a database path configured, the SQLite store uses WAL journalling and
synchronous = FULL. The default in-memory database is for tests and does not
survive a restart.

The release chain
-----------------

Every accepted update creates a release with a parent. Three identities stay
separate: ``release_id`` names Reef's publication decision, ``content_id`` names
the selected model or harness content, and ``runtime_load_id`` names a concrete
serving-engine weight load. Every release refers to durable bytes: Reef serves
only weights that a training job saved and that passed candidate evaluation.

The training component decides which checkpoints a job saves. A Slime or
Tinker job saves a checkpoint every ``save_interval`` optimizer steps when
the job's parameters set it, and one when the job ends. Each saved checkpoint becomes a candidate.
A selected candidate becomes a release. Reef commits a rejected candidate
without a release, and serving stays on the previous release. The next job
starts from the newest saved checkpoint whether or not it was selected:
evaluation decides only what is served. A restart resumes from the committed
state. The step counter, the weight trainer's state, and record progress
survive.

A release binds one or more named components. Every shipped recipe declares
one, so its releases are flat: the artifact is the component and its
``content_id`` is the component's. A recipe that declares several components
(``weights`` and ``harness``, say) serves releases that keep one directory
per component and carry a component manifest in their metadata. Each step
publishes one component and the committer carries the others forward from
the previous checkpoint, so the release still binds the whole combination:
its ``content_id`` derives from the component content ids, a rollback
restores every component together, and only a component whose content the
engine does not already serve is loaded or activated.

Durable releases are Git-backed, one ref per scenario, with LFS patterns for
weight files and a ``reef-artifact.json`` manifest in every release. Heads move
only by compare-and-swap: ``advance_current`` requires the expected head,
``publish`` requires the expected parent, and the push carries a lease, so a
stale publication conflicts instead of overwriting. Rollback does not rewrite
history. It activates an earlier release's ``content_id`` and publishes it under
a new ``release_id``, keeping step numbers monotonic.

Commit ordering
---------------

``ScenarioStore`` owns a scenario's committed state, its ``RecordStore`` and
its task stores. It validates step progression and settles record
consumption with each commit.
Deployment assembly supplies ``ScenarioStorage``; ``ScenarioFactory`` handles
registration and release selection, opens a storage session, and returns the
complete ``Scenario``. The scenario owns that session and
coordinates trainer and artifact operations. Both built-in storages
assemble their record store with ``CommitHistoryScenarioStore`` and keep
commits in a ``scenario_commit`` table of the record database.
``SQLiteScenarioStorage`` keeps one database file per scenario;
``PostgresScenarioStorage`` keeps every scenario in one schema. The committed
row is the commit point. A committed step records its step number, artifact
ref, checkpoint flag, algorithm state, record high-water mark, consumed record
IDs, metrics, training job ID, and, in a scenario of several trainers, the
component that made it.

``commit_step(expected_step=..., commit=...)`` validates the expected scenario
step before accepting its successor. Identical retries (ignoring only
``recorded_at``) return the original commit; different content for that step
or a stale expected step conflicts.

``ScenarioCommitter`` keeps artifact operations outside the store. For a
durable store, the order is:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Commit kind
     - Order after trainer preparation
   * - Selected checkpoint
     - Stage the bytes, load them into serving, publish the durable release,
       commit the store, install the committed heads, synchronize the backend
       pointer, apply trainer state.
   * - Rollback or promote
     - For a rollback to other weights, import the release into the training
       component. Load the release into serving if it holds other content,
       publish it as a new release, commit the store, install the committed
       heads, apply trainer state. After a rollback, the next job starts from
       the imported release.
   * - No artifact
     - Commit the store, apply trainer state; keep the artifact head. A job
       submission, a rejected checkpoint, and a job's end commit this way.
   * - Pending checkpoint
     - Publish durable bytes, commit the store, apply trainer state;
       leave serving and checkpoint heads in place.

Every trainer is bound to a release component (``records`` when the recipe
serves none). With several trainers, every commit record names the
``component`` that made it. In a scenario of one trainer, commit records,
checkpoint metadata and drop receipts name no component.

A scenario runs one training step at a time, whatever the number of its
trainers:

- Each trainer has a worker. A worker cycle holds the scenario's cycle lock
  from preparation to commit.
- A weight trainer commits in short cycles: one submits a job, one decides
  each saved checkpoint, and one records the job's end. While its job runs,
  the worker reads the job's status every 2 seconds. Each read is a short
  cycle, and the lock is free between them.
- A rollback or a promote refuses to run while any job data is reserved.

Thus the head does not move under a step, and every step commits on the
release that it was prepared against.

When a commit fails, the worker rebuilds the scenario from durable state at
once. A cycle that still runs on the old instance keeps that instance open
until it ends. Then it commits nothing and looks again on the rebuilt
instance, which has not read its rows. A cycle whose scenario is deleted
under it also ends without a commit.

A commit retires no row, so the trainers share every stored row and each
keeps its own consumption: a row one trainer consumed is still there for the
others. Job data that a trainer drops as stale goes on a consumption receipt
that names its component. On restart each trainer recovers its state and
read cursor from its own commits and skips the rows its own commits and
receipts name, which needs durable commit storage.

Without a durable store, a saved release advances the serving head before
the in-memory commit settles. A conflicting head therefore rejects the step
before its records are consumed. In-memory commit history supports retries
only for the lifetime of that session.

Artifact metadata stores scenario registration and checkpoint commit data
under ``scenario_commit_record``. Recovery decodes it into a ``CommitRecord`` and
reconciles it with the store before restoring the trainer. Initial registration
has no commit and passes ``None``. A committed checkpoint ahead of the backend pointer
repairs that pointer. An older checkpoint ahead of the commit history can
be adopted into the history. Recovery replays retained records through the
committed watermark while excluding every committed step's consumed IDs, and
restores the read cursor. The watermark is a read position:
retained records behind it can still belong to incomplete job data.

Training jobs run outside the transaction. A job's ID is a hash of the
scenario, the backend, the start, the records of the job's data, and the parameters.
A retry after a crash therefore submits the same job, and the component
ignores a job ID it already has. After an ambiguous commit error, history is
authoritative. Callers must reconcile or retry the same commit before
publishing another version. Each scenario still needs exactly one logical Reef
writer; expected-step validation does not provide distributed leases or make
external training operations atomic. Those operations must be idempotent or
reconcilable after a crash. See the `Python storage contract
<../reference/python-api.rst#scenario-stores>`__ for custom adapters.

These guarantees require persistent storage. `Configuration
<../reference/configuration.rst>`__ lists the paths, and `Operate a deployment
<../user-guide/operate.rst#restart-and-recovery>`__ tabulates what each one
survives.
