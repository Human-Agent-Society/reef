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

A scenario's first release holds the *creation artifact* of each component.
Every later release has a parent. A release binds one *artifact*, the content
of one component, for each component. Four identities stay separate: ``release_id`` names the release, ``artifact_id``
names an artifact, ``content_id`` names the model or harness content, and
``runtime_load_id`` names a concrete serving-engine weight load. In a
scenario of one component, ``artifact_id`` is the ``release_id`` of the
release that made the artifact; in a scenario of several components, it is
``<release_id>:<component>``. A release that carries a component forward
keeps that component's ``artifact_id``. Every release refers to durable
bytes.

Serving and training each read artifacts of the chain:

- The *active release* serves. It binds the active artifact of each
  component. ``activate`` makes an existing artifact active and leaves every
  trainer's start alone.
- Each trainer has a *training start*, the artifact that its next job or
  evolve step starts from. The first training start is the creation artifact.
  After each job or step, the trainer moves its start to its own newest
  output. ``train-from`` sets the start to an existing artifact and leaves the
  active release alone.

The training component decides which checkpoints a job saves. A Slime or
Tinker job saves a checkpoint every ``save_interval`` optimizer steps when
the job's parameters set it, and one when the job ends. Each saved checkpoint
becomes an artifact. Reef activates a checkpoint when the deployment's
activation rule says so. Any other checkpoint stays an inactive artifact,
and the active release keeps serving. The next job starts from the newest
saved checkpoint whether or not it was activated: the rule decides only what
serves. A
restart resumes from the committed state. The step counter, the weight
trainer's state, and record progress survive.

A release binds one or more named components. Every shipped recipe declares
one, so its releases are flat: the release holds one artifact, and its
``content_id`` is the artifact's. A recipe that declares several components
(``weights`` and ``harness``, say) serves releases that keep one directory
per component and carry a component manifest in their metadata. Each step
publishes one component and the committer carries the others forward from
the chain tip, so the release still binds the whole combination:
its ``content_id`` derives from the component content ids, and only a
component whose content the engine does not already serve is loaded. An
activation in such a scenario publishes a new release: the active release
with the activated component bound to the artifact. It makes no new content
and does not move the chain tip.

Durable releases are Git-backed, one ref per scenario, with LFS patterns for
weight files and a ``reef-artifact.json`` manifest in every release. The
repository keeps two heads: the active release, and the *chain tip*. The chain
tip is the newest release on the chain, and the parent of the next release
that a training step publishes. A training step that makes an artifact and
activates it moves both heads to its release. The release of an inactive
artifact, and the release that an activation publishes in a scenario of
several components, are published beside the chain: they move neither the
chain tip nor the backend head. Heads move only by compare-and-swap:
``activate`` requires the expected active release, ``publish`` requires the
expected parent, and the push carries a lease, so a stale publication
conflicts instead of overwriting. In a scenario of one component, activation
writes no artifact or release: it appends an ``activate`` record and moves
only the active release. The commit history records every
activation, and recovery reads the active release back from the history: it
is the release of the newest activation. A training step that makes an
artifact and activates it counts as an activation.

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
   * - Activated artifact
     - Stage the bytes, load them into serving, publish the durable release,
       commit the store with ``activated: true``, install the committed heads,
       synchronize the backend pointer, apply trainer state.
   * - Activation
     - Check the artifact as at its publication; in a scenario of several
       components, publish the release that binds it. Load the release into
       serving, commit an ``activate`` record, make the release active. A
       crash before the record leaves serving on the release that the history
       names.
   * - Train-from
     - The trainer takes the artifact as its start and cancels its running
       job: a weight trainer imports the artifact with ``import_start``, and
       a harness trainer reads the entries of its tree. Commit a ``train_from``
       record. A restart before the trainer's next training commit takes the
       start again from that record.
   * - No artifact
     - Commit the store, apply trainer state; keep the active release and the
       chain tip. A job submission, a rejected harness candidate, and a job's
       end commit this way.
   * - Inactive artifact
     - Publish durable bytes, commit the store with ``activated: false``,
       move the chain tip, apply trainer state; keep the active release. An
       artifact that the activation rule does not activate commits this way.
       A later activation is its own record.

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
- An activation or a train-from refuses to run while a training result
  waits for its commit.

Thus the chain tip does not move under a step, and every step commits on the
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

Without a durable store, a saved release advances the active release and the
chain tip before the in-memory commit settles. A conflicting chain tip
therefore rejects the step before its records are consumed. In-memory commit history supports retries
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
