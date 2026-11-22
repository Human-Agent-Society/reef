State Model: Records, Artifacts, and Operations
==============================================

This page is for readers who write a storage or artifact backend, or who must
reason about recovery. It states what Reef stores for a scenario and the order
in which a commit becomes durable. Operators find the tasks in `Operate a
deployment <../user-guide/operate.rst>`__.

Records
-------

Reef stores every inference exchange and every report as an ``AgentRecord``:
a record ID, a scenario, the inference payload or the feedback, and metadata
such as the request type and the artifact that served the request.

``RecordStore`` defines record append, replay and audit. It knows nothing about
scenario lifecycle, commits, artifact publication or training. A scenario's
records are record items of its owner in the data store; ``ItemRecordStore``
gives the records of one scenario over that store.

- **Consumption** does not delete a record. Reads include every retained
  record, and each trainer keeps its own consumption in its commits (see
  `Commit ordering`_).
- **Retention** deletes the oldest record items only when the deployment-wide
  byte budget is full (``DataStore.prune_records``). It keeps the commit
  metadata. A retry of a deleted record stores it again.
- **Durability.** With a database path, the SQLite scenario database uses WAL
  journalling and ``synchronous = FULL``. The default in-memory database is for
  tests and does not survive a restart.

The snapshot chain
------------------

The public state model has three parts: immutable artifacts, scenario
selections and an append-only operation history. The repository also keeps a
publication chain internally.

An *artifact* is one output of one component. Its ``artifact_id`` is the
only public artifact identity. Training, import and harness sync create
artifacts. Scenario creation creates the starting artifacts from the
configuration or an artifact item. A component keeps its ID while another
component changes. Callers treat IDs as opaque strings; repository revisions
and content checksums are private storage details.

A *selection* maps every component name to one artifact. For example,
``{weights: A, harness: B}`` selects weights A and harness B. Activating
harness C records ``{weights: A, harness: C}`` atomically. The operation
creates no artifact or repository revision. A runtime load ID identifies an
engine load internally; inference records identify the answering artifact
with ``artifact_id``.

Serving and training use separate selections:

- The active selection supplies the default routes. An ``activate``
  operation replaces one selected artifact and leaves every trainer's start
  unchanged. Inference uses the selection that completed startup; a failed
  load remains visible in startup status until control retries it.
- Each trainer has a training start: the artifact its next job or evolve
  step starts from. It begins at the creation artifact and moves to its own
  newest output after training. ``train-from`` selects an existing artifact
  and leaves serving unchanged. A request's ``train_from`` gives one job its
  own start: that job does not move the training start, so it can run beside
  the other jobs of the scenario.
- A ``serve`` operation adds an artifact beside the active selection. It
  records the selected component mapping without publishing bytes. Later
  activation does not change that served mapping. A ``stop_serving``
  operation removes the extra route.
- The ``served`` setting of ``POST /reef/scenarios/{scenario}/update`` sets
  what a scenario serves of one component: a mapping from artifact IDs to
  weights. The largest weight names the active artifact, and each other
  artifact serves beside it. Reef appends the ``activate``, ``serve`` and
  ``stop_serving`` operations that lead to the mapping. With more than one
  weight above 0, the weights are the traffic split, which the coordination
  store holds outside the commit history.
- ``import`` creates an artifact from external files and records its source.
  It changes neither serving nor the trainers' starts.

The activation rule decides whether a new training artifact serves.
A Slime or Tinker job saves checkpoints at ``save_interval`` optimizer steps,
when configured, and when the job ends. Each checkpoint becomes an artifact.
The next job starts from the newest checkpoint whether or not it was
activated. A harness sync creates an artifact and sets both its serving
selection and training start in one operation.

An *operation* records a state transition or training outcome at a scenario
step. Activation, serving and training-start changes do not appear as new
artifacts in the artifact catalog. Their operations preserve history and
provide the complete selection needed for recovery. Rejected or skipped
training steps remain operations even when they create no artifact.

The built-in repository stores durable bytes in Git, with LFS patterns for
weight files and a ``reef-artifact.json`` manifest. Backend interfaces use
``SnapshotRef`` with ``revision`` and ``parent_revision``. These fields locate
stored bytes; component IDs remain independent of storage. The publication head
(the chain tip) orders new snapshots independently of the active selection.
Publication checks its expected parent and uses a Git push lease, so a stale
writer conflicts instead of overwriting another publication.

Old commit records remain readable, with their stored locations. A new commit
record stores only its component artifact selection. Recovery reads each
location from the repository's binding of an artifact ID. A binding never
changes and Reef never removes one, so a committed record always resolves. A
retry of a failed commit publishes nothing new: one artifact ID names one
stored snapshot.

Commit ordering
---------------

``ScenarioStore`` owns a scenario's committed state, its ``RecordStore`` and
its task stores. It validates the order of commit numbers and settles record
consumption with each commit. Deployment assembly supplies a
``ScenarioStorage``. ``ScenarioFactory`` writes a new scenario's root snapshot,
opens a storage session and returns the complete ``Scenario``, which owns that
session.

Both built-in storages keep commits in a ``scenario_commit`` table of the
record database (``CommitHistoryScenarioStore``). ``SQLiteScenarioStorage``
keeps one database file per scenario; ``PostgresScenarioStorage`` keeps every
scenario in one schema. The committed row is the commit point. A commit
records:

- its commit number, operation and component selection;
- the private artifact locator and the checkpoint flag;
- the algorithm state and the record high-water mark;
- the consumed IDs: record IDs, and ``sample:<sample id>`` for each sample
  that its job trained;
- metrics and the training job ID;
- in a scenario of several trainers, the component that made it.

``append_commit(expected_commit_number=..., commit=...)`` validates the
expected commit number before it accepts the successor. An identical retry
(ignoring only ``recorded_at``) returns the original commit. Different content
for that commit number, or a stale expected commit number, conflicts.

``ScenarioCommitter`` keeps artifact operations outside the store. For a
durable store, the order is:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Commit kind
     - Order after trainer preparation
   * - Activated artifact
     - Stage the bytes, publish the durable snapshot,
       commit the store with ``activated: true``, install the committed heads,
       synchronize the backend pointer, apply trainer state.
   * - Activation
     - Check the selected artifact. Inference loads it after the selection
       commits. Append an ``activate`` operation with the complete component
       mapping.
       Publish no artifact snapshot. Recovery restores the committed mapping,
       including every unchanged component.
   * - Train-from
     - The trainer takes the artifact as its start and cancels its running
       job: a weight trainer names the artifact's snapshot as its start, which the
       training service imports with ``import_start`` for the next job, and
       a harness trainer reads the entries of its tree. Commit a ``train_from``
       record. A restart before the trainer's next training commit takes the
       start again from that record.
   * - No artifact
     - Commit the store, apply trainer state; keep the active selection and the
       chain tip. A job submission, a rejected harness candidate, and a job's
       end commit this way.
   * - Inactive artifact
     - Publish durable bytes, commit the store with ``activated: false``,
       move the chain tip, apply trainer state; keep the active selection. An
       artifact that the activation rule does not activate commits this way.
       A later activation is its own record.

Every trainer is bound to a snapshot component (``records`` when the recipe
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
snapshot that it was prepared against.

When a commit fails, the worker rebuilds the scenario from durable state at
once. A cycle that still runs on the old instance keeps that instance open
until it ends. Then it commits nothing and looks again on the rebuilt
instance, which has not read its rows. A cycle whose scenario is deleted
under it also ends without a commit.

A commit retires no row, so the trainers share every stored row and each
keeps its own consumption: a row one trainer consumed is still there for the
others. On restart each trainer recovers its state and
read cursor from its own commits and skips the rows that they name, which needs
durable commit storage.

Without a durable store, a saved snapshot advances the active selection and
the chain tip before the in-memory commit settles. A conflicting chain tip
therefore rejects the step before its records are consumed. In-memory commit
history supports retries only for the lifetime of that session.

Artifact metadata stores scenario registration and checkpoint commit data
under ``scenario_commit_record`` (format ``reef-scenario/8``): the root's
``source``, and after commit number zero the root snapshot as ``start_artifact``.
Recovery refuses other formats. It decodes the metadata into a ``CommitRecord`` and
reconciles it with the store before restoring the trainer. Initial registration
has no commit and passes ``None``. A committed checkpoint ahead of the backend pointer
repairs that pointer. An older checkpoint ahead of the commit history can
be adopted into the history. Recovery replays retained records through the
committed watermark while excluding every commit's consumed IDs and
the records that the trainer's stages consumed (its training state), and
restores the read cursor. The watermark is a read position:
retained records behind it can still belong to incomplete job data.

Training jobs run outside the transaction. A job's ID is a hash of the
scenario, the backend, the start, the records of the job's data, and the parameters.
A retry after a crash therefore submits the same job, and the component
ignores a job ID it already has. After an ambiguous commit error, history is
authoritative: callers must reconcile or retry the same commit before they
publish another version.

Each scenario still needs exactly one logical Reef writer. The check of the
expected commit number provides no distributed lease and does not make external
training operations atomic. Those operations must be idempotent or
reconcilable after a crash. See the `Python storage contract
<../reference/python-api-scenario-stores.rst#scenario-stores>`__ for custom adapters.

These guarantees require persistent storage. `Configuration
<../reference/configuration.rst>`__ lists the paths, and `Operate a deployment
<../user-guide/operate.rst#restart-and-recovery>`__ tabulates what each one
survives.
