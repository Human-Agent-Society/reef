State Model: Records, Commits, and Releases
=============================================

Records
-------

Reef stores every exchange of inference and every report as ``AgentRecord``.
It includes a record id, a scenario, an inference payload (for inference
exchange) or feedback (for report), and necessary metadata (e.g. request type
or artifact identifier used for serving).

``RecordStore`` defines record append, replay, compaction, and audit operations.
It does not know about trainers, artifact publication, or committed scenario
steps. ``SQLiteRecordStore`` and ``PostgresRecordStore`` share SQL record operations
while supplying their own database connections, schemas, and transactions.

Compaction retires records from training while retaining their bodies for audit.
It retires only rows the processor marks releasable, and Reef recomputes that
set from current state on every read. Separate retention maintenance physically
purges old compacted bodies while keeping retry hashes and commit metadata.

With a database path configured, the SQLite store uses WAL journalling and
synchronous = FULL. The default in-memory database is for tests and does not
survive a restart.

The release chain
-----------------

Every accepted update creates a release with a parent. Three identities stay
separate: ``release_id`` names Reef's publication decision, ``content_id`` names
the selected model or harness content, and ``runtime_load_id`` names a concrete
serving-engine weight load. A release may refer to durable bytes or to live
weights held only by the current process; its identity is stored durably either
way.

.. code:: mermaid

   flowchart TB
       accTitle: When releases become durable
       subgraph START["1. Durable start"]
           direction LR
           C0[("Checkpoint r0")] -->|"scenario starts"| S0["Serving r0"]
       end
       subgraph LIVE["2. Engine memory (restart restores r0)"]
           direction LR
           V1["Live release r1 / load l1"] -->|"step 2: train and sync"| V2["Live release r2 / load l2"]
       end
       subgraph NEXT["3. Next durable release"]
           direction LR
           C1[("Checkpoint r3")] -->|"continue serving"| S1["Serving r3"]
       end
       START -->|"step 1: train and sync"| LIVE
       LIVE -->|"step 3: export and publish"| NEXT
       class C0,C1 durable
       class V1,V2 volatile

Checkpoint cadence controls when live weights become durable, not how often they
change; any number of live steps may occur between checkpoints. A live release's
``runtime_load_id`` is an opaque ``<incarnation>:<sequence>`` token, where the
incarnation keeps tokens unique across training-group restarts. The release
record is durable; the bytes are not, so a restart restores the last checkpoint.
The step counter, algorithm state, and record progress do survive.

Durable releases are Git-backed, one ref per scenario, with LFS patterns for
weight files and a ``reef-artifact.json`` manifest in every release. Heads move
only by compare-and-swap: ``advance_current`` requires the expected head,
``publish`` requires the expected parent, and the push carries a lease, so a
stale publication conflicts instead of overwriting. Rollback does not rewrite
history. It activates an earlier release's ``content_id`` and publishes it under
a new ``release_id``, keeping step numbers monotonic.

Commit ordering
---------------

``ScenarioStore`` owns a scenario's committed state and its ``RecordStore``.
It validates step progression and settles record consumption with each commit.
Deployment assembly supplies ``ScenarioStorage``; ``ScenarioFactory`` handles
registration and release selection, then ``recover_scenario`` opens a session
and returns the complete ``Scenario``. The scenario owns that session and
coordinates trainer and artifact operations. The default ``SQLiteScenarioStorage`` assembles
``SQLiteRecordStore`` with ``CommitLogScenarioStore``. The commit log store accepts
any ``RecordStore`` and keeps commits in append-only JSONL; its fsynced append
remains the commit point. A committed step
records its step number, artifact ref, checkpoint flag, algorithm state, record
high-water mark, consumed record IDs, compaction retirements, metrics, and
training job identity. Existing databases and logs require no conversion.

``commit_step(expected_step=..., commit=...)`` validates the expected scenario
step before accepting its successor. Identical retries return the original
commit; different content for that step or a stale expected step conflicts.
The default adapter appends the commit before applying SQLite compaction.
These are separate writes: once the append is durable, a compaction failure
does not undo the commit. Retrying the same commit or running startup recovery
repairs the remaining compaction. A database adapter may settle both in one
database transaction while preserving this public contract.

``ScenarioCommitter`` keeps artifact operations outside the store. For a
durable store, the order is:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Commit kind
     - Order after trainer preparation
   * - Live weights
     - Prepare the live ref, commit the store, advance the serving head,
       apply trainer state.
   * - Saved checkpoint or rollback
     - Publish durable bytes, commit the store, install the committed heads,
       synchronize the backend pointer, apply trainer state.
   * - Local saved artifact
     - Stage bytes, commit the store, advance the process-local head,
       apply trainer state.
   * - No artifact
     - Commit the store, apply trainer state; keep the artifact head.
   * - Pending checkpoint
     - Publish durable bytes, commit the store, apply trainer state;
       leave serving and checkpoint heads in place.

Without a durable store, live and local saved releases advance the serving
head before settling the in-memory commit. A conflicting head therefore
rejects the step before its records are compacted. In-memory commit history
supports retries only for the lifetime of that session.

Artifact metadata stores scenario registration and checkpoint commit data
under ``scenario_commit_record``. Recovery decodes it into a ``CommitRecord`` and
reconciles it with the store before restoring the trainer. Initial registration
has no commit and passes ``None``. A committed checkpoint ahead of the backend pointer
repairs that pointer. An older checkpoint ahead of the local log can
be adopted into the log. Recovery repairs compaction, replays retained records
through the committed watermark while excluding every committed step's
consumed IDs, and restores the read cursor. The watermark is a read position:
retained records behind it can still belong to an incomplete batch.

Runtime work happens outside the transaction, so the training step must succeed
before its batch is acknowledged. After an ambiguous commit error, history is
authoritative: callers must reconcile or retry the same commit before
publishing another version. Each scenario still needs exactly one logical Reef
writer; expected-step validation does not provide distributed leases or make
external training operations atomic. Those operations must be idempotent or
reconcilable after a crash. See the `Python storage contract
<../reference/python-api.rst#scenario-stores>`__ for custom adapters.

These guarantees require persistent storage. `Configuration
<../reference/configuration.rst>`__ lists the paths, and `Operate a deployment
<../user-guide/operate.rst#restart-and-recovery>`__ tabulates what each one
survives.
