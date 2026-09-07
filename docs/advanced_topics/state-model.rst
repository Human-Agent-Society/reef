State Model: Records, Commits, and Releases
=============================================

Records
-------

Reef stores every exchange of inference and every report as ``AgentRecord``.
It includes a record id, a scenario, an inference payload (for inference
exchange) or feedback (for report), and necessary metadata (e.g. request type
or artifact identifier used for serving).

Compaction is the only operation that deletes records. It deletes only rows
the processor marks releasable, and Reef recomputes that set from current state
on every read.

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

Each scenario has an append-only JSONL commit log, and the fsynced append is the
commit point. A committed step records its step number, artifact ref, checkpoint
flag, algorithm state, record high-water mark, compaction deletions, and
metrics. Every other store is derived from that log, and the ordering around the
append is fixed per step kind, so a crash in any gap replays cleanly.

Runtime work happens outside the transaction, so the training step must succeed
before its batch is acknowledged. A publish that fails discards the staged
artifact and reloads from durable state, replaying the uncompacted records. Each
scenario needs exactly one logical Reef writer, and external training operations
must be idempotent or reconcilable after a crash.

These guarantees require persistent storage. `Configuration
<../reference/configuration.rst>`__ lists the paths, and `Operate a deployment
<../user-guide/operate.rst#restart-and-recovery>`__ tabulates what each one
survives.
