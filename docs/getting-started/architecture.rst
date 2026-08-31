Architecture
============

An agent is a model plus its harness. Reef sits between the harness and the
runtime that executes the model. It records the release that served
each response, accepts feedback that refers to those responses, and lets a
scenario's recipe use eligible records to produce the next release.

Which package holds which code is `Codebase structure
<../contributing/codebase-structure.rst>`__.

The core loop
-------------

.. code:: mermaid

   sequenceDiagram
       accTitle: How Reef serves, records, trains, and publishes
       autonumber
       participant H as Harness
       participant S as Scenario
       participant I as Inference
       participant T as Trainer
       participant G as Training*

       opt Harness recipe: pull the served tree
         H->>S: GET /reef/harness for scenario
         S-->>H: Harness tree and release
         Note over H: Agent runs on that tree
       end
       Note over H,I: Serve and record each request
       H->>S: Inference request for scenario
       S->>S: Freeze current release
       S->>I: Provider-native request
       I-->>S: Provider response
       S->>S: Validate frozen release and store record
       S-->>H: Response and receipt
       H->>S: Feedback quotes the receipt
       S->>T: Eligible record
       opt Processor has a batch
         Note over S,G: Update and commit
         T->>G: Prepared step
         G-->>T: Trained artifact ready
         T->>S: Commit new release
       end

The inference runtime is always required. The training runtime exists only for
recipes that change weights; recipes that change text run their step in Reef's
own process, with no GPU.

The opening pull is for recipes whose artifact is the harness tree: the agent
fetches the currently served tree (``GET /reef/harness``, or the install script
built on it) and runs on that release, so the harness it uses is the one whose
receipts it will later report against. Weight recipes skip this pull because
their artifact lives in the inference runtime. Requests reach the new release
there directly.

The request path
----------------

Before calling the model, Reef reads the scenario's current artifact ref and
builds the request against that release. The stored exchange uses the same ref,
so an update completing mid-request does not change what the receipt records.

Reef validates a response before recording it. ``prepare_request`` transforms
the outgoing payload, and Reef forwards *and records* the transformed payload.
``verify_response`` checks the provider's answer against the frozen release. On
failure Reef records nothing and returns the error.

For live weights, Reef asks the engine to report the ``runtime_load_id`` for each
generated token span. A response may cover several runtime loads if an update lands
mid-generation; Reef accepts it only when the span information accounts for
every generated token and is consistent with the frozen release. Missing or
inconsistent spans are a backend contract error and return HTTP 409.

Pass-through streaming cannot do that check. Reef leaves ``return_meta_info``
disabled when ``stream`` is true and records a plain SSE exchange. The training
backend buffers its stream instead and validates the complete response against
the frozen release before recording.

Records
-------

Every inference and every report is an ``AgentRecord``: an id, a scenario, a
request type, a payload, and the frozen ``artifact_ref`` for inference. The
record id is the receipt.

Appending the same id with different content returns a conflict rather than
overwriting. A separate table tracks the ids each batch consumed, so retried
reports and late reports whose references already trained are not counted twice.

Compaction is the only operation that deletes records. It deletes only rows the
processor marks releasable, and Reef recomputes that set from current state on
every read.

With a database path configured, the SQLite store uses WAL journalling and
``synchronous = FULL``. The default in-memory database is for tests and does not
survive a restart.

Scenarios
---------

A scenario isolates the records, trainer, and release chain for one workload.
The first request creates it, names its recipe, and may pin a starting release.
Those bindings never change; a request naming a different recipe returns HTTP
409. The surface, runtime, inference backend, and optional report schema chosen
when the recipe is constructed are fixed with it.

A Reef process runs at most one scenario that trains full weights, on a single
thread, so preparation, remote execution, and commit never interleave. It may
run any number of scenarios that produce no updates or that update text
artifacts in process. Each one grows and commits on its own background thread,
so record acceptance never waits for artifact evolution.

Surfaces
--------

A surface delivers a published artifact to whoever uses it.

+---------------+---------------------------------------------------------------+
| Surface       | Delivery                                                      |
+===============+===============================================================+
| record-only   | no loader, inference hooks, or file tree                      |
+---------------+---------------------------------------------------------------+
| weights       | pushed into the serving engine by the training runtime; the   |
|               | surface owns runtime-load fencing policy                      |
+---------------+---------------------------------------------------------------+
| harness files | an adapter-specific tree, pulled by the client                |
+---------------+---------------------------------------------------------------+
| skill files   | a layered skill tree, pulled by the client or injected        |
|               | server-side into each request                                 |
+---------------+---------------------------------------------------------------+

Every scenario binds the same ``Surface`` type; its *fields*, not its type
identity, advertise what it supports, and ``None`` means the capability is
absent. Artifact admission is a separate binding the recipe selects: the commit
path validates candidates and rollback sources before they enter the chain.

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

Durability
----------

Each scenario has an append-only JSONL commit log, and the fsynced append is the
commit point. A committed step records its step number, artifact ref, checkpoint
flag, algorithm state, record high-water mark, compaction deletions, and
metrics. Every other store is derived from that log, and the ordering around the
append is fixed per step kind, so a crash in any gap replays cleanly.

+----------------------+-----------------------------------------------+
| State                | Guarantee                                     |
+======================+===============================================+
| Records              | persisted before the processor sees them      |
+----------------------+-----------------------------------------------+
| Algorithm state      | restored from the log's head record;          |
|                      | checkpoint snapshot metadata is the fallback  |
+----------------------+-----------------------------------------------+
| Record progress      | resumes at the head record's high-water mark; |
|                      | consumed rows are never re-trained            |
+----------------------+-----------------------------------------------+
| Pending runtime work | not durably recoverable                       |
+----------------------+-----------------------------------------------+

Runtime work happens outside the transaction, so the training step must succeed
before its batch is acknowledged. A publish that fails discards the staged
artifact and reloads from durable state, replaying the uncompacted records. Each
scenario needs exactly one logical Reef writer, and external training operations
must be idempotent or reconcilable after a crash.

These guarantees require persistent storage. See `Configuration
<../reference/configuration.rst>`__ for the paths.

Evaluation
----------

Evaluation is part of the trainer workflow and gates whether a candidate is accepted.
Whether a candidate is *accepted* is the recipe's decision, made through its
``candidate_evaluation`` plugin.

A deployment names its plugin in the ``evaluation`` section of its config
(`Configuration <../reference/configuration.rst#the-evaluation-section>`__);
`Write a recipe <../developer-guide/write-a-recipe.rst#gate-a-candidate>`__
walks through building one.
