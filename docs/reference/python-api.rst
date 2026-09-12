Python API for custom learning methods
======================================

The Python API is the set of extension points a learning method plugs into. Reef
owns everything around them: accepting and replaying records, holding a batch
until it is acknowledged, committing algorithm state, running the backend, and
publishing the next version.

`Write a recipe <../developer-guide/write-a-recipe.rst>`__ is the executable tutorial.

What to implement
-----------------

Start with Recipe and add only what the method actually needs.

+---------------------------------------------+--------------------------------------------------+------------------------------+
| Need                                        | Component                                        | Required for                 |
+=============================================+==================================================+==============================+
| configure deployment behavior               | `Recipe <#recipe>`__                             | every deployment             |
+---------------------------------------------+--------------------------------------------------+------------------------------+
| validate feedback at ingress                | `Report <#report>`__                             | methods whose signal arrives |
|                                             |                                                  | in reports                   |
+---------------------------------------------+--------------------------------------------------+------------------------------+
| turn records into a typed batch             | `Processor <#processor>`__                       | every method producing       |
|                                             |                                                  | updates                      |
+---------------------------------------------+--------------------------------------------------+------------------------------+
| carry data to the backend                   | `Batch <#batch>`__                               | every method producing       |
|                                             |                                                  | updates; subclass only for a |
|                                             |                                                  | new shape                    |
+---------------------------------------------+--------------------------------------------------+------------------------------+
| turn a batch into a signal                  | `Step preparer <#step-preparer>`__               | weight-training methods      |
+---------------------------------------------+--------------------------------------------------+------------------------------+
| deliver a new artifact medium               | `Surface <#surface>`__                           | only a new evolution medium  |
+---------------------------------------------+--------------------------------------------------+------------------------------+
| propose a harness edit and grade an episode | `Harness method <#harness-method>`__             | harness-evolution methods    |
+---------------------------------------------+--------------------------------------------------+------------------------------+
| gate a produced candidate                   | `Candidate evaluation <#candidate-evaluation>`__ | optional, any recipe         |
+---------------------------------------------+--------------------------------------------------+------------------------------+
| a new tensor objective                      | `Loss family                                     | rarely                       |
|                                             | <../developer-guide/loss-families.rst>`__        |                              |
+---------------------------------------------+--------------------------------------------------+------------------------------+

Method code should depend only on what this page documents. Anything else under
``reef.`` is an implementation detail and may change.

Recipe
------

.. code:: python

   from reef.recipe import Recipe, WeightTrainingRecipe, config_field
   from reef.train.cordis_backend import CordisRecipe

A recipe is one frozen dataclass configuring the serving and evolution behavior
for every scenario in a deployment.

.. code:: text

   Recipe                       record-only by default   reef.recipe
   ├── WeightTrainingRecipe     step preparer, loss family, TrainingRuntime
   │   ├── SAORecipe                                        recipes.sao.recipe
   │   ├── TTTDRecipe                                       recipes.tttd.recipe
   │   └── OpenClawRLRecipe                                 recipes.openclawrl.recipe
   └── CordisRecipe             harness tree + episodes  reef.train.cordis_backend
       └── SkillClawRecipe                                recipes.skillclaw.recipe

Choose the narrowest class whose assumptions all hold. Inheriting ``Recipe``
starts without the extra contracts of a specialized base; it does not force the
recipe to remain record-only.

+---------------------------------+------------------------------------------------------+
| What you are building           | Where to start                                       |
+=================================+======================================================+
| record traffic, no updates      | ``Recipe`` as-is                                     |
+---------------------------------+------------------------------------------------------+
| train and publish weights       | subclass ``WeightTrainingRecipe``                    |
+---------------------------------+------------------------------------------------------+
| use Reef's harness loop         | configure ``CordisRecipe``; subclass it only         |
|                                 | for a named preset or extra validation               |
+---------------------------------+------------------------------------------------------+
| evolve a different artifact     | subclass ``Recipe``, override ``build()`` and        |
|                                 | ``build_surface()``                                  |
+---------------------------------+------------------------------------------------------+
| serve an externally produced    | subclass ``Recipe``, override ``build_surface()``    |
| artifact                        | only                                                 |
+---------------------------------+------------------------------------------------------+

Common members
~~~~~~~~~~~~~~

+---------------------------------------------------+-----------------------------+--------------------------------+
| Member                                            | Type                        | Contract                       |
+===================================================+=============================+================================+
| ``name``                                          | ``str``                     | instance field; the default    |
|                                                   |                             | registry key                   |
+---------------------------------------------------+-----------------------------+--------------------------------+
| ``runtime``                                       | ``InferenceRuntime | None`` | narrowed to a required         |
|                                                   |                             | ``TrainingRuntime`` by         |
|                                                   |                             | ``WeightTrainingRecipe``       |
+---------------------------------------------------+-----------------------------+--------------------------------+
| ``checkpoint_strategy``                           | ``CheckpointStrategy``      | defaults to                    |
|                                                   |                             | ``EveryNVersions(1)``          |
+---------------------------------------------------+-----------------------------+--------------------------------+
| ``build(scenario, records, algorithm_state=...)`` | ``Trainer``                 | construct the scenario trainer |
+---------------------------------------------------+-----------------------------+--------------------------------+
| ``build_surface(scenario)``                       | ``Surface``                 | the delivery contract for one  |
|                                                   |                             | named scenario                 |
+---------------------------------------------------+-----------------------------+--------------------------------+
| ``build_artifact_validator()``                    | ``ArtifactValidator``       | artifact admission, enforced   |
|                                                   |                             | before publication and         |
|                                                   |                             | rollback; defaults to          |
|                                                   |                             | ``AcceptAnyArtifact()``        |
+---------------------------------------------------+-----------------------------+--------------------------------+
| ``serving_status()``                              | ``Mapping | None``          | runtime-wide state for         |
|                                                   |                             | ``/reef/status``               |
+---------------------------------------------------+-----------------------------+--------------------------------+

Every recipe may declare ``report_type``, the ``ReportBase`` subclass its
reports parse as (``None`` keeps ingress open). Weight-training recipes add
``training_spec()``, which binds the processor, the registered or dotted step
preparer, and the backend loss family; ``max_staleness``, the accepted
producing-to-serving version lag, which must match the runtime; and
``candidate_evaluation``, the optional plugin configured by the deployment's
``evaluation`` section.

.. code:: python

   @dataclass(frozen=True, kw_only=True)
   class MyMethodRecipe(WeightTrainingRecipe):
       name: str = "my_method"

       @property
       def report_type(self) -> type[MyMethodReport]:
           return MyMethodReport

       @classmethod
       def training_spec(cls) -> WeightTrainingSpec:
           return WeightTrainingSpec(
               processor=MyMethodProcessor,
               step_preparer="my_method.prepare:prepare_step",
               loss_family="my_method",
           )

``frozen=True`` is required by the base. ``kw_only=True`` keeps later fields
keyword-only while the training runtime stays the positional dependency. Call
``super().__post_init__()`` first when adding validation.

Configuration
~~~~~~~~~~~~~

``config_field()`` is a ``dataclasses.field`` carrying a default, a type-aware
parser, and an optional environment fallback:

.. code:: python

   batch_size: int = config_field(4, env="REEF_MY_METHOD_BATCH_SIZE")

Precedence is explicit configuration, then environment, then the default.
``from_environment()`` builds the recipe. On weight-training recipes,
``service_config()`` forwards declared fields and shared artifact settings
from the service configuration; override ``processor_config()`` when a
processor needs renamed or derived keys. Never read deployment YAML from
inside a processor.

Report
------

.. code:: python

   from reef.core.reports import ReportBase, ReportValidationError, ScoredRolloutReport

Method-specific report contracts live in their method package; Reef does not
import or re-export them.

A report type declares the feedback a method accepts, so malformed input fails
at ingress with HTTP 400.

.. code:: python

   @dataclass(frozen=True)
   class MyMethodReport(ReportBase):
       score: float
       task_id: str
       rubric: str = ""

       def validate(self) -> None:
           if not self.task_id.strip():
               raise ReportValidationError("metadata.task_id must be non-empty")

Declare it through the recipe's ``report_type``. ``score`` uses the top-level
``score`` channel; every other field uses ``metadata.<field>``. Producers may
attach extra fields the schema does not declare.

+-----------------------+---------------------+------------------------------------+
| Annotation            | Accepted JSON       | Validation                         |
+=======================+=====================+====================================+
| ``float``             | number              | must be finite; ints normalize     |
+-----------------------+---------------------+------------------------------------+
| ``int``               | number              | integral value; booleans rejected  |
+-----------------------+---------------------+------------------------------------+
| ``str``               | string              | no coercion                        |
+-----------------------+---------------------+------------------------------------+
| ``bool``              | boolean             | no numeric substitutes             |
+-----------------------+---------------------+------------------------------------+
| ``Mapping[str, str]`` | object              | every key and value a string       |
+-----------------------+---------------------+------------------------------------+

Each supported type may also be written as ``T | None``. Any other annotation,
any other Union included, is a declaration error and raises ``TypeError`` when
Reef first inspects the type. A field without a default is required; a field
with one may be absent; JSON ``null`` is accepted only when the annotation
permits it.

The same type serves both sides: a producer constructs it and calls
``to_dict()``; a processor receives the parsed instance as
``context.parsed_report``.

Scenario stores
---------------

``RecordStore`` in ``reef.records`` is the abstract base for appending, reading,
and retiring records. ``ScenarioStore`` composes a ``RecordStore`` with committed
scenario state, settling each step and its record progress together. Recipes
and trainers use the record interface; ``Scenario`` coordinates training and
artifact publication through its supplied scenario store.

Storage implementations explicitly subclass ``RecordStore``,
``ScenarioStore``, and ``ScenarioStoreFactory`` and override their abstract
methods and properties. Incomplete subclasses cannot be instantiated. The
bundled ``SQLiteRecordStore``, ``CommitLogScenarioStore``, and
``SQLiteScenarioStoreFactory`` inherit those bases, respectively.

.. code:: python

   from pathlib import Path

   from reef.dispatcher import Dispatcher
   from reef.storage.factory import SQLiteScenarioStoreFactory

   store_factory = SQLiteScenarioStoreFactory(Path(".reef/agent-record"))
   dispatcher = Dispatcher(
       recipe,
       repository_backend_factory,
       agent_record_dir=Path(".reef/agent-record"),
       scenario_store_factory=store_factory,
   )

``Dispatcher`` requires ``scenario_store_factory`` and passes it to scenario
creation. It never selects or imports a record storage implementation. The
``build_default_dispatcher`` convenience helper also requires this argument.
Deployment assembly chooses SQLite or PostgreSQL from ``reef.record_backend``;
embedded callers supply their chosen factory explicitly. For ephemeral records
and commits, pass ``SQLiteScenarioStoreFactory()`` without a directory. Passing
a directory preserves SQLite schemas, JSONL format, and scenario filenames.

With an explicit store factory, ``agent_record_dir`` independently selects the
local directory for scenario model settings. Configure it to retain those
settings across restarts even when records and commits use a remote adapter.

Direct ``Scenario`` construction requires ``store=...``. Replace the former
``records=...`` and ``commit_log=...`` arguments with one store opened by a
``ScenarioStoreFactory``. The supplied trainer must use that store's ``records``.
``Scenario`` does not choose a storage backend. Read committed history through
``scenario.store.history()`` and persistence capability through
``scenario.store.durable``; the former ``scenario.commit_log`` property has been
removed. The concrete commit log store's ``commit_log`` is for backend diagnostics.

The abstract bases ``ScenarioStore`` and ``ScenarioStoreFactory`` are exported from
``reef.scenario``, alongside ``ScenarioStoreConflict``. Concrete storage classes
live under ``reef.storage``: ``commit_log.CommitLogScenarioStore`` accepts a
``RecordStore`` and a JSONL ``CommitLog``; ``factory.SQLiteScenarioStoreFactory``
assembles it with ``sqlite.SQLiteRecordStore``. The scenario package depends only
on storage contracts. Direct commit log callers now import ``CommitLog`` from
``reef.storage.commit_log``; ``reef.scenario.commit_log`` is removed. The session
contract is:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Member
     - Contract
   * - ``records``
     - A ``RecordStore`` implementation preserving append deduplication,
       ordered replay, scenario isolation, audit reads, and compaction receipts.
   * - ``durable``
     - Whether committed history survives session/process restart. A durable
       store requires a repository backend supporting staged releases.
   * - ``history()``
     - Committed ``CommitRecord`` values in step order, including rollback and
       pending releases. This is authoritative after an ambiguous commit error.
   * - ``training_run_position()``
     - The last rollback step and the count of training commits after it, for
       experiment run numbering.
   * - ``commit_step(expected_step=..., commit=...)``
     - Settle the ``CommitRecord`` and its record compaction, returning the
       canonical accepted record. Its step must equal ``expected_step + 1``.
   * - ``recover(snapshot=..., checkpoint_head=...)``
     - Reconcile checkpoint metadata with history, repair interrupted record
       compaction, and return the head ``CommitRecord`` or ``None`` for a fresh
       scenario.
   * - ``close()``
     - Release the session's record and commit resources; repeated calls are safe.

``CommitRecord`` carries the artifact ref, algorithm state, record watermark,
consumed and compacted IDs, checkpoint/pending flags, operation and rollback
target, metrics, and training job identity. The store must atomically validate
the current step before accepting a new successor: two different commits
prepared from the same step cannot both succeed. An identical recorded retry
returns the original record, even after later steps have committed; only
``recorded_at`` is excluded from retry comparison. A new commit with a stale
expected step, or a retry with conflicting content, raises
``ScenarioStoreConflict`` without advancing the store.

The commit log adapter serializes its writers with a local POSIX file lock;
direct writes through ``CommitLog`` bypass this store contract.
The default adapter fsyncs the JSONL record before applying SQLite compaction.
It therefore provides recoverable settlement across two files, rather than a
single SQL transaction. A failure after the append may leave the step committed
while compaction still needs repair. Retry the exact commit or recover the
session; do not infer rollback from an exception. Recovery reapplies recorded
compaction and uses all committed ``consumed_ids`` to keep retained audit rows
out of training. A future database adapter can commit the step and record
progress together in one database transaction.

Artifact bytes and backend head movement remain outside ``ScenarioStore``.
``ScenarioCommitProtocol`` owns their order around store settlement, including
pending releases, rollback, and checkpoint reconciliation. See `Commit ordering
<../advanced_topics/state-model.rst#commit-ordering>`__. Step validation does not
provide distributed leases or authorize multiple active training writers.

The factory exposes ``durable``, ``open(scenario)``, ``archive(scenario)``,
``prune(days=..., max_bytes=...)``, and ``close()``. Archive retires that scenario's
stored state so its name can be reused; prune applies retained-body limits to
active and archived state. Custom backends implement those operations instead
of exposing local file paths. Each opened session belongs to its scenario and
closes after its trainer, including on construction or recovery failure. The
dispatcher closes its factory on shutdown. Application assembly should use
the scenario store factory; direct record store construction is described below.

Record storage and audit
------------------------

``reef.records`` defines ``RecordStore``, record result and error types, and
``RecordRetention``. It has no dependency on scenario coordination, training,
or concrete storage adapters. Both ``reef.RecordStore`` and
``reef.records.RecordStore`` refer to this abstract base.

``reef.storage.sql_records.SQLRecordStore`` implements the shared SQLAlchemy
Core record operations. ``reef.storage.sqlite.SQLiteRecordStore`` inherits
those operations and supplies SQLite connection setup, schema migration,
transactions, and dialect behavior. When opening older SQLite schemas, the
adapter uses Alembic's operations API to add missing columns on its connection.
Retention query logic is shared in the SQL module; the SQLite adapter owns
database-file discovery and maintenance connections.

``reef.storage.postgres.PostgresRecordStore`` inherits those same operations.
It supplies PostgreSQL schema types, pooled transactions, and conflict insertion
through psycopg 3. Install ``reef-infra[postgres]`` to enable the driver. The
``RecordTables.scope`` mapping isolates named stores in shared SQL tables;
SQLite leaves it empty because each store owns a database.

.. code-block:: python

   import os
   from contextlib import closing
   from reef.storage.postgres import PostgresRecordStore

   with closing(PostgresRecordStore(
       os.environ["REEF_RECORD_DATABASE_URL"],
       schema="reef_records",
       name="my-record-store",
   )) as records:
       rows = records.replay("my-scenario")

The URL constructor owns its pool. To share a pool across sessions, construct
``PostgresRecordDatabase(url, schema=...)`` and pass that object instead of a URL;
close sessions before closing the database. Both close operations are idempotent.
``PostgresScenarioStoreFactory(url, directory, schema=...)`` combines PostgreSQL
records with ``CommitLogScenarioStore`` and local JSONL logs. Pass it through
``Dispatcher(..., scenario_store_factory=...)`` for embedded use. Scenario code
continues to depend only on the abstract store contracts. Deployment configuration
selects this factory with ``reef.record_backend: postgres``; see
`operation settings <../user-guide/operate.rst>`__ for connection and storage setup.

Writes lock their store generation before allocating append sequences and hold
the lock through commit. Reads use a consistent transaction snapshot. PostgreSQL
receipt keys hash large compacted id sets, with complete canonical content checked
by the shared SQL layer. PostgreSQL timestamps use double precision, and sequences
use 64-bit identities. Retention applies the same age and byte-budget policy to
compacted bodies across active and archived generations in the deployment schema.

Existing SQLite databases, record encodings, and record methods remain
compatible; no database conversion is required. Direct callers must replace
``RecordStore(path)`` with ``SQLiteRecordStore(path)``; the abstract base cannot
be instantiated. ``SQLiteRecordStore`` is also exported from ``reef``:

.. code:: python

   from pathlib import Path

   from reef.storage.sqlite import SQLiteRecordStore

   with SQLiteRecordStore(Path(".reef/records.sqlite3")) as records:
       retained = records.audit_page("math")

``RecordRetention`` in ``reef.records`` holds and validates the
``days`` and ``max_bytes`` limits. Store factories apply those limits through
their ``prune`` method.

``reef.records.RecordStore`` separates the training record set from retained
trace history. ``compact(scenario, ids)`` sets ``compacted_at`` and keeps the
original payload, response, references, and artifact reference. Hash tombstones
and optional compaction receipts are committed atomically with that transition.
Repeated compaction preserves the first timestamp.

``get``, ``replay``, ``replay_page``, and ``count`` expose only records whose
``compacted_at`` is ``None``. Training and restart recovery continue to use
those methods. Use these explicit methods for audit and retention work:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Method
     - Result
   * - ``get_for_audit(scenario, agent_record_id)``
     - A ``StoredRecord``, or ``None`` if no body is retained in that scenario.
   * - ``audit_page(scenario, after_sequence=0, limit=256)``
     - A bounded tuple of ``StoredRecord`` entries, in append order, including
       compacted bodies. Advance the cursor using the last entry's ``sequence``.
   * - ``purge_compacted(scenario, before=timestamp, limit=256)``
     - The number of bodies physically deleted, at most ``limit``. Only records
       with ``compacted_at < before`` are eligible. The cutoff must be a finite
       Unix timestamp and the limit a positive integer.

``StoredRecord`` contains ``sequence``, ``item`` (the original ``AgentRecord``),
and ``compacted_at`` (a Unix timestamp or ``None``). Audit reads never restore a
record to the training set. A missing body may have been purged or never stored;
the read API does not guess which. Compaction includes terminal or excluded
records as well as trained records. Use the commit log's per-step
``consumed_ids`` to determine learning participation.

For example, inspect one trace without making it available to training again:

.. code:: python

   entry = scenario.records.get_for_audit(scenario.name, record_id)
   if entry is not None:
       payload = entry.item.payload
       references = entry.item.references
       retired_at = entry.compacted_at

With the default SQLite factory, the HTTP service runs background retention at
startup and every 60 seconds.
It removes bodies older than 7 days, then the oldest remaining bodies to meet
a shared 20 GiB budget across scenario databases in ``agent_record_dir``,
including ``archived/``. The budget measures UTF-8 JSON payloads, references,
and artifact references. Limits are configurable in `Configuration <configuration.rst>`__.

For embedded Python deployments, use
``dispatcher.prune_record_archives(RecordRetention(days=7, max_bytes=20 * 1024**3))``
with ``RecordRetention`` imported from ``reef.records``. This runs one sweep
through the configured store factory and serializes it with scenario archival.
Standalone ``SQLiteRecordStore`` and ``Dispatcher`` construction do not start a
maintenance task. ``create_app``
accepts ``record_retention=RecordRetention(...)`` to enable service maintenance.

The former ``RecordRetention.prune(directory)`` method has been removed: the
retention value no longer opens databases or accepts a filesystem path. Use
``dispatcher.prune_record_archives(retention)`` for a running dispatcher. For
standalone SQLite maintenance, pass the directory to the factory:

.. code:: python

   from pathlib import Path

   from reef.records import RecordRetention
   from reef.storage.factory import SQLiteScenarioStoreFactory

   retention = RecordRetention(days=7, max_bytes=20 * 1024**3)
   store_factory = SQLiteScenarioStoreFactory(Path(".reef/agent-record"))
   try:
       purged = store_factory.prune(days=retention.days, max_bytes=retention.max_bytes)
   finally:
       store_factory.close()

The caller must serialize standalone maintenance with any scenario file moves.

Retention preserves active records, retry hashes, and compaction receipts.
An identical retry after purge still deduplicates, and conflicting content
still fails. Deletes commit in batches of 256. Concurrent compaction can exceed
the budget until the next sweep. SQLite may reuse freed pages, but purging does
not shrink the database file; active records, indexes, and other metadata also
use disk space. HTTP audit routes remain a separate integration. See
`Configuration <configuration.rst>`__ for migration and rollback constraints.

Processor
---------

.. code:: python

   from reef.train.processors import ComputedFeedbackProcessor, ReportedFeedbackProcessor

A processor turns durable records into typed batches. Reef owns replay,
retention, deduplication, pending batches, and exactly-once consumption; the
method implements only the hooks below. They run synchronously on the trainer
thread, so they must not block on network or model latency.

+-------------------+-------------------------------+-------------------------------+
|                   | ``ReportedFeedbackProcessor`` | ``ComputedFeedbackProcessor`` |
+===================+===============================+===============================+
| signal arrives as | a report referencing          | information reconstructed     |
|                   | inference records             | from recorded traffic         |
+-------------------+-------------------------------+-------------------------------+
| judgment          | synchronous, on available     | ``async`` model or service    |
|                   | data                          | call                          |
+-------------------+-------------------------------+-------------------------------+
| method owns       | eligibility, optional         | correlation, slow judgment,   |
|                   | grouping, batch shaping       | sample and batch shaping      |
+-------------------+-------------------------------+-------------------------------+
| Reef owns         | waiting index, retry dedup,   | worker lifecycle, queues,     |
|                   | grouping state, retention,    | retention, replay             |
|                   | replay                        |                               |
+-------------------+-------------------------------+-------------------------------+

``DataProcessor.training_mode`` selects automatic, instruction-triggered or
combined batching on the same processor. Declare ``supported_training_modes`` and
implement ``make_training_batch(batch_number, request)`` to select inputs;
``request`` is the queued instruction in ``manual`` and ``hybrid`` and ``None``
for an automatic batch. Ingestion, acknowledgement, retention,
compaction and background derivation are shared. See
`Processors <../developer-guide/processors.rst>`__ for the instruction queue and batch contract.

Every processor gets the scenario's experiment logger as
``self.experiment_logger``. Log finite numeric metrics under the ``processor``
namespace; processor code never imports W&B, and the logger is a no-op when
tracking is off.

Reported feedback
~~~~~~~~~~~~~~~~~

.. code:: mermaid

   sequenceDiagram
       accTitle: How reported feedback becomes a typed batch
       participant Reef as Reported-feedback processor
       participant Method as Method processor
       Reef->>Method: judge(ReportContext)
       alt referenced inference is missing
           Method-->>Reef: WAIT: park this report
       else report can never train
           Method-->>Reef: NEVER: release this report
       else report is accepted
           Method-->>Reef: ReportDecision.train(value, ...)
           opt group_key supplied
               Reef->>Method: decide_group(key, candidates)
               Method-->>Reef: GroupDecision<br/>INCOMPLETE / READY / DISCARD
           end
           Reef->>Method: make_batch(ready units, batch_number)
       end

+------------------------------------------------------+----------------------------------+
| Hook                                                 | Contract                         |
+======================================================+==================================+
| ``judge(context) -> ReportDecision``                 | return ``TRAIN``, ``WAIT``, or   |
|                                                      | ``NEVER``                        |
+------------------------------------------------------+----------------------------------+
| ``make_batch(units, batch_number) -> TrainingBatch`` | shape accepted candidates        |
+------------------------------------------------------+----------------------------------+
| ``decide_group(key, candidates) -> GroupDecision``   | required only when ``judge()``   |
|                                                      | supplies a ``group_key``         |
+------------------------------------------------------+----------------------------------+
| ``output_schema``                                    | the exact batch class returned   |
+------------------------------------------------------+----------------------------------+
| ``exclusive_sources``                                | when true, a terminal report     |
|                                                      | owns and releases its sources    |
+------------------------------------------------------+----------------------------------+
| ``ordered_groups``                                   | when true, ready groups batch by |
|                                                      | sortable group key               |
+------------------------------------------------------+----------------------------------+

Begin a score-based ``judge()`` with ``context.eligibility()``: it returns
``WAIT`` while a referenced inference is missing, ``NEVER`` for permanently
ineligible input, and ``None`` when the method should decide. Use
``ReportDecision.never(reason)`` for a rejection an operator should be able to
diagnose. The processor logs each new reason and counts repeats.

``ReportDecision.train(value)`` creates a singleton candidate. Supply
``group_key`` and an idempotent ``slot`` when the training unit is a complete
comparison group; ``decide_group()`` then returns ``READY``, ``INCOMPLETE``, or
``DISCARD``. Report-level ``WAIT`` and ``GroupDecision.INCOMPLETE`` differ: the
first waits for a report's missing references, the second holds accepted
candidates until their group fills.

Computed feedback
~~~~~~~~~~~~~~~~~

Use this engine when later traffic completes an earlier record and judging it
calls a model or another slow service.

+---------------------------------------+-----------------------------------+
| Hook                                  | Contract                          |
+=======================================+===================================+
| ``ingest(record)``                    | correlate records; must not block |
+---------------------------------------+-----------------------------------+
| ``async judge(job)``                  | slow judgment, on the processor's |
|                                       | own worker                        |
+---------------------------------------+-----------------------------------+
| ``make_sample(record, judgment)``     | a ``PolicySample``, or ``None``   |
|                                       | to retire the record              |
+---------------------------------------+-----------------------------------+
| ``make_batch(samples, batch_number)`` | the declared batch type           |
+---------------------------------------+-----------------------------------+
| ``expire(now)``                       | optionally return receipts whose  |
|                                       | completion window ended           |
+---------------------------------------+-----------------------------------+
| ``required_request_types``            | optionally restrict which record  |
|                                       | types reach the processor         |
+---------------------------------------+-----------------------------------+

Every ``ingest()`` starts with ``catch_up(now)``, then uses ``track(record)``,
``tracked_record(receipt)``, ``dispatch(job)``, ``retire(receipt)``, and
``abandon(receipt)``. A failed judgment or a ``None`` sample retires the record
instead of failing a training step. All correlation state must be
reconstructible from replay.

Batch
-----

.. code:: python

   from reef.train.types import (
       GroupedPolicyBatch, PolicyBatch, PolicySample,
       TraceBatch, TraceSample, TrainingBatch,
   )

A batch is a frozen dataclass with a stable ``batch_id``. A processor returns
the same pending batch until that id is acknowledged, so batch content must not
depend on mutable external state.

.. code:: text

   TrainingBatch
   ├── PolicyBatch          samples: tuple[PolicySample, ...]
   ├── GroupedPolicyBatch   comparison_sets: tuple[tuple[PolicySample, ...], ...]
   └── TraceBatch           samples: tuple[TraceSample, ...]

+------------------------+-------------------------------------------------+
| Type                   | Typical use                                     |
+========================+=================================================+
| ``PolicyBatch``        | singleton rollout and session-derived weight    |
|                        | updates                                         |
+------------------------+-------------------------------------------------+
| ``GroupedPolicyBatch`` | group-relative objectives                       |
+------------------------+-------------------------------------------------+
| ``TraceBatch``         | local harness-artifact evolution                |
+------------------------+-------------------------------------------------+

A ``PolicySample`` carries ``source_agent_record_id``, ``tokens``,
``loss_mask``, ``rollout_log_probs``, and ``reward``. When available, it also carries
``runtime_load_id``, ``action_mask``, ``rollout_created_at``, ``turn_count``,
``topk_indices`` / ``topk_log_probs``, ``runtime_load_spans``, and ``extras``,
the field a processor uses for its own loss family. Use the shared assembly
helpers in ``reef.train.processors.reported`` and
``reef.train.processors.common`` rather than re-parsing provider responses.

A ``TraceSample`` is not tokenized: it carries ``source_agent_record_id``, the
recorded ``payload`` unchanged, and the resolved ``score``.

Subclass ``TrainingBatch`` only when the processor and backend genuinely share a
different data contract; set ``output_schema`` to the new class and keep the
batch serializable, with no handles to services, files, threads, or models.

Step preparer
-------------

.. code:: python

   from reef.train.algos import StepSignal

A preparer turns a reserved batch into a pure, backend-neutral signal. Normally
it is a plain function named by the recipe as ``package.module:callable``, and
its module must be importable in both the service and the training process.

.. code:: python

   def prepare_step(batch: TrainingBatch, state: Mapping[str, Any]) -> StepSignal:
       if not isinstance(batch, PolicyBatch):
           raise TypeError(f"my_method requires PolicyBatch, got {type(batch).__name__}")
       steps = next_steps(state)
       return StepSignal(
           action="train",
           loss_family="my_method",
           advantages=tuple(sample.reward for sample in batch.samples),
           next_algorithm_state={"steps": steps},
           metrics={"steps": steps},
       )

+--------------------------+-------------------------------------------------+
| Field                    | Contract                                        |
+==========================+=================================================+
| ``action``               | ``train`` runs a backend step; ``skip`` commits |
|                          | a state-only transition                         |
+--------------------------+-------------------------------------------------+
| ``loss_family``          | the backend objective for this step             |
+--------------------------+-------------------------------------------------+
| ``advantages``           | optional per-sample values, in batch order      |
+--------------------------+-------------------------------------------------+
| ``next_algorithm_state`` | committed only after the step succeeds          |
+--------------------------+-------------------------------------------------+
| ``metrics``              | method telemetry carried to the commit record   |
+--------------------------+-------------------------------------------------+
| ``scheduling``           | how a runtime materializes a grouped batch:     |
|                          | ``unit`` is ``comparison_set`` or ``sample``,   |
|                          | ``batch_size`` is ``configured``, ``actual``,   |
|                          | or a positive int                               |
+--------------------------+-------------------------------------------------+

A preparer owns method math and nothing else. It must not import a runtime, Ray,
torch, or Slime; execute a training job; read deployment configuration; mutate
the reserved batch; or commit state outside ``next_algorithm_state``. The
recipe's loss family, the signal's loss family, the driver environment, and the
backend flags must agree. Reef rejects a mismatch at startup or during step
preparation.

Use a ``StepPreparer`` subclass with ``@register_step_preparer`` only when
several recipes need a stable shared name.

Harness method
--------------

A harness-evolution method fills three slots. Reef runs the loop around them:
snapshot, apply, run the paired episodes, record, publish or revert.

.. code:: python

   def propose(nodes, samples, models) -> Mutation | Sequence[Mutation] | None: ...
   def evaluate(task, result) -> float: ...
   class Selection:  # optional
       def decide(self, candidate, evaluation) -> SelectionDecision: ...

+------------------------+----------------------------------------------------------+
| Member                 | Contract                                                 |
+========================+==========================================================+
| ``nodes``              | the tree as ``(kind, config)`` pairs                     |
+------------------------+----------------------------------------------------------+
| ``samples``            | the batch of ``TraceSample`` records                     |
+------------------------+----------------------------------------------------------+
| ``models``             | the method's only path to a model: ``models.served`` is  |
|                        | the model under test, ``models["teacher"]`` comes from   |
|                        | ``evolution.models``                                     |
+------------------------+----------------------------------------------------------+
| ``manifest``           | optional, keyword-only: the previous step's              |
|                        | ``FailureManifest``, the per-task record of which        |
|                        | episodes failed and how                                  |
+------------------------+----------------------------------------------------------+
| ``Mutation``           | ``create``, ``update``, or ``remove`` on one root-level  |
|                        | entry; a sequence applies as one composite proposal      |
+------------------------+----------------------------------------------------------+
| ``result``             | one finished episode: exit code, stdout, stderr, and the |
|                        | parsed ``trajectory``                                    |
+------------------------+----------------------------------------------------------+
| ``evaluation.metrics`` | guarantees ``candidate_scores`` and ``current_scores``:  |
|                        | per-task score lists in task order, ``None`` for an      |
|                        | episode that could not run                               |
+------------------------+----------------------------------------------------------+

Returning ``None`` from ``propose`` skips the step. The worked examples are in
`Evolve your harness <../user-guide/evolve-your-harness.rst#write-a-method>`__.

Candidate evaluation
--------------------

.. code:: python

   from reef import CandidateEvaluationPlugin, EvaluationResult, SelectionDecision

A plugin measures a produced candidate before it is published, and decides. Reef
enforces the fixed evaluate-then-decide order and verifies the decision kept the
exact result it was given.

+-----------------------------------+-----------------------------------------------+
| Member                            | Contract                                      |
+===================================+===============================================+
| ``evaluate(candidate)``           | returns an ``EvaluationResult``: evaluator    |
|                                   | name, version, and a metrics mapping          |
+-----------------------------------+-----------------------------------------------+
| ``decide(candidate, evaluation)`` | returns a ``SelectionDecision`` whose         |
|                                   | ``outcome`` is ``select`` or ``reject``, and  |
|                                   | which must carry the evaluation it was given  |
+-----------------------------------+-----------------------------------------------+

A rejection leaves the previous artifact serving. An exception is fail-closed:
Reef aborts the prepared candidate rather than publishing it. Make evaluations
idempotent by ``candidate.candidate_id`` because recovery may repeat work whose
result was not durably committed. The deployment names the factory in its
``evaluation`` section (`Configuration
<configuration.rst#the-evaluation-section>`__).

Surface
-------

.. code:: python

   from reef.surface import (
       Surface, create_harness_surface, create_skill_surface, create_weight_surface,
   )

A surface binds one frozen release to its consumers.
``WeightTrainingRecipe.build_surface()`` already calls
``create_weight_surface()``, and ``CordisRecipe`` calls
``create_harness_surface()``, so most methods never touch this.

``Surface`` is a frozen dataclass whose capabilities are fields, not subclass
identity. ``None`` means the capability is absent, and bare ``Surface()`` is the
complete record-only configuration.

+---------------+---------------------------+----------------------------------------------+
| Field         | Type                      | Contract                                     |
+===============+===========================+==============================================+
| ``loader``    | ``ArtifactLoader | None`` | recover the serving head, load rollback      |
|               |                           | checkpoints                                  |
+---------------+---------------------------+----------------------------------------------+
| ``inference`` | ``InferenceHooks | None`` | prepare provider requests, verify responses  |
+---------------+---------------------------+----------------------------------------------+
| ``files``     | ``FileTree | None``       | back client pulls                            |
+---------------+---------------------------+----------------------------------------------+

Two optional protocols extend those structurally, and the scenario checks for
them with ``isinstance``. ``ArtifactActivator`` adds ``loader.activate(artifact,
runtime, source=...)``, making a version servable once it is final.
``LeasingInferenceHooks`` adds ``inference.begin_request(artifact, path)``,
returning a lease the service releases when the attempt ends, so serving state
such as a resident adapter stays protected for its duration.

A surface does not decide which records train, compute candidates, execute a
training job, admit an artifact, or mutate the release chain. Artifact admission
is separate, through ``Recipe.build_artifact_validator()``. Native streaming
behavior stays unchanged. A method should not add an HTTP proxy or copy Reef's
record store.
