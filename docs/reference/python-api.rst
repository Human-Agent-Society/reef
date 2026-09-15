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
| evaluate a produced candidate               | `Candidate evaluation <#candidate-evaluation>`__ | optional, any recipe         |
+---------------------------------------------+--------------------------------------------------+------------------------------+
| a new tensor objective                      | `Loss family                                     | rarely                       |
|                                             | <../developer-guide/loss-families.rst>`__        |                              |
+---------------------------------------------+--------------------------------------------------+------------------------------+

Method code should depend only on what this page documents. Anything else under
``reef.`` is an implementation detail and may change.

Backend and runtime contracts
-----------------------------

Native backends implement model operations; Reef runtimes expose the scheduling
interface used by recipes and serving. The two sides are independent:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Layer
     - Training
     - Inference
   * - Native backend
     - ``TrainingBackend``
     - ``InferenceBackend``
   * - Reef scheduling
     - ``TrainingRuntime``
     - ``InferenceRuntime``

.. code:: python

   from reef.runtime.interfaces import InferenceBackend, InferenceRuntime, TrainingBackend, TrainingRuntime

The native contracts are abstract base classes in ``reef/runtime/interfaces.py``.
The engine supervision hooks an inference integration also implements
(``InferenceEngines``, ``InferenceMonitor``, ``WeightUpdateConnection``,
``EngineHealthChecks``, ``EngineHealthTarget``) live in
``reef/runtime/recovery.py``, next to the objects that drive them.
Implementations inherit the corresponding interface and provide every abstract
operation. Slime's ``SlimeTrainingBackend`` inherits ``TrainingBackend``;
SGLang's ``SGLangInferenceBackend`` inherits ``InferenceBackend``. Reef's
coordinator owns publication and recovery ordering across them. Component
allocation and service shutdown belong to ``TrainingService`` and
``InferenceService`` in ``reef/runtime/deployment.py``. Those service interfaces
also require explicit inheritance; inference factory discovery rejects objects
that only happen to expose similarly named methods.

Two other extension points have separate names and purposes:

- ``reef.train.CandidateBackend`` prepares, evaluates and settles recipe updates,
  including harness edits. ``RuntimeCandidateBackend`` connects weight updates
  to Reef's scheduler; ``Trainer`` accepts it as ``candidate_backend``.
- ``reef.runtime.interfaces.InferenceHandler`` executes one buffered or streaming
  request. Runtime and recipe objects expose it as ``inference_handler``;
  ``create_app`` accepts the same keyword for an injected handler.

Native backend selectors remain ``training.backend`` and ``inference.backend``.
Custom request adapters use ``inference.handler-factory`` and
``inference.handler-config``. The factory path names an ``InferenceHandler``
subclass. Reef calls its ``from_config(upstream_url, *, model_path, timeout_s,
**config)`` class method; arbitrary functions are not accepted. Handlers injected
directly into ``create_app`` need only implement the inference methods.

Streaming handlers return ``InferenceStream``; its ``record_response`` and
``record_response_pending`` fields carry buffered or deferred recording state.
Implementations and test doubles must initialize these fields through the
stream constructor. SGLang handler subclasses that preserve raw reasoning tags
can set ``SPLIT_REASONING = False`` (renamed from ``_SPLIT_REASONING``).

These configuration names replace the previous request-adapter
``backend-factory`` and ``backend-config`` names. Python extensions migrate the
old recipe-facing ``TrainingBackend`` to ``CandidateBackend`` and the old
request-facing ``InferenceBackend`` to ``InferenceHandler``; the names now
reserved for native interfaces must not be used as replacement import aliases
for those different contracts.

Python extension contracts use abstract base classes and explicit inheritance.
This also applies to repository factories, candidate evaluation plugins, harness
plugins, and surface capabilities. Implement optional capabilities only when
supported: adapter weight residency, artifact activation, and request leases
remain separate interfaces. CI rejects ``typing.Protocol``,
``typing_extensions.Protocol``, and ``runtime_checkable`` in all first-party
Python files, including scripts and tutorials; these findings cannot be baselined.
Third-party, generated result and golden fixture exclusions follow the
contribution policy.

Recipe
------

.. code:: python

   from reef.recipe import Recipe, WeightTrainingRecipe, config_field
   from reef.recipe.cordis import CordisRecipe
   from reef.recipe.reefine import ReefineRecipe

A recipe is one frozen dataclass configuring the serving and evolution behavior
for every scenario in a deployment.

.. code:: text

   Recipe                       record-only by default   reef.recipe
   ├── WeightTrainingRecipe     step preparer, loss family, separate runtimes
   │   ├── SAORecipe                                        recipes.sao.recipe
   │   ├── TTTDRecipe                                       recipes.tttd.recipe
   │   └── OpenClawRLRecipe                                 recipes.openclawrl.recipe
   └── CordisRecipe             harness tree + episodes  reef.recipe.cordis
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
| ``runtime``                                       | ``InferenceRuntime`` or     | required inference component   |
|                                                   | ``None``                    | for ``WeightTrainingRecipe``   |
+---------------------------------------------------+-----------------------------+--------------------------------+
| ``training_runtime``                              | ``TrainingRuntime`` or      | required training component    |
|                                                   | ``None``                    | for ``WeightTrainingRecipe``   |
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

``RecordStore`` in ``reef.storage.records`` is the abstract base for appending, reading,
and retiring records. ``ScenarioStore`` composes a ``RecordStore`` with committed
scenario state, settling each step and its record progress together. Recipes
and trainers use the record interface; ``Scenario`` coordinates training and
artifact publication through its supplied scenario store.

Storage implementations explicitly subclass ``RecordStore``,
``ScenarioStore``, and ``ScenarioStorage`` and override their abstract
methods and properties. Incomplete subclasses cannot be instantiated. The
bundled ``SQLiteRecordStore``, ``CommitLogScenarioStore``, and
``SQLiteScenarioStorage`` inherit those bases, respectively.

.. code:: python

   from pathlib import Path

   from reef.dispatcher import Dispatcher
   from reef.storage.sqlite import SQLiteScenarioStorage

   storage = SQLiteScenarioStorage(Path(".reef/agent-record"))
   dispatcher = Dispatcher(
       recipe,
       repository_backend_factory,
       agent_record_dir=Path(".reef/agent-record"),
       scenario_storage=storage,
   )

``Dispatcher`` requires ``scenario_storage`` and passes it to scenario
creation. It never selects or imports a record storage implementation. The
``build_default_dispatcher`` convenience helper also requires this argument.
Deployment assembly chooses SQLite or PostgreSQL from ``reef.record_backend``;
embedded callers supply their chosen storage service explicitly. For ephemeral records
and commits, pass ``SQLiteScenarioStorage()`` without a directory. Passing
a directory preserves SQLite schemas, JSONL format, and scenario filenames.

With an explicit storage service, ``agent_record_dir`` independently selects the
local directory for scenario model settings. Configure it to retain those
settings across restarts even when records and commits use a remote adapter.

``Recipe.with_model_config(config)`` accepts the concrete ``ModelConfig`` from
``reef.inference.model_config``. ``ModelConfig.from_value(value)`` validates the
model override; its ``runtime`` field holds an ``InferenceProxyRuntime``, or
``None`` for the recipe default. ``view()`` returns the credential-redacted
API representation. The type holds no file path and performs no file I/O.
It replaces ``ScenarioModelConfig`` and the former read-only configuration
protocol; there is no separate configuration collection class.

The scenario registry caches one configuration object per scenario and passes
it to the factory during creation and recovery. Recipes retain that object so
new work observes updates while already resolved runtime snapshots remain
unchanged. Use ``Dispatcher.configure_scenario_model`` to persist updates;
it validates the value and recipe compatibility and writes the private JSON
file before replacing the active runtime. ``reef.storage.model_config`` owns
file reads, atomic writes, and archival. Existing file paths, JSON values,
and owner-only permissions remain unchanged.

Direct ``Scenario`` construction requires ``store=...``. Replace the former
``records=...`` and ``commit_log=...`` arguments with one store opened by a
``ScenarioStorage``. The supplied trainer must use that store's ``records``.
``Scenario`` does not choose a storage backend. Read committed history through
``scenario.store.history()`` and persistence capability through
``scenario.store.durable``; the former ``scenario.commit_log`` property has been
removed. The concrete commit log store's ``commit_log`` is for backend diagnostics.

The abstract bases ``ScenarioStore`` and ``ScenarioStorage`` are exported from
``reef.storage.scenario``, alongside ``ScenarioStoreConflict``. Concrete storage classes
live under ``reef.storage``: ``commit_log.CommitLogScenarioStore`` accepts a
``RecordStore`` and a JSONL ``CommitLog``; ``sqlite.SQLiteScenarioStorage`` and
``postgres.PostgresScenarioStorage`` assemble their respective record backends.
The scenario package depends only
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
   * - ``recover(checkpoint=...)``
     - Accept a checkpoint ``CommitRecord`` (``None`` at initial registration),
       reconcile it with history, repair interrupted record
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
``ScenarioCommitter`` owns their order around store settlement, including
pending releases, rollback, and checkpoint reconciliation. See `Commit ordering
<../advanced_topics/state-model.rst#commit-ordering>`__. Step validation does not
provide distributed leases or authorize multiple active training writers.

The storage service exposes ``durable``, ``open(scenario)``, ``archive(scenario)``,
``prune(days=..., max_bytes=...)``, and ``close()``. Archive retires that scenario's
stored state so its name can be reused; prune applies retained-body limits to
active and archived state. Custom backends implement those operations instead
of exposing local file paths. Each opened session belongs to its scenario and
closes after its trainer, including on construction or recovery failure. The
dispatcher closes its storage service on shutdown. Application assembly should use
the scenario storage service; direct record store construction is described below.

Record storage and audit
------------------------

``reef.storage.records`` defines ``RecordStore``, record result and error types, and
``RecordRetention``. It has no dependency on scenario coordination, training,
or concrete storage adapters. Both ``reef.RecordStore`` and
``reef.storage.records.RecordStore`` refer to this abstract base.

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
``PostgresScenarioStorage(url, directory, schema=...)`` combines PostgreSQL
records with ``CommitLogScenarioStore`` and local JSONL logs. Pass it through
``Dispatcher(..., scenario_storage=...)`` for embedded use. Scenario code
continues to depend only on the abstract store contracts. Deployment configuration
selects this storage service with ``reef.record_backend: postgres``; see
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

``RecordRetention`` in ``reef.storage.records`` holds and validates the
``days`` and ``max_bytes`` limits. Store factories apply those limits through
their ``prune`` method.

``reef.storage.records.RecordStore`` separates the training record set from retained
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

With the default SQLite storage service, the HTTP service runs background retention at
startup and every 60 seconds.
It removes bodies older than 7 days, then the oldest remaining bodies to meet
a shared 20 GiB budget across scenario databases in ``agent_record_dir``,
including ``archived/``. The budget measures UTF-8 JSON payloads, references,
and artifact references. Limits are configurable in `Configuration <configuration.rst>`__.

For embedded Python deployments, use
``dispatcher.prune_record_archives(RecordRetention(days=7, max_bytes=20 * 1024**3))``
with ``RecordRetention`` imported from ``reef.storage.records``. This runs one sweep
through the configured storage service and serializes it with scenario archival.
Standalone ``SQLiteRecordStore`` and ``Dispatcher`` construction do not start a
maintenance task. ``create_app(dispatcher, ...)`` requires an existing dispatcher
and accepts ``record_retention=RecordRetention(...)`` to enable service maintenance.
It closes the dispatcher on cleanup only when ``close_dispatcher=True``.
The deployment assembly selects the storage backend and passes the dispatcher
to the HTTP app; ``create_app`` never creates a default backend.

The former ``RecordRetention.prune(directory)`` method has been removed: the
retention value no longer opens databases or accepts a filesystem path. Use
``dispatcher.prune_record_archives(retention)`` for a running dispatcher. For
standalone SQLite maintenance, pass the directory to the storage service:

.. code:: python

   from pathlib import Path

   from reef.storage.records import RecordRetention
   from reef.storage.sqlite import SQLiteScenarioStorage

   retention = RecordRetention(days=7, max_bytes=20 * 1024**3)
   storage = SQLiteScenarioStorage(Path(".reef/agent-record"))
   try:
       purged = storage.prune(days=retention.days, max_bytes=retention.max_bytes)
   finally:
       storage.close()

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

Reported feedback supplies a ``make_sample`` hook that assembles valid reports;
computed feedback supplies an asynchronous ``judge`` that derives new feedback.
Both engines own their buffering, reservations, retention, and replay. Reports
must reference existing inference records in the same scenario.

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

``ReportedFeedbackProcessor`` exposes these recipe hooks:

.. list-table::
   :header-rows: 1

   * - Hook
     - Contract
   * - ``make_sample(context) -> TrainDataItem``
     - Assemble one valid report and its resolved inferences; raise on data errors.
   * - ``make_batch(items, batch_number) -> TrainingBatch``
     - Shape the flat tuple of selected training items into a batch.
   * - ``grouping(context) -> (group_key, slot)``
     - Optional collection group and retry slot; defaults to an independent report.
   * - ``decide_group(key, items) -> GroupDecision``
     - Required when grouping supplies a group key; return READY, INCOMPLETE, or DISCARD.

``ReportContext`` carries ``report``, ordered ``inferences``, optional ``score``,
and the recipe's ``parsed_report``. ``require_score()`` returns a finite reward or
raises if the training method cannot use the supplied feedback.

``make_sample`` returns a trajectory or task directly. The processor adds the
report id and ordered references to ``source_agent_record_ids``. Grouping and
consumption state stay private; recipes never receive buffered report objects.
``grouping`` returns ``(None, None)`` by default. A group key collects reports,
and a slot deduplicates retries within that group (None uses the report id).
This collection group differs from ``TrajectoryItem.group_id``: TTTD waits for
a complete step, then trains several comparison groups within it.
``make_batch`` receives items in group/arrival order. Acknowledgement consumes
all selected reports, even when the recipe omits constant-reward groups from
its output. Reports arriving after reservation remain for a later batch.
``output_schema`` declares the batch type, ``exclusive_sources`` controls source
release for terminal group/duplicate reports, and ``ordered_groups`` orders ready
groups by their keys.

There is no report-level ``judge``, ``WAIT``, ``NEVER``, or eligibility flag.
Reference validation runs at admission; incomplete groups wait for more valid
samples. Training input violations raise instead of silently filtering reports.

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
| ``make_sample(record, judgment)``     | a ``TrajectoryItem``, or ``None`` |
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

   from pathlib import Path
   from reef.train.types import (
       TaskItem, TrainDataItem, TrainingBatch, TrajectoryItem,
       trajectories, trajectory_groups,
   )

``TrainDataItem = TrajectoryItem | TaskItem``. Every processor returns a frozen
``TrainingBatch(batch_id, items, request=None)`` with an ordered tuple of items.
The same batch may contain both kinds. ``request`` is keyword-only and carries
an optional explicit training instruction; a request-only batch may be empty.
A processor returns the same pending batch until its id is acknowledged.

.. code:: python

   batch = TrainingBatch(
       "scenario:batch:1",
       (
           TrajectoryItem(atif_document, source_agent_record_ids=("inference-1", "report-1")),
           TaskItem(Path("tasks/example"), source_agent_record_ids=("record-2",)),
       ),
   )

A ``TrajectoryItem.trajectory`` is an
`ATIF document <https://github.com/harbor-framework/harbor/blob/main/rfcs/0001-trajectory-format.md>`__
as a JSON-compatible mapping. All shipped processors and recipes use this
representation. The item checks the envelope; consumers validate the fields
their algorithms require. Standard steps describe messages, tool calls and
observations. Captured records use ATIF-v1.7, matching the pinned Harbor version.

Reef-specific fields live inside ``trajectory.extra.reef``:

- ``reward`` and ``feedback`` hold the supplied or computed training signal;
- ``source_agent_record_id`` identifies the primary source;
- ``records`` preserves ordered source ids, timestamps and original provider
  payloads, including provider-specific fields;
- ``training`` holds exact captured ``tokens``, ``loss_mask``,
  ``rollout_log_probs``, ``runtime_load_id`` and ``runtime_load_spans``.
  Recipes may also supply ``action_mask``, ``rollout_created_at``, ``turn_count``,
  ``topk_indices``, ``topk_log_probs`` and method-specific ``extras``.

``item.metadata`` and ``item.training`` read these extension objects.
``with_metadata`` and ``with_training`` return updated documents while preserving
other extensions. Training arrays are JSON lists, and runtime load spans are
objects with ``start``, ``end`` and ``runtime_load_id``.

``TrajectoryItem.group_id`` is an optional string, scoped to the batch.
None means an independent trajectory. Group-relative policy algorithms require
an explicit id on every trajectory and contiguous items for each group, so
advantages and rows stay aligned. TTTD and CORAL preserve their existing group
barriers and ordering. SAO and OpenClaw-RL explicitly schedule each trajectory
independently, even when group metadata is present. ``trajectory_groups(batch)`` reads these comparison groups;
``trajectories(batch)`` reads ATIF items in flat item order.

A ``TaskItem`` names a `Harbor task directory <https://www.harborframework.com/docs/tasks>`__
with ``task_path: Path``: the directory contains the instruction, environment
and verifier configuration/files. The path is interpreted in the consuming
algorithm's execution environment. The value also accepts optional ``metadata``.
It performs no file I/O or rollout. Algorithms supporting tasks own validation,
environment execution and conversion to trajectories. Current built-in policy
and harness backends explicitly reject task items; accepting the container
contract does not imply that every algorithm supports every item kind.

Both item kinds may carry ``source_agent_record_ids``. These record derivation
inputs; processor acknowledgement remains authoritative for consumption,
including inputs removed by a recipe's existing group rules.

The ``reef.core.trajectories`` helpers construct and consume ATIF:
``make_trajectory(records, reward, feedback)`` retains original exchanges;
``recorded_payload`` / ``recorded_payloads`` project exchanges for harness
methods, including external ATIF without captured records; and
``trajectory_reward`` reads a finite policy reward. Policy assembly uses
``make_policy_trajectory`` or ``make_multi_turn_policy_trajectory`` from
``reef.train.processors.common`` to add exact captured tensors. Missing required
tensors fail at the training boundary; these helpers do not invent tokens or
log probabilities from text. Both batch readers reject unsupported item kinds
instead of silently filtering mixed batches.

Migration: the former policy/trace sample types and their batch subclasses are
removed. Construct ATIF items directly and return ``TrainingBatch(id, items)``.
Use ``group_id`` on each member for grouped batches. Read captured tensors from
``item.training`` and rewards through ``trajectory_reward(item)``.
``output_schema`` defaults to ``TrainingBatch``. Keep batches serializable, with
no handles to services, files, threads or models.

Step preparer
-------------

.. code:: python

   from reef.train.algos import StepSignal
   from reef.core.trajectories import trajectory_reward

A preparer turns a reserved batch into a pure, backend-neutral signal. Normally
it is a plain function named by the recipe as ``package.module:callable``, and
its module must be importable in both the service and the training process.

.. code:: python

   def prepare_step(batch: TrainingBatch, state: Mapping[str, Any]) -> StepSignal:
       samples = trajectories(batch)
       steps = next_steps(state)
       return StepSignal(
           action="train",
           loss_family="my_method",
           advantages=tuple(trajectory_reward(sample) for sample in samples),
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

   def propose(nodes, samples, models) -> Mutation | Sequence[Mutation] | StepProposal | None: ...
   def evaluate(task, result) -> float: ...
   class Selection:  # optional
       def decide(self, candidate, evaluation) -> SelectionDecision: ...

+------------------------+----------------------------------------------------------+
| Member                 | Contract                                                 |
+========================+==========================================================+
| ``nodes``              | the tree as ``(kind, config)`` pairs                     |
+------------------------+----------------------------------------------------------+
| ``samples``            | the batch of ATIF ``TrajectoryItem`` values              |
+------------------------+----------------------------------------------------------+
| ``models``             | the method's only path to a model: ``models.served`` is  |
|                        | the model under test, ``models["teacher"]`` comes from   |
|                        | ``evolution.models``                                     |
+------------------------+----------------------------------------------------------+
| ``manifest``           | optional, keyword-only: the previous step's              |
|                        | ``FailureManifest``, the per-task record of which        |
|                        | episodes failed and how                                  |
+------------------------+----------------------------------------------------------+
| ``entries``            | optional, keyword-only: the tree as entry options,       |
|                        | ``{"id", "name", "config"}`` mappings in tree order, so  |
|                        | an ``update`` or ``remove`` can name its entry           |
+------------------------+----------------------------------------------------------+
| ``Mutation``           | ``create``, ``update``, or ``remove`` on one root-level  |
|                        | entry; a sequence applies as one composite proposal      |
+------------------------+----------------------------------------------------------+
| ``StepProposal``       | the mutations plus ``notes``, a JSON mapping the step    |
|                        | records under ``proposal_notes`` and never reads; empty  |
|                        | mutations skip the step                                  |
+------------------------+----------------------------------------------------------+
| ``result``             | one finished episode: exit code, stdout, stderr, and the |
|                        | parsed ``trajectory``                                    |
+------------------------+----------------------------------------------------------+
| ``evaluation.metrics`` | guarantees ``candidate_scores`` and ``current_scores``:  |
|                        | per-task score lists in task order, ``None`` for an      |
|                        | episode that could not run; ``current_scores`` is empty  |
|                        | under ``selection: floor``, which runs the candidate     |
|                        | alone                                                    |
+------------------------+----------------------------------------------------------+

Returning ``None`` from ``propose`` skips the step. The worked examples are in
`Evolve your harness <../user-guide/evolve-your-harness.rst#write-a-method>`__.

Candidate evaluation
--------------------

.. code:: python

   from reef import CandidateEvaluationPlugin, EvaluationResult, SelectionDecision

A plugin explicitly inherits ``CandidateEvaluationPlugin``, measures a produced
candidate before it is published, and decides. ``CandidateEvaluator`` and
``CandidateSelector`` remain separate abstract capabilities; implementing only
one does not make an object a plugin. Reef
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

``CandidateEvaluationPluginFactory`` is an abstract base class whose
``build(config, *, runtime, training_runtime, scenario, environ)`` returns one
scenario-local plugin. ``evaluation.module`` names its subclass or instance;
factory subclasses must construct without arguments and without allocating
model resources. Plain callable factories and objects that merely expose
matching methods are rejected.

Harness recipes instead use ``reef.train.evaluation.CandidatePluginFactory``:
its ``build(candidate_backend)`` binds a plugin to an existing candidate
backend. ``AlwaysSelectPluginFactory`` and ``ScoreComparisonPluginFactory``
provide the built-in policies; custom factory classes explicitly inherit the
same interface.

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

Runtime responsibilities
------------------------

``TrainingRuntime`` and ``InferenceRuntime`` are independent interfaces. Neither
inherits from the other, and there is no aggregate runtime.

* ``TrainingRuntime`` prepares batches, produces candidate checkpoints, rejects
  candidates and restores training weights/optimizer state. It receives serving
  versions as values; it does not own an inference endpoint or request backend.
* ``InferenceRuntime`` executes requests, manages admission and reconnection,
  loads selected weights or adapters, and reports serving versions. It restores
  serving weights without restoring optimizer state.
* The existing ``RuntimeCandidateBackend`` coordinates both: prepare/train,
  evaluate, activate or reject, delegating scheduling and durable publication
  acknowledgement to ``RuntimeScheduler``. ``ScenarioCommitter`` coordinates rollback across both runtimes and
  commits the restored artifact before reopening inference.

Weight recipes hold ``training_runtime: TrainingRuntime`` and
``runtime: InferenceRuntime`` separately. Construction is explicit:

.. code:: python

   recipe = SAORecipe(training_runtime=training, runtime=inference)

Training deployment factories return ``(training_runtime, inference_runtime)``;
inference-only factories return an ``InferenceRuntime``. ``connect_ray_runtime``
and ``connect_executor_runtimes`` in ``reef.service.runtime`` return the same pair.
``ExecutorTrainingRuntime`` in ``reef.train.runtime`` and
``ExecutorInferenceRuntime`` in ``reef.inference.runtime`` use the existing
``CoordinatorClient`` control connection for their respective operations. That
legacy RPC connection still exposes both training and publication operations;
it is not a public runtime or a new backend-neutral weight transport.

Migration: split implementations of the former combined training runtime into
these two interfaces and inject both into the recipe. The aggregate runtime
classes and the ``RayRuntime`` alias are removed. The ``executor_training`` and
``ray_training`` config kinds, control RPCs and stored artifact formats remain
unchanged. Adding another backend combination still requires compatible native
weight transport and recovery behavior.

``reef.runtime`` is a namespace package without an import facade. Its five modules
are ``interfaces``, ``scheduler``, ``deployment``, ``publication`` and ``recovery``;
``executor/`` contains worker transports. Import the ABCs and shared values from
``reef.runtime.interfaces`` and the factory registry from
``reef.runtime.deployment``. Concrete scheduling connections live in their owning
integration: ``SlimeTrainingRuntime`` in ``reef.train.slime_backend.runtime`` and
``SGLangInferenceRuntime`` in ``reef.inference.sglang.runtime``. These remain
distinct from the native ``TrainingBackend``/``InferenceBackend`` pair consumed by
Reef's coordinator.

Tinker integration
------------------

``reef.train.tinker_backend.launch.TinkerDeployment`` implements the optional
``tinker`` backend. Its ``runtime_factory`` builds, from ``TinkerConfig`` and only
after deployment selection, a ``TinkerTrainingRuntime`` and, through the
``tinker`` inference kind, a ``reef.inference.tinker.TinkerInferenceRuntime``.
The two hold no shared object: the training runtime branches every candidate
from the incumbent it remembers on disk, learning commits through
``TrainingRuntime.commit_candidate`` and rollbacks through
``restore_checkpoint``; the inference runtime reads each candidate's or
artifact's ``tinker-checkpoint.json`` manifest, samples from the immutable
sampler it names, activates selected candidates and binds the head Reef
publishes through ``activate_checkpoint``. With a local inference engine the
same deployment instead returns a model-driver plan: ``TinkerTrainingService``
supplies ``TinkerTrainingBackend``, a ``reef.runtime.interfaces.TrainingBackend``
that delivers each published adapter as a PEFT directory through
``adapter_files``; Reef's publisher then calls the receiver's
``InferenceBackend.load_adapter_files`` (the ``reef-adapter-files-v1``
transfer any engine that loads adapter directories can declare), so the
trainer never holds an engine handle. The HTTP service connects through the
``coordinator_training`` runtime kind like any coordinator-driven trainer. ``TrainingDeployment``
defaults ``requires_local_model`` to true; hosted integrations set it to false
to preserve remote model identifiers during deployment resolution.

Methods implement ``TinkerLoss`` from ``reef.train.tinker_backend.losses``
and register it with ``register_loss_family_ref(name, "package.module:CLASS",
backend="tinker")`` from ``reef.train.algos.registry``, the same table that
holds their Slime reference. ``loss_fn`` names the Tinker built-in loss the
family trains with; ``inputs(rows, base_logprobs, kl_coef=...)`` returns that
function's token-aligned ``loss_fn_inputs``, one dictionary per row; and
``needs_base_logprobs`` requests frozen-base scores for a nonzero KL
coefficient. For ``importance_sampling``, ``TokenRow.inputs`` builds the
dictionary from response-token advantages, shifting prediction targets by one
token and padding prompt positions with zeros. A ``TinkerCustomLoss`` adds
``loss(data, logprobs)``, which the SDK boundary passes to
``forward_backward_custom`` so the family computes its loss tensor on the
Reef host. ``Datum`` objects are constructed only after this shaping. See
`Train with Tinker <../user-guide/tinker.rst>`__ for supported contracts.
