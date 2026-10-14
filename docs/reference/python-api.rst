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
| turn records into a typed batch             | `Pipeline stages <#pipeline-stages>`__           | every method producing       |
|                                             |                                                  | updates                      |
+---------------------------------------------+--------------------------------------------------+------------------------------+
| carry data to the backend                   | `Batch <#batch>`__                               | every method producing       |
|                                             |                                                  | updates; subclass only for a |
|                                             |                                                  | new shape                    |
+---------------------------------------------+--------------------------------------------------+------------------------------+
| turn a batch into a signal                  | `Training objective <#training-objective>`__     | weight-training methods      |
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

The native contracts are abstract base classes in ``reef/runtime/interfaces.py``;
``InferenceEngine`` there names the operations one engine actor offers, which
``ReefSGLangEngine`` and ``ReefVLLMEngine`` implement.
The engine supervision hooks an inference integration also implements
(``InferenceEngineGroup``, ``InferenceMonitor``, ``WeightUpdateConnection``,
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
   ├── WeightTrainingRecipe     training objective, loss family, separate runtimes
   │   ├── SAORecipe                                        recipes.sao.recipe
   │   ├── TTTDRecipe                                       recipes.tttd.recipe
   │   ├── SDFTRecipe                                       recipes.sdft.recipe
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
| evolve weights and a harness    | configure ``CompositeRecipe`` with one recipe per    |
| (or configuration) together     | component; each trainer runs as its own worker       |
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
| ``build_trainers(scenario, records,``             | ``tuple[ComponentTrainer,   | one trainer per release        |
| ``surface=..., algorithm_states=...)``            | ...]``                      | component; defaults to the     |
|                                                   |                             | single trainer ``build``       |
|                                                   |                             | returns, bound to the          |
|                                                   |                             | surface's one component        |
+---------------------------------------------------+-----------------------------+--------------------------------+
| ``build_surface(scenario)``                       | ``Surface``                 | the delivery contract for one  |
|                                                   |                             | named scenario: its release    |
|                                                   |                             | components, each with its      |
|                                                   |                             | admission check (default       |
|                                                   |                             | ``AcceptAnyArtifact()``)       |
+---------------------------------------------------+-----------------------------+--------------------------------+
| ``build_artifact_validator()``                    | ``ArtifactValidator``       | kept from before component     |
|                                                   |                             | checks: joins the check of the |
|                                                   |                             | one component served, admits   |
|                                                   |                             | the whole release of a recipe  |
|                                                   |                             | serving none (inside a         |
|                                                   |                             | composite, that recipe's       |
|                                                   |                             | component); an override on a   |
|                                                   |                             | recipe serving several is      |
|                                                   |                             | refused at build               |
+---------------------------------------------------+-----------------------------+--------------------------------+
| ``serving_status()``                              | ``Mapping | None``          | runtime-wide state for         |
|                                                   |                             | ``/reef/status``               |
+---------------------------------------------------+-----------------------------+--------------------------------+

Every recipe may declare ``report_type``, the ``ReportBase`` subclass its
reports parse as (``None`` keeps ingress open), and ``harness_adapter``, the
adapter a client installs its harness tree with (``None``, the default, for a
recipe with no harness tree; ``CordisRecipe`` names ``evolution.adapter``).
The scenario list names it for every scenario. Every recipe declares
``pipeline_spec()``, the default class of each pipeline stage (see `Pipeline
stages <#pipeline-stages>`__). Weight-training recipes add ``training_spec()``,
which binds the registered or dotted training objective, which declares the
backend loss family, and the ``StepScheduling`` the runtime cuts each batch with; ``max_staleness``, the accepted
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
               objective="my_method.objective:MyObjective",
               scheduling=StepScheduling(unit="sample"),
           )

       @classmethod
       def pipeline_spec(cls) -> PipelineSpec:
           return PipelineSpec(
               record2record=MyMethodRecord2Record,
               record2task=MyMethodTasks,
               task2batch=MyMethodBatches,
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
from the service configuration. A setting that a pipeline stage reads belongs
in the stage's settings, under the deployment's ``pipeline`` section, not in a
recipe field. Never read deployment YAML from inside a stage.

Report
------

.. code:: python

   from reef.core.reports import ReportBase, ReportValidationError, ScoredRolloutReport, TeacherContextReport

``ScoredRolloutReport`` is the shared score-only contract and
``TeacherContextReport`` the shared contract of the distilling recipes: a
rollout's receipt and the ``teacher_context`` its teacher sees (a demonstration,
environment feedback, or nothing for a separate teacher).
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

``RecordStore.append_many(items)`` atomically appends an ordered batch from one
scenario and returns an ``AppendResult`` for each input. Conflicts roll back the
batch; identical retries and retired records follow ``append_result`` semantics.
SQLite and PostgreSQL implement this with one write transaction. Other adapters
must implement atomic batch append to support ``POST /reef/records/batch``;
the default raises ``NotImplementedError`` before writing. Single-record
callers remain compatible with existing adapters.

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
       ordered replay, scenario isolation, audit reads, and consumption receipts.
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
     - Persist the ``CommitRecord`` and its consumption progress, returning the
       canonical accepted record. Its step must equal ``expected_step + 1``.
   * - ``recover(checkpoint=...)``
     - Accept a checkpoint ``CommitRecord`` (``None`` at initial registration),
       reconcile it with history, and return the head ``CommitRecord`` or ``None`` for a fresh
       scenario.
   * - ``close()``
     - Release the session's record and commit resources; repeated calls are safe.

``CommitRecord`` carries the artifact ref, algorithm state, record watermark,
consumed IDs, checkpoint/pending flags, operation and rollback
target, metrics, and training job identity. The store must atomically validate
the current step before accepting a new successor: two different commits
prepared from the same step cannot both succeed. An identical recorded retry
returns the original record, even after later steps have committed; only
``recorded_at`` is excluded from retry comparison. A new commit with a stale
expected step, or a retry with conflicting content, raises
``ScenarioStoreConflict`` without advancing the store.

The commit log adapter serializes its writers with a local POSIX file lock;
direct writes through ``CommitLog`` bypass this store contract.
The default adapter fsyncs the JSONL record as the durable commit point.
A failure after append can leave the step committed. Retry the exact commit or
recover the session; do not infer rollback from an exception. Recovery uses
committed ``consumed_ids`` and separate skipped-batch receipts to rebuild
processor memory without repeating work. It does not modify stored bodies.

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
SQLite leaves it empty because each store owns a database. Custom SQL adapters
must also supply ``RecordTables.eviction`` with scope/scenario keys and the
``record_count``, ``body_bytes``, ``first_sequence`` and ``last_sequence`` columns.
These totals are updated atomically with capacity deletion and retry hashes.

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
receipt keys hash large consumed ID sets, with complete canonical content checked
by the shared SQL layer. PostgreSQL timestamps use double precision, and sequences
use 64-bit identities. Retention applies the same byte-budget policy to
all bodies across active and archived generations in the deployment schema.

Existing SQLite databases remain readable without copying record bodies. Direct callers must replace
``RecordStore(path)`` with ``SQLiteRecordStore(path)``; the abstract base cannot
be instantiated. ``SQLiteRecordStore`` is also exported from ``reef``:

.. code:: python

   from pathlib import Path

   from reef.storage.sqlite import SQLiteRecordStore

   with SQLiteRecordStore(Path(".reef/records.sqlite3")) as records:
       retained = records.audit_page("math")

``RecordRetention`` in ``reef.storage.records`` holds and validates the
``max_bytes`` capacity budget. ``days`` remains a validated compatibility
argument but does not expire data. Store factories apply capacity limits through
``prune`` independently of record consumption.

Training commits persist consumption progress without changing record
visibility or deleting bodies. Processors expose ``releasable_record_ids()``
and ``release_records(ids)`` for memory management. Storage controls capacity
independently of these buffers. With several trainers in one scenario, a
stale drop's consumption receipt carries ``component`` in its metadata and a
``<component>:<batch id>`` receipt id, and each trainer skips on recovery only
the rows its own commits and receipts consumed.

``get``, ``replay``, ``replay_page``, and ``count`` read all retained records.
A one-time upgrade converts older retirement markers and receipts into
consumption records. It preserves original bodies, drops the old schema,
and keeps consumption metadata out of record reads.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Method
     - Result
   * - ``get_for_audit(scenario, agent_record_id)``
     - A ``StoredRecord``, or ``None`` if no body is retained in that scenario.
   * - ``audit_page(scenario, after_sequence=0, limit=256)``
     - A bounded tuple of ``StoredRecord`` entries, in append order, including
       consumed bodies. Advance the cursor using the last entry's ``sequence``.
   * - ``loss(scenario)``
     - Durable ``RecordLoss`` totals: ``record_count``, ``body_bytes``,
       ``first_sequence`` and ``last_sequence`` for capacity-evicted bodies.
       Counts include consumed and unconsumed records.
   * - ``record_consumption(scenario, ids, receipt_id=..., metadata=...)``
     - Persist a skipped batch's consumed IDs without modifying its records.
       Identical retries are idempotent; conflicting metadata is rejected.
   * - ``consumption_receipts(scenario)``
     - Ordered receipts with ``receipt_id``, ``consumed_ids``, ``metadata``,
       and ``recorded_at``. Includes consumption from legacy receipts.

``StoredRecord`` contains ``sequence`` and ``item`` (the original ``AgentRecord``).
Reading a record does not change a consumer's progress. A missing body may have been purged or never stored;
the read API does not guess which. Use the commit log's per-step
``consumed_ids`` to determine consumption, including intentional skips; it is
not proof that every named record produced a model update.

For example, inspect one trace without making it available to training again:

.. code:: python

   entry = scenario.records.get_for_audit(scenario.name, record_id)
   if entry is not None:
       payload = entry.item.payload
       references = entry.item.references

With the default SQLite storage service, the HTTP service runs background retention at
startup and every 60 seconds.
It evicts oldest bodies only when their total exceeds a shared 20 GiB budget
across scenario databases in ``agent_record_dir``,
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

Capacity eviction includes active records and logs warnings with counts, byte
sizes and sequence ranges. Retry hashes and commit/receipt metadata survive,
so identical retries still deduplicate and conflicting content still fails.
SQLite deletes commit in batches of 256; PostgreSQL serializes a sweep with
writers in one transaction. Concurrent writes can exceed the budget between
sweeps. SQLite reuses freed pages without shrinking its file. Indexes and other
metadata also require disk space. HTTP audit routes remain a separate integration. See
`Configuration <configuration.rst>`__ for migration and rollback constraints.

Pipeline stages
---------------

.. code:: python

   from reef.record2record import GroupDecision, RecordJudging, ReportContext, ReportGrouping, ReportRecord2Record
   from reef.record2task import JudgedTasks, ReportTasks, SkippedReport
   from reef.task2batch import DistillBatches, Task2Batch

A training pipeline (``reef.train.pipeline.TrainingPipeline``) turns durable
records into typed batches through three stages. A recipe names a default class
for each in ``pipeline_spec()`` (``PipelineSpec(record2record=...,
record2task=..., task2batch=...)``):

- ``record2record``: the record-to-record stage
  (``reef.record2record``). It decides which records train and in what order:
  it groups reports, judges records, and drops records that can no longer
  train. Without one, each record passes through and nothing trains.
- ``record2task``: the record-to-task stage
  (``reef.record2task.Record2Task``). It turns each record set into tasks
  (``TrainDataItem`` values). Reported feedback supplies a ``make_sample`` hook
  on ``ReportTasks`` that assembles valid reports; computed feedback supplies
  a ``JudgedTasks`` after an asynchronous ``judge`` derives new feedback.
- ``task2batch``: the task-to-batch stage (``reef.task2batch.Task2Batch``).
  Its ``make_batch(tasks, batch_number)`` shapes the tasks of the selected
  sets into a batch.

Reef builds the stages for each scenario, in this order, with the class
methods below; a later stage receives the stages built before it:

- ``Record2Record.from_config(config, *, context, environ)``
- ``Record2Task.from_config(config, *, context, record2record, environ)``
- ``Task2Batch.from_config(config, *, context, record2record, record2task, environ)``

``config`` is the stage's config from the deployment's ``pipeline`` section;
``environ`` holds the environment the deployment was read with. A stage
declares its keys in a frozen ``reef.core.plugins.PluginSettings`` subclass
with ``reef.core.config.config_option`` fields and validation in
``__post_init__``. In ``from_config``, parse the config with
``Settings.parse(config, plugin_class=cls)`` (``NoSettings`` for a stage
without config) and give the settings object to the constructor. A bad config
fails when the first scenario builds its pipeline. A later stage that needs an
earlier stage as a specific class checks it with ``isinstance`` and raises
``PluginConfigError`` otherwise. A value that several stages need
belongs to the first of them; later stages read it from the stage object
they receive. A ``Record2Task`` that keeps
per-scenario state also implements ``scenario_state_dirs(config, scenario)``,
so a scenario delete archives it. Reef owns replay, retention, deduplication, pending batches,
and exactly-once consumption; the method implements only the hooks below.
They run synchronously on the trainer thread, so they must not block on
network or model latency. Reports must reference existing inference records
in the same scenario.

``TrainingPipeline.training_mode`` selects automatic, instruction-triggered or
combined batching. A ``Task2Batch`` declares ``supported_training_modes``. In
``manual`` an instruction's batch has no tasks; in ``hybrid`` it takes the sets
an automatic batch would. Ingestion, acknowledgement, retention, buffer release
and background derivation are shared. See
`Training pipeline <../developer-guide/processors.rst>`__ for the instruction queue and batch contract.

Every stage gets the scenario's experiment logger as
``context.experiment_logger`` (``self.experiment_logger`` on a ``Task2Batch``).
Log finite numeric metrics under the method's namespace; stage code never
imports W&B, and the logger is a no-op when tracking is off.

Reported feedback
~~~~~~~~~~~~~~~~~

A reported-feedback recipe implements these hooks:

.. list-table::
   :header-rows: 1

   * - Hook
     - Contract
   * - ``ReportTasks.make_sample(context) -> TrainDataItem | SkippedReport``
     - Assemble one valid report and its resolved inferences; raise on data errors. Return
       ``SkippedReport(reason)`` for a valid report that cannot train; the stage releases it.
   * - ``Task2Batch.make_batch(tasks, batch_number) -> TrainingBatch``
     - Shape the flat tuple of selected tasks into a batch.

The recipe owns the report policy. Its record stage is a subclass of
``ReportRecord2Record(context, grouping, *, exclusive_sources=False,
ordered_groups=False)``; ``ReportRecord2Record(context, ReportGrouping())``
trains every report by itself. A ``ReportGrouping`` runs in the
record-to-record stage and reads only records:

.. list-table::
   :header-rows: 1

   * - Method
     - Contract
   * - ``placement(context) -> (group_key, slot)``
     - Optional collection group and retry slot; defaults to an independent report.
   * - ``decide(key, members) -> GroupDecision``
     - Required when ``placement`` supplies a group key. ``members`` are the group's
       ``ReportContext`` values in arrival order. Return READY, INCOMPLETE, or DISCARD.
   * - ``is_training_report(report) -> bool``
     - Optional; ``False`` for a valid report that is not this method's training data. The stage releases it and
       never assembles it; its sources go with it only under ``exclusive_sources`` or when it references more than
       one inference.
   * - ``status() -> Mapping``
     - Optional JSON-safe group outcomes, such as steps that can never train. The pipeline adds them to its status.

``ReportContext`` carries ``report``, ordered ``inferences``, optional ``score``,
and the recipe's ``parsed_report``. ``require_score()`` returns a finite reward or
raises if the training method cannot use the supplied feedback.

``make_sample`` returns a trajectory or task directly. ``ReportTasks`` adds the
report id and ordered references to ``source_agent_record_ids``. Grouping and
consumption state stay in the stage; recipes never receive buffered report objects.
``placement`` returns ``(None, None)`` by default. A group key collects reports,
and a slot deduplicates retries within that group (None uses the report id).
This collection group differs from ``TrajectoryItem.group_id``: TTTD waits for
a complete step, then trains several comparison groups within it.
``make_batch`` receives tasks in group/arrival order. Acknowledgement consumes
all selected reports, even when the recipe omits constant-reward groups from
its output. Reports arriving after reservation remain for a later batch.
``output_schema`` declares the batch type. On the stage, ``exclusive_sources``
controls source release for terminal group/duplicate reports, and
``ordered_groups`` orders ready groups by their keys.

There is no report-level ``judge``, ``WAIT``, ``NEVER``, or eligibility flag.
Reference validation runs at admission; incomplete groups wait for more valid
samples. Training input violations raise instead of silently filtering reports.

Computed feedback
~~~~~~~~~~~~~~~~~

Use this engine when later traffic completes an earlier record and judging it
calls a model or another slow service. The recipe writes a ``RecordJudging``.
Its record stage is a ``JudgingRecord2Record(context, judging, worker)``,
its record-to-task stage is a ``JudgedTasks``, and its ``Task2Batch`` shapes
the judged samples into batches.

+------------------------------------------+-----------------------------------+
| Hook                                     | Contract                          |
+==========================================+===================================+
| ``RecordJudging.observe(record, now,``   | correlate one record with the     |
| ``stage)``                               | stage's verbs; must not block     |
+------------------------------------------+-----------------------------------+
| ``async RecordJudging.judge(job)``       | slow judgment, on the worker      |
|                                          | thread                            |
+------------------------------------------+-----------------------------------+
| ``RecordJudging.expire(now)``            | optionally return receipts whose  |
|                                          | completion window ended           |
+------------------------------------------+-----------------------------------+
| ``JudgedTasks.make_sample(record,``      | a ``TrajectoryItem``, or ``None`` |
| ``judgment)``                            | to retire the record              |
+------------------------------------------+-----------------------------------+
| ``make_batch(tasks, batch_number)``      | the declared batch type           |
+------------------------------------------+-----------------------------------+
| ``required_request_types``               | optionally restrict which record  |
|                                          | types reach the pipeline          |
+------------------------------------------+-----------------------------------+

Before each ``observe``, the stage retires expired records and absorbs finished
judgments. ``observe`` then uses the stage's ``track(record)``,
``tracked_record(receipt)``, ``dispatch(job)``, ``retire(receipt)``, and
``abandon(receipt)``. A failed judgment or a ``None`` sample retires the record
instead of failing a training step. All correlation state must be
reconstructible from replay.

While no batch is out, the stage retires candidates that are more than
``max_staleness`` versions behind the newest candidate, by record arrival. The
recipe's ``max_staleness`` reaches the stages as
``PipelineContext.max_staleness``. With the default value 0, only candidates of
the newest version stay. Runtime admission still checks each batch against the
serving version.

Task generation
~~~~~~~~~~~~~~~

.. code:: python

   from reef.core.tasks import (
       GeneratedTask, HarborTask, PlayArm, TaskGenerationRequest, TaskGenerator,
       TaskPlayResult, TaskValidationResult,
   )

``TaskGenerator`` is the plugin base class that a deployment names in
``pipeline.record2task.config.generator``. Subclasses implement:

- the class method ``from_config(config, *, scenario, environ)``, which returns
  the generator of one scenario;
- ``async generate(request: TaskGenerationRequest) -> tuple[GeneratedTask, ...]``;
- ``async validate(task: HarborTask) -> TaskValidationResult``.

``async record_results(results: tuple[TaskPlayResult, ...])`` is optional. The
default keeps nothing. The built-in generated-task recipe,
``reef.recipe.generated_tasks:GeneratedTaskRecipe``, calls these methods.
See `Training pipeline <../developer-guide/processors.rst#task-generation-contract>`__
for the round and its failure behavior.

``TaskGenerationRequest(source_records, description, assets=(), asset_files={})``
carries source records and non-empty requirements text.
Each ``AgentRecord`` must have a distinct ID and belong to the same scenario.
Use an empty record tuple for description-only generation.
For local materials, pass a tuple of ``Path`` values in ``assets``.
For materials in memory, use ``asset_files`` to map relative file names to
UTF-8 text. Supplying both fields is an error. The constructor validates
fields without reading files.

Record2dataset accepts the same input through ``DesignerRequest(inputs=request,
...)``. Its HTTP client reads local materials and sends their contents. The
service rebuilds ``TaskGenerationRequest`` from the JSON fields. The generated
task preserves source record IDs in order.

``GeneratedTask(task, measurement_arms=())`` holds one task and the
measurement arms the generator wants played on it. ``PlayArm(name, count,
extra_instruction_files=())`` names an arm, its number of plays, and the task
files shown after the instruction. Arm names are unique for each task.

``TaskPlayResult(task, training_rewards=(), measurement_rewards={}, error="")``
holds the rewards of the training plays and, by arm name, of each measurement
arm. Each tuple holds only the episodes that ran. ``is_played`` is true when
``error`` is empty. A task that was not played has no rewards.

``TaskValidationResult(errors=())`` exposes ``is_valid``. Each error is a
non-empty explanation of a task defect. Failures to execute the checks raise
exceptions instead.

Migration: ``reef.train.processors.TaskGenerationProcessor`` is removed. Move
its ``generate`` and ``validate`` methods to a ``TaskGenerator`` subclass:

- ``generate`` returns a tuple of ``GeneratedTask`` values instead of one task.
- ``validate`` receives the ``HarborTask`` instead of a directory path.
- Import ``TaskValidationResult`` from ``reef.core.tasks``.

Name the class in ``pipeline.record2task.config.generator`` and use the generated-task recipe. The recipe plays and trains, so the processor half of the method is
not needed.

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
``reef.record2task.samples`` to add exact captured tensors. Missing required
tensors fail at the training boundary; these helpers do not invent tokens or
log probabilities from text. Both batch readers reject unsupported item kinds
instead of silently filtering mixed batches.

Completed text-only streaming responses contribute their aggregated
``response.message`` to the ATIF agent step. The original SSE body remains
in the captured record. Matching assistant messages in later request history
are reused as context without duplicating those steps. An explicit
``training.response_message`` takes precedence over the stream summary.

Migration: the former policy/trace sample types and their batch subclasses are
removed. Construct ATIF items directly and return ``TrainingBatch(id, items)``.
Use ``group_id`` on each member for grouped batches. Read captured tensors from
``item.training`` and rewards through ``trajectory_reward(item)``.
``output_schema`` defaults to ``TrainingBatch``. Keep batches serializable, with
no handles to services, files, threads or models.

Training objective
------------------

Each method defines a ``TrainingObjective`` in its own ``objective.py``. It
owns full-batch signal preparation and declares the backend loss family.
Recipes bind it with ``WeightTrainingSpec(objective="my_method.objective:MyObjective",
scheduling=StepScheduling(...))``. The dotted reference names a class with a zero-argument
constructor or an existing instance; its module must be importable in both the
service and training processes.

.. code:: python

   from reef.train.algos import TrainingObjective, StepSignal
   from reef.train.algos.helpers import next_steps
   from reef.train.types import TrainingBatch, trajectories
   from reef.core.trajectories import trajectory_reward

   class MyObjective(TrainingObjective):
       name = "my_method"
       loss_family = "my_method"

       def prepare(self, batch, state) -> StepSignal:
           samples = trajectories(batch)
           steps = next_steps(state)
           return StepSignal(
               action="train",
               advantages=tuple(trajectory_reward(sample) for sample in samples),
               next_algorithm_state={"steps": steps},
               metrics={"steps": steps},
           )

For a shared short name, decorate the class with ``@register_objective`` from
``reef.train.algos.registry``; importing its package registers it. The same
function accepts an explicit instance. Core never imports method packages.
``WeightTrainingSpec.loss_family`` is derived from the objective, so neither
the recipe nor the signal repeats that choice.

``WeightTrainingSpec.scheduling`` is the recipe's ``StepScheduling``
(``reef.core.batches``): the rollout unit (``comparison_set`` or ``sample``),
rollouts per optimizer step (``configured``, ``actual`` or an int), ``epochs``,
``shuffle`` and ``remainder`` handling. It is deployment configuration rather
than method math, so it lives on the recipe and the runtime carries it to the
backend with the objective reference. An objective declares only what its loss
tolerates: ``supports_multiple_epochs`` (default ``False``) says whether passes
after the first, which train off-policy against log-probs computed once, are
valid, as they are for a clipped ratio. ``validate_scheduling`` rejects
``epochs > 1`` otherwise, at recipe build and again in each backend before
preparation.

``StepSignal`` carries:

- ``action``: ``train`` runs backend training; ``skip`` commits a state-only
  transition.
- ``advantages``: optional per-trajectory values, in batch order.
- ``next_algorithm_state``: proposed state committed through the trainer's
  existing step lifecycle.
- ``metrics``: method telemetry carried to the commit record.

``prepare`` runs on the complete reserved batch before optimizer or worker
partitioning. Compute group-relative statistics here. Keep this entry point
free of torch, Ray, Slime and Tinker imports. It must not mutate the batch,
execute training, or commit state. Model-dependent terms (critic-based GAE,
reference-model KL) stay in the backend implementation after the required
forward passes. Both Slime and Tinker use the same objective preparation;
the selected backend resolves its own loss implementation lazily.

Migration: move the former ``StepPreparer.__call__`` body to
``TrainingObjective.prepare`` in the method's ``objective.py``, declare
``loss_family`` on that class, and remove it from ``StepSignal``. Move the
``StepScheduling`` the preparer returned to the recipe:
``WeightTrainingSpec(step_preparer=..., loss_family=...)`` becomes
``WeightTrainingSpec(objective=..., scheduling=...)``, and ``StepSignal`` no
longer carries ``scheduling``. Declare ``supports_multiple_epochs = True`` on
an objective whose loss is clipped for off-policy passes. Plain function
references are replaced by objective class/instance references. Runtime
preparation's string argument is now named ``objective`` and is followed by
the recipe's ``scheduling`` in every ``prepare_training_step`` signature,
including the coordinator RPC. Update custom runtimes and upgrade coordinators
and workers together. Experiment backend metadata now uses ``objective`` and
``scheduling`` instead of ``step_preparer``; committed algorithm state,
training payloads, checkpoints and artifact formats are unchanged.

Harness method
--------------

A harness-evolution method fills three slots with plugin classes. Reef runs
the loop around them: snapshot, apply, run the paired episodes, record,
publish or revert.

.. code:: python

   from reef.evaluation import SelectionPlugin
   from reef.evaluation.harness import EpisodeScorer
   from reef.train.cordis_backend import Proposer

   class MyProposer(Proposer):
       def propose(self, nodes, samples, models, *, manifest=None, rejected=(), sources=(),
                   requests=(), entries=(), agent_host=None) -> Mutation | Sequence[Mutation] | StepProposal | None: ...

   class MyScorer(EpisodeScorer):
       def score(self, task, result) -> float: ...

   class MySelection(SelectionPlugin):  # optional
       @classmethod
       def from_config(cls, config, *, candidate_backend) -> "MySelection": ...
       def evaluate(self, candidate) -> EvaluationResult: ...
       def decide(self, candidate, evaluation) -> SelectionDecision: ...

A deployment names each class as ``package.module:ClassName``, or as a mapping
with ``module`` and ``config``. Reef builds it with the class method
``from_config``. ``Proposer``, ``EpisodeScorer`` and ``Promoter`` inherit
``reef.core.plugins.ConfiguredPlugin``, whose default ``from_config(config)``
builds the class with no arguments and refuses a non-empty ``config``.

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
| ``manifest``           | keyword-only: the previous step's ``FailureManifest``,   |
|                        | the per-task record of which episodes failed and how;    |
|                        | ``None`` before the first settled step                   |
+------------------------+----------------------------------------------------------+
| ``reads_requests``     | property, ``False`` by default: ``True`` when the        |
|                        | proposer answers the ``requests`` of a manual step       |
+------------------------+----------------------------------------------------------+
| ``runs_agent``         | property, ``False`` by default: ``True`` when the        |
|                        | proposer uses ``agent_host`` to run a coding agent       |
+------------------------+----------------------------------------------------------+
| ``entries``            | keyword-only: the tree as entry options,                 |
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

Returning ``None`` from ``propose`` skips the step. Reef passes every keyword
on every call.

Migration:

- Wrap a plain ``propose`` or ``evaluate`` function in a ``Proposer`` or
  ``EpisodeScorer`` subclass. Its ``propose`` or ``score`` method calls the
  function. Name the class in the deployment file.
- Rename ``__call__`` on a ``Proposer``, ``EpisodeScorer`` or ``Promoter``
  subclass to ``propose``, ``score`` or ``promote``.
- A proposer that named the ``requests`` or ``agent_host`` keyword overrides
  ``reads_requests`` or ``runs_agent``.
- Move the body of a ``CandidatePluginFactory`` or
  ``CandidateEvaluationPluginFactory`` to the class method ``from_config`` on
  the plugin. The plugin then inherits ``SelectionPlugin`` or
  ``ConfiguredEvaluationPlugin``. The worked examples are in
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
result was not durably committed. The deployment names the plugin class in
its ``evaluation`` section (`Configuration
<configuration.rst#the-evaluation-section>`__).

A configured plugin inherits ``ConfiguredEvaluationPlugin``. It implements the
class method ``from_config(config, *, runtime, training_runtime, scenario,
environ)``, which returns one scenario-local plugin. Reef checks the class
when it loads the configuration and calls ``from_config`` for each scenario.
Reef rejects functions, abstract classes and objects that only expose
matching methods.

Harness recipes instead name a ``reef.evaluation.SelectionPlugin``
class in ``evolution.selection``. Its ``from_config(config, *,
candidate_backend)`` binds the plugin to the scenario's candidate backend.
``BackendAlwaysSelectPlugin`` and ``reef.evaluation.harness``'s
``ScoreComparisonPlugin`` and ``FloorPlugin`` are the built-in policies
``always``, ``score_comparison`` and ``floor``.

Surface
-------

.. code:: python

   from reef.surface import (
       ComponentSurface, Surface, create_config_surface, create_harness_surface,
       create_skill_surface, create_weight_surface,
   )

A surface binds one frozen release to its consumers.
``WeightTrainingRecipe.build_surface()`` already calls
``create_weight_surface()``, and ``CordisRecipe`` calls
``create_harness_surface()``, so most methods never touch this.

``Surface`` is a frozen dataclass mapping each named release component to a
``ComponentSurface``, whose capabilities are fields, not subclass identity.
``None`` means the capability is absent, and bare ``Surface()`` is the
complete record-only configuration. A one-component surface serves a flat
release; a surface with several components serves a release with one
directory per component, and ``Surface.loader``, ``inference``, and ``files``
route to their components.

Migration: code written before components keeps working.
``Surface(loader=..., inference=..., files=...)`` builds a one-component
surface whose component is named ``release``; beside one component in
``components``, those keywords replace that component's fields, as
``dataclasses.replace`` does. A surface of several components refuses them:
set the capabilities on each ``ComponentSurface``.

+---------------+-----------------------------+----------------------------------------------+
| Field         | Type                        | Contract                                     |
+===============+=============================+==============================================+
| ``validator`` | ``ArtifactValidator``       | admit the component before publication and   |
|               |                             | rollback; the default accepts any artifact   |
+---------------+-----------------------------+----------------------------------------------+
| ``loader``    | ``ArtifactLoader | None``   | recover the serving head, load rollback      |
|               |                             | checkpoints                                  |
+---------------+-----------------------------+----------------------------------------------+
| ``inference`` | ``InferenceHooks | None``   | prepare provider requests, verify responses  |
+---------------+-----------------------------+----------------------------------------------+
| ``files``     | ``FileTree | None``         | back client pulls                            |
+---------------+-----------------------------+----------------------------------------------+

Two optional protocols extend those structurally, and the scenario checks for
them with ``isinstance``. ``ArtifactActivator`` adds ``loader.activate(artifact,
runtime, source=...)``, making a version servable once it is final.
``LeasingInferenceHooks`` adds ``inference.begin_request(artifact, path)``,
returning a lease the service releases when the attempt ends, so serving state
such as a resident adapter stays protected for its duration.

A surface does not decide which records train, compute candidates, execute a
training job, or mutate the release chain. Native streaming
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
  serving weights without restoring optimizer state. Its ``model_path`` property
  supplies the default model name for harness bindings; the base implementation
  returns an empty string for runtimes whose requests select the model.
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
