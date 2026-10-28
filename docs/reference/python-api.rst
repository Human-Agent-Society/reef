Python API for custom learning methods
======================================

The Python API is the set of extension points a learning method plugs into. Reef
owns everything around them: accepting and replaying records, holding a job's data
until it is acknowledged, committing algorithm state, submitting training jobs,
and publishing the selected checkpoints.

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
| turn records into typed job data            | `Pipeline stages <#pipeline-stages>`__           | every method producing       |
|                                             |                                                  | updates                      |
+---------------------------------------------+--------------------------------------------------+------------------------------+
| carry data to the backend                   | `Job data <#job-data>`__                         | every method producing       |
|                                             |                                                  | updates; subclass only for a |
|                                             |                                                  | new shape                    |
+---------------------------------------------+--------------------------------------------------+------------------------------+
| turn job data into a signal                 | `Training objective <#training-objective>`__     | weight-training methods      |
+---------------------------------------------+--------------------------------------------------+------------------------------+
| deliver a new artifact medium               | `Surface <#surface>`__                           | only a new evolution medium  |
+---------------------------------------------+--------------------------------------------------+------------------------------+
| propose a harness edit and grade a rollout  | `Harness method <#harness-method>`__             | harness-evolution methods    |
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

Training and inference are separate contracts. Neither holds the other, and
there is no aggregate runtime:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Layer
     - Training
     - Inference
   * - Reef contract
     - ``TrainingComponent``
     - ``InferenceRuntime``
   * - Native integration
     - the component's own framework (Slime, Tinker)
     - ``InferenceBackend``

.. code:: python

   from reef.runtime.interfaces import InferenceBackend, InferenceRuntime
   from reef.runtime.jobs import JobStatus, SavedCheckpoint, TrainingComponent, TrainingJob

A ``TrainingComponent`` runs bounded training jobs (see `Training jobs and
components`_). An ``InferenceRuntime`` executes requests, owns admission, and
loads published releases into its engines (see `Runtime responsibilities`_).
``InferenceBackend`` controls the serving engines of one deployment: it loads
``WeightFiles`` into every engine (``load_weight_files``), pauses and resumes
generation, unloads adapters, and reports the runtime load IDs that the engines
serve. ``InferenceEngine`` names the operations one engine actor offers, which
``ReefSGLangEngine`` and ``ReefVLLMEngine`` implement. The engine supervision
hooks an inference integration also implements (``InferenceEngineGroup``,
``InferenceMonitor``, ``EngineHealthChecks``, ``EngineHealthTarget``) live in
``reef/runtime/recovery.py``, next to the objects that drive them.
Implementations inherit the corresponding interface and provide every abstract
operation. SGLang's ``SGLangInferenceBackend`` inherits ``InferenceBackend``
through ``EngineInferenceBackend``. ``InferenceService`` in
``reef/runtime/deployment.py`` owns engine start and shutdown. It also requires
explicit inheritance; inference factory discovery rejects objects that only
happen to expose similarly named methods.

Two other extension points have separate names and purposes:

- ``reef.train.CandidateBackend`` prepares, evaluates and settles harness
  updates; ``Trainer`` accepts it as ``candidate_backend``. A weight trainer
  takes a ``JobLoop`` as ``job_loop`` instead (see `Training jobs and
  components`_); a trainer takes one or the other, not both.
  ``Trainer.build(..., default_job_samples=...)`` takes the default
  ``job_samples`` of the sample-to-job stage: a harness recipe passes
  ``evolution.batch_size``, and a weight recipe passes none, so its
  deployment sets ``pipeline.sample2job.config.job_samples``.
- ``reef.runtime.interfaces.InferenceHandler`` executes one buffered or streaming
  request. Runtime and recipe objects expose it as ``inference_handler``;
  ``create_app`` and ``build_app`` accept the same keyword for an injected
  handler (see `Embed the service`_).

The deployment selects native backends with ``training.backend`` and ``inference.backend``.
Custom request adapters use ``inference.handler-factory`` and
``inference.handler-config``. The factory path names an ``InferenceHandler``
subclass. Reef calls its ``from_config(upstream_url, *, model_path, timeout_s,
**config)`` class method; arbitrary functions are not accepted. Handlers injected
directly into ``create_app`` or ``build_app`` need only implement the inference methods.

Streaming handlers return ``InferenceStream``; its ``record_response`` and
``record_response_pending`` fields carry buffered or deferred recording state.
Implementations and test doubles must initialize these fields through the
stream constructor. SGLang handler subclasses that preserve raw reasoning tags
can set ``SPLIT_REASONING = False`` (renamed from ``_SPLIT_REASONING``).

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
   from reef.recipe.sft import SFTRecipe

A recipe is one frozen dataclass configuring the serving and evolution behavior
for every scenario in a deployment.

.. code:: text

   Recipe                       record-only by default   reef.recipe
   ├── WeightTrainingRecipe     training component, inference runtime, objective
   │   ├── SFTRecipe                                        reef.recipe.sft
   │   ├── SAORecipe                                        recipes.sao.recipe
   │   ├── TTTDRecipe                                       recipes.tttd.recipe
   │   ├── SDFTRecipe                                       recipes.sdft.recipe
   │   └── OpenClawRLRecipe                                 recipes.openclawrl.recipe
   └── CordisRecipe             harness tree + rollouts  reef.recipe.cordis
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
| ``training_component``                            | ``TrainingComponent`` or    | required training component    |
|                                                   | ``None``                    | for ``WeightTrainingRecipe``;  |
|                                                   |                             | its first positional argument  |
+---------------------------------------------------+-----------------------------+--------------------------------+
| ``build(scenario, records, algorithm_state=...,`` | ``Trainer``                 | construct the scenario trainer |
| ``task_storage=...)``                             |                             | and give ``task_storage`` to   |
|                                                   |                             | ``Trainer.build``              |
+---------------------------------------------------+-----------------------------+--------------------------------+
| ``build_trainers(scenario, records,``             | ``tuple[ComponentTrainer,   | one trainer per release        |
| ``surface=..., algorithm_states=...,``            | ...]``                      | component; defaults to the     |
| ``task_storage=...)``                             |                             |                                |
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
reports parse as (``None`` keeps ingress open). It may also declare
``harness_adapter``, the adapter a client installs its harness tree with.
``None``, the default, is for a recipe with no harness tree;
``CordisRecipe`` names ``evolution.adapter``. The scenario list names the
adapter for every scenario. Every recipe declares ``pipeline_spec()``, the
default class of each pipeline stage (see `Pipeline stages
<#pipeline-stages>`__). Weight-training recipes add these members:

- ``training_spec()`` binds the registered or dotted training objective.
  The objective declares the backend loss family.
- ``job_plan()`` returns the ``JobPlan`` that chooses the parameters of each
  training job (see `Training objective`_).
- ``candidate_evaluation`` is the optional plugin that the deployment's
  ``evaluation`` section configures.

.. code:: python

   @dataclass(frozen=True, kw_only=True)
   class MyMethodRecipe(WeightTrainingRecipe):
       name: str = "my_method"

       @property
       def report_type(self) -> type[MyMethodReport]:
           return MyMethodReport

       @classmethod
       def training_spec(cls) -> WeightTrainingSpec:
           return WeightTrainingSpec(objective="my_method.objective:MyObjective")

       @classmethod
       def pipeline_spec(cls) -> PipelineSpec:
           return PipelineSpec(
               record2record=MyMethodRecord2Record,
               record2sample=MyMethodTasks,
               sample2job=MyMethodJobData,
           )

``frozen=True`` is required by the base. ``kw_only=True`` keeps later fields
keyword-only while the training component stays the positional dependency. Call
``super().__post_init__()`` first when adding validation.

Configuration
~~~~~~~~~~~~~

``config_field()`` is a ``dataclasses.field`` carrying a default, a type-aware
parser, and an optional environment fallback:

.. code:: python

   clip_ratio: float = config_field(0.2, env="REEF_MY_METHOD_CLIP_RATIO")

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

Embed the service
-----------------

An application can run the complete Reef service in its own process.
``reef.service.assembly.build_app`` makes the dispatcher and the HTTP app from
a ``ServiceConfig``. The app includes record retention, inference retries,
the evaluation token, readiness of local cycles, and cleanup.

.. code:: python

   from aiohttp import web

   from reef.service.app import get_dispatcher
   from reef.service.assembly import build_app

   app = build_app(settings, inference_handler=handler)

   async def scenario_count(request: web.Request) -> web.Response:
       dispatcher = get_dispatcher(request.app)
       return web.json_response({"scenarios": len(dispatcher.list_scenarios())})

   app.router.add_get("/my-app/scenario-count", scenario_count)
   web.run_app(app, host=settings.host, port=settings.port)

- ``inference_handler`` is optional. If you give one, it runs every
  inference request in place of the recipe runtime's handler.
- Add middleware and routes before the app starts. The service tokens also
  protect the routes that you add.
- ``get_dispatcher(app)`` returns the dispatcher of an app that ``build_app``
  or ``create_app`` made.
- ``build_app`` owns the dispatcher that it makes. The app closes it on
  cleanup. ``create_app(dispatcher, ...)`` closes the dispatcher only if you
  set ``close_dispatcher=True``.
- You own the inference handler that you give. Reef does not close it.
  Release it after the app stops.
- If ``build_app`` fails, it closes the resources that it made and raises the
  original error.

Scenario stores
---------------

``RecordStore`` in ``reef.storage.records`` is the abstract base for appending,
reading and auditing records, and for recording consumption receipts. ``ScenarioStore`` composes a ``RecordStore`` with committed
scenario state, settling each step and its record progress together. Recipes
and trainers use the record interface; ``Scenario`` coordinates training and
artifact publication through its supplied scenario store.

``RecordStore.append_many(items)`` atomically appends an ordered batch from one
scenario and returns an ``AppendResult`` for each input. Conflicts roll back the
batch; identical retries and evicted records follow ``append_result`` semantics.
SQLite and PostgreSQL implement this with one write transaction. Other adapters
must implement atomic batch append to support ``POST /reef/records/batch``;
the default raises ``NotImplementedError`` before writing. Single-record
callers remain compatible with existing adapters.

Storage implementations explicitly subclass ``RecordStore``,
``ScenarioStore``, and ``ScenarioStorage`` and override their abstract
methods and properties. Incomplete subclasses cannot be instantiated. The
bundled ``SQLiteRecordStore``, ``CommitHistoryScenarioStore``, and
``SQLiteScenarioStorage`` inherit those bases, respectively.

.. code:: python

   from pathlib import Path

   from reef.dispatcher import Dispatcher
   from reef.storage.sqlite import SQLiteScenarioStorage

   storage = SQLiteScenarioStorage(Path(".reef/agent-record"))
   dispatcher = Dispatcher(
       recipe,
       repository_backend_factory,
       scenario_storage=storage,
   )

``Dispatcher`` requires ``scenario_storage`` and passes it to scenario
creation. It never selects or imports a record storage implementation. The
``build_default_dispatcher`` convenience helper also requires this argument.
Deployment assembly builds the storage that ``storage.record_backend`` names:
SQLite, PostgreSQL, or a ``ScenarioStorage`` plugin; embedded callers supply
their chosen storage service explicitly. For ephemeral records, commits and
settings, pass ``SQLiteScenarioStorage()`` without a directory. With a
directory, each scenario has one SQLite database file for its records,
commits, settings, tasks and uploaded trajectories.

The storage service keeps all per-scenario state. This includes the
``ScenarioSettings`` of each scenario: the model configuration and the
selected training mode. The scenario registry uses ``read_settings`` and
``write_settings``. The per-scenario state also includes the published
Harbor tasks of its task stores (see `Task stores`_). ``archive``
retires the settings and tasks with the records and commits.

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
it validates the value and recipe compatibility and writes the scenario
settings through the storage service before replacing the active runtime.

Direct ``Scenario`` construction requires ``store=...``. Replace the former
``records=...`` and ``commit_log=...`` arguments with one store opened by a
``ScenarioStorage``. The supplied trainer must use that store's ``records``.
``Scenario`` does not choose a storage backend. Read committed history through
``scenario.store.history()`` and persistence capability through
``scenario.store.durable``; the former ``scenario.commit_log`` property has been
removed. ``CommitHistoryScenarioStore.commit_history`` is for backend diagnostics.

The abstract bases ``ScenarioStore`` and ``ScenarioStorage`` are exported from
``reef.storage.scenario``, alongside ``ScenarioSettings`` and ``ScenarioStoreConflict``.
Concrete storage classes live under ``reef.storage``:
``commit_history.CommitHistoryScenarioStore`` accepts a ``RecordStore``, a
``TaskStorage`` (``tasks=``), and a ``CommitHistory``. The bundled record
stores give their ``TaskStorage`` from ``task_storage()`` and their
``sql_commits.SQLCommitHistory`` from ``commit_history()``; the commits are a
``scenario_commit`` table in the same database as the records. ``sqlite.SQLiteScenarioStorage`` and
``postgres.PostgresScenarioStorage`` assemble their respective backends. The
scenario package depends only on storage contracts. The storage contract is:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Member
     - Contract
   * - ``from_config(config, *, directory, task_files)``
     - Class method that builds the storage a deployment names in
       ``storage.record_backend``. ``config`` is the plugin's own mapping;
       ``directory`` is ``storage.agent_record_dir``. ``task_files`` is the
       ``TaskFileStore`` of the deployment (see `Task file stores`_). The
       storage owns it: ``archive`` and ``close`` also apply to it.
   * - ``durable``
     - Whether new sessions and settings survive a process restart.
   * - ``open(scenario)``
     - Open an independently owned session (the ``ScenarioStore`` below).
   * - ``read_settings(scenario)``
     - The scenario's ``ScenarioSettings``; defaults for a scenario with none.
   * - ``write_settings(scenario, settings)``
     - Replace the settings durably. The caller serializes writers of one scenario.
   * - ``archive(scenario)``
     - Retire the scenario's records, commits, tasks and settings, so a
       recreated scenario starts empty; return the archive locations.
   * - ``prune(max_bytes=...)``
     - Evict the oldest record bodies over the byte budget.
   * - ``close()``
     - Release shared resources; repeated calls are safe.

The session contract is:

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
   * - ``tasks``
     - The ``TaskStorage`` that opens the task stores of the
       scenario (see `Task stores`_). The stores close with the
       session and survive restarts when ``durable``.
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

``CommitHistoryScenarioStore`` serializes writers through ``CommitHistory.locked()``.
``SQLCommitHistory`` holds one write transaction of the record store, which
SQLite locks with ``BEGIN IMMEDIATE`` and PostgreSQL with a row lock on the
store generation. It reads the history and inserts the commit row in that
transaction. The commit is durable when the transaction ends; an exception
before then stores no commit. Direct writes through a ``CommitHistory``
bypass this store contract.
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
``prune(max_bytes=...)``, and ``close()``. Archive retires that scenario's
stored state so its name can be reused; prune applies retained-body limits to
active and archived state. Custom backends implement those operations instead
of exposing local file paths. Each opened session belongs to its scenario and
closes after its trainer, including on construction or recovery failure. The
dispatcher closes its storage service on shutdown. Application assembly should use
the scenario storage service; direct record store construction is described below.

Task stores
~~~~~~~~~~~

A ``TaskStore`` holds published Harbor tasks, each in the ``train`` or
``eval`` split. A record-to-sample stage that generates tasks saves them in a
task store before any rollout. Thus generation and rollout can recover
independently after a restart. A caller can also add a task directory with
``add_task``. The contract and its values are in ``reef.storage.tasks``.

To open a store, call ``task_storage.open(name)``. Each record-to-sample stage
uses its own ``name``.

A pipeline stage gets ``task_storage`` from ``PipelineContext.task_storage``.
The scenario factory gives ``session.tasks`` to ``Recipe.build_trainers``,
which gives it to ``Trainer.build``. A recipe that overrides ``build`` must
give ``task_storage`` to ``Trainer.build`` too. A composite recipe gives each
component a ``TaskStorage`` that opens ``<component>/<name>``. Thus two
components never share a store, except ``imported-tasks``
(``reef.storage.tasks.IMPORTED_TASK_STORE``): the scenario's imported tasks,
which every component opens by that name and only the service writes.

A store holds generations. A generation holds its source record ids and its
candidates. ``save_candidates`` saves a copy of each candidate task directory
in the task file store. Each ``TaskCandidate`` holds the task ``name``, the
``files_digest`` of its directory, and its outcome:

- A published candidate has a ``split``, ``train`` or ``eval``. The stage
  chooses the split, and the store keeps it.
- A rejected candidate has an ``error`` that gives the reason.
- A pending candidate has neither. After a restart, the stage validates the
  candidates in ``GenerationState.pending_indices``.

A task is a Harbor task directory. The database keeps the state of the store
and the digest of each task directory. The task file store keeps the files,
text or binary, under that digest. ``publish_task`` records the publication of
a saved candidate. ``add_task`` saves a task directory, such as one with
images, and then records its publication. A ``StoredTask`` holds the task
``name``, its ``split`` and its ``files_digest``.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Member
     - Contract
   * - ``begin_generation(generation=..., source_record_ids=...)``
     - Create the checkpoint of a generation. A retry with the same source
       record ids returns the checkpoint.
   * - ``save_candidates(generation=..., directories=...)``
     - Save a copy of each candidate task directory once, before validation or
       rollouts. The name of a candidate is the name of its directory.
   * - ``publish_task(generation=..., index=..., split=...)``
     - Publish one candidate and return its ``StoredTask``. Each task name is
       published once. A later candidate with the same files must give the
       same split. It then returns the published task.
   * - ``add_task(directory=..., split=...)``
     - Publish a Harbor task directory under its directory name, and return
       its ``StoredTask``. The caller checks that the directory is a Harbor
       task. A retry with the same files and split returns the published task.
   * - ``reject_task(generation=..., index=..., reason=...)``
     - Save the rejection of one candidate.
   * - ``complete_generation(generation=...)``
     - Mark the generation complete. Each candidate must have an outcome.
   * - ``generation(generation)``, ``generations()``
     - Return one checkpoint or all checkpoints.
   * - ``tasks(split=None)``
     - Return the published tasks in publication order. Give a ``split`` to
       return only the tasks of that split.
   * - ``path(files_digest)``
     - Return the local task directory of a candidate or a published task,
       for example for ``read_harbor_task`` or ``TaskRolloutRunner.run``. Do not
       change it.
   * - ``mark_trained(task_names=...)``, ``trained_task_names()``
     - Mark published train tasks as trained, or return the trained ones. The
       rollout of a task pool marks a task once the commit of its training step
       is durable, so a restart rolls out only the others.
   * - ``write_document(key=..., document=...)``, ``documents()``
     - Save a text document under a non-empty key, or return every document by
       key. Each key holds one document; another document under a used key
       raises ``TaskStoreError``. The store does not read a document: the stage
       that writes the store keeps its own state about its tasks here.

Each write is durable when it returns. An identical retry of a write changes
nothing. A write that conflicts with the stored state raises
``TaskStoreError``. If a task name is published with other files or in the
other split, the store raises ``TaskConflict``, a subclass of
``TaskStoreError``. The caller serializes the writers of one store.

SQLite keeps the task tables in the database file of the scenario.
PostgreSQL keeps them in the deployment schema, with a key for each store
generation. ``archive`` retires the task tables and the task files with the
records and commits. Record retention does not delete tasks.

Task file stores
~~~~~~~~~~~~~~~~

A ``TaskFileStore`` (``reef.storage.tasks``) keeps the directories of the
published tasks, addressed by the digest of their files. The digest covers the
directory name, and the path and bytes of each file. Thus a repeated save of
the same files changes nothing. A deployment selects the store in
``storage.task_file_backend`` (see `Choose the task file store
<../user-guide/operate.rst#choose-the-task-file-store>`__).
``LocalTaskFileStore`` is the built-in store. An embedded caller can give a
store to ``SQLiteScenarioStorage(..., task_files=...)`` or
``PostgresScenarioStorage(..., task_files=...)``. Without one, the storage
uses a ``LocalTaskFileStore`` in ``task-files`` under its directory.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Member
     - Contract
   * - ``from_config(config, *, directory)``
     - Class method that builds the store a deployment names. ``directory``
       is ``storage.task_file_dir``. A remote store can use it as a local cache.
   * - ``put(scenario, source)``
     - Save a copy of the task directory ``source`` and return the digest.
       The copy is durable when the method returns. Raise ``TaskStoreError``
       for a symbolic link or a special file.
   * - ``path(scenario, digest)``
     - Return a local directory with the files of ``digest``. The directory
       has the name of the saved directory. Raise ``TaskStoreError`` if the
       store does not hold the digest.
   * - ``archive(scenario)``
     - Retire the files of the scenario and return the archive locations.
   * - ``close()``
     - Release the resources of the store. A second call does nothing.

Artifact backends
-----------------

An artifact backend stores the releases of each scenario. By default, Reef
uses Git LFS. A deployment selects a different backend in
``storage.artifact_backend`` (see `Choose the artifact backend
<../user-guide/operate.rst#choose-the-artifact-backend>`__).

A backend plugin is a subclass of
``reef.artifact.ConfigurableRepositoryBackendFactory``. Calling the factory
with a scenario name returns the ``RepositoryBackend`` of that scenario. The
plugin implements one class method:

.. code:: python

   @classmethod
   def from_config(cls, config, *, context): ...

- ``config`` is the plugin's own ``config`` mapping. Raise ``ValueError`` for
  an invalid value.
- ``context`` is a ``RepositoryBackendContext``. ``work_dir`` and ``cache_dir``
  are ``storage.artifact_work_dir`` and ``storage.artifact_cache_dir``.
  ``bootstrap_files`` are the files of the recipe's base artifact. A new
  scenario starts from these files. ``bootstrap_component`` is the release
  component of a bootstrap model snapshot, or ``None`` for the release root.

The optional capabilities are separate base classes. Inherit only the ones
that the backend supports:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Base class
     - Capability
   * - ``RegistrationAwareRepositoryBackendFactory``
     - ``has_registration(scenario)``: find a stored scenario without making
       its backend.
   * - ``EnumerableRepositoryBackendFactory``
     - ``list_registrations()``: list the stored scenarios. The service
       loads them when it starts.
   * - ``CachedRepositoryBackendFactory``
     - Both capabilities, and one cached backend for each scenario.
   * - ``StagedReleaseRepositoryBackend``
     - The backend side: ``publish(advance_head=False)`` stores a release
       without moving the head, and ``commit_release`` moves the head
       after the scenario commit. A durable scenario store requires it.

``tests/artifact/test_reef_artifacts.py`` and
``tests/artifact/test_reef_git_lfs.py`` show the publication and recovery
behavior that a backend must have. ``GitLFSRepositoryBackendFactory`` is the
built-in plugin. Reef does not call a close method on a backend factory.

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
``RecordTables.settings`` holds the store's settings as ``name`` and
``value_json`` rows under the scope keys; ``SQLRecordStore.read_settings()`` and
``write_settings(settings)`` read and replace them.

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
``PostgresScenarioStorage(url, directory, schema=...)`` keeps records, commits
(the ``scenario_commit`` table) and settings (the ``scenario_setting`` table) in
the schema. It uses ``directory`` only to import the model configuration
files of earlier versions on start. Pass it through
``Dispatcher(..., scenario_storage=...)`` for embedded use. Scenario code
continues to depend only on the abstract store contracts. Deployment configuration
selects this storage service with ``storage.record_backend: postgres``; see
`Choose the storage backend <../user-guide/operate.rst#choose-the-storage-backend>`__
for connection and storage setup.

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
``max_bytes`` capacity budget. Store factories apply capacity limits through
``prune`` independently of record consumption.

Training commits persist consumption progress without changing record
visibility or deleting bodies. Processors expose ``releasable_record_ids()``
and ``release_records(ids)`` for memory management. Storage controls capacity
independently of these buffers. An earlier Reef version wrote a consumption
receipt for each stale batch drop. With several trainers in one scenario, such
a receipt carries ``component`` in its metadata, and each trainer skips on
recovery only the rows its own commits and receipts consumed.

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
   * - ``consumption_receipts(scenario)``
     - Ordered receipts with ``receipt_id``, ``consumed_ids``, ``metadata``,
       and ``recorded_at``. Reef no longer writes receipts. A store that an
       earlier version wrote holds them for skipped batches and legacy
       retirement markers.

``StoredRecord`` contains ``sequence`` and ``item`` (the original ``AgentRecord``).
Reading a record does not change a consumer's progress. A missing body may have been purged or never stored;
the read API does not guess which. Use the commit history's per-step
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
``dispatcher.prune_record_archives(RecordRetention(max_bytes=20 * 1024**3))``
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

   retention = RecordRetention(max_bytes=20 * 1024**3)
   storage = SQLiteScenarioStorage(Path(".reef/agent-record"))
   try:
       purged = storage.prune(max_bytes=retention.max_bytes)
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
   from reef.record2sample import JudgedSamples, ReportSamples, SkippedReport
   from reef.sample2job import DistillJobData, Sample2Job

A training pipeline (``reef.train.pipeline.TrainingPipeline``) turns durable
records into typed job data through three stages. A recipe names a default class
for each in ``pipeline_spec()`` (``PipelineSpec(record2record=...,
record2sample=..., sample2job=...)``):

- ``record2record``: the record-to-record stage
  (``reef.record2record``). It decides which records train and in what order:
  it groups reports, judges records, and drops records that can no longer
  train, including data more than ``max_staleness`` versions behind the
  serving version. ``max_staleness`` is the stage's own setting, from
  ``Record2RecordSettings`` (see `the staleness check
  <../developer-guide/processors.rst#staleness>`__). Without
  one, the pipeline forms no record set and nothing trains.
- ``record2sample``: the record-to-sample stage
  (``reef.record2sample.Record2Sample``). It turns each record set into samples
  (``Sample`` values). Reported feedback supplies a ``make_sample`` hook
  on ``ReportSamples`` that assembles valid reports; computed feedback supplies
  a ``JudgedSamples`` after an asynchronous ``judge`` derives new feedback.
  The pipeline pushes each record set (``push``) and pulls the results
  (``pull``). Most stages inherit ``ImmediateRecord2Sample`` and implement
  ``make_samples``; a stage with slow work returns a result later (see `slow
  record-to-sample work <../developer-guide/processors.rst#slow-record-to-sample-work>`__).
  ``uploaded_sample(trajectory) -> TrajectorySample`` turns an uploaded ATIF
  trajectory into its training sample. It must be pure, and it raises
  ``ValueError`` to refuse the upload (see `Uploaded trajectories
  <../developer-guide/processors.rst#uploaded-trajectories>`__).
- ``sample2job``: the sample-to-job stage (``reef.sample2job.Sample2Job``).
  Its ``make_job_data(samples, data_id)`` shapes the samples of the selected
  sets into one job's data with the ID ``data_id``. The pipeline assigns the
  ID. Automatic job data gets ``<scenario>:data:<number>``. An instruction's
  data gets ``<scenario>:instruction:<request id>``. The number counts the
  job data of one pipeline. It is not a training step.
  ``Sample2Job.__init__(context, buffer_size, job_samples, *, max_job_samples=None)``
  sets the most samples that wait for a job and the fewest samples of one
  job. A ``job_samples`` of ``None`` takes ``context.default_job_samples``; a
  ``buffer_size`` of ``None`` holds the data of four jobs
  (``DEFAULT_BUFFER_JOBS``). The constructor raises ``PluginConfigError`` for
  a buffer smaller than ``job_samples`` and for a ``job_samples`` above
  ``max_job_samples`` (an SDPO or TTTD grid). A stage with the class
  attribute ``trains_own_job_data_only`` takes neither size.
  ``Trainer.build`` raises ``PluginConfigError`` for a training trainer whose
  other stage has no ``job_samples``. ``Sample2JobSettings`` parses the
  ``buffer_size`` and ``job_samples`` keys, and a stage with more keys
  extends it. See `Job size <../developer-guide/processors.rst#job-size>`__.

Reef builds the stages for each scenario, in this order, with the class
methods below; a later stage receives the stages built before it:

- ``Record2Record.from_config(config, *, context, environ)``
- ``Record2Sample.from_config(config, *, context, record2record, environ)``
- ``Sample2Job.from_config(config, *, context, record2record, record2sample, environ)``

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
they receive. A ``Record2Sample`` that keeps
per-scenario state also implements ``scenario_state_dirs(config, scenario)``,
so a scenario delete archives it. Reef owns replay, retention, deduplication, pending job data,
and exactly-once consumption; the method implements only the hooks below.
They run synchronously on the trainer thread, so they must not block on
network or model latency. Reports must reference existing inference records
in the same scenario.

``TrainingPipeline.training_mode`` selects automatic, instruction-triggered or
combined training. A ``Sample2Job`` declares ``supported_training_modes``. In
``manual`` an instruction's data has no tasks; in ``hybrid`` it takes the sets
that automatic job data would. Record push, acknowledgement, retention, buffer release
and background derivation are shared. See
`Training pipeline <../developer-guide/processors.rst>`__ for the instruction queue and job data contract.

Every stage gets the scenario's experiment logger as
``context.experiment_logger`` (``self.experiment_logger`` on a ``Sample2Job``).
Log finite numeric metrics under the method's namespace; stage code never
imports W&B, and the logger is a no-op when tracking is off.

Reported feedback
~~~~~~~~~~~~~~~~~

A reported-feedback recipe implements these hooks:

.. list-table::
   :header-rows: 1

   * - Hook
     - Contract
   * - ``ReportSamples.make_sample(context) -> Sample | SkippedReport``
     - Assemble one valid report and its resolved inferences; raise on data errors. Return
       ``SkippedReport(reason)`` for a valid report that cannot train; the stage releases it.
   * - ``Sample2Job.make_job_data(samples, data_id) -> JobData``
     - Shape the flat tuple of selected tasks into job data with the ID ``data_id``.

The recipe owns the report policy. Its record stage is a subclass of
``ReportRecord2Record(context, grouping, *, exclusive_sources=False,
ordered_groups=False)``; ``ReportRecord2Record(context, ReportGrouping())``
trains every report by itself. ``ExclusiveReportRecord2Record(context)`` also
trains every report by itself, and a report that ends owns its inference
records. No inference is in two record sets: a report that references an
inference of another report ends without training. A ``ReportGrouping`` runs in the
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

``make_sample`` returns a trajectory or task directly. ``ReportSamples`` adds the
report id and ordered references to ``source_agent_record_ids``. Grouping and
consumption state stay in the stage; recipes never receive buffered report objects.
``placement`` returns ``(None, None)`` by default. A group key collects reports,
and a slot deduplicates retries within that group (None uses the report id).
This collection group differs from ``TrajectorySample.group_id``: TTTD waits for
a complete step, then trains several comparison groups within it.
``make_job_data`` receives samples in group/arrival order. Acknowledgement consumes
all selected reports, even when the recipe omits constant-reward groups from
its output. Reports arriving after reservation remain for later job data.
``output_schema`` declares the job data type. On the stage, ``exclusive_sources``
controls source release for terminal group/duplicate reports, and
``ordered_groups`` orders ready groups by their keys.

There is no report-level ``judge``, ``WAIT``, ``NEVER``, or eligibility flag.
Reference validation runs at admission; incomplete groups wait for more valid
samples. Training input violations raise instead of silently filtering reports.

Computed feedback
~~~~~~~~~~~~~~~~~

Use this engine when later traffic completes an earlier record and judging it
calls a model or another slow service. The recipe writes a ``RecordJudging``.
Its record stage is a ``JudgingRecord2Record(judging, worker)``,
its record-to-sample stage is a ``JudgedSamples``, and its ``Sample2Job`` shapes
the judged samples into job data.

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
| ``JudgedSamples.make_sample(record,``    | a ``TrajectorySample``, or        |
| ``judgment)``                            | ``None`` to retire the record     |
+------------------------------------------+-----------------------------------+
| ``make_job_data(samples, data_id)``      | the declared job data type        |
+------------------------------------------+-----------------------------------+
| ``required_request_types``               | optionally restrict which record  |
|                                          | types reach the pipeline          |
+------------------------------------------+-----------------------------------+
| ``accepts_trajectories``                 | optionally set True to train on   |
|                                          | uploaded ATIF trajectories        |
+------------------------------------------+-----------------------------------+

Before each ``observe``, the stage retires expired records and absorbs finished
judgments. ``observe`` then uses the stage's ``track(record)``,
``tracked_record(receipt)``, ``dispatch(job)``, ``retire(receipt)``, and
``abandon(receipt)``. A failed judgment or a ``None`` sample retires the record
instead of failing a training step. All correlation state must be
reconstructible from replay.

Task generation
~~~~~~~~~~~~~~~

.. code:: python

   from reef.core.tasks import HarborTask
   from reef.recipe.generated_tasks import TaskGenerator, GeneratedTaskResult, TaskValidationResult
   from reef.record2sample.generator import TaskGenerationRequest

``TaskGenerator`` is the plugin base class that a deployment names in
``pipeline.record2sample.config.generator``. Subclasses implement:

- the class method ``from_config(config, *, scenario, environ)``, which returns
  the generator of one scenario;
- ``async generate(request: TaskGenerationRequest) -> tuple[HarborTask, ...]``;
- ``async validate(task: HarborTask) -> TaskValidationResult``.

``async record_results(results: tuple[GeneratedTaskResult, ...], *, generation: int)``
is optional. It receives the results of one earlier ``generate`` call, named by
``TaskGenerationRequest.generation``, possibly after later calls and after a
restart. A second call for the same generation must have no effect. The
default keeps nothing. The built-in generated-task recipe,
``reef.recipe.generated_tasks:GeneratedTaskRecipe``, calls these methods.
See `Training pipeline <../developer-guide/processors.rst#task-generation-contract>`__
for the round and its failure behavior.

``TaskGenerationRequest(source_records, description, assets=(), asset_files={}, generation=0)``
carries source records, non-empty requirements text, and the number of the generation.
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

``GeneratedTaskResult`` is ``RejectedTask(task, reason)``, ``HeldOutTask(task)``,
``RolledOutTask(task, rollout_rewards)`` or ``FailedTask(task, error,
infrastructure_failure)``. ``rollout_rewards`` is a non-empty tuple of finite
numbers; ``reason`` and ``error`` are non-empty. See `the task generation
contract <../developer-guide/processors.rst#task-generation-contract>`__ for
when the recipe sends each case.

``TaskValidationResult(errors=())`` exposes ``is_valid``. Each error is a
non-empty explanation of a task defect. Failures to execute the checks raise
exceptions instead.

Migration: ``reef.train.processors.TaskGenerationProcessor`` is removed. Move
its ``generate`` and ``validate`` methods to a ``TaskGenerator`` subclass:

- ``generate`` returns a tuple of ``HarborTask`` values instead of one task.
- ``validate`` receives the ``HarborTask`` instead of a directory path.
- Import ``TaskValidationResult`` from ``reef.recipe.generated_tasks``.

Name the class in ``pipeline.record2sample.config.generator`` and use the generated-task recipe. The recipe runs the tasks and trains, so the processor half of the method is
not needed.

Job data
--------

.. code:: python

   from pathlib import Path
   from reef.train.types import (
       TaskSample, Sample, JobData, TrajectorySample,
       trajectories, trajectory_groups,
   )

``Sample = TrajectorySample | TaskSample``. Every processor returns a frozen
``JobData(data_id, samples, request=None)`` with an ordered tuple of samples.
The same job data may contain both kinds. ``request`` is keyword-only and carries
an optional explicit training instruction; request-only job data may be empty.
The pipeline returns the same pending job data until its id is acknowledged.
Reef submits one training job for each value. The training component cuts the
samples into its own optimizer batches.

.. code:: python

   data = JobData(
       "scenario:data:1",
       (
           TrajectorySample(atif_document, source_agent_record_ids=("inference-1", "report-1")),
           TaskSample(Path("tasks/example"), source_agent_record_ids=("record-2",)),
       ),
   )

A ``TrajectorySample.trajectory`` is an
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

``TrajectorySample.group_id`` is an optional string, scoped to the job data.
None means an independent trajectory. Group-relative policy algorithms require
an explicit id on every trajectory and contiguous samples for each group, so
advantages and rows stay aligned. TTTD and CORAL preserve their existing group
barriers and ordering. SAO and OpenClaw-RL explicitly schedule each trajectory
independently, even when group metadata is present. ``trajectory_groups(data)`` reads these comparison groups;
``trajectories(data)`` reads the trajectory samples in order.

A ``TaskSample`` names a `Harbor task directory <https://www.harborframework.com/docs/tasks>`__
with ``task_path: Path``: the directory contains the instruction, environment
and verifier configuration/files. The path is interpreted in the consuming
algorithm's execution environment. The value also accepts optional ``metadata``.
It performs no file I/O or rollout. Algorithms supporting tasks own validation,
environment execution and conversion to trajectories. The Tinker component
rolls task samples out in task jobs; the Slime component and the harness
backends refuse them. Accepting ``JobData`` does not imply that every algorithm
supports both sample kinds.

Both sample kinds may carry ``source_agent_record_ids``. These record derivation
inputs; processor acknowledgement remains authoritative for consumption,
including inputs removed by a recipe's existing group rules.

The ``reef.core.trajectories`` helpers construct and consume ATIF:
``make_trajectory(records, reward, feedback)`` retains original exchanges;
``recorded_payload`` / ``recorded_payloads`` project exchanges for harness
methods, including external ATIF without captured records; and
``trajectory_reward`` reads a finite policy reward. Policy assembly uses
``make_policy_trajectory`` or ``make_multi_turn_policy_trajectory`` from
``reef.record2sample.samples`` to add exact captured tensors. Missing required
tensors fail at the training boundary; these helpers do not invent tokens or
log probabilities from text. Both job data readers reject unsupported sample kinds
instead of silently filtering mixed job data.

Completed text-only streaming responses contribute their aggregated
``response.message`` to the ATIF agent step. The original SSE body remains
in the captured record. Matching assistant messages in later request history
are reused as context without duplicating those steps. An explicit
``training.response_message`` takes precedence over the stream summary.

Migration: the former policy/trace sample types and their batch subclasses are
removed. Construct ATIF trajectory samples directly and return ``JobData(id, samples)``.
Use ``group_id`` on each member for grouped job data. Read captured tensors from
``item.training`` and rewards through ``trajectory_reward(item)``.
``output_schema`` defaults to ``JobData``. Keep job data serializable, with
no handles to services, files, threads or models.

Training objective
------------------

Each method defines a ``TrainingObjective`` in its own ``objective.py``. It
owns signal preparation over a job's full data and declares the backend loss family.
Recipes bind it with ``WeightTrainingSpec(objective="my_method.objective:MyObjective")``.
The dotted reference names a class with a zero-argument
constructor or an existing instance; its module must be importable in both the
service and training processes.

.. code:: python

   from reef.train.algos import TrainingObjective, TrainingSignal
   from reef.train.algos.helpers import next_steps
   from reef.train.types import JobData, trajectories
   from reef.core.trajectories import trajectory_reward

   class MyObjective(TrainingObjective):
       name = "my_method"
       loss_family = "my_method"

       def prepare(self, data, state) -> TrainingSignal:
           samples = trajectories(data)
           steps = next_steps(state)
           return TrainingSignal(
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

The training component cuts each job's data into optimizer steps. The
backend's batch size (Slime's ``--global-batch-size``, Tinker's
``batch_size``) is the rollouts in one step, or the packs in one step when the
job sets ``packing_length``. Two class attributes of the
objective set what the component may do with the job's data:

- ``rollout_unit`` is the rows that one step keeps together. ``"sample"``, the
  default, makes each sample its own rollout. ``"comparison_set"`` keeps the
  samples that share a ``group_id`` in one step.
- ``supports_multiple_epochs`` (default ``False``) says whether passes after
  the first are valid. Those passes train off-policy against log-probs
  computed once, which a clipped ratio tolerates.

A job's optional ``scheduling`` parameter is a ``StepScheduling``
(``reef.train.algos.schedule``) as an object:

.. config::

   epochs | 1 | the passes over the job's data, a positive int
   shuffle | ``false`` | reorder the rollouts in every pass, deterministically from the data's rows
   remainder | ``"partial"`` | a last step with fewer rollouts (or packs) than the batch size: ``"partial"`` trains it as a smaller step, ``"drop"`` leaves it out, ``"error"`` refuses the job's data
   packing_length | ``null`` | pack the rollouts into packs of at most this many tokens (``training["tokens"]`` of each trajectory) before the steps are cut; one step then trains batch-size packs. A pack never splits a rollout, and a longer rollout is a pack of its own. ``shuffle`` and ``remainder`` apply to the packs.

A component reads a job's ``method`` and ``scheduling`` parameters with
``reef.train.algos.registry.job_training(parameters)``. It raises
``ValueError`` when ``method`` is missing or is not an object, when
``scheduling`` has another key or a value out of range, and when the
objective's ``validate_scheduling`` refuses the schedule: ``epochs > 1``
without ``supports_multiple_epochs``. Slime and Tinker call it when they
accept a job and again when they run it.

``reef.train.algos.schedule.pack_rollouts(token_counts, packing_length)`` is
the packing: best-fit decreasing, which puts each rollout, longest first, into
the pack whose free tokens it fills most closely. The packs decide only which
rollouts share a step. The loss and the micro-batches stay the component's.

``WeightTrainingRecipe.job_plan()`` returns the recipe's ``JobPlan``
(``reef.train.jobs``). When a job's data is ready and no job runs, the weight
trainer calls ``next_job(context)``. ``context`` is a ``JobPlanContext`` with
three fields:

- ``data``: the ready job data that the job trains.
- ``finished_jobs``: the number of this trainer's jobs that ended, finished
  or failed.
- ``last_job``: the ``JobOutcome`` (``reef.core.job_data``) of the last job
  that ended, or ``None`` before the first one. A ``JobOutcome`` holds
  ``job_id``, ``backend``, ``event`` (``finished`` or ``failed``),
  ``metrics``, and ``error``, which is set exactly when the job failed.

The plan always returns a ``JobRequest(parameters, restart=False)``. Reef reads
none of the ``parameters``; it submits them to the deployment's training
component, whose name is the job's ``backend``. With ``restart=True``, the job
starts from the component's ``fresh_start`` of the trainer's start. ``TrainingMethod`` (``reef.runtime.interfaces``)
is an objective reference and an optional ``LearningRateSchedule``.

The default ``FixedJobPlan`` submits one job each time job data is ready. Its
parameters are ``method`` (``TrainingMethod(objective).to_dict()``) and the
recipe's ``job_parameters`` config, such as ``scheduling`` or
``save_interval``.

``next_job`` must be a function of its argument. The request is part of the
job's ID, so a retry after a restart must get the same request. See `Write a
recipe <../developer-guide/write-a-recipe.rst#switch-methods-within-a-run>`__
for a plan that changes the method between jobs.

``TrainingSignal`` carries:

- ``action``: ``train`` runs backend training; ``skip`` commits a state-only
  transition.
- ``advantages``: optional per-trajectory values, in the job data's order.
- ``next_algorithm_state``: proposed state committed through the trainer's
  existing step lifecycle.
- ``metrics``: method telemetry carried to the commit record.

``prepare`` runs on the complete reserved job data before optimizer or worker
partitioning. Compute group-relative statistics here. Keep this entry point
free of torch, Ray, Slime and Tinker imports. It must not mutate the job data,
execute training, or commit state. Model-dependent terms (critic-based GAE,
reference-model KL) stay in the backend implementation after the required
forward passes. Both Slime and Tinker use the same objective preparation;
the selected backend resolves its own loss implementation lazily.

Harness method
--------------

A harness-evolution method fills three slots with plugin classes. Reef runs
the loop around them: snapshot, apply, run the paired rollouts, record,
publish or revert.

.. code:: python

   from reef.evaluation import SelectionPlugin
   from reef.evaluation.harness import RolloutScorer
   from reef.train.cordis_backend import Proposer

   class MyProposer(Proposer):
       def propose(self, nodes, samples, models, *, manifest=None, rejected=(), sources=(),
                   requests=(), entries=(), agent_host=None) -> Mutation | Sequence[Mutation] | StepProposal | None: ...

   class MyScorer(RolloutScorer):
       def score(self, task, result) -> float: ...

   class MySelection(SelectionPlugin):  # optional
       @classmethod
       def from_config(cls, config, *, candidate_backend) -> "MySelection": ...
       def evaluate(self, candidate) -> EvaluationResult: ...
       def decide(self, candidate, evaluation) -> SelectionDecision: ...

A deployment names each class as ``package.module:ClassName``, or as a mapping
with ``module`` and ``config``. Reef builds it with the class method
``from_config``. ``Proposer``, ``RolloutScorer`` and ``Promoter`` inherit
``reef.core.plugins.ConfiguredPlugin``, whose default ``from_config(config)``
builds the class with no arguments and refuses a non-empty ``config``.

+------------------------+----------------------------------------------------------+
| Member                 | Contract                                                 |
+========================+==========================================================+
| ``nodes``              | the tree as ``(kind, config)`` pairs                     |
+------------------------+----------------------------------------------------------+
| ``samples``            | the job data's ATIF ``TrajectorySample`` values          |
+------------------------+----------------------------------------------------------+
| ``models``             | the method's only path to a model: ``models.served`` is  |
|                        | the model under test, ``models["teacher"]`` comes from   |
|                        | ``evolution.models``                                     |
+------------------------+----------------------------------------------------------+
| ``manifest``           | keyword-only: the previous step's ``FailureManifest``,   |
|                        | the per-task record of which rollouts failed and how;    |
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
| ``result``             | one finished rollout: exit code, stdout, stderr, and the |
|                        | parsed ``trajectory``                                    |
+------------------------+----------------------------------------------------------+
| ``evaluation.metrics`` | guarantees ``candidate_scores`` and ``current_scores``:  |
|                        | ``RolloutScores`` in JSON, each rollout a number or      |
|                        | ``"failed"``; ``current_scores`` is empty under          |
|                        | ``selection: floor``, which runs the candidate alone     |
+------------------------+----------------------------------------------------------+

Returning ``None`` from ``propose`` skips the step. Reef passes every keyword
on every call.

Migration:

- Wrap a plain ``propose`` or ``evaluate`` function in a ``Proposer`` or
  ``RolloutScorer`` subclass. Its ``propose`` or ``score`` method calls the
  function. Name the class in the deployment file.
- Rename ``__call__`` on a ``Proposer``, ``RolloutScorer`` or ``Promoter``
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
|                                   | ``selected`` is a boolean, and which must     |
|                                   | carry the evaluation it was given             |
+-----------------------------------+-----------------------------------------------+

A rejection leaves the previous artifact serving, and Reef publishes the
candidate as a release held for review (``POST
/reef/scenarios/{scenario}/promote`` serves it). An exception is fail-closed:
Reef does not publish the candidate. Make evaluations
idempotent by ``candidate.candidate_id`` because recovery may repeat work whose
result was not durably committed. The deployment names the plugin class in
its ``evaluation`` section (`Configuration
<configuration.rst#the-evaluation-section>`__).

Without an ``evaluation`` section, a weight recipe uses
``AlwaysSelectCheckpointPlugin`` (``reef.evaluation.evaluators``): it selects
every saved checkpoint and adds no metrics. ``ManualPromotionPlugin`` selects
none, so a person promotes each release. The commit already records the
checkpoint's ``training_metrics``, and it refuses an evaluation or decision
metric whose key is a training metric or ``selected``, ``selection`` or
``training_job``. A training component only trains: it runs no evaluation of
an artifact.

A configured plugin inherits ``ConfiguredEvaluationPlugin``. It implements the
class method ``from_config(config, *, runtime, scenario, environ)``, which
returns one scenario-local plugin. Reef checks the class
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
``create_weight_surface(kind=...)`` with its runtime's ``served_weight_kind``,
and ``CordisRecipe`` calls
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
| ``loader``    | ``ArtifactLoader | None``   | recover the serving head, make published     |
|               |                             | releases servable                            |
+---------------+-----------------------------+----------------------------------------------+
| ``inference`` | ``InferenceHooks | None``   | prepare provider requests, verify responses  |
+---------------+-----------------------------+----------------------------------------------+
| ``files``     | ``FileTree | None``         | back client pulls                            |
+---------------+-----------------------------+----------------------------------------------+

``loader.activate(artifact, runtime, source=...)`` makes a version servable
once it is published, and returns the runtime load ID that serves it. One
optional interface extends the hooks, and the scenario checks for it with
``isinstance``: ``LeasingInferenceHooks`` adds ``inference.begin_request(artifact, path)``,
returning a lease the service releases when the attempt ends, so serving state
such as a resident adapter stays protected for its duration.

A surface does not decide which records train, compute candidates, execute a
training job, or mutate the release chain. Native streaming
behavior stays unchanged. A method should not add an HTTP proxy or copy Reef's
record store.

Runtime responsibilities
------------------------

``TrainingComponent`` and ``InferenceRuntime`` are independent interfaces.
Neither inherits from the other, and no trainer holds a serving engine.

* A ``TrainingComponent`` runs training jobs and saves checkpoints. It cuts
  each job's data into optimizer batches by its own batch size, from its
  training options (Slime's ``--global-batch-size``, Tinker's
  ``batch_size``). Reef does not read that size; the deployment sets the job
  size in ``pipeline.sample2job.config.job_samples``.
* An ``InferenceRuntime`` executes requests, manages admission, and loads
  published releases. It loads a release when a
  scenario starts, when a release is committed, and when one is rolled back.
  Its ``model_path`` property supplies the default model name for harness
  bindings; the base implementation returns an empty string for runtimes whose
  requests select the model. ``per_scenario_adapters`` is true for a runtime
  that serves each scenario its own adapter. ``served_weight_kind`` is the
  kind of the weights that the runtime serves from a release: ``model``, the
  default, for full weights, or ``model_adapter`` for a LoRA adapter on its
  base model. A runtime that serves adapters must override it.
  ``artifact_model(artifact)`` is the ``PolicyModel`` that the runtime serves
  for an artifact on request, without loading it, or None, the default, for a
  runtime that serves only its loaded artifact. A runtime whose handler answers
  each call from the artifact that the call names overrides it, as the Tinker
  runtime does; the `artifact evaluation <http-api.rst#evaluate-an-artifact>`__
  of a model artifact needs it.
  ``shared_by_training_scenarios`` is true for a runtime that serves each
  training scenario its own weights.
* ``ScenarioCommitter`` orders both around each commit. It publishes a
  selected checkpoint, has the runtime serve it
  (``WeightRuntime.activate_checkpoint``), and then writes the commit record.
  A failed publication therefore leaves serving unchanged. For a rollback it
  first imports the release into the training component, then publishes and
  serves the target the same way.

``reef.runtime.serving`` holds the two runtimes for saved weights, both over an
``InferenceBackend``:

* ``AdapterServingRuntime`` serves each scenario its own PEFT adapter on one
  base model. A release holds the adapter in ``adapter/``. At most
  ``adapter_capacity`` adapters stay resident.
  ``reef.runtime.residency.AdapterResidencyManager`` evicts a leaked
  adapter first, then the oldest-activated adapter that is no scenario's
  current one.
* ``FullWeightServingRuntime`` serves one scenario's full Hugging Face
  weights. After it serves a release, it cannot return to the base model,
  because the engines do not keep the base model's files. Roll back to a
  saved release instead.

Two training scenarios can share a deployment only if its training component
allows it (``shares_scenarios``) and its runtime serves each scenario its own
weights. The Tinker component allows it, because each job carries its own
start. The Slime component does not: its workers hold one set of weights, so
a Slime deployment trains one scenario.

Each runtime passes a release's directory as a ``WeightFiles`` value, with
the ``"adapter"`` or ``"full"`` layout, to
``InferenceBackend.load_weight_files``, under a new runtime load ID that every
engine then reports. The runtime kind ``remote_serving``
(``reef.inference.remote.runtime``) builds one of them over the engines of an
inference component, as the release format says: ``AdapterServingRuntime``
for ``adapter`` releases, ``FullWeightServingRuntime`` for ``full`` ones.
``serving_engine_config`` builds the engines' input from the ``inference``
settings and the release format. With a ``service``, each runtime checks its
health every ``health_interval_s`` seconds (default 30). After a failed check
it calls ``recover_engines``: it pauses generation, lets the backend replace
lost or dead engines (``InferenceBackend.recover``, which for an inference
component also covers a restarted component), loads what it serves into them
again (``reload_served``), and resumes.

Weight recipes hold ``training_component: TrainingComponent`` and
``runtime: InferenceRuntime`` separately. Construction is explicit:

.. code:: python

   recipe = SAORecipe(training, runtime=inference)

A training deployment's runtime factory returns the ``TrainingComponent``
alone; a serving factory returns an ``InferenceRuntime``. ``RuntimeFactory``
and ``RuntimeRegistry`` in ``reef.runtime.deployment`` define and build both.
The HTTP service builds the training component first, then the serving
runtime that ``reef.service.deploy.serving.serving_runtime_config`` describes
from the ``inference`` settings and the component's ``release_format``.

``reef.runtime`` is a namespace package without an import facade. Its modules
are ``interfaces`` (inference contracts, ``TrainingMethod``, runtime load
IDs), ``jobs`` (the training component contract), ``serving`` (the
saved-weight serving runtimes), ``residency`` (adapter residency),
``deployment`` (runtime factories and inference service contracts) and
``recovery`` (durable JSON files and engine supervision); ``executor/``
contains worker transports. Concrete integrations live with their owners:
``SlimeTrainingComponent`` in ``reef.train.slime_backend.component``,
``TinkerTrainingComponent`` in ``reef.train.tinker_backend.component``, and
``SGLangInferenceBackend`` in ``reef.inference.sglang.backend``.

Training jobs and components
----------------------------

``reef.runtime.jobs`` defines bounded training jobs. A ``TrainingJob`` has
these fields:

- ``job_id`` and ``backend``, the name of the training component.
- ``data``, a ``JobData``: the data snapshot that the job trains.
- ``start``: the directory of the weights the job starts from, or ``None``
  for the backend's base model.
- ``parameters``: the backend's parameters. Reef reads none of them; they
  override the backend's own configuration.

``job_id_for`` names a job by a hash of its backend, its start, its data and
its parameters. The weight trainer puts the scenario and the records of the
job's data into the data identity, so the ID covers the scenario, the
backend, the start, those records and the parameters. A retry of the same request is
therefore the same job.

A ``TrainingComponent`` runs whole jobs with its framework's own training
loop:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Member
     - Contract
   * - ``name``
     - The backend name that the component's jobs carry.
   * - ``release_format``
     - A ``ReleaseFormat``: the weight files that each saved checkpoint's
       release holds, ``layout`` ``full`` (a Hugging Face checkpoint) or
       ``adapter`` (a PEFT adapter in ``adapter/``, with ``adapter_rank`` and
       ``adapter_target_modules``). ``None`` when the releases hold no weight
       files that an engine loads, such as Tinker's without
       ``export_adapter``. The serving assembly configures the engines from
       it and refuses a serving runtime that cannot load it.
   * - ``submit(job)``
     - Record the job durably and return; the job runs in the background.
       Submitting a job ID again does nothing.
   * - ``status(job_id)``
     - A ``JobStatus``: ``state`` (``running``, ``finished`` or ``failed``),
       the ``SavedCheckpoint`` values so far in step order, the backend's
       ``metrics``, and ``error``, set exactly when the job failed. Raise
       ``KeyError`` for a job the component never saw. A restarted component
       resumes each unfinished job or reports it failed.
   * - ``cancel(job_id)``
     - Stop a running job, which then reports ``failed``.
   * - ``import_start(release)``
     - Copy a published release into the component's own storage and return
       it as a start (``None`` for a release without weights). Reef calls it
       for a rollback. Raise ``ValueError`` when no job can start from those
       weights.
   * - ``fresh_start(start)``
     - Return a start with the weights of ``start`` and no training state: a
       new optimizer state, the learning-rate schedule at step 0, and an
       empty objective state. The same ``start`` always gives the same
       result. Raise ``ValueError`` when the component cannot do this; the
       default refuses. Slime implements it; Tinker does not.
   * - ``close()``
     - Release the component's resources; safe to call repeatedly.

A ``SavedCheckpoint`` has a ``step`` within its job, a ``path`` that a later
job starts from, a ``release_path`` that Reef publishes and serving loads, and
the backend's ``metrics``. A finished job without checkpoints trained nothing,
for example because its objective skipped the job's data.

``reef.train.local_jobs.LocalJobComponent`` is the base of the built-in
components. It keeps one directory per job (``job.json``,
``checkpoints.json``, and ``result.json`` once the job ended). It runs one job
at a time in a worker thread, and it resumes unfinished jobs after a restart.
A subclass implements ``check_job(job)`` and ``run(job, directory)``. It must
call ``start_jobs()`` once it can train. Until then no job runs, including
the jobs that a restart left.

``reef.train.jobs.JobLoop`` is the control plane's side. It holds the weight
trainer's one training component, the recipe's ``JobPlan`` and the candidate
evaluation plugin. It returns the result that the trainer commits:

1. When no job runs, the trainer reserves ready job data. The plan names the
   job, the loop submits it, and the trainer commits the submission, which
   consumes the records of the job's data.
2. While the job runs, each new saved checkpoint becomes a
   ``CheckpointCandidate`` (``candidate_id`` ``"<job_id>:<step>"``). The
   evaluation plugin decides, and the trainer commits the decision. A selected
   checkpoint's ``release_path`` becomes the new release.
3. When the job ends, the trainer commits its outcome.

Training is linear: the next job starts from the newest saved checkpoint,
whether or not it was selected. A restarted job replaces the trainer's start
with its fresh start. A job continues the objective state of its start only
when it trains the same objective. The trainer commits the loop's state as its
algorithm state. This ``JobLoopState`` has four fields:

- ``start``: the start of the next job, or ``None`` for the base model.
- ``job``: the running job as a ``RunningJob``, or ``None``.
- ``finished_jobs``: the number of jobs that ended, finished or failed.
- ``last_job``: the outcome of the last job that ended, or ``None`` before
  the first one.

A failed job ends like a finished one. Reef consumed the records of its data when
it submitted the job, so they do not train again.

Once the commit of a job's end is durable, the trainer calls
``Sample2Job.job_ended(outcome)`` with ``last_job``. The default does nothing.
A trainer that starts with no running job calls it again for the last job
that ended, before any other call to a new stage. The stage takes it as the
end of the last job that it may have handed out.
A stage whose job data holds ``TaskSample`` values (Harbor tasks) reads each
task's rollout results from the outcome's ``metrics``: the Tinker component
reports ``rollout_rewards``, the rewards of each task's rollouts, by task
name. A component that cannot roll out tasks refuses such a job in
``check_job``; Slime does.

For a rollback, Reef first imports the release with ``import_start``. It then
writes the rollback commit and cancels the running job. The next job starts
from the imported release. A component that cannot load a release raises
``ValueError``, and Reef refuses the rollback with HTTP 409.

Tinker integration
------------------

``reef.train.tinker_backend.launch.TinkerDeployment`` implements the optional
``tinker`` backend as an ``InProcessTrainingDeployment``. Its
``component_factory`` builds, from ``TinkerConfig`` and only after deployment
selection, a ``TinkerTrainingComponent``. Each job
trains remotely from the job's start manifest. Its parameters are ``method``,
the optional ``scheduling``, optional ``batch_size``, ``kl_coef`` and
``learning_rate`` overrides, and ``save_interval``, the optimizer steps between saved
checkpoints (by default, one checkpoint when the job ends). A saved checkpoint
is both the start of a later job and the release Reef publishes. A job of
Harbor tasks also takes ``rollout``, the rollouts of each task.

``TinkerDeployment.default_serving`` is ``trainer``: its
``trainer_serving_config`` selects ``reef.inference.tinker.TinkerInferenceRuntime``
(runtime kind ``tinker``), which reads each release's
``tinker-checkpoint.json`` manifest and samples from the immutable sampler it
names. With ``TinkerConfig.export_adapter``, the component also exports each
checkpoint's PEFT adapter into ``adapter/``, and Tinker's adapter as Tinker
saved it into ``tinker-adapter/`` (``TinkerClient.download``); its ``release_format`` is
``ReleaseFormat("adapter", lora_rank, ("all",))``; without it, the release
format is ``None``. Reef's engines can then serve the adapters
(``inference.serving: engines``). Rollback works: the component imports the
release's manifest as a start.

``TrainingDeployment`` (``reef.train.deployment``) is a training backend's
deployment definition. ``default_serving`` (``engines`` unless the backend
says otherwise) is the serving of a deployment that names none and sets no
engine setting, and ``trainer_serving_config(settings)`` the runtime config
of the backend's own serving, or ``None`` without one. ``requires_local_model``
defaults to true; hosted integrations set it to false to preserve remote
model identifiers during deployment resolution.

Methods implement ``TinkerLoss`` from ``reef.train.tinker_backend.losses``
and register it with ``register_loss_family_ref(name, "package.module:CLASS",
backend="tinker")`` from ``reef.train.algos.registry``, the same table that
holds their Slime reference. ``loss_fn`` names the Tinker built-in loss the
family trains with; ``inputs(rows, base_logprobs, kl_coef=...)`` returns that
function's token-aligned ``loss_fn_inputs``, one dictionary per row; and
``needs_base_logprobs`` requests frozen-base scores for a nonzero KL
coefficient. ``check_kl_coef(kl_coef)`` raises ``ValueError`` for a
coefficient the family cannot train; Reef calls it when it prepares a job,
before any remote work. The default accepts every coefficient. For ``importance_sampling``, ``TokenRow.inputs`` builds the
dictionary from response-token advantages, shifting prediction targets by one
token and padding prompt positions with zeros. A ``TinkerCustomLoss`` adds
``loss(data, logprobs)``, which the SDK boundary passes to
``forward_backward_custom`` so the family computes its loss tensor on the
Reef host. ``Datum`` objects are constructed only after this shaping. See
`Train with Tinker <../user-guide/tinker.rst>`__ for supported contracts.
