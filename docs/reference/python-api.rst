Python API
==========

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
| bind a scenario's behavior                  | `Recipe <#recipe>`__                             | every recipe                 |
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
|                                             | <../developer-guide/loss-families.rst>`__                           |                              |
+---------------------------------------------+--------------------------------------------------+------------------------------+

Method code should depend only on what this page documents. Anything else under
``reef.`` is an implementation detail and may change.

Recipe
------

.. code:: python

   from reef.recipe import Recipe, WeightTrainingRecipe, config_field
   from reef.harness_evolve import HarnessEvolveRecipe

A recipe is one frozen dataclass binding a scenario to its serving and evolution
behavior.

.. code:: text

   Recipe                       record-only by default   reef.recipe
   ├── WeightTrainingRecipe     step preparer, loss family, TrainingRuntime
   │   ├── SAORecipe                                        reef.sao
   │   ├── TTTDRecipe                                       reef.tttd
   │   └── OpenClawRLRecipe                                 reef.openclawrl
   └── HarnessEvolveRecipe      harness surface + backend   reef.harness_evolve

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
| use Reef's harness loop         | configure ``HarnessEvolveRecipe``; subclass it only  |
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
| ``build_surface(scenario)``                       | ``Surface``                 | the delivery boundary for one  |
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

Weight-training recipes add ``training_spec()``, which binds the processor, the
registered or dotted step preparer, and the backend loss family;
``report_type``, the declared ``ReportBase`` subclass; ``max_staleness``, the
accepted producing-to-serving version lag, which must match the runtime; and
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
``from_environment()`` builds the recipe; ``service_config()`` forwards declared
fields and shared artifact settings from the service configuration. Override
``processor_config()`` when a processor needs renamed or derived keys. Never
read deployment YAML from inside a processor.

Report
------

.. code:: python

   from reef.core.reports import ReportBase, ReportValidationError
   from reef.core.reports import GroupedRolloutReport, ScoredRolloutReport   # bundled contracts

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

Every processor gets the scenario's experiment logger as
``self.experiment_logger``. Log finite numeric metrics under the ``processor``
namespace; processor code never imports W&B, and the logger is a no-op when
tracking is off.

Reported feedback
~~~~~~~~~~~~~~~~~

.. code:: mermaid

   sequenceDiagram
       accTitle: How the reported-feedback processor turns records into a typed batch
       participant Reef as Reported-feedback processor
       participant Method as Method processor
       Reef->>Method: judge(ReportContext)
       alt referenced inference is missing
           Method-->>Reef: WAIT — park this report
       else report can never train
           Method-->>Reef: NEVER — release this report
       else report is accepted
           Method-->>Reef: ReportDecision.train(value, ...)
           opt group_key supplied
               Reef->>Method: decide_group(key, candidates)
               Method-->>Reef: GroupDecision<br/>INCOMPLETE · READY · DISCARD
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
``weight_version``, ``action_mask``, ``rollout_created_at``, ``turn_count``,
``topk_indices`` / ``topk_log_probs``, ``weight_version_spans``, and ``extras``,
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
|                          | ``batch_size`` is ``configured`` or ``actual``  |
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

A surface binds one frozen artifact version to its consumers.
``WeightTrainingRecipe.build_surface()`` already calls
``create_weight_surface()``, and ``HarnessEvolveRecipe`` calls
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
training job, admit an artifact, or mutate the version chain. Artifact admission
is separate, through ``Recipe.build_artifact_validator()``. Native streaming
behavior stays unchanged. A method should not add an HTTP proxy or copy Reef's
record store.
