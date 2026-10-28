Write a recipe
==============

A recipe is the method a deployment runs: how recorded traffic is processed,
how it becomes the data of a training job, what signal that data carries, and whether the
candidate it produces replaces the served version.

`Quickstart <../getting-started/quickstart.rst>`__ defines the term.

Accepting records, replaying them after a restart, holding a job's data until it is
acknowledged, committing algorithm state, submitting training jobs, and
publishing the next version are Reef's job, not the method's.

This page covers a **weight** recipe. For a harness-evolution method with a
proposer, a rollout scorer, and a selection policy against a fixed model, see
`Evolve your harness <../user-guide/evolve-your-harness.rst#write-a-method>`__.

.. code:: mermaid

   flowchart TB
       accTitle: How a recipe collects records and publishes an update
       subgraph COLLECT["1. Collect"]
           direction LR
           REQ["Inference request"]
           REP["Optional report"]
           STORE[("Records")]
           REQ --> STORE
           REP --> STORE
       end
       subgraph RESOLVE["2. Resolve and select"]
           direction LR
           ENG["Resolve reported<br/>or computed feedback"]
           SAMPLE["Make sample"]
           BATCH["Make job data"]
           ENG -->|"resolved feedback"| SAMPLE
           SAMPLE --> BATCH
       end
       subgraph EVOLVE["3. Train and publish"]
           direction LR
           PLAN["Plan job"]
           JOB["Training component<br/>runs the job"]
           EVAL["Evaluate checkpoint"]
           SELECT["Select or reject"]
           VER["New release<br/>serves later requests ↻"]
           PLAN -->|"job data + method"| JOB
           JOB -->|"saved checkpoint"| EVAL
           EVAL --> SELECT
           SELECT -->|"select"| VER
       end
       COLLECT -->|"stored records"| RESOLVE
       RESOLVE -->|"typed job data"| EVOLVE
       class SAMPLE,BATCH,PLAN,EVAL,SELECT user-owned

The shaded steps are the method's. A reported-feedback processor assembles
valid reports and their already stored inference records into samples, then
puts complete units into job data. The recipe's job plan gives the parameters of the
training job for each job's ready data. The training component runs the job: the
method's objective prepares the signal, and the backend trains it and saves
checkpoints. The backend validates the required training fields. A contract
failure raises an error; the backend does not silently drop reports. For each
saved checkpoint, the evaluator measures it and the selector decides whether
to publish it.

Before you write one
--------------------

If an existing recipe's processor, objective, candidate evaluation, and
surface already match your method, change its config instead. Re-read
`Choosing a recipe <../user-guide/recipes.rst>`__.

Build one
---------

A weight recipe is four pieces plus the class that binds them.

.. config::

   training objective | a ``TrainingObjective`` declaring a loss family and implementing ``prepare(data, state)`` to return advantages, metrics, and proposed state in a ``TrainingSignal``. It also declares ``rollout_unit``, the rows that one optimizer step keeps together (``"sample"`` by default, or ``"comparison_set"``), and ``supports_multiple_epochs``, whether its loss accepts more than one pass over a job's data. No torch, Ray, or Slime import.
   pipeline stages | the record stage, the record-to-sample stage, and the sample-to-job stage that turn valid reports and referenced records into samples and typed job data; ``pipeline_spec()`` names them
   report type | the ``ReportBase`` subclass Reef validates at ingress, so a malformed report is HTTP 400 rather than a training-time surprise; a scenario of several components admits a report any of them accepts, and a trainer releases one shaped for another component
   candidate evaluation | measures each checkpoint that a training job saved and decides select or reject. Every recipe carries one; the default, ``AlwaysSelectCheckpointPlugin``, selects every checkpoint
   recipe class | a frozen dataclass whose ``pipeline_spec()`` names the stage classes and whose ``training_spec()`` names the objective (by registered name or dotted class/instance path); ``job_plan()`` returns the ``JobPlan`` that gives each training job's parameters

`Python API <../reference/python-api.rst>`__ is the contract for each.
``recipes/sao/`` is the smallest cookbook implementation and the one to read
alongside this page. Its ``recipe.py``, ``processor.py`` and ``objective.py``
total fewer than 200 lines; the ``slime/`` package holds its loss family.

Every weight recipe takes a ``training_component`` as its first positional
argument and an inference ``runtime`` as a keyword argument. Reef builds both
from the deployment config and passes them in. The training component runs
the recipe's jobs; the runtime serves the weights that Reef publishes.

Configure it
~~~~~~~~~~~~

Keep your module in a package installed in the environment of the Reef
service, and verify the import it will perform:

.. code:: bash

   python -c "from my_pkg.my_method import MyMethodRecipe"

The training component runs in the service's process. Slime's Ray workers get
the service's ``PYTHONPATH``, so they can import a cookbook loss family too.

Copy a weight-training config as described in `Evolve your model
<../user-guide/evolve-your-model.rst>`__ and select the class by dotted path:

.. code:: yaml

   recipe:
     implementation: "my_pkg.my_method:MyMethodRecipe"
   pipeline:
     sample2job:
       config:
         job_samples: ${training.config.global_batch_size}
   training:
     config:
       global_batch_size: 4
     options:
       global-batch-size: "${training.config.global_batch_size}"

This fragment shows only the new keys; keep the inference, storage and training
settings from the config you copied. ``pipeline.sample2job.config.job_samples``
is required: Reef submits a job when it holds this many samples. The batch
size belongs to the training component. For Slime it is
``training.options.global-batch-size``, and Slime cuts each job's data into
optimizer steps of this size. With the same value for both, one job is one
optimizer step (see `Job size <processors.rst#job-size>`__). The recipe does
not configure optimizer steps. Add the Slime flags your
loss family requires (`the mapping <loss-families.rst#family-to-slime-flags>`__).
Reef gets the loss family from the class's ``training_spec()``. Do not repeat
the recipe or its family in the environment. Reef has no global
recipe-implementation registry.

.. code:: bash

   reef serve -c path/to/my-method.yaml

Declare each method setting once with ``config_field``:

.. code:: python

   from dataclasses import dataclass
   from reef.recipe import WeightTrainingRecipe, config_field

   @dataclass(frozen=True)
   class MyMethodRecipe(WeightTrainingRecipe):
       clip_ratio: float = config_field(0.2, env="MY_CLIP_RATIO", help="Bound on the policy ratio.")
       temperature: float = config_field(0.5, allow_nonfinite=False)
       tags: tuple[str, ...] = config_field(())

The shared parser reads annotations, defaults, help and environment fallback
from these declarations. Supported types are ``str``, ``int``, ``float``,
``bool``, ``tuple[str, ...]``, ``Mapping[str, Any]`` / ``dict[str, Any]``, and
optional forms. Object fields can use ``config_field(default_factory=dict)``.
Keep range and cross-field checks in ``__post_init__``. Do not repeat scalar
conversion in the service layer or recipe hooks.

``WeightTrainingRecipe.service_config`` translates the flat deployment layout
and rejects unknown fields; it does not convert scalar values. The service
resolves those fields before it builds the training component and passes them
to ``from_resolved_config``. ``from_environment`` uses that same resolution
path for standalone recipe construction. Custom construction hooks should
consume the resolved values; ``_recipe_kwargs`` owns non-field sections.

Switch methods within a run
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A weight trainer has one training component. By default, ``job_plan()``
returns a ``FixedJobPlan``. It submits one job each time job data is ready.
Every job trains ``training_spec().objective``. Its parameters are ``method``
and the ``recipe.config.job_parameters`` setting, for example a
``scheduling`` or a ``save_interval``. The component's defaults fill the
rest, such as its configured learning rate.

A recipe that trains in phases, such as SFT and then RL or OPD, overrides
``job_plan`` and returns its own ``JobPlan``. Reef calls the plan's
``next_job(context)`` when a job's data is ready and no job runs. The plan returns
a ``JobRequest`` with the job's parameters. The ``method`` parameter is the
``TrainingMethod`` of the job's phase. The same component switches the
objective from one job to the next. The job's ``backend`` is always the
component's name, so the request does not name it.

This plan trains SFT for the first 200 jobs and then the policy objective.
The first policy job restarts training:

.. code:: python

   from reef.recipe import WeightTrainingRecipe
   from reef.runtime.interfaces import LearningRateSchedule, TrainingMethod
   from reef.train.jobs import JobPlan, JobPlanContext, JobRequest

   SUPERVISED_JOBS = 200
   POLICY = LearningRateSchedule("policy", 1e-6, warmup_steps=10)


   class SupervisedThenPolicy(JobPlan):
       def next_job(self, context: JobPlanContext) -> JobRequest:
           if context.finished_jobs < SUPERVISED_JOBS:
               method = TrainingMethod("reef.recipe.sft:SftObjective")
           else:
               method = TrainingMethod("my_pkg.objectives:PolicyObjective", POLICY)
           restart = context.finished_jobs == SUPERVISED_JOBS
           return JobRequest({"method": method.to_dict()}, restart=restart)


   class MyMethodRecipe(WeightTrainingRecipe):
       def job_plan(self) -> JobPlan:
           return SupervisedThenPolicy()

- ``context`` holds the ready job ``data``, ``finished_jobs`` and ``last_job``.
  ``finished_jobs`` counts the jobs that ended, finished or failed.
  ``last_job`` is the outcome of the last job that ended, or ``None`` before
  the first one.
- ``next_job`` always returns a ``JobRequest``. It must depend only on its
  argument. A retry after a restart calls it again, and the request is part
  of the job's identity.
- The component resolves the objective in its own process. Use a dotted
  reference for an objective from another package, such as the built-in
  `SFT <../user-guide/recipes/sft.rst>`__ objective above.
- ``restart=True`` starts the job from the same weights with no training
  state: a new optimizer state, the learning-rate schedule at step 0, and an
  empty objective state. Use it at a switch between phases. Reef asks the
  component for this start with ``TrainingComponent.fresh_start``. Slime
  supports it; Tinker refuses it, and the job is not submitted.
- A request can also give the optional ``scheduling`` parameter, an object
  of ``epochs`` (a positive int, default 1), ``shuffle`` (default ``false``),
  ``remainder`` (``"partial"``, the default, ``"drop"`` or ``"error"``) and
  ``packing_length`` (default ``null``). It sets how the component trains the
  job's data: the passes over it, the rollout order, and what happens to a
  last step with fewer rollouts than the batch size. With
  ``packing_length``, the component first packs the rollouts into packs of
  at most that many tokens, and one step trains batch-size packs. The
  component refuses ``epochs`` above 1 for an objective that does not
  declare ``supports_multiple_epochs``.
- ``save_interval`` (Slime and Tinker) saves a checkpoint after each
  ``save_interval`` optimizer steps of the job.
- The component refuses a method that it cannot train, and the job fails.
  `Per-job families <loss-families.rst#per-job-families>`__ and `Train with
  Tinker <../user-guide/tinker.rst>`__ list what each backend supports.

A ``LearningRateSchedule`` counts optimizer steps. A job that selects the
active schedule continues it, also on a retry or after a restart. A job that
selects another schedule starts that schedule at step 0. A job that selects no
schedule keeps the active one. The component keeps the schedule's progress
with each checkpoint it saves. Starting a schedule does not reset the
optimizer state; only ``restart=True`` does. A job continues the objective
state of the previous job only when it trains the same objective. A job of
another objective starts with an empty objective state. While a schedule is active, job metrics include
``learning_rate`` and ``learning_rate_schedule``.

Evaluate a candidate
~~~~~~~~~~~~~~~~~~~~

Each checkpoint that a training job saves is a candidate, a
``CheckpointCandidate`` (``reef.core.evaluation``). Its ``release_path`` is
the directory that Reef publishes if the candidate is selected, and
``training_metrics`` holds the component's metrics for the checkpoint. The
commit records those metrics beside the evaluation's, so an evaluation adds
only its own metrics. A commit refuses an evaluation metric whose key is
already a training metric. The recipe's ``candidate_evaluation``
decides what happens to it. A weight recipe declares its evaluator in the
deployment config. The top-level ``evaluation`` section serves weight recipes
only. A harness recipe names its plugin in ``evolution.selection`` instead
(see `Write a harness method
<write-a-harness-method.rst#selection-policies>`__):

.. code:: yaml

   evaluation:
     module: my_pkg.evaluation:BenchmarkEvaluator
     config:
       benchmark: gsm8k
       threshold: 0.8

The reference names a ``ConfiguredEvaluationPlugin`` subclass. Reef imports
and checks the class when it loads the recipe config. It then calls the class
method ``from_config`` once per scenario with the opaque ``config``,
``runtime``, ``scenario``, and ``environ``. Allocate model resources in
``from_config``, not when the module is imported:

.. code:: python

   from reef import ConfiguredEvaluationPlugin


   class BenchmarkEvaluator(ConfiguredEvaluationPlugin):
       def __init__(self, benchmark, threshold, runtime, scenario):
           ...

       @classmethod
       def from_config(cls, config, *, runtime, scenario, environ):
           return cls(config["benchmark"], config["threshold"], runtime, scenario)

       def evaluate(self, candidate): ...

       def decide(self, candidate, evaluation): ...

The class must implement both ``evaluate`` and ``decide``. Reef rejects
functions, abstract classes, and objects that only have matching method names.
The trainer calls ``evaluate`` and then ``decide`` for each saved checkpoint,
before Reef publishes it. A rejection leaves the previous version serving and
publishes the checkpoint as a release held for review, which a person can
promote. The next job still starts from the rejected checkpoint: evaluation
decides only what is served. `Python API
<../reference/python-api.rst#candidate-evaluation>`__ documents the plugin
contract: ``evaluate``, ``decide``, the fail-closed rule, and idempotency by
``candidate.candidate_id``. The section fields are in `Configuration
<../reference/configuration.rst#the-evaluation-section>`__.

Where the feedback comes from
-----------------------------

Reef never invents feedback. Use whatever already judges your agent; for the
numeric ``score`` field, a consistent scale where higher is better. If you have
no number, `Choosing a recipe <../user-guide/recipes.rst>`__ lists the methods that need
none.

External method services
------------------------

A configuration file describes components and their parameters. It does not
accept ``service``, ``services`` or ``execution.services``. HTTP settings belong
in ``reef``. Reef assembles its HTTP process and supported inference/training
backends. Additional method services run independently of Reef's orchestrator.

Declare a recipe field for the external endpoint and implement the client in
the method package. For example, OpenClawRL uses ``recipe.config.prm-url`` and
``recipe.config.prm-tokenizer-path`` to call its independently served PRM.
CLI overrides such as ``--recipe.config.prm-url http://localhost:23001`` use the
same field declarations and validation as YAML. Request timeouts and scoring
failure behavior belong to that recipe's client.

The method's deployment tools own external service startup, readiness, resources
and shutdown. Reef does not register their processes, probe their health or
reserve their GPUs. The OpenClawRL example's ``docker-compose.yaml`` starts PRM
and user-model containers on devices separate from Reef/Slime. A remote endpoint
can be substituted without changing Reef's process topology. Reef shutdown or a
training startup failure leaves independently deployed services running.

Backend environment defaults belong to their integration; Slime's defaults
live in ``reef/train/slime_backend/launch.py`` and honor explicit environment
overrides.


Training backend deployment
---------------------------

``training.backend`` selects one ``TrainingDeployment`` from
``reef.train.deployment``. The built-in names are ``slime`` (the default) and
``tinker``. A definition lives under its integration. It describes the
deployment and does not allocate model resources:

* ``prepare(config, settings)`` receives the resolved deployment plus parsed
  service settings as a mapping. It validates backend combinations, binds
  derived values in ``config`` and returns the process definitions that the
  HTTP service waits for: for Slime, the optional teacher engine; for an
  in-process backend, none. It must not download models, allocate devices or
  construct a component.
* ``component_config(settings, *, loss_family)`` returns the config that
  ``reef.runtime.deployment.build_training_component`` builds in the HTTP
  process. Its ``type`` is the dotted reference of the component's
  ``TrainingComponentFactory``. ``loss_family`` is the recipe's startup loss
  family.
* ``http_environment(settings)`` returns variables for the HTTP service
  process. A value can name ``${endpoints.<service>}`` of a service that
  ``prepare`` returned.
* ``trainer_serving_config(settings)`` returns the runtime config of the
  backend's own serving, such as Tinker's sampler, or ``None`` when the
  backend has none (the default). ``inference.serving: trainer`` selects it.

The class attributes ``requires_local_model`` (default true),
``supports_teacher_engine`` and ``default_serving`` declare what the
integration needs and accepts. A hosted integration sets
``requires_local_model`` to false, so deployment resolution keeps a remote
model identifier. ``default_serving`` (``engines`` unless overridden) is the
serving of a deployment that sets no ``inference.serving`` and no engine
setting.

The ``TrainingComponentFactory`` that the config's ``type`` names returns the
``TrainingComponent`` alone. It runs in the HTTP service's process; Reef
passes it to the recipe as ``training_component``. The backend does not build
serving: the ``inference`` section selects it, and the component's
``release_format`` tells the serving runtime what each release holds.
`Training components <executors.rst#training-components>`__ defines the
component contract, and `Serving saved weights
<executors.rst#serving-saved-weights>`__ defines how published weights reach
the engines.

The built-in components derive from
``reef.train.local_jobs.LocalJobComponent``. It keeps each job in its own
directory (``job.json``, ``checkpoints.json`` and ``result.json``). It runs
one job at a time in a worker thread, and it resumes unfinished jobs after a
restart. A subclass implements these methods:

* ``check_job(job)`` raises ``ValueError`` for a job the component cannot
  train. ``submit`` then records nothing.
* ``run(job, directory)`` trains the job to its end, records each checkpoint
  with ``save_checkpoint``, and returns the job's metrics. A resumed job finds
  its earlier checkpoints with ``saved(directory)``. An exception ends the job
  as failed.
* ``close_resources()`` optionally releases what the subclass owns.

The subclass must call ``start_jobs()`` once it can train, for example at the
end of its constructor. Until then no job runs, including the jobs that a
restart left.

An in-process integration can reuse the supplied definition:

.. code:: python

   from reef.train.deployment import InProcessTrainingDeployment

   class Deployment(InProcessTrainingDeployment):
       component_type = "my_backend.runtime:factory"

Here ``factory`` is a lightweight ``TrainingComponentFactory`` instance. Its
``config_type()`` declarations validate ``training.options`` using the shared
parser. Import execution libraries only when the factory constructs the
component, and provide any backend-owned defaults there. The base ``prepare``
refuses Ray and training executor settings. The ``inference`` section stays
the serving's: the integration neither reads nor refuses it.

Select ``--training.backend my_backend.launch:Deployment`` directly, or register
an installed name in the integration distribution:

.. code:: toml

   [project.entry-points."reef.training_backends"]
   my-backend = "my_backend.launch:Deployment"

This enables ``--training.backend my-backend``. Only the selected definition
is imported. Missing or ambiguous names fail explicitly, without falling back
to Slime. Do not introduce a separate user-facing runtime selector for weight
training: runtime wiring belongs to the selected backend. Method dependencies
remain the recipe's responsibility.
