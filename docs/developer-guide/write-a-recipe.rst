Write a recipe
==============

A recipe is the method that a deployment runs. It says which records train,
how they become the data of a training job, and how that job changes the
weights or the harness tree. This page shows what a recipe holds, which
service calls each part, and which page defines each part.

:For: authors of a weight method or a harness method
:Before you start: the core loop from `Quickstart <../getting-started/quickstart.rst>`__, and `Choose a recipe <../user-guide/recipes.rst>`__ to check that no existing recipe fits
:You will have: a map of the plug-in points, a minimal recipe of each kind, and the pages to read next

What a recipe is
----------------

A recipe is configuration. It is a frozen dataclass (``reef.recipe.Recipe``)
that names classes and holds settings. It runs nothing by itself.

A recipe does not:

- implement a service, a training component or an inference runtime. Reef
  builds them from the deployment config;
- accept records, replay them after a restart, submit jobs, commit state or
  publish artifacts. Reef does that for every recipe;
- decide which artifact serves. The deployment's activation rule decides.

A recipe fills plug-in points. Reef's services call them.

Which service calls which plug-in
---------------------------------

A deployment runs seven services: ``control``, ``inference``, ``samples``,
``training``, ``evaluation``, ``data`` and ``harness`` (see `Run the services
in separate processes
<../user-guide/operate.rst#run-the-services-in-separate-processes>`__). By
default ``reef serve`` runs all of them in one process, on the same
interfaces. A plug-in must not assume that another plug-in runs in its
process.

.. list-table::
   :header-rows: 1
   :widths: 12 22 26 20 20

   * - Service
     - Plug-in
     - When the service calls it
     - Default
     - Page
   * - ``control``
     - Report type (``Recipe.report_type``)
     - On each ``POST /reef/report``, before Reef stores the report
     - ``None``: every well-formed report is accepted
     - `Control service <recipe-control-service.rst#report-type>`__
   * - ``samples``
     - Record-to-sample stage (``Record2Sample``)
     - For each record selection: after new records, a commit, a data write,
       or the end of the stage's background work
     - ``NoSamples``: no sample, so no job
     - `Samples service <recipe-samples-service.rst>`__
   * - ``samples``
     - Staleness rule (``Record2Sample.is_stale``)
     - After a commit changes what the scenario serves, for each group of
       untrained samples
     - On-policy: only data of the served artifact trains
     - `Samples service <recipe-samples-service.rst#staleness>`__
   * - ``samples``
     - Task generator (``TaskGenerator``)
     - In each round of the built-in generated-task recipe
     - None: ``rounds`` is 0, so nothing is generated
     - `Task generators <task-generators.rst>`__
   * - ``control``
     - Sample-to-job stage (``Sample2Job``): job size, job data, job
       parameters
     - In each trainer cycle, when enough samples wait or a training request
       is queued
     - ``NoJobData``: never ready
     - `Control service <recipe-control-service.rst#sample-to-job-stage>`__
   * - ``control``
     - Activation rule (``ActivationRule``)
     - When a training step publishes an artifact, and when an artifact
       evaluation finishes
     - ``always``
     - `Control service <recipe-control-service.rst#activation-rule>`__
   * - Each service that loads a scenario
     - Scenario components (``Recipe.build_components``)
     - When the service creates or loads a scenario
     - ``Recipe``: none. ``WeightTrainingRecipe``: ``weights``.
       ``HarnessEvolutionRecipe``: ``harness``
     - `Scenario components <components.rst>`__
   * - ``training``, ``control``
     - Admission check (``ComponentSpec.validator``)
     - ``training``: on each checkpoint, before Reef stores it.
       ``control``: on each import
     - Accept every artifact
     - `Scenario components <components.rst#admission-checks>`__
   * - ``training``
     - Weight method: the objective and its loss family
     - Once per job (``prepare``), then in each optimizer step (the loss)
     - No objective. ``sft`` is built in
     - `Training service <recipe-training-service.rst>`__,
       `Loss families <loss-families.rst>`__
   * - ``training``
     - Harness method (``Method``), with its scorer, collector and
       publication policy
     - Once per harness job (``Method.run``); the scorer once per run
     - ``evolution.method`` is required. Policy ``compare``
     - `Write a harness method <write-a-harness-method.rst>`__

The ``evaluation`` service runs the rollouts of artifact evaluations. It
calls no recipe plug-in: control reads the scores and asks the activation
rule. The ``inference``, ``data`` and ``harness`` services call none either.
Inference and harness read the declared components.

The path of a record through the plug-ins:

.. code:: mermaid

   flowchart LR
       accTitle: The services and the recipe plug-ins that a record passes
       REP["report"] -->|"report type"| CTRL1["control<br/>stores the record"]
       CTRL1 --> SAMP["samples<br/>record-to-sample stage"]
       SAMP -->|"samples"| CTRL2["control<br/>sample-to-job stage"]
       CTRL2 -->|"job data + parameters"| TRAIN["training<br/>objective or Method"]
       TRAIN -->|"checkpoint"| CTRL3["control<br/>activation rule"]
       CTRL3 -->|"active snapshot"| INF["inference"]
       class SAMP,CTRL2,TRAIN,CTRL3 user-owned

Choose the kind of method
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * - Question
     - Weight method
     - Harness method
   * - What changes
     - model weights
     - the harness tree: rules, skills, commands, extensions
   * - What it needs
     - a training backend: Slime on your GPUs, or Tinker
     - a model endpoint
   * - What you write
     - a ``WeightTrainingRecipe`` subclass, its two stages, an objective,
       and usually a loss family
     - a ``Method`` class. The recipe is the built-in
       ``HarnessEvolutionRecipe``
   * - What to read
     - this page, then `Samples service <recipe-samples-service.rst>`__,
       `Control service <recipe-control-service.rst>`__, `Training service
       <recipe-training-service.rst>`__ and `Loss families
       <loss-families.rst>`__
     - this page, then `Write a harness method
       <write-a-harness-method.rst>`__

Read `Scenario components <components.rst>`__ only when your recipe serves
something other than ``weights`` or ``harness``. Read `Training backends
<training-backends.rst>`__ only when you add a backend beside Slime and
Tinker. A scenario that evolves weights and a harness together uses
``CompositeRecipe``, which maps each component name to one of the two kinds
(see `Recipe configuration
<../reference/configuration-recipe.rst#recipe-configuration>`__).

A minimal weight recipe
-----------------------

``recipes/sdft/`` is a complete weight method in four files:

- ``recipe.py``: the recipe class, with the report type, the stages and the
  default job parameters. Every service loads it.
- ``processor.py``: the record-to-sample stage ``SDFTSamples``. The samples
  service runs it.
- ``slime/__init__.py``: the objective ``SdftObjective`` and the loss family
  ``SdftAlgorithm``. The training service runs them.
- ``slime/objective.py``: the loss hooks that Slime's workers call.

The package ``__init__.py`` registers the objective by reference, so that
importing the package loads no backend code:

.. code:: python

   from reef.train.slime_backend.objectives import register_objective

   register_objective("sdft", "recipes.sdft.slime:SdftObjective")

The recipe class binds the plug-ins (imports omitted; see
``recipes/sdft/recipe.py``):

.. code:: python

   @dataclass(frozen=True, kw_only=True)
   class SDFTRecipe(WeightTrainingRecipe):
       name: str = "sdft"

       @property
       def report_type(self) -> type[ReportBase]:
           return TeacherContextReport

       @classmethod
       def default_parameters(cls) -> Mapping[SampleKind, Mapping[str, Any]]:
           return {kind: {"objective": "sdft"} for kind in SAMPLE_KINDS}

       @classmethod
       def pipeline_spec(cls) -> PipelineSpec:
           return PipelineSpec(record2sample=SDFTSamples, sample2job=DistillJobData)

The deployment selects the class and sets the job size:

.. code:: yaml

   recipe:
     implementation: recipes.sdft.recipe:SDFTRecipe
   pipeline:
     sample2job:
       config:
         job_samples: ${training.config.global_batch_size}
   training:
     backend: slime
     config:
       global_batch_size: 32
     options:
       global-batch-size: "${training.config.global_batch_size}"
       loss-type: "custom_loss"

- ``pipeline.sample2job.config.job_samples`` is required for a weight
  recipe: no scenario's trainer builds without it (see `Job size
  <recipe-control-service.rst#job-size>`__).
- ``training.options`` holds the backend's native flags. They must match the
  loss family (see `Family to Slime flags
  <loss-families.rst#family-to-slime-flags>`__).
- Install the package where every service of the deployment can import it.
  Check the import that Reef performs:
  ``python -c "from recipes.sdft.recipe import SDFTRecipe"``.

``recipes/sdft/examples/skill_stream/serve.yaml`` is the complete config.

A minimal harness method
------------------------

A harness method needs no recipe class. The deployment selects
``HarnessEvolutionRecipe`` and names the ``Method`` in the ``evolution``
section:

.. code:: python

   from reef.train.harness_backend import Method, MethodJob


   class MyMethod(Method):
       def run(self, job: MethodJob) -> None:
           ...  # edit job.workspace.harness; an unchanged tree skips the job

.. code:: yaml

   recipe:
     implementation: reef.recipe.harness:HarnessEvolutionRecipe
     config:
       evolution:
         adapter: pi
         seed: methods/seed
         method: methods.my_method:MyMethod
         evaluation:
           tasks:
             - "[fib] Compute fib(90) exactly. Reply with the integer alone on the last line."
           scorer: methods.scoring:ExactAnswer
           policy: compare
   inference:
     upstream-url: ${REEF_UPSTREAM_URL}
     upstream-model: qwen3-8b

`Write a harness method <write-a-harness-method.rst>`__ defines the
``Method`` contract, the scorer and the policy.
``recipes/gepa/examples/aime/gepa.yaml`` is a complete config.

Recipe settings
---------------

Declare each setting of a recipe class once, with ``config_field``. The
field is the YAML key under ``recipe.config``, the CLI flag, the environment
fallback and the parser:

.. code:: python

   from dataclasses import dataclass

   from reef.recipe import WeightTrainingRecipe, config_field


   @dataclass(frozen=True)
   class MyMethodRecipe(WeightTrainingRecipe):
       clip_ratio: float = config_field(0.2, env="MY_CLIP_RATIO", help="Bound on the policy ratio.")
       temperature: float = config_field(0.5, allow_nonfinite=False)
       tags: tuple[str, ...] = config_field(())

- Supported types: ``str``, ``int``, ``float``, ``bool``, ``tuple[str,
  ...]``, ``Mapping[str, Any]`` or ``dict[str, Any]``, and their optional
  forms. An object field uses ``config_field(default_factory=dict)``.
- Put range checks and checks between fields in ``__post_init__``.
- One loader builds every recipe (``reef.recipe.loading.load_recipe``). Do
  not add a loading method to a recipe class.
- An unknown key, a wrong type or a failed check raises
  ``RecipeConfigError`` when the deployment loads.

A recipe class can also declare:

- ``section_field(path, required=...)``: a field that holds one config
  section, parsed into the ``Settings`` dataclass of its annotation.
- ``reader_field(reader)``: a field that a ``ConfigReader`` reads from the
  config, the environment and the parsed sections, such as a file that a
  section names. The value is configuration, never an object that runs.
- ``section_defaults``: a class attribute with the class's own defaults for
  its sections. The deployment's config merges over them, key by key.
- ``refused_sections``: a class attribute that maps a section name to an
  error message. The loader raises ``RecipeConfigError`` with it when the
  deployment sets the section.

Every recipe also has ``training_mode`` (``auto`` or ``manual``, default
``auto``), the ``pipeline`` section and the ``activation`` section.

A method that calls a service of its own, such as a judge model, declares a
field for the endpoint and implements the client in its package. Reef does
not start, probe or stop that service. OpenClaw-RL does this with
``recipe.config.prm-url``.

The rest of the deployment config
---------------------------------

The ``recipe`` section selects the method. A deployment config also needs:

- ``reef``: host, port, token and run directory of the HTTP service.
- ``inference``: what serves the model: local engines (``model-path``) or an
  upstream endpoint (``upstream-url``, ``upstream-model``).
- ``pipeline``: the config of each stage, and optionally another stage class.
- ``training``: for a weight method, the backend, its timeouts and its
  native ``options``.
- ``activation``: the activation rule; ``always`` when absent.
- ``execution`` and ``executors``: where training, rollouts and harness runs
  execute.
- ``storage``: the record store, the artifact repository and the work
  directories.

`Configuration <../reference/configuration-files.rst#configuration-file-layout>`__
lists every section and key. `Evolve your model
<../user-guide/evolve-your-model.rst>`__ and `Evolve your harness
<../user-guide/evolve-your-harness.rst>`__ give complete configs to copy.

Where the feedback comes from
-----------------------------

Reef never invents feedback. A report carries what already judges your
agent. For the numeric ``score`` field, use one scale on which higher is
better. A method without a number computes its own feedback (see `Computed
feedback <recipe-samples-service.rst#computed-feedback>`__) or needs none,
as SFT does.

Package a method
----------------

A method in this repository is one package under ``recipes/<name>/``, with a
``README.md`` and runnable deployments in ``examples/``. Reef does not import
``recipes/`` at start and does not ship it in the wheel. A deployment
selects a method by its dotted ``package.module:ClassName``. An installed
package of your own works the same way. `Python API
<../reference/python-api.rst>`__ lists the members of each contract.
