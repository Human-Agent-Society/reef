Recipe configuration
====================

The keys of the ``recipe``, ``activation`` and ``pipeline`` sections: the recipe that a deployment runs, its training mode and job parameters, the rule that decides which new artifacts serve, and the config of each pipeline stage.

A configuration file selects the recipe with ``recipe.implementation``, puts the recipe's declared fields under ``recipe.config`` and the runtime under ``recipe.runtime``. A field that the selected recipe does not declare is an error. Harness settings live in ``recipe.config.evolution`` (`Harness evolution keys <configuration-harness.rst#harness-evolution-keys>`__).

**Select a recipe.** ``recipe.implementation`` takes one of three forms:

- **The core record-only recipe:** ``recipe``, the default.
- **A dotted class:** ``"my_pkg.my_method:MyMethodRecipe"``. The class does not have to be pip-installed. ``reef serve`` walks up from the config file's directory to the nearest ancestor that holds ``<package>/__init__.py`` and appends that directory to ``PYTHONPATH`` for every service. This is how ``recipes.sao.recipe:SAORecipe`` resolves from a source checkout. Entries already in ``PYTHONPATH`` keep their precedence.
- **A named preset:** ``my-preset``, read from ``my-preset.yaml`` under ``REEF_RECIPE_CONFIG_DIR``. The variable has no default, and Reef bundles no presets. A bare name other than ``recipe`` is always a preset name; it never imports a learning method.

**Training mode.** Every recipe has ``recipe.config.training_mode``:

.. container:: config

   ``recipe.config.training_mode`` : auto
       ``auto`` starts a job when the `next job <../developer-guide/recipe-control-service.rst#the-next-job>`__ holds ``job_samples`` untrained samples, and refuses ``POST /reef/train``. ``manual`` starts a job only for a ``POST /reef/train`` request

- A ``manual`` job of a weight recipe trains the samples that the request names (`Choose the samples that train <http-api-samples.rst#choose-the-samples-that-train>`__). The request text goes into the commit's ``training_request`` metrics.
- Harness evolution runs a ``manual`` request at once. The method gets the request and the traces that it names, possibly none.
- Records arrive and stages make samples in both modes. The samples wait as data items until a job trains them, so a switch from ``manual`` to ``auto`` can train on data already collected.
- A recipe's ``Sample2Job`` declares ``supported_training_modes``; an unsupported mode raises ``NotImplementedError``. Harness evolution supports both modes. ``manual`` on an inference-only recipe creates no training backend.
- The mode is independent of the `activation rule <#the-activation-section>`__.
- ``POST /reef/scenarios/{scenario}/update`` selects another mode at runtime. The change applies to later job data; reserved job data completes under its original mode. Reef stores the selected mode in the scenario's settings. The stored mode survives restarts and takes precedence over the configured one. A stored mode that the recipe does not support keeps the configured mode and logs a warning.

.. code:: yaml

   recipe:
     config:
       training_mode: manual

**Job parameters.** ``recipe.config.job_parameters`` is an object that a weight recipe sends with every job, over its default parameters. Reef does not read it. The training component owns it, and ``GET /reef/scenarios/{scenario}/contract`` serves its JSON Schema. The component refuses a parameter that it does not take, and the job is not submitted. A recipe that overrides ``configured_parameters`` decides where the setting goes.

.. list-table::
   :header-rows: 1

   * - Parameter
     - Slime
     - Tinker
   * - ``objective``
     - required: a registered name or a dotted reference
     - required: a registered name or a dotted reference
   * - ``learning_rate_schedule``
     - optional
     - optional
   * - ``scheduling``
     - optional (`Step scheduling <python-api-training.rst#training-objective>`__)
     - optional
   * - ``save_interval``
     - optional
     - optional
   * - ``batch_size``, ``learning_rate``, ``kl_coef``
     - no
     - optional
   * - ``dataset``
     - a job without samples (`Dataset jobs on Slime <configuration-training.rst#dataset-jobs-on-slime>`__)
     - no
   * - ``harbor``
     - required on a task job, refused on another job
     - no
   * - ``rollout``
     - no
     - required on a task job, refused on another job

.. code:: yaml

   recipe:
     config:
       job_parameters:
         scheduling:
           packing_length: 8192
         save_interval: 50

**Presets.** A preset uses the configuration file layout and is read as-is: ``${VAR}`` interpolates in a deployment config, never in a preset.

- A preset carries ``recipe.implementation`` (``recipe`` or a dotted class) and ``recipe.config``. ``recipe.config`` holds the training mode and the recipe's own sections, and ``pipeline`` holds the stage config.
- An optional ``recipe.runtime`` makes the recipe build its own runtime instead of using the deployment's upstream proxy. Such a preset must name its model (``inference.upstream-model`` or ``inference.model-path``).
- A preset that uses the deployment's runtime may omit the model and inherit the runtime's. An explicit model takes precedence.
- Presets suit recipes whose runtime can be built from the preset or the deployment's upstream proxy. Select a weight-training recipe by dotted class in the deployment config, so that the service can build its training component.

.. code:: yaml

   recipe:
     implementation: reef.recipe.harness:HarnessEvolutionRecipe
     config:
       evolution:
         adapter: pi
         method: methods.mine:MyMethod
         evaluation:
           tasks: ["..."]
           scorer: methods.mine:MyScorer
   inference:
     upstream-model: qwen3-8b

**Composite recipes.** A composite recipe serves and evolves several snapshot components in one scenario, one recipe per recipe component. Its ``components`` object maps each recipe component name to a section with the shape of the top-level ``recipe`` section: ``implementation`` names the recipe class, and ``config`` holds the recipe's fields and sections, including its own ``pipeline`` stage config.

- ``components`` holds at least two recipe components. At most one trains weights, at most one is a harness evolution recipe, and none is a composite. Reef refuses any other ``components`` object when it loads the configuration.
- A recipe component has no ``runtime``: every recipe component uses the deployment's model, inference runtime and training component.
- A composite whose recipe component trains weights is deployed the way that recipe is: ``training.backend`` selects the backend, and a top-level ``pipeline`` or ``evaluation`` section goes to that recipe component.
- A scenario's root snapshot keeps one directory per recipe component, with that component's default start. A recipe component whose recipe seeds nothing starts empty.
- Each recipe component's trainer commits into the same snapshot chain, so every step checkpoints, and the recipe components share one ``training_mode``.
- Reef admits a report when the contract of any recipe component accepts it. Each trainer keeps the reports that its own contract parses.

.. code:: yaml

   recipe:
     implementation: reef.recipe.composite:CompositeRecipe
     config:
       components:
         weights:
           implementation: recipes.sao.recipe:SAORecipe
           config:
             pipeline: {sample2job: {config: {job_samples: 8, buffer_size: 32}}}
         harness:
           implementation: reef.recipe.harness:HarnessEvolutionRecipe
           config:
             evolution: {adapter: pi, method: methods.mine:MyMethod, evaluation: {tasks: ["..."], scorer: methods.mine:MyScorer}}
   inference:
     model-path: Qwen/Qwen3-8B
   training:
     backend: slime

A composite without a recipe component that trains weights has the same shape, with ``inference.upstream-model`` in place of ``inference.model-path`` and no ``training`` section.

The ``activation`` section
--------------------------

Training makes artifacts: every checkpoint that a weight training job saves, and every tree that a harness trainer publishes (`Harness evolution keys <configuration-harness.rst#harness-evolution-keys>`__). Training never decides whether an artifact serves. The activation rule that this section names decides that, for every recipe, a composite included.

- Without the section, the rule is ``always``.
- A scenario can select one of the built-in rules for itself with ``POST /reef/scenarios/{scenario}/update`` (`Scenario updates <http-api-scenarios.rst#scenario-updates>`__). The section's rule applies to each scenario that selects none.
- An artifact that the rule does not activate stays inactive. A person can activate it with the ``served`` setting of the same request (`Set what a scenario serves <../user-guide/operate.rst#set-what-a-scenario-serves>`__).
- Activation does not change where training continues.

.. code:: yaml

   activation:
     module: always
     config:
       review_kinds: [extension]

.. container:: config

   ``activation.module`` : always
       ``always``, ``never``, ``evaluated``, or a ``package.module:ClassName`` reference to an ``ActivationRule`` subclass (``reef.core.activation``)

   ``activation.config``
       the rule's settings, below; a class of your own receives the mapping in ``from_config``

The built-in rules:

- ``always`` activates every new artifact, except a harness tree that changes a kind in ``review_kinds`` (a list, default empty). Each name is a kind of the ``evolve`` section of the adapter's ``adapter.yaml``, such as ``rules``, ``config``, ``skill``, ``command``, ``extension`` or native's ``native_*`` kinds; the harness recipe refuses a name that the adapter does not evolve. A tree that changes a kind that is always reviewed, such as native's ``native_loop``, waits for a person whether or not the list names it.
- ``never`` activates no artifact, for fully manual activation. It takes no config.
- ``evaluated`` activates an artifact once an evaluation of it finishes with a ``mean_score`` of at least ``min_mean_score`` (required). Set the scenario's ``artifact_evaluation`` (`Automatic evaluation <http-api-evaluation.rst#automatic-evaluation>`__) so Reef evaluates each new artifact. A scenario without ``artifact_evaluation`` under this rule activates no new artifact, and Reef logs a warning when it creates or loads the scenario.

A rule of your own subclasses ``ActivationRule`` (`Activation rule <../developer-guide/recipe-control-service.rst#activation-rule>`__).

Reef checks the class and builds the rule when it loads the deployment config, so a bad rule fails at startup. Reef rejects functions, abstract classes and structural lookalikes.

The ``pipeline`` section
------------------------

A recipe's pipeline has two stages (see `Training pipeline <../developer-guide/write-a-recipe.rst>`__). The recipe names a default class for each stage. This section gives each stage its config, and it can name another class for a stage. Every stage reads only its own ``config``; the recipe's ``recipe.config`` holds no stage settings.

.. container:: config

   ``pipeline.record2sample``
       the record-to-sample stage, which decides which records train, in which groups, and makes their samples: ``config``, and optionally ``module``, a ``Record2Sample`` subclass

   ``pipeline.sample2job``
       the sample-to-job stage: ``config``, and optionally ``module``, a ``Sample2Job`` subclass

.. code:: yaml

   recipe:
     implementation: recipes.tttd.recipe:TTTDRecipe
   pipeline:
     record2sample:
       config:
         groups_per_step: 8
         rollouts_per_group: 64

Rules of the section:

- A stage without a key is the recipe's class with no config. Any other key under ``pipeline`` fails the startup.
- Each recipe's page lists the keys of its stages. Keys are written as the stage reads them; a hyphen spelling is also accepted.
- Reef checks each stage when it loads the recipe config, before a scenario starts. It imports a named class, rejects functions and abstract classes, and parses the stage's ``config``. An unknown key or a value out of range fails the startup.
- A stage key under ``recipe.config``, such as ``groups_per_step``, is an unknown recipe key and also fails the startup.
- A stage class needs the class method ``from_config``. The sample-to-job stage receives the record-to-sample stage and reads the values that the record-to-sample stage owns from it. For example, the TTTD step grid is set once, in ``pipeline.record2sample.config``.
- A scenario can replace keys of ``pipeline.record2sample.config`` for itself with its ``record2sample_config`` setting (`Record-to-sample config <http-api-scenarios.rst#record-to-sample-config>`__).

**Staleness.** Each record-to-sample stage decides which of its samples are stale; no setting chooses the rule. Most built-in stages are on-policy: only data of the artifact that the scenario serves trains. An SFT stage and the harness trace stages never drop a sample. SAO accepts a bounded lag, its own ``max_artifact_lag`` (`SAO <../user-guide/recipes/sao.rst>`__). Startup refuses a ``pipeline.record2sample.config.max_staleness`` key: remove it. A training framework's own off-policy limit goes in that backend's ``training.options``; no built-in backend has one now. See `Staleness <../developer-guide/recipe-samples-service.rst#staleness>`__.

**Job size.** Reef submits one training job for each job's data. Every built-in sample-to-job stage that trains on records reads these two keys; a recipe's page lists the other keys of its stage.

.. container:: config

   ``pipeline.sample2job.config.job_samples`` : required for weight recipes; 1 for the harness evolution stages
       the fewest samples of one job, a positive int; at most the stage's grid for TTTD and SDPO. A weight recipe without it fails when Reef builds a scenario's trainer, with a ``PluginConfigError`` that names the key

   ``pipeline.sample2job.config.buffer_size`` : ``job_samples``
       the most samples that one automatic job takes, beside the rest of a group that those samples touch; at least ``job_samples``. Uploaded trajectories count toward it. Samples beyond it wait for later jobs.

The training component cuts each job's data into optimizer batches by its own batch size: Slime's ``--global-batch-size`` (``training.config.global_batch_size`` in the examples) or Tinker's ``batch_size``. Usually, set ``job_samples`` to that batch size, so that one job is one optimizer step:

.. code:: yaml

   pipeline:
     sample2job:
       config:
         job_samples: ${training.config.global_batch_size}

The two values can differ. Job data takes up to ``buffer_size`` samples and the rest of each group that those samples touch, so a group of rollouts trains in one job. A job with more samples than the batch size trains more than one optimizer step, and a job with fewer trains one partial step. The ``remainder`` field of a job's optional ``scheduling`` parameter decides what happens to a shorter last step; by default, the component trains it as a smaller step. See `Job size <../developer-guide/recipe-control-service.rst#job-size>`__.
