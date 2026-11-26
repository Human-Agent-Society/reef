Recipe configuration
====================

How a deployment selects its recipe, and the rules of the ``recipe``, ``activation`` and ``pipeline`` sections that their keys do not show. `Settings reference <configuration-settings.rst#recipe>`__ lists the keys. `Write a recipe <../developer-guide/write-a-recipe.rst>`__ describes the recipe contract.

``recipe.implementation`` selects the recipe. The recipe's declared fields go under ``recipe.config``, and a field that the recipe does not declare is an error. It takes one of three forms:

- **The core recipe:** ``recipe``, the default. It records, and it trains nothing.
- **A dotted class**, such as ``my_pkg.my_method:MyMethodRecipe``. The package does not have to be installed. ``reef serve`` walks up from the directory of the config file to the nearest directory that holds ``<package>/__init__.py``, and appends that directory to ``PYTHONPATH`` for every service. Entries already in ``PYTHONPATH`` keep their precedence.
- **A preset name**, such as ``my-preset``: the file ``my-preset.yaml`` under ``REEF_RECIPE_CONFIG_DIR``. The variable has no default, and Reef bundles no presets. A bare name other than ``recipe`` is always a preset name.

Rules for all recipes:

- With ``recipe`` or a preset name, Reef refuses the ``pipeline`` and ``activation`` sections of the deployment config.
- ``reef serve --print-config`` does not check the ``pipeline`` section, the ``activation`` section or the value of ``recipe.config.training-mode``. The service checks them when it starts.
- ``recipe.runtime`` selects the runtime of a dotted recipe that trains no weights: a ``type`` (``inference_proxy``, ``tinker``, or the ``package.module:ClassName`` of a ``RuntimeFactory``) and the settings of that type.
- `Manual training <http-api-training.rst#manual-training>`__ describes training requests. `Parameters that Slime and Tinker take <../developer-guide/recipe-training-service.rst#parameters-that-slime-and-tinker-take>`__ lists the names that ``recipe.config.job-parameters`` takes.

Presets
-------

A preset is a file in the configuration file layout that holds a recipe and its settings. Reef reads it as it is: ``${VAR}`` interpolates in a deployment config, never in a preset.

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

- A preset holds ``recipe.implementation``, which is ``recipe`` or a dotted class, and ``recipe.config``. Its stage config goes in its own ``pipeline`` section.
- An optional ``recipe.runtime`` makes the recipe build its own runtime, in place of the deployment's upstream proxy. Such a preset must name its model (``inference.upstream-model`` or ``inference.model-path``). A preset without ``recipe.runtime`` can omit the model and takes the model of the deployment's runtime.
- A preset cannot select a weight-training recipe. Select that recipe by dotted class in the deployment config, so that the service can build its training component.

Composite recipes
-----------------

A composite recipe serves and evolves several snapshot components in one scenario, with one recipe for each. ``recipe.config.components`` maps each component name to a section with ``implementation``, the recipe class, and ``config``, the recipe's fields and sections with its own ``pipeline``.

.. code:: yaml

   recipe:
     implementation: reef.recipe.composite:CompositeRecipe
     config:
       components:
         weights:
           implementation: recipes.sao.recipe:SAORecipe
           config:
             pipeline: {sample2job: {config: {job_samples: 8}}}
         harness:
           implementation: reef.recipe.harness:HarnessEvolutionRecipe
           config:
             evolution: {adapter: pi, method: methods.mine:MyMethod, evaluation: {tasks: ["..."], scorer: methods.mine:MyScorer}}
   inference:
     model-path: Qwen/Qwen3-8B
   training:
     backend: slime
     options:
       global-batch-size: 8

- ``components`` holds at least two recipes. At most one trains weights, at most one is a harness evolution recipe, and none is a composite.
- Reef refuses the component names ``train``, ``step``, ``reef`` and ``operations``, because experiment tracking uses them as metric prefixes.
- A component has no ``runtime``: every component uses the deployment's model, inference runtime and training component.
- The components share the ``training-mode`` of the composite.
- With a component that trains weights, ``training.backend`` selects the backend, and a top-level ``pipeline`` section replaces the ``pipeline`` of that component.
- Without one, the deployment sets ``inference.upstream-model`` in place of ``inference.model-path`` and has no ``training`` section. It ignores a top-level ``pipeline`` section.

The ``activation`` section
--------------------------

Training makes artifacts, and the activation rule of this section decides which new artifacts serve (`Activation rule <../developer-guide/recipe-control-service.rst#activation-rule>`__). Without the section, the rule is ``always``. A scenario can select a built-in rule for itself (`Scenario updates <http-api-scenarios.rst#scenario-updates>`__).

.. code:: yaml

   activation:
     module: evaluated
     config:
       min_mean_score: 0.8

The keys of ``activation.config`` take underscores only.

.. container:: config

   ``activation.module`` : always
       ``always``, ``never``, ``evaluated``, or the ``package.module:ClassName`` of an ``ActivationRule`` subclass. A class of your own receives ``activation.config`` in ``from_config``; ``never`` takes no config.

   ``activation.config.review_kinds`` : []
       For ``always``: the kinds of harness tree parts whose change waits for a person, such as ``rules``, ``skill`` or ``extension``. The harness recipe refuses a kind that the adapter does not evolve.

   ``activation.config.min_mean_score``
       For ``evaluated``, required: the least ``mean_score`` with which a finished evaluation activates its artifact.

``evaluated`` needs the scenario's ``artifact_evaluation`` (`Automatic evaluation <http-api-evaluation.rst#automatic-evaluation>`__), so that Reef evaluates each new artifact. A scenario without it activates no new artifact, and Reef logs a warning when it creates or loads the scenario.

The ``pipeline`` section
------------------------

A recipe's pipeline has two stages, and the recipe names a default class for each (`Training pipeline <../developer-guide/write-a-recipe.rst>`__). This section gives each stage its config, and it can name another class for a stage.

.. code:: yaml

   recipe:
     implementation: recipes.tttd.recipe:TTTDRecipe
   pipeline:
     record2sample:
       config:
         groups_per_step: 8
         rollouts_per_group: 64

- ``pipeline.record2sample`` is the record-to-sample stage, and ``pipeline.sample2job`` is the sample-to-job stage. Each takes ``config`` and an optional ``module``, the ``package.module:ClassName`` of a ``Record2Sample`` or ``Sample2Job`` subclass. Another key under ``pipeline`` fails the startup.
- A stage without a key is the recipe's class with no config.
- Each stage reads only its own ``config``. A stage key under ``recipe.config`` is an unknown recipe key and fails the startup.
- A stage parses its ``config`` when a scenario builds its pipeline. An unknown key or a value out of range fails then, not at startup.
- The settings reference lists the keys of the built-in stages, and the page of each recipe lists the keys of its own stages. A scenario can replace keys of ``pipeline.record2sample.config`` for itself (`Record-to-sample config <http-api-scenarios.rst#record-to-sample-config>`__).
- `Job size <../developer-guide/recipe-control-service.rst#job-size>`__ describes ``job-samples`` and ``buffer-size``, and `Staleness <../developer-guide/recipe-samples-service.rst#staleness>`__ describes which samples are stale.
