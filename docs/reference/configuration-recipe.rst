Recipe configuration
====================

Select a recipe, configure its pipeline, and choose when new artifacts start serving.
For all keys and defaults, see `Settings reference <configuration-settings.rst#recipe>`__.

Set ``recipe.implementation`` to one of the following values.
Put the recipe's fields under ``recipe.config``. Unknown fields are errors.

- **Core recipe:** ``recipe`` (default). Records requests and reports without training.
- **Dotted class:** for example, ``my_pkg.my_method:MyMethodRecipe``. The package can be installed or located above the config file. Reef searches parent directories for ``<package>/__init__.py`` and appends the nearest match to each service's ``PYTHONPATH``. Existing entries take priority.
- **Preset:** a bare name other than ``recipe``, such as ``my-preset``. Reef reads ``$REEF_RECIPE_CONFIG_DIR/my-preset.yaml``. Set this variable explicitly; Reef bundles no presets.

Rules for all recipes:

- With ``recipe`` or a preset name, Reef refuses the ``pipeline`` and ``activation`` sections of the deployment config.
- ``reef serve --print-config`` does not check the ``pipeline`` section, the ``activation`` section or the value of ``recipe.config.training-mode``. The service checks them when it starts.
- ``recipe.runtime`` selects the runtime of a dotted recipe that trains no weights: a ``type`` (``inference_proxy``, ``tinker``, or the ``package.module:ClassName`` of a ``RuntimeFactory``) and the settings of that type.
- `Manual training <http-api-training.rst#manual-training>`__ describes training requests. `Parameters that Slime and Tinker take <../developer-guide/recipe-training-service.rst#parameters-that-slime-and-tinker-take>`__ lists the names that ``recipe.config.job-parameters`` takes.

Presets
-------

A preset uses the deployment configuration layout. Reef does not expand ``${VAR}`` references inside presets.

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

- Set ``recipe.implementation`` to ``recipe`` or a dotted class. Put recipe fields under ``recipe.config`` and stage fields under ``pipeline``.
- With ``recipe.runtime``, also set ``inference.upstream-model`` or ``inference.model-path``. Otherwise, the preset can use the deployment's model and runtime.
- Presets cannot select weight-training recipes. Select those by dotted class in the deployment config.

Composite recipes
-----------------

A composite recipe serves and evolves several components in one scenario.
Under ``recipe.config.components``, give each component an ``implementation`` and ``config``.
Put its pipeline inside that component's ``config``.

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

- Use at least two components. At most one can train weights and one can evolve a harness. Components cannot be composites.
- Reef refuses the component names ``train``, ``step``, ``reef`` and ``operations``, because experiment tracking uses them as metric prefixes.
- A component has no ``runtime``: every component uses the deployment's model, inference runtime and training component.
- The components share the ``training-mode`` of the composite.
- With a component that trains weights, ``training.backend`` selects the backend, and a top-level ``pipeline`` section replaces the ``pipeline`` of that component.
- Without one, the deployment sets ``inference.upstream-model`` in place of ``inference.model-path`` and has no ``training`` section. It ignores a top-level ``pipeline`` section.

The ``activation`` section
--------------------------

The activation rule decides which new training artifacts serve. The default is ``always``.
A scenario can override it with a built-in rule through `Scenario updates <http-api-scenarios.rst#scenario-updates>`__.

.. code:: yaml

   activation:
     module: evaluated
     config:
       min_mean_score: 0.8

The keys of ``activation.config`` take underscores only.

.. container:: config

   ``activation.module`` : always
       ``always``, ``never``, ``evaluated``, or an ``ActivationRule`` subclass as ``package.module:ClassName``. Custom classes receive ``activation.config`` in ``from_config``. ``never`` takes no config.

   ``activation.config.review_kinds`` : []
       For ``always``: part kinds that require manual activation, such as ``rules``, ``skill`` or ``extension``. Each kind must be evolvable by the adapter.

   ``activation.config.min_mean_score``
       Required for ``evaluated``: the minimum ``mean_score`` needed to activate an artifact after evaluation.

For ``evaluated``, configure the scenario's ``artifact_evaluation`` (`Automatic evaluation <http-api-evaluation.rst#automatic-evaluation>`__).
Without it, no new artifact activates. Reef logs a warning when it creates or loads the scenario.

The ``pipeline`` section
------------------------

Use ``pipeline`` to configure or replace a recipe's two stages: record-to-sample and sample-to-job.
The recipe supplies each stage's default class.

.. code:: yaml

   recipe:
     implementation: recipes.tttd.recipe:TTTDRecipe
   pipeline:
     record2sample:
       config:
         groups_per_step: 8
         rollouts_per_group: 64

- ``record2sample`` and ``sample2job`` accept ``config`` and optional ``module``. Other keys under ``pipeline`` fail startup.
- ``module`` names a ``Record2Sample`` or ``Sample2Job`` subclass as ``package.module:ClassName``. Omitted stages use the recipe's class with no config.
- Put each stage's fields in its own ``config``. Stage keys under ``recipe.config`` fail startup.
- Stage config is validated when a scenario builds its pipeline. Unknown keys and out-of-range values fail then.
- The settings reference lists the keys of the built-in stages, and the page of each recipe lists the keys of its own stages. A scenario can replace keys of ``pipeline.record2sample.config`` for itself (`Record-to-sample config <http-api-scenarios.rst#record-to-sample-config>`__).
- `Job size <../developer-guide/recipe-control-service.rst#job-size>`__ describes ``job-samples`` and ``buffer-size``, and `Staleness <../developer-guide/recipe-samples-service.rst#staleness>`__ describes which samples are stale.
