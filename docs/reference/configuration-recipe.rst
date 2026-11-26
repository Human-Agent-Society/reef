Recipe configuration
====================

Put recipe fields under ``recipe.config``. Unknown fields fail validation.
Choose a ``recipe.implementation``:

- ``recipe`` (default) records requests and reports without training.
- A class such as ``my_pkg.method:MyRecipe`` loads from an installed package or a parent of the config directory. Existing ``PYTHONPATH`` entries take priority.
- A bare name loads a preset from ``$REEF_RECIPE_CONFIG_DIR/<name>.yaml``. Set that directory explicitly.

Only dotted classes accept top-level ``pipeline`` and ``activation`` sections.
The service validates these sections and ``recipe.config.training-mode`` at startup. ``--print-config`` skips this validation.
For backend job parameters see `Training parameters <../developer-guide/recipe-training-service.rst#parameters-that-slime-and-tinker-take>`__.

Presets
-------

A preset uses deployment YAML with literal values. It can select ``recipe`` or a dotted class that does not train weights.
Put stage settings in the preset's own ``pipeline``.

A preset inherits the deployment runtime. To give it a separate runtime, set ``recipe.runtime.type``
to ``inference_proxy``, ``tinker`` or a ``RuntimeFactory`` class. Include that runtime's settings and either ``inference.upstream-model`` or ``inference.model-path``.

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

Use at least two components. At most one trains weights and one evolves a harness. Nested composites and the names ``train``, ``step``, ``reef`` and ``operations`` are invalid.
All components share the model, runtime, training component and training mode.

With a weight component, set ``training.backend``. A top-level ``pipeline`` replaces that component's pipeline.
Otherwise set ``inference.upstream-model``, omit ``training``, and configure pipelines per component.

The ``activation`` section
--------------------------

By default, ``always`` activates new training artifacts. Use ``never`` to keep them inactive.
A scenario can override the rule through `scenario updates <http-api-scenarios.rst#scenario-updates>`__.

.. code:: yaml

   activation:
     module: evaluated
     config: {min_mean_score: 0.8}

``evaluated`` requires a minimum mean score and `automatic evaluation <http-api-evaluation.rst#automatic-evaluation>`__.
Without a scenario evaluation, Reef logs a warning and leaves new artifacts inactive.
For ``always``, use ``review_kinds`` to require manual activation for selected harness part kinds supported by the adapter.
``never`` takes no config. A custom ``ActivationRule`` receives the config in ``from_config``. Use underscores in config keys.

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

``record2sample`` and ``sample2job`` each accept ``config`` and an optional ``module``.
The module names a ``Record2Sample`` or ``Sample2Job`` subclass. Omitted stages use the recipe defaults.
Misplaced stage keys fail startup. Invalid stage config fails when a scenario builds its pipeline.

See `job size <../developer-guide/recipe-control-service.rst#job-size>`__,
`staleness <../developer-guide/recipe-samples-service.rst#staleness>`__ and
`per-scenario overrides <http-api-scenarios.rst#record-to-sample-config>`__.
