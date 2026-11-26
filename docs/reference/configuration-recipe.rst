Recipe configuration
====================

Put recipe fields under ``recipe.config``. Unknown fields fail validation.
Choose a ``recipe.implementation``:

- ``recipe`` (default) records requests and reports without training.
- A class such as ``my_pkg.method:MyRecipe`` loads from an installed package or a parent of the config directory. Existing ``PYTHONPATH`` entries take priority.
- A bare name loads a preset from ``$REEF_RECIPE_CONFIG_DIR/<name>.yaml``. Presets contain literal deployment YAML. They can select the default recipe or a class that does not train weights.

Only dotted classes accept top-level ``pipeline`` and ``activation`` sections.
The service validates these sections and ``recipe.config.training-mode`` at startup. ``--print-config`` skips this validation.
For backend job parameters see `Training parameters <../developer-guide/recipe-training-service.rst#parameters-that-slime-and-tinker-take>`__.

Composite recipes
-----------------

Use ``reef.recipe.composite:CompositeRecipe`` for two or more components in one scenario.
Each entry in ``recipe.config.components`` needs an ``implementation`` and ``config``.
Put its ``pipeline`` inside that component's config.

All components share a model and training mode. At most one can train weights and one can evolve a harness.
Set ``training.backend`` if a component trains weights. Otherwise set ``inference.upstream-model`` and omit ``training``.

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

Omit ``pipeline`` to use the recipe's stages. To override one, set ``pipeline.record2sample`` or ``pipeline.sample2job``.
Each accepts ``config`` and an optional ``module`` naming the stage's class.

See `job size <../developer-guide/recipe-control-service.rst#job-size>`__,
`staleness <../developer-guide/recipe-samples-service.rst#staleness>`__ and
`per-scenario overrides <http-api-scenarios.rst#record-to-sample-config>`__.
