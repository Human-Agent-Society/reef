Write a recipe
==============

A recipe selects sample stages, training parameters and `artifact components <components.rst>`__.
Check `existing recipes <../user-guide/recipes.rst>`__ first.

Choose an extension
-------------------

- Subclass ``WeightTrainingRecipe`` for weight recipes.
- Use ``HarnessEvolutionRecipe`` with a `Method <write-a-harness-method.rst>`__ for harness recipes.
- Combine both with `CompositeRecipe <../reference/configuration-recipe.rst#composite-recipes>`__.

Recipes are frozen configuration dataclasses. Keep clients and workers outside them.
Stages exchange data through the shared store.

.. code:: mermaid

   flowchart TB
       accTitle: Recipe stages and their service owners
       accDescr: The samples service turns records into stored samples with Record2Sample. The control service turns stored samples into JobData with Sample2Job.
       records["Records"] -->|"Record2Sample<br/>samples service"| samples["Stored samples"]
       samples -->|"Sample2Job<br/>control service"| job["JobData for training"]

Extend `sample creation <recipe-samples-service.rst>`__,
`job data and activation <recipe-control-service.rst>`__, or
`training objectives <recipe-training-service.rst>`__.

A minimal weight recipe
-----------------------

Adapt `SDFT's recipe <../../recipes/sdft/recipe.py>`__ and `objective registration <../../recipes/sdft/__init__.py>`__.
Select the class in deployment YAML.

.. code:: yaml

   recipe:
     implementation: recipes.sdft.recipe:SDFTRecipe

The `runnable SDFT example <../../recipes/sdft/examples/skill_stream/README.md>`__ supplies required job sizes and backend settings.

Settings and packaging
-----------------------

Use ``config_field`` for settings. Raise ``RecipeConfigError`` for invalid values in ``__post_init__``.
Install recipes on every service because the Reef wheel excludes them.
Handle records removed by `retention <../user-guide/storage.rst#record-retention>`__.
