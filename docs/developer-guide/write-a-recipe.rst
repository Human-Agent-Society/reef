Write a recipe
==============

A recipe defines how records become samples, jobs and artifacts.
Reef handles storage, submission, commits and recovery.
First check the `existing recipes <../user-guide/recipes.rst>`__.

Choose an extension
-------------------

- **Weights:** subclass ``WeightTrainingRecipe`` and select sample stages and an objective.
- **Harness:** use ``HarnessEvolutionRecipe`` with a `Method <write-a-harness-method.rst>`__.
- **Both:** combine recipes with `CompositeRecipe <../reference/configuration-recipe.rst#composite-recipes>`__.

Recipes are frozen configuration dataclasses. Keep running clients and workers
out of their fields: services can run separately.

.. list-table::
   :header-rows: 1

   * - Service
     - Extension
   * - Control
     - `Reports, job data and activation <recipe-control-service.rst>`__
   * - Samples
     - `Record selection and sample creation <recipe-samples-service.rst>`__
   * - Training
     - `Objectives and losses <recipe-training-service.rst>`__
   * - Every service loading a scenario
     - `Artifact declarations <components.rst>`__

A minimal weight recipe
-----------------------

This excerpt binds SDFT's report, stages and objective.
See the `complete implementation <../../recipes/sdft/recipe.py>`__ for imports.

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

Register objectives by import reference in the package's ``__init__.py``:

.. code:: python

   from reef.train.slime_backend.objectives import register_objective

   register_objective("sdft", "recipes.sdft.slime:SdftObjective")

Set ``recipe.implementation`` to the recipe's ``package.module:ClassName``.
Configure stages under ``pipeline.record2sample`` and ``pipeline.sample2job``.
SDFT requires ``sample2job.config.job_samples``.
Use the `SDFT deployment <../../recipes/sdft/examples/skill_stream/serve.yaml>`__
for backend and engine settings.

Settings and packaging
-----------------------

Declare configurable fields with ``config_field``, for example
``clip_ratio: float = config_field(0.2, env="MY_CLIP_RATIO")``.
Validate ranges in ``__post_init__``. Invalid settings raise ``RecipeConfigError``.
Use the `shared configuration loader <../../reef/recipe/__init__.py>`__.

Put methods in ``recipes/<name>/`` with a README and runnable examples.
The wheel excludes these packages; install them where every service can import them.
Record-following plug-ins must handle data removed by
`retention <../user-guide/storage.rst#record-retention>`__.
