Write a recipe
==============

A recipe selects how records become samples, jobs and artifacts.
Reef handles storage, submission, commits and recovery.
First check whether an `existing recipe <../user-guide/recipes.rst>`__ fits.

What a recipe is
----------------

A recipe is a frozen configuration dataclass. Subclass
``WeightTrainingRecipe`` for weights, or configure ``HarnessEvolutionRecipe``
with a ``Method`` for harness updates.
Keep clients and workers out of recipe fields: services can run separately.
The activation rule decides which artifacts serve.

Which service calls which plug-in
---------------------------------

.. list-table::
   :header-rows: 1

   * - Service
     - Extension contract
   * - ``control``
     - `Report validation, job data and activation <recipe-control-service.rst>`__
   * - ``samples``
     - `Record selection, samples and staleness <recipe-samples-service.rst>`__
   * - ``training``
     - `Objectives and losses <recipe-training-service.rst>`__ or `harness methods <write-a-harness-method.rst>`__
   * - Every service loading a scenario
     - `Artifact declarations <components.rst>`__

.. code:: text

   report → samples → job → artifact → activation

Plug-ins must handle records removed by
`retention <../user-guide/storage.rst#record-retention>`__.

Choose the kind of method
-------------------------

Weight methods usually reuse a backend and inference runtime.
Add a `loss family <loss-families.rst>`__ only when existing losses do not fit.
Use `CompositeRecipe <../reference/configuration-recipe.rst#composite-recipes>`__
to combine weight and harness methods.

A minimal weight recipe
-----------------------

This excerpt binds SDFT's report, stages and objective. Imports and complete
code are in `the SDFT recipe <../../recipes/sdft/recipe.py>`__:

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

Register the objective by import reference in the package's ``__init__.py``:

.. code:: python

   from reef.train.slime_backend.objectives import register_objective

   register_objective("sdft", "recipes.sdft.slime:SdftObjective")

Select the recipe and set its required job size:

.. code:: yaml

   recipe:
     implementation: recipes.sdft.recipe:SDFTRecipe
   pipeline:
     sample2job:
       config:
         job_samples: 32
   training:
     backend: slime
     options:
       global-batch-size: 32
       loss-type: custom_loss

Use the `complete deployment
<../../recipes/sdft/examples/skill_stream/serve.yaml>`__ for engine settings.
Native training flags must match the `loss family <loss-families.rst#family-to-slime-flags>`__.

A minimal harness method
------------------------

Implement ``Method.run(job)`` to edit ``job.workspace.harness``.
An unchanged tree skips the job.
Select the class through ``recipe.config.evolution.method``.
See the runnable example in `Write a harness method <write-a-harness-method.rst>`__.

Recipe settings
---------------

Declare fields with ``config_field`` for YAML, CLI and environment parsing:

.. code:: python

   from dataclasses import dataclass
   from reef.recipe import WeightTrainingRecipe, config_field


   @dataclass(frozen=True)
   class MyMethodRecipe(WeightTrainingRecipe):
       clip_ratio: float = config_field(0.2, env="MY_CLIP_RATIO")

Validate ranges and cross-field constraints in ``__post_init__``.
Invalid configuration raises ``RecipeConfigError`` when loaded.
Use the shared loader; do not add a recipe-specific loader.
The `configuration contracts <../../reef/recipe/__init__.py>`__ describe
supported types, nested sections, readers and defaults.

The rest of the deployment config
---------------------------------

Declare external service endpoints in configuration; implement clients in
your method package. Reef does not manage those services.
Use the `Settings reference <../reference/configuration-settings.rst>`__
for inference, training, execution and storage.

Where the feedback comes from
-----------------------------

Reports carry your judge's feedback. Use a consistent scale where higher
scores are better. Methods can instead
`compute feedback <recipe-samples-service.rst#computed-feedback>`__ or train
without scores, as SFT does.

Package a method
----------------

Put repository methods in ``recipes/<name>/``, with a README and runnable
``examples/``. The wheel excludes these packages.
Install your method where every service can import ``package.module:ClassName``.

Stages can log finite numeric metrics through their ``experiment_logger``;
``Record2Sample`` accesses it through ``context``.
Tracking-disabled loggers do nothing. Each namespace has its own event axis.
