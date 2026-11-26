Write a recipe
==============

A recipe selects how records become samples, training jobs and new artifacts.
It is a frozen configuration dataclass. Reef runs its plug-ins and handles
record storage, job submission, commits and recovery.

Start with `Choose a recipe <../user-guide/recipes.rst>`__ if an existing
method may fit your task.

What a recipe is
----------------

Subclass ``WeightTrainingRecipe`` for a weight method. For harness evolution,
use ``HarnessEvolutionRecipe`` and supply a ``Method`` class.

Keep live clients and workers out of recipe fields. Services can run in
separate processes, so plug-ins must not depend on shared process memory.
The deployment's activation rule decides which artifacts serve.

Which service calls which plug-in
---------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Service
     - Extension
     - Contract
   * - ``control``
     - Report validation, job data and activation
     - `Control service <recipe-control-service.rst>`__
   * - ``samples``
     - Record selection, sample creation and staleness
     - `Samples service <recipe-samples-service.rst>`__
   * - ``training``
     - Weight objective and loss, or harness method
     - `Training service <recipe-training-service.rst>`__;
       `harness methods <write-a-harness-method.rst>`__
   * - Each service that loads a scenario
     - Artifact declarations
     - `Scenario components <components.rst>`__

The normal path is:

.. code:: text

   report → record-to-sample stage → sample-to-job stage
          → training job → artifact → activation rule

Record retention can remove data between reads. Plug-ins must handle missing
records when they follow report references. See `Record retention
<../user-guide/storage.rst#record-retention>`__.

Choose the kind of method
-------------------------

- **Weights:** implement the recipe, sample stages and a backend objective.
  Add a `loss family <loss-families.rst>`__ when existing losses do not fit.
- **Harness:** implement ``Method.run`` to edit a source tree. See
  `Write a harness method <write-a-harness-method.rst>`__.
- **Both:** use ``CompositeRecipe`` to bind one recipe per component.
  See `Recipe configuration <../reference/configuration-recipe.rst#recipe-configuration>`__.

A new method usually reuses a training backend and inference runtime.
Only implement those interfaces when adding an integration.

A minimal weight recipe
-----------------------

The SDFT recipe binds a report type, two stages and an objective.
This excerpt omits imports; the complete implementation is
``recipes/sdft/recipe.py``:

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

Register the objective by import reference in the package's ``__init__.py``.
This keeps backend imports out of services that only load the recipe:

.. code:: python

   from reef.train.slime_backend.objectives import register_objective

   register_objective("sdft", "recipes.sdft.slime:SdftObjective")

Select the recipe and set its job size:

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

Set ``job_samples`` explicitly for this recipe. Native training flags must match the
`loss family <loss-families.rst#family-to-slime-flags>`__.
Use ``recipes/sdft/examples/skill_stream/serve.yaml`` for a complete deployment.

A minimal harness method
------------------------

A harness method edits ``job.workspace.harness``:

.. code:: python

   from reef.train.harness_backend import Method, MethodJob


   class MyMethod(Method):
       def run(self, job: MethodJob) -> None:
           ...  # Edit the harness tree. An unchanged tree skips the job.

Select ``reef.recipe.harness:HarnessEvolutionRecipe`` and set
``recipe.config.evolution.method`` to ``methods.my_method:MyMethod``.
Configure its seed, adapter and evaluation as described in
`Write a harness method <write-a-harness-method.rst>`__.

Recipe settings
---------------

Use ``config_field`` to declare a YAML key, CLI flag, environment fallback
and parser together:

.. code:: python

   from dataclasses import dataclass

   from reef.recipe import WeightTrainingRecipe, config_field


   @dataclass(frozen=True)
   class MyMethodRecipe(WeightTrainingRecipe):
       clip_ratio: float = config_field(0.2, env="MY_CLIP_RATIO")
       temperature: float = config_field(0.5, allow_nonfinite=False)
       tags: tuple[str, ...] = config_field(())

Supported types are ``str``, ``int``, ``float``, ``bool``, ``tuple[str, ...]``,
string-keyed object mappings, and their optional forms.
Use ``config_field(default_factory=dict)`` for an object default.
Put range and cross-field checks in ``__post_init__``.
Unknown keys, invalid types and failed checks raise ``RecipeConfigError``
when the deployment loads.

For larger configurations:

- ``section_field(path, required=...)`` parses a section into its annotated
  ``Settings`` dataclass.
- ``reader_field(reader)`` uses a ``ConfigReader`` to read configuration,
  such as a referenced file. It must return configuration, not a running object.
- ``section_defaults`` supplies defaults that deployment values override.
- ``refused_sections`` maps unsupported section names to error messages.

Use the shared recipe loader; do not add a loader to the recipe class.
For an external judge or other service, declare its endpoint and implement
the client in the method package. Reef does not manage that service.

The rest of the deployment config
---------------------------------

Use `Settings reference <../reference/configuration-settings.rst>`__ for
inference, training, execution and storage settings. Complete configurations
appear in `Evolve your model <../user-guide/evolve-your-model.rst>`__ and
`Evolve your harness <../user-guide/evolve-your-harness.rst>`__.

Where the feedback comes from
-----------------------------

Reports carry feedback from your task's existing judge. Use a consistent
numeric scale where higher scores are better. A method can instead
`compute feedback <recipe-samples-service.rst#computed-feedback>`__ or train
without scores, as SFT does.

Package a method
----------------

Put repository methods under ``recipes/<name>/``, with a README and runnable
``examples/``. The Reef wheel does not include these packages.
Install your method where every deployment service can import it.
Select it with ``package.module:ClassName``.

Pipeline stages can log finite numeric metrics with the scenario's experiment
logger. ``Record2Sample`` receives it through ``context.experiment_logger``;
``Sample2Job`` exposes ``self.experiment_logger``:

.. code:: python

   self.experiment_logger.log({"accepted": 12}, namespace="processor")

The logger drops nonfinite values and does nothing when tracking is disabled.
Each namespace has its own ``<namespace>/event`` axis.
