Python API for custom learning methods
======================================

Use ``reef_client.ReefClient`` to send requests and report feedback. The `quickstart <../getting-started/quickstart.rst>`__ shows a complete example.

Use the interfaces below to extend Reef itself. Inherit the documented abstract base class for each capability.
Its docstrings define the methods and their contracts. Other internal classes can change.

Learning methods
----------------

- `Recipe <../developer-guide/write-a-recipe.rst>`__ selects the pipeline and components for a scenario.
- `Record2Sample <../developer-guide/recipe-samples-service.rst>`__ selects records and makes samples.
  `Sample2Job <../developer-guide/recipe-control-service.rst>`__ groups samples into a training job.
- A `training objective <../developer-guide/recipe-training-service.rst>`__ prepares the training signal.
  A `loss family <../developer-guide/loss-families.rst>`__ computes the loss.
- A harness `Method <../developer-guide/write-a-harness-method.rst>`__ writes a candidate tree in the job workspace.
- `ComponentSpec <../developer-guide/components.rst>`__ declares a component in a scenario snapshot.

Integrations and plugins
------------------------

Select plugins by built-in name or ``package.module:ClassName``.

- ``training.backend`` selects a `TrainingDeployment <../developer-guide/training-backends.rst>`__. Its ``TrainingComponent`` runs jobs.
- `InferenceRuntime <../developer-guide/serving-runtimes.rst>`__ binds artifacts to serving endpoints.
- `Executor <../developer-guide/executors.rst>`__ starts workers. Configure the executor for each role under ``execution``.
- `TaskGenerator <../developer-guide/task-generators.rst>`__ builds tasks from agent activity.
- `Harness adapters <../developer-guide/harness-adapters.rst>`__ use descriptor files to define how agents install and run a harness.

For other integration points, see `storage backends <../user-guide/storage.rst>`__ and
`deployment settings <configuration-settings.rst>`__.
