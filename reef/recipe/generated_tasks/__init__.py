"""The built-in generated-task recipe: train the served model on the tasks that a task generator makes.

The deployment configures the three stages in its ``pipeline`` section, and
names the generator (:class:`reef.recipe.generated_tasks.TaskGenerator`) in
``pipeline.record2sample.config.generator``. The recipe does not depend on the
generator's method:

- ``generator``: the contract of the task generator: ``TaskGenerator``,
  ``TaskValidationResult`` and ``GeneratedTaskResult``, one of
  ``RejectedTask``, ``HeldOutTask``, ``RolledOutTask`` and ``FailedTask``. The request,
  ``TaskGenerationRequest``, is the generator service's
  (``reef.record2sample.generator.inputs``).
- ``processor``: the three stages. ``GeneratedTaskRecord2Record`` holds the
  source sessions and starts each task-generation round when it is due;
  ``GeneratedTaskSamples`` runs the rounds (generate, validate, send the results
  back); and ``GeneratedTaskJobData`` hands the saved tasks to the training
  component as task jobs, which the component rolls out on its own sampler.
- ``results``: each train task's first rollout result and the generations
  delivered to the generator, kept in the task store's documents.
- ``objective``: group relative advantages per task group on Tinker's
  importance sampling loss.
- ``recipe``: the objective and the report contract.

The generator service (``reef.record2sample.generator``) writes and checks the tasks; the training component
runs their training rollouts.
"""

from reef.recipe.generated_tasks.generator import (
    FailedTask,
    GeneratedTaskResult,
    HeldOutTask,
    RejectedTask,
    RolledOutTask,
    TaskGenerator,
    TaskValidationResult,
)
from reef.recipe.generated_tasks.objective import TaskGroupObjective
from reef.recipe.generated_tasks.processor import (
    GeneratedTaskJobData,
    GeneratedTaskRecord2Record,
    GeneratedTaskSamples,
)
from reef.recipe.generated_tasks.recipe import GeneratedTaskRecipe

__all__ = [
    "FailedTask",
    "GeneratedTaskJobData",
    "GeneratedTaskRecipe",
    "GeneratedTaskRecord2Record",
    "GeneratedTaskResult",
    "GeneratedTaskSamples",
    "HeldOutTask",
    "RejectedTask",
    "RolledOutTask",
    "TaskGenerator",
    "TaskGroupObjective",
    "TaskValidationResult",
]
