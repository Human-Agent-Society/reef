"""The built-in generated-task recipe: train the served model on the tasks that a task generator makes.

The deployment configures the three stages in its ``pipeline`` section, and
names the generator (:class:`reef.recipe.generated_tasks.TaskGenerator`) in
``pipeline.record2task.config.generator``. The recipe does not depend on the
generator's method:

- ``generator``: the contract of the task generator: ``TaskGenerator``,
  ``TaskValidationResult`` and ``GeneratedTaskResult``, one of
  ``RejectedTask``, ``HeldOutTask``, ``RolledOutTask`` and ``FailedTask``. The request,
  ``TaskGenerationRequest``, is the generator service's
  (``reef.record2task.generator.inputs``).
- ``processor``: the three stages. ``GeneratedTaskRecord2Record`` holds the
  source sessions and starts each task-generation round when it is due;
  ``GeneratedTaskTasks`` runs the rounds (generate, validate, send the results
  back) and each saved task's episodes through the generator service; and
  ``GeneratedTaskBatches`` rolls saved tasks out and batches the samples of
  their episodes, which are not records.
- ``results``: each train task's first rollout result and the generations
  delivered to the generator, kept in the task store's documents.
- ``objective``: group relative advantages per task group on Tinker's
  importance sampling loss.
- ``recipe``: the objective, the step schedule, and the report contract.

The recipe runs the episodes of its tasks through the generator service (``reef.record2task.generator``).
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
from reef.recipe.generated_tasks.processor import GeneratedTaskBatches, GeneratedTaskRecord2Record, GeneratedTaskTasks
from reef.recipe.generated_tasks.recipe import GeneratedTaskRecipe

__all__ = [
    "FailedTask",
    "GeneratedTaskBatches",
    "GeneratedTaskRecipe",
    "GeneratedTaskRecord2Record",
    "GeneratedTaskResult",
    "GeneratedTaskTasks",
    "HeldOutTask",
    "RejectedTask",
    "RolledOutTask",
    "TaskGenerator",
    "TaskGroupObjective",
    "TaskValidationResult",
]
