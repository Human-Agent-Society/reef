"""The built-in generated-task recipe: train the served model on the tasks that a task generator makes.

The deployment configures the three stages in its ``pipeline`` section, and
names the generator (:class:`reef.core.tasks.TaskGenerator`) in
``pipeline.record2task.config.generator``. The recipe does not depend on the
generator's method:

- ``processor``: the three stages. ``GeneratedTaskRecord2Record`` groups the
  training plays by task and starts each task-generation round when it is
  due; ``GeneratedTaskTasks`` runs the rounds (generate, validate, play
  through the generator service, send the results back) and turns the
  training arm's episodes into tasks; and
  ``GeneratedTaskBatches`` batches complete task groups.
- ``objective``: group relative advantages per task group on Tinker's
  importance sampling loss.
- ``recipe``: the objective, the step schedule, and the report contract.

The recipe plays tasks through the generator service (``reef.record2task.generator``).
"""

from reef.recipe.generated_tasks.objective import TaskGroupObjective
from reef.recipe.generated_tasks.processor import GeneratedTaskBatches, GeneratedTaskRecord2Record, GeneratedTaskTasks
from reef.recipe.generated_tasks.recipe import GeneratedTaskRecipe

__all__ = [
    "GeneratedTaskBatches",
    "GeneratedTaskRecipe",
    "GeneratedTaskRecord2Record",
    "GeneratedTaskTasks",
    "TaskGroupObjective",
]
