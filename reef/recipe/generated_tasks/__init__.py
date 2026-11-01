"""The built-in generated-task recipe: train the served model on the tasks that a task generator makes.

The deployment configures the two stages in its ``pipeline`` section, and
names the generator (:class:`reef.recipe.generated_tasks.TaskGenerator`) in
``pipeline.record2sample.config.generator``. The recipe does not depend on the
generator's method:

- ``generator``: the contract of the task generator: ``TaskGenerator``,
  ``TaskValidationResult`` and ``GeneratedTaskResult``, one of
  ``RejectedTask``, ``HeldOutTask``, ``RolledOutTask`` and ``FailedTask``. The request,
  ``TaskGenerationRequest``, is the generator service's
  (``reef.record2sample.generator.inputs``).
- ``processor``: the two stages. ``GeneratedTaskSamples`` holds the source
  sessions, starts each task-generation round when it is due, runs the rounds
  (generate, validate, send the results back), and hands the saved train tasks
  to the sample store; and
  ``GeneratedTaskJobData`` makes task jobs of the buffer's tasks, which the
  training component rolls out on its own sampler.
- ``results``: each train task's first rollout result and the generations
  delivered to the generator, kept in the task store's documents.
- ``objective``: group relative advantages per task group on the
  ``importance_sampling`` loss family: Tinker's built-in loss, or on Slime
  ``reef.train.slime_backend.importance_sampling``, whose reference this
  package registers.
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
from reef.recipe.generated_tasks.processor import GeneratedTaskJobData, GeneratedTaskSamples, GeneratedTaskSettings
from reef.recipe.generated_tasks.recipe import GeneratedTaskRecipe
from reef.train.algos.registry import register_loss_family_ref

# Slime's implementation of the objective's loss family, by reference: only the training driver and the workers
# import the Slime code.
register_loss_family_ref(
    "importance_sampling", "reef.train.slime_backend.importance_sampling:ImportanceSamplingAlgorithm"
)

__all__ = [
    "FailedTask",
    "GeneratedTaskJobData",
    "GeneratedTaskRecipe",
    "GeneratedTaskResult",
    "GeneratedTaskSamples",
    "GeneratedTaskSettings",
    "HeldOutTask",
    "RejectedTask",
    "RolledOutTask",
    "TaskGenerator",
    "TaskGroupObjective",
    "TaskValidationResult",
]
