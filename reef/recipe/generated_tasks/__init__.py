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
  sessions, starts each task-generation round when it is due, and hands the saved train tasks
  to the sample store; and
  ``GeneratedTaskJobData`` makes task jobs of the generated tasks, which the
  training component rolls out on its own sampler. ``GenerationRunner`` owns
  generation, validation, task publication and result delivery on the stage's worker.
- ``results``: each train task's first rollout result and the generations
  delivered to the generator, kept in the task store's documents.
- ``recipe``: the report contract. The deployment names the objective in the
  recipe's ``job_parameters``, as a dotted reference to a method that it provides.

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
from reef.recipe.generated_tasks.processor import GeneratedTaskJobData, GeneratedTaskSamples, GeneratedTaskSettings
from reef.recipe.generated_tasks.recipe import GeneratedTaskRecipe

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
    "TaskValidationResult",
]
