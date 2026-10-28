"""The scenario's imported tasks as a recipe reads them: the task of one task job, and the artifact evaluation.

The tasks of the scenario's ``imported-tasks`` store arrive by ``POST /reef/scenarios/{scenario}/tasks`` or by
the deployment's ``task-import`` section. ``imported_task_item`` is the task that a task job rolls out for one of
them. ``ImportedTaskEvaluation`` evaluates every artifact that the trainer publishes on the store's eval tasks:
an evaluation job from the artifact's checkpoint, which rolls the tasks out and trains nothing. The SFT-then-RL
recipe and the generated-task recipe use them.
"""

from __future__ import annotations

from collections.abc import Mapping

from reef.core.evaluation import CheckpointCandidate
from reef.core.job_data import JobData, TaskSample
from reef.storage.tasks import IMPORTED_TASK_STORE, SplitName, StoredTask, TaskStorage, TaskStore
from reef.train.jobs import ArtifactEvaluation, EvaluationJob


def imported_task_item(store: TaskStore, stored: StoredTask) -> TaskSample:
    """The task that a task job rolls out for the imported task ``stored``: its directory, named by its task name."""
    return TaskSample(
        store.path(stored.files_digest),
        metadata={"task": {"name": stored.name, "files_digest": stored.files_digest, "split": stored.split}},
    )


class ImportedTaskEvaluation(ArtifactEvaluation):
    """Evaluate every artifact on the eval tasks of the scenario's ``imported-tasks`` store.

    Values: ``task_store`` is that store, and ``parameters`` the task job parameters (with ``rollout``),
    a mapping. Operation: ``evaluation_job(candidate)`` is ``None`` while the store holds no eval task, and else
    the job of every eval task, in publication order, with ``parameters`` and ``evaluation_of`` set to the
    candidate's ``candidate_id``, which the training component scores without training. Purpose:
    ``SFTThenRLRecipe.artifact_evaluation`` and ``GeneratedTaskRecipe.artifact_evaluation`` make it, and the
    weight trainer's job loop asks it at each decision.
    """

    imported_task_splits: frozenset[SplitName] = frozenset({"eval"})

    def __init__(self, task_store: TaskStore, parameters: Mapping[str, object]) -> None:
        self.task_store = task_store
        self.parameters = dict(parameters)

    @classmethod
    def for_storage(
        cls, task_storage: TaskStorage | None, parameters: Mapping[str, object]
    ) -> ImportedTaskEvaluation | None:
        """The evaluation over ``task_storage``'s imported tasks, or ``None`` without task storage."""
        return None if task_storage is None else cls(task_storage.open(IMPORTED_TASK_STORE), parameters)

    def evaluation_job(self, candidate: CheckpointCandidate) -> EvaluationJob | None:
        held_out = self.task_store.tasks(split="eval")
        if not held_out:
            return None
        items = tuple(imported_task_item(self.task_store, stored) for stored in held_out)
        return EvaluationJob(
            JobData(f"{candidate.candidate_id}:evaluation", items),
            {**self.parameters, "evaluation_of": candidate.candidate_id},
        )
