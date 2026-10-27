"""The generated-task recipe's feedback state: each train task's first rollout, and the generations delivered.

For a task store, let ``train(g)`` be the names of the train tasks that generation ``g`` published, and
``complete(g)`` say that every candidate of ``g`` has an outcome. Published names are unique, so a train task
name ``n`` belongs to exactly one generation. The recipe keeps two values in the store's documents, which the
store saves but does not read:

- ``rollout : N ⇀ TaskRolloutResult``, the result of the first rollout of train task ``n`` that ended, under
  the key ``rollout/<n>``. ``n ∈ dom rollout`` implies ``n ∈ train(g)`` for some ``g``, and ``rollout(n)`` is
  written once.
- ``Δ ⊆ {0, 1, 2, ...}``, the generations whose results reached the generator, under the key
  ``delivered/<g>``. ``g ∈ Δ`` implies ``complete(g)`` and ``train(g) ⊆ dom rollout``.

A generation ``g`` is deliverable when ``complete(g)``, ``train(g) ⊆ dom rollout`` and ``g ∉ Δ``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from reef.core.tasks import HarborTask
from reef.recipe.generated_tasks.generator import FailedTask, RolledOutTask, check_rewards
from reef.storage.tasks import TaskStore

ROLLOUT_KEY_PREFIX = "rollout/"
DELIVERED_KEY_PREFIX = "delivered/"


@dataclass(frozen=True)
class TaskRolloutResult:
    """The result of the first rollout of one train task that ended: its rewards, or why no episode ran.

    Values: ``(task_name, rollout_rewards, error, infrastructure_failure)`` with ``task_name`` a non-empty task
    name, in exactly one of two cases. The episodes ran: ``rollout_rewards`` is a non-empty tuple of finite
    numbers, ``error`` is empty and ``infrastructure_failure`` is false. No episode ran: ``rollout_rewards`` is
    empty and ``error`` is non-empty; ``infrastructure_failure`` says whether the task could not run at all.
    Purpose: the record-to-sample stage makes it when a train task's rollout ends, or when a restart finds a task
    that trained without one. It saves it as ``rollout(task_name)``; the delivery turns it into the generator's
    ``RolledOutTask`` or ``FailedTask`` (:meth:`generated`).
    """

    task_name: str
    rollout_rewards: tuple[float, ...] = ()
    error: str = ""
    infrastructure_failure: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.task_name, str) or not self.task_name:
            raise ValueError("a rollout result needs its task name")
        if not isinstance(self.error, str) or not isinstance(self.infrastructure_failure, bool):
            raise TypeError("error must be text and infrastructure_failure a bool")
        if self.error:
            if self.rollout_rewards != ():
                raise ValueError("a rollout that ran no episode has no rewards")
        else:
            check_rewards(self.rollout_rewards)
            if self.infrastructure_failure:
                raise ValueError("an infrastructure failure needs the error that stopped the task")

    def generated(self, task: HarborTask) -> RolledOutTask | FailedTask:
        """The generator's result for ``task``, the published train task named ``task_name``."""
        if task.name != self.task_name:
            raise ValueError(f"the rollout result of {self.task_name!r} is not the result of {task.name!r}")
        if self.error:
            return FailedTask(task, self.error, self.infrastructure_failure)
        return RolledOutTask(task, self.rollout_rewards)

    def document(self) -> str:
        """The result as canonical JSON text: an identical result gives the same text."""
        fields = {
            "task_name": self.task_name,
            "rollout_rewards": list(self.rollout_rewards),
            "error": self.error,
            "infrastructure_failure": self.infrastructure_failure,
        }
        return json.dumps(fields, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_document(cls, text: str) -> TaskRolloutResult:
        """Read the text of :meth:`document`."""
        fields = json.loads(text)
        return cls(
            task_name=fields["task_name"],
            rollout_rewards=tuple(fields["rollout_rewards"]),
            error=fields["error"],
            infrastructure_failure=fields["infrastructure_failure"],
        )


def rollout_results(store: TaskStore) -> dict[str, TaskRolloutResult]:
    """Return ``rollout``: the saved rollout result of each train task, by task name."""
    return {
        key.removeprefix(ROLLOUT_KEY_PREFIX): TaskRolloutResult.from_document(document)
        for key, document in store.documents().items()
        if key.startswith(ROLLOUT_KEY_PREFIX)
    }


def save_rollout_result(store: TaskStore, result: TaskRolloutResult) -> None:
    """Set ``rollout(result.task_name) = result``. Raises ``TaskStoreError`` if the task has another result."""
    store.write_document(key=f"{ROLLOUT_KEY_PREFIX}{result.task_name}", document=result.document())


def delivered_generations(store: TaskStore) -> frozenset[int]:
    """Return ``Δ``: the generations whose results reached the generator."""
    return frozenset(
        int(key.removeprefix(DELIVERED_KEY_PREFIX))
        for key in store.documents()
        if key.startswith(DELIVERED_KEY_PREFIX)
    )


def mark_delivered(store: TaskStore, generation: int) -> None:
    """Add ``generation`` to ``Δ``; a repeated call changes nothing."""
    store.write_document(key=f"{DELIVERED_KEY_PREFIX}{generation}", document=json.dumps({"generation": generation}))
