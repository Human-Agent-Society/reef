"""The generated-task recipe's feedback state: each train task's first rollout, the generations delivered, and the
Designer replies that train.

For a task store, let ``train(g)`` be the names of the train tasks that generation ``g`` published, and
``complete(g)`` say that every candidate of ``g`` has an outcome. Published names are unique, so a train task
name ``n`` belongs to exactly one generation. The recipe keeps these values in the store's documents, which the
store saves but does not read:

- ``rollout : N ⇀ TaskRolloutResult``, the result of the first rollout of train task ``n`` that ended, under
  the key ``rollout/<n>``. ``n ∈ dom rollout`` implies ``n ∈ train(g)`` for some ``g``, and ``rollout(n)`` is
  written once.
- ``Δ ⊆ {0, 1, 2, ...}``, the generations whose results reached the generator, under the key
  ``delivered/<g>``. ``g ∈ Δ`` implies ``complete(g)`` and ``train(g) ⊆ dom rollout``.
- ``designer``, under the key ``designer/<g>``, only when the stage carries Designer replies: ``designer(g)`` is the
  ``DesignerReplies`` of generation ``g``, the advantages that the generator gave the Designer replies of ``g`` after
  it took the results of ``g``, and the delayed sample of each such reply. The stage writes ``designer(g)`` before it
  marks ``g`` delivered, so a delivered generation has ``designer(g)`` when the stage carried Designer replies at
  that time.

A generation ``g`` is deliverable when ``complete(g)``, ``train(g) ⊆ dom rollout`` and ``g ∉ Δ``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass

from reef.core.job_data import DelayedSample
from reef.core.tasks import HarborTask
from reef.recipe.generated_tasks.generator import FailedTask, RolledOutTask, check_rewards
from reef.storage.tasks import TaskStore

ROLLOUT_KEY_PREFIX = "rollout/"
DELIVERED_KEY_PREFIX = "delivered/"
DESIGNER_KEY_PREFIX = "designer/"


@dataclass(frozen=True)
class TaskRolloutResult:
    """The result of the first rollout of one train task that ended: its rewards, or why no rollout ran.

    Values: ``(task_name, rollout_rewards, error, infrastructure_failure)`` with ``task_name`` a non-empty task
    name, in exactly one of two cases. The rollouts ran: ``rollout_rewards`` is a non-empty tuple of finite
    numbers, ``error`` is empty and ``infrastructure_failure`` is false. No rollout ran: ``rollout_rewards`` is
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
                raise ValueError("a rollout that ran no rollout has no rewards")
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
        for key, document in store.documents(prefix=ROLLOUT_KEY_PREFIX).items()
    }


def save_rollout_result(store: TaskStore, result: TaskRolloutResult) -> None:
    """Set ``rollout(result.task_name) = result``. Raises ``TaskStoreError`` if the task has another result."""
    store.write_document(key=f"{ROLLOUT_KEY_PREFIX}{result.task_name}", document=result.document())


def delivered_generations(store: TaskStore) -> frozenset[int]:
    """Return ``Δ``: the generations whose results reached the generator."""
    return frozenset(
        int(key.removeprefix(DELIVERED_KEY_PREFIX)) for key in store.documents(prefix=DELIVERED_KEY_PREFIX)
    )


def mark_delivered(store: TaskStore, generation: int) -> None:
    """Add ``generation`` to ``Δ``; a repeated call changes nothing."""
    store.write_document(key=f"{DELIVERED_KEY_PREFIX}{generation}", document=json.dumps({"generation": generation}))


def checked_advantages(advantages: object) -> dict[str, float]:
    """Return ``advantages`` as a dict of floats; raise ``ValueError`` unless it maps non-empty record IDs to finite
    numbers."""
    if not isinstance(advantages, Mapping):
        raise ValueError("Designer advantages must map record IDs to numbers")
    checked: dict[str, float] = {}
    for record_id, advantage in advantages.items():
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("a Designer advantage needs the ID of its inference record")
        if isinstance(advantage, bool) or not isinstance(advantage, (int, float)) or not math.isfinite(advantage):
            raise ValueError(f"the Designer advantage of record {record_id!r} must be a finite number")
        checked[record_id] = float(advantage)
    return checked


@dataclass(frozen=True)
class DesignerReplies:
    """The Designer replies of one generation that train: their advantages, and their delayed samples.

    Values: ``advantages`` maps the ID of the inference record of each Designer reply that trains to its advantage, a
    finite float, as the generator gave it. ``samples`` holds one ``DelayedSample`` for each reply of ``advantages``
    whose record the stage held with its exact sample, in record ID order: the trajectory of that record alone,
    which names the record as its only source, with the reply's advantage. A reply without a sample does not train.
    The constructor refuses other values.
    Purpose: ``GenerationRunner`` saves it as ``designer(g)`` when the generator took the results of ``g``;
    ``GeneratedTaskJobData`` reads its samples into the data of the task job that carries them.
    """

    advantages: Mapping[str, float]
    samples: tuple[DelayedSample, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "advantages", checked_advantages(self.advantages))
        if not isinstance(self.samples, tuple) or not all(isinstance(item, DelayedSample) for item in self.samples):
            raise TypeError("DesignerReplies.samples must be a tuple of DelayedSample values")
        record_ids = [item.trajectory.source_agent_record_ids for item in self.samples]
        if any(len(sources) != 1 or sources[0] not in self.advantages for sources in record_ids):
            raise ValueError("each Designer sample names exactly one record that has an advantage")
        if [sources[0] for sources in record_ids] != sorted({sources[0] for sources in record_ids}):
            raise ValueError("the Designer samples are in record ID order, each record once")
        if any(item.advantage != self.advantages[item.trajectory.source_agent_record_ids[0]] for item in self.samples):
            raise ValueError("a Designer sample trains with the advantage of its reply")

    def document(self) -> str:
        """The replies as canonical JSON text: equal replies give the same text."""
        fields = {"advantages": dict(self.advantages), "samples": [item.to_dict() for item in self.samples]}
        return json.dumps(fields, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_document(cls, text: str) -> DesignerReplies:
        """Read the text of :meth:`document`."""
        fields = json.loads(text)
        return cls(fields["advantages"], tuple(DelayedSample.from_dict(entry) for entry in fields["samples"]))


def designer_replies(store: TaskStore, generation: int) -> DesignerReplies | None:
    """Return ``designer(generation)``, or None when the store has none."""
    document = store.document(f"{DESIGNER_KEY_PREFIX}{generation}")
    return None if document is None else DesignerReplies.from_document(document)


def save_designer_replies(store: TaskStore, generation: int, replies: DesignerReplies) -> None:
    """Set ``designer(generation) = replies``. Raises ``TaskStoreError`` if the generation has other replies."""
    store.write_document(key=f"{DESIGNER_KEY_PREFIX}{generation}", document=replies.document())
