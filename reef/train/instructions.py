"""Training instructions in the instructions queue, not in the record store (RFC #1054).

A training instruction (``POST /reef/train``) is control's work, not an observed record. Each instruction is one item
of the ``instructions`` work queue (``reef.storage.work_queue``):

- ``id`` is the instruction ID: the client's ``agent_record_id``, or a new one;
- ``scenario`` and ``serial_key`` are its scenario, so the instructions of one scenario settle in the order they came;
- ``kind`` is ``instruction_kind(scenario)``, so control takes only the items of the scenario that it settles;
- ``payload`` is the normalized request (``normalize_training_request_payload``).

An instruction is queued until a training commit consumes it; then control finishes its item. A commit records the
instruction in its ``metrics["training_request"]``, so the history of an instruction outlives its queue item, which
control deletes after the work retention.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, cast

from reef.core.training_request import TrainingRequest
from reef.storage.records import RecordConflict
from reef.storage.work_queue import JsonValue, NewWork, WorkConflict, WorkItem, WorkQueue
from reef.train.trainer import Trainer

logger = logging.getLogger(__name__)

#: The queue of training instructions.
INSTRUCTIONS_QUEUE = "instructions"


def instruction_kind(scenario: str) -> str:
    """The work-item kind of the instructions of ``scenario``."""
    return f"train_instruction:{scenario}"


@dataclass(frozen=True)
class InstructionQueue:
    """The instructions queue of every scenario of one deployment.

    Values: ``queue`` is the work queue store that holds ``INSTRUCTIONS_QUEUE``.
    Purpose: the dispatcher makes it from its coordination store and gives it to the scenario factory, which gives
    each scenario its own instructions (``of``).
    """

    queue: WorkQueue

    def of(self, scenario: str) -> ScenarioInstructions:
        """The instructions of ``scenario``."""
        return ScenarioInstructions(self.queue, scenario)


@dataclass(frozen=True)
class ScenarioInstructions:
    """The training instructions of one scenario, as items of the instructions queue.

    Values: ``queue`` is the work queue store; ``scenario`` a non-empty scenario name; ``worker_id`` the worker that
    takes an item to finish it. The items are those of ``INSTRUCTIONS_QUEUE`` with this scenario, as the module
    docstring defines them.
    Purpose: the dispatcher files instructions (``file``); a scenario gives the queued ones to its trainers and
    finishes those that a commit consumed (``give``); the request page reads one (``get``).
    """

    queue: WorkQueue
    scenario: str
    worker_id: str = "control"

    def file(self, request_id: str, payload: Mapping[str, Any]) -> WorkItem:
        """Queue the instruction ``request_id`` with the normalized ``payload``; return its item.

        The same ID with the same payload returns the item that exists; with another payload it raises
        ``RecordConflict``, as a record with the same ID and other content does.
        """
        work = NewWork(
            INSTRUCTIONS_QUEUE,
            request_id,
            self.scenario,
            instruction_kind(self.scenario),
            cast(dict[str, JsonValue], dict(payload)),
            serial_key=self.scenario,
        )
        try:
            return self.queue.add(work)
        except WorkConflict as exc:
            raise RecordConflict(f"instruction {request_id!r} already has different content") from exc

    def get(self, request_id: str) -> WorkItem | None:
        """The item of the instruction ``request_id`` of this scenario, in any state, or None."""
        item = self.queue.get(INSTRUCTIONS_QUEUE, request_id)
        return item if item is not None and item.scenario == self.scenario else None

    def queued(self) -> tuple[TrainingRequest, ...]:
        """The instructions that no commit consumed yet, oldest first."""
        return tuple(
            TrainingRequest.from_dict(item.payload, request_id=item.id)
            for item in self.queue.items(
                INSTRUCTIONS_QUEUE, scenario=self.scenario, kind=instruction_kind(self.scenario)
            )
            if not item.final
        )

    def settle(self, consumed: Iterable[str]) -> None:
        """Finish the items of the queued instructions whose IDs are in ``consumed``, oldest first.

        Instructions train in the order they came, so the consumed ones are the oldest queued; settling stops at
        the first queued instruction that is not consumed.
        """
        consumed_ids = set(consumed)
        for request in self.queued():
            if request.id not in consumed_ids:
                return
            item = self.queue.take(INSTRUCTIONS_QUEUE, self.worker_id, (instruction_kind(self.scenario),))
            if item is None or item.id != request.id:
                # Another control holds it; its lease returns it, and the next settle finishes it.
                logger.warning(
                    "scenario %r: instruction %s is held elsewhere; it settles later", self.scenario, request.id
                )
                return
            self.queue.finish(INSTRUCTIONS_QUEUE, item.id, self.worker_id, {"consumed": True})

    def give(self, trainers: Iterable[Trainer]) -> None:
        """Give every trainer of ``trainers`` the queued instructions that none of them consumed, and finish the items
        of those that one consumed (``settle``). A trainer's commit, or its recovery, consumes an instruction."""
        trainers = tuple(trainers)
        queued = self.queued()
        ids = [request.id for request in queued]
        consumed: set[str] = set()
        for trainer in trainers:
            consumed |= trainer.consumed_among(ids)
        self.settle(consumed)
        pending = tuple(request for request in queued if request.id not in consumed)
        for trainer in trainers:
            trainer.queue_instructions(pending)


__all__ = ["INSTRUCTIONS_QUEUE", "InstructionQueue", "ScenarioInstructions", "instruction_kind"]
