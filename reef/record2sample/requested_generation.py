"""Task generation on request: in ``manual`` mode, a client names the source sessions of each generation round.

A record-to-sample stage that generates tasks implements ``RequestedTaskGeneration``. In ``manual`` training mode
such a stage starts no round on its own. A client asks for one round
(``POST /reef/scenarios/{scenario}/task-generations``), and names the reports of the sessions that the generator
reads. Control checks the request (``Trainer.check_task_generation``) and adds a ``task_generation`` item to the
``samples`` queue (``reef.record2sample.service``). A ``TaskGenerationWorker`` of the samples service gives the
round to the stage (``Trainer.request_task_generation``) and holds the item until the round's result is ready.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from reef.core.records_types import AgentRecord
from reef.record2sample.reports import ReportContext


@dataclass(frozen=True)
class RequestedRoundResult:
    """How a requested task-generation round ended.

    Values: ``generation`` is a non-negative int, the generation of the scenario's task store that the round
    made or continued. When a complete generation already used every source session of the round, it is the
    newest such generation. It is None for a round that the stage refused to start, such as one past the round
    limit. ``train_tasks`` holds the distinct names of that generation's published train tasks, in candidate order.
    ``error`` is empty when the generation is complete. Otherwise it names why the generation ran out of its
    attempts or why the round did not start, and ``train_tasks`` is empty. The constructor refuses other values.

    Purpose: a ``RequestedTaskGeneration`` stage makes it when a requested round ends; the samples service writes it
    as the result of the request's work item, or fails the item with ``error``.
    """

    generation: int | None
    train_tasks: tuple[str, ...] = ()
    error: str = ""

    def __post_init__(self) -> None:
        if self.generation is not None and (
            isinstance(self.generation, bool) or not isinstance(self.generation, int) or self.generation < 0
        ):
            raise ValueError("generation must be None or a non-negative integer")
        if self.generation is None and not self.error:
            raise ValueError("a round that did not start names its error")
        if any(not isinstance(name, str) or not name for name in self.train_tasks) or len(
            set(self.train_tasks)
        ) != len(self.train_tasks):
            raise ValueError("train_tasks holds distinct non-empty task names")
        if not isinstance(self.error, str) or (self.error and self.train_tasks):
            raise ValueError("an incomplete generation names its error and has no train task")


class RequestedTaskGeneration(ABC):
    """A record-to-sample stage that generates tasks in rounds, and runs a round that a client requests.

    Values: the stage holds each requested round from ``request_round`` until ``take_round_result`` returns its
    result. It runs the requested rounds one at a time, in the order they came, and before any round of its own.
    In ``manual`` mode it starts no round of its own.

    Operations:

    * ``source_session(report, calls)`` is the source session of ``report`` with ``calls``, the calls of its
      trajectories, oldest first. It raises ``ValueError`` when the stage does not take the session as a source.
    * ``check_request(sources)`` raises ``ValueError`` when the stage cannot run a round of ``sources`` now: too
      many sources, no generator, or the round limit reached. It changes nothing.
    * ``request_round(request_id, sources)`` checks the request as ``check_request`` does, then holds the round.
      A request that the stage holds already is unchanged.
    * ``holds_request(request_id)`` is whether the stage holds the request.
    * ``take_round_result(request_id)`` returns the result of an ended round and stops holding the request. It
      returns None while the round waits or runs, and for a request that the stage does not hold.

    Purpose: ``Trainer.check_task_generation`` and ``Trainer.request_task_generation`` call it; the route and the
    samples service reach the stage only through the trainer.
    """

    @abstractmethod
    def source_session(self, report: AgentRecord, calls: tuple[AgentRecord, ...]) -> ReportContext: ...

    @abstractmethod
    def check_request(self, sources: tuple[ReportContext, ...]) -> None: ...

    @abstractmethod
    def request_round(self, request_id: str, sources: tuple[ReportContext, ...]) -> None: ...

    @abstractmethod
    def holds_request(self, request_id: str) -> bool: ...

    @abstractmethod
    def take_round_result(self, request_id: str) -> RequestedRoundResult | None: ...


__all__ = ["RequestedRoundResult", "RequestedTaskGeneration"]
