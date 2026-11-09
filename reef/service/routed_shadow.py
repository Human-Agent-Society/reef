"""The shadow calls of a routed deployment: their limits and the bounded set of shadow tasks that runs them.

A Scenario's ``shadow_target`` (``reef.service.routing``) makes Reef also call that target once for each routed
request whose primary target is another one. A shadow call starts after the primary target and its handler are
bound and before the primary call; the client never waits for it and never receives it. Each shadow call runs as
one task that ends only after its own record write has finished or failed, so a slow record store holds a slot as
long as a slow model. There is no queue: a request that finds no free slot skips its shadow call.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

#: What ``ShadowTasks.start`` did with a shadow call: ran it, or skipped it because no slot was free or the
#: service is stopping.
ShadowStart = Literal["started", "saturated", "closed"]


@dataclass(frozen=True)
class ShadowLimits:
    """How much shadow work one routed service runs.

    Values: ``max_concurrency`` is an int from 1 to 4096, the most shadow tasks that run at once.
    ``timeout_seconds`` is a finite positive number: how long one shadow call waits for its provider's answer; it
    does not bound the record write that follows. ``max_response_bytes`` is a positive int: the most body bytes,
    success or error, that a shadow call collects before it refuses the body unparsed. The defaults are
    conservative settings, not measured model or GPU capacities.
    Purpose: ``create_app`` gets it from ``reef.routing_deployment``; ``ShadowTasks`` reads ``max_concurrency``
    and each shadow call reads the other two.
    """

    max_concurrency: int = 4
    timeout_seconds: float = 300.0
    max_response_bytes: int = 8 * 1024 * 1024

    def __post_init__(self) -> None:
        if type(self.max_concurrency) is not int or not 1 <= self.max_concurrency <= 4096:
            raise ValueError("shadow max_concurrency must be an integer from 1 to 4096")
        if (
            type(self.timeout_seconds) not in (int, float)
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("shadow timeout_seconds must be finite and positive")
        if type(self.max_response_bytes) is not int or self.max_response_bytes <= 0:
            raise ValueError("shadow max_response_bytes must be a positive integer")


class ShadowTasks:
    """The shadow tasks of one routed service, on one event loop.

    State: ``limits`` is fixed at construction. ``tasks`` holds each task that ``start`` created until its done
    callback ``finished`` runs, so a task that has just ended stays in it until the event loop runs that
    callback; ``len(tasks) ≤ limits.max_concurrency`` always holds. ``closed`` is false until ``close`` and then
    stays true. ``skipped`` counts the calls ``start`` refused, by reason; ``cancelled`` counts tasks that ended
    cancelled, also before their first step; ``failed`` counts tasks that ended with an exception, which a shadow
    call raises only when it could not write its record.
    Operations: ``start(call)`` runs the coroutine ``call`` as a task and returns ``"started"`` when not closed
    and a slot is free; otherwise it closes ``call`` without running it and returns why. ``cancel`` cancels each
    running task that is not already cancelling, and leaves admission open: a task cancelled once keeps its slot
    while it writes its ``cancelled`` record, and a later ``cancel`` or ``close`` does not interrupt that write.
    ``close(timeout_seconds)`` closes admission, cancels like ``cancel`` and returns whether every task ended
    within ``timeout_seconds``. A task cancelled before its first step never calls its provider and writes no
    record; its slot is released all the same. The service and its request acceptance share one instance.
    """

    def __init__(self, limits: ShadowLimits) -> None:
        self.limits = limits
        self.tasks: set[asyncio.Task[None]] = set()
        self.closed = False
        self.skipped: dict[ShadowStart, int] = {"saturated": 0, "closed": 0}
        self.cancelled = 0
        self.failed = 0

    def start(self, call: Coroutine[object, object, None]) -> ShadowStart:
        if self.closed or len(self.tasks) >= self.limits.max_concurrency:
            call.close()
            reason: ShadowStart = "closed" if self.closed else "saturated"
            self.skipped[reason] += 1
            return reason
        task = asyncio.create_task(call)
        self.tasks.add(task)
        task.add_done_callback(self.finished)
        return "started"

    def finished(self, task: asyncio.Task[None]) -> None:
        self.tasks.discard(task)
        if task.cancelled():
            self.cancelled += 1
            return
        error = task.exception()
        if error is not None:
            self.failed += 1
            # The type only: an error message could repeat a provider body or a prompt.
            logger.warning("a routed shadow call ended without its record: %s", type(error).__name__)

    def cancel(self) -> None:
        for task in list(self.tasks):
            if not task.done() and not task.cancelling():
                task.cancel()

    async def close(self, timeout_seconds: float) -> bool:
        self.closed = True
        self.cancel()
        running = set(self.tasks)
        if not running:
            return True
        _, pending = await asyncio.wait(running, timeout=max(0.0, timeout_seconds))
        return not pending


__all__ = ["ShadowLimits", "ShadowStart", "ShadowTasks"]
