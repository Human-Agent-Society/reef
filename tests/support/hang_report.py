"""Temporary diagnostics for the serving-gate flake: a timeline of serving and training events, and a hang report.

A test installs ``Timeline`` on its module (an autouse fixture) and adds ``hang_report`` to the message of a wait
that timed out. The report holds the timeline, the stack of every thread, the scenario's serving work, its route
and its artifact status.
"""

from __future__ import annotations

import sys
import threading
import time
import traceback
from collections.abc import Callable
from typing import Any

import pytest

from reef.core.served_models import ScenarioModel
from reef.dispatcher import Dispatcher
from reef.storage.served_models import SERVING_QUEUE


class Timeline:
    """Timestamped events of one test, from every thread."""

    def __init__(self) -> None:
        self.start = time.monotonic()
        self.lock = threading.Lock()
        self.events: list[str] = []
        self.last: dict[str, str] = {}

    def add(self, text: str, *, dedupe_key: str | None = None) -> None:
        with self.lock:
            if dedupe_key is not None:
                if self.last.get(dedupe_key) == text:
                    return
                self.last[dedupe_key] = text
            self.events.append(f"[{time.monotonic() - self.start:8.3f}] {threading.current_thread().name}: {text}")

    def wrap(self, monkeypatch: pytest.MonkeyPatch, owner: type, name: str, describe: Callable[..., str | None]):
        original = getattr(owner, name)
        timeline = self

        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                result = original(self, *args, **kwargs)
            except BaseException as exc:
                timeline.add(f"{owner.__name__}.{name} raised {type(exc).__name__}: {exc}")
                raise
            text = describe(self, args, kwargs, result)
            if text is not None:
                timeline.add(f"{owner.__name__}.{name} {text}", dedupe_key=f"{owner.__name__}.{name}")
            return result

        monkeypatch.setattr(owner, name, wrapper)

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from reef.dispatcher.served import ServedModelsPublisher
        from reef.dispatcher.serving import ServingWorker
        from reef.scenario.committer import ScenarioCommitter
        from reef.scenario.scenario import Scenario
        from reef.storage.sql_coordination import SQLServedModels

        self.wrap(
            monkeypatch,
            SQLServedModels,
            "request",
            lambda s, a, k, r: f"{a[0]} -> {r.id} {r.state} artifact={dict(a[1]).get('artifact_id')}",
        )
        self.wrap(
            monkeypatch,
            SQLServedModels,
            "complete_request",
            lambda s, a, k, r: f"{a[0].id} routes={sorted(r.result or {})}",
        )
        self.wrap(monkeypatch, ServingWorker, "run_item", lambda s, a, k, r: f"{a[0].id} {a[0].kind} ran")
        self.wrap(monkeypatch, ServedModelsPublisher, "run_once", lambda s, a, k, r: f"held={r}")
        self.wrap(monkeypatch, ScenarioCommitter, "commit", lambda s, a, k, r: f"step={s.step}")

        def gate(scenario: Any, args: Any, kwargs: Any, result: Any) -> str:
            if result is not None:
                return "result"
            return f"None served={scenario.served_artifact_id()} selected={scenario.serving_artifact_id()}"

        self.wrap(monkeypatch, Scenario, "prepare_training_step", gate)


TIMELINE: Timeline | None = None


def install_timeline(monkeypatch: pytest.MonkeyPatch) -> None:
    global TIMELINE
    TIMELINE = Timeline()
    TIMELINE.install(monkeypatch)


def hang_report(dispatcher: Dispatcher, scenario: str) -> str:
    """The timeline, every thread's stack, and the scenario's serving work, route and artifact status."""
    lines = ["", "===== HANG REPORT ====="]
    if TIMELINE is not None:
        lines.append("--- timeline")
        lines.extend(TIMELINE.events[-200:])
    try:
        routes = dispatcher.coordination.served_models
        lines.append(f"--- current request: {routes.current_request(scenario)!r}")
        lines.append(f"--- route: {routes.read(ScenarioModel(scenario))!r}")
        for item in dispatcher.queue.items(SERVING_QUEUE, scenario=scenario)[-10:]:
            lines.append(
                f"--- serving item {item.id} kind={item.kind} state={item.state} attempts={item.attempt_count} "
                f"error={item.error!r} artifact={dict(item.payload).get('artifact_id')}"
            )
        current = dispatcher.get_or_create_scenario(scenario)
        lines.append(f"--- artifact_status: {current.artifact_status()}")
        lines.append(f"--- served_artifact_id: {current.served_artifact_id()}")
    except Exception as exc:
        lines.append(f"--- state read failed: {type(exc).__name__}: {exc}")
    names = {thread.ident: thread.name for thread in threading.enumerate()}
    for ident, frame in sys._current_frames().items():
        lines.append(f"--- thread {names.get(ident, ident)}")
        lines.extend(line.rstrip() for line in traceback.format_stack(frame)[-12:])
    return "\n".join(lines)
