"""Bounded Ray health probes and identity-safe retirement of SGLang engine groups."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import suppress
from typing import Any

import ray

from reef.runtime.health_monitor import EngineHealthTarget


class SGLangEngineHealthChecks:
    """Snapshot node-0 probe targets with every node of the same logical engine."""

    def __init__(self, group: Any) -> None:
        self._group = group

    def targets(self) -> Sequence[EngineHealthTarget]:
        group = self._group
        width = group.nodes_per_engine
        return [
            _SGLangEngineHealthTarget(group, offset, tuple(group.all_engines[offset : offset + width]))
            for offset in range(0, len(group.all_engines), width)
            if group.all_engines[offset] is not None
        ]


class _SGLangEngineHealthTarget:
    def __init__(self, group: Any, offset: int, engines: tuple[Any, ...]) -> None:
        self._group = group
        self._offset = offset
        self._engines = engines

    def check(self, timeout: float) -> None:
        result = ray.get(self._engines[0].health_generate.remote(timeout=timeout), timeout=timeout)
        if result is not True:
            raise RuntimeError("inference health probe did not report success")

    def retire(self, timeout: float) -> None:
        current = self._group.all_engines
        if any(
            self._offset + index >= len(current) or current[self._offset + index] is not engine
            for index, engine in enumerate(self._engines)
        ):
            return
        pending = []
        for engine in self._engines:
            if engine is not None:
                with suppress(Exception):
                    pending.append(engine.shutdown.remote())
        if pending:
            with suppress(Exception):
                ray.get(pending, timeout=timeout)
        errors = []
        for index, engine in enumerate(self._engines):
            if engine is None:
                continue
            # Kill the captured handle even if shutdown failed. Never replace
            # it with the current occupant after waiting for an RPC.
            try:
                ray.kill(engine, no_restart=True)
            except Exception as exc:
                errors.append(exc)
                continue
            slot = self._offset + index
            if slot < len(current) and current[slot] is engine:
                current[slot] = None
        if errors:
            raise errors[0]
