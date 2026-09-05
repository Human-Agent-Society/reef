"""One ordinary worker in the caller's process, without GPU allocation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from reef.runtime.executor.local import LocalExecutor


class UniProcExecutor(LocalExecutor):
    def _init_executor(self) -> None:
        if len(self.config.workers) > 1:
            raise ValueError("UniProcExecutor accepts at most one worker")
        super()._init_executor()

    @classmethod
    def from_workers(cls, workers: Sequence[Any], *, owned: bool = False) -> LocalExecutor:
        if len(workers) > 1:
            raise ValueError("UniProcExecutor accepts at most one worker")
        return super().from_workers(workers, owned=owned)
