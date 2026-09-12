"""Default SGLang rollout executor, with all SGLang serving ownership here."""

from __future__ import annotations

from reef.runtime.executor.delegating import DelegatingExecutor
from reef.runtime.executor.uniproc import UniProcExecutor


class SGLangExecutor(DelegatingExecutor):
    """One control rank manages multi-node SGLang engine groups.

    The control object lives in the inference control actor; engine processes are Ray
    actors allocated from supplied reservations. Alternative executors implement the same control
    RPC vocabulary without inheriting SGLang's server or actor classes.
    """

    def _init_executor(self) -> None:
        from reef.runtime.sglang.worker import SGLangWorker

        self._worker = SGLangWorker(**dict(self.config.options))
        self._rpc = UniProcExecutor.from_workers([self._worker], owned=True)

    def check_health(self, timeout: float | None = None) -> None:
        self._rpc.rpc(0, "check_health", timeout=timeout)
