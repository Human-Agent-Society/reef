"""In-memory actor doubles for tests of distributed coordinators, not a backend."""

from reef.runtime.executor import Executor, ExecutorConfig, ExecutorFuture, resolve
from reef.runtime.executor.uniproc import UniProcExecutor


class _GroupFuture(ExecutorFuture):
    def __init__(self, pending):
        self.pending = pending

    def result(self, timeout=None):
        return resolve(self.pending, timeout=timeout)


class AttachedTestGroup(Executor):
    def _init_executor(self):
        self.ranks = []
        self._closed = False

    @classmethod
    def from_workers(cls, workers, *, owned=False):
        group = cls(ExecutorConfig(backend=cls))
        group.ranks = [UniProcExecutor.from_workers([worker], owned=owned) for worker in workers]
        return group

    def collective_rpc(self, method, *, args=(), kwargs=None, timeout=None, non_block=False):
        pending = [rank.rpc(0, method, args=args, kwargs=kwargs, non_block=True) for rank in self.ranks]
        future = _GroupFuture(pending)
        return future if non_block else future.result(timeout)

    def rpc(self, rank, method, **kwargs):
        return self.ranks[rank].rpc(0, method, **kwargs)

    def check_health(self, timeout=None):
        for rank in self.ranks:
            rank.check_health(timeout)

    def shutdown(self):
        self._closed = True
        for rank in self.ranks:
            rank.shutdown()
