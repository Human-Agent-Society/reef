"""Backend-neutral worker launch, control RPC, and result handling.

Executors coordinate workers; tensor collectives and checkpoint semantics stay
with the training or serving backend. Importing this package does not load Ray.
"""

from reef.execution.executor.base import Executor, ExecutorFuture, ExecutorLaunch, WorkerSpec, resolve
from reef.execution.executor.failure import ExecutorFailedError, ExecutorFailure, ExecutorFailureListener
from reef.execution.executor.requirements import ExecutionRequirements

__all__ = [
    "ExecutionRequirements",
    "Executor",
    "ExecutorFailedError",
    "ExecutorFailure",
    "ExecutorFailureListener",
    "ExecutorFuture",
    "ExecutorLaunch",
    "WorkerSpec",
    "resolve",
]
