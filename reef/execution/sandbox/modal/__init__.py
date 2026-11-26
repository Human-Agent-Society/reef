"""Modal program execution. Harbor loads the trial environment from ``modal.trial``."""

from reef.execution.sandbox.modal import executor
from reef.execution.sandbox.modal.executor import ModalExecutor, ModalSession

DEFAULT_APP = executor.DEFAULT_APP
KILL_AFTER_S = executor.KILL_AFTER_S
NODE_IMAGE = executor.NODE_IMAGE
SANDBOX_MAX_TIMEOUT_S = executor.SANDBOX_MAX_TIMEOUT_S
TIMEOUT_STATUSES = executor.TIMEOUT_STATUSES

__all__ = ["ModalExecutor", "ModalSession"]
