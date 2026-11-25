"""Compatibility path for the Modal executor and its settings."""

import sys

from reef.execution.sandbox.providers.modal import executor

DEFAULT_APP = executor.DEFAULT_APP
KILL_AFTER_S = executor.KILL_AFTER_S
NODE_IMAGE = executor.NODE_IMAGE
SANDBOX_MAX_TIMEOUT_S = executor.SANDBOX_MAX_TIMEOUT_S
TIMEOUT_STATUSES = executor.TIMEOUT_STATUSES
ModalExecutor = executor.ModalExecutor
ModalSession = executor.ModalSession

# Both import paths must see changes to module-level settings and hooks.
sys.modules[__name__] = executor
