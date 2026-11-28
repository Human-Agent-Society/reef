"""E2B program execution and the Harbor environment for E2B trials."""

from reef.execution.sandbox.agentenv import build, compose
from reef.execution.sandbox.e2b import executor, trial
from reef.execution.sandbox.e2b.executor import (
    TRAFFIC_ACCESS_TOKEN_HEADER,
    E2BExecutor,
    E2BSession,
    SandboxKeeper,
    ensure_template,
    template_alias,
)
from reef.execution.sandbox.e2b.trial import (
    AENV_TEMPLATE_SUFFIX,
    COPY_MODES_TEMPLATE_SUFFIX,
    KillOnTimeoutE2BEnvironment,
    TemplateNotPreparedError,
    build_task_main_image,
    build_task_template,
    dockerfile_with_copy_modes,
)

SANDBOX_MAX_TIMEOUT_S = executor.SANDBOX_MAX_TIMEOUT_S
SANDBOX_RENEW_MARGIN_S = executor.SANDBOX_RENEW_MARGIN_S
SYSTEMD_TMP_CLEANUP_WAIT = trial.SYSTEMD_TMP_CLEANUP_WAIT
SYSTEMD_TMP_CLEANUP_WAIT_SECONDS = trial.SYSTEMD_TMP_CLEANUP_WAIT_SECONDS
VERIFIER_CAPACITY_FIRST_PAUSE_SECONDS = trial.VERIFIER_CAPACITY_FIRST_PAUSE_SECONDS
VERIFIER_CAPACITY_MAX_PAUSE_SECONDS = trial.VERIFIER_CAPACITY_MAX_PAUSE_SECONDS
VERIFIER_CAPACITY_WAIT_SHARE = trial.VERIFIER_CAPACITY_WAIT_SHARE
VERIFIER_SESSION_MARKER = trial.VERIFIER_SESSION_MARKER

# Trials look these functions up here so callers can supply their own build and startup operations.
build_image = build.build_image
ensure_aenv_template = build.ensure_aenv_template
delete_sandbox = compose.delete_sandbox
prepare_guest = compose.prepare_guest

__all__ = [
    "AENV_TEMPLATE_SUFFIX",
    "COPY_MODES_TEMPLATE_SUFFIX",
    "TRAFFIC_ACCESS_TOKEN_HEADER",
    "E2BExecutor",
    "E2BSession",
    "KillOnTimeoutE2BEnvironment",
    "SandboxKeeper",
    "TemplateNotPreparedError",
    "build_task_main_image",
    "build_task_template",
    "dockerfile_with_copy_modes",
    "ensure_template",
    "template_alias",
]
