"""Compatibility path for E2B executors and saved Harbor trials."""

import sys

from reef.execution.sandbox.providers.e2b import trial

SANDBOX_MAX_TIMEOUT_S = trial.SANDBOX_MAX_TIMEOUT_S
SANDBOX_RENEW_MARGIN_S = trial.SANDBOX_RENEW_MARGIN_S
TRAFFIC_ACCESS_TOKEN_HEADER = trial.TRAFFIC_ACCESS_TOKEN_HEADER
E2BExecutor = trial.E2BExecutor
E2BSession = trial.E2BSession
SandboxKeeper = trial.SandboxKeeper
ensure_template = trial.ensure_template
template_alias = trial.template_alias
AENV_TEMPLATE_SUFFIX = trial.AENV_TEMPLATE_SUFFIX
COPY_MODES_TEMPLATE_SUFFIX = trial.COPY_MODES_TEMPLATE_SUFFIX
SYSTEMD_TMP_CLEANUP_WAIT = trial.SYSTEMD_TMP_CLEANUP_WAIT
SYSTEMD_TMP_CLEANUP_WAIT_SECONDS = trial.SYSTEMD_TMP_CLEANUP_WAIT_SECONDS
VERIFIER_CAPACITY_FIRST_PAUSE_SECONDS = trial.VERIFIER_CAPACITY_FIRST_PAUSE_SECONDS
VERIFIER_CAPACITY_MAX_PAUSE_SECONDS = trial.VERIFIER_CAPACITY_MAX_PAUSE_SECONDS
VERIFIER_CAPACITY_WAIT_SHARE = trial.VERIFIER_CAPACITY_WAIT_SHARE
VERIFIER_SESSION_MARKER = trial.VERIFIER_SESSION_MARKER
KillOnTimeoutE2BEnvironment = trial.KillOnTimeoutE2BEnvironment
TemplateNotPreparedError = trial.TemplateNotPreparedError
build_task_main_image = trial.build_task_main_image
build_task_template = trial.build_task_template
dockerfile_with_copy_modes = trial.dockerfile_with_copy_modes

# Both import paths must see changes to module-level settings and hooks.
sys.modules[__name__] = trial
