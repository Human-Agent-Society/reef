"""Harbor's E2B task environment, with each sandbox deleted at its timeout.

Harbor's ``E2BEnvironment`` creates its sandbox with no lifecycle setting, so the service decides what happens when
the sandbox reaches its timeout. E2B deletes it. A self-hosted service that speaks E2B's API can pause it instead,
as AgentENV does, and then the sandbox of a trial whose process was killed stays on the service.
``KillOnTimeoutE2BEnvironment`` asks for the deletion when it creates the sandbox. Everything else is Harbor's.

``reef.execution.sandbox.task_environment.environment_config`` names this class as the ``import_path`` of an
``e2b`` trial, and Harbor imports it from there. Two callers use that config: the task rollout runner
(``reef.harness.harbor.rollouts``) for its in-process trials, and the generator's checks
(``reef.record2sample.generator.harbor``) for their ``harbor run`` children. A trial that names only the type
``e2b`` gets Harbor's own class. The module needs the e2b package and Harbor 0.25 (``reef-infra[e2b]``).
"""

from __future__ import annotations

from typing import override

from e2b import AsyncSandbox
from harbor.environments.e2b import E2BEnvironment
from harbor.models.task.config import NetworkMode
from tenacity import retry, stop_after_attempt, wait_exponential


class KillOnTimeoutE2BEnvironment(E2BEnvironment):
    """Harbor's ``E2BEnvironment`` whose sandbox is created with ``lifecycle={"on_timeout": "kill"}``.

    Values: those of ``E2BEnvironment``. Harbor makes one from a trial's environment config, so the template, the
    sandbox's lifetime (``sandbox_timeout_secs``), its metadata, its startup environment and its network options are
    the parent's, and ``type()`` is ``e2b``.
    Purpose: Harbor's ``EnvironmentFactory`` makes one for an ``e2b`` trial whose environment config names this class
    as its ``import_path``. ``start`` creates the sandbox with ``_create_sandbox``, and ``stop`` deletes it, as before.
    """

    # Harbor 0.25.0 names this method and the attributes that it reads; an override cannot rename them. The body is
    # Harbor's create call with one more argument, and the decorator is Harbor's retry policy, because Harbor's
    # decorator wraps its own method and not this one. When Harbor adds an argument to the call, add it here too: the
    # parity test in tests/sandbox/test_task_environment.py compares the two calls.
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=10), reraise=True)
    @override
    async def _create_sandbox(self) -> None:
        self._sandbox = await AsyncSandbox.create(
            template=self._template_name,
            metadata={"environment_name": self.environment_name, "session_id": self.session_id},
            envs=self._startup_env(),
            timeout=self._sandbox_timeout,
            allow_internet_access=self.network_policy.network_mode != NetworkMode.NO_NETWORK,
            network=self._sandbox_create_network_options(),
            lifecycle={"on_timeout": "kill"},
        )


__all__ = ["KillOnTimeoutE2BEnvironment"]
