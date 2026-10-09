"""Native no-network transport using Harbor 0.20's Docker boundary hooks.

Harbor's protected Compose and egress-selection hooks are required to retain its
image, mount, resource, and cleanup behavior without the egress-control service.
"""

from __future__ import annotations

import shlex
from collections.abc import Sequence
from pathlib import Path
from typing import override

from harbor.environments.base import ExecResult, OutputCallback
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.environments.docker import COMPOSE_NO_NETWORK_PATH
from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.task.config import NetworkMode, NetworkPolicy, TaskOS


class NoNetworkDockerEnvironment(DockerEnvironment):
    """Run a single Linux container with only its loopback interface."""

    @staticmethod
    @override
    def _requires_egress_control(
        *, startup_network_policy: NetworkPolicy, phase_network_policies: Sequence[NetworkPolicy]
    ) -> bool:
        for policy in (startup_network_policy, *phase_network_policies):
            if policy.network_mode != NetworkMode.NO_NETWORK or policy.allowed_hosts:
                raise ValueError("AgentCL transport requires no-network without allowed hosts")
        return False

    @property
    @override
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(disable_internet=True, mounted=True)

    @override
    def validate_network_policy_support(self, network_policy: NetworkPolicy | None = None) -> None:
        policy = self.network_policy if network_policy is None else network_policy
        if policy.network_mode != NetworkMode.NO_NETWORK or policy.allowed_hosts:
            raise ValueError("AgentCL transport requires no-network without allowed hosts")
        super().validate_network_policy_support(policy)

    @override
    def _validate_definition(self) -> None:
        if self.task_env_config.os != TaskOS.LINUX:
            raise ValueError("AgentCL transport requires Linux containers")
        if self._environment_docker_compose_path.exists() or self.extra_docker_compose_paths:
            raise ValueError("AgentCL transport does not accept task or extra Docker Compose files")
        super()._validate_definition()

    @property
    @override
    def _docker_compose_paths(self) -> list[Path]:
        return [*super()._docker_compose_paths, COMPOSE_NO_NETWORK_PATH]

    @override
    async def _run_docker_compose_command(
        self,
        command: list[str],
        check: bool = True,
        timeout_sec: int | None = None,
        stdin_data: bytes | None = None,
        on_output: OutputCallback | None = None,
    ) -> ExecResult:
        if command and command[0] in ("down", "stop"):
            explicit_timeout = any(
                argument in ("--timeout", "-t") or argument.startswith(("--timeout=", "-t"))
                for argument in command[1:]
            )
            if not explicit_timeout:
                # The sandbox keepalive does not need a long shutdown grace period.
                command = [command[0], "--timeout", "1", *command[1:]]
        return await super()._run_docker_compose_command(
            command,
            check=check,
            timeout_sec=timeout_sec,
            stdin_data=stdin_data,
            on_output=on_output,
        )

    @override
    async def start(self, force_build: bool) -> None:
        await super().start(force_build)
        script = (
            "import socket,sys; interfaces = [name for _, name in socket.if_nameindex()]; "
            "print(interfaces); sys.exit(interfaces != ['lo'])"
        )
        result = await self.exec(
            "env -i PATH=/usr/local/bin:/usr/bin:/bin python3 -I -S -c " + shlex.quote(script),
            user="root",
            timeout_sec=10,
        )
        if result.return_code != 0:
            raise RuntimeError("AgentCL container must expose only the loopback network interface")
