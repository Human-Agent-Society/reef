"""CPU contracts for AgentCL's native no-network provider; no Docker commands."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import shlex
import stat
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
import yaml

from harbor.environments.base import ExecResult
from harbor.environments.docker import COMPOSE_EGRESS_CONTROL_PATH, COMPOSE_NO_NETWORK_PATH
from harbor.environments.docker.docker import DockerEnvironment
from harbor.environments.factory import EnvironmentFactory
from harbor.models.task.config import EnvironmentConfig, NetworkMode, NetworkPolicy, TaskOS
from harbor.models.trial.config import EnvironmentConfig as TrialEnvironmentConfig
from harbor.models.trial.paths import TrialPaths
from harbor.trial.network_policy import TrialNetworkPlan
from harbor.trial.trial import Trial


@pytest.fixture(params=["sdft", "sdpo", "opd"])
def provider(request):
    return importlib.import_module(f"recipes.{request.param}.examples.agentcl.harness.no_network")


def create_environment(
    provider, tmp_path, *, policy=None, phases=(), task_os=TaskOS.LINUX, extra_compose=(), keep_containers=False
):
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir(exist_ok=True)
    (environment_dir / "Dockerfile").write_text("FROM python:3.12-slim\n")
    return provider.NoNetworkDockerEnvironment(
        environment_dir=environment_dir,
        environment_name="fixture",
        session_id="fixture__agent",
        trial_paths=TrialPaths(trial_dir=tmp_path / "trial"),
        task_env_config=EnvironmentConfig(network_mode="no-network", os=task_os, cpus=1, memory_mb=512),
        network_policy=policy,
        phase_network_policies=phases,
        extra_docker_compose=extra_compose,
        keep_containers=keep_containers,
    )


def no_network_policy():
    return NetworkPolicy(network_mode=NetworkMode.NO_NETWORK)


def test_native_none_overlay_and_capabilities(provider, tmp_path):
    environment = create_environment(provider, tmp_path, policy=no_network_policy())
    paths = environment._docker_compose_paths
    assert paths[-1] == COMPOSE_NO_NETWORK_PATH
    assert COMPOSE_EGRESS_CONTROL_PATH not in paths
    assert yaml.safe_load(paths[-1].read_text()) == {"services": {"main": {"network_mode": "none"}}}
    assert environment.capabilities.disable_internet
    assert not environment.capabilities.dynamic_network_policy
    assert not environment.capabilities.network_allowlist
    assert not environment.capabilities.windows
    assert environment._write_egress_control_services_compose_file() is None
    assert environment._effective_cpus == 1
    assert environment._effective_memory_mb == 512


@pytest.mark.parametrize("mode", [None, "public", "allowlist"])
@pytest.mark.parametrize("phase_override", [False, True])
def test_rejects_non_no_network_startup_and_phase_policies(provider, tmp_path, mode, phase_override):
    if mode is None:
        policy = None
    else:
        policy = NetworkPolicy(network_mode=mode, allowed_hosts=["example.com"] if mode == "allowlist" else [])
    if phase_override:
        phases = (NetworkPolicy() if policy is None else policy,)
        startup = no_network_policy()
    else:
        phases = ()
        startup = policy
    with pytest.raises(ValueError, match="requires no-network"):
        create_environment(provider, tmp_path, policy=startup, phases=phases)


def test_rejects_windows(provider, tmp_path):
    with pytest.raises(ValueError, match="Linux"):
        create_environment(provider, tmp_path, policy=no_network_policy(), task_os=TaskOS.WINDOWS)


@pytest.mark.parametrize("network_mode", ["host", "bridge", "none", "service:other", "container:other"])
def test_rejects_task_authored_compose_networking(provider, tmp_path, network_mode):
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    (environment_dir / "docker-compose.yaml").write_text(
        yaml.safe_dump({"services": {"main": {"network_mode": network_mode}}})
    )
    with pytest.raises(ValueError, match="does not accept"):
        create_environment(provider, tmp_path, policy=no_network_policy())


def test_rejects_extra_compose(provider, tmp_path):
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text("services:\n  main:\n    privileged: true\n")
    with pytest.raises(ValueError, match="extra-docker-compose"):
        create_environment(provider, tmp_path, policy=no_network_policy(), extra_compose=(overlay,))


@pytest.mark.parametrize("return_code", [0, 1, 127])
def test_start_checks_actual_interfaces_before_return(provider, tmp_path, monkeypatch, return_code):
    inherited_start = AsyncMock()
    execution = AsyncMock(return_value=ExecResult(return_code=return_code))
    monkeypatch.setattr(DockerEnvironment, "start", inherited_start)
    monkeypatch.setattr(DockerEnvironment, "exec", execution)
    environment = create_environment(provider, tmp_path, policy=no_network_policy())
    if return_code == 0:
        asyncio.run(environment.start(force_build=False))
    else:
        with pytest.raises(RuntimeError, match="loopback"):
            asyncio.run(environment.start(force_build=False))
    inherited_start.assert_awaited_once_with(False)
    command = execution.await_args.args[0]
    assert "python3 -I -S -c" in command
    script = shlex.split(command)[-1]
    assert "socket.if_nameindex()" in script
    assert "interfaces != ['lo']" in script
    compile(script, "network-check", "exec")
    assert execution.await_args.kwargs == {"user": "root", "timeout_sec": 10}


@pytest.mark.parametrize("mode", ["public", "allowlist"])
def test_rejects_policy_change_after_start(provider, tmp_path, mode):
    environment = create_environment(provider, tmp_path, policy=no_network_policy())
    with pytest.raises(ValueError, match="requires no-network"):
        asyncio.run(environment.set_network_policy(NetworkPolicy(network_mode=mode)))
    assert environment.network_policy == no_network_policy()


def test_campaign_lab_override_uses_custom_provider(provider, tmp_path, monkeypatch):
    method = provider.__name__.split(".")[1]
    driver = importlib.import_module(f"recipes.{method}.examples.agentcl.run")
    laboratory = SimpleNamespace(run=AsyncMock())
    monkeypatch.setattr("reef_eval.Lab", Mock(return_value=laboratory))
    arguments = argparse.Namespace(
        run_root=tmp_path / "run",
        data_root=tmp_path / "data",
        service_url="http://fixture",
        scenario="fixture",
        max_turns=8,
    )
    backend = driver.HarborBackend(arguments)
    monkeypatch.setattr(backend, "recover", Mock(return_value={"outcome": "completed"}))
    asyncio.run(
        backend.run(
            {"task_path": "task", "sampling_seed": 42, "category": "raw", "id": "fixture"},
            "episode",
            "baseline",
            {"release_id": "release"},
        )
    )
    assert laboratory.run.await_args.kwargs["environment"] == {
        "import_path": "harness.detached:DetachedNoNetworkEnvironment"
    }


def test_agent_and_separate_verifier_inherit_detached_provider_and_native_none(provider, tmp_path, monkeypatch):
    detached = importlib.import_module(provider.__package__ + ".detached")
    environment_dir = tmp_path / "tests"
    environment_dir.mkdir()
    (environment_dir / "Dockerfile").write_text("FROM python:3.12-slim\n")
    control_paths = []

    async def _start_with_control(environment, force_build):
        await inherited_start(force_build)
        assert force_build is False
        mounts = json.loads(environment._write_mounts_compose_file().read_text())
        control_mount = mounts["services"]["main"]["volumes"][-1]
        assert control_mount["target"] == "/agentcl-control"
        assert control_mount["read_only"] is False
        control = Path(control_mount["source"])
        assert control.stat().st_uid == 0
        assert stat.S_IMODE(control.stat().st_mode) == 0o700
        assert (control / "command_runner.py").read_bytes() == Path(detached.__file__).with_name(
            "command_runner.py"
        ).read_bytes()
        control_paths.append(control)

    inherited_start = AsyncMock()
    execution = AsyncMock(return_value=ExecResult(return_code=0))
    cleanup = AsyncMock(return_value=ExecResult(return_code=0))
    ownership = AsyncMock()
    monkeypatch.setattr(DockerEnvironment, "start", _start_with_control)
    monkeypatch.setattr(DockerEnvironment, "exec", execution)
    monkeypatch.setattr(DockerEnvironment, "prepare_logs_for_host", ownership)
    monkeypatch.setattr(DockerEnvironment, "_run_docker_compose_command", cleanup)
    runtime_config = TrialEnvironmentConfig(
        import_path=f"{detached.__name__}:DetachedNoNetworkEnvironment",
        extra_docker_compose=[tmp_path / "unused-agent-overlay.yaml"],
    )
    plan = TrialNetworkPlan(no_network_policy(), no_network_policy(), no_network_policy(), no_network_policy())
    trial = SimpleNamespace(
        config=SimpleNamespace(environment=runtime_config),
        task=SimpleNamespace(short_name="fixture"),
        paths=TrialPaths(trial_dir=tmp_path / "trial"),
        logger=None,
        _id=uuid4(),
        _environment_build_timeout_sec=10,
        _verifier_env_build_context=lambda _step: environment_dir,
        _separate_verifier_session_id=lambda _key: "fixture__verifier",
        _verifier_env_mounts=lambda _config: [],
        _validate_separate_verifier_env_policies=Mock(),
    )
    agent = EnvironmentFactory.create_environment_from_config(
        config=runtime_config.model_copy(update={"extra_docker_compose": []}),
        environment_dir=environment_dir,
        environment_name="fixture",
        session_id="fixture__agent",
        trial_paths=trial.paths,
        task_env_config=EnvironmentConfig(network_mode="no-network"),
        network_policy=no_network_policy(),
    )
    assert isinstance(agent, detached.DetachedNoNetworkEnvironment)
    assert isinstance(agent, provider.NoNetworkDockerEnvironment)
    assert agent._docker_compose_paths[-1] == COMPOSE_NO_NETWORK_PATH
    assert yaml.safe_load(COMPOSE_NO_NETWORK_PATH.read_text())["services"]["main"]["network_mode"] == "none"

    async def _check_verifier():
        try:
            await agent.start(force_build=False)
            async with Trial._separate_verifier_env(
                trial, EnvironmentConfig(network_mode="no-network"), key="verifier", plan=plan
            ) as environment:
                assert isinstance(environment, detached.DetachedNoNetworkEnvironment)
                assert isinstance(environment, provider.NoNetworkDockerEnvironment)
                assert environment.extra_docker_compose_paths == []
                assert environment._docker_compose_paths[-1] == COMPOSE_NO_NETWORK_PATH
                assert environment.network_policy == no_network_policy()
                assert environment.control_temp_dir.name != agent.control_temp_dir.name
        finally:
            await agent.stop(delete=True)

    asyncio.run(_check_verifier())
    assert inherited_start.await_count == 2
    assert execution.await_count == 2
    for call in execution.await_args_list:
        assert "socket.if_nameindex()" in shlex.split(call.args[0])[-1]
        assert call.kwargs == {"user": "root", "timeout_sec": 10}
    assert ownership.await_count == 2
    assert cleanup.await_count == 2
    cleanup.assert_awaited_with(
        ["down", "--timeout", "1", "--rmi", "local", "--volumes", "--remove-orphans"],
        check=True,
        timeout_sec=None,
        stdin_data=None,
        on_output=None,
    )
    assert len(control_paths) == 2
    assert all(not control.exists() for control in control_paths)


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        (["down", "--remove-orphans"], ["down", "--timeout", "1", "--remove-orphans"]),
        (["stop", "main"], ["stop", "--timeout", "1", "main"]),
    ],
)
def test_compose_teardown_preserves_arguments_and_result(provider, tmp_path, monkeypatch, command, expected):
    result = ExecResult(return_code=125, stdout="removal failed", stderr="fixture error")
    inherited = AsyncMock(return_value=result)
    output = AsyncMock()
    monkeypatch.setattr(DockerEnvironment, "_run_docker_compose_command", inherited)
    environment = create_environment(provider, tmp_path, policy=no_network_policy())
    original = list(command)
    returned = asyncio.run(environment._run_docker_compose_command(command, False, 37, b"fixture stdin", output))
    assert returned is result
    assert command == original
    inherited.assert_awaited_once_with(
        expected, check=False, timeout_sec=37, stdin_data=b"fixture stdin", on_output=output
    )


@pytest.mark.parametrize("operation", ["down", "stop"])
@pytest.mark.parametrize("timeout_arguments", [["--timeout", "9"], ["--timeout=9"], ["-t", "9"], ["-t9"]])
def test_compose_teardown_preserves_explicit_timeout(provider, tmp_path, monkeypatch, operation, timeout_arguments):
    inherited = AsyncMock(return_value=ExecResult(return_code=0))
    monkeypatch.setattr(DockerEnvironment, "_run_docker_compose_command", inherited)
    environment = create_environment(provider, tmp_path, policy=no_network_policy())
    command = [operation, *timeout_arguments]
    asyncio.run(environment._run_docker_compose_command(command))
    assert inherited.await_args.args[0] is command
    inherited.assert_awaited_once_with(command, check=True, timeout_sec=None, stdin_data=None, on_output=None)


@pytest.mark.parametrize(
    "command", [["build"], ["up", "--detach", "--wait"], ["exec", "main", "true"], ["cp", "source", "main:/target"]]
)
def test_compose_non_teardown_commands_are_unchanged(provider, tmp_path, monkeypatch, command):
    inherited = AsyncMock(return_value=ExecResult(return_code=0))
    monkeypatch.setattr(DockerEnvironment, "_run_docker_compose_command", inherited)
    environment = create_environment(provider, tmp_path, policy=no_network_policy())
    asyncio.run(environment._run_docker_compose_command(command))
    assert inherited.await_args.args[0] is command
    inherited.assert_awaited_once_with(command, check=True, timeout_sec=None, stdin_data=None, on_output=None)


@pytest.mark.parametrize("failure", [RuntimeError("removal failed"), asyncio.CancelledError("fixture cancellation")])
def test_compose_teardown_propagates_failure_and_cancellation(provider, tmp_path, monkeypatch, failure):
    inherited = AsyncMock(side_effect=failure)
    monkeypatch.setattr(DockerEnvironment, "_run_docker_compose_command", inherited)
    environment = create_environment(provider, tmp_path, policy=no_network_policy())
    with pytest.raises(type(failure)) as caught:
        asyncio.run(environment._run_docker_compose_command(["down"], timeout_sec=37))
    assert caught.value is failure
    inherited.assert_awaited_once_with(
        ["down", "--timeout", "1"], check=True, timeout_sec=37, stdin_data=None, on_output=None
    )


@pytest.mark.parametrize(
    ("delete", "keep_containers", "expected"),
    [
        (True, False, ["down", "--timeout", "1", "--rmi", "local", "--volumes", "--remove-orphans"]),
        (False, False, ["down", "--timeout", "1"]),
        (True, True, ["stop", "--timeout", "1"]),
        (False, True, ["stop", "--timeout", "1"]),
    ],
)
def test_agent_stop_preserves_cleanup_semantics(provider, tmp_path, monkeypatch, delete, keep_containers, expected):
    inherited = AsyncMock(return_value=ExecResult(return_code=0))
    ownership = AsyncMock()
    monkeypatch.setattr(DockerEnvironment, "_run_docker_compose_command", inherited)
    monkeypatch.setattr(DockerEnvironment, "prepare_logs_for_host", ownership)
    environment = create_environment(provider, tmp_path, policy=no_network_policy(), keep_containers=keep_containers)
    mounts = environment._write_mounts_compose_file()
    resources = environment._write_resources_compose_file()
    startup_env = environment._write_env_compose_file()
    asyncio.run(environment.stop(delete=delete))
    ownership.assert_awaited_once()
    inherited.assert_awaited_once_with(expected, check=True, timeout_sec=None, stdin_data=None, on_output=None)
    assert not mounts.exists()
    assert not resources.exists()
    assert not startup_env.exists()


def test_method_provider_sources_are_identical():
    root = Path(__file__).resolve().parents[1]
    assert (root / "recipes/sdft/examples/agentcl/harness/no_network.py").read_bytes() == (
        root / "recipes/sdpo/examples/agentcl/harness/no_network.py"
    ).read_bytes()
