"""CPU-only private detached command contracts; no container clients execute."""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import pwd
import shlex
import signal
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from harbor.environments.base import ExecResult
from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.task.config import EnvironmentConfig, NetworkPolicy
from harbor.models.trial.paths import TrialPaths


@pytest.fixture(params=["sdft", "sdpo", "opd"])
def provider(request):
    return importlib.import_module(f"recipes.{request.param}.examples.agentcl.harness.detached")


@pytest.fixture
def environment(provider, tmp_path):
    if os.geteuid() != 0:
        pytest.skip("Root-local private transport")
    definition = tmp_path / "environment"
    definition.mkdir()
    (definition / "Dockerfile").write_text("FROM python:3.12-slim\n")
    instance = provider.DetachedNoNetworkEnvironment(
        environment_dir=definition,
        environment_name="fixture",
        session_id="Fixture__agent",
        trial_paths=TrialPaths(trial_dir=tmp_path / "trial"),
        task_env_config=EnvironmentConfig(network_mode="no-network", workdir="/tmp"),
        network_policy=NetworkPolicy(network_mode="no-network"),
    )
    instance._write_mounts_compose_file()
    yield instance
    if instance.control_temp_dir is not None:
        instance.control_temp_dir.cleanup()
    instance._cleanup_mounts_compose_file()


@pytest.fixture
def runner(provider):
    return importlib.import_module(provider.__package__ + ".command_runner")


def request_job(runner, control, command, *, user=None, env=None, cwd="/tmp", seconds=2):
    nonce = uuid4().hex
    job = control / nonce
    job.mkdir(mode=0o700)
    runner.write_atomic(
        job / "request.json",
        {
            "nonce": nonce,
            "command": command,
            "cwd": cwd,
            "env": env or {},
            "user": user,
            "deadline": time.time() + seconds,
        },
    )
    return job


def invoke_runner(runner, job):
    return subprocess.run(
        [sys.executable, "-I", "-S", runner.__file__, str(job / "request.json")], capture_output=True, timeout=4
    )


def test_private_mount_and_preserved_volumes(environment, provider, monkeypatch, tmp_path):
    overlay = tmp_path / "mounts:fixture.json"
    original = {"services": {"main": {"volumes": ["/trusted:/hidden:ro"]}}}
    overlay.write_text(json.dumps(original))

    def inherited(self):
        return overlay

    monkeypatch.setattr(DockerEnvironment, "_write_mounts_compose_file", inherited)
    path = environment._write_mounts_compose_file()
    compose = json.loads(path.read_text())
    assert compose["services"]["main"]["volumes"][0] == "/trusted:/hidden:ro"
    mount = compose["services"]["main"]["volumes"][1]
    assert mount == {
        "type": "bind",
        "source": environment.control_temp_dir.name,
        "target": "/agentcl-control",
        "read_only": False,
    }
    control = Path(mount["source"])
    assert control.parent == Path("/tmp")
    assert control.stat().st_uid == 0
    assert stat.S_IMODE(control.stat().st_mode) == 0o700
    assert stat.S_IMODE((control / "command_runner.py").stat().st_mode) == 0o644
    assert (control / "command_runner.py").read_bytes() == Path(provider.__file__).with_name(
        "command_runner.py"
    ).read_bytes()


@pytest.mark.parametrize("return_code", [0, 7, 127])
def test_runner_exact_shell_environment_cwd_and_codes(environment, runner, return_code):
    control = Path(environment.control_temp_dir.name)
    command = 'printf "%s|%s|%s" "$VALUE" "$INHERITED" "$PWD"; printf error >&2; exit ' + str(return_code)
    job = request_job(runner, control, command, env={"VALUE": 'literal $HOME : "value"', "INHERITED": "merged"})
    process = invoke_runner(runner, job)
    result = json.loads((job / "result.json").read_text())
    started = json.loads((job / "started.json").read_text())
    assert process.returncode == 0 and process.stdout == process.stderr == b""
    assert result["stdout"] == 'literal $HOME : "value"|merged|/tmp'
    assert result["stderr"] == "error" and result["return_code"] == return_code
    assert result["nonce"] == job.name and result["pid"] == started["pid"]
    assert not result["timed_out"]
    for name in ("request.json", "started.json", "result.json", "claimed"):
        assert stat.S_IMODE((job / name).stat().st_mode) == 0o600
    assert not list(job.glob("*.tmp"))


def test_runner_inherits_then_merges_environment(environment, runner, monkeypatch):
    monkeypatch.setenv("DETACHED_TEST_INHERIT", "normal")
    job = request_job(runner, Path(environment.control_temp_dir.name), 'printf "%s" "$DETACHED_TEST_INHERIT"')
    assert invoke_runner(runner, job).returncode == 0
    assert json.loads((job / "result.json").read_text())["stdout"] == "normal"


def test_numeric_user_groups_and_control_cannot_be_forged(environment, runner):
    account = pwd.getpwnam("nobody")
    control = Path(environment.control_temp_dir.name)
    source = "import os; print(os.getuid(), os.getgid(), sorted(os.getgroups())); "
    source += f"print(os.access({str(control)!r}, os.R_OK)); "
    source += f"open({str(control / 'forged')!r}, 'w')"
    command = "/usr/bin/python3 -I -S -c " + shlex.quote(source)
    job = request_job(runner, control, command, user=account.pw_uid)
    invoke_runner(runner, job)
    result = json.loads((job / "result.json").read_text())
    expected = f"{account.pw_uid} {account.pw_gid} {sorted(os.getgrouplist(account.pw_name, account.pw_gid))}\nFalse\n"
    assert result["stdout"] == expected
    assert result["return_code"] == 1 and "PermissionError" in result["stderr"]
    assert not (control / "forged").exists()


def test_student_name_resolves_uid_and_groups(environment, runner, monkeypatch):
    account = pwd.getpwnam("nobody")
    monkeypatch.setattr(runner.pwd, "getpwnam", lambda name: account)
    job = request_job(runner, Path(environment.control_temp_dir.name), "id -u; id -g; id -G", user="student")
    runner.run(job / "request.json")
    result = json.loads((job / "result.json").read_text())
    assert result["stdout"].splitlines()[:2] == [str(account.pw_uid), str(account.pw_gid)]
    assert set(map(int, result["stdout"].splitlines()[2].split())) == set(
        os.getgrouplist(account.pw_name, account.pw_gid)
    )


@pytest.mark.parametrize("user", ["root:root", "nobody", -1, True])
def test_invalid_users_do_not_execute(environment, runner, user):
    job = request_job(runner, Path(environment.control_temp_dir.name), "printf should-not-run", user=user)
    result = invoke_runner(runner, job)
    assert result.returncode != 0 and result.stdout == result.stderr == b""
    assert not (job / "result.json").exists()


def test_nonce_corruption_and_duplicate_launch(environment, runner):
    control = Path(environment.control_temp_dir.name)
    job = request_job(runner, control, "printf once")
    assert invoke_runner(runner, job).returncode == 0
    before = (job / "result.json").read_bytes()
    assert invoke_runner(runner, job).returncode != 0
    assert (job / "result.json").read_bytes() == before
    bad = request_job(runner, control, "printf never")
    request = json.loads((bad / "request.json").read_text())
    request["nonce"] = uuid4().hex
    (bad / "request.json").write_text(json.dumps(request))
    assert invoke_runner(runner, bad).returncode != 0
    assert not (bad / "started.json").exists()


@pytest.mark.parametrize("cancelled", [False, True])
def test_cancelled_or_expired_request_never_spawns_child(environment, runner, monkeypatch, cancelled):
    marker = Path(environment.control_temp_dir.name) / "side-effect"
    job = request_job(
        runner,
        Path(environment.control_temp_dir.name),
        "touch " + shlex.quote(str(marker)),
        seconds=2 if cancelled else -1,
    )
    if cancelled:
        runner.write_atomic(job / "cancel.json", {"nonce": job.name})
    child_launch = Mock(side_effect=AssertionError("Cancelled command must not spawn a child"))
    monkeypatch.setattr(runner.subprocess, "Popen", child_launch)
    runner.run(job / "request.json")
    child_launch.assert_not_called()
    assert (job / "claimed").is_file()
    assert not (job / "started.json").exists() and not marker.exists()


def test_cancelled_request_subprocess_does_not_execute_or_replay(environment, runner):
    control = Path(environment.control_temp_dir.name)
    marker = control / "side-effect"
    job = request_job(runner, control, "touch " + shlex.quote(str(marker)))
    runner.write_atomic(job / "cancel.json", {"nonce": job.name})
    assert invoke_runner(runner, job).returncode == 0
    assert invoke_runner(runner, job).returncode != 0
    assert not marker.exists() and not (job / "started.json").exists()


def test_timeout_kills_child_process_group(environment, runner):
    job = request_job(runner, Path(environment.control_temp_dir.name), "sleep 10 & wait", seconds=0.2)
    started_at = time.monotonic()
    invoke_runner(runner, job)
    result = json.loads((job / "result.json").read_text())
    assert result["timed_out"] and result["return_code"] == -signal.SIGKILL
    assert time.monotonic() - started_at < 1
    with pytest.raises(ProcessLookupError):
        os.kill(result["pid"], 0)


def test_cancel_marker_drains_child(environment, runner):
    job = request_job(runner, Path(environment.control_temp_dir.name), "sleep 10", seconds=2)
    process = subprocess.Popen([sys.executable, "-I", "-S", runner.__file__, str(job / "request.json")])
    deadline = time.monotonic() + 1
    while not (job / "started.json").exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    runner.write_atomic(job / "cancel.json", {"nonce": job.name})
    assert process.wait(timeout=1) == 0
    assert json.loads((job / "result.json").read_text())["timed_out"]


@pytest.mark.parametrize("acknowledgement", ["zero", "nonzero", "stall"])
def test_completion_not_acknowledgement_no_replay(environment, runner, monkeypatch, acknowledgement):
    calls, children = [], []
    control = Path(environment.control_temp_dir.name)

    async def native(arguments):
        calls.append(arguments)
        if arguments[0] == "ps":
            return await asyncio.create_subprocess_exec(
                sys.executable, "-c", "print('a' * 64)", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
        request = control / Path(arguments[-1]).parent.name / "request.json"
        children.append(
            subprocess.Popen(
                [sys.executable, "-I", "-S", runner.__file__, str(request)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        )
        code = (
            "import time; time.sleep(2)"
            if acknowledgement == "stall"
            else "raise SystemExit(" + ("255" if acknowledgement == "nonzero" else "0") + ")"
        )
        return await asyncio.create_subprocess_exec(
            sys.executable, "-c", code, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

    monkeypatch.setattr(environment, "native_process", native)
    callback = AsyncMock()
    monkeypatch.setattr(environment, "_output_callback", lambda: callback)
    started_at = time.monotonic()
    result = asyncio.run(environment.exec("printf 42; printf err >&2; exit 7", user="root", timeout_sec=1))
    assert result == ExecResult(stdout="42", stderr="err", return_code=7)
    assert time.monotonic() - started_at < 1
    assert len([call for call in calls if call[0] == "exec"]) == 1
    assert calls[1][:5] == ["exec", "--detach", "--user", "root", "a" * 64]
    assert "printf 42" not in str(calls) and "err >&2" not in str(calls)
    callback.assert_any_await("42", "stdout")
    callback.assert_any_await("err", "stderr")
    for child in children:
        assert child.wait(timeout=1) == 0


@pytest.mark.parametrize("corruption", [None, "nonce", "pid"])
def test_missing_or_corrupt_status_is_unknown_and_never_replayed(environment, runner, monkeypatch, corruption):
    calls = []
    cleanup = AsyncMock(return_value=ExecResult(return_code=0))
    environment.container_id = "a" * 64
    monkeypatch.setattr(environment, "_run_docker_compose_command", cleanup)

    async def native(arguments):
        calls.append(arguments)
        job = Path(environment.control_temp_dir.name) / Path(arguments[-1]).parent.name
        if corruption:
            runner.write_atomic(job / "started.json", {"nonce": job.name, "pid": 42})
            runner.write_atomic(
                job / "result.json",
                {
                    "nonce": uuid4().hex if corruption == "nonce" else job.name,
                    "pid": 43 if corruption == "pid" else 42,
                    "return_code": 0,
                    "timed_out": False,
                    "stdout": "forged",
                    "stderr": "",
                    "completed_at": time.time(),
                },
            )
        return await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "raise SystemExit(255)",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    monkeypatch.setattr(environment, "native_process", native)
    with pytest.raises(RuntimeError, match="outcome is unknown"):
        asyncio.run(environment.exec("sensitive command", timeout_sec=0.15))
    assert len(calls) == 1
    cleanup.assert_awaited_once_with(["down", "--volumes", "--remove-orphans"], check=False, timeout_sec=2)


def test_unspecified_command_limit_preserves_outer_task_deadline(environment, runner, monkeypatch):
    children = []
    environment.container_id = "a" * 64

    async def native(arguments):
        request = Path(environment.control_temp_dir.name) / Path(arguments[-1]).parent.name / "request.json"
        assert json.loads(request.read_text())["deadline"] is None
        children.append(subprocess.Popen([sys.executable, "-I", "-S", runner.__file__, str(request)]))
        return await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "raise SystemExit(0)",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    monkeypatch.setattr(environment, "native_process", native)
    result = asyncio.run(environment.exec("printf verified", user="root"))
    assert result == ExecResult(stdout="verified", stderr=None, return_code=0)
    assert len(children) == 1
    assert children[0].wait(timeout=2) == 0


def test_base_resolution_non_main_delegation_and_stop_order(environment, monkeypatch):
    compose = AsyncMock(return_value=ExecResult(return_code=0))
    monkeypatch.setattr(DockerEnvironment, "_compose_exec", compose)
    asyncio.run(
        environment.service_exec("true", service="other", cwd="/other", env={"X": "y"}, user=123, timeout_sec=9)
    )
    compose.assert_awaited_once_with("true", service="other", cwd="/other", env={"X": "y"}, user=123, timeout_sec=9)
    control = Path(environment.control_temp_dir.name)

    async def stop(self, delete):
        assert control.exists()
        assert delete

    monkeypatch.setattr(DockerEnvironment, "stop", stop)
    asyncio.run(environment.stop(True))
    assert not control.exists() and environment.control_temp_dir is None


@pytest.mark.parametrize("cancellation_count", [1, 2])
def test_cancelled_stop_drains_command_and_teardown_before_releasing_mount(
    environment, runner, monkeypatch, cancellation_count
):
    control = Path(environment.control_temp_dir.name)
    environment.container_id = "a" * 64
    children = []
    jobs = []
    native_clients = []
    thread_errors = []
    teardown_reached = threading.Event()
    teardown_release = threading.Event()
    cancellation_sent = threading.Event()

    async def native(arguments):
        job = control / Path(arguments[-1]).parent.name
        jobs.append(job)
        children.append(subprocess.Popen([sys.executable, "-I", "-S", runner.__file__, str(job / "request.json")]))
        client = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "import time; time.sleep(10)",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        native_clients.append(client)
        return client

    async def inherited_stop(self, delete):
        assert delete and control.exists()
        assert not self.active_commands
        assert (jobs[0] / "cancel.json").is_file()
        await asyncio.to_thread(children[0].wait, timeout=2)
        assert json.loads((jobs[0] / "result.json").read_text())["timed_out"]
        assert all(client.returncode is not None for client in native_clients)
        teardown_reached.set()
        assert await asyncio.to_thread(teardown_release.wait, 2)
        assert control.exists()

    monkeypatch.setattr(environment, "native_process", native)
    monkeypatch.setattr(DockerEnvironment, "stop", inherited_stop)

    async def exercise():
        execution = asyncio.create_task(environment.exec("sleep 10 & wait", user="root"))
        deadline = time.monotonic() + 2
        while not jobs or not (jobs[0] / "started.json").exists():
            assert time.monotonic() < deadline
            await asyncio.sleep(0.01)
        loop = asyncio.get_running_loop()
        stopping = asyncio.create_task(environment.stop(True))

        def cancel_during_teardown():
            try:
                assert teardown_reached.wait(2)
                for _ in range(cancellation_count):
                    loop.call_soon_threadsafe(stopping.cancel, "fixture cancellation")
                    time.sleep(0.02)
                cancellation_sent.set()
            except AssertionError as error:
                thread_errors.append(error)
            finally:
                teardown_release.set()

        thread = threading.Thread(target=cancel_during_teardown)
        thread.start()
        try:
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(stopping, 4)
            with pytest.raises(asyncio.CancelledError):
                await execution
            assert cancellation_sent.is_set() and not thread_errors
            assert not control.exists() and environment.control_temp_dir is None
            assert environment.container_id is None
        finally:
            teardown_release.set()
            execution.cancel()
            await asyncio.gather(execution, return_exceptions=True)
            await asyncio.to_thread(thread.join, 2)

    try:
        asyncio.run(exercise())
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=2)


def test_failed_teardown_retains_control_mount_for_retry(environment, monkeypatch):
    control = Path(environment.control_temp_dir.name)
    failure = RuntimeError("teardown failed")
    inherited = AsyncMock(side_effect=failure)
    monkeypatch.setattr(DockerEnvironment, "stop", inherited)
    with pytest.raises(RuntimeError) as captured:
        asyncio.run(environment.stop(True))
    assert captured.value is failure
    assert control.exists() and environment.control_temp_dir is not None
    inherited.side_effect = None
    asyncio.run(environment.stop(True))
    assert not control.exists() and environment.control_temp_dir is None


def test_example_sources_identical():
    root = Path(__file__).resolve().parents[1]
    for name in ("command_runner.py", "detached.py"):
        assert (root / "recipes/sdft/examples/agentcl/harness" / name).read_bytes() == (
            root / "recipes/sdpo/examples/agentcl/harness" / name
        ).read_bytes()
