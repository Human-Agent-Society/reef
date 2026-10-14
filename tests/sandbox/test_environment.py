"""Reef execution uses Harbor's actual environment and result contracts."""

import asyncio
import os
import shlex
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from harbor.environments.base import BaseEnvironment, ExecResult

from reef.sandbox.e2b import E2BExecutor, E2BSession
from reef.sandbox.environment import ExecutorEnvironment, run_sync
from reef.sandbox.process import EpisodeLaunchError, EpisodeTimeout, LocalExecutor, ProcessOutcome


def test_harbor_lifecycle_commands_and_file_transfers(tmp_path: Path) -> None:
    root = tmp_path / "environment"
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    source = tmp_path / "input.txt"
    source.write_text("input")
    environment: BaseEnvironment = ExecutorEnvironment(LocalExecutor(), root=root, workspace=workspace)

    async def _run() -> None:
        await environment.start(force_build=False)
        await environment.upload_file(source, "input.txt")
        result = await environment.exec("cat input.txt > output.txt; printf stdout; printf stderr >&2; exit 7")
        assert isinstance(result, ExecResult)
        assert (result.return_code, result.stdout, result.stderr) == (7, "stdout", "stderr")
        await environment.download_file("output.txt", tmp_path / "download.txt")
        await environment.download_dir(".", tmp_path / "downloaded")
        await environment.stop(delete=True)
        await environment.stop(delete=True)
        with pytest.raises(EpisodeLaunchError, match="not running"):
            await environment.exec("true")

    run_sync(_run())
    assert (tmp_path / "download.txt").read_text() == "input"
    assert (tmp_path / "downloaded/input.txt").read_text() == "input"
    assert workspace.is_dir()


def test_command_workdir_environment_and_quoted_arguments(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workdir = workspace / "with spaces"
    workdir.mkdir(parents=True)
    environment = ExecutorEnvironment(LocalExecutor(), root=tmp_path, workspace=workspace, env={"SETTING": "default"})

    async def _run() -> None:
        await environment.start(force_build=False)
        result = await environment.exec('pwd; printf "%s" "$SETTING"', cwd="with spaces", env={"SETTING": "value"})
        assert result.stdout == f"{workdir}\nvalue"
        with pytest.raises(ValueError, match="execution root"):
            await environment.exec("true", cwd=str(tmp_path.parent))
        with pytest.raises(ValueError, match="executor user"):
            await environment.exec("true", user="root")
        await environment.stop(delete=True)

    run_sync(_run())


def test_timeout_does_not_leave_a_process_running(tmp_path: Path) -> None:
    environment = ExecutorEnvironment(LocalExecutor(), root=tmp_path, workspace=tmp_path, timeout_seconds=0.05)

    async def _run() -> None:
        await environment.start(force_build=False)
        with pytest.raises(EpisodeTimeout):
            await environment.exec(f"exec {shlex.quote(sys.executable)} -c 'import time; time.sleep(30)'")
        assert environment.process_owner.process_groups == set()
        await environment.stop(delete=True)

    run_sync(_run())


def test_async_cancellation_terminates_the_process_before_returning(tmp_path: Path) -> None:
    environment = ExecutorEnvironment(LocalExecutor(), root=tmp_path, workspace=tmp_path)

    async def _run() -> None:
        await environment.start(force_build=False)
        pending = asyncio.create_task(
            environment.exec(f"exec {shlex.quote(sys.executable)} -c 'import time; time.sleep(30)'")
        )

        async def _wait_for_process() -> tuple[int, ...]:
            while True:
                with environment.process_owner.lock:
                    groups = tuple(environment.process_owner.process_groups)
                if groups:
                    return groups
                await asyncio.sleep(0.01)

        groups = await asyncio.wait_for(_wait_for_process(), timeout=5)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        for process_group in groups:
            with pytest.raises(ProcessLookupError):
                os.killpg(process_group, 0)
        with pytest.raises(EpisodeLaunchError, match="not running"):
            await environment.exec("true")

    run_sync(_run())


class RecordingSession(E2BSession):
    def __init__(self) -> None:
        self.environment: dict[str, str] = {}
        self.pulled: list[tuple[Path, str]] = []
        self.close_calls = 0
        self.fail_close = False

    def launch(
        self,
        argv: Sequence[str],
        *,
        root: Path,
        workspace: Path,
        env: Mapping[str, str],
        timeout: float,
        writable_paths: Sequence[Path] = (),
        readonly_paths: Sequence[Path] = (),
    ) -> ProcessOutcome:
        self.environment = dict(env)
        return ProcessOutcome(0, "remote", "")

    def pull(self, root: Path, relative: str = "") -> None:
        self.pulled.append((root, relative))

    def close(self) -> None:
        self.close_calls += 1
        if self.fail_close:
            raise EpisodeLaunchError("provider cleanup failed")


@dataclass(frozen=True)
class RecordingExecutor(E2BExecutor):
    session: RecordingSession = field(default_factory=RecordingSession)
    deadlines: list[float] = field(default_factory=list)

    def preflight(self) -> None:
        return None

    def open(self) -> E2BSession:
        self.deadlines.append(self.timeout_s)
        return self.session


def test_owned_remote_environment_opens_a_session_and_reports_cleanup_failure(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    executor = RecordingExecutor(env={"SETTING": "deployment"}, timeout_s=1)
    environment = ExecutorEnvironment(
        executor, root=tmp_path, workspace=workspace, env={"SETTING": "caller"}, timeout_seconds=10
    )

    async def _run() -> None:
        await environment.start(force_build=False)
        await environment.start(force_build=False)
        assert executor.deadlines == [10]
        result = await environment.exec("true")
        # The session applies the deployment's own settings; the environment passes the caller's.
        assert result.stdout == "remote" and executor.session.environment["SETTING"] == "caller"
        executor.session.fail_close = True
        with pytest.raises(EpisodeLaunchError, match="cleanup failed"):
            await environment.stop(delete=True)
        executor.session.fail_close = False
        await environment.stop(delete=True)
        assert executor.session.close_calls == 2

    run_sync(_run())


def test_borrowed_remote_session_survives_a_trial_and_refreshes_its_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session = RecordingSession()
    environment: BaseEnvironment = ExecutorEnvironment(session, root=tmp_path, workspace=workspace)

    async def _run() -> None:
        await environment.start(force_build=False)
        await environment.exec("true")
        await environment.download_dir(".", workspace)
        await environment.stop(delete=True)

    run_sync(_run())
    assert session.pulled == [(tmp_path, "workspace")]
    assert session.close_calls == 0


def test_file_transfers_reject_paths_outside_the_root(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    environment = ExecutorEnvironment(LocalExecutor(), root=workspace, workspace=workspace)
    with pytest.raises(ValueError, match="execution root"):
        run_sync(environment.upload_file(tmp_path / "source", "../escaped"))
    with pytest.raises(ValueError, match="execution root"):
        run_sync(environment.download_dir("..", tmp_path / "copy"))
    with pytest.raises(ValueError, match="outside its source"):
        run_sync(environment.download_dir(".", workspace / "copy"))
