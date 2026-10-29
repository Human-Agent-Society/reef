"""Harbor environments backed by Reef's existing execution implementations."""

from __future__ import annotations

import asyncio
import contextlib
import math
import shlex
import shutil
from collections.abc import Coroutine, Mapping, Sequence
from pathlib import Path
from threading import Event, Lock
from typing import TypeVar

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

from reef.execution.lifecycle import Execution, validate_wait_timeout
from reef.execution.sandbox.process import (
    PROCESS_OWNER,
    ProcessOutcome,
    ProcessOwner,
    ProgramExecutor,
    RolloutLaunchError,
    RolloutTimeout,
)

ResultType = TypeVar("ResultType")


def run_sync(operation: Coroutine[object, object, ResultType]) -> ResultType:
    """Enter Harbor's async API from Reef's synchronous execution callers."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is None:
        return asyncio.run(operation)
    operation.close()
    raise RuntimeError("use the asynchronous environment API from a running event loop")


class ExecutorEnvironment(BaseEnvironment):
    """Use Harbor's API without changing an executor's isolation boundary.

    Paths refer to the caller's staged root, including for cloud execution.
    File transfers preserve the remote root mapping. The caller owns the staged
    root; stopping this environment never deletes caller-owned local files.
    Starting opens the executor's session (:meth:`ProgramExecutor.open_session`).
    When that opens a new one, such as an E2B VM, the environment owns it and
    stopping closes it. An executor that is already a session, such as an open
    E2B session, is borrowed, so stopping this view leaves it running.
    Cancelling execution retires the environment and its processes, including
    a borrowed session.
    """

    def __init__(
        self,
        executor: ProgramExecutor,
        *,
        root: Path,
        workspace: Path,
        env: Mapping[str, str] | None = None,
        timeout_seconds: float = 600,
        writable_paths: Sequence[Path] = (),
        readonly_paths: Sequence[Path] = (),
    ) -> None:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        self.root = root.absolute()
        self.workspace = workspace.absolute()
        if not self.workspace.is_relative_to(self.root):
            raise ValueError("workspace must be inside the execution root")
        self.executor = executor
        self.timeout_seconds = timeout_seconds
        self.writable_paths = tuple(path.absolute() for path in writable_paths)
        self.readonly_paths = tuple(path.absolute() for path in readonly_paths)
        if any(not path.is_relative_to(self.root) for path in (*self.writable_paths, *self.readonly_paths)):
            raise ValueError("execution state and inputs must be inside the execution root")
        self.process_owner = ProcessOwner()
        self.started = False
        self.closed = False
        self.owns_session = False
        super().__init__(
            environment_dir=self.root,
            environment_name=self.root.name,
            session_id=self.root.name,
            trial_paths=TrialPaths(trial_dir=self.root),
            task_env_config=EnvironmentConfig(),
            persistent_env=dict(env or {}),
        )

    @staticmethod
    def type() -> str:
        return "reef"

    # Harbor requires this named override to validate the environment definition.
    def _validate_definition(self) -> None:
        if not self.root.is_dir() or not self.workspace.is_dir():
            raise FileNotFoundError("the execution root and workspace must already exist")

    async def start(self, force_build: bool) -> None:
        if self.closed:
            raise RolloutLaunchError("the execution environment was stopped")
        if self.started:
            return
        if force_build:
            raise ValueError("Reef executors do not support forced template rebuilding")
        await asyncio.to_thread(self.executor.preflight)
        executor = self.executor
        opening = asyncio.create_task(asyncio.to_thread(executor.open_session, timeout_s=self.timeout_seconds))
        try:
            opened = await asyncio.shield(opening)
        except asyncio.CancelledError:
            opened = await asyncio.shield(opening)
            self.executor = opened
            self.owns_session = opened is not executor
            if opened is not executor:
                await asyncio.to_thread(opened.close)
            self.closed = True
            raise
        self.executor = opened
        self.owns_session = opened is not executor
        self.started = True

    async def stop(self, delete: bool) -> None:
        if self.closed:
            return
        if self.owns_session and not delete:
            raise ValueError("retaining a stopped sandbox session is not supported")
        self.process_owner.stop()
        await asyncio.to_thread(self.process_owner.scope.wait_closed)
        if self.owns_session:
            await asyncio.to_thread(self.executor.close)
        self.closed = True
        self.started = False

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        if user is not None:
            raise ValueError("Reef environments execute as the configured executor user")
        workdir = self.workspace if cwd is None else Path(cwd)
        if not workdir.is_absolute():
            workdir = self.workspace / workdir
        if not workdir.resolve().is_relative_to(self.root.resolve()):
            raise ValueError("cwd must be inside the execution root")
        return await self.exec_argv(
            ["/bin/sh", "-c", f"cd {shlex.quote(str(workdir))} && {command}"], env=env, timeout_sec=timeout_sec
        )

    async def exec_argv(
        self, argv: Sequence[str], env: dict[str, str] | None = None, timeout_sec: int | None = None
    ) -> ExecResult:
        """Preserve argv semantics for existing synchronous executor callers."""
        if not self.started or self.closed:
            raise RolloutLaunchError("the execution environment is not running")
        if not argv:
            raise ValueError("the execution command must contain a binary")
        timeout = self.timeout_seconds if timeout_sec is None else float(timeout_sec)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout_sec must be finite and positive")
        token = PROCESS_OWNER.set(self.process_owner)
        try:
            pending = asyncio.create_task(
                asyncio.to_thread(
                    self.executor.launch,
                    argv,
                    root=self.root,
                    workspace=self.workspace,
                    env=self._merge_env(env) or {},
                    timeout=timeout,
                    writable_paths=self.writable_paths,
                    readonly_paths=self.readonly_paths,
                )
            )
            try:
                outcome = await asyncio.shield(pending)
            except asyncio.CancelledError:
                self.process_owner.stop()
                self.started = False
                self.owns_session = True
                try:
                    await asyncio.to_thread(self.executor.close)
                finally:
                    with contextlib.suppress(RolloutLaunchError, RolloutTimeout):
                        await asyncio.shield(pending)
                self.closed = True
                raise
        finally:
            PROCESS_OWNER.reset(token)
        return ExecResult(stdout=outcome.stdout, stderr=outcome.stderr, return_code=outcome.exit_code)

    def staged_path(self, path: str) -> Path:
        if self.closed:
            raise RolloutLaunchError("the execution environment was stopped")
        target = Path(path)
        if not target.is_absolute():
            target = self.workspace / target
        if not target.resolve().is_relative_to(self.root.resolve()):
            raise ValueError("file transfer paths must be inside the execution root")
        return target

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        target = self.staged_path(target_path)
        await asyncio.to_thread(copy_path, Path(source_path), target, directory=False)
        await asyncio.to_thread(self.executor.push, self.root)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        target = self.staged_path(target_dir)
        await asyncio.to_thread(copy_path, Path(source_dir), target, directory=True)
        await asyncio.to_thread(self.executor.push, self.root)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        source = self.staged_path(source_path)
        relative = source.parent.relative_to(self.root).as_posix()
        await asyncio.to_thread(self.executor.pull, self.root, "" if relative == "." else relative)
        target = Path(target_path)
        await asyncio.to_thread(copy_path, source, target, directory=False)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        source = self.staged_path(source_dir)
        relative = source.relative_to(self.root).as_posix()
        await asyncio.to_thread(self.executor.pull, self.root, "" if relative == "." else relative)
        target = Path(target_dir)
        await asyncio.to_thread(copy_path, source, target, directory=True)


class ProgramExecution(Execution):
    """One program run with its own environment lifecycle and concrete ProcessOutcome.

    environment is an unstarted ExecutorEnvironment supplied by the caller;
    this run exclusively owns that view. argv is a nonempty tuple of NUL-free
    strings with a nonempty executable. run or run_async may start it once.
    A stop before start closes admission without opening the environment.
    A stop during the run cancels its task; the environment reaps local work
    and closes the remote session according to its ownership contract.

    gate protects started, stopped, loop and task. finished starts unset and
    is set after cleanup settles. cleanup_error is None unless cleanup raised;
    wait_closed raises that error rather than claiming successful closure.
    run returns ProcessOutcome after cleanup, or propagates launch, timeout,
    cancellation or cleanup errors. The executor itself remains a factory or
    session resource, not an Execution.
    """

    def __init__(self, environment: ExecutorEnvironment, argv: Sequence[str]) -> None:
        if not isinstance(environment, ExecutorEnvironment) or environment.started or environment.closed:
            raise ValueError("program execution requires an unstarted environment")
        if isinstance(argv, str):
            raise ValueError("argv must be a sequence of arguments, not text")
        self.argv = tuple(argv)
        if not self.argv or not all(isinstance(arg, str) and "\0" not in arg for arg in self.argv) or not self.argv[0]:
            raise ValueError("argv must contain a nonempty executable and NUL-free arguments")
        self.environment = environment
        self.gate = Lock()
        self.started = False
        self.stopped = False
        self.loop: asyncio.AbstractEventLoop | None = None
        self.task: asyncio.Task[object] | None = None
        self.finished = Event()
        self.cleanup_error: Exception | None = None

    def request_stop(self) -> None:
        """Cancel this run once; cancellation completes only after environment cleanup."""
        with self.gate:
            if self.stopped or self.finished.is_set():
                return
            self.stopped = True
            if not self.started:
                self.finished.set()
            elif self.loop is not None and self.task is not None:
                self.loop.call_soon_threadsafe(self.task.cancel)

    def wait_closed(self, timeout_seconds: float | None = None) -> bool:
        """Wait for this run's cleanup, raising any failure to release its resources."""
        validate_wait_timeout(timeout_seconds)
        if not self.finished.wait(timeout_seconds):
            return False
        if self.cleanup_error is not None:
            raise self.cleanup_error
        return True

    def run(self) -> ProcessOutcome:
        """Start this run synchronously and return its concrete result after cleanup."""
        return run_sync(self.run_async())

    async def run_async(self) -> ProcessOutcome:
        """Start this run on the current loop; callers on that loop must not block in wait_closed."""
        with self.gate:
            if self.started or self.stopped:
                raise RolloutLaunchError("the program execution was already started or stopped")
            self.started = True
            self.loop = asyncio.get_running_loop()
            self.task = asyncio.current_task()
        try:
            await self.environment.start(force_build=False)
            result = await self.environment.exec_argv(self.argv)
            return ProcessOutcome(result.return_code, result.stdout or "", result.stderr or "")
        finally:
            # A stop during cleanup must not abandon the cleanup coroutine.
            cleanup = asyncio.create_task(self.environment.stop(delete=True))
            try:
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    await cleanup
                    raise
            except Exception as exc:
                self.cleanup_error = exc
                raise
            finally:
                with self.gate:
                    self.loop = None
                    self.task = None
                    self.finished.set()


def execute(
    executor: ProgramExecutor,
    argv: Sequence[str],
    *,
    root: Path,
    workspace: Path,
    env: Mapping[str, str],
    timeout: float,
    writable_paths: Sequence[Path] = (),
    readonly_paths: Sequence[Path] = (),
) -> ProcessOutcome:
    """Run an existing rollout caller through Harbor's environment interface."""
    environment = ExecutorEnvironment(
        executor,
        root=root,
        workspace=workspace,
        env=env,
        timeout_seconds=timeout,
        writable_paths=writable_paths,
        readonly_paths=readonly_paths,
    )

    return ProgramExecution(environment, argv).run()


def download_directory(executor: ProgramExecutor, root: Path, relative: str) -> None:
    """Refresh a running executor's staged directory through Harbor's API."""
    environment: BaseEnvironment = ExecutorEnvironment(executor, root=root, workspace=root)
    target = root / relative
    run_sync(environment.download_dir(str(target), target))


def copy_path(source: Path, target: Path, *, directory: bool) -> None:
    """Copy staged files or directory trees without copying a tree into itself."""
    if source.resolve() == target.resolve():
        return
    if directory and target.resolve().is_relative_to(source.resolve()):
        raise ValueError("a directory transfer destination must be outside its source")
    target.parent.mkdir(parents=True, exist_ok=True)
    if directory:
        shutil.copytree(source, target, symlinks=True, dirs_exist_ok=True)
    else:
        shutil.copy2(source, target)
