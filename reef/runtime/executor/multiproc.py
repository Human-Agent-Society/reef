"""Spawned worker ranks with ordered, non-retrying control RPC.

No Ray/Torch dependency, forked CUDA state, or model parallel rendezvous.
Workers and RPC arguments must be pickleable/importable under Python spawn.
"""

from __future__ import annotations

import multiprocessing
import os
import pickle
import signal
import threading
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from multiprocessing.connection import Connection, wait
from time import monotonic
from typing import Any, Literal, overload

from reef.runtime.executor.base import Executor, ExecutorFuture, resolve_class
from reef.runtime.executor.failure import ExecutorFailure, FailureState
from reef.runtime.executor.uniproc import ConcurrentExecutorFuture


def _terminate_worker(signum, frame) -> None:
    raise SystemExit("worker owner stopped")


def _watch_parent() -> None:
    parent = multiprocessing.parent_process()
    if parent is not None:
        wait([parent.sentinel])
        os.kill(os.getpid(), signal.SIGTERM)


def _reply(connection: Connection, ok: bool, value: Any) -> None:
    try:
        data = pickle.dumps((ok, value))
    except Exception:
        data = pickle.dumps((False, RuntimeError("worker returned an unpickleable result or exception")))
    connection.send_bytes(data)


def _worker_main(connection: Connection, spec_data: bytes, cuda_visible_devices: str | None) -> None:
    # Set visibility before unpickling/importing the worker or scorer module.
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    signal.signal(signal.SIGTERM, _terminate_worker)
    threading.Thread(target=_watch_parent, daemon=True, name="reef-worker-owner").start()
    worker = None
    try:
        spec = pickle.loads(spec_data)
        worker = resolve_class(spec.worker_cls)(*spec.args, **dict(spec.kwargs))
        _reply(connection, True, None)
        while True:
            method, args, kwargs = pickle.loads(connection.recv_bytes())
            if method is None:
                break
            try:
                result = getattr(worker, method)(*args, **kwargs)
            except Exception as exc:
                _reply(connection, False, exc)
            else:
                _reply(connection, True, result)
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        with suppress(Exception):
            _reply(connection, False, exc)
    finally:
        if worker is not None:
            with suppress(Exception):
                shutdown = getattr(worker, "shutdown", None)
                if shutdown is not None:
                    shutdown()
        connection.close()


class _Rank:
    def __init__(
        self, context, spec_data: bytes, index: int, failure_state: FailureState, cuda_visible_devices=None
    ) -> None:
        self.failure_state = failure_state
        self.index = index
        self.connection, child = context.Pipe()
        self.process = context.Process(
            target=_worker_main, args=(child, spec_data, cuda_visible_devices), name=f"reef-worker-{index}"
        )
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"reef-rpc-{index}")
        try:
            self.process.start()
        except BaseException:
            self.connection.close()
            self.pool.shutdown()
            raise
        finally:
            child.close()

    def receive(self) -> Any:
        try:
            ok, value = pickle.loads(self.connection.recv_bytes())
        except (EOFError, OSError) as exc:
            self.failure_state.fail(ExecutorFailure("MultiprocExecutor", "worker exited or disconnected", self.index))
            raise RuntimeError(f"worker process {self.process.pid} exited or disconnected") from exc
        if not ok:
            raise value
        return value

    def call(self, data: bytes) -> Any:
        self.failure_state.check()
        try:
            self.connection.send_bytes(data)
        except (BrokenPipeError, OSError) as exc:
            self.failure_state.fail(ExecutorFailure("MultiprocExecutor", "worker disconnected", self.index))
            raise RuntimeError(f"worker process {self.process.pid} disconnected") from exc
        return self.receive()

    def request_stop(self) -> None:
        # Runs after already submitted RPCs. Busy workers get a bounded grace
        # period before SIGTERM, whose Python handler allows finally cleanup.
        with suppress(OSError):
            self.connection.send_bytes(pickle.dumps((None, (), {})))

    def enqueue_stop(self) -> None:
        try:
            self.pool.submit(self.request_stop)
        except RuntimeError:
            # Python joins RPC threads before atexit/GC finalizers. In that
            # case no RPC thread still owns the pipe.
            self.request_stop()


class MultiprocExecutor(Executor):
    def _init_executor(self) -> None:
        self._ranks: list[_Rank] = []
        self._closed = False
        self._monitor_stop = threading.Event()
        self._shutdown_lock = threading.Lock()
        self._shutdown_complete = threading.Event()
        self._monitor_thread: threading.Thread | None = None
        if self.config.options or any(set(spec.options) - {"cuda_visible_devices"} for spec in self.config.workers):
            raise ValueError("MultiprocExecutor accepts only per-worker cuda_visible_devices, not cluster options")
        for spec in self.config.workers:
            if "cuda_visible_devices" in spec.options and not isinstance(spec.options["cuda_visible_devices"], str):
                raise ValueError("cuda_visible_devices must be a string")
        # Serialize every worker before starting any child; no silent fallback
        # to fork, cloudpickle or shared in-process objects for closures/locks.
        try:
            specs = [pickle.dumps(spec) for spec in self.config.workers]
        except Exception as exc:
            raise TypeError(
                "mp workers must be spawn-pickleable; use importable classes/scorers or one uni worker"
            ) from exc
        context = multiprocessing.get_context("spawn")
        timeout = self.config.launch_timeout_s
        deadline = None if timeout is None else monotonic() + timeout
        try:
            for index, spec_data in enumerate(specs):
                self._ranks.append(
                    _Rank(
                        context,
                        spec_data,
                        index,
                        self._failure_state,
                        self.config.workers[index].options.get("cuda_visible_devices"),
                    )
                )
            for rank in self._ranks:
                remaining = None if deadline is None else max(0.0, deadline - monotonic())
                if not rank.connection.poll(remaining):
                    raise TimeoutError("mp worker startup timed out")
                rank.receive()
            if self._ranks:
                self._monitor_thread = threading.Thread(
                    target=self._monitor_workers, daemon=True, name="reef-mp-monitor"
                )
                self._monitor_thread.start()
        except BaseException:
            self.shutdown()
            raise

    def _ensure_open(self) -> None:
        self._failure_state.check()
        if self._closed:
            raise RuntimeError("executor is shut down")

    def _monitor_workers(self) -> None:
        sentinels = [rank.process.sentinel for rank in self._ranks]
        while not self._monitor_stop.is_set():
            died = wait(sentinels, timeout=0.1)
            if self._monitor_stop.is_set():
                return
            if died or self.failure is not None:
                rank = sentinels.index(died[0]) if died else None
                self._fail("worker exited unexpectedly", rank=rank)
                self.shutdown()
                return

    @overload
    def collective_rpc(
        self,
        method: str,
        *,
        args: tuple[Any, ...] = (),
        kwargs: Mapping[str, Any] | None = None,
        timeout: float | None = None,
        non_block: Literal[False] = False,
    ) -> list[Any]: ...

    @overload
    def collective_rpc(
        self,
        method: str,
        *,
        args: tuple[Any, ...] = (),
        kwargs: Mapping[str, Any] | None = None,
        timeout: float | None = None,
        non_block: Literal[True],
    ) -> ExecutorFuture: ...

    @overload
    def collective_rpc(
        self,
        method: str,
        *,
        args: tuple[Any, ...] = (),
        kwargs: Mapping[str, Any] | None = None,
        timeout: float | None = None,
        non_block: bool,
    ) -> list[Any] | ExecutorFuture: ...

    def collective_rpc(
        self,
        method: str,
        *,
        args: tuple[Any, ...] = (),
        kwargs: Mapping[str, Any] | None = None,
        timeout: float | None = None,
        non_block: bool = False,
    ) -> list[Any] | ExecutorFuture:
        self._ensure_open()
        data = pickle.dumps((method, args, dict(kwargs or {})))
        pending = [self._failure_state.track(rank.pool.submit(rank.call, data)) for rank in self._ranks]
        future = ConcurrentExecutorFuture(pending, timeout=timeout)
        return future if non_block else future.result()

    def rpc(
        self,
        rank: int,
        method: str,
        *,
        args: tuple[Any, ...] = (),
        kwargs: Mapping[str, Any] | None = None,
        timeout: float | None = None,
        non_block: bool = False,
    ) -> Any:
        self._ensure_open()
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0 or rank >= len(self._ranks):
            raise IndexError(f"worker rank {rank!r} is outside the executor group")
        data = pickle.dumps((method, args, dict(kwargs or {})))
        worker = self._ranks[rank]
        pending = self._failure_state.track(worker.pool.submit(worker.call, data))
        future = ConcurrentExecutorFuture([pending], single=True, timeout=timeout)
        return future if non_block else future.result()

    def check_health(self, timeout: float | None = None) -> None:
        self._ensure_open()
        for rank in self._ranks:
            if not rank.process.is_alive():
                raise RuntimeError(f"worker process {rank.process.pid} exited with code {rank.process.exitcode}")

    def shutdown(self) -> None:
        with self._shutdown_lock:
            already_closed = self._closed
            if not already_closed:
                self._closed = True
                self._monitor_stop.set()
                self._failure_state.close()
        if already_closed:
            # The caller may arrive while the monitor is retiring peer ranks.
            # A monitor must not wait for a caller that is joining that monitor.
            if threading.current_thread() is not self._monitor_thread:
                self._shutdown_complete.wait()
            return
        try:
            self._shutdown_workers()
        finally:
            self._shutdown_complete.set()

    def _shutdown_workers(self) -> None:
        if self._monitor_thread is not None and self._monitor_thread is not threading.current_thread():
            self._monitor_thread.join()
        for rank in self._ranks:
            rank.enqueue_stop()
        if self.failure is not None:
            for rank in self._ranks:
                if rank.process.is_alive():
                    rank.process.terminate()
        deadline = monotonic() + 5
        for rank in self._ranks:
            rank.process.join(max(0, deadline - monotonic()))
        for rank in self._ranks:
            if rank.process.is_alive():
                rank.process.terminate()
        deadline = monotonic() + 5
        for rank in self._ranks:
            rank.process.join(max(0, deadline - monotonic()))
            if rank.process.is_alive():
                rank.process.kill()
                rank.process.join()
            rank.pool.shutdown(wait=True, cancel_futures=True)
            rank.connection.close()
            rank.process.close()
