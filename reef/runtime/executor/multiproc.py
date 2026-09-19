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
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from multiprocessing.connection import Connection, wait
from time import monotonic
from typing import Any

from reef.runtime.executor.base import ExecutorFuture, SubmittingExecutor, check_rank, remaining_time, resolve_class
from reef.runtime.executor.failure import ExecutorFailure, FailureState
from reef.runtime.executor.uniproc import ConcurrentExecutorFuture, shutdown_worker

#: How long ranks get to exit on their own before SIGTERM, and after SIGTERM before SIGKILL.
STOP_GRACE_S = 5.0

# -- Worker process --------------------------------------------------------


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


def _serve(connection: Connection, worker: Any) -> None:
    """Answer RPCs until the owner sends the ``None`` stop request."""
    while True:
        method, args, kwargs = pickle.loads(connection.recv_bytes())
        if method is None:
            return
        try:
            result = getattr(worker, method)(*args, **kwargs)
        except Exception as exc:
            _reply(connection, False, exc)
        else:
            _reply(connection, True, result)


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
        _serve(connection, worker)
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        with suppress(Exception):
            _reply(connection, False, exc)
    finally:
        if worker is not None:
            with suppress(Exception):
                shutdown_worker(worker)
        connection.close()


# -- Owner side ------------------------------------------------------------


class _Rank:
    """The owner's end of one spawned worker: its process, pipe, and RPC thread."""

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

    def _disconnected(self, reason: str, exc: Exception) -> RuntimeError:
        self.failure_state.fail(ExecutorFailure("MultiprocExecutor", reason, self.index))
        return RuntimeError(f"worker process {self.process.pid} {reason}")

    def receive(self) -> Any:
        try:
            ok, value = pickle.loads(self.connection.recv_bytes())
        except (EOFError, OSError) as exc:
            raise self._disconnected("exited or disconnected", exc) from exc
        if not ok:
            raise value
        return value

    def call(self, data: bytes) -> Any:
        self.failure_state.check()
        try:
            self.connection.send_bytes(data)
        except (BrokenPipeError, OSError) as exc:
            raise self._disconnected("disconnected", exc) from exc
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

    def close(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.connection.close()
        self.process.close()


def _join_all(ranks: Iterable[_Rank], grace_s: float) -> None:
    deadline = monotonic() + grace_s
    for rank in ranks:
        rank.process.join(remaining_time(deadline))


def _alive(ranks: Iterable[_Rank]) -> list[_Rank]:
    return [rank for rank in ranks if rank.process.is_alive()]


class MultiprocExecutor(SubmittingExecutor):
    """Spawn one process per worker and monitor them for unexpected exit."""

    def _init_executor(self) -> None:
        self._ranks: list[_Rank] = []
        self._monitor_stop = threading.Event()
        self._shutdown_lock = threading.Lock()
        self._shutdown_complete = threading.Event()
        self._monitor_thread: threading.Thread | None = None
        if self.config.options:
            raise ValueError("MultiprocExecutor accepts only per-worker cuda_visible_devices, not cluster options")
        specs = [self._spawn_spec(spec) for spec in self.config.workers]
        context = multiprocessing.get_context("spawn")
        deadline = None if self.config.launch_timeout_s is None else monotonic() + self.config.launch_timeout_s
        try:
            for index, (spec_data, devices) in enumerate(specs):
                self._ranks.append(_Rank(context, spec_data, index, self._failure_state, devices))
            for rank in self._ranks:
                if not rank.connection.poll(remaining_time(deadline)):
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

    @staticmethod
    def _spawn_spec(spec: Any) -> tuple[bytes, str | None]:
        """Validate one worker's options and serialize it before any child starts.

        Every worker is pickled up front: there is no silent fallback to fork,
        cloudpickle, or shared in-process objects for closures and locks.
        """
        if set(spec.options) - {"cuda_visible_devices"}:
            raise ValueError("MultiprocExecutor accepts only per-worker cuda_visible_devices, not cluster options")
        devices = spec.options.get("cuda_visible_devices")
        if devices is not None and not isinstance(devices, str):
            raise ValueError("cuda_visible_devices must be a string")
        try:
            return pickle.dumps(spec), devices
        except Exception as exc:
            raise TypeError(
                "mp workers must be spawn-pickleable; use importable classes/scorers or one uni worker"
            ) from exc

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

    def _pending(self, rank: _Rank, data: bytes) -> Any:
        return self._failure_state.track(rank.pool.submit(rank.call, data))

    def _submit(
        self, rank: int, method: str, args: tuple[Any, ...], kwargs: dict[str, Any], timeout: float | None
    ) -> ExecutorFuture:
        check_rank(rank, len(self._ranks))
        data = pickle.dumps((method, args, kwargs))
        return ConcurrentExecutorFuture([self._pending(self._ranks[rank], data)], single=True, timeout=timeout)

    def _submit_all(
        self, method: str, args: tuple[Any, ...], kwargs: dict[str, Any], timeout: float | None
    ) -> ExecutorFuture:
        data = pickle.dumps((method, args, kwargs))
        return ConcurrentExecutorFuture([self._pending(rank, data) for rank in self._ranks], timeout=timeout)

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
            # Peers of a dead rank get no grace: the group cannot continue.
            for rank in _alive(self._ranks):
                rank.process.terminate()
        _join_all(self._ranks, STOP_GRACE_S)
        for rank in _alive(self._ranks):
            rank.process.terminate()
        _join_all(self._ranks, STOP_GRACE_S)
        for rank in self._ranks:
            if rank.process.is_alive():
                rank.process.kill()
                rank.process.join()
            rank.close()
