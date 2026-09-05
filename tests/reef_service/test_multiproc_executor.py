from __future__ import annotations

import multiprocessing
import os
import sys
import threading
import time
from pathlib import Path

import pytest

from reef.runtime.executor import Executor, ExecutorConfig, ExecutorFailedError, WorkerSpec, resolve


class Counter:
    def __init__(self, rank=0, fail=False, marker=None):
        if fail:
            raise ValueError("constructor failed")
        self.rank = rank
        self.count = 0
        self.marker = marker

    def identity(self):
        return self.rank, os.getpid(), "ray" in sys.modules

    def increment(self, delay=0):
        time.sleep(delay)
        self.count += 1
        return self.count

    def fail(self):
        self.count += 1
        raise ValueError("not retried")

    def count_calls(self):
        return self.count

    def crash(self):
        os._exit(7)

    def shutdown(self):
        if self.marker:
            Path(self.marker).write_text("stopped")


class Rendezvous:
    def __init__(self, directory, rank):
        self.directory = Path(directory)
        self.rank = rank

    def meet(self):
        (self.directory / str(self.rank)).touch()
        deadline = time.monotonic() + 10
        while len(list(self.directory.iterdir())) < 2:
            if time.monotonic() > deadline:
                raise TimeoutError("collective was not dispatched concurrently")
            time.sleep(0.01)
        return self.rank


class EpisodeSleeper:
    def run(self, directory):
        from reef.harness.executor import EPISODE_OWNER_LEASE, LocalExecutor

        root = Path(directory)
        token = EPISODE_OWNER_LEASE.set(True)
        try:
            return LocalExecutor().launch(
                [
                    sys.executable,
                    "-c",
                    "import os,time; from pathlib import Path; "
                    "Path('child.pid').write_text(str(os.getpid())); time.sleep(120)",
                ],
                root=root,
                workspace=root,
                env={},
                timeout=120,
            )
        finally:
            EPISODE_OWNER_LEASE.reset(token)


def test_uni_is_one_in_process_worker():
    executor = Executor.create(ExecutorConfig(backend="uni", workers=(WorkerSpec(Counter),)))
    try:
        assert executor.rpc(0, "identity")[1] == os.getpid()
    finally:
        executor.shutdown()
    with pytest.raises(ValueError, match="at most one"):
        Executor.create(ExecutorConfig(backend="uni", workers=(WorkerSpec(Counter), WorkerSpec(Counter))))


def test_mp_spawns_distinct_cpu_ranks_with_ordered_rpc_and_cleanup(tmp_path):
    executor = Executor.create(
        ExecutorConfig(
            backend="mp",
            workers=tuple(
                WorkerSpec(Counter, args=(rank,), kwargs={"marker": str(tmp_path / str(rank))}) for rank in range(2)
            ),
        )
    )
    try:
        identities = executor.collective_rpc("identity")
        pids = [row[1] for row in identities]
        assert len(set(pids)) == 2 and os.getpid() not in pids
        assert [row[0] for row in identities] == [0, 1]
        assert not any(row[2] for row in identities)
        futures = [executor.rpc(0, "increment", non_block=True) for _ in range(4)]
        assert resolve(futures) == [1, 2, 3, 4]
        with pytest.raises(ValueError, match="not retried"):
            executor.rpc(0, "fail")
        assert executor.rpc(0, "count_calls") == 5
        executor.check_health()
    finally:
        executor.shutdown()
    executor.shutdown()
    assert all((tmp_path / str(rank)).read_text() == "stopped" for rank in range(2))
    assert not any(child.pid in pids for child in multiprocessing.active_children())


def test_mp_collective_dispatches_all_ranks_before_waiting(tmp_path):
    executor = Executor.create(
        ExecutorConfig(
            backend="mp", workers=tuple(WorkerSpec(Rendezvous, args=(str(tmp_path), rank)) for rank in range(2))
        )
    )
    try:
        assert executor.collective_rpc("meet", timeout=15) == [0, 1]
    finally:
        executor.shutdown()


def test_mp_timeout_does_not_retry_or_cancel_work():
    executor = Executor.create(ExecutorConfig(backend="mp", workers=(WorkerSpec(Counter),)))
    try:
        future = executor.rpc(0, "increment", args=(0.2,), non_block=True)
        with pytest.raises(TimeoutError):
            future.result(timeout=0.01)
        assert future.result(timeout=5) == 1
        assert executor.rpc(0, "count_calls") == 1
        with pytest.raises(IndexError):
            executor.rpc(-1, "identity")
    finally:
        executor.shutdown()


def test_mp_constructor_failure_rolls_back_started_ranks(tmp_path):
    before = {child.pid for child in multiprocessing.active_children()}
    with pytest.raises(ValueError, match="constructor failed"):
        Executor.create(
            ExecutorConfig(
                backend="mp",
                workers=(
                    WorkerSpec(Counter, kwargs={"marker": str(tmp_path / "stopped")}),
                    WorkerSpec(Counter, kwargs={"fail": True}),
                ),
            )
        )
    assert (tmp_path / "stopped").read_text() == "stopped"
    assert {child.pid for child in multiprocessing.active_children()} == before


def test_mp_worker_death_fails_rpc_and_health_without_hanging():
    executor = Executor.create(ExecutorConfig(backend="mp", workers=(WorkerSpec(Counter),)))
    try:
        with pytest.raises(ExecutorFailedError, match=r"exited|disconnected"):
            executor.rpc(0, "crash", timeout=5)
        with pytest.raises(RuntimeError, match="exited"):
            executor.check_health()
    finally:
        executor.shutdown()


def test_idle_rank_death_proactively_fails_other_rank_and_notifies_once():
    class Listener:
        def __init__(self):
            self.events = []
            self.received = threading.Event()

        def on_executor_failure(self, failure):
            self.events.append(failure)
            self.received.set()

    executor = Executor.create(ExecutorConfig(backend="mp", workers=(WorkerSpec(Counter),) * 2))
    listener = Listener()
    executor.register_failure_listener(listener)
    pids = [row[1] for row in executor.collective_rpc("identity")]
    try:
        pending = executor.rpc(0, "increment", args=(120,), non_block=True)
        queued = executor.rpc(0, "increment", non_block=True)
        executor._ranks[1].process.kill()  # No RPC on this idle rank to discover its death.
        assert listener.received.wait(5)
        assert listener.events[0].rank == 1
        for future in (pending, queued):
            with pytest.raises(ExecutorFailedError):
                future.result(timeout=2)
        with pytest.raises(ExecutorFailedError):
            executor.rpc(0, "identity")
        executor.register_failure_listener(listener)
        late = Listener()
        executor.register_failure_listener(late)
        assert late.events == listener.events
        assert len(listener.events) == 1
    finally:
        executor.shutdown()
    assert not any(child.pid in pids for child in multiprocessing.active_children())


def test_mp_rejects_unpickleable_worker_before_starting_any_rank():
    before = {child.pid for child in multiprocessing.active_children()}
    with pytest.raises(TypeError, match="spawn-pickleable"):
        Executor.create(ExecutorConfig(backend="mp", workers=(WorkerSpec(Counter, kwargs={"marker": lambda: None}),)))
    assert {child.pid for child in multiprocessing.active_children()} == before


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group leases")
def test_killed_worker_does_not_leave_running_harness(tmp_path):
    executor = Executor.create(ExecutorConfig(backend="mp", workers=(WorkerSpec(EpisodeSleeper),)))
    try:
        pending = executor.rpc(0, "run", args=(str(tmp_path),), non_block=True)
        marker = tmp_path / "child.pid"
        deadline = time.monotonic() + 10
        while not marker.exists():
            if time.monotonic() > deadline:
                pytest.fail("harness failed to start")
            time.sleep(0.01)
        pid = int(marker.read_text())
        executor._ranks[0].process.kill()
        with pytest.raises(RuntimeError):
            pending.result(timeout=5)
        while time.monotonic() < deadline:
            # A killed orphan can briefly remain a zombie on CI; it must not run.
            import subprocess

            state = subprocess.run(
                ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True
            ).stdout.strip()
            if not state or state.startswith("Z"):
                break
            time.sleep(0.02)
        else:
            pytest.fail("harness survived worker death")
    finally:
        executor.shutdown()
