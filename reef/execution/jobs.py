"""Asynchronous jobs off the trainer thread: a job runner and the worker that runs it.

Pipeline stages use the worker for slow work that must not block the trainer,
such as judge calls or task-generation rounds. The generator service runs its
proposals, checks and rollouts on one worker. Its process ownership uses the
reef.execution.processes module and shared lifecycle contract.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from threading import Event, Lock, Thread

from reef.execution.lifecycle import Execution, validate_wait_timeout
from reef.execution.processes import CURRENT_PROCESS_SCOPE, ProcessScope

logger = logging.getLogger(__name__)

#: How long ``close()`` lets in-flight jobs finish before cancelling.
CLOSE_GRACE_S = 5.0
PROCESS_CLOSE_GRACE_S = 10.0


@dataclass(frozen=True)
class SupportsReceipt:
    """The receipt of a job or of its result. The worker reads only this field.

    The caller defines the job and result types. Both types, and
    :class:`Failed`, use this class as their base.
    """

    receipt: str


@dataclass(frozen=True)
class Failed(SupportsReceipt):
    """The worker's result for a job that never finished."""


class JobContext(Execution):
    """Resources and cancellation state of one accepted execution.

    processes is a fresh ProcessScope; cancelled is an initially unset Event.
    request_stop sets cancelled and closes processes, idempotently. finish
    waits for process reaping after the runner exits, then sets finished.
    finished is an initially unset Event; wait_closed waits for that event.
    cleanup_error records a cleanup exception, which wait_closed re-raises.
    gate protects the optional loop and task, bound once by the worker.
    A stop cancels that task on its owning loop. The worker creates one context per running
    submission and binds its scope across async calls and asyncio.to_thread.
    Runners must await their own asynchronous cleanup and cannot detach work.
    """

    def __init__(self) -> None:
        self.processes = ProcessScope()
        self.cancelled = Event()
        self.finished = Event()
        self.cleanup_error: Exception | None = None
        self.gate = Lock()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.task: asyncio.Task[object] | None = None
        self.bound = False

    def bind_task(self) -> None:
        """Bind the current worker task once so another thread can request its cancellation."""
        with self.gate:
            if self.bound or self.finished.is_set():
                raise RuntimeError("job context was already bound or finished")
            self.bound = True
            self.loop = asyncio.get_running_loop()
            self.task = asyncio.current_task()

    def request_stop(self) -> None:
        with self.gate:
            if self.finished.is_set() or self.cancelled.is_set():
                return
            self.cancelled.set()
            if self.loop is not None and self.task is not None:
                self.loop.call_soon_threadsafe(self.task.cancel)
        self.processes.request_stop(grace_seconds=PROCESS_CLOSE_GRACE_S)

    def wait_closed(self, timeout_seconds: float | None = None) -> bool:
        """Wait until the runner and its owned processes have finished cleanup."""
        validate_wait_timeout(timeout_seconds)
        if not self.finished.wait(timeout_seconds):
            return False
        if self.cleanup_error is not None:
            raise self.cleanup_error
        return True

    async def finish(self) -> None:
        """Close and reap owned processes before the worker publishes the result."""
        with self.gate:
            self.loop = None
            self.task = None
        self.processes.request_stop(grace_seconds=PROCESS_CLOSE_GRACE_S)
        cleanup = asyncio.create_task(asyncio.to_thread(self.processes.wait_closed))
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
            self.finished.set()


class BackgroundJobRunner(ABC):
    """An asynchronous job implementation consumed by BackgroundJobWorker.

    run returns one SupportsReceipt with the submitted receipt or raises.
    run_in_context adds owned resources without changing existing run callers.
    Implementations must cooperate with cancellation, await cleanup, and not
    leave detached threads, tasks or remote resources after returning.
    """

    @abstractmethod
    async def run(self, job: SupportsReceipt) -> SupportsReceipt:
        """Return the result for job; exceptions become Failed in the worker."""

    async def run_in_context(self, job: SupportsReceipt, context: JobContext) -> SupportsReceipt:
        """Run with worker-owned resources; existing runners retain their run interface."""
        return await self.run(job)


class BackgroundJobWorker(Execution):
    """A queue of submissions, at most concurrency runners, and a queue of results.

    runner implements BackgroundJobRunner; thread_name is nonempty text and
    concurrency a positive integer. Each accepted submission gets exactly one
    result after runner and owned-process cleanup, even for repeated receipts.
    Exceptions and cancellation yield Failed. poll consumes results once in
    completion order. closed means admission stopped; broken means the worker
    loop is retiring or has exited. Neither alone proves cleanup: wait_closed checks thread exit.
    stopping is set when admission closes. cleanup_error is None or a reported
    job cleanup failure; wait_closed re-raises it after thread exit.
    A runner that refuses cancellation can keep wait_closed returning False.
    """

    def __init__(self, runner: BackgroundJobRunner, *, thread_name: str, concurrency: int = 8) -> None:
        if not isinstance(runner, BackgroundJobRunner):
            raise ValueError("runner must implement BackgroundJobRunner")
        if not isinstance(thread_name, str) or not thread_name:
            raise ValueError("thread_name must be nonempty text")
        if type(concurrency) is not int or concurrency < 1:
            raise ValueError("concurrency must be a positive integer")
        self.runner = runner
        self.thread_name = thread_name
        self.concurrency = concurrency
        self.jobs: queue.SimpleQueue[SupportsReceipt | None] = queue.SimpleQueue()
        self.results: queue.SimpleQueue[SupportsReceipt] = queue.SimpleQueue()
        self.thread: Thread | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.wake: asyncio.Event | None = None
        self.gate = Lock()
        self.broken = False
        self.closed = False
        self.stopping = Event()
        self.cleanup_error: Exception | None = None

    def submit(self, job: SupportsReceipt) -> bool:
        """Accept one submission unless closing or broken; repeated receipts are independent submissions."""
        with self.gate:
            if self.closed or self.broken:
                return False
            if self.thread is None:
                self.thread = Thread(target=self.run_thread, name=self.thread_name, daemon=True)
                self.thread.start()
            self.jobs.put(job)
            if self.loop is not None and self.wake is not None:
                self.loop.call_soon_threadsafe(self.wake.set)
            return True

    def poll(self) -> list[SupportsReceipt]:
        """Consume currently available results without blocking."""
        results: list[SupportsReceipt] = []
        try:
            while True:
                results.append(self.results.get_nowait())
        except queue.Empty:
            return results

    def request_stop(self) -> None:
        """Stop admission and wake the worker; running jobs receive CLOSE_GRACE_S before cancellation."""
        with self.gate:
            if self.closed:
                return
            self.closed = True
            self.stopping.set()
            self.jobs.put(None)
            if self.loop is not None and self.wake is not None:
                self.loop.call_soon_threadsafe(self.wake.set)

    def wait_closed(self, timeout_seconds: float | None = None) -> bool:
        """Wait for admission to stop and the worker thread to exit, sharing one deadline."""
        validate_wait_timeout(timeout_seconds)
        deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
        if not self.stopping.wait(timeout_seconds):
            return False
        with self.gate:
            thread = self.thread
        if thread is None:
            return True
        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        thread.join(timeout=remaining)
        if thread.is_alive():
            return False
        if self.cleanup_error is not None:
            raise self.cleanup_error
        return True

    def close(self) -> None:
        """Stop and wait; raise TimeoutError if a runner or its cleanup is still active.

        A later close or wait_closed can finish the wait. Timeout never reports
        successful cleanup and never makes new submissions admissible.
        """
        self.request_stop()
        if not self.wait_closed(CLOSE_GRACE_S + PROCESS_CLOSE_GRACE_S + 5.0):
            raise TimeoutError(f"background job worker {self.thread_name} is still closing")

    def run_thread(self) -> None:
        try:
            asyncio.run(self.run_jobs())
        except Exception:
            logger.exception("background job worker %s failed; queued jobs resolve as Failed", self.thread_name)
        finally:
            with self.gate:
                self.broken = True
                self.closed = True
                self.stopping.set()
            while True:
                try:
                    job = self.jobs.get_nowait()
                except queue.Empty:
                    break
                if job is not None:
                    self.results.put(Failed(job.receipt))

    async def run_jobs(self) -> None:
        semaphore = asyncio.Semaphore(self.concurrency)
        wake = asyncio.Event()
        with self.gate:
            self.loop = asyncio.get_running_loop()
            self.wake = wake
        tasks: dict[asyncio.Task[SupportsReceipt], tuple[SupportsReceipt, JobContext]] = {}

        def _complete(task: asyncio.Task[SupportsReceipt]) -> None:
            submitted, context = tasks.pop(task)
            if context.cleanup_error is not None:
                self.cleanup_error = context.cleanup_error
            try:
                result = task.result()
            except asyncio.CancelledError:
                result = Failed(submitted.receipt)
            except Exception:
                logger.exception("background job cleanup failed for receipt %s", submitted.receipt)
                result = Failed(submitted.receipt)
            self.results.put(result)

        try:
            while True:
                try:
                    job = self.jobs.get_nowait()
                except queue.Empty:
                    await wake.wait()
                    wake.clear()
                    continue
                if job is None:
                    break
                if self.closed:
                    self.results.put(Failed(job.receipt))
                    continue
                context = JobContext()
                task = asyncio.create_task(self.run_one(job, semaphore, context))
                tasks[task] = (job, context)
                task.add_done_callback(_complete)
                await asyncio.sleep(0)
            if tasks:
                await asyncio.wait(tasks, timeout=CLOSE_GRACE_S)
        finally:
            with self.gate:
                self.broken = True
                self.loop = None
                self.wake = None
            remaining = tuple(tasks)
            executions: tuple[Execution, ...] = tuple(tasks[task][1] for task in remaining)
            for execution in executions:
                execution.request_stop()
            if remaining:
                await asyncio.gather(*remaining, return_exceptions=True)

    async def run_one(
        self, job: SupportsReceipt, semaphore: asyncio.Semaphore, context: JobContext
    ) -> SupportsReceipt:
        result: SupportsReceipt = Failed(job.receipt)
        context.bind_task()
        token = CURRENT_PROCESS_SCOPE.set(context.processes)
        try:
            async with semaphore:
                if not self.closed and not context.cancelled.is_set():
                    result = await self.runner.run_in_context(job, context)
                    if result.receipt != job.receipt:
                        raise ValueError("a job result must preserve its receipt")
        except asyncio.CancelledError:
            context.request_stop()
            result = Failed(job.receipt)
        except Exception:
            result = Failed(job.receipt)
            logger.exception("background job for receipt %s failed", job.receipt)
        finally:
            try:
                await context.finish()
            finally:
                CURRENT_PROCESS_SCOPE.reset(token)
        return Failed(job.receipt) if context.cancelled.is_set() else result
