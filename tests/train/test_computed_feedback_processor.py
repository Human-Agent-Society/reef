"""Focused unit tests for the computed-feedback pipeline's state machine.

tests.recipes.openclawrl.test_openclawrl_contracts pins the real recipe's behavior; these tests pin the
mechanics of a pipeline with a ``JudgedSamples`` stage of a
toy recipe, where recipe policy cannot blur the state machine.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from tests.support._trajectories import policy_trajectory

from reef.core import AgentRecord, RequestType
from reef.core.trajectories import source_record_id
from reef.execution.jobs import BackgroundJobRunner, BackgroundJobWorker, Failed, JobContext, SupportsReceipt
from reef.execution.sandbox.process import LocalExecutor, ProgramLaunchError
from reef.pipeline_context import PipelineContext
from reef.record2sample import JudgedSamples, RecordJudging
from reef.sample2job import Sample2Job
from reef.train.pipeline import TrainingPipeline
from reef.train.types import JobData, Sample, TrajectorySample

pytestmark = pytest.mark.unit


@dataclass(frozen=True)
class _Job(SupportsReceipt):
    receipt: str


@dataclass(frozen=True)
class _Judgment(SupportsReceipt):
    receipt: str
    good: bool = True


class _FakeWorker:
    def __init__(self) -> None:
        self.jobs: list[_Job] = []
        self.closed = False
        self._judgments: list[object] = []

    def submit(self, job) -> bool:
        self.jobs.append(job)
        return True

    def poll(self):
        judgments, self._judgments = self._judgments, []
        return judgments

    def close(self) -> None:
        self.closed = True

    def push(self, judgment) -> None:
        self._judgments.append(judgment)


class _ToyJudging(RecordJudging):
    """A record tracks itself; a record whose payload names ``completes`` finishes that receipt."""

    def observe(self, item: AgentRecord, now: float, stage: JudgedSamples) -> None:
        completes = item.payload.get("completes")
        if completes:
            stage.dispatch(_Job(completes))
        if item.payload.get("track"):
            stage.track(item)
        else:
            stage.retire(item.agent_record_id)

    async def judge(self, job: SupportsReceipt) -> SupportsReceipt:
        return _Judgment(job.receipt)


class _ToySamples(JudgedSamples):
    """One-token stub samples; a judgment that is not good cannot train."""

    def make_sample(self, record: AgentRecord, judgment: _Judgment) -> TrajectorySample | None:
        if not judgment.good:
            return None
        return policy_trajectory(
            source_agent_record_id=record.agent_record_id,
            tokens=(1, 2),
            loss_mask=(1,),
            rollout_log_probs=(-0.1,),
            reward=1.0,
            artifact_id=record.payload.get("version", "v1"),
        )


class ToyJobData(Sample2Job):
    """Batches the judged samples in the order that the stage selected them."""

    def make_job_data(self, samples: tuple[Sample, ...], data_id: str) -> JobData:
        return JobData(data_id, samples)


class _ToyProcessor(TrainingPipeline):
    """Trains the judged records of ``_ToyJudging``. Samples are one-token stubs."""

    def __init__(self, context: PipelineContext, *, worker: Any, max_staleness: int = 0) -> None:
        super().__init__(
            context,
            record2sample=_ToySamples(_ToyJudging(), worker, max_staleness=max_staleness),
            sample2job=ToyJobData(context, None, None),
        )


def _record(agent_record_id: str, **payload) -> AgentRecord:
    return AgentRecord.create(
        scenario="s", request_type=RequestType.INFERENCE, payload=payload, agent_record_id=agent_record_id
    )


def _engine(worker=None) -> tuple[_ToyProcessor, _FakeWorker]:
    worker = worker if worker is not None else _FakeWorker()
    processor = _ToyProcessor(PipelineContext("s"), worker=worker)
    return processor, worker


def test_track_complete_judge_batch_acknowledge() -> None:
    processor, worker = _engine()
    processor.push(_record("r1", track=True))
    assert not processor.background_jobs_running()
    processor.push(_record("r2", completes="r1", track=True))
    assert [job.receipt for job in worker.jobs] == ["r1"]
    assert processor.background_jobs_running()

    worker.push(_Judgment("r1"))
    processor.produce()
    assert processor.ready(1)
    batch = processor.build_job_data(1)
    assert source_record_id(batch.samples[0]) == "r1"
    processor.acknowledge(batch.data_id)
    decision = processor.releasable_record_ids()
    assert "r1" in decision
    assert "r2" not in decision  # still tracked


def test_untracked_records_are_terminal_on_sight() -> None:
    processor, worker = _engine()
    processor.push(_record("noise"))
    assert "noise" in processor.releasable_record_ids()
    assert not worker.jobs


def test_failed_and_declined_judgments_retire_the_record() -> None:
    processor, worker = _engine()
    processor.push(_record("a", track=True))
    processor.push(_record("b", track=True))
    processor.push(_record("done", completes="a"))
    processor.push(_record("done2", completes="b"))
    worker.push(Failed("a"))
    worker.push(_Judgment("b", good=False))
    processor.produce()
    assert not processor.ready(1)
    releasable = processor.releasable_record_ids()
    assert {"a", "b"} <= set(releasable)


def test_refused_submission_retires_immediately() -> None:
    class _Refusing(_FakeWorker):
        def submit(self, job) -> bool:
            return False

    processor, _ = _engine(worker=_Refusing())
    processor.push(_record("r1", track=True))
    processor.push(_record("r2", completes="r1"))
    assert "r1" in processor.releasable_record_ids()


def test_judged_samples_behind_the_serving_version_leave_the_buffer_before_a_batch() -> None:
    processor, worker = _engine()
    processor.observe_serving_artifact_id("v1")
    processor.push(_record("old", track=True, version="v1"))
    processor.push(_record("done1", completes="old"))
    processor.observe_serving_artifact_id("v2")
    processor.push(_record("new", track=True, version="v2"))
    processor.push(_record("done2", completes="new"))
    worker.push(_Judgment("new"))
    worker.push(_Judgment("old"))
    processor.produce()
    assert not processor.ready(2)  # the stale sample is not in the buffer
    assert {"old", "new"} <= processor.releasable_record_ids()
    assert [source_record_id(sample) for sample in processor.build_job_data(1).samples] == ["new"]


def test_judged_sets_within_max_staleness_of_the_serving_version_stay() -> None:
    worker = _FakeWorker()
    processor = _ToyProcessor(PipelineContext("s"), worker=worker, max_staleness=1)
    for name, version in (("v3", "e:3"), ("v4", "e:4"), ("v5", "e:5")):
        processor.observe_serving_artifact_id(version)
        processor.push(_record(name, track=True, version=version))
        processor.push(_record(f"done-{name}", completes=name))
        worker.push(_Judgment(name))
    processor.produce()
    assert not processor.ready(3)  # e:3 is two artifacts behind e:5
    assert {"v3", "v4", "v5"} <= processor.releasable_record_ids()
    assert [source_record_id(sample) for sample in processor.build_job_data(2).samples] == ["v4", "v5"]


def test_correlate_only_mode_without_a_worker() -> None:
    processor = _ToyProcessor(PipelineContext("s"), worker=None)
    processor.push(_record("r1", track=True))
    processor.push(_record("r2", completes="r1"))
    assert "r1" in processor.releasable_record_ids()
    processor.close()


def test_close_propagates_to_the_worker() -> None:
    processor, worker = _engine()
    processor.close()
    assert worker.closed


class _Answers(BackgroundJobRunner):
    async def run(self, job: SupportsReceipt) -> SupportsReceipt:
        return _Judgment(job.receipt)


class _Crashes(BackgroundJobRunner):
    async def run(self, job: SupportsReceipt) -> SupportsReceipt:
        raise RuntimeError("boom")


def test_real_worker_runs_the_judge_coroutine() -> None:
    worker = BackgroundJobWorker(_Answers(), thread_name="test-jobs")
    assert worker.submit(_Job("r1"))

    deadline = time.monotonic() + 10.0
    judgments: list = []
    while not judgments and time.monotonic() < deadline:
        judgments = worker.poll()
        time.sleep(0.01)
    assert [judgment.receipt for judgment in judgments] == ["r1"]
    worker.close()


def test_real_worker_resolves_a_crashing_judge_as_failed() -> None:
    worker = BackgroundJobWorker(_Crashes(), thread_name="test-jobs")
    assert worker.submit(_Job("r1"))

    deadline = time.monotonic() + 10.0
    judgments: list = []
    while not judgments and time.monotonic() < deadline:
        judgments = worker.poll()
        time.sleep(0.01)
    assert judgments == [Failed("r1")]
    worker.close()


class BlockingJobRunner(BackgroundJobRunner):
    """A runner with a cooperative or deliberately delayed cancellation path."""

    def __init__(self, *, defer_cancellation: bool = False) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.cancelled = threading.Event()
        self.defer_cancellation = defer_cancellation
        self.calls: list[str] = []

    async def run(self, job: SupportsReceipt) -> SupportsReceipt:
        self.calls.append(job.receipt)
        self.started.set()
        try:
            while not self.release.is_set():
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            self.cancelled.set()
            if self.defer_cancellation:
                while not self.release.is_set():
                    await asyncio.sleep(0.01)
            raise
        return SupportsReceipt(job.receipt)


def test_worker_cancels_queued_jobs_without_running_them(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("reef.execution.jobs.CLOSE_GRACE_S", 0.01)
    runner = BlockingJobRunner()
    worker = BackgroundJobWorker(runner, thread_name="queue-close", concurrency=1)
    assert worker.submit(SupportsReceipt("running"))
    assert runner.started.wait(5)
    assert worker.submit(SupportsReceipt("queued"))
    worker.close()
    assert runner.calls == ["running"]
    results = worker.poll()
    assert sorted(result.receipt for result in results) == ["queued", "running"]
    assert all(isinstance(result, Failed) for result in results)
    assert worker.poll() == []
    assert worker.wait_closed(0)


def test_worker_reports_incomplete_close_until_runner_finishes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("reef.execution.jobs.CLOSE_GRACE_S", 0.01)
    runner = BlockingJobRunner(defer_cancellation=True)
    worker = BackgroundJobWorker(runner, thread_name="deferred-close")
    worker.submit(SupportsReceipt("job"))
    assert runner.started.wait(5)
    try:
        worker.request_stop()
        assert runner.cancelled.wait(5)
        assert not worker.wait_closed(0.01)
        assert worker.poll() == []
        assert not worker.submit(SupportsReceipt("late"))
    finally:
        runner.release.set()
        worker.close()
    assert worker.poll() == [Failed("job")]
    assert worker.wait_closed(0)


class ProcessJobRunner(BackgroundJobRunner):
    """Run a local process through the executor adapter inside a worker job."""

    def __init__(self, root: Path) -> None:
        self.root = root

    async def run(self, job: SupportsReceipt) -> SupportsReceipt:
        pending = asyncio.create_task(
            asyncio.to_thread(
                LocalExecutor().launch,
                [
                    sys.executable,
                    "-c",
                    "import os,pathlib,time; pathlib.Path('pid').write_text(str(os.getpid())); time.sleep(60)",
                ],
                root=self.root,
                workspace=self.root,
                env={},
                timeout=60,
            )
        )
        try:
            await asyncio.shield(pending)
        except asyncio.CancelledError:
            with contextlib.suppress(ProgramLaunchError):
                await pending
            raise
        return SupportsReceipt(job.receipt)


def test_worker_reaps_executor_process_before_publishing_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("reef.execution.jobs.CLOSE_GRACE_S", 0.01)
    worker = BackgroundJobWorker(ProcessJobRunner(tmp_path), thread_name="process-close")
    worker.submit(SupportsReceipt("job"))
    deadline = time.monotonic() + 5
    while not (tmp_path / "pid").exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    try:
        assert (tmp_path / "pid").exists()
        pid = int((tmp_path / "pid").read_text())
    finally:
        worker.close()
    assert worker.poll() == [Failed("job")]
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


class SaturatedJobRunner(BackgroundJobRunner):
    """A one-thread executor held by a job until cancellation releases it."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    async def run(self, job: SupportsReceipt) -> SupportsReceipt:
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
        pending = asyncio.create_task(asyncio.to_thread(self.release.wait))
        self.started.set()
        try:
            await asyncio.shield(pending)
        finally:
            self.release.set()
            await pending
        return SupportsReceipt(job.receipt)


def test_worker_shutdown_does_not_need_an_available_executor_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("reef.execution.jobs.CLOSE_GRACE_S", 0.01)
    runner = SaturatedJobRunner()
    worker = BackgroundJobWorker(runner, thread_name="saturated-worker", concurrency=1)
    worker.submit(SupportsReceipt("job"))
    assert runner.started.wait(5)
    try:
        worker.request_stop()
        assert worker.wait_closed(3)
        assert worker.poll() == [Failed("job")]
    finally:
        runner.release.set()
        worker.close()


def test_repeated_receipts_are_distinct_submissions() -> None:
    worker = BackgroundJobWorker(_Answers(), thread_name="repeated-receipts")
    assert worker.submit(_Job("same"))
    assert worker.submit(_Job("same"))
    results: list[SupportsReceipt] = []
    deadline = time.monotonic() + 5
    try:
        while len(results) < 2 and time.monotonic() < deadline:
            results.extend(worker.poll())
            time.sleep(0.01)
        assert results == [_Judgment("same"), _Judgment("same")]
        assert worker.poll() == []
    finally:
        worker.close()


def test_worker_lifecycle_wait_can_start_before_stop() -> None:
    from reef.execution.lifecycle import Execution

    worker = BackgroundJobWorker(_Answers(), thread_name="lifecycle-wait")
    lifecycle: Execution = worker
    assert not lifecycle.wait_closed(0)
    with ThreadPoolExecutor() as pool:
        waiting = pool.submit(lifecycle.wait_closed, None)
        lifecycle.request_stop()
        lifecycle.request_stop()
        assert waiting.result(timeout=5)
    assert not worker.submit(SupportsReceipt("closed"))


class ContextJobRunner(BlockingJobRunner):
    """Expose a running job's context to test cancellation through Execution."""

    def __init__(self) -> None:
        super().__init__()
        self.context: JobContext | None = None

    async def run_in_context(self, job: SupportsReceipt, context: JobContext) -> SupportsReceipt:
        self.context = context
        return await self.run(job)


def test_job_execution_can_stop_without_stopping_its_worker() -> None:
    from reef.execution.lifecycle import Execution

    runner = ContextJobRunner()
    worker = BackgroundJobWorker(runner, thread_name="individual-job")
    try:
        assert worker.submit(SupportsReceipt("first"))
        assert runner.started.wait(5)
        assert runner.context is not None
        execution: Execution = runner.context
        execution.request_stop()
        assert execution.wait_closed(5)
        assert runner.cancelled.is_set()
        assert not worker.closed
        runner.release.set()
        assert worker.submit(SupportsReceipt("second"))
        deadline = time.monotonic() + 5
        while "second" not in runner.calls and time.monotonic() < deadline:
            time.sleep(0.01)
        assert "second" in runner.calls
    finally:
        worker.close()
    results = worker.poll()
    assert Failed("first") in results
    assert SupportsReceipt("second") in results
