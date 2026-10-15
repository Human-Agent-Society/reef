"""A composite trainer rebuilt from durable state decides its rows as it did live.

What a trainer releases between its commits is decided in memory, around
the rows it read with them: a retry retired because its request was seen, a
step discarded once complete, an inference released with a report another
role owns. A commit retires no row, so a trainer rebuilt after a reload reads
every row again, skips the rows its own commits consumed, and decides the rest
as it did live, whatever its sibling trained meanwhile. Each test runs one
sequence with and without a reload at one point and checks that both runs
train the same and keep the same rows.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from tests.scenario.test_component_trainers import (
    HARNESS,
    WEIGHTS,
    _ComponentBackend,
    _JudgedAtOnce,
    _Judgment,
    _TwoTrainerRecipe,
)
from tests.support._threshold_processor import threshold_pipeline
from tests.support._trajectories import policy_trajectory

from recipes.tttd import TTTDBatches, TTTDGridSettings, TTTDGroupedRolloutReport, TTTDRecord2Record, TTTDTasks
from reef.artifact import ArtifactRef, InMemoryRepositoryBackend
from reef.background_jobs import SupportsReceipt
from reef.core import AgentRecord, RequestType
from reef.dispatcher import Dispatcher
from reef.pipeline_context import PipelineContext
from reef.record2record import (
    GroupDecision,
    JudgingRecord2Record,
    RecordJudging,
    ReportContext,
    ReportGrouping,
    ReportRecord2Record,
)
from reef.record2task import JudgedTasks
from reef.record2task.assembly import SampleAssemblySettings
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.task2batch import Task2Batch
from reef.train import ComponentTrainer, Trainer
from reef.train.pipeline import TrainingPipeline
from reef.train.types import TrainDataItem, TrainingBatch


class _SessionJudging(RecordJudging):
    """OpenClawRL's tagged path: a turn equal to its session's last request is a client retry and retires; the
    next turn of a session dispatches the one before it; reports retire."""

    def __init__(self) -> None:
        self.sessions: dict[str, tuple[str, int]] = {}

    def observe(self, item: AgentRecord, now: float, stage: JudgingRecord2Record) -> None:
        if item.request_type is not RequestType.INFERENCE or "session" not in item.payload:
            stage.retire(item.agent_record_id)
            return
        session, key = item.payload["session"], int(item.payload["key"])
        last = self.sessions.get(session)
        if last is not None and last[1] == key:
            stage.retire(item.agent_record_id)
            return
        if last is not None and stage.tracked_record(last[0]) is not None:
            stage.dispatch(_Judgment(last[0]))
        self.sessions[session] = (item.agent_record_id, key)
        stage.track(item)

    async def judge(self, job: SupportsReceipt) -> SupportsReceipt:
        return job


class _SessionTasks(JudgedTasks):
    """One-token stub samples; a turn marked declined is unusable."""

    def make_sample(self, record: AgentRecord, judgment: _Judgment) -> Any:
        if record.payload.get("declined"):
            return None
        return policy_trajectory(
            source_agent_record_id=record.agent_record_id,
            tokens=(1, 2),
            loss_mask=(0, 1),
            rollout_log_probs=(-0.1,),
            reward=1.0,
            runtime_load_id="v1",
        )


class SessionBatches(Task2Batch):
    """Batches the judged turns of ``_SessionJudging``."""

    def make_batch(self, tasks: tuple[TrainDataItem, ...], batch_number: int) -> TrainingBatch:
        return TrainingBatch(f"{self.scenario}:session:{batch_number}", tasks)


def session_pipeline(context: PipelineContext, batch_size: int = 1) -> TrainingPipeline:
    """Trains the judged turns of ``_SessionJudging``; a turn marked declined is unusable."""
    return TrainingPipeline(
        context,
        record2record=JudgingRecord2Record(_SessionJudging(), _JudgedAtOnce()),
        record2task=_SessionTasks(),
        task2batch=SessionBatches(context, batch_size),
    )


class _StepGrouping(ReportGrouping):
    """Reports with a ``metadata.step`` group by step and slot; a step of two trains unless its feedback is mixed."""

    def is_training_report(self, report: AgentRecord) -> bool:
        return "step" in (report.payload.get("metadata") or {})

    def placement(self, context: ReportContext) -> tuple[Hashable | None, Hashable | None]:
        metadata = context.report.payload["metadata"]
        return metadata["step"], metadata["slot"]

    def decide(self, key: Hashable, members: tuple[ReportContext, ...]) -> GroupDecision:
        if len(members) < 2:
            return GroupDecision.INCOMPLETE
        if any(member.report.payload.get("feedback") == "mixed" for member in members):
            return GroupDecision.DISCARD
        return GroupDecision.READY


def step_pipeline(context: PipelineContext, batch_size: int = 1) -> TrainingPipeline:
    """TTTD's shape: a step is one group of two (step, slot) reports, sources are exclusive, a step with a report
    marked mixed is discarded once complete, and a report that names no step is another role's and released."""
    stage = ReportRecord2Record(context, _StepGrouping(), exclusive_sources=True, ordered_groups=True)
    return threshold_pipeline(context, batch_size=batch_size, record2record=stage)


def tttd_pipeline(context: PipelineContext) -> TrainingPipeline:
    """The TTTD recipe's stages, one group of two rollouts per step."""
    record2record = TTTDRecord2Record(context, TTTDGridSettings(groups_per_step=1, rollouts_per_group=2))
    return TrainingPipeline(
        context,
        record2record=record2record,
        record2task=TTTDTasks(SampleAssemblySettings(), record2record),
        task2batch=TTTDBatches(context, record2record),
    )


@dataclass(frozen=True)
class _Recipe(_TwoTrainerRecipe):
    pipelines: tuple[tuple[str, Any], ...] = ()

    def build_trainers(self, scenario, records, *, surface, algorithm_states, experiment_logger=None):
        pipelines = dict(self.pipelines)

        def factory_for(component: str):
            build_pipeline = pipelines[component]
            if build_pipeline is tttd_pipeline:
                return tttd_pipeline
            return lambda context: build_pipeline(context, batch_size=1)

        return tuple(
            ComponentTrainer(
                component,
                Trainer.build(
                    scenario,
                    records,
                    pipeline_factory=factory_for(component),
                    candidate_backend=backend,
                    algorithm_state=algorithm_states.get(component),
                    experiment_logger=experiment_logger,
                    report_type=TTTDGroupedRolloutReport if pipelines[component] is tttd_pipeline else None,
                ),
            )
            for component, backend in self.backends.items()
        )


def _dispatcher(tmp_path: Path, pipelines: dict[str, Any]) -> Dispatcher:
    initial = tmp_path / "initial"
    for component in (WEIGHTS, HARNESS):
        (initial / component).mkdir(parents=True, exist_ok=True)
        (initial / component / f"{component}.txt").write_text(f"{component} seed", encoding="utf-8")
    backends = {component: _ComponentBackend(component, tmp_path / "candidates") for component in (WEIGHTS, HARNESS)}
    return Dispatcher(
        _Recipe(backends=backends, pipelines=tuple(pipelines.items())),
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "staged",
        scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
    )


def _turn(record_id: str, session: str, key: int, **extra: Any) -> AgentRecord:
    payload = {"tokens": [1, 2], "loss_mask": [0, 1], "rollout_log_probs": [-0.2], "session": session, "key": key}
    payload.update(extra)
    return AgentRecord.create(
        scenario="agent", request_type=RequestType.INFERENCE, payload=payload, agent_record_id=record_id
    )


def _inference(record_id: str) -> AgentRecord:
    payload = {"tokens": [1, 2], "loss_mask": [0, 1], "rollout_log_probs": [-0.2]}
    return AgentRecord.create(
        scenario="agent", request_type=RequestType.INFERENCE, payload=payload, agent_record_id=record_id
    )


def _report(record_id: str, reference: str, metadata: dict[str, Any] | None = None, feedback: str = "") -> AgentRecord:
    payload: dict[str, Any] = {"score": 1.0, "references": [reference]}
    if metadata is not None:
        payload["metadata"] = metadata
    if feedback:
        payload["feedback"] = feedback
    return AgentRecord.create(
        scenario="agent",
        request_type=RequestType.REPORT,
        payload=payload,
        agent_record_id=record_id,
        references=(reference,),
    )


def _step(scenario, component: str) -> list[tuple[str, ...]] | None:
    result = scenario.prepare_training_step(component)
    if result is None:
        return None
    batch = scenario.trainer_for(component).pending_batch
    sources = [tuple(item.source_agent_record_ids) for item in batch.items]
    scenario.commit(result, component=component)
    return sources


def _stored(scenario) -> list[str]:
    return sorted(record.agent_record_id for record in scenario.records.replay("agent"))


def _retry_after_a_sibling_commit(tmp_path: Path, reload: bool) -> tuple[Any, ...]:
    dispatcher = _dispatcher(tmp_path, {WEIGHTS: threshold_pipeline, HARNESS: session_pipeline})
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        for record in (_turn("X", "u", 1), _turn("X2", "u", 2), _turn("A", "s", 1, declined=True), _report("rA", "A")):
            scenario.records.append(record)
        assert _step(scenario, HARNESS) == [("X",)]
        for record in (_turn("A-retry", "s", 1), _turn("B", "s", 2)):
            scenario.records.append(record)
        assert _step(scenario, HARNESS) is None  # A-retry retired as a retry, A judged unusable
        assert _step(scenario, WEIGHTS) == [("A", "rA")]
        if reload:
            scenario = dispatcher._registry.reload("agent")
        return _step(scenario, HARNESS), _stored(scenario)
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_retry_a_session_processor_retired_stays_retired_after_a_reload(tmp_path: Path) -> None:
    """The harness retires A-retry as a retry of A and judges A unusable, with no commit to record either; the
    weights trainer then trains A. Rebuilt from its last commit, the harness reads A-retry again: it trains nothing,
    as live, and never trains the client's retry."""
    live = _retry_after_a_sibling_commit(tmp_path / "live", reload=False)
    rebuilt = _retry_after_a_sibling_commit(tmp_path / "rebuilt", reload=True)
    assert live[0] is None and rebuilt == live


def _discarded_step(tmp_path: Path, reload: bool) -> tuple[Any, ...]:
    dispatcher = _dispatcher(tmp_path, {WEIGHTS: threshold_pipeline, HARNESS: step_pipeline})
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        for record in (
            _inference("iA"),
            _report("rA", "iA", {"step": 9, "slot": 0}),
            _inference("iB"),
            _report("rB", "iB", {"step": 9, "slot": 1}),
            _inference("i1"),
            _report("r1", "i1", {"step": 0, "slot": 0}),
            _inference("i2"),
            _report("r2", "i2", {"step": 0, "slot": 1}, feedback="mixed"),
        ):
            scenario.records.append(record)
        assert _step(scenario, HARNESS) == [("iA", "rA"), ("iB", "rB")]
        assert _step(scenario, HARNESS) is None  # step 0 discarded
        for _ in range(3):
            _step(scenario, WEIGHTS)
        if reload:
            scenario = dispatcher._registry.reload("agent")
        assert _step(scenario, HARNESS) is None
        trained = _step(scenario, WEIGHTS)
        metrics = scenario.trainer_for(HARNESS).pipeline.operational_metrics()
        return trained, metrics["unreserved_reports"], _stored(scenario)
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_step_a_reported_processor_discarded_stays_discarded_after_a_reload(tmp_path: Path) -> None:
    """The harness discards step 0 and releases its rows with no commit to record it; the weights trainer trains
    r1. Rebuilt, the harness discards step 0 again as it was: no report waits in a step that can never complete,
    and every row stays stored."""
    live = _discarded_step(tmp_path / "live", reload=False)
    rebuilt = _discarded_step(tmp_path / "rebuilt", reload=True)
    stored = sorted(["iA", "rA", "iB", "rB", "i1", "r1", "i2", "r2"])
    assert live == ([("i2", "r2")], 0, stored) and rebuilt == live


def _another_roles_report(tmp_path: Path, reload: bool) -> tuple[Any, ...]:
    dispatcher = _dispatcher(tmp_path, {WEIGHTS: session_pipeline, HARNESS: step_pipeline})
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        for record in (
            _inference("iA"),
            _report("rA", "iA", {"step": 9, "slot": 0}),
            _inference("iB"),
            _report("rB", "iB", {"step": 9, "slot": 1}),
            _turn("i3", "s", 1),
            _report("r3", "i3"),
            _turn("X", "u", 1),
            _turn("X2", "u", 2),
        ):
            scenario.records.append(record)
        assert _step(scenario, HARNESS) == [("iA", "rA"), ("iB", "rB")]
        assert _step(scenario, HARNESS) is None  # r3 is another role's: released with i3
        assert _step(scenario, WEIGHTS) == [("X",)]
        if reload:
            scenario = dispatcher._registry.reload("agent")
        assert _step(scenario, HARNESS) is None
        scenario.records.append(_turn("i3b", "s", 2))
        trained = _step(scenario, WEIGHTS)
        released = scenario.trainer_for(HARNESS).pipeline.releasable_record_ids()
        return trained, "i3" in released, _stored(scenario)
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_an_inference_released_with_another_roles_report_stays_released_after_a_reload(tmp_path: Path) -> None:
    """The harness releases r3 and i3 with it (exclusive sources, a report of another role) with no commit to
    record it. Rebuilt, the harness does not read i3 as an inference waiting for a report: it releases it again,
    as live."""
    live = _another_roles_report(tmp_path / "live", reload=False)
    rebuilt = _another_roles_report(tmp_path / "rebuilt", reload=True)
    assert live[0] == [("i3",)] and live[1] is True and rebuilt == live


def _retry_at_a_discarded_step(tmp_path: Path, point: str, reload: bool) -> tuple[Any, ...]:
    dispatcher = _dispatcher(tmp_path, {WEIGHTS: threshold_pipeline, HARNESS: step_pipeline})
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        records = [
            _inference("i1"),
            _report("r1", "i1", {"step": 0, "slot": 0}),
            _inference("i2"),
            _report("r2", "i2", {"step": 0, "slot": 1}, feedback="mixed"),
        ]
        for record in records:
            scenario.records.append(record)
        first = _step(scenario, HARNESS)
        if point == "trained":
            # The weights trainer trains the discarded rows before the rebuild.
            for _ in range(3):
                _step(scenario, WEIGHTS)
        if reload:
            scenario = dispatcher._registry.reload("agent")
        # The client retries step 0 at both slots, clean this time.
        for record in (
            _inference("iX"),
            _report("rX", "iX", {"step": 0, "slot": 0}),
            _inference("iY"),
            _report("rY", "iY", {"step": 0, "slot": 1}),
        ):
            scenario.records.append(record)
        return first, _step(scenario, HARNESS)
    finally:
        dispatcher.close()


@pytest.mark.unit
@pytest.mark.parametrize("point", ["settled", "trained"])
def test_a_retry_at_a_step_a_reported_processor_discarded_stays_terminal_after_a_reload(
    tmp_path: Path, point: str
) -> None:
    """Live, every later report of a discarded group key is terminal. A rebuilt harness reads the discarded step's
    rows again, whether or not the weights trainer trained them, finds step 0 discarded, and never trains the
    client's retry of it."""
    live = _retry_at_a_discarded_step(tmp_path / "live", point, reload=False)
    rebuilt = _retry_at_a_discarded_step(tmp_path / "rebuilt", point, reload=True)
    assert live[1] is None and rebuilt == live


def _tttd_inference(record_id: str, release: str) -> AgentRecord:
    return AgentRecord.create(
        scenario="agent",
        request_type=RequestType.INFERENCE,
        agent_record_id=record_id,
        artifact_ref=ArtifactRef(content_id=f"c-{release}", release_id=release, parent_release_id=None),
        payload={"tokens": [1, 2], "loss_mask": [0, 1], "rollout_log_probs": [-0.2]},
    )


def _tttd_report(record_id: str, reference: str, rollout: int, score: float) -> AgentRecord:
    metadata = {
        "algorithm": "ttt-discover",
        "comparison_set": "tttd-step-0-group-0",
        "step": 0,
        "group": 0,
        "rollout": rollout,
        "groups_per_step": 1,
        "rollouts_per_group": 2,
    }
    return AgentRecord.create(
        scenario="agent",
        request_type=RequestType.REPORT,
        agent_record_id=record_id,
        references=(reference,),
        payload={"score": score, "references": [reference], "metadata": metadata},
    )


def _tttd_failed_step(tmp_path: Path, reload: bool) -> tuple[Any, ...]:
    dispatcher = _dispatcher(tmp_path, {WEIGHTS: threshold_pipeline, HARNESS: tttd_pipeline})
    try:
        scenario = dispatcher.get_or_create_scenario("agent")
        # Step 0 spans two releases: TTTD fails it.
        for record in (
            _tttd_inference("i0", "rel-1"),
            _tttd_report("r0", "i0", 0, 1.0),
            _tttd_inference("i1", "rel-2"),
            _tttd_report("r1", "i1", 1, 0.0),
        ):
            scenario.records.append(record)
        first = _step(scenario, HARNESS)
        if reload:
            scenario = dispatcher._registry.reload("agent")
        # The controller samples step 0 again, on one release.
        for record in (
            _tttd_inference("j0", "rel-2"),
            _tttd_report("s0", "j0", 0, 1.0),
            _tttd_inference("j1", "rel-2"),
            _tttd_report("s1", "j1", 1, 0.0),
        ):
            scenario.records.append(record)
        resampled = _step(scenario, HARNESS)
        return first, scenario.trainer_for(HARNESS).pipeline_status()["failed_steps"], resampled
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_tttd_step_failed_on_two_releases_stays_failed_after_a_reload(tmp_path: Path) -> None:
    """A rebuilt TTTD harness reads step 0 again, reports it failed with its releases, and never trains a resample
    of it."""
    live = _tttd_failed_step(tmp_path / "live", reload=False)
    rebuilt = _tttd_failed_step(tmp_path / "rebuilt", reload=True)
    assert live[1] == [{"step": 0, "reason": "mixed_release_ids", "release_ids": ["rel-1", "rel-2"]}]
    assert live[2] is None and rebuilt == live
