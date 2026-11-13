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

from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, Self

import pytest
from tests.support.component_trainers import (
    HARNESS,
    MAX_STALENESS,
    WEIGHTS,
    JudgedAtOnce,
    Judgment,
    TextHarnessRecipe,
    WeightsRecipe,
    _dispatcher,
    _scenario,
    checked_commit,
    component_stages,
    finish_job,
    harness_recipe,
    seeded_start,
    two_trainer_recipe,
)
from tests.support.records import report, token_inference
from tests.support.samples_service import make_samples
from tests.support.serving import wait_for_serving
from tests.support.threshold_processor import ThresholdJobData, ThresholdSamples
from tests.support.trajectories import policy_trajectory

from recipes.tttd import TTTDGroupedRolloutReport, TTTDJobData, TTTDSamples
from reef.core import AgentRecord, RequestType
from reef.core.reports import ReportBase
from reef.dispatcher import Dispatcher
from reef.evaluation.harness import EvaluationSettings
from reef.execution.jobs import SupportsReceipt
from reef.execution.workers import SERVICE_COMPONENTS
from reef.pipeline_context import PipelineContext
from reef.record2sample import GroupDecision, JudgedSamples, RecordJudging, ReportContext, ReportGrouping
from reef.record2sample.assembly import SampleAssemblySettings
from reef.scenario import Scenario
from reef.train.harness_backend.settings import EvolutionSettings
from reef.train.pipeline import PipelineSettings
from reef.train.types import JobData, Sample


class _SessionJudging(RecordJudging):
    """OpenClawRL's tagged path: a turn equal to its session's last request is a client retry and retires; the
    next turn of a session dispatches the one before it; reports retire."""

    def __init__(self) -> None:
        self.sessions: dict[str, tuple[str, int]] = {}

    def observe(self, item: AgentRecord, now: float, stage: JudgedSamples) -> None:
        if item.request_type is not RequestType.INFERENCE or "session" not in item.payload:
            stage.retire(item.agent_record_id)
            return
        session, key = item.payload["session"], int(item.payload["key"])
        last = self.sessions.get(session)
        if last is not None and last[1] == key:
            stage.retire(item.agent_record_id)
            return
        if last is not None and stage.tracked_record(last[0]) is not None:
            stage.dispatch(Judgment(last[0]))
        self.sessions[session] = (item.agent_record_id, key)
        stage.track(item)

    async def judge(self, job: SupportsReceipt) -> SupportsReceipt:
        return job


class _SessionSamples(JudgedSamples):
    """One-token stub samples for the judged turns of a ``_SessionJudging``, each judged at the next poll; a turn
    marked declined is unusable."""

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, context: PipelineContext, environ: Mapping[str, str]) -> Self:
        return cls(_SessionJudging(), JudgedAtOnce(), max_staleness=MAX_STALENESS)

    def make_sample(self, record: AgentRecord, judgment: Judgment) -> Any:
        if record.payload.get("declined"):
            return None
        return policy_trajectory(
            source_agent_record_id=record.agent_record_id,
            tokens=(1, 2),
            loss_mask=(0, 1),
            rollout_log_probs=(-0.1,),
            reward=1.0,
            artifact_id=record.artifact_id,
        )


class SessionJobData(ThresholdJobData):
    """Batches the judged turns of ``_SessionJudging``."""

    def make_job_data(self, samples: tuple[Sample, ...], data_id: str) -> JobData:
        return JobData(data_id, samples)


def session_stages() -> PipelineSettings:
    """Trains the judged turns of ``_SessionJudging``; a turn marked declined is unusable."""
    return component_stages(_SessionSamples, SessionJobData, record2sample_config={})


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


class _StepSamples(ThresholdSamples):
    """``ThresholdSamples`` in ``_StepGrouping``'s groups, with exclusive sources."""

    def __init__(self, context: PipelineContext, settings: SampleAssemblySettings | None = None) -> None:
        super().__init__(context, settings, grouping=_StepGrouping())
        self.exclusive_sources = True


def step_stages() -> PipelineSettings:
    """TTTD's shape: a step is one group of two (step, slot) reports, sources are exclusive, a step with a report
    marked mixed is discarded once complete, and a report that names no step is another role's and released."""
    return component_stages(record2sample=_StepSamples)


def tttd_stages() -> PipelineSettings:
    """The TTTD recipe's stages, one group of two rollouts per step."""
    config = {"groups_per_step": 1, "rollouts_per_group": 2, "max_staleness": MAX_STALENESS}
    return component_stages(TTTDSamples, TTTDJobData, record2sample_config=config)


@dataclass(frozen=True)
class _TTTDHarnessRecipe(TextHarnessRecipe):
    """The text harness recipe whose trainer parses each report as a TTTD rollout report."""

    @property
    def report_type(self) -> type[ReportBase]:  # type: ignore[override]
        return TTTDGroupedRolloutReport


def composite_dispatcher(
    tmp_path: Path, *, weights: PipelineSettings | None = None, harness: TextHarnessRecipe | None = None
) -> Dispatcher:
    """A dispatcher over the two trainer recipe whose weights trainer runs the stages ``weights`` and whose harness
    recipe is ``harness``; each default runs the threshold stages. Each trainer reads the stored samples of its
    component, so a rebuilt trainer finds the samples made live."""
    recipe = two_trainer_recipe(weights=WeightsRecipe(pipeline_stages=weights), harness=harness)
    dispatcher, _ = _dispatcher(
        tmp_path,
        recipe=recipe,
        # Loading a scenario wakes its workers: held, they leave every step to the test.
        hold_local_cycles=True,
        # No samples worker: only the test makes samples, in ``step``.
        components=frozenset(SERVICE_COMPONENTS) - {"samples"},
    )
    return dispatcher


def opened_scenario(dispatcher: Dispatcher) -> Scenario:
    """Scenario ``agent``, created, once its start artifact serves: an automatic job waits for it."""
    scenario = _scenario(dispatcher)
    wait_for_serving(dispatcher, "agent")
    return scenario


def turn(record_id: str, session: str, key: int, **extra: Any) -> AgentRecord:
    return token_inference(record_id, scenario="agent", session=session, key=key, **extra)


agent_inference = partial(token_inference, scenario="agent")
agent_report = partial(report, scenario="agent")


def step(scenario, component: str) -> list[tuple[str, ...]] | None:
    # The samples service makes the samples of the rows appended since the last step, as its worker would.
    make_samples(scenario)
    result = checked_commit(scenario, component)
    if result is None:
        return None
    batch = scenario.trainer_for(component).pending_job_data
    sources = [tuple(item.source_agent_record_ids) for item in batch.samples]
    scenario.commit(result, component=component)
    finish_job(scenario, component)
    return sources


def append_records(scenario: Scenario, records: Iterable[AgentRecord]) -> None:
    """Append ``records``. An inference that names no artifact is answered by the artifact that ``scenario`` serves
    now, so its sample is not stale."""
    for record in records:
        if record.request_type is RequestType.INFERENCE and record.artifact_id is None:
            record = replace(record, artifact_id=scenario.serving_artifact_id())
        scenario.records.append(record)


def step_nine() -> tuple[AgentRecord, ...]:
    """The two reported rows of step 9, a complete group that trains."""
    return (
        agent_inference("iA"),
        agent_report("rA", "iA", metadata={"step": 9, "slot": 0}),
        agent_inference("iB"),
        agent_report("rB", "iB", metadata={"step": 9, "slot": 1}),
    )


def stored_ids(scenario) -> list[str]:
    return sorted(record.agent_record_id for record in scenario.records.replay("agent"))


def retry_after_a_sibling_commit(tmp_path: Path, reload: bool) -> tuple[Any, ...]:
    dispatcher = composite_dispatcher(tmp_path, harness=harness_recipe(session_stages()))
    try:
        scenario = opened_scenario(dispatcher)
        append_records(
            scenario,
            (turn("X", "u", 1), turn("X2", "u", 2), turn("A", "s", 1, declined=True), agent_report("rA", "A")),
        )
        assert step(scenario, HARNESS) == [("X",)]
        append_records(scenario, (turn("A-retry", "s", 1), turn("B", "s", 2)))
        assert step(scenario, HARNESS) is None  # A-retry retired as a retry, A judged unusable
        assert step(scenario, WEIGHTS) == [("A", "rA")]
        if reload:
            scenario = dispatcher.registry.reload("agent")
        return step(scenario, HARNESS), stored_ids(scenario)
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_retry_a_session_processor_retired_stays_retired_after_a_reload(tmp_path: Path) -> None:
    """The harness retires A-retry as a retry of A and judges A unusable, with no commit to record either; the
    weights trainer then trains A. Rebuilt from its last commit, the harness reads A-retry again: it trains nothing,
    as live, and never trains the client's retry."""
    live = retry_after_a_sibling_commit(tmp_path / "live", reload=False)
    rebuilt = retry_after_a_sibling_commit(tmp_path / "rebuilt", reload=True)
    assert live[0] is None and rebuilt == live


def discarded_step(tmp_path: Path, reload: bool) -> tuple[Any, ...]:
    dispatcher = composite_dispatcher(tmp_path, harness=harness_recipe(step_stages()))
    try:
        scenario = opened_scenario(dispatcher)
        append_records(scenario, step_nine())
        assert step(scenario, HARNESS) == [("iA", "rA"), ("iB", "rB")]
        # The rows of step 0 arrive after the first commit, so no commit records their discard.
        append_records(
            scenario,
            (
                agent_inference("i1"),
                agent_report("r1", "i1", metadata={"step": 0, "slot": 0}),
                agent_inference("i2"),
                agent_report("r2", "i2", metadata={"step": 0, "slot": 1}, feedback="mixed"),
            ),
        )
        assert step(scenario, HARNESS) is None  # step 0 discarded
        for _ in range(3):
            step(scenario, WEIGHTS)
        if reload:
            scenario = dispatcher.registry.reload("agent")
        assert step(scenario, HARNESS) is None
        trained = step(scenario, WEIGHTS)
        metrics = scenario.trainer_for(HARNESS).pipeline.operational_metrics()
        return trained, metrics["waiting_reports"], stored_ids(scenario)
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_step_a_reported_processor_discarded_stays_discarded_after_a_reload(tmp_path: Path) -> None:
    """The harness discards step 0 and releases its rows with no commit to record it; the weights trainer trains
    r1. Rebuilt, the harness discards step 0 again as it was: no report waits in a step that can never complete,
    and every row stays stored."""
    live = discarded_step(tmp_path / "live", reload=False)
    rebuilt = discarded_step(tmp_path / "rebuilt", reload=True)
    stored = sorted(["iA", "rA", "iB", "rB", "i1", "r1", "i2", "r2"])
    assert live == ([("i2", "r2")], 0, stored) and rebuilt == live


def another_roles_report(tmp_path: Path, reload: bool) -> tuple[Any, ...]:
    dispatcher = composite_dispatcher(tmp_path, weights=session_stages(), harness=harness_recipe(step_stages()))
    try:
        scenario = opened_scenario(dispatcher)
        append_records(scenario, step_nine())
        assert step(scenario, HARNESS) == [("iA", "rA"), ("iB", "rB")]
        # These rows arrive after the first commit, so no commit records the snapshot of r3 and i3.
        append_records(scenario, (turn("i3", "s", 1), agent_report("r3", "i3"), turn("X", "u", 1), turn("X2", "u", 2)))
        assert step(scenario, HARNESS) is None  # r3 is another role's: released with i3
        assert step(scenario, WEIGHTS) == [("X",)]
        if reload:
            scenario = dispatcher.registry.reload("agent")
        assert step(scenario, HARNESS) is None
        append_records(scenario, (turn("i3b", "s", 2),))
        trained = step(scenario, WEIGHTS)
        released = scenario.trainer_for(HARNESS).pipeline.record2sample.releasable_record_ids()
        return trained, "i3" in released, stored_ids(scenario)
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_an_inference_released_with_another_roles_report_stays_released_after_a_reload(tmp_path: Path) -> None:
    """The harness artifacts r3 and i3 with it (exclusive sources, a report of another role) with no commit to
    record it. Rebuilt, the harness does not read i3 as an inference waiting for a report: it releases it again,
    as live."""
    live = another_roles_report(tmp_path / "live", reload=False)
    rebuilt = another_roles_report(tmp_path / "rebuilt", reload=True)
    assert live[0] == [("i3",)] and live[1] is True and rebuilt == live


def retry_at_a_discarded_step(tmp_path: Path, point: str, reload: bool) -> tuple[Any, ...]:
    dispatcher = composite_dispatcher(tmp_path, harness=harness_recipe(step_stages()))
    try:
        scenario = opened_scenario(dispatcher)
        records = [
            agent_inference("i1"),
            agent_report("r1", "i1", metadata={"step": 0, "slot": 0}),
            agent_inference("i2"),
            agent_report("r2", "i2", metadata={"step": 0, "slot": 1}, feedback="mixed"),
        ]
        append_records(scenario, records)
        first = step(scenario, HARNESS)
        if point == "trained":
            # The weights trainer trains the discarded rows before the rebuild.
            for _ in range(3):
                step(scenario, WEIGHTS)
        if reload:
            scenario = dispatcher.registry.reload("agent")
        # The client retries step 0 at both slots, clean this time.
        append_records(
            scenario,
            (
                agent_inference("iX"),
                agent_report("rX", "iX", metadata={"step": 0, "slot": 0}),
                agent_inference("iY"),
                agent_report("rY", "iY", metadata={"step": 0, "slot": 1}),
            ),
        )
        return first, step(scenario, HARNESS)
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
    live = retry_at_a_discarded_step(tmp_path / "live", point, reload=False)
    rebuilt = retry_at_a_discarded_step(tmp_path / "rebuilt", point, reload=True)
    assert live[1] is None and rebuilt == live


def tttd_inference(record_id: str, artifact_id: str | None = None) -> AgentRecord:
    return AgentRecord.create(
        scenario="agent",
        request_type=RequestType.INFERENCE,
        agent_record_id=record_id,
        artifact_id=artifact_id,
        payload={"tokens": [1, 2], "loss_mask": [0, 1], "rollout_log_probs": [-0.2]},
    )


def tttd_report(record_id: str, reference: str, rollout: int, score: float) -> AgentRecord:
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


def tttd_failed_step(tmp_path: Path, reload: bool) -> tuple[Any, ...]:
    harness = _TTTDHarnessRecipe(
        evolution=EvolutionSettings(),
        evaluation=EvaluationSettings(policy="none"),
        pipeline_stages=tttd_stages(),
        start=seeded_start(HARNESS),
    )
    dispatcher = composite_dispatcher(tmp_path, harness=harness)
    try:
        scenario = opened_scenario(dispatcher)
        served = scenario.serving_artifact_id()
        # Step 0 spans two snapshots, ``rel-1`` and the one that serves: TTTD fails it.
        append_records(
            scenario,
            (
                tttd_inference("i0", "rel-1"),
                tttd_report("r0", "i0", 0, 1.0),
                tttd_inference("i1"),
                tttd_report("r1", "i1", 1, 0.0),
            ),
        )
        first = step(scenario, HARNESS)
        if reload:
            scenario = dispatcher.registry.reload("agent")
        # The controller samples step 0 again, on the snapshot that serves alone.
        append_records(
            scenario,
            (
                tttd_inference("j0"),
                tttd_report("s0", "j0", 0, 1.0),
                tttd_inference("j1"),
                tttd_report("s1", "j1", 1, 0.0),
            ),
        )
        resampled = step(scenario, HARNESS)
        return first, scenario.trainer_for(HARNESS).pipeline_status()["failed_steps"], resampled, served
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_tttd_step_failed_on_two_artifacts_stays_failed_after_a_reload(tmp_path: Path) -> None:
    """A rebuilt TTTD harness reads step 0 again, reports it failed with its snapshots, and never trains a resample
    of it."""
    live = tttd_failed_step(tmp_path / "live", reload=False)
    rebuilt = tttd_failed_step(tmp_path / "rebuilt", reload=True)
    # Each run's scenario has its own artifact IDs.
    for run in (live, rebuilt):
        assert run[1] == [{"step": 0, "reason": "mixed_artifact_ids", "artifact_ids": sorted(["rel-1", run[3]])}]
    assert live[2] is None and (rebuilt[0], rebuilt[2]) == (live[0], live[2])
