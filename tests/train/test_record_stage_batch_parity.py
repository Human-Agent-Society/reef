"""Batch goldens for the record-to-record migration (RFC #765).

Each stream appends records to a record store and drives a ``Trainer`` through
reserve, execute, prepare-commit, and commit, as the dispatcher does. For each
committed batch, the test saves the batch ID, a stable digest of each item, and
the record IDs that the commit consumed and released.

The goldens in ``data/record_stage_batch_goldens.json`` come from the processor
engines before grouping, judging, eligibility, and selection move into the
record-to-record stage. The migration must keep them unchanged. Change a golden
only for an intended behavior change, and give the reason in the pull request.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Self

import pytest
from tests.recipes.sdpo.test_sdpo_pipeline import SDPOTokenizer
from tests.support._threshold_processor import ThresholdBatches, ThresholdTasks
from tests.support._trajectories import policy_trajectory

from recipes.beta.coral.recipe import CoralRecipe
from recipes.sdpo import SDPOReport
from recipes.sdpo.recipe import SDPORecipe
from recipes.tttd import TTTDGroupedRolloutReport
from recipes.tttd.recipe import TTTDRecipe
from reef.artifact import ArtifactRef
from reef.background_jobs import SupportsReceipt
from reef.core import AgentRecord, RequestType
from reef.core.plugins import NoSettings
from reef.core.reports import ReportBase, TeacherContextReport
from reef.recipe.generated_tasks.recipe import GeneratedTaskRecipe
from reef.record2record import (
    ExclusiveReportRecord2Record,
    GroupDecision,
    JudgingRecord2Record,
    Record2Record,
    RecordJudging,
    ReportContext,
    ReportGrouping,
    ReportRecord2Record,
)
from reef.record2task import JudgedTasks, Record2Task, ReportTasks
from reef.record2task.assembly import SampleAssembly, SampleAssemblySettings
from reef.record2task.distill import DistillTasks
from reef.storage.sqlite import SQLiteRecordStore
from reef.task2batch import BufferSettings, DistillBatches, Task2Batch
from reef.train import PipelineContext, Trainer
from reef.train.backend import CandidateBackend, PreparedStep
from reef.train.cordis_backend.processor import TraceBatches, TraceTasks
from reef.train.pipeline import PipelineConfig, PipelineSpec
from reef.train.types import TaskItem, TrainDataItem, TrainingBatch, TrajectoryItem

GOLDENS_PATH = Path(__file__).parent / "data" / "record_stage_batch_goldens.json"

CYCLE = "cycle"
RELOAD = "reload"


@dataclass(frozen=True)
class Serve:
    """The scenario serves ``runtime_load_id`` from this step on; the pipeline's staleness check measures against it."""

    runtime_load_id: str


Step = AgentRecord | Serve | str


# ------------------------------------------------------------------- the driver


class CountingBackend(CandidateBackend):
    """A dispatched backend that trains nothing and counts steps, so each reserved batch commits."""

    @property
    def dispatched(self) -> bool:
        return True

    def initial_state(self) -> dict[str, Any]:
        return {"steps": 0}

    def prepare_step(self, batch: TrainingBatch, state: Mapping[str, Any], scenario_step: int) -> PreparedStep:
        return PreparedStep.skipped(state={"steps": int(state.get("steps", 0)) + 1}, metrics={})

    def evaluate(self, candidate: object) -> object:
        raise AssertionError("the counting backend produces no candidates")

    def settle_step(self, prepared: PreparedStep, decision: object) -> object:
        raise AssertionError("the counting backend settles no candidates")

    def abort_step(self, prepared: PreparedStep) -> None:
        return None


def json_value(value: object) -> object:
    """Convert tuples, paths, and other values to the JSON form the goldens store."""
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def describe_item(item: TrainDataItem) -> dict[str, object]:
    """The deterministic parts of one batch item."""
    if isinstance(item, TaskItem):
        return {"task": str(item.task_path), "sources": list(item.source_agent_record_ids)}
    training = item.training
    metadata = item.metadata
    training_fields = (
        "tokens",
        "loss_mask",
        "rollout_log_probs",
        "runtime_load_id",
        "teacher_tokens",
        "distill_sample_weight",
    )
    return {
        "sources": list(item.source_agent_record_ids),
        "group": item.group_id,
        "reward": json_value(metadata.get("reward")),
        "training": {name: json_value(training[name]) for name in training_fields if name in training},
        "recipe": {name: json_value(metadata[name]) for name in ("tttd", "coral", "sdpo", "task") if name in metadata},
    }


def stage_configs(**configs: Mapping[str, object]) -> dict[str, dict[str, Mapping[str, object]]]:
    """A deployment's ``pipeline`` section: each named stage's config, for the stage class of the spec."""
    return {stage: {"config": config} for stage, config in configs.items()}


@dataclass
class StreamRun:
    """One stream's trainer, its record store, and what its commits recorded.

    ``spec`` names the three stage classes, as a recipe does; ``pipeline`` is
    the deployment's ``pipeline`` section over it. ``batch_size`` is the
    trainer's samples in one step.
    """

    spec: PipelineSpec
    pipeline: Mapping[str, object] = field(default_factory=dict)
    report_type: type[ReportBase] | None = None
    training_mode: str = "auto"
    max_staleness: int = 0
    batch_size: int = 1
    serving_runtime_load_id: str | None = None
    records: SQLiteRecordStore = field(default_factory=SQLiteRecordStore)
    consumed: set[str] = field(default_factory=set)
    high_water: tuple[int, int] | None = None
    commits: list[dict[str, object]] = field(default_factory=list)

    def build(self) -> Trainer:
        return Trainer.build(
            "s",
            self.records,
            pipeline_factory=PipelineConfig.from_setting(self.pipeline, spec=self.spec, environ={}).build,
            candidate_backend=CountingBackend(),
            report_type=self.report_type,
            training_mode=self.training_mode,
            max_staleness=self.max_staleness,
            batch_size=self.batch_size,
        )

    def reload(self, trainer: Trainer) -> Trainer:
        """Rebuild the trainer from durable state, in the order scenario recovery uses."""
        trainer.close()
        rebuilt = self.build()
        if self.high_water is not None:
            sequence, offset = self.high_water
            rebuilt.replay_retained(up_to_sequence=sequence, consumed_ids=frozenset(self.consumed))
            rebuilt.restore_record_progress(after_sequence=sequence, offset=offset)
        elif self.consumed:
            rebuilt.replay_retained(up_to_sequence=0, consumed_ids=frozenset(self.consumed))
        return rebuilt

    def drain(self, trainer: Trainer) -> None:
        """Commit every batch the trainer can reserve now."""
        while (
            batch := trainer.reserve_training_batch(serving_runtime_load_id=self.serving_runtime_load_id)
        ) is not None:
            execution = trainer.execute_reserved_step(len(self.commits))
            if execution.result is None:
                raise AssertionError(f"batch {batch.batch_id} produced no result")
            prepared = trainer.prepare_commit(execution.result)
            trainer.commit(prepared)
            self.consumed |= prepared.consumed_ids
            self.high_water = (prepared.high_water_sequence, prepared.high_water_offset)
            self.commits.append(
                {
                    "batch_id": batch.batch_id,
                    "request": None if batch.request is None else batch.request.id,
                    "items": [describe_item(item) for item in batch.items],
                    "consumed": sorted(prepared.consumed_ids),
                    "released": sorted(prepared.released_ids),
                }
            )

    def run(self, steps: Sequence[Step]) -> list[dict[str, object]]:
        trainer = self.build()
        try:
            for step in steps:
                if isinstance(step, AgentRecord):
                    self.records.append(step)
                elif isinstance(step, Serve):
                    self.serving_runtime_load_id = step.runtime_load_id
                    trainer.pipeline.observe_serving_runtime_load_id(step.runtime_load_id)
                elif step == CYCLE:
                    self.drain(trainer)
                elif step == RELOAD:
                    trainer = self.reload(trainer)
                else:
                    raise ValueError(f"unknown stream step {step!r}")
            self.drain(trainer)
        finally:
            trainer.close()
        return self.commits


# ---------------------------------------------------------------- record builders


def inference(record_id: str, *, version: str | None = None, artifact_ref: ArtifactRef | None = None) -> AgentRecord:
    training: dict[str, object] = {"tokens": [1, 2, 3], "loss_mask": [0, 1, 1], "rollout_log_probs": [-0.1, -0.2]}
    if version is not None:
        training["runtime_load_id"] = version
    return AgentRecord.create(
        scenario="s",
        request_type=RequestType.INFERENCE,
        agent_record_id=record_id,
        artifact_ref=artifact_ref,
        payload={"response": {"training": training}},
    )


def report(record_id: str, *references: str, score: float = 1.0, **metadata: object) -> AgentRecord:
    payload: dict[str, object] = {"score": score, "references": list(references)}
    if metadata:
        payload["metadata"] = metadata
    return AgentRecord.create(
        scenario="s",
        request_type=RequestType.REPORT,
        agent_record_id=record_id,
        references=references,
        payload=payload,
    )


def instruction(record_id: str) -> AgentRecord:
    return AgentRecord.create(
        scenario="s",
        request_type=RequestType.TRAIN,
        agent_record_id=record_id,
        payload={"text": record_id, "session": "session-1", "release_id": "release-1"},
    )


def scored(prefix: str, index: int, score: float = 1.0, **metadata: object) -> list[AgentRecord]:
    """An inference and the report that scores it."""
    return [
        inference(f"i-{prefix}-{index}"),
        report(f"r-{prefix}-{index}", f"i-{prefix}-{index}", score=score, **metadata),
    ]


# ------------------------------------------------------------- in-file pipelines


class SlotGrouping(ReportGrouping):
    """Groups by ``metadata.group`` with slots from ``metadata.slot``; ready at two, discarded on mixed feedback."""

    def placement(self, context: ReportContext) -> tuple[Hashable | None, Hashable | None]:
        metadata = context.report.payload.get("metadata", {})
        return metadata.get("group"), metadata.get("slot")

    def decide(self, key: Hashable, members: tuple[ReportContext, ...]) -> GroupDecision:
        if any(member.report.payload.get("feedback") == "mixed" for member in members):
            return GroupDecision.DISCARD
        return GroupDecision.READY if len(members) >= 2 else GroupDecision.INCOMPLETE


class AnyGroupReady(SlotGrouping):
    """Every group is ready as soon as it has a report."""

    def decide(self, key: Hashable, members: tuple[ReportContext, ...]) -> GroupDecision:
        return GroupDecision.READY


class AssembledTasks(ReportTasks):
    """One report as one assembled policy sample."""

    def __init__(self) -> None:
        super().__init__()
        self.assembly = SampleAssembly(SampleAssemblySettings())

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        context: PipelineContext,
        record2record: Record2Record,
        environ: Mapping[str, str],
    ) -> AssembledTasks:
        NoSettings.parse(config, plugin_class=cls)
        return cls()

    def make_sample(self, context: ReportContext) -> TrajectoryItem:
        return self.assembly.build(context, context.require_score())


class SlotBatches(Task2Batch):
    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        context: PipelineContext,
        record2record: Record2Record,
        record2task: Record2Task,
        environ: Mapping[str, str],
    ) -> Self:
        return cls(context, BufferSettings.parse(config, plugin_class=cls).buffer_size)

    def make_batch(self, tasks: tuple[TrainDataItem, ...], batch_id: str) -> TrainingBatch:
        return TrainingBatch(batch_id, tasks)


class SlotGroupRecord2Record(ReportRecord2Record):
    """Holds the groups of ``SlotGrouping``; a terminal report owns its sources."""

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], *, context: PipelineContext, environ: Mapping[str, str]
    ) -> SlotGroupRecord2Record:
        NoSettings.parse(config, plugin_class=cls)
        return cls(context, SlotGrouping(), exclusive_sources=True)


class OrderedMixRecord2Record(ReportRecord2Record):
    """Reports with a ``metadata.group`` form groups that batch in key order after singletons."""

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], *, context: PipelineContext, environ: Mapping[str, str]
    ) -> OrderedMixRecord2Record:
        NoSettings.parse(config, plugin_class=cls)
        return cls(context, AnyGroupReady(), exclusive_sources=True, ordered_groups=True)


# Each recipe builds its three stages. These specs name the same stages, as the recipe would.

THRESHOLD_STAGES = PipelineSpec(
    record2record=ReportRecord2Record, record2task=ThresholdTasks, task2batch=ThresholdBatches
)
SLOT_GROUP_STAGES = PipelineSpec(
    record2record=SlotGroupRecord2Record, record2task=AssembledTasks, task2batch=SlotBatches
)
ORDERED_MIX_STAGES = PipelineSpec(
    record2record=OrderedMixRecord2Record, record2task=AssembledTasks, task2batch=SlotBatches
)
DISTILL_STAGES = PipelineSpec(
    record2record=ExclusiveReportRecord2Record, record2task=DistillTasks, task2batch=DistillBatches
)
#: The harness-evolution pipeline of reported traces.
TRACE_STAGES = PipelineSpec(record2record=ReportRecord2Record, record2task=TraceTasks, task2batch=TraceBatches)


@dataclass(frozen=True)
class ToyJob(SupportsReceipt):
    pass


@dataclass(frozen=True)
class ToyJudgment(SupportsReceipt):
    good: bool = True


class ImmediateWorker:
    """A worker that judges each job when the pipeline next polls."""

    def __init__(self) -> None:
        self.ready: list[ToyJudgment] = []

    def submit(self, job: ToyJob) -> bool:
        self.ready.append(ToyJudgment(job.receipt, good=not job.receipt.startswith("bad")))
        return True

    def poll(self) -> list[ToyJudgment]:
        judgments, self.ready = self.ready, []
        return judgments

    def close(self) -> None:
        return None


class ToyJudging(RecordJudging):
    """A record tracks itself; a record whose payload names ``completes`` finishes that receipt."""

    def observe(self, item: AgentRecord, now: float, stage: JudgingRecord2Record) -> None:
        completes = item.payload.get("completes")
        if completes:
            stage.dispatch(ToyJob(completes))
        if item.payload.get("track"):
            stage.track(item)
        else:
            stage.retire(item.agent_record_id)

    async def judge(self, job: SupportsReceipt) -> SupportsReceipt:
        raise AssertionError("the immediate worker judges without the coroutine")


class ToyTasks(JudgedTasks):
    """One-token stub samples; a judgment that is not good cannot train."""

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        context: PipelineContext,
        record2record: Record2Record,
        environ: Mapping[str, str],
    ) -> ToyTasks:
        NoSettings.parse(config, plugin_class=cls)
        return cls()

    def make_sample(self, record: AgentRecord, judgment: SupportsReceipt) -> TrajectoryItem | None:
        if not isinstance(judgment, ToyJudgment) or not judgment.good:
            return None
        return policy_trajectory(
            source_agent_record_id=record.agent_record_id,
            tokens=(1, 2),
            loss_mask=(1,),
            rollout_log_probs=(-0.1,),
            reward=1.0,
            runtime_load_id=record.payload.get("version"),
        )


class ToyBatches(Task2Batch):
    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        context: PipelineContext,
        record2record: Record2Record,
        record2task: Record2Task,
        environ: Mapping[str, str],
    ) -> Self:
        return cls(context, BufferSettings.parse(config, plugin_class=cls).buffer_size)

    def make_batch(self, tasks: tuple[TrainDataItem, ...], batch_id: str) -> TrainingBatch:
        return TrainingBatch(batch_id, tasks)


class ToyJudgingRecord2Record(JudgingRecord2Record):
    """Judges the records of ``ToyJudging`` at the next poll."""

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], *, context: PipelineContext, environ: Mapping[str, str]
    ) -> ToyJudgingRecord2Record:
        NoSettings.parse(config, plugin_class=cls)
        return cls(ToyJudging(), ImmediateWorker())


#: Trains the judged records of ``ToyJudging``.
TOY_COMPUTED_STAGES = PipelineSpec(record2record=ToyJudgingRecord2Record, record2task=ToyTasks, task2batch=ToyBatches)


def turn(record_id: str, *, version: str, completes: str | None = None, track: bool = True) -> AgentRecord:
    payload: dict[str, object] = {"version": version, "track": track}
    if completes is not None:
        payload["completes"] = completes
    return AgentRecord.create(
        scenario="s", request_type=RequestType.INFERENCE, agent_record_id=record_id, payload=payload
    )


# ----------------------------------------------------------------------- streams


def ungrouped_stream() -> StreamRun:
    return StreamRun(THRESHOLD_STAGES, batch_size=2)


def ungrouped_steps() -> list[Step]:
    return [
        *scored("a", 0),
        *scored("a", 1, 0.5),
        CYCLE,
        *scored("a", 2, 0.0),
        CYCLE,
        *scored("a", 3),
        *scored("a", 4),
    ]


def slot_steps() -> list[Step]:
    return [
        *scored("g1", 0, group="g1", slot="x"),
        # A retry in a taken slot is terminal.
        inference("i-g1-retry"),
        report("r-g1-retry", "i-g1-retry", score=0.0, group="g1", slot="x"),
        *scored("g1", 1, group="g1", slot="y"),
        # A group with mixed feedback is discarded, and a later report for it is terminal.
        inference("i-g2-0"),
        AgentRecord.create(
            scenario="s",
            request_type=RequestType.REPORT,
            agent_record_id="r-g2-0",
            references=("i-g2-0",),
            payload={"score": 1.0, "references": ["i-g2-0"], "feedback": "mixed", "metadata": {"group": "g2"}},
        ),
        *scored("g2", 1, group="g2"),
        *scored("g2", 2, group="g2"),
        CYCLE,
        # A consumed group reopens for later reports.
        *scored("g1", 2, group="g1", slot="z"),
        *scored("g1", 3, group="g1", slot="w"),
    ]


def ordered_steps() -> list[Step]:
    return [
        *scored("z", 0, group="g-z"),
        *scored("a", 0, group="g-a"),
        *scored("single", 0),
    ]


def tttd_inference(step: int, group: int, rollout: int, release: str) -> AgentRecord:
    return replace(
        inference(f"i-{step}-{group}-{rollout}"),
        artifact_ref=ArtifactRef(f"artifact-{release}", release, None),
    )


def tttd_report(step: int, group: int, rollout: int, score: float) -> AgentRecord:
    return report(
        f"r-{step}-{group}-{rollout}",
        f"i-{step}-{group}-{rollout}",
        score=score,
        algorithm="ttt-discover",
        comparison_set=f"tttd-step-{step}-group-{group}",
        step=step,
        group=group,
        rollout=rollout,
        groups_per_step=2,
        rollouts_per_group=2,
    )


def tttd_steps() -> list[Step]:
    steps: list[Step] = []
    for step, mixed in ((0, True), (1, False)):
        for group in range(2):
            for rollout in range(2):
                release = "v2" if mixed and (group, rollout) == (1, 1) else "v1"
                steps += [
                    tttd_inference(step, group, rollout, release),
                    tttd_report(step, group, rollout, float(rollout)),
                ]
        if step == 1:
            steps.insert(len(steps) - 2, RELOAD)
    return steps


def coral_report(record_id: str, reference: str, score: float, *, commit: str, parent: str | None) -> AgentRecord:
    coral = {
        "agent_id": "agent-1",
        "commit_hash": commit,
        "parent_hash": parent,
        "status": "improved",
        "run_id": "run-1",
    }
    return report(record_id, reference, score=score, coral=coral)


def coral_steps() -> list[Step]:
    return [
        inference("i1", version="wv-1"),
        coral_report("r1", "i1", 0.3, commit="c-a", parent="p0"),
        inference("i1-retry", version="wv-1"),
        coral_report("r1-retry", "i1-retry", 0.1, commit="c-a", parent="p0"),
        inference("i2", version="wv-1"),
        coral_report("r2", "i2", 0.9, commit="c-b", parent="p0"),
        inference("i3", version="wv-1"),
        coral_report("r3", "i3", 0.5, commit="c-c", parent=None),
        inference("i4", version="wv-1"),
        coral_report("r4", "i4", 0.6, commit="c-d", parent=None),
    ]


def task_report(record_id: str, reference: str, task: str, score: float, *, arm: str = "plain") -> AgentRecord:
    return report(
        record_id,
        reference,
        score=score,
        task={"name": task, "path": f"/tasks/{task}", "digest": "ab" * 32},
        episode={"id": record_id, "labels": {"arm": arm}},
    )


def task_steps() -> list[Step]:
    steps: list[Step] = []
    for index, (task, score, arm) in enumerate(
        (
            ("t0", 1.0, "plain"),
            ("t1", 0.0, "plain"),
            ("t0", 0.0, "measure"),
            ("t0", 0.0, "plain"),
            ("t1", 1.0, "plain"),
        )
    ):
        steps += [inference(f"i-{index}", version="r1"), task_report(f"r-{index}", f"i-{index}", task, score, arm=arm)]
    return steps


def distill_inference(record_id: str, question: str = "What is the boiling point of water?") -> AgentRecord:
    return AgentRecord.create(
        scenario="s",
        request_type=RequestType.INFERENCE,
        agent_record_id=record_id,
        payload={
            "messages": [{"role": "user", "content": question}],
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
            "response": {"choices": [{"message": {"role": "assistant", "content": "About 90 degrees."}}]},
            "tokens": [5, 6, 7, 1, 2, 3],
            "loss_mask": [1, 1, 1],
            "rollout_log_probs": [-0.1, -0.2, -0.3],
        },
    )


def distill_report(record_id: str, reference: str, teacher_context: str) -> AgentRecord:
    body = TeacherContextReport(teacher_context=teacher_context).to_dict(references=(reference,))
    return AgentRecord.create(
        scenario="s", request_type=RequestType.REPORT, agent_record_id=record_id, references=(reference,), payload=body
    )


def distill_steps() -> list[Step]:
    return [
        distill_inference("i0"),
        distill_report("r0", "i0", "100 degrees Celsius."),
        # A long request: the teacher sequence overflows the window, and the report is discarded.
        distill_inference("i1", "x" * 400),
        distill_report("r1", "i1", "Water boils at 100 degrees."),
        distill_inference("i2"),
        distill_report("r2", "i2", "Water boils at 100 degrees."),
    ]


def sdpo_records(step: int, rollout: int, *, score: float, question: str = "What is 2+2?") -> list[AgentRecord]:
    name = f"s{step}r{rollout}"
    record = AgentRecord.create(
        scenario="s",
        request_type=RequestType.INFERENCE,
        agent_record_id=name,
        payload={
            "messages": [{"role": "user", "content": question}],
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
            "tokens": [5, 6, 7, 1, 2, 3],
            "loss_mask": [1, 1, 1],
            "rollout_log_probs": [-0.1, -0.2, -0.3],
            "response": {"choices": [{"message": {"role": "assistant", "content": f"Answer from {name}"}}]},
        },
        artifact_ref=ArtifactRef("artifact-v3", "slime-v3", None),
    )
    body = SDPOReport(step=step, group=0, rollout=rollout, score=score, teacher_context="").to_dict(references=(name,))
    return [
        record,
        AgentRecord.create(
            scenario="s",
            request_type=RequestType.REPORT,
            agent_record_id=f"report-{name}",
            references=(name,),
            payload=body,
        ),
    ]


def sdpo_steps() -> list[Step]:
    return [
        *sdpo_records(0, 0, score=1.0),
        *sdpo_records(0, 1, score=0.0),
        # Siblings asked different questions: the step is discarded.
        *sdpo_records(1, 0, score=1.0),
        *sdpo_records(1, 1, score=0.0, question="What is 3+3?"),
        *sdpo_records(2, 1, score=0.0),
        *sdpo_records(2, 0, score=1.0),
    ]


def computed_steps() -> list[Step]:
    return [
        Serve("e:1"),
        turn("a", version="e:1"),
        Serve("e:2"),
        turn("b", version="e:2", completes="a"),
        Serve("e:3"),
        turn("c", version="e:3", completes="b"),
        turn("bad-d", version="e:3", completes="c"),
        turn("e", version="e:3", completes="bad-d"),
        RELOAD,
        turn("f", version="e:3", completes="e"),
        turn("closer", version="e:3", completes="f", track=False),
    ]


def cordis_steps(mode_requests: bool) -> list[Step]:
    steps: list[Step] = []
    for index in range(6):
        steps += scored("c", index, float(index))
    if mode_requests:
        steps += [instruction("train-1"), CYCLE, *scored("c", 6), instruction("train-2")]
    return steps


def streams() -> dict[str, tuple[StreamRun, list[Step]]]:
    """Every stream, built fresh: a run keeps its own record store and commits."""
    grid = {"groups_per_step": 2, "rollouts_per_group": 2}
    sdpo = stage_configs(
        record2record={"groups_per_step": 1, "rollouts_per_group": 2},
        record2task={"tokenizer_path": "/models/s", "max_teacher_tokens": 18432},
    )
    distill = stage_configs(record2task={"tokenizer_path": "/models/s", "max_teacher_tokens": 30})
    generated = stage_configs(record2record={"plays_per_task": 2})
    return {
        "ungrouped": (ungrouped_stream(), ungrouped_steps()),
        "ungrouped_reload": (ungrouped_stream(), [*ungrouped_steps()[:5], RELOAD, *ungrouped_steps()[5:]]),
        "slots_and_discards": (
            StreamRun(SLOT_GROUP_STAGES),
            slot_steps(),
        ),
        "ordered_groups": (
            StreamRun(ORDERED_MIX_STAGES, batch_size=3),
            ordered_steps(),
        ),
        "tttd": (
            StreamRun(TTTDRecipe.pipeline_spec(), stage_configs(record2record=grid), TTTDGroupedRolloutReport),
            tttd_steps(),
        ),
        "coral": (
            StreamRun(CoralRecipe.pipeline_spec(), stage_configs(record2record={"group_size": 2})),
            coral_steps(),
        ),
        "generated_tasks": (StreamRun(GeneratedTaskRecipe.pipeline_spec(), generated, batch_size=4), task_steps()),
        "distill_overflow": (StreamRun(DISTILL_STAGES, distill, TeacherContextReport), distill_steps()),
        "sdpo": (StreamRun(SDPORecipe.pipeline_spec(), sdpo, SDPOReport), sdpo_steps()),
        "computed_exact_version": (StreamRun(TOY_COMPUTED_STAGES, batch_size=2), computed_steps()),
        "computed_window": (StreamRun(TOY_COMPUTED_STAGES, max_staleness=1, batch_size=2), computed_steps()),
        "cordis_manual_cap": (
            StreamRun(TRACE_STAGES, training_mode="manual"),
            cordis_steps(True),
        ),
        "cordis_hybrid": (
            StreamRun(TRACE_STAGES, training_mode="hybrid", batch_size=2),
            cordis_steps(True),
        ),
    }


STREAM_NAMES = sorted(streams())


@pytest.fixture(autouse=True)
def counting_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=SDPOTokenizer()))


def run_stream(name: str) -> list[dict[str, object]]:
    stream, steps = streams()[name]
    commits = json_value(stream.run(steps))
    if not isinstance(commits, list):
        raise TypeError("a stream run returns a list of commits")
    return commits


@pytest.mark.unit
@pytest.mark.parametrize("name", STREAM_NAMES)
def test_committed_batches_match_the_recorded_goldens(name: str) -> None:
    goldens = json.loads(GOLDENS_PATH.read_text(encoding="utf-8"))
    assert run_stream(name) == goldens[name]


@pytest.mark.unit
def test_every_stream_has_a_golden() -> None:
    assert sorted(json.loads(GOLDENS_PATH.read_text(encoding="utf-8"))) == STREAM_NAMES
