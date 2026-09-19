"""SPADE training data: the Reasoning Agent's episodes grouped by task, the Designer's reports by generation, and the generations run through a generator."""

from __future__ import annotations

import asyncio
import json
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import pytest
from reef_service.runtime_stubs import StubTrainingRuntime

from recipes.beta.spade import SpadeDesignerProcessor, SpadeDesignerRecipe, SpadeObjective, SpadeProcessor, SpadeRecipe
from recipes.beta.spade.designer_processor import generation_label, reported_generation
from recipes.beta.spade.processor import REFUSAL_SCORE, GenerationJob, reported_task_name
from reef.core import AgentRecord, RequestType
from reef.core.reports import ScoredRolloutReport
from reef.core.tasks import (
    HarborTask,
    HarborTaskConflict,
    read_harbor_task,
    read_split_manifest,
    write_harbor_task,
    write_split_manifest,
)
from reef.core.trajectories import make_trajectory
from reef.harness.client.tasks import TaskPlay
from reef.inference.http import InferenceProxyRuntime
from reef.record2dataset import (
    DesignerRequest,
    DuplicateTask,
    Generator,
    GeneratorError,
    OracleResult,
    ProposedTask,
    TaskNameConflict,
    WrittenTask,
    content_hash,
    split_generation,
)
from reef.train.algos.registry import resolve_objective
from reef.train.processors.computed import Failed, JudgingWorker, SupportsReceipt
from reef.train.processors.reported import GroupDecision
from reef.train.types import ProcessorContext, TrainingBatch, TrajectoryItem, trajectory_groups

# ------------------------------------------------------------------------------------------- the reported half


def inference(record_id: str) -> AgentRecord:
    payload = {
        "messages": [{"role": "user", "content": "ls"}],
        "response": {"choices": [{"message": {"role": "assistant", "content": "ls -la"}}]},
        "training": {
            "tokens": [1, 2, 3],
            "loss_mask": [0, 1, 1],
            "rollout_log_probs": [0.0, -0.5, -0.7],
            "runtime_load_id": "r1",
        },
    }
    return AgentRecord.create(
        scenario="spade", request_type=RequestType.INFERENCE, payload=payload, agent_record_id=record_id
    )


def report(
    record_id: str,
    reference: str,
    task: str | None,
    score: float,
    *,
    arm: str | None = "plain",
    role: str | None = None,
    **extra: object,
) -> AgentRecord:
    payload: dict[str, object] = {"score": score, "feedback": f"verifier reward {score}", "references": [reference]}
    metadata: dict[str, object] = {}
    if task is not None:
        metadata["task"] = {"name": task, "path": f"/tasks/{task}", "digest": "ab" * 32}
    if arm is not None:
        metadata["episode"] = {"id": record_id, "labels": {"arm": arm}}
    if role is not None:
        metadata["role"] = role
    metadata.update(extra)
    if metadata:
        payload["metadata"] = metadata
    return AgentRecord.create(
        scenario="spade", request_type=RequestType.REPORT, payload=payload, agent_record_id=record_id
    )


def processor(**config: object) -> SpadeProcessor:
    return SpadeProcessor(ProcessorContext("spade", {"tasks_per_step": 2, "rollouts_per_task": 2, **config}))


def played(processor: SpadeProcessor, task: str, index: int, score: float, **fields: object) -> None:
    processor.ingest(inference(f"rec-{task}-{index}"))
    processor.ingest(report(f"rep-{task}-{index}", f"rec-{task}-{index}", task, score, **fields))  # type: ignore[arg-type]


def test_reported_task_name_reads_the_task_players_metadata() -> None:
    assert reported_task_name(report("r", "i", "harbor-00000-000", 1.0)) == "harbor-00000-000"
    assert reported_task_name(report("r", "i", None, 1.0)) is None


def test_episodes_of_one_task_form_a_group_and_a_batch_holds_complete_groups_only() -> None:
    p = processor()
    played(p, "harbor-00000-000", 0, 1.0)
    assert not p.ready(), "one episode of a task is not a group"
    played(p, "harbor-00000-001", 0, 0.0)
    played(p, "harbor-00000-000", 1, 0.0)
    assert not p.ready(), "one complete group is not a batch of two"
    played(p, "harbor-00000-001", 1, 1.0)
    assert p.ready()
    batch = p.build_batch()
    assert isinstance(batch, TrainingBatch) and batch.batch_id == "spade:spade:1"
    groups = trajectory_groups(batch)
    assert [[sample.group_id for sample in group] for group in groups] == [
        ["harbor-00000-000", "harbor-00000-000"],
        ["harbor-00000-001", "harbor-00000-001"],
    ]
    assert [sample.metadata["reward"] for group in groups for sample in group] == [1.0, 0.0, 0.0, 1.0]


def test_a_report_without_a_task_is_refused_and_keeps_its_report_retryable() -> None:
    p = processor()
    p.ingest(inference("rec-x"))
    with pytest.raises(ValueError, match=r"requires metadata\.task\.name"):
        p.ingest(report("rep-x", "rec-x", None, 1.0))


def test_the_designers_reports_and_the_hint_arms_are_released_not_trained() -> None:
    p = processor()
    played(p, "harbor-00000-000", 0, 0.0, arm="hint")
    played(p, "harbor-00000-000", 1, 1.0, arm="hint")
    assert not p.ready(), "hint episodes measure the task; they never form a group"
    p.ingest(inference("rec-designer"))
    p.ingest(report("rep-designer", "rec-designer", None, 0.5, arm=None, role="designer"))
    p.ingest(inference("rec-designer-refused"))
    p.ingest(
        report("rep-designer-refused", "rec-designer-refused", "harbor-00000-000", 0.0, arm=None, role="designer")
    )
    assert not p.ready()
    released = p.retention_decision().releasable_agent_record_ids
    assert {"rep-designer", "rec-designer", "rep-designer-refused", "rec-designer-refused"} <= released
    assert {"rec-harbor-00000-000-0", "rec-harbor-00000-000-1"} <= released, "the hint arm's calls are released too"
    played(p, "harbor-00000-000", 2, 1.0)
    played(p, "harbor-00000-000", 3, 0.0)
    played(p, "harbor-00000-001", 0, 1.0)
    played(p, "harbor-00000-001", 1, 0.0)
    assert p.ready() and len(p.build_batch().items) == 4


def test_decide_group_waits_for_rollouts_per_task_episodes() -> None:
    p = processor(rollouts_per_task=3)
    one: tuple[TrajectoryItem, ...] = (make_trajectory((inference("a"),), 1.0),)
    assert p.decide_group("t", one) is GroupDecision.INCOMPLETE
    assert p.decide_group("t", one * 3) is GroupDecision.READY


def test_a_rounds_plays_are_one_batch_ordered_by_task_whatever_tasks_per_step_says() -> None:
    p = processor(tasks_per_step=4, rollouts_per_task=3)
    played(p, "harbor-00003-000", 9, 1.0, arm="hint", generation=3)
    assert not p.ready()
    assert "rec-harbor-00003-000-9" in p.retention_decision().releasable_agent_record_ids, "the hint arm is released"
    plays = [
        ("harbor-00003-001", 1.0),
        ("harbor-00003-000", 0.0),
        ("harbor-00003-001", 0.0),
        ("harbor-00003-000", 1.0),
    ]
    stamp = {"generation": 3, "round": "generation-00003", "round_plays": len(plays)}
    for index, (task, score) in enumerate(plays):
        played(p, task, index, score, **stamp)
        assert p.ready() == (index == len(plays) - 1), "a round is ready at its size, not at rollouts_per_task"
    batch = p.build_batch()
    assert [sample.group_id for sample in batch.items] == ["harbor-00003-000"] * 2 + ["harbor-00003-001"] * 2
    assert [sample.metadata["reward"] for sample in batch.items] == [0.0, 1.0, 1.0, 0.0]
    assert all(sample.metadata["round"] == "generation-00003" for sample in batch.items)
    assert all(sample.metadata["round_plays"] == 4 for sample in batch.items)
    assert resolve_objective("spade").prepare(batch, {}).advantages == (-1.0, 1.0, 1.0, -1.0)
    p.acknowledge(batch.batch_id)
    assert not p.ready()
    played(p, "harbor-00003-002", 0, 1.0)
    played(p, "harbor-00003-002", 1, 0.0)
    played(p, "harbor-00003-002", 2, 0.0)
    assert not p.ready(), "a report without a round groups by task, as before"


@pytest.mark.parametrize("round_plays", [0, -2, "4", 2.0, True, None])
def test_a_round_report_without_its_size_is_refused(round_plays: object) -> None:
    p = processor()
    stamp: dict[str, object] = {"round": "generation-00003"}
    if round_plays is not None:
        stamp["round_plays"] = round_plays
    with pytest.raises(ValueError, match="round_plays"):
        played(p, "harbor-00003-000", 0, 1.0, **stamp)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"tasks_per_step": 0}, "tasks_per_step must be positive"),
        ({"rollouts_per_task": 1}, "at least two"),
        ({"generations": 1}, "generator_url"),
        ({"generations": 1, "generator_url": "${endpoints.generator}"}, "generator_url"),
        ({"generations": 1, "generator_url": "http://127.0.0.1:8910"}, "description"),
        (
            {
                "generations": 1,
                "generator_url": "http://127.0.0.1:8910",
                "description": "x",
                "state_dir": "s",
                "count": 0,
            },
            "count",
        ),
    ],
)
def test_a_processor_configuration_that_cannot_train_is_refused(config: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        processor(**config)


def test_the_objective_centers_and_scales_rewards_within_each_task_group() -> None:
    p = processor()
    for index, score in enumerate((1.0, 0.0)):
        played(p, "harbor-00000-000", index, score)
    for index, score in enumerate((0.5, 0.5)):
        played(p, "harbor-00000-001", index, score)
    batch = p.build_batch()
    objective = resolve_objective("spade")
    assert isinstance(objective, SpadeObjective) and objective.loss_family == "importance_sampling"
    signal = objective.prepare(batch, {})
    assert signal.action == "train"
    assert signal.advantages == (1.0, -1.0, 0.0, 0.0)
    assert signal.metrics["constant_groups"] == 1


def test_a_batch_whose_groups_are_all_constant_is_skipped_and_keeps_the_step_count() -> None:
    p = processor()
    for task in ("harbor-00000-000", "harbor-00000-001"):
        for index in range(2):
            played(p, task, index, 1.0)
    signal = resolve_objective("spade").prepare(p.build_batch(), {"steps": 3})
    assert signal.action == "skip" and signal.advantages is None
    assert signal.next_algorithm_state == {"steps": 3}
    assert signal.metrics == {"constant_groups": 2, "skipped": "every group is constant"}


def test_the_recipe_binds_the_processor_the_objective_and_its_schedule() -> None:
    spec = SpadeRecipe.training_spec()
    assert spec.processor is SpadeProcessor and spec.objective == "spade"
    assert spec.loss_family == "importance_sampling"
    assert spec.scheduling.unit == "sample" and spec.scheduling.batch_size == "actual"
    assert SpadeRecipe.report_type.fget(SpadeRecipe) is ScoredRolloutReport  # type: ignore[union-attr]
    runtime = InferenceProxyRuntime(model_path="Qwen/Qwen3-8B", base_url="http://127.0.0.1:8000")
    training_runtime = StubTrainingRuntime()
    recipe = SpadeRecipe(training_runtime=training_runtime, runtime=runtime)
    config = recipe.processor_config()
    assert config["tasks_per_step"] == 4 and config["rollouts_per_task"] == 4 and config["scaffold_tolerance"] == 8
    assert config["generations"] == 0 and config["generator_url"] == "" and config["skills"] == ()
    assert config["report_plays_after_generation"] is False, "each play reports as it ends unless asked otherwise"
    with pytest.raises(ValueError, match="scaffold_tolerance"):
        SpadeRecipe(training_runtime=training_runtime, runtime=runtime, scaffold_tolerance=-1)
    with pytest.raises(ValueError, match="generator_url"):
        SpadeRecipe(
            training_runtime=training_runtime, runtime=runtime, generations=2, generator_url="${endpoints.generator}"
        )
    with pytest.raises(ValueError, match="description and a state_dir"):
        SpadeRecipe(training_runtime=training_runtime, runtime=runtime, generations=2, generator_url="http://g:8910")
    generating = SpadeRecipe(
        training_runtime=training_runtime,
        runtime=runtime,
        generations=2,
        generator_url="http://g:8910",
        description="shell tasks",
        state_dir="/tmp/spade",
        skills=("inspection", "repair"),
    )
    assert generating.processor_config()["skills"] == ("inspection", "repair")


def test_the_assembly_spans_an_episodes_turns_and_realigns_the_think_scaffold() -> None:
    assembly = processor()._assembly
    assert assembly.accept_multi_turn and assembly.scaffold_tolerance == 8
    assert processor(scaffold_tolerance=2)._assembly.scaffold_tolerance == 2


# ------------------------------------------------------------------------------------------- the Designer's half


def designer_report(record_id: str, reference: str, score: float, **metadata: object) -> AgentRecord:
    payload: dict[str, object] = {
        "score": score,
        "feedback": {"task": None, "round": {"generation": 0, "previous": None}},
        "references": [reference],
    }
    if metadata:
        payload["metadata"] = dict(metadata)
    return AgentRecord.create(
        scenario="spade", request_type=RequestType.REPORT, payload=payload, agent_record_id=record_id
    )


def designer_processor(**config: object) -> SpadeDesignerProcessor:
    return SpadeDesignerProcessor(ProcessorContext("spade", dict(config)))


def proposed(
    processor: SpadeDesignerProcessor, generation: int, index: int, score: float, proposals: int, **extra: object
) -> None:
    processor.ingest(inference(f"designer-{generation}-{index}"))
    processor.ingest(
        designer_report(
            f"regret-{generation}-{index}",
            f"designer-{generation}-{index}",
            score,
            generation=generation,
            proposals=proposals,
            **extra,
        )
    )


def test_reported_generation_reads_the_generations_report_metadata() -> None:
    assert reported_generation(designer_report("r", "i", 0.5, generation=3, proposals=8)) == (3, 8)
    assert generation_label(3) == "generation-00003" and generation_label(3, "repair") == "generation-00003-repair"


def test_a_generations_proposals_form_a_group_and_a_refusal_is_a_member() -> None:
    p = designer_processor()
    proposed(p, 0, 0, 0.5, 3, designer_version={"kind": "runtime", "id": "load-7"}, opponent={"model": "m"})
    proposed(p, 0, 1, REFUSAL_SCORE, 3, refusal="reply refused: no json block")
    assert not p.ready(), "two of three proposals are not a generation"
    assert p.status() == {"ready_groups": 0, "groups": {"0": 2}}
    proposed(p, 0, 2, -0.25, 3)
    assert p.ready() and p.status() == {"ready_groups": 1, "groups": {"0": 3}}
    batch = p.build_batch()
    assert batch.batch_id == "spade:spade-designer:1"
    groups = trajectory_groups(batch)
    assert [[sample.group_id for sample in group] for group in groups] == [["generation-00000"] * 3]
    first = groups[0][0]
    assert first.metadata["generation"] == 0 and first.metadata["proposals"] == 3
    assert first.metadata["designer_version"] == {"kind": "runtime", "id": "load-7"}
    assert first.metadata["opponent"] == {"model": "m"} and "refusal" not in first.metadata
    assert first.metadata["feedback"] == {"task": None, "round": {"generation": 0, "previous": None}}
    assert [sample.metadata["reward"] for sample in groups[0]] == [0.5, -1.0, -0.25]
    assert first.metadata["references"] == ["designer-0-0"], "one chat call, one reference, no multi turn assembly"
    p.acknowledge(batch.batch_id)
    assert p.status() == {"ready_groups": 0, "groups": {}} and not p.ready()


def test_designer_proposals_compare_within_one_generation_and_one_skill() -> None:
    p = designer_processor()
    proposed(p, 2, 0, 0.5, 3, skill="inspection")
    proposed(p, 2, 1, 0.0, 3, skill="repair")
    assert p.status() == {"ready_groups": 0, "groups": {"2": 2}}, "a generation is one unit, ready at proposals"
    proposed(p, 2, 2, 1.0, 3, skill="repair")
    batch = p.build_batch()
    groups = trajectory_groups(batch)
    assert [[sample.group_id for sample in group] for group in groups] == [
        ["generation-00002-inspection"],
        ["generation-00002-repair", "generation-00002-repair"],
    ]
    signal = resolve_objective("spade").prepare(batch, {})
    assert signal.action == "train" and signal.advantages == (0.0, -1.0, 1.0)
    assert signal.metrics["constant_groups"] == 1, "a skill with one proposal has no relative regret"


def test_generations_per_step_batches_that_many_complete_generations() -> None:
    p = designer_processor(generations_per_step=2)
    for index, score in enumerate((1.0, 0.0)):
        proposed(p, 4, index, score, 2)
    assert not p.ready(), "one complete generation is not a batch of two"
    proposed(p, 5, 0, 0.5, 2)
    assert not p.ready()
    proposed(p, 5, 1, 0.5, 2)
    assert p.ready()
    groups = trajectory_groups(p.build_batch())
    assert [[sample.group_id for sample in group] for group in groups] == [
        ["generation-00004"] * 2,
        ["generation-00005"] * 2,
    ]


@pytest.mark.parametrize(
    ("metadata", "key"),
    [
        ({"proposals": 2}, "generation"),
        ({"generation": 1}, "proposals"),
        ({"generation": 1, "proposals": 0}, "proposals"),
        ({"generation": True, "proposals": 2}, "generation"),
    ],
)
def test_a_designer_report_without_its_generation_or_size_is_refused(metadata: dict[str, object], key: str) -> None:
    p = designer_processor()
    p.ingest(inference("designer-x"))
    with pytest.raises(ValueError, match=rf"requires metadata\.{key}"):
        p.ingest(designer_report("regret-x", "designer-x", 0.5, **metadata))
    p.ingest(inference("designer-y"))
    with pytest.raises(ValueError, match=r"requires metadata\.generation and metadata\.proposals"):
        p.ingest(designer_report("regret-y", "designer-y", 0.5))


def test_a_generations_per_step_that_cannot_train_is_refused() -> None:
    with pytest.raises(ValueError, match="generations_per_step must be positive"):
        designer_processor(generations_per_step=0)


def test_the_objective_centers_regret_within_one_generation() -> None:
    p = designer_processor()
    for index, score in enumerate((0.5, 0.5, -0.5, -0.5)):
        proposed(p, 7, index, score, 4)
    signal = resolve_objective("spade").prepare(p.build_batch(), {})
    assert signal.action == "train" and signal.advantages == (1.0, 1.0, -1.0, -1.0)
    assert signal.metrics["constant_groups"] == 0


def test_the_designer_recipe_binds_its_processor_the_shared_objective_and_its_schedule() -> None:
    spec = SpadeDesignerRecipe.training_spec()
    assert spec.processor is SpadeDesignerProcessor and spec.objective == "spade"
    assert spec.loss_family == "importance_sampling"
    assert spec.scheduling.unit == "sample" and spec.scheduling.batch_size == "actual"
    assert SpadeDesignerRecipe.report_type.fget(SpadeDesignerRecipe) is ScoredRolloutReport  # type: ignore[union-attr]
    runtime = InferenceProxyRuntime(model_path="Qwen/Qwen3-8B", base_url="http://127.0.0.1:8001")
    training_runtime = StubTrainingRuntime()
    recipe = SpadeDesignerRecipe(training_runtime=training_runtime, runtime=runtime)
    assert recipe.name == "spade_designer" and recipe.processor_config() == {"generations_per_step": 1}
    assert SpadeDesignerRecipe(
        training_runtime=training_runtime, runtime=runtime, generations_per_step=3
    ).processor_config() == {"generations_per_step": 3}
    with pytest.raises(ValueError, match="generations_per_step"):
        SpadeDesignerRecipe(training_runtime=training_runtime, runtime=runtime, generations_per_step=0)


# ----------------------------------------------------------------------------------------- the generation half

DOCUMENT = {
    "instruction": (
        "A service on this machine writes the port it listens on under /var/run. Find that file and write the "
        "port number, and nothing else, to /workspace/port.txt."
    ),
    "environment": {
        "Dockerfile": "FROM python:3.12-slim\nRUN apt-get update && apt-get install -y tmux && echo 8471 > /var/run/app.port\nWORKDIR /workspace\n"
    },
    "tests": {
        "test.sh": '#!/bin/sh\nmkdir -p /logs/verifier\ntest "$(cat /workspace/port.txt)" = 8471 && echo 1 > /logs/verifier/reward.txt || echo 0 > /logs/verifier/reward.txt\n'
    },
    "solution": {"solve.sh": "#!/bin/sh\ncat /var/run/app.port > /workspace/port.txt\n"},
    "hint": "Look under /var/run for what the service left behind.",
}


def task_for(name: str, record_id: str, port: int, *, skill: str | None, hint: bool = True) -> HarborTask:
    document = json.loads(json.dumps(DOCUMENT).replace("8471", str(port)))
    solution = dict(document["solution"])
    if hint:
        solution["hint.txt"] = document["hint"] + "\n"
    metadata: dict[str, object] = {"generation": int(name.split("-")[1]), "index": int(name.split("-")[2]), "step": 0}
    if skill is not None:
        metadata["skill"] = skill
    return HarborTask(
        name=name,
        instruction=document["instruction"] + "\n",
        tests=document["tests"],
        environment=document["environment"],
        solution=solution,
        metadata=metadata,
        source_agent_record_ids=(record_id,),
    )


class StandInGenerator(Generator):
    """A generator service in memory: scripted refusals, checks and rewards; tasks written under a root."""

    def __init__(
        self,
        root: Path,
        *,
        refusals: Mapping[int, str] | None = None,
        same_port: bool = False,
        is_solvable: bool = True,
        plain: float = 0.25,
        hint: float = 0.75,
        play_error: str = "",
        fail_after: int | None = None,
        taken: Sequence[str] = (),
        check_fail_after: int | None = None,
    ) -> None:
        self.root = root
        self.refusals = dict(refusals or {})
        self.same_port = same_port
        # Names no write ever takes, the way a directory that is no task of reef's holds one.
        self.taken = set(taken)
        self.is_solvable = is_solvable
        self.rewards = {"plain": plain, "hint": hint}
        self.play_error = play_error
        self.fail_after = fail_after
        self.check_fail_after = check_fail_after
        self.proposals: list[dict[str, object]] = []
        self.checks: list[Path] = []
        self.plays: list[dict[str, object]] = []
        self.reports: list[dict[str, object]] = []
        self.play_reports: list[dict[str, object]] = []
        self.deleted: list[str] = []
        self.manifests: list[dict[str, object]] = []
        # The order of the play and report calls, which says whether a play was held.
        self.sequence: list[str] = []
        self.episodes = 0

    async def propose(
        self, request: DesignerRequest, *, scenario, generation, index, tags, model=None
    ) -> ProposedTask:
        self.proposals.append(
            {
                "request": request,
                "scenario": scenario,
                "generation": generation,
                "index": index,
                "tags": dict(tags),
                "model": model,
            }
        )
        if self.fail_after is not None and len(self.proposals) > self.fail_after:
            raise GeneratorError("the generator went away")
        record_id = f"designer-{len(self.proposals)}"
        if index in self.refusals:
            return ProposedTask(record_id=record_id, task=None, refusal=self.refusals[index])
        name = f"harbor-{generation:05d}-{index:03d}" + (f"-{request.skill}" if request.skill else "")
        port = 8471 if self.same_port else 8471 + len(self.proposals)
        return ProposedTask(record_id=record_id, task=task_for(name, record_id, port, skill=request.skill), refusal="")

    async def report_proposal(self, record_id, *, scenario, score, metadata, feedback=None) -> str:
        self.reports.append(
            {
                "record_id": record_id,
                "scenario": scenario,
                "score": score,
                "metadata": dict(metadata),
                "feedback": feedback,
            }
        )
        return f"report-{len(self.reports)}"

    async def write_task(self, task: HarborTask) -> WrittenTask:
        known = (
            {
                content_hash(read_harbor_task(entry))
                for entry in self.root.iterdir()
                if entry.is_dir() and (entry / "task.toml").is_file()
            }
            if self.root.is_dir()
            else set()
        )
        if content_hash(task) in known:
            raise DuplicateTask("duplicate of a task already under the root")
        if task.name in self.taken:
            raise TaskNameConflict(f"a different task holds the name {task.name}: it is not a task reef wrote")
        try:
            path = write_harbor_task(task, self.root)
        except HarborTaskConflict as exc:
            raise TaskNameConflict(f"a different task holds the name {task.name}: {exc}") from exc
        return WrittenTask(path=path, name=task.name, digest=task.digest)

    async def delete_task(self, name: str) -> None:
        self.deleted.append(name)
        shutil.rmtree(self.root / name, ignore_errors=True)

    async def check(self, task_path: Path) -> OracleResult:
        self.checks.append(task_path)
        if self.check_fail_after is not None and len(self.checks) > self.check_fail_after:
            raise GeneratorError("job 7 failed: OracleUnavailable: the harbor command line is not installed")
        if self.is_solvable:
            return OracleResult(is_solvable=True, reason="", oracle_reward=1.0, nop_reward=0.0)
        return OracleResult(is_solvable=False, reason="the oracle scored 0", oracle_reward=0.0)

    async def play(self, task_path, *, scenario, arm, plays, is_reporting, extra_instruction_files, tags, model=None):
        self.sequence.append("play")
        self.plays.append(
            {
                "task": task_path.name,
                "arm": arm,
                "plays": plays,
                "is_reporting": is_reporting,
                "extra": tuple(extra_instruction_files),
                "tags": dict(tags),
                "model": model,
            }
        )
        reward = None if self.play_error else self.rewards[arm]
        rows = []
        for _ in range(plays):
            self.episodes += 1
            rows.append(
                TaskPlay(
                    task_path=task_path,
                    name=task_path.name,
                    episode_id=f"episode-{self.episodes}",
                    reward=reward,
                    rewards={} if reward is None else {"reward": reward},
                    error=self.play_error,
                    receipts=() if self.play_error else (f"rec-{self.episodes}",),
                    failed_calls=0,
                    report_agent_record_ids=(f"rep-{self.episodes}",) if is_reporting and not self.play_error else (),
                    trial_uri=None,
                    labels={**tags, "arm": arm},
                )
            )
        return tuple(rows)

    async def report_plays(self, plays, *, scenario, score_of, metadata, model=None):
        self.sequence.append("report_plays")
        self.play_reports.append(
            {
                "plays": tuple(plays),
                "scenario": scenario,
                "scores": dict(score_of),
                "metadata": dict(metadata),
                "model": model,
            }
        )
        return tuple(
            (
                replace(play, report_agent_record_ids=(f"rep-{play.episode_id}",))
                if play.episode_id in score_of and play.receipts
                else play
            )
            for play in plays
        )

    async def write_manifest(self, *, generation, names: Sequence[str], eval_fraction, seed) -> Path:
        self.manifests.append(
            {"generation": generation, "names": list(names), "eval_fraction": eval_fraction, "seed": seed}
        )
        split = split_generation(
            [read_harbor_task(self.root / name) for name in names], eval_fraction=eval_fraction, seed=seed
        )
        path = self.root / f"manifest-{generation:05d}.json"
        write_split_manifest(path, split)
        return path


def looked(p: SpadeProcessor) -> bool:
    """Look for a batch twice: the inline worker's outcome lands on the look after the one that submitted it."""
    p.ready()
    return p.ready()


class InlineWorker(JudgingWorker):
    """Runs each generation the moment it is submitted, on this thread, so a test reads the outcome right away."""

    def __init__(self) -> None:
        super().__init__(self.unbound, concurrency=1)
        self.processor: SpadeProcessor | None = None
        self.submitted: list[SupportsReceipt] = []
        self._outcomes: list[SupportsReceipt] = []

    async def unbound(self, job: SupportsReceipt) -> SupportsReceipt:
        raise RuntimeError("no processor bound")

    def submit(self, job: SupportsReceipt) -> bool:
        self.submitted.append(job)
        if self.processor is None:
            return False
        try:
            self._outcomes.append(asyncio.run(self.processor.run_generation(job)))  # type: ignore[arg-type]
        except Exception:
            self._outcomes.append(Failed(job.receipt))
        return True

    def poll(self) -> list[SupportsReceipt]:
        outcomes, self._outcomes = self._outcomes, []
        return outcomes

    def close(self) -> None:
        self.processor = None


class DeferredWorker(JudgingWorker):
    """Takes a generation the moment it is submitted and runs it on ``finish``; the outcome lands on the next poll."""

    def __init__(self) -> None:
        super().__init__(self.unbound, concurrency=1)
        self.processor: SpadeProcessor | None = None
        self.submitted: list[GenerationJob] = []
        self.waiting: list[GenerationJob] = []
        self.finished: list[SupportsReceipt] = []

    async def unbound(self, job: SupportsReceipt) -> SupportsReceipt:
        raise RuntimeError("no processor bound")

    def submit(self, job: SupportsReceipt) -> bool:
        if not isinstance(job, GenerationJob):
            raise TypeError("the generation worker takes GenerationJob")
        self.submitted.append(job)
        self.waiting.append(job)
        return True

    def finish(self) -> None:
        """Run every generation taken so far, on this thread; the next poll hands their outcomes back."""
        if self.processor is None:
            raise RuntimeError("no processor bound")
        jobs, self.waiting = self.waiting, []
        for job in jobs:
            try:
                self.finished.append(asyncio.run(self.processor.run_generation(job)))
            except Exception:
                self.finished.append(Failed(job.receipt))

    def poll(self) -> list[SupportsReceipt]:
        outcomes, self.finished = self.finished, []
        return outcomes

    def close(self) -> None:
        self.processor = None


def generating(
    tmp_path: Path,
    generator: StandInGenerator | None = None,
    *,
    worker: InlineWorker | DeferredWorker | None = None,
    **config: object,
) -> tuple[SpadeProcessor, StandInGenerator]:
    generator = generator if generator is not None else StandInGenerator(tmp_path / "tasks")
    worker = worker if worker is not None else InlineWorker()
    built = SpadeProcessor(
        ProcessorContext(
            "spade",
            {
                "tasks_per_step": 2,
                "rollouts_per_task": 2,
                "generations": 3,
                "count": 3,
                "hint_plays": 1,
                "description": "shell tasks with hidden state under /var and /etc",
                "skills": ("inspection", "repair"),
                "eval_fraction": 0.3,
                "seed": 7,
                "state_dir": str(tmp_path / "state"),
                "served_model": "Qwen/Qwen3-8B",
                **config,
            },
        ),
        generator=generator,
        worker=worker,
    )
    worker.processor = built
    return built, generator


def test_the_first_look_for_a_batch_runs_generation_zero_end_to_end(tmp_path: Path) -> None:
    p, generator = generating(tmp_path)
    assert p.status()["generation"] == {"in_flight": None, "next": 0, "completed": 0, "of": 3, "last_error": ""}
    assert not looked(p)
    assert not p.derivation_pending(), "the inline worker finished before ready() returned"
    assert p.status()["generation"] == {"in_flight": None, "next": 1, "completed": 1, "of": 3, "last_error": ""}

    assert [call["index"] for call in generator.proposals] == [0, 1, 2]
    assert [call["request"].skill for call in generator.proposals] == ["inspection", "repair", "inspection"]
    assert generator.proposals[0]["tags"] == {"role": "designer", "generation": "0", "skill": "inspection"}
    assert generator.proposals[0]["scenario"] == "spade" and generator.proposals[0]["model"] == "Qwen/Qwen3-8B"
    assert "nothing recorded yet" in generator.proposals[0]["request"].experience_text
    assert generator.proposals[0]["request"].target == "shell tasks with hidden state under /var and /etc"

    names = ["harbor-00000-000-inspection", "harbor-00000-001-repair", "harbor-00000-002-inspection"]
    assert generator.checks == [tmp_path / "tasks" / name for name in names]
    arms = [
        (call["task"], call["arm"], call["plays"], call["is_reporting"], call["extra"]) for call in generator.plays
    ]
    assert arms[0] == (names[0], "plain", 2, True, ())
    assert arms[1] == (names[0], "hint", 1, True, ("solution/hint.txt",))
    assert all(call["tags"] == {"generation": "0", "skill": call["task"].split("-")[-1]} for call in generator.plays)

    assert [r["record_id"] for r in generator.reports] == ["designer-1", "designer-2", "designer-3"]
    assert [r["score"] for r in generator.reports] == [0.5, 0.5, 0.5]
    metadata = generator.reports[0]["metadata"]
    assert metadata["task"]["name"] == names[0] and metadata["outcome"] == "frontier" and metadata["regret"] == 0.5
    assert metadata["skill"] == "inspection" and metadata["generation"] == 0 and metadata["proposals"] == 3
    assert generator.reports[0]["feedback"] == {
        "task": names[0],
        "refusal": "",
        "outcome": "frontier",
        "regret": 0.5,
        "return_without_hint": 0.25,
        "return_with_hint": 0.75,
        "round": {"generation": 0, "previous": None},
    }

    assert generator.manifests == [{"generation": 0, "names": names, "eval_fraction": 0.3, "seed": 7}]
    manifest = read_split_manifest(tmp_path / "tasks" / "manifest-00000.json")
    assert sorted([*manifest.train, *manifest.eval]) == names and len(manifest.eval) == 1

    document = json.loads((tmp_path / "state" / "generation-00000.json").read_text())
    assert document["manifest"] == str(tmp_path / "tasks" / "manifest-00000.json") and document["error"] == ""
    assert [task["name"] for task in document["tasks"]] == names and len(document["experience"]) == 3
    assert [proposal["designer_report_id"] for proposal in document["proposals"]] == [
        "report-1",
        "report-2",
        "report-3",
    ]


def test_without_the_option_every_play_reports_as_it_ends(tmp_path: Path) -> None:
    p, generator = generating(tmp_path, skills=())
    assert not looked(p)
    assert all(call["is_reporting"] for call in generator.plays) and generator.play_reports == []
    assert generator.sequence == ["play"] * 6
    document = json.loads((tmp_path / "state" / "generation-00000.json").read_text())
    assert document["held_plays_reported"] is None
    assert [task["held_plays"] for task in document["tasks"]] == [{"plain": 0, "hint": 0}] * 3


def posted(p: SpadeProcessor, play: TaskPlay, metadata: Mapping[str, object]) -> None:
    """The report the task player posts for a held play, ingested after its receipt."""
    p.ingest(inference(play.receipts[0]))
    payload: dict[str, object] = {
        "score": play.reward,
        "feedback": f"verifier reward {play.reward} on {play.name}",
        "references": list(play.receipts),
        "metadata": {
            **metadata,
            "task": {"name": play.name, "path": str(play.task_path), "digest": "ab" * 32},
            "episode": {"id": play.episode_id, "labels": dict(play.labels)},
        },
    }
    p.ingest(
        AgentRecord.create(
            scenario="spade",
            request_type=RequestType.REPORT,
            payload=payload,
            agent_record_id=f"rep-{play.episode_id}",
        )
    )


def test_held_plays_report_as_one_round_once_the_generation_lands_and_train_as_one_batch(tmp_path: Path) -> None:
    p, generator = generating(tmp_path, skills=(), report_plays_after_generation=True)
    assert not looked(p)
    assert generator.sequence == ["play"] * 6 + ["report_plays"] * 2, "nothing is reported until the loop ends"
    assert not any(call["is_reporting"] for call in generator.plays)
    names = ["harbor-00000-000", "harbor-00000-001", "harbor-00000-002"]
    plain, hint = generator.play_reports
    assert plain["scenario"] == "spade" and plain["model"] == "Qwen/Qwen3-8B"
    assert plain["metadata"] == {"arm": "plain", "generation": 0, "round": "generation-00000", "round_plays": 6}
    assert [play.name for play in plain["plays"]] == [name for name in names for _ in range(2)]
    assert all(play.labels == {"generation": "0", "arm": "plain"} for play in plain["plays"])
    assert plain["scores"] == {play.episode_id: 0.25 for play in plain["plays"]}, "the verifier's reward is the score"
    assert hint["metadata"] == {"arm": "hint", "generation": 0}
    assert [play.name for play in hint["plays"]] == names
    assert hint["scores"] == {play.episode_id: 0.75 for play in hint["plays"]}
    assert [r["score"] for r in generator.reports] == [0.5, 0.5, 0.5], "the Designer's regret is measured as before"
    document = json.loads((tmp_path / "state" / "generation-00000.json").read_text())
    assert document["held_plays_reported"] == 9
    assert [task["held_plays"] for task in document["tasks"]] == [{"plain": 2, "hint": 1}] * 3

    for play in hint["plays"]:
        posted(p, play, hint["metadata"])
    assert not p.ready()
    released = p.retention_decision().releasable_agent_record_ids
    assert {play.receipts[0] for play in hint["plays"]} <= released, "the hint reports are released"
    for count, play in enumerate(plain["plays"], 1):
        posted(p, play, plain["metadata"])
        assert p.ready() == (count == 6), "two complete tasks are no batch of two: the round is one unit"
    batch = p.build_batch()
    assert [item.group_id for item in batch.items] == [name for name in names for _ in range(2)]
    assert all(item.metadata["round"] == "generation-00000" for item in batch.items)
    p.acknowledge(batch.batch_id)
    assert not looked(p)
    assert p.status()["generation"]["completed"] == 2, "the round was the one batch generation 1 waited for"
    assert len(generator.play_reports) == 4


def test_the_next_generation_waits_for_batches_per_generation_and_carries_the_experience(tmp_path: Path) -> None:
    p, generator = generating(tmp_path, batches_per_generation=2, skills=())
    assert not looked(p) and p.status()["generation"]["completed"] == 1
    assert len(generator.proposals) == 3
    assert not looked(p), "no batch was acknowledged yet"
    assert len(generator.proposals) == 3
    for task in ("harbor-00000-000", "harbor-00000-001"):
        played(p, task, 0, 1.0)
        played(p, task, 1, 0.0)
    p.acknowledge(p.build_batch().batch_id)
    assert not looked(p) and len(generator.proposals) == 3, "one batch of two is not enough"
    for task in ("harbor-00000-002", "harbor-00000-003"):
        played(p, task, 0, 1.0)
        played(p, task, 1, 0.0)
    p.acknowledge(p.build_batch().batch_id)
    assert not looked(p)
    assert len(generator.proposals) == 6 and p.status()["generation"]["completed"] == 2
    text = generator.proposals[3]["request"].experience_text
    assert "harbor-00000-000: without hint +0.25, with hint +0.75" in text and "Within reach" in text
    assert generator.proposals[3]["tags"] == {"role": "designer", "generation": "1"}
    assert generator.reports[3]["feedback"]["round"] == {
        "generation": 1,
        "previous": {"generation": 0, "mean_regret": 0.5, "measured": 3, "refused": 0},
    }
    assert sorted(entry.name for entry in (tmp_path / "state").iterdir()) == [
        "generation-00000.json",
        "generation-00001.json",
    ]


def test_batches_trained_while_a_generation_runs_count_toward_the_next_one(tmp_path: Path) -> None:
    worker = DeferredWorker()
    p, generator = generating(tmp_path, worker=worker, batches_per_generation=2, skills=())
    assert not p.ready() and [job.generation for job in worker.submitted] == [0]
    assert p.derivation_pending() and p.status()["generation"]["in_flight"] == 0
    for tasks in (("harbor-00000-000", "harbor-00000-001"), ("harbor-00000-002", "harbor-00000-003")):
        for task in tasks:
            played(p, task, 0, 1.0)
            played(p, task, 1, 0.0)
        p.acknowledge(p.build_batch().batch_id)
        assert not p.ready() and [job.generation for job in worker.submitted] == [0], "one generation at a time"
    worker.finish()
    assert not p.ready(), "generation 0 lands on this look and generation 1 starts on it"
    assert [job.generation for job in worker.submitted] == [0, 1]
    assert p.status()["generation"] == {"in_flight": 1, "next": 2, "completed": 1, "of": 3, "last_error": ""}
    worker.finish()
    assert len(generator.proposals) == 6
    assert "harbor-00000-000: without hint +0.25" in generator.proposals[3]["request"].experience_text
    assert not p.ready() and p.status()["generation"]["completed"] == 2
    assert [job.generation for job in worker.submitted] == [0, 1], "no batch trained while generation 1 ran"


def test_a_restarted_processor_carries_on_from_the_reports_on_disk(tmp_path: Path) -> None:
    p, generator = generating(tmp_path, generations=2)
    assert not looked(p) and p.status()["generation"]["completed"] == 1
    p.close()
    again, generator = generating(tmp_path, StandInGenerator(tmp_path / "tasks", same_port=True), generations=2)
    assert again.status()["generation"] == {"in_flight": None, "next": 1, "completed": 1, "of": 2, "last_error": ""}
    assert not looked(again)
    assert generator.proposals[0]["generation"] == 1
    assert "harbor-00000-000-inspection" in generator.proposals[0]["request"].experience_text
    assert generator.reports[0]["feedback"]["round"]["previous"] == {
        "generation": 0,
        "mean_regret": 0.5,
        "measured": 3,
        "refused": 0,
    }, "the last generation's summary is read back from its report on disk"
    assert again.status()["generation"]["completed"] == 2
    assert not looked(again), "the cap is reached"
    assert len(generator.proposals) == 3


def test_refused_proposals_are_reported_at_the_refusal_floor_and_never_stay_under_the_root(tmp_path: Path) -> None:
    generator = StandInGenerator(tmp_path / "tasks", refusals={0: "reply refused: no json"}, same_port=True)
    p, _ = generating(tmp_path, generator, skills=(), count=3, tasks_per_step=1)
    assert not looked(p)
    document = json.loads((tmp_path / "state" / "generation-00000.json").read_text())
    refusals = [proposal["refusal"] for proposal in document["proposals"]]
    assert refusals == ["reply refused: no json", "", "refused: duplicate of a task already under the root"]
    assert [r["score"] for r in generator.reports] == [REFUSAL_SCORE, 0.5, REFUSAL_SCORE] and REFUSAL_SCORE == -1.0
    assert generator.reports[0]["metadata"]["refusal"] == "reply refused: no json"
    assert generator.reports[0]["metadata"]["proposals"] == 3 and "task" not in generator.reports[0]["metadata"]
    assert generator.reports[0]["feedback"] == {
        "task": None,
        "refusal": "reply refused: no json",
        "outcome": None,
        "regret": None,
        "return_without_hint": None,
        "return_with_hint": None,
        "round": {"generation": 0, "previous": None},
    }
    played(p, "harbor-00000-001", 0, 1.0)
    played(p, "harbor-00000-001", 1, 0.0)
    p.acknowledge(p.build_batch().batch_id)
    assert not looked(p), "a batch trained, so generation 1 runs and its reports carry generation 0's summary"
    assert generator.reports[3]["feedback"]["round"]["previous"] == {
        "generation": 0,
        "mean_regret": 0.5,
        "measured": 1,
        "refused": 2,
    }
    assert [task["name"] for task in document["tasks"]] == ["harbor-00000-001"]
    assert sorted(e.name for e in (tmp_path / "tasks").iterdir() if e.name != ".staging") == [
        "harbor-00000-001",
        "manifest-00000.json",
    ]


def test_a_rerun_generation_replaces_the_tasks_its_earlier_attempt_wrote(tmp_path: Path) -> None:
    root = tmp_path / "tasks"
    earlier = task_for("harbor-00000-000", "designer-earlier", 1, skill=None)
    write_harbor_task(earlier, root)
    p, generator = generating(tmp_path, StandInGenerator(root), skills=())
    assert not looked(p)
    assert generator.deleted == ["harbor-00000-000"]
    assert read_harbor_task(root / "harbor-00000-000").digest != earlier.digest
    document = json.loads((tmp_path / "state" / "generation-00000.json").read_text())
    assert [task["name"] for task in document["tasks"]] == ["harbor-00000-000", "harbor-00000-001", "harbor-00000-002"]
    assert [proposal["refusal"] for proposal in document["proposals"]] == ["", "", ""] and document["error"] == ""

    kept = StandInGenerator(tmp_path / "again" / "tasks", taken=("harbor-00000-000",))
    p, generator = generating(tmp_path / "again", kept, skills=())
    assert not looked(p)
    assert generator.deleted == ["harbor-00000-000"], "replaced once; a second conflict is a refusal"
    document = json.loads((tmp_path / "again" / "state" / "generation-00000.json").read_text())
    assert document["proposals"][0]["refusal"].startswith("refused: a different task holds the name harbor-00000-000")
    assert [task["name"] for task in document["tasks"]] == ["harbor-00000-001", "harbor-00000-002"]
    assert [r["score"] for r in generator.reports] == [REFUSAL_SCORE, 0.5, 0.5] and document["error"] == ""


def test_a_task_the_oracle_refuses_or_the_agent_cannot_play_is_removed(tmp_path: Path) -> None:
    refusing = StandInGenerator(tmp_path / "tasks", is_solvable=False)
    p, generator = generating(tmp_path, refusing, count=1, generations=1)
    assert not looked(p)
    assert generator.deleted == ["harbor-00000-000-inspection"] and generator.plays == []
    assert generator.reports[0]["score"] == REFUSAL_SCORE
    assert generator.reports[0]["metadata"]["refusal"] == "oracle check refused: the oracle scored 0"
    assert generator.manifests == []

    unplayable = StandInGenerator(tmp_path / "tasks2", play_error="Failed to start tmux session. Error: None")
    p, _ = generating(tmp_path / "second", unplayable, count=1, generations=1)
    assert not looked(p)
    assert [call["arm"] for call in unplayable.plays] == [
        "plain"
    ], "the hint arm is not played for a task that cannot run"
    assert unplayable.deleted == ["harbor-00000-000-inspection"]
    assert unplayable.reports[0]["metadata"]["refusal"].startswith("the Reasoning Agent could not play the task")


def test_a_task_whose_hint_hurt_is_reported_with_its_negative_regret_above_a_refusal(tmp_path: Path) -> None:
    hurting = StandInGenerator(tmp_path / "tasks", plain=0.75, hint=0.25)
    p, generator = generating(tmp_path, hurting, count=1, generations=1, skills=())
    assert not looked(p)
    report = generator.reports[0]
    assert report["score"] == -0.5 and report["metadata"]["regret"] == -0.5, "the regret is reported as measured"
    assert report["feedback"]["regret"] == -0.5 and report["feedback"]["outcome"] == "frontier"
    assert report["score"] > REFUSAL_SCORE, "a refusal stays below a task whose hint hurt"
    document = json.loads((tmp_path / "state" / "generation-00000.json").read_text())
    assert document["tasks"][0]["regret"] == -0.5


def test_a_dropped_batch_does_not_count_toward_the_next_generation(tmp_path: Path) -> None:
    worker = DeferredWorker()
    p, _ = generating(tmp_path, worker=worker, skills=())
    assert not p.ready() and [job.generation for job in worker.submitted] == [0]
    for task in ("harbor-00000-000", "harbor-00000-001"):
        played(p, task, 0, 1.0)
        played(p, task, 1, 0.0)
    batch_id = p.build_batch().batch_id
    p.dropped(batch_id)
    p.acknowledge(batch_id)
    worker.finish()
    assert not p.ready(), "generation 0 lands; the dropped batch trained nothing"
    assert [job.generation for job in worker.submitted] == [0]
    for task in ("harbor-00000-002", "harbor-00000-003"):
        played(p, task, 0, 1.0)
        played(p, task, 1, 0.0)
    p.acknowledge(p.build_batch().batch_id)
    assert not p.ready(), "a trained batch counts"
    assert [job.generation for job in worker.submitted] == [0, 1]


def test_a_generation_that_measured_no_task_does_not_block_the_next_one(tmp_path: Path) -> None:
    worker = DeferredWorker()
    refusing = StandInGenerator(tmp_path / "tasks", is_solvable=False)
    p, generator = generating(tmp_path, refusing, worker=worker, count=1)
    assert not p.ready() and [job.generation for job in worker.submitted] == [0]
    worker.finish()
    assert not p.ready(), "generation 0 lands without a task and generation 1 starts on the same look"
    assert [job.generation for job in worker.submitted] == [0, 1], "no batch can come; none was acknowledged"
    assert p.status()["generation"] == {"in_flight": 1, "next": 2, "completed": 1, "of": 3, "last_error": ""}
    assert generator.deleted == ["harbor-00000-000-inspection"] and generator.manifests == []


def test_a_generator_that_goes_away_ends_the_generation_with_what_ran(tmp_path: Path) -> None:
    p, generator = generating(tmp_path, StandInGenerator(tmp_path / "tasks", fail_after=1), skills=())
    assert not looked(p)
    status = p.status()["generation"]
    assert status["completed"] == 1 and status["last_error"] == "the generator went away"
    document = json.loads((tmp_path / "state" / "generation-00000.json").read_text())
    assert [task["name"] for task in document["tasks"]] == ["harbor-00000-000"] and document[
        "error"
    ] == "the generator went away"
    assert generator.manifests == [], "the manifest is written after the last proposal"


def test_a_check_that_could_not_run_ends_the_generation_with_the_error_and_no_score_for_that_proposal(
    tmp_path: Path,
) -> None:
    generator = StandInGenerator(tmp_path / "tasks", check_fail_after=1)
    p, _ = generating(tmp_path, generator, skills=())
    assert not looked(p)
    error = "job 7 failed: OracleUnavailable: the harbor command line is not installed"
    status = p.status()["generation"]
    assert status["completed"] == 1 and status["last_error"] == error
    assert len(generator.proposals) == 2 and len(generator.checks) == 2, "the loop stops at the check that failed"
    assert [(r["record_id"], r["score"]) for r in generator.reports] == [("designer-1", 0.5)]
    assert [call["task"] for call in generator.plays] == ["harbor-00000-000"] * 2, "the unchecked task is not played"
    document = json.loads((tmp_path / "state" / "generation-00000.json").read_text())
    assert document["error"] == error and [task["name"] for task in document["tasks"]] == ["harbor-00000-000"]
    assert [proposal["refusal"] for proposal in document["proposals"]] == [""], "an unchecked task is no refusal"
    assert generator.manifests == []


def test_a_generation_that_never_finishes_is_named_in_the_status(tmp_path: Path) -> None:
    p, _ = generating(tmp_path)
    p.close()
    assert not looked(p)
    assert "no further generation runs" in p.status()["generation"]["last_error"]


def test_the_real_worker_runs_a_generation_off_the_trainers_thread(tmp_path: Path) -> None:
    generator = StandInGenerator(tmp_path / "tasks")
    p = SpadeProcessor(
        ProcessorContext(
            "spade",
            {
                "generations": 1,
                "count": 1,
                "description": "shell tasks",
                "state_dir": str(tmp_path / "state"),
                "generator_url": "http://127.0.0.1:1",
            },
        ),
        generator=generator,
    )
    try:
        assert not p.ready()
        assert p.derivation_pending() and p.status()["generation"]["in_flight"] == 0
        deadline = time.monotonic() + 10.0
        while p.derivation_pending() and time.monotonic() < deadline:
            time.sleep(0.02)
            p.ready()
        assert not p.derivation_pending() and p.status()["generation"]["completed"] == 1
        assert (tmp_path / "state" / "generation-00000.json").is_file()
        assert p.operational_metrics()["generations_completed"] == 1
    finally:
        p.close()


def turn(record_id: str, tokens: list[int], loss_mask: list[int]) -> AgentRecord:
    payload = {
        "messages": [{"role": "user", "content": "ls"}],
        "response": {"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        "training": {
            "tokens": tokens,
            "loss_mask": loss_mask,
            "rollout_log_probs": [-0.1] * sum(loss_mask),
            "runtime_load_id": "r1",
        },
    }
    return AgentRecord.create(
        scenario="spade", request_type=RequestType.INFERENCE, payload=payload, agent_record_id=record_id
    )


def forked_episode(processor: SpadeProcessor, task: str, tag: str, **metadata: object) -> None:
    """Two turns whose second prompt does not extend the first: no sample can be built from them."""
    processor.ingest(turn(f"{tag}-a", [1, 2, 3], [0, 1, 1]))
    processor.ingest(turn(f"{tag}-b", [9, 9, 9, 4], [0, 0, 0, 1]))
    payload: dict[str, object] = {
        "score": 1.0,
        "feedback": "verifier reward 1.0",
        "references": [f"{tag}-a", f"{tag}-b"],
        "metadata": {"task": {"name": task, "path": f"/tasks/{task}", "digest": "ab" * 32}, **metadata},
    }
    processor.ingest(
        AgentRecord.create(
            scenario="spade", request_type=RequestType.REPORT, payload=payload, agent_record_id=f"{tag}-rep"
        )
    )


def test_an_episode_that_cannot_be_assembled_is_given_up_and_still_counts_toward_its_task_group() -> None:
    p = processor(tasks_per_step=1, rollouts_per_task=2)
    played(p, "harbor-00000-000", 0, 1.0)
    forked_episode(p, "harbor-00000-000", "fork")
    assert p.status()["unassembled_episodes"] == 1
    assert p.ready(), "one sample plus one given up episode complete a group of two"
    batch = p.build_batch()
    assert [sample.group_id for sample in batch.items] == ["harbor-00000-000"]
    releasable = p.retention_decision().releasable_agent_record_ids
    assert {"fork-a", "fork-b", "fork-rep"} <= releasable, "the given up episode and its report are released"


def test_a_given_up_episode_counts_toward_its_round_unit() -> None:
    p = processor(tasks_per_step=4, rollouts_per_task=3)
    p.ingest(inference("rec-0"))
    p.ingest(report("rep-0", "rec-0", "harbor-00003-000", 1.0, round="generation-00003", round_plays=2))
    assert not p.ready()
    forked_episode(p, "harbor-00003-001", "fork", round="generation-00003", round_plays=2)
    assert p.ready(), "the round's two plays arrived, one of them given up"
    assert [sample.group_id for sample in p.build_batch().items] == ["harbor-00003-000"]


def test_the_designer_processor_keeps_the_engines_default_and_retries_an_unassembled_report() -> None:
    p = designer_processor()
    p.ingest(turn("d-a", [1, 2, 3], [0, 1, 1]))
    p.ingest(turn("d-b", [9, 9, 9, 4], [0, 0, 0, 1]))
    payload: dict[str, object] = {
        "score": 0.5,
        "feedback": "SPADE Designer regret",
        "references": ["d-a", "d-b"],
        "metadata": {"generation": 1, "proposals": 1},
    }
    report_record = AgentRecord.create(
        scenario="spade", request_type=RequestType.REPORT, payload=payload, agent_record_id="d-rep"
    )
    with pytest.raises(ValueError, match="cannot assemble"):
        p.ingest(report_record)
    assert "d-rep" in p.retention_decision().protected_agent_record_ids, "retained for the next drain"
