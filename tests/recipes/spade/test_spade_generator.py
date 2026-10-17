"""SPADE as a task generator: the Designer's generations, the hint arm, and the regret reports from the results."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from tests.support.task_service_stubs import StandInGenerator, task_for

from recipes.beta.spade import SpadeTaskGenerator
from reef.core import AgentRecord, RequestType
from reef.core.plugins import PluginConfigError
from reef.core.tasks import PlayArm, TaskGenerationRequest, TaskPlayResult, read_harbor_task, write_harbor_task
from reef.pipeline_context import PipelineContext
from reef.recipe.generated_tasks import GeneratedTaskTasks
from reef.recipe.generated_tasks.processor import (
    GeneratedTaskRecord2Record,
    RoundJob,
    RoundSettings,
    TaskGroupSettings,
)
from reef.record2task.generator import TaskNameConflict

DESCRIPTION = "shell tasks with hidden state under /var and /etc"
REQUEST = TaskGenerationRequest((), DESCRIPTION)


def spade(tmp_path: Path, service: StandInGenerator | None = None, **fields: Any) -> SpadeTaskGenerator:
    return SpadeTaskGenerator(
        scenario="spade",
        service=service if service is not None else StandInGenerator(tmp_path / "tasks"),
        state_dir=tmp_path / "state",
        **{
            "count": 3,
            "hint_plays": 1,
            "skills": ("inspection", "repair"),
            "served_model": "Qwen/Qwen3-8B",
            **fields,
        },
    )


def played(generator: SpadeTaskGenerator, *, plain: float = 0.25, hint: float = 0.75) -> tuple[TaskPlayResult, ...]:
    """Generate, validate, write and play every task the way the generated-task recipe does, with fixed rewards."""
    results = []
    for item in asyncio.run(generator.generate(REQUEST)):
        validation = asyncio.run(generator.validate(item.task))
        if not validation.is_valid:
            results.append(TaskPlayResult(item.task, error="; ".join(validation.errors)))
            continue
        asyncio.run(generator.service.write_task(item.task))
        measurement_rewards = {arm.name: (hint,) * arm.count for arm in item.measurement_arms}
        results.append(TaskPlayResult(item.task, (plain, plain), measurement_rewards))
    return tuple(results)


def report(tmp_path: Path, generation: int) -> dict[str, Any]:
    return json.loads((tmp_path / "state" / f"generation-{generation:05d}.json").read_text())


def test_a_generation_asks_the_designer_count_times_across_the_skills_and_asks_for_the_hint_arm(
    tmp_path: Path,
) -> None:
    service = StandInGenerator(tmp_path / "tasks")
    tasks = asyncio.run(spade(tmp_path, service).generate(REQUEST))
    assert [call["index"] for call in service.proposals] == [0, 1, 2]
    assert [call["request"].skill for call in service.proposals] == ["inspection", "repair", "inspection"]
    assert service.proposals[0]["tags"] == {"role": "designer", "generation": "0", "skill": "inspection"}
    assert service.proposals[0]["scenario"] == "spade" and service.proposals[0]["model"] == "Qwen/Qwen3-8B"
    assert "nothing recorded yet" in service.proposals[0]["request"].experience_text
    assert service.proposals[0]["request"].target == DESCRIPTION
    assert [item.task.name for item in tasks] == [
        "harbor-00000-000-inspection",
        "harbor-00000-001-repair",
        "harbor-00000-002-inspection",
    ]
    assert all(item.measurement_arms == (PlayArm("hint", 1, ("solution/hint.txt",)),) for item in tasks)
    assert asyncio.run(spade(tmp_path / "off", hint_plays=0).generate(REQUEST))[0].measurement_arms == ()


def test_validation_checks_a_written_copy_and_removes_it(tmp_path: Path) -> None:
    service = StandInGenerator(tmp_path / "tasks")
    generator = spade(tmp_path, service, count=1)
    (item,) = asyncio.run(generator.generate(REQUEST))
    assert asyncio.run(generator.validate(item.task)).is_valid
    assert service.checks == [tmp_path / "tasks" / item.task.name]
    assert service.deleted == [item.task.name] and not (tmp_path / "tasks" / item.task.name).exists()

    refusing = StandInGenerator(tmp_path / "refusing", is_solvable=False)
    generator = spade(tmp_path / "second", refusing, count=1)
    (item,) = asyncio.run(generator.generate(REQUEST))
    assert asyncio.run(generator.validate(item.task)).errors == ("oracle check refused: the oracle scored 0",)
    assert refusing.deleted == [item.task.name]


def test_a_duplicate_is_invalid_and_a_rerun_replaces_the_task_its_earlier_attempt_wrote(tmp_path: Path) -> None:
    root = tmp_path / "tasks"
    earlier = task_for("harbor-00000-000", "designer-earlier", 1)
    write_harbor_task(earlier, root)
    service = StandInGenerator(root, same_port=True)
    generator = spade(tmp_path, service, skills=(), count=2)
    first, second = asyncio.run(generator.generate(REQUEST))
    assert asyncio.run(generator.validate(first.task)).is_valid
    assert service.deleted == ["harbor-00000-000", "harbor-00000-000"], "the earlier attempt, then the checked copy"
    write_harbor_task(first.task, root)
    assert asyncio.run(generator.validate(second.task)).errors == (
        "refused: duplicate of a task already under the root",
    )

    taken = StandInGenerator(tmp_path / "taken", taken=("harbor-00000-000",))
    generator = spade(tmp_path / "again", taken, skills=(), count=1)
    (item,) = asyncio.run(generator.generate(REQUEST))
    errors = asyncio.run(generator.validate(item.task)).errors
    assert errors[0].startswith("refused: a different task holds the name harbor-00000-000")
    assert taken.deleted == ["harbor-00000-000"], "replaced once; a second conflict is a refusal"


def test_a_history_task_never_replaces_a_conflicting_name(tmp_path: Path) -> None:
    service = StandInGenerator(tmp_path / "tasks", taken=("harbor-history-1",))
    task = task_for("harbor-history-1", "designer-1", 1)
    history = type(task)(**{**task.__dict__, "metadata": {"designer_record_id": "designer-1"}})
    with pytest.raises(TaskNameConflict):
        asyncio.run(spade(tmp_path, service).write(history))
    assert service.deleted == []


def test_the_results_close_the_generation_with_regret_reports_and_the_next_experience(
    tmp_path: Path,
) -> None:
    service = StandInGenerator(tmp_path / "tasks", refusals={1: "reply refused: no json"})
    generator = spade(tmp_path, service)
    asyncio.run(generator.record_results(played(generator)))

    assert [(r["record_id"], r["score"]) for r in service.reports] == [
        ("designer-1", 0.5),
        ("designer-2", 0.0),
        ("designer-3", 0.5),
    ]
    metadata = service.reports[0]["metadata"]
    assert metadata["task"]["name"] == "harbor-00000-000-inspection" and metadata["outcome"] == "frontier"
    assert metadata["regret"] == 0.5 and metadata["skill"] == "inspection" and metadata["generation"] == 0
    assert service.reports[1]["metadata"]["refusal"] == "reply refused: no json"
    names = ["harbor-00000-000-inspection", "harbor-00000-002-inspection"]
    assert service.manifests == [], "the task store owns the split manifest"

    document = report(tmp_path, 0)
    assert document["error"] == "" and [task["name"] for task in document["tasks"]] == names
    assert [proposal["refusal"] for proposal in document["proposals"]] == ["", "reply refused: no json", ""]
    assert [proposal["designer_report_id"] for proposal in document["proposals"]] == [
        "report-1",
        "report-2",
        "report-3",
    ]

    asyncio.run(generator.generate(REQUEST))
    text = service.proposals[3]["request"].experience_text
    assert "harbor-00000-000-inspection (inspection): without hint +0.25, with hint +0.75" in text
    assert service.proposals[3]["tags"]["generation"] == "1"


def test_an_unplayed_task_scores_zero_with_its_error(tmp_path: Path) -> None:
    service = StandInGenerator(tmp_path / "tasks")
    generator = spade(tmp_path, service, count=1)
    (item,) = asyncio.run(generator.generate(REQUEST))
    asyncio.run(generator.record_results((TaskPlayResult(item.task, error="the agent could not play the task"),)))
    assert service.reports[0]["score"] == 0.0
    assert service.reports[0]["metadata"]["refusal"] == "the agent could not play the task"
    assert service.manifests == [] and report(tmp_path, 0)["tasks"] == []


@pytest.mark.unit
def test_eval_tasks_receive_no_designer_reward_or_training_experience(tmp_path: Path) -> None:
    service = StandInGenerator(tmp_path / "tasks")
    generator = spade(tmp_path, service, count=1)
    (generated,) = asyncio.run(generator.generate(REQUEST))
    assert asyncio.run(generator.validate(generated.task)).is_valid
    asyncio.run(generator.record_results((TaskPlayResult(task=generated.task, split="eval"),)))
    assert service.reports == []
    assert service.manifests == []
    assert generator.experience == ()
    document = report(tmp_path, 0)
    assert document["tasks"] == []
    assert document["proposals"][0]["split"] == "eval"
    assert document["proposals"][0]["refusal"] == ""


def test_a_restarted_generator_carries_on_from_the_reports_on_disk(tmp_path: Path) -> None:
    generator = spade(tmp_path, count=1)
    asyncio.run(generator.record_results(played(generator)))
    service = StandInGenerator(tmp_path / "tasks")
    again = spade(tmp_path, service, count=1)
    assert again.generation == 1
    asyncio.run(again.generate(REQUEST))
    assert service.proposals[0]["generation"] == 1
    assert "harbor-00000-000-inspection" in service.proposals[0]["request"].experience_text


def test_a_service_that_goes_away_ends_the_generation_with_what_ran(tmp_path: Path) -> None:
    service = StandInGenerator(tmp_path / "tasks", fail_after=1)
    generator = spade(tmp_path, service, skills=())
    results = played(generator)
    assert [result.task.name for result in results] == ["harbor-00000-000"]
    asyncio.run(generator.record_results(results))
    document = report(tmp_path, 0)
    assert document["error"] == "the generator went away" and [t["name"] for t in document["tasks"]] == [
        "harbor-00000-000"
    ]
    assert service.manifests == [], "a generation that stopped writes no manifest"


def test_a_check_that_could_not_run_raises_and_is_named_in_the_report(tmp_path: Path) -> None:
    service = StandInGenerator(tmp_path / "tasks", check_fail_after=0)
    generator = spade(tmp_path, service, count=1)
    (item,) = asyncio.run(generator.generate(REQUEST))
    with pytest.raises(Exception, match="the harbor command line is not installed"):
        asyncio.run(generator.validate(item.task))
    asyncio.run(generator.record_results(()))
    document = report(tmp_path, 0)
    assert "OracleUnavailable" in document["error"] and document["proposals"] == []
    assert service.reports == [], "an unchecked task is no refusal"


def test_history_from_another_scenario_is_refused_before_the_designer_call(tmp_path: Path) -> None:
    service = StandInGenerator(tmp_path / "tasks")
    source = AgentRecord.create(scenario="other", request_type=RequestType.INFERENCE, payload={})
    with pytest.raises(ValueError, match="generator's scenario"):
        asyncio.run(spade(tmp_path, service).generate(TaskGenerationRequest((source,), "rebuild")))
    assert service.proposals == []


def test_from_config_reads_its_section_and_keeps_one_state_directory_for_each_scenario(tmp_path: Path) -> None:
    generator = SpadeTaskGenerator.from_config(
        {
            "generator_url": "http://127.0.0.1:8910",
            "state_dir": str(tmp_path),
            "count": 2,
            "skills": "inspection, repair",
        },
        scenario="one",
        environ={},
    )
    assert generator.state_dir == tmp_path / "one" and generator.count == 2
    assert generator.skills == ("inspection", "repair") and generator.hint_plays == 2
    for config, message in (
        ({"state_dir": "s"}, "generator_url"),
        ({"generator_url": "${endpoints.generator}", "state_dir": "s"}, "generator_url"),
        ({"generator_url": "http://g"}, "state_dir"),
        ({"generator_url": "http://g", "state_dir": "s", "rollouts": 2}, "does not take rollouts"),
        ({"generator_url": "http://g", "state_dir": "s", "count": 0}, "count must be positive"),
    ):
        with pytest.raises(PluginConfigError, match=message):
            SpadeTaskGenerator.from_config(config, scenario="one", environ={})


def test_a_round_of_the_generated_task_recipe_runs_spade_end_to_end(tmp_path: Path) -> None:
    service = StandInGenerator(tmp_path / "tasks", refusals={1: "reply refused: no json"})
    generator = spade(tmp_path, service)
    context = PipelineContext("spade")
    generation = GeneratedTaskTasks(
        context,
        RoundSettings(rounds=1, description=DESCRIPTION, state_dir=str(tmp_path / "rounds")),
        GeneratedTaskRecord2Record(context, TaskGroupSettings(plays_per_task=2)),
        task_generator=generator,
        task_player=service,
    )
    outcome = asyncio.run(generation.run_round(RoundJob(receipt="round-00000", round_number=0)))
    generation.close()
    assert outcome.error == "" and [result.is_played for result in outcome.results] == [True, True]
    plays = [(call["task"], call["arm"], call["plays"], call["extra"]) for call in service.plays]
    assert plays[:2] == [
        ("harbor-00000-000-inspection", "plain", 2, ()),
        ("harbor-00000-000-inspection", "hint", 1, ("solution/hint.txt",)),
    ]
    assert [(r["record_id"], r["score"]) for r in service.reports] == [
        ("designer-1", 0.5),
        ("designer-2", 0.0),
        ("designer-3", 0.5),
    ]
    assert service.manifests == [] and report(tmp_path, 0)["error"] == ""
    assert (tmp_path / "rounds" / "spade" / "store" / "split-manifest.json").is_file()
    assert read_harbor_task(tmp_path / "tasks" / "harbor-00000-000-inspection").name == "harbor-00000-000-inspection"
