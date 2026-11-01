"""Drive the pinned Terminal-Bench search through Reef scenario commits."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import shutil
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from harness import benchmark
from recipes.meta_harness import MetaHarness
from recipes.meta_harness.method import search_population
from reef.artifact import GitLFSRepositoryBackend
from reef.core import AgentRecord, RequestType
from reef.dispatcher import Dispatcher
from reef.evaluation.harness import History
from reef.harness import run_task
from reef.harness.adapters import get_adapter
from reef.harness.tree import HarnessTree, render_tree
from reef.inference.http import InferenceProxyRuntime
from reef.recipe.config import recipe_config_from_file_layout
from reef.recipe.harness import HarnessEvolutionRecipe
from reef.recipe.registry import build_recipe
from reef.scenario.release_files import component_files
from reef.scenario.scenario import Scenario
from reef.service.deploy.config_utils import load_config
from reef.storage.sql_coordination import open_coordination
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.train.harness_backend.component import HISTORY_DIR
from reef.train.jobs import JobLoopState

HERE = Path(__file__).resolve().parent
SCENARIO = "terminal-bench"
SEED = HERE / "seed"


def configuration(tasks: tuple[str, ...], iterations: int | None, repeats: int | None) -> dict[str, Any]:
    """Fill the stack's dataset and keep the evaluation run budget consistent with it."""
    sections = recipe_config_from_file_layout(load_config(HERE / "terminal_bench.yaml"))
    evolution = sections["evolution"]
    evaluation = evolution["evaluation"]
    evolution["seed"] = str(SEED)
    evaluation["tasks"] = list(tasks)
    if repeats is not None:
        evaluation["repeats"] = repeats
    method = evolution["method"]["config"]
    if iterations is not None:
        method["max_candidates"] = iterations
        evolution["max_steps"] = 2 * iterations
    # The cache keeps the incumbent's scores: the seed runs once, then each candidate once.
    method["max_target_runs"] = len(tasks) * evaluation["repeats"] * (method["max_candidates"] + 1)
    return sections


def campaign_id(sections: Mapping[str, Any]) -> str:
    """Different tasks, models, seeds, or budgets must never share search state."""
    identity = json.loads(json.dumps(sections))
    identity["evolution"]["seed"] = HarnessTree.read(SEED).checksum()
    identity["target_url"] = os.environ["REEF_UPSTREAM_URL"]
    identity["target_model"] = os.environ["REEF_MODEL"]
    identity["dataset_revision"] = benchmark.MANIFEST["revision"]
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def load_recipe(sections: Mapping[str, Any], directory: Path) -> HarnessEvolutionRecipe:
    """Build the recipe with deployment credentials, never candidate credentials; its history lives in ``directory``."""
    config = json.loads(json.dumps(sections))
    config["evolution"]["state_dir"] = str(directory / "harness")
    runtime = InferenceProxyRuntime(
        model_path=os.environ["REEF_MODEL"],
        base_url=os.environ["REEF_UPSTREAM_URL"],
        api_key=os.environ.get("REEF_UPSTREAM_API_KEY") or None,
        inference_timeout_s=config["evolution"]["evaluation"]["task_timeout_s"],
    )
    recipe = build_recipe(config["implementation"], config=config, runtime=runtime)
    if not isinstance(recipe, HarnessEvolutionRecipe) or not isinstance(recipe.method, MetaHarness):
        raise TypeError("the example runs the Meta-Harness method")
    return recipe


def open_campaign(recipe: HarnessEvolutionRecipe, directory: Path) -> Dispatcher:
    """Use persistent artifacts and SQLite records so a new process can resume.

    The recipe's seed is the root release of a new scenario, as the service makes it, so the search starts there.

    ``search`` prepares and commits every step itself, so the dispatcher runs no training cycle of its own: a
    cycle that its preload starts would commit the same replayed step a second time.
    """
    return Dispatcher(
        recipe,
        GitLFSRepositoryBackend.factory(
            directory / "artifacts.git",
            work_dir=directory / "artifact-work",
            cache_dir=directory / "artifact-cache",
        ),
        scenario_storage=SQLiteScenarioStorage(directory / "reef-data"),
        hold_local_cycles=True,
        # The jobs queue is on disk, so a restarted campaign finds the job of its step.
        coordination=open_coordination(
            "sqlite", agent_record_dir=str(directory / "reef-data"), record_database_url=None
        ),
    )


def steps_done(scenario: Scenario) -> int:
    """The harness steps that ended: each step is one training job of the scenario's trainer."""
    return JobLoopState.from_dict(scenario.trainer.state).finished_jobs


def search_history(scenario: Scenario, recipe: HarnessEvolutionRecipe) -> History:
    """The search history that the harness training component keeps for the scenario, as its last step left it."""
    root = recipe.harness_state_dir(scenario.name, scenario.surface.files_component or "harness")
    if root is None:
        raise TypeError("the example's scenario evolves a harness")
    return History.open(root / HISTORY_DIR)


def commit_job(scenario: Scenario) -> bool:
    """Commit the scenario's next training job, or the one a previous process started, from its submission to its
    end; False when no job starts. The dispatcher's training service runs the job meanwhile."""
    trainer = scenario.trainer
    if not trainer.job_running:
        submitted = scenario.prepare_training_step()
        if submitted is None:
            return False
        scenario.commit(submitted)
    while trainer.job_running:
        result = scenario.prepare_training_step()
        if result is None:
            time.sleep(1.0)
            continue
        scenario.commit(result)
    return True


def served_tree(scenario: Scenario) -> HarnessTree:
    """The head release's source tree: the tree the next step builds on."""
    artifact = scenario.repository.resolve(scenario.repository.require_current_artifact())
    return HarnessTree(component_files(scenario.surface, artifact, scenario.surface.files_component or "harness"))


def record_run(scenario: Scenario, recipe: HarnessEvolutionRecipe) -> None:
    """Record one real served-harness run and its verifier feedback.

    Stable ids let a restart finish an interrupted report without rerunning an
    already recorded run. The next task is a function of the committed step.
    """
    if recipe.scorer is None:
        raise ValueError("the example scores its runs with evolution.evaluation.scorer, which is not set")
    step = steps_done(scenario)
    tasks = recipe.configured_tasks()
    task = tasks[step % len(tasks)]
    inference_id = f"terminal-bench-run-{step}"
    inference = scenario.records.get(scenario.name, inference_id)
    if inference is None:
        descriptor = get_adapter(recipe.evolution.adapter)
        tree = served_tree(scenario)
        files = render_tree(tree, descriptor, recipe.model_binding().config_overlays(descriptor))
        result = run_task(
            descriptor,
            files,
            task,
            binary=recipe.evolution.binary,
            timeout=recipe.evaluation.task_timeout_s,
            executor=recipe.executor,
        )
        littered = recipe.evaluation.forbid_residue and bool(result.residue)
        score = 0.0 if littered else float(recipe.scorer.score(task, result))
        if not math.isfinite(score):
            raise ValueError("run scorer returned a non-finite score")
        inference = scenario.records.append(
            AgentRecord.create(
                scenario=scenario.name,
                agent_record_id=inference_id,
                request_type=RequestType.INFERENCE,
                payload={
                    "source": "terminus_run",
                    "task": task,
                    "tree": dict(tree.files),
                    "run": dataclasses.asdict(result),
                    "score": score,
                },
            )
        )
    scenario.records.append(
        AgentRecord.create(
            scenario=scenario.name,
            agent_record_id=f"terminal-bench-report-{step}",
            request_type=RequestType.REPORT,
            payload={
                "score": inference.payload["score"],
                "feedback": {"task": task, "source": "Harbor verifier"},
                "references": [inference.agent_record_id],
            },
        )
    )


def write_summary(scenario: Scenario, recipe: HarnessEvolutionRecipe, directory: Path) -> dict[str, Any]:
    """Rebuild a disposable summary from the latest durable commit and the history of the steps it covers."""
    done = steps_done(scenario)
    if not done:
        return {}
    step = done + 1
    history = search_history(scenario, recipe)
    head = served_tree(scenario).checksum()
    members = search_population(history, head, step, recipe.configured_tasks())
    summary = {
        "scenario": scenario.name,
        "step": done,
        "served_id": head,
        "generated_candidates": sum(member.record.origin == "method" for member in members),
        "evaluation_runs": sum(task_run.step < step for task_run in history.runs),
        "candidates": [
            {
                "id": member.record.tree,
                "parent_id": member.parent,
                "outcome": member.outcome,
                "scores": [task_run.score.to_json() for task_run in member.runs],
                "mean": member.mean_score,
            }
            for member in members
        ],
    }
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(directory / "summary.json")
    return summary


def search(scenario: Scenario, recipe: HarnessEvolutionRecipe, directory: Path) -> dict[str, Any]:
    """Resume pending feedback, then request bounded proposal and evaluation steps."""
    method = recipe.method
    if not isinstance(method, MetaHarness):
        raise TypeError("the example drives the Meta-Harness method")
    summary = write_summary(scenario, recipe, directory)
    while steps_done(scenario) < recipe.evolution.max_steps:
        # The method's own budget check, before a run is spent on a step that would propose nothing.
        if method.spent_budget(search_history(scenario, recipe), steps_done(scenario) + 1) is not None:
            break
        # A previous process may have recorded the report but not committed its
        # step, or started its job. Let Reef replay it before spending on another run.
        scenario.produce_samples()
        if not commit_job(scenario):
            record_run(scenario, recipe)
            scenario.produce_samples()
            if not commit_job(scenario):
                raise RuntimeError("the recorded run did not produce a training step")
        summary = write_summary(scenario, recipe, directory)
        print(
            json.dumps({key: summary[key] for key in ("step", "generated_candidates", "evaluation_runs", "served_id")})
        )
    return summary


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks-root", type=Path, default=Path(os.environ.get("REEF_TERMINAL_BENCH_DIR", "/opt/terminal-bench-2"))
    )
    parser.add_argument("--task", action="append", help="run a smaller suite using names from the pinned manifest")
    parser.add_argument("--iterations", type=positive_int)
    parser.add_argument("--repeats", type=positive_int)
    parser.add_argument(
        "--dry-run", action="store_true", help="validate tasks and render the seed without model calls"
    )
    args = parser.parse_args()
    tasks = benchmark.task_paths(args.tasks_root, args.task)
    sections = configuration(tasks, args.iterations, args.repeats)
    directory = Path(os.environ["REEF_WORK"]).resolve() / campaign_id(sections)
    descriptor = get_adapter("terminus")
    files = render_tree(HarnessTree.read(SEED), descriptor)
    evolution = sections["evolution"]
    method = evolution["method"]["config"]
    print(
        json.dumps(
            {
                "directory": str(directory),
                "tasks": len(tasks),
                "repeats": evolution["evaluation"]["repeats"],
                "candidate_limit": method["max_candidates"],
                "evaluation_run_limit": method["max_target_runs"],
                "run_limit": evolution["max_steps"],
                "seed_files": sorted(files),
            },
            indent=2,
        )
    )
    if args.dry_run:
        return
    if not all(Path(task).is_relative_to("/usr") or Path(task).is_relative_to("/opt") for task in tasks):
        raise ValueError("place the task checkout under /usr or /opt so Reef's sandbox can read it")
    if shutil.which("reef-terminus") is None:
        raise RuntimeError("reef-terminus is missing; install Reef and this example in the same environment")
    recipe = load_recipe(sections, directory)
    dispatcher = open_campaign(recipe, directory)
    try:
        scenario = dispatcher.get_or_create_scenario(SCENARIO)
        if scenario is None:
            raise RuntimeError("Reef did not create the campaign scenario")
        search(scenario, recipe, directory)
    finally:
        dispatcher.close()


if __name__ == "__main__":
    main()
