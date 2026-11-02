"""Run the pinned Terminal-Bench search: one Reef training request per step, each step a durable scenario commit."""

from __future__ import annotations

import argparse
import hashlib
import json
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
from reef.dispatcher import Dispatcher
from reef.evaluation.harness import History
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
#: The text of each step's training request, which the proposer reads as its instruction.
STEP_REQUEST = "Propose the next candidate of the search."
#: The steps a run takes unless --iterations says otherwise, as in the comparison: four full-history iterations.
DEFAULT_STEPS = 4


def configuration(tasks: tuple[str, ...], repeats: int | None) -> dict[str, Any]:
    """The deployment's recipe sections with the seed path, the selected tasks and the repeats filled in."""
    sections = recipe_config_from_file_layout(load_config(HERE / "terminal_bench.yaml"))
    evolution = sections["evolution"]
    evaluation = evolution["evaluation"]
    evolution["seed"] = str(SEED)
    evaluation["tasks"] = list(tasks)
    if repeats is not None:
        evaluation["repeats"] = repeats
    return sections


def campaign_id(sections: Mapping[str, Any]) -> str:
    """Different tasks, models, seeds, or search settings must never share search state; more steps continue one."""
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
    cycle would commit the same step a second time.
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


def search(dispatcher: Dispatcher, scenario: Scenario, recipe: HarnessEvolutionRecipe, directory: Path, steps: int):
    """Run the search until ``steps`` steps ended: queue one training request per step and commit its job.

    Each request has the ID of its step. A restarted process queues the request of an unfinished step again, which
    returns the request already queued, and ``commit_job`` finishes a job that a previous process started.
    """
    summary = write_summary(scenario, recipe, directory)
    while steps_done(scenario) < steps:
        step = steps_done(scenario) + 1
        dispatcher.accept_instruction(
            scenario.name, {"text": STEP_REQUEST, "session": "", "release_id": ""}, request_id=f"step-{step}"
        )
        if not commit_job(scenario):
            raise RuntimeError(f"the training request of step {step} started no job")
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
    parser.add_argument(
        "--iterations", type=positive_int, default=DEFAULT_STEPS, help="the steps the search takes in all"
    )
    parser.add_argument("--repeats", type=positive_int)
    parser.add_argument(
        "--dry-run", action="store_true", help="validate tasks and render the seed without model calls"
    )
    args = parser.parse_args()
    tasks = benchmark.task_paths(args.tasks_root, args.task)
    sections = configuration(tasks, args.repeats)
    directory = Path(os.environ["REEF_WORK"]).resolve() / campaign_id(sections)
    files = render_tree(HarnessTree.read(SEED), get_adapter("terminus"))
    print(
        json.dumps(
            {
                "directory": str(directory),
                "tasks": len(tasks),
                "repeats": sections["evolution"]["evaluation"]["repeats"],
                "steps": args.iterations,
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
        search(dispatcher, scenario, recipe, directory, args.iterations)
    finally:
        dispatcher.close()


if __name__ == "__main__":
    main()
