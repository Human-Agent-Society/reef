"""Drive the pinned Terminal-Bench search through Reef scenario commits."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from harness import benchmark
from recipes.meta_harness.population import Population, population_for_step
from recipes.meta_harness.proposer import MetaHarnessProposer
from recipes.meta_harness.recipe import MetaHarnessRecipe
from reef.artifact import GitLFSRepositoryBackend
from reef.core import AgentRecord, RequestType
from reef.dispatcher import Dispatcher
from reef.harness import run_episode
from reef.harness.adapters import get_adapter
from reef.harness.tree import HarnessTree, render_tree
from reef.inference.http import InferenceProxyRuntime
from reef.recipe.config import recipe_config_from_file_layout
from reef.recipe.registry import build_recipe
from reef.scenario.release_files import component_files
from reef.scenario.scenario import Scenario
from reef.service.deploy.config_utils import load_config
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.train.harness_backend.state import HarnessState

HERE = Path(__file__).resolve().parent
SCENARIO = "terminal-bench"
SEED = HERE / "seed"


def configuration(tasks: tuple[str, ...], iterations: int | None, repeats: int | None) -> dict[str, Any]:
    """Fill the stack's dataset and keep the evaluation episode budget consistent with it."""
    sections = recipe_config_from_file_layout(load_config(HERE / "terminal_bench.yaml"))
    evolution = sections["evolution"]
    evolution["seed"] = str(SEED)
    evolution["tasks"] = list(tasks)
    if repeats is not None:
        evolution["episode_repeats"] = repeats
    method = evolution["meta_harness"]
    if iterations is not None:
        method["max_candidates"] = iterations
        evolution["max_steps"] = 2 * iterations
    method["max_target_episodes"] = 2 * len(tasks) * evolution["episode_repeats"] * method["max_candidates"]
    return sections


def campaign_id(sections: Mapping[str, Any]) -> str:
    """Different tasks, models, seeds, or budgets must never share search state."""
    identity = json.loads(json.dumps(sections))
    identity["evolution"]["seed"] = HarnessTree.read(SEED).checksum()
    identity["target_url"] = os.environ["REEF_UPSTREAM_URL"]
    identity["target_model"] = os.environ["REEF_MODEL"]
    identity["dataset_revision"] = benchmark.MANIFEST["revision"]
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def load_recipe(sections: Mapping[str, Any]) -> MetaHarnessRecipe:
    """Build the shared recipe with deployment credentials, never candidate credentials."""
    config = json.loads(json.dumps(sections))
    runtime = InferenceProxyRuntime(
        model_path=os.environ["REEF_MODEL"],
        base_url=os.environ["REEF_UPSTREAM_URL"],
        api_key=os.environ.get("REEF_UPSTREAM_API_KEY") or None,
        inference_timeout_s=config["evolution"]["episode_timeout_s"],
    )
    recipe = build_recipe(config["implementation"], config=config, runtime=runtime)
    if not isinstance(recipe, MetaHarnessRecipe):
        raise TypeError("the example requires MetaHarnessRecipe")
    return recipe


def open_campaign(recipe: MetaHarnessRecipe, directory: Path) -> Dispatcher:
    """Use persistent artifacts and SQLite records so a new process can resume.

    The recipe's seed is the base release of a new scenario, as the service makes it, so the search starts there.

    ``search`` prepares and commits every step itself, so the dispatcher runs no training cycle of its own: a
    cycle that its preload starts would commit the same replayed step a second time.
    """
    return Dispatcher(
        recipe,
        GitLFSRepositoryBackend.factory(
            directory / "artifacts.git",
            work_dir=directory / "artifact-work",
            cache_dir=directory / "artifact-cache",
            bootstrap_files=recipe.base_artifact_files() or {},
        ),
        scenario_storage=SQLiteScenarioStorage(directory / "reef-data"),
        hold_local_cycles=True,
    )


def committed_population(algorithm_state: Mapping[str, Any] | None) -> Population:
    """The population a committed step left, with the evaluation it recorded folded in."""
    state = HarnessState() if algorithm_state is None else HarnessState.from_json(algorithm_state)
    return population_for_step(state.method_state, state.last_outcome)


def population(scenario: Scenario) -> Population:
    """The search state as the scenario's trainer holds it after its latest commit."""
    return committed_population(scenario.trainer.state)


def served_tree(scenario: Scenario) -> HarnessTree:
    """The head release's source tree: the tree the next step builds on."""
    artifact = scenario.repository.resolve(scenario.repository.require_current_artifact())
    return HarnessTree(component_files(scenario.surface, artifact, scenario.surface.files_component or "harness"))


def record_rollout(scenario: Scenario, recipe: MetaHarnessRecipe) -> None:
    """Record one real served-harness episode and its verifier feedback.

    Stable ids let a restart finish an interrupted report without rerunning an
    already recorded episode. The next task is a function of the committed step.
    """
    step = scenario.scenario_step
    task = recipe.tasks[step % len(recipe.tasks)]
    inference_id = f"terminal-bench-rollout-{step}"
    inference = scenario.records.get(scenario.name, inference_id)
    if inference is None:
        descriptor = get_adapter(recipe.adapter)
        tree = served_tree(scenario)
        files = render_tree(tree, descriptor, recipe.model_binding().config_overlays(descriptor))
        result = run_episode(
            descriptor,
            files,
            task,
            binary=recipe.binary,
            timeout=recipe.episode_timeout_s,
            executor=recipe.executor,
        )
        score = 0.0 if recipe.forbid_residue and result.residue else float(recipe.score_episode.score(task, result))
        if not math.isfinite(score):
            raise ValueError("rollout scorer returned a non-finite score")
        inference = scenario.records.append(
            AgentRecord.create(
                scenario=scenario.name,
                agent_record_id=inference_id,
                request_type=RequestType.INFERENCE,
                payload={
                    "source": "terminus_episode",
                    "task": task,
                    "tree": dict(tree.files),
                    "episode": dataclasses.asdict(result),
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


def write_summary(scenario: Scenario, directory: Path) -> dict[str, Any]:
    """Rebuild a disposable summary solely from the latest durable commit."""
    history = scenario.store.history()
    if not history:
        return {}
    committed = history[-1]
    search = committed_population(committed.algorithm_state)
    summary = {
        "scenario": scenario.name,
        "step": committed.step,
        "served_id": search.served_id,
        "generated_candidates": search.generated_candidates,
        "proposer_calls": search.proposer_calls,
        "evaluation_episodes": search.episode_calls,
        "candidates": [
            {
                "id": candidate.candidate_id,
                "parent_id": candidate.parent_id,
                "scores": candidate.scores,
                "mean": None if candidate.scores is None else sum(candidate.scores) / len(candidate.scores),
            }
            for candidate in search.candidates
        ],
    }
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(directory / "summary.json")
    return summary


def search(scenario: Scenario, recipe: MetaHarnessRecipe, directory: Path) -> dict[str, Any]:
    """Resume pending feedback, then request bounded proposal/evaluation steps."""
    summary = write_summary(scenario, directory)
    if not isinstance(recipe.propose, MetaHarnessProposer):
        raise TypeError("the example drives the Meta-Harness proposer")
    settings = recipe.propose.settings
    while scenario.scenario_step < recipe.max_steps:
        committed = population(scenario)
        if committed.generated_candidates >= settings.max_candidates:
            break
        if committed.episode_calls + settings.evaluation_cost > settings.max_target_episodes:
            break
        # A previous process may have recorded the report but not committed its
        # step. Let Reef replay it before spending on another rollout.
        result = scenario.prepare_training_step()
        if result is None:
            record_rollout(scenario, recipe)
            result = scenario.prepare_training_step()
        if result is None:
            raise RuntimeError("the recorded rollout did not produce a training step")
        scenario.commit(result)
        summary = write_summary(scenario, directory)
        print(
            json.dumps(
                {key: summary[key] for key in ("step", "generated_candidates", "evaluation_episodes", "served_id")}
            )
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
    print(
        json.dumps(
            {
                "directory": str(directory),
                "tasks": len(tasks),
                "repeats": evolution["episode_repeats"],
                "candidate_limit": evolution["meta_harness"]["max_candidates"],
                "evaluation_episode_limit": evolution["meta_harness"]["max_target_episodes"],
                "rollout_limit": evolution["max_steps"],
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
    recipe = load_recipe(sections)
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
