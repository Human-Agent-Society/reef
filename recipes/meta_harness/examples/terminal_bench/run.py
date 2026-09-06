"""Run the Terminal-Bench comparison through Reef's durable scenario lifecycle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from recipes.meta_harness.examples.terminal_bench.campaign import (
    CAMPAIGN_STATE_KEY,
    TerminalBenchRecipe,
    score_episode,
)
from recipes.meta_harness.examples.terminal_bench.journal_storage import CompressedDispatcher
from recipes.meta_harness.examples.terminal_bench.runtime import runtime_fingerprint
from recipes.meta_harness.examples.terminal_bench.tasks import parse_tasks, read_tasks
from recipes.meta_harness.recipe import _UnboundProposer
from reef.artifact import GitLFSRepositoryBackend
from reef.core import AgentRecord, RequestType
from reef.harness.executor import LocalExecutor
from reef.harness.model_binding import ModelBinding
from reef.harness.terminus.runner import pinned_task
from reef.runtime.adapters.inference_proxy import InferenceProxyRuntime
from reef.train.cordis_backend.strategies import resolve_episode_scorer

SCENARIO = "terminal-bench-meta-harness"


def code_fingerprint():
    root = Path(__file__).resolve().parents[4]
    paths = sorted(
        {
            *root.joinpath("recipes/meta_harness").glob("*.py"),
            *root.joinpath("recipes/meta_harness/examples/terminal_bench").glob("*.py"),
            *root.joinpath("reef/harness/terminus").glob("*.py"),
            *root.joinpath("reef/artifact").glob("*.py"),
            root / "reef/train/trainer.py",
            root / "reef/scenario/commit_protocol.py",
            root / "reef/scenario/factory.py",
        }
    )
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def make_recipe(arguments, *, fingerprint=None):
    tasks = read_tasks(arguments.tasks_file) if arguments.tasks_file else parse_tasks(arguments.tasks or "")
    if not tasks:
        raise ValueError("name at least one Terminal-Bench task")
    for name in ("trials", "iterations", "concurrency", "max_attempts"):
        if getattr(arguments, name) < 1:
            raise ValueError(f"{name} must be positive")
    if not math.isfinite(arguments.episode_timeout_s) or arguments.episode_timeout_s < 14400:
        raise ValueError("episode timeout must be at least 14400 seconds to preserve task phase budgets")
    if not math.isfinite(arguments.max_observed_cost_usd) or arguments.max_observed_cost_usd <= 0:
        raise ValueError("cost cap must be finite and positive")
    if arguments.concurrency > 32:
        raise ValueError("concurrency must fit the shared 32-sandbox project allocation")
    if len(set(tasks)) != len(tasks):
        raise ValueError("task IDs must be unique; use --trials for repeats")
    for task in tasks:
        pinned_task(task, "69671fbaac6d67a7ef0dfec016cc38a64ef7a77c")
    plan = {
        "tasks": list(tasks),
        "trials": arguments.trials,
        "iterations": arguments.iterations,
        "concurrency": arguments.concurrency,
        "max_attempts": arguments.max_attempts,
        "episode_timeout_s": arguments.episode_timeout_s,
        "mode": arguments.mode,
        "target_model": arguments.target_model,
        "target_url": arguments.target_url,
        "proposer_model": arguments.proposer_model,
        "proposer_url": arguments.proposer_url,
        "proposer_effort": arguments.proposer_effort,
        "max_observed_cost_usd": arguments.max_observed_cost_usd,
        "runtime": fingerprint if fingerprint is not None else runtime_fingerprint(),
        "code": code_fingerprint(),
        "sandbox": "e2b",
        "adapter_attempts": 1,
        "dataset_commit": "69671fbaac6d67a7ef0dfec016cc38a64ef7a77c",
    }
    executor = LocalExecutor()
    kinds = ("rules", "skill", "agent_command")
    seed = ()
    if getattr(arguments, "agent_failure_policy", None):
        from .agent_failure_policy import SUPPORTED

        if arguments.agent_failure_policy not in SUPPORTED:
            raise ValueError("unknown agent failure scoring policy")
        plan["agent_failure_policy"] = arguments.agent_failure_policy
    if getattr(arguments, "benchmark_sandbox_limit", None) is not None:
        if type(arguments.benchmark_sandbox_limit) is not int or arguments.benchmark_sandbox_limit not in (32, 64):
            raise ValueError("benchmark sandbox allocation must be 32 or 64")
        plan["benchmark_sandbox_limit"] = arguments.benchmark_sandbox_limit
    if getattr(arguments, "episode_schedule", None):
        plan["episode_schedule"] = arguments.episode_schedule
    if getattr(arguments, "proposer_api", None):
        plan["proposer_api"] = arguments.proposer_api
    if getattr(arguments, "proposer_timeout_s", None) is not None:
        if not math.isfinite(arguments.proposer_timeout_s) or arguments.proposer_timeout_s <= 0:
            raise ValueError("proposer time allowance must be finite and positive")
        plan["proposer_timeout_s"] = arguments.proposer_timeout_s
    if getattr(arguments, "max_proposer_cost_usd", None) is not None:
        from .proposer_usage import freeze_pricing

        if not math.isfinite(arguments.max_proposer_cost_usd) or arguments.max_proposer_cost_usd <= 0:
            raise ValueError("proposer spend cap must be finite and positive")
        plan["max_proposer_cost_usd"] = arguments.max_proposer_cost_usd
        plan["proposer_pricing"] = freeze_pricing(arguments.proposer_model)
    if getattr(arguments, "e2b_runtime_receipt", None):
        from .e2b_executor import E2BEpisodeExecutor

        receipt = json.loads(Path(arguments.e2b_runtime_receipt).read_text())
        plan["runner"] = {key: receipt[key] for key in ("snapshot_id", "manifest_sha256")}
        if receipt.get("verifier_compat"):
            plan["runner"]["verifier_compat"] = receipt["verifier_compat"]
        executor = E2BEpisodeExecutor(**plan["runner"])
        executor.preflight()
        if arguments.concurrency > 16:
            raise ValueError("each E2B runner also needs a task sandbox; concurrency must be at most 16")
        if getattr(arguments, "executable_harness", False):
            kinds = ("code_extension",)
            from .runtime_source import source_bundle, source_manifest

            plan["proposer_source_manifest"] = source_manifest(source_bundle())
        plan["candidate_kinds"] = list(kinds)
    elif getattr(arguments, "executable_harness", False):
        raise ValueError("executable candidates require --e2b-runtime-receipt")
    if getattr(arguments, "seed_agent", None):
        if not getattr(arguments, "executable_harness", False):
            raise ValueError("a Python seed requires --executable-harness")
        code = Path(arguments.seed_agent).read_text()
        seed = ({"id": "python-harness", "name": "code_extension", "config": {"name": "candidate", "code": code}},)
        plan["seed_entries"] = list(seed)
    key = os.environ.get("OPENAI_API_KEY")
    return TerminalBenchRecipe(
        propose=_UnboundProposer(),
        score_episode=resolve_episode_scorer(score_episode),
        tasks=tuple(tasks),
        adapter="terminus",
        episode_repeats=arguments.trials,
        episode_workers=arguments.concurrency,
        episode_timeout_s=arguments.episode_timeout_s,
        executor=executor,
        runtime=InferenceProxyRuntime(model_path=arguments.target_model, base_url=arguments.target_url, api_key=key),
        models={
            "proposer": ModelBinding(
                base_url=arguments.proposer_url,
                model=arguments.proposer_model,
                api_key=key,
                api=plan.get("proposer_api", "openai"),
            )
        },
        archive_dir=Path(arguments.output_dir).resolve(),
        output=Path(arguments.output_dir).resolve(),
        plan=plan,
        mode=arguments.mode,
        kinds=kinds,
        seed=seed,
        batch_policy="records",
        batch_size=1,
    )


def advance(scenario):
    # Scheduling records contain no search plan or scores: the backend chooses
    # work from its committed state. A recovered pending trigger is idempotent.
    identity = f"campaign-step-{scenario.scenario_step + 1}"
    scenario.records.append_result(
        AgentRecord.create(
            scenario=scenario.name,
            request_type=RequestType.INFERENCE,
            payload={"messages": [{"role": "user", "content": "Advance the committed campaign."}]},
            agent_record_id=identity,
        )
    )
    result = scenario.prepare_training_step()
    if result is None:
        raise RuntimeError("campaign scheduling record did not produce a step")
    scenario.commit(result)
    return scenario.trainer.state[CAMPAIGN_STATE_KEY]


def baseline_ready(data):
    """Recognize the committed boundary before any proposer reservation."""
    return (
        data["status"] == "running"
        and not data["unknown_usage"]
        and data["evaluation"] is None
        and data["reservation"] is None
        and len(data["rounds"]) == 1
        and data["rounds"][0]["kind"] == "baseline"
        and not data["proposals"]
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    tasks = parser.add_mutually_exclusive_group(required=True)
    tasks.add_argument("--tasks")
    tasks.add_argument("--tasks-file")
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--benchmark-sandbox-limit", type=int, choices=(32, 64), default=32)
    from .agent_failure_policy import SUPPORTED

    parser.add_argument("--agent-failure-policy", choices=SUPPORTED)
    parser.add_argument(
        "--episode-schedule",
        choices=("rolling", "waves"),
        default="rolling",
        help="Commit completed trials and refill available workers",
    )
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--episode-timeout-s", type=float, default=28800)
    parser.add_argument("--max-observed-cost-usd", type=float, required=True)
    parser.add_argument(
        "--max-proposer-cost-usd",
        type=float,
        default=20,
        help="API proposer sub-cap included in the total observed cap",
    )
    parser.add_argument("--mode", choices=("full_history", "incumbent_only"), default="full_history")
    parser.add_argument("--target-model", default="gpt-5.6-luna")
    parser.add_argument("--target-url", default="https://api.openai.com")
    parser.add_argument("--proposer-model", default="gpt-5.6-sol")
    parser.add_argument("--proposer-url", default="https://api.openai.com")
    parser.add_argument("--proposer-effort", default="xhigh")
    parser.add_argument("--proposer-api", choices=("responses", "openai"), default="responses")
    parser.add_argument("--proposer-timeout-s", type=float, default=2400)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--e2b-runtime-receipt", help="Prepared E2B runner snapshot receipt, frozen into the plan")
    parser.add_argument(
        "--executable-harness", action="store_true", help="Evolve one self-contained Python Agent module"
    )
    parser.add_argument("--seed-agent", type=Path, help="Freeze a Python Agent module as the evaluation seed")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--stop-after-baseline",
        action="store_true",
        help="Stop successfully at the committed baseline; omit on resume to continue the frozen plan",
    )
    arguments = parser.parse_args(argv)
    recipe = make_recipe(arguments)
    if arguments.dry_run:
        print(json.dumps(recipe.plan, indent=2))
        return 0
    if not os.environ.get("OPENAI_API_KEY") or not os.environ.get("E2B_API_KEY"):
        parser.error("OPENAI_API_KEY and E2B_API_KEY are required for a live run")
    output = recipe.output.resolve()
    if output.exists() and any(output.iterdir()) and not (output / "records").exists():
        parser.error("this output directory is not a durable campaign; preserve it and use a new directory")
    factory = GitLFSRepositoryBackend.factory(
        output / "artifacts.git", work_dir=output / "artifact-work", cache_dir=output / "artifact-cache"
    )
    dispatcher = CompressedDispatcher(recipe, factory, agent_record_dir=output / "records")
    try:
        scenario = dispatcher.get_or_create_scenario(SCENARIO)
        data = scenario.trainer.state[CAMPAIGN_STATE_KEY]
        while data["status"] == "running":
            if arguments.stop_after_baseline and baseline_ready(data):
                print(
                    json.dumps({"status": "baseline_ready", "observed_cost_usd": data["observed_cost_usd"]}),
                    flush=True,
                )
                return 0
            data = advance(scenario)
            print(
                json.dumps(
                    {
                        "step": scenario.scenario_step,
                        "status": data["status"],
                        "observed_cost_usd": data["observed_cost_usd"],
                        "rounds": data["rounds"],
                    }
                ),
                flush=True,
            )
        return 0 if data["status"] == "complete" else 2
    finally:
        dispatcher.close()


if __name__ == "__main__":
    raise SystemExit(main())
