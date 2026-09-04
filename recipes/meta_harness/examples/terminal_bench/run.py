"""Drive the Reef arm of the Terminal-Bench 2 reproduction.

Runs the Meta-Harness search directly over ``MetaHarnessBackend``: propose,
evaluate the candidate against the incumbent, settle, repeat. Reef's Trainer
does this in production; here the loop is explicit so an experiment can be
resumed, capped, and inspected.

The seed is evaluated before the first proposal. Upstream always seeds its
frontier from the baselines ("Always seed frontier from baselines if results
exist"), and against an empty frontier its comparison is ``avg > -1``, so the
first candidate would win unconditionally. Reef's paired gate measures the
incumbent alongside every candidate, which supplies the same number, but only
if step one actually runs before anything is judged.

    OPENAI_API_KEY=... E2B_API_KEY=... \
        python -m recipes.meta_harness.examples.terminal_bench.run \
            --tasks extract-elf,password-recovery --trials 1 --iterations 2 \
            --max-observed-cost-usd 5 --output-dir /tmp/tb2-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from reef.harness.adapters import get_adapter
from reef.harness.executor import LocalExecutor
from reef.harness.model_binding import ModelBinding, ModelBindings
from reef.train.evaluation import DefaultCandidateEvaluationPlugin
from reef.train.types import TraceBatch

from recipes.meta_harness.backend import POPULATION_STATE_KEY, MetaHarnessBackend
from recipes.meta_harness.method import MetaHarnessProposer, MetaHarnessSelector
from recipes.meta_harness.population import PopulationStore

from .budget import ObservedCostLedger, SpendCapReached
from .tasks import parse_tasks, read_tasks

#: The seed is the empty tree, which the terminus adapter renders as stock
#: Terminus 2. That is upstream's `baseline_terminus2`, so both arms start
#: from the same agent.
SEED: tuple = ()


class EpisodeScorer:
    """Reads the verifier's reward, and charges what the episode cost.

    The cost has to be taken here because the episode root is deleted once
    ``run_episode`` returns: the trial record the runner wrote is the last
    place it exists, and it reaches this scorer on the trajectory.
    """

    def __init__(self, ledger: ObservedCostLedger) -> None:
        self._ledger = ledger
        self._episode = 0

    def __call__(self, task: str, result: Any) -> float:
        del task  # the reward is per-episode; the task is already on the record
        self._episode += 1
        for event in getattr(result, "trajectory", ()) or ():
            if event.get("type") != "verifier":
                continue
            cost = event.get("cost_usd")
            if isinstance(cost, (int, float)):
                self._ledger.record_trial(f"episode-{self._episode}", float(cost))
            reward = event.get("reward")
            return 0.0 if reward is None else float(reward)
        return 0.0


def build(arguments: argparse.Namespace, ledger: ObservedCostLedger) -> tuple[MetaHarnessBackend, PopulationStore]:
    output = Path(arguments.output_dir)
    store = PopulationStore(output / "population.json")
    descriptor = get_adapter("terminus")
    tasks = read_tasks(arguments.tasks_file) if arguments.tasks_file else parse_tasks(arguments.tasks or "")
    if not tasks:
        raise SystemExit("--tasks must name at least one Terminal-Bench task")

    served = ModelBinding(
        base_url=arguments.target_url,
        model=arguments.target_model,
        api_key=os.environ["OPENAI_API_KEY"],
    )
    proposer_binding = ModelBinding(
        base_url=arguments.proposer_url,
        model=arguments.proposer_model,
        api_key=os.environ["OPENAI_API_KEY"],
    )
    proposer = MetaHarnessProposer(
        store=store,
        descriptor=descriptor,
        tasks=tasks,
        episode_repeats=arguments.trials,
        mode=arguments.mode,
        kinds=("rules", "skill", "agent_command"),
        max_candidates=arguments.iterations,
    )
    backend = MetaHarnessBackend(
        population_store=store,
        descriptor=descriptor,
        propose=proposer,
        score_episode=EpisodeScorer(ledger),
        tasks=tasks,
        models=ModelBindings(served=served, named={"proposer": proposer_binding}),
        episode_repeats=arguments.trials,
        episode_timeout_s=arguments.episode_timeout_s,
        # Pairings run in one pool. Without this the evaluation is sequential,
        # and 600 episodes at two minutes each is a day rather than an hour.
        episode_workers=arguments.concurrency,
        executor=LocalExecutor(),
        seed=SEED,
    )
    return backend, store


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tb2-reef-arm", description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--tasks", help="comma-separated Harbor task ids")
    source.add_argument("--tasks-file", help="file of Harbor task ids, one per line or comma-separated")
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument(
        "--max-episode-failure-rate",
        type=float,
        default=0.5,
        help="stop when this share of an iteration's episodes produce no score at all",
    )
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--mode", default="full_history")
    parser.add_argument("--target-model", default="gpt-5.6-luna")
    parser.add_argument("--target-url", default="https://api.openai.com")
    parser.add_argument("--proposer-model", default="gpt-5.6-sol")
    parser.add_argument("--proposer-url", default="https://api.openai.com")
    parser.add_argument("--episode-timeout-s", type=float, default=1800.0)
    parser.add_argument("--concurrency", type=int, default=8, help="pairings evaluated in parallel")
    parser.add_argument("--max-observed-cost-usd", type=float, required=True)
    parser.add_argument("--output-dir", required=True)
    arguments = parser.parse_args(argv)

    output = Path(arguments.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ledger = ObservedCostLedger(output / "observed-cost.json", arguments.max_observed_cost_usd)
    backend, store = build(arguments, ledger)

    state: dict[str, Any] = dict(backend.initial_state())
    log = (output / "iterations.jsonl").open("a", encoding="utf-8")
    evaluator = DefaultCandidateEvaluationPlugin(backend, MetaHarnessSelector(store))
    search(
        backend,
        evaluator,
        ledger,
        state,
        arguments.iterations,
        log,
        arguments.max_episode_failure_rate,
    )
    log.close()
    population = state.get(POPULATION_STATE_KEY, {})
    (output / "final-population.json").write_text(json.dumps(population, indent=2, sort_keys=True) + "\n")
    print(f"served candidate: {population.get('served_id')}", flush=True)
    return 0


def search(
    backend: Any,
    evaluator: Any,
    ledger: ObservedCostLedger,
    state: dict[str, Any],
    iterations: int,
    log: Any,
    max_episode_failure_rate: float = 1.0,
) -> dict[str, Any]:
    """Propose, evaluate, settle and commit, once per iteration.

    ``state`` is updated in place so a caller keeps the last committed value
    even when the loop stops early.
    """
    accounted_before, free_before = ledger.trial_tally()
    for iteration in range(1, iterations + 1):
        try:
            ledger.before_trial(f"iteration-{iteration}")
        except SpendCapReached as exc:
            print(f"stopping: {exc}", flush=True)
            break
        started = time.time()
        prepared = backend.prepare_step(TraceBatch(batch_id=f"iteration-{iteration}", samples=()), state, iteration)
        if prepared.outcome == "skip":
            print(f"iteration {iteration}: no proposal ({prepared.metrics})", flush=True)
            state.clear()
            state.update(prepared.state)
            continue
        try:
            evaluation = evaluator.evaluate(prepared.candidate)
            decision = evaluator.decide(prepared.candidate, evaluation)
            result = backend.settle_step(prepared, decision)
        except BaseException:
            backend.abort_step(prepared)
            raise
        # settle_step returns speculative state and aborts the staged
        # population; commit_applied is what installs it. Reef's scenario
        # commit calls that once its durable record exists, and this loop
        # stands in for the scenario, so it has to call it too. Without this
        # the next iteration's begin() sees state that never reached commit
        # and refuses to continue - correctly.
        backend.commit_applied(result.state)
        state.clear()
        state.update(result.state)
        # An episode that produced no score never ran. Nothing else in this row
        # distinguishes a weak candidate from an outage -- a full sandbox quota,
        # a provider refusing every call -- so a run that scored near zero
        # because nothing executed reads back as a search result.
        failures = int(evaluation.metrics.get("episode_failures", 0) or 0)
        episodes = len(evaluation.metrics.get("candidate_scores", ())) + len(
            evaluation.metrics.get("current_scores", ())
        )
        # An episode can produce a score without running: the adapter exits
        # before it reaches the model and the scorer reads a reward of zero.
        # `episode_failures` only counts a missing score, so that case is
        # invisible to it -- two iterations here reported zero failures while
        # claiming 120 episodes in 74 seconds. Cost is the ground truth for
        # whether an episode called the model at all.
        accounted, free = ledger.trial_tally()
        ran = max(0, accounted - accounted_before)
        free_now = max(0, free - free_before)
        accounted_before, free_before = accounted, free
        unmeasured = failures + free_now
        failure_rate = unmeasured / episodes if episodes else 0.0
        row = {
            "iteration": iteration,
            "selected": bool(decision.selected),
            "metrics": {k: v for k, v in (decision.metrics or {}).items() if isinstance(v, (int, float, str))},
            "episodes": episodes,
            "episode_failures": failures,
            "episodes_accounted": ran,
            "episodes_without_cost": free_now,
            "seconds": round(time.time() - started, 1),
        }
        log.write(json.dumps(row, sort_keys=True) + "\n")
        log.flush()
        print(json.dumps(row, sort_keys=True), flush=True)
        if failure_rate > max_episode_failure_rate:
            raise SystemExit(
                f"iteration {iteration}: {unmeasured} of {episodes} episodes did not measure the "
                f"agent ({failures} produced no score, {free_now} produced one without reaching "
                f"the model). Scores built from that are not a comparison; fix the environment "
                f"and rerun rather than spending the remaining iterations."
            )
    return state


if __name__ == "__main__":
    sys.exit(main())
