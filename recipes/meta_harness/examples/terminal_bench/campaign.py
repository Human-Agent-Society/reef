"""A Terminal-Bench search scheduled entirely through Reef scenario commits.

    commit frozen proposal -> commit episode reservations -> evaluate a wave
    -> commit its observations/spend -> select only complete valid evaluations

An unacknowledged reservation after a restart has unknown external effects.
It stops recovery rather than silently reissuing paid calls or inventing zero
spend. All exported JSON is derived from the scenario's committed state.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import os
import tempfile
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from recipes.meta_harness.backend import POPULATION_STATE_KEY, MetaHarnessBackend
from recipes.meta_harness.method import MetaHarnessProposer, MetaHarnessSelector, mutations_between
from recipes.meta_harness.population import PopulationStore
from recipes.meta_harness.recipe import MetaHarnessRecipe
from reef.harness.episode import run_episode
from reef.harness.model_binding import ModelBindings
from reef.harness.render import render_composition
from reef.train.backend import PreparedStep
from reef.train.cordis_backend import HarnessCandidate
from reef.train.evaluation.contracts import EvaluationResult

from .history import HistoryBinding, ProposerBudgetReached
from .runtime import check_capacity

CAMPAIGN_STATE_KEY = "terminal_bench_campaign"
_LOG = logging.getLogger(__name__)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def score_episode(task, result) -> float:
    del task
    verifier = next(event for event in result.trajectory if event.get("type") == "verifier")
    if not (verifier.get("outcome") or {}).get("valid"):
        raise ValueError("Terminal-Bench evaluation did not produce a valid measurement")
    score = verifier.get("reward")
    if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score):
        raise ValueError("Terminal-Bench verifier reward must be finite")
    return float(score)


class TerminalBenchBackend(MetaHarnessBackend):
    def __init__(self, *, plan, output: Path, **kwargs):
        if plan.get("runner"):
            from .isolated_adapter import isolated_descriptor

            kwargs["descriptor"] = isolated_descriptor(kwargs["descriptor"], kwargs["executor"])
        super().__init__(**kwargs)
        self.plan = json.loads(json.dumps(plan))
        self.output = output
        self.owner = uuid.uuid4().hex
        self._started_reservations = set()
        self._rolling_pool = None
        self._episode_futures = {}
        self._descriptor = dataclasses.replace(
            self._descriptor,
            env={
                **self._descriptor.env,
                "REEF_TERMINUS_TASK_COMMIT": self.plan["dataset_commit"],
                "REEF_TERMINUS_MAX_ATTEMPTS": "1",  # the committed campaign owns retries
            },
        )

    def initial_state(self):
        return {
            **super().initial_state(),
            CAMPAIGN_STATE_KEY: {
                "schema_version": 1,
                "plan": self.plan,
                "status": "running",
                "evaluation": None,
                "reservation": None,
                "records": {},
                "rounds": [],
                "proposals": [],
                "observed_cost_usd": 0.0,
                "proposer_cost_usd": 0.0,
                "unknown_usage": False,
            },
        }

    def prepare_step(self, batch, state, scenario_step):
        del batch
        # Clone before doing anything: no speculative mutation may leak into
        # Trainer.state or the PopulationStore's last committed object.
        next_state = json.loads(json.dumps(state))
        data = next_state[CAMPAIGN_STATE_KEY]
        if data["plan"] != self.plan:
            raise ValueError("run plan differs from committed Reef state; use a new output directory")
        population = self._population_store.begin(state[POPULATION_STATE_KEY])
        population.sync_served(state["entries"], step=0)
        next_state["steps"] = int(state.get("steps", 0)) + 1
        keep_transaction = False
        try:
            if data["status"] != "running":
                pass
            elif data["reservation"] is not None:
                reservation = data["reservation"]
                reservation_id = hashlib.sha256(json.dumps(reservation, sort_keys=True).encode()).hexdigest()
                rolling = reservation["kind"] == "episodes" and self.plan.get("episode_schedule") == "rolling"
                if reservation["owner"] != self.owner or (
                    not rolling and reservation_id in self._started_reservations
                ):
                    data.update(
                        status="interrupted",
                        unknown_usage=True,
                        reason="a prior process reserved external work without committing its result",
                    )
                elif reservation["kind"] == "proposal":
                    self._started_reservations.add(reservation_id)
                    # Baseline and earlier evidence are already committed.
                    binding = self._models["proposer"]
                    visible = data["records"]
                    if self.plan["mode"] == "incumbent_only":
                        visible = {
                            key: value
                            for key, value in visible.items()
                            if value["candidate_id"] == population.served_id
                        }
                    allowance = min(
                        self.plan["max_observed_cost_usd"] - data["observed_cost_usd"],
                        self.plan.get("max_proposer_cost_usd", float("inf")) - data.get("proposer_cost_usd", 0),
                    )
                    sources = None
                    if "proposer_source_manifest" in self.plan:
                        from .runtime_source import verified_source

                        sources = verified_source(self.plan["proposer_source_manifest"])
                    history = HistoryBinding(
                        binding,
                        visible,
                        effort=self.plan["proposer_effort"],
                        pricing=self.plan.get("proposer_pricing"),
                        remaining_cost_usd=allowance,
                        executable="code_extension" in self.plan.get("candidate_kinds", ()),
                        sources=sources,
                        timeout_s=self.plan.get("proposer_timeout_s"),
                    )
                    proposer = MetaHarnessProposer(
                        store=self._population_store,
                        descriptor=self._descriptor,
                        tasks=self._tasks,
                        episode_repeats=self._episode_repeats,
                        mode=self.plan["mode"],
                        kinds=tuple(self.plan.get("candidate_kinds", ("rules", "skill", "agent_command"))),
                    )
                    error = None
                    try:
                        proposer((), (), ModelBindings(served=self._models.served, named={"proposer": history}))
                    except Exception as exc:
                        error = exc
                    data["proposer_cost_usd"] = data.get("proposer_cost_usd", 0) + history.cost_usd
                    data["observed_cost_usd"] += history.cost_usd
                    data["proposals"].append(
                        {
                            "iteration": population.proposer_calls,
                            "visible_record_ids": list(visible),
                            "exchanges": history.audit,
                            "cost_usd": history.cost_usd,
                            "unknown_usage": history.unknown_usage,
                            "error_type": type(error).__name__ if error else None,
                        }
                    )
                    data["reservation"] = None
                    if history.unknown_usage:
                        population.discard_pending()
                        data.update(status="usage_unknown", unknown_usage=True, reason="proposer usage is unknown")
                    elif error:
                        population.discard_pending()
                        data.update(
                            status=(
                                "budget_exhausted" if isinstance(error, ProposerBudgetReached) else "proposal_failed"
                            ),
                            reason=f"proposer stopped ({type(error).__name__})",
                        )
                    elif population.pending_id:
                        data["evaluation"] = self._new_evaluation(population.pending_id, baseline=False)
                    elif population.attempts[-1]["status"] != "duplicate":
                        data.update(status="proposal_failed", reason="proposer returned an invalid composition")
                elif rolling:
                    if not self._refill_reservation(data):
                        self._execute_rolling(data, population)
                else:
                    self._started_reservations.add(reservation_id)
                    self._execute_wave(data, population)
            elif data["unknown_usage"]:
                data.update(status="usage_unknown", reason="an executed episode returned unknown usage")
            elif data["evaluation"] is None:
                if population.served.scores is None:
                    data["evaluation"] = self._new_evaluation(population.served_id, baseline=True)
                elif population.proposer_calls >= self.plan["iterations"]:
                    data["status"] = "complete"
                elif data["observed_cost_usd"] >= self.plan["max_observed_cost_usd"] or data.get(
                    "proposer_cost_usd", 0
                ) >= self.plan.get("max_proposer_cost_usd", float("inf")):
                    data["status"] = "budget_exhausted"
                else:
                    data["reservation"] = {
                        "kind": "proposal",
                        "owner": self.owner,
                        "iteration": population.proposer_calls + 1,
                    }
            else:
                evaluation = data["evaluation"]
                missing = [slot for slot in evaluation["slots"] if slot["accepted"] is None]
                if not missing:
                    from .agent_failure_policy import score as benchmark_score

                    scores = [
                        benchmark_score(data["records"][slot["accepted"]], self.plan.get("agent_failure_policy"))
                        for slot in evaluation["slots"]
                    ]
                    if evaluation["baseline"]:
                        population.served.scores = tuple(scores)
                        data["rounds"].append(
                            {
                                "iteration": 0,
                                "candidate_id": population.served_id,
                                "mean": sum(scores) / len(scores),
                                "selected": False,
                                "kind": "baseline",
                            }
                        )
                        data["evaluation"] = None
                    else:
                        current = tuple(population.served.entries)
                        proposed = tuple(population.pending.entries)
                        next_state[POPULATION_STATE_KEY] = population.to_dict()
                        prepared = PreparedStep.with_candidate(
                            HarnessCandidate(
                                candidate_id=population.pending_id,
                                candidate_files=render_composition(self._nodes_from(proposed), self._descriptor),
                                current_files=render_composition(self._nodes_from(current), self._descriptor),
                                candidate_entries=proposed,
                                current_entries=current,
                                mutations=mutations_between(current, proposed),
                                gate_tasks=self._tasks,
                            ),
                            state=next_state,
                            metrics={"iteration": population.proposer_calls},
                        )
                        keep_transaction = True
                        return prepared
                elif any(slot["attempts"] >= self._attempt_limit(data, slot) for slot in missing):
                    data.update(status="evaluation_failed", reason="unmeasured episodes exhausted their retries")
                elif data["observed_cost_usd"] >= self.plan["max_observed_cost_usd"]:
                    data["status"] = "budget_exhausted"
                else:
                    self._check_capacity(
                        min(len(missing), self._episode_workers) * (2 if self.plan.get("runner") else 1)
                    )
                    wave = []
                    for slot in missing[: self._episode_workers]:
                        slot["attempts"] += 1
                        identity = hashlib.sha256(
                            json.dumps(
                                [
                                    evaluation["candidate_id"],
                                    slot["task"],
                                    slot["repeat"],
                                    slot["attempts"],
                                ]
                            ).encode()
                        ).hexdigest()
                        wave.append(
                            {
                                "id": identity,
                                "task": slot["task"],
                                "repeat": slot["repeat"],
                                "attempt": slot["attempts"],
                            }
                        )
                    data["reservation"] = {"kind": "episodes", "owner": self.owner, "wave": wave}
            next_state[POPULATION_STATE_KEY] = population.to_dict()
            return PreparedStep.skipped(
                state=next_state,
                metrics={
                    "status": data["status"],
                    "observed_cost_usd": data["observed_cost_usd"],
                    "target_episode_calls": population.episode_calls,
                },
            )
        finally:
            if not keep_transaction:
                self._population_store.abort()

    def _check_capacity(self, count):
        options = (
            {"benchmark_limit": self.plan["benchmark_sandbox_limit"]} if "benchmark_sandbox_limit" in self.plan else {}
        )
        check_capacity(count, **options)

    def _attempt_limit(self, data, slot):
        return self.plan["max_attempts"]

    def _new_evaluation(self, candidate_id, *, baseline):
        return {
            "candidate_id": candidate_id,
            "baseline": baseline,
            "slots": [
                {"task": task, "repeat": repeat, "attempts": 0, "accepted": None}
                for task in self._tasks
                for repeat in range(self._episode_repeats)
            ],
        }

    def _execute_wave(self, data, population):
        evaluation = data["evaluation"]
        candidate = population.by_id(evaluation["candidate_id"])
        files = self._render_for_episode(candidate.entries)
        wave = data["reservation"]["wave"]
        with ThreadPoolExecutor(max_workers=self._episode_workers) as pool:
            results = list(pool.map(lambda item: self._episode(files, item), wave))
        self._record_results(data, population, wave, results)
        data["reservation"] = None

    def _record_results(self, data, population, wave, results):
        evaluation = data["evaluation"]
        candidate = population.by_id(evaluation["candidate_id"])
        for item, record in zip(wave, results, strict=True):
            record.update(candidate_id=candidate.candidate_id, side="seed" if evaluation["baseline"] else "candidate")
            data["records"][item["id"]] = record
            population.episode_calls += 1
            cost = record["cost_usd"]
            if cost is None:
                data["unknown_usage"] = True
            else:
                data["observed_cost_usd"] += cost
            from .agent_failure_policy import eligible

            if eligible(record, self.plan.get("agent_failure_policy")):
                slot = next(
                    slot
                    for slot in evaluation["slots"]
                    if slot["task"] == item["task"] and slot["repeat"] == item["repeat"]
                )
                slot["accepted"] = item["id"]

    def _refill_reservation(self, data):
        """Stage more slots; only the next committed prepare may launch them."""
        wave = data["reservation"]["wave"]
        available = self._episode_workers - len(wave)
        if (
            not available
            or data["unknown_usage"]
            or data["observed_cost_usd"] >= self.plan["max_observed_cost_usd"]
            or any(item["id"] not in self._episode_futures for item in wave)
        ):
            return False
        if any(self._episode_futures[item["id"]].done() for item in wave):
            return False  # commit already available costs before reserving more
        active = {(item["task"], item["repeat"]) for item in wave}
        missing = [
            slot
            for slot in data["evaluation"]["slots"]
            if slot["accepted"] is None and (slot["task"], slot["repeat"]) not in active
        ]
        if not missing or any(slot["attempts"] >= self._attempt_limit(data, slot) for slot in missing):
            return False  # drain already admitted work before reporting an invalid evaluation
        selected = missing[:available]
        self._check_capacity(len(selected) * (2 if self.plan.get("runner") else 1))
        for slot in selected:
            slot["attempts"] += 1
            identity = hashlib.sha256(
                json.dumps(
                    [data["evaluation"]["candidate_id"], slot["task"], slot["repeat"], slot["attempts"]]
                ).encode()
            ).hexdigest()
            wave.append({"id": identity, "task": slot["task"], "repeat": slot["repeat"], "attempt": slot["attempts"]})
        return True

    def _execute_rolling(self, data, population):
        """Collect completed trials while slower, durably reserved trials run."""
        wave = data["reservation"]["wave"]
        if self._rolling_pool is None:
            self._rolling_pool = ThreadPoolExecutor(max_workers=self._episode_workers)
        candidate = population.by_id(data["evaluation"]["candidate_id"])
        files = self._render_for_episode(candidate.entries)
        for item in wave:
            if item["id"] not in self._episode_futures:
                self._episode_futures[item["id"]] = self._rolling_pool.submit(self._episode, files, dict(item))
        wait([self._episode_futures[item["id"]] for item in wave], timeout=15, return_when=FIRST_COMPLETED)
        completed = [item for item in wave if self._episode_futures[item["id"]].done()]
        results = []
        for item in completed:
            try:
                result = self._episode_futures[item["id"]].result()
            except Exception as exc:
                result = {
                    **item,
                    "valid": False,
                    "reward": None,
                    "cost_usd": None,
                    "phase": "execution_failure",
                    "error": type(exc).__name__,
                    "trajectory": [],
                }
            results.append(result)
        self._record_results(data, population, completed, results)
        finished = {item["id"] for item in completed}
        remaining = [item for item in wave if item["id"] not in finished]
        data["reservation"] = {**data["reservation"], "wave": remaining} if remaining else None

    def _episode(self, files, identity):
        started = time.monotonic()
        record = {**identity, "valid": False, "reward": None, "cost_usd": None, "trajectory": []}
        try:
            result = run_episode(
                self._descriptor,
                files,
                identity["task"],
                binary=self._binary,
                timeout=self._episode_timeout_s,
                executor=self._executor,
            )
            verifier = next((event for event in result.trajectory if event.get("type") == "verifier"), {})
            outcome = verifier.get("outcome") or {}
            record.update(
                {
                    **outcome,
                    "cost_usd": verifier.get("observed_cost_usd"),
                    "trajectory": list(result.trajectory),
                    "exit_code": result.exit_code,
                    "residue": list(result.residue),
                    "runner_stdout": result.stdout[-131072:],
                    "runner_stderr": result.stderr[-131072:],
                    "runner_output_truncated": len(result.stdout) > 131072 or len(result.stderr) > 131072,
                }
            )
            if "valid" not in outcome:
                record.update(
                    valid=False, phase="missing_diagnostics", error="runner did not supply trial diagnostics"
                )
            if self._forbid_residue and result.residue:
                record.update(valid=False, phase="residue", error="episode left forbidden residue")
            if record["valid"]:
                score = float(self._score_episode(identity["task"], result))
                if not math.isfinite(score):
                    raise ValueError("episode scorer returned a non-finite score")
                record["reward"] = score
            cost = record["cost_usd"]
            if cost is not None and (
                isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0
            ):
                raise ValueError("episode returned invalid cost")
        except Exception as exc:
            record.update(valid=False, phase="episode_failure", error=str(exc), reward=None)
            from .e2b_transport import failure_diagnostics, failure_evidence

            if diagnostics := failure_diagnostics(exc):
                record["transport_diagnostics"] = diagnostics
                record["cost_usd"] = diagnostics.get("observed_cost_usd")
            if evidence := failure_evidence(exc):
                record["trajectory"] = evidence.get("trajectory", [])
            if isinstance(exc, ValueError):
                raise
        record["elapsed_s"] = time.monotonic() - started
        if getattr(self, "plan", {}).get("agent_failure_policy"):
            from .agent_failure_policy import annotate

            record = annotate(record, self.plan["agent_failure_policy"])
        return record

    def evaluate(self, candidate):
        # Selection reads only observations already committed by previous
        # waves. There are no model calls in this phase and no incumbent leg.
        state = self._committed_campaign
        evaluation = state["evaluation"]
        if evaluation["candidate_id"] != candidate.candidate_id:
            raise ValueError("selection candidate differs from committed evaluation")
        records = [state["records"][slot["accepted"]] for slot in evaluation["slots"]]
        from .agent_failure_policy import eligible
        from .agent_failure_policy import score as benchmark_score

        if not records or not all(eligible(record, self.plan.get("agent_failure_policy")) for record in records):
            raise ValueError("cannot select from an incomplete or invalid evaluation")
        return EvaluationResult(
            evaluator="committed_terminal_bench",
            evaluator_version="1",
            metrics={
                "candidate_scores": tuple(
                    benchmark_score(record, self.plan.get("agent_failure_policy")) for record in records
                ),
                "current_scores": self._population_store.active.served.scores,
                "episode_calls": 0,  # already charged by the episode commits
            },
        )

    def settle_step(self, prepared, decision):
        result = super().settle_step(prepared, decision)
        state = json.loads(json.dumps(result.state))
        data = state[CAMPAIGN_STATE_KEY]
        data["rounds"].append(
            {
                "iteration": state[POPULATION_STATE_KEY]["proposer_calls"],
                "candidate_id": prepared.candidate.candidate_id,
                "mean": decision.metrics["candidate_mean"],
                "selected": bool(decision.selected),
                "kind": "candidate",
            }
        )
        data["evaluation"] = None
        # This standalone experiment serves compositions from committed
        # algorithm state. It does not publish a serving endpoint; the full
        # composition is already in that state and is rendered on recovery.
        # Avoid moving an artifact head before the campaign journal commits.
        return dataclasses.replace(result, state=state, artifact=None, metrics={**result.metrics, "published": False})

    def restore(self, state):
        if state[CAMPAIGN_STATE_KEY]["plan"] != self.plan:
            raise ValueError("run plan differs from committed Reef state; use a new output directory")
        self._population_store.restore_committed(state[POPULATION_STATE_KEY])
        self._committed_campaign = json.loads(json.dumps(state[CAMPAIGN_STATE_KEY]))
        self._loader.root.update([dict(entry) for entry in state["entries"]])

    def commit_applied(self, state):
        super().commit_applied(state)
        self.restore(state)
        # Remove completed handles only after their observations are durable.
        # A failed commit retains them for a retry without repeating an episode.
        for identity in list(self._episode_futures):
            if identity in state[CAMPAIGN_STATE_KEY]["records"]:
                del self._episode_futures[identity]
        if not self._episode_futures and self._rolling_pool is not None:
            self._rolling_pool.shutdown(wait=False)
            self._rolling_pool = None
        try:
            self.export(state)
        except OSError as exc:
            _LOG.warning("could not refresh committed campaign mirrors: %s", exc)

    def export(self, state):
        """Post-commit mirrors only; recovery never reads these files."""
        data = state[CAMPAIGN_STATE_KEY]
        self._population_store.persist()
        atomic_json(self.output / "run.json", data)
        atomic_json(
            self.output / "observed-cost.json",
            {
                "schema_version": 2,
                "max_observed_cost_usd": self.plan["max_observed_cost_usd"],
                "observed_cost_usd": data["observed_cost_usd"],
                "proposer_cost_usd": data.get("proposer_cost_usd", 0),
                "carried_cost_usd": data.get("carried_cost_usd", 0),
                "new_cost_usd": data["observed_cost_usd"] - data.get("carried_cost_usd", 0),
                "unknown_usage": data["unknown_usage"],
                "trials": {key: row["cost_usd"] for key, row in data["records"].items()},
            },
        )
        if data["status"] == "complete":
            atomic_json(self.output / "final-population.json", state[POPULATION_STATE_KEY])
        else:
            (self.output / "final-population.json").unlink(missing_ok=True)


@dataclasses.dataclass(frozen=True)
class TerminalBenchRecipe(MetaHarnessRecipe):
    plan: dict[str, Any] = dataclasses.field(default_factory=dict)
    output: Path = Path(".")

    def build(self, scenario, records, *, algorithm_state=None, experiment_logger=None):
        # The one campaign's mirror path is fixed; the scenario commit log's
        # own name encoding remains authoritative for durable recovery.
        store = PopulationStore(self.output / "population.json")
        bound = dataclasses.replace(self, candidate_selector=MetaHarnessSelector(store))
        backend = TerminalBenchBackend(
            population_store=store,
            plan=self.plan,
            output=self.output,
            **bound._backend_kwargs(),
        )
        if algorithm_state is not None:
            backend.restore(algorithm_state)
            backend.export(algorithm_state)
        return bound._build_trainer(
            scenario, records, backend, algorithm_state=algorithm_state, experiment_logger=experiment_logger
        )
