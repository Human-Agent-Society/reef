"""Differential check of recipes/gepa against upstream GEPA, iteration by iteration.

Upstream's ``gepa.optimize`` and the Reef method run the same synthetic task
with deterministic scoring and a deterministic reflection model, from the
same seed. Every draw upstream makes must land in the same place here:
which parent it picks, which training problems it shows the reflection
model, whether the child is accepted, and what the archive ends up holding.
Runs only where the upstream package is installed; it is a development
check, not a runtime dependency.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from tests.support._trajectories import recorded_trajectory

gepa = pytest.importorskip("gepa")

from gepa.core.adapter import EvaluationBatch
from gepa.core.state import GEPAState
from recipes.gepa.archive import Archive
from recipes.gepa.proposer import Feedback, GEPAProposer, GEPASettings, archive_for_step, sync
from recipes.gepa.selection import GEPASelection
from reef.core.evaluation import EvaluationResult, RolloutScore, RolloutScores
from reef.evaluation.harness import HarnessCandidate, ScoredRollout
from reef.harness.adapters import get_adapter
from reef.harness.rollouts.model_binding import ModelBinding, ModelBindings
from reef.harness.tree import HarnessTree, TreeKind
from reef.train.harness_backend.contract import ProposalContext, SampleSource, StepOutcome, TrialRunner

TRAIN = [{"input": f"train problem {i}", "answer": "### 1"} for i in range(45)]
VAL = [{"input": f"validation problem {i}", "answer": "### 1"} for i in range(45)]
SEED_TEXT = "Answer the question."


def _digest(text: str, task: str) -> str:
    return hashlib.sha256(f"{text}|{task}".encode()).hexdigest()[:8]


def output_for(text: str, task: str) -> str:
    """What a composition answers: a token that encodes (text, task)."""
    return f"work... ### {_digest(text, task)}"


def score_of(output: str) -> float:
    """About two in five answers are right, decided by the token alone."""
    token = output.rsplit("### ", 1)[-1]
    return 1.0 if int(token, 16) % 5 < 2 else 0.0


def feedback(task: str, output: str, score: float) -> str:
    return "correct" if score >= 1.0 else f"wrong for {task}"


class ReefFeedback(Feedback):
    def feedback(self, task: str, output: str, score: float) -> str:
        return feedback(task, output, score)


def reflect(prompt: str) -> str:
    """A reflection model that rewrites deterministically from what it was shown."""
    return f"```\nrules v{hashlib.sha256(prompt.encode()).hexdigest()[:10]}\n```"


class UpstreamAdapter:
    propose_new_texts = None

    def evaluate(self, batch, candidate, capture_traces=False):
        outputs = [output_for(candidate["rules"], example["input"]) for example in batch]
        scores = [score_of(output) for output in outputs]
        trajectories = None
        if capture_traces:
            trajectories = [
                {"input": example["input"], "output": output, "score": score}
                for example, output, score in zip(batch, outputs, scores, strict=True)
            ]
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        return {
            component: [
                {
                    "Inputs": t["input"],
                    "Generated Outputs": t["output"],
                    "Feedback": feedback(t["input"], t["output"], t["score"]),
                }
                for t in eval_batch.trajectories
            ]
            for component in components_to_update
        }


def run_upstream(run_dir: Path, budget: int):
    result = gepa.optimize(
        seed_candidate={"rules": SEED_TEXT},
        trainset=list(TRAIN),
        valset=list(VAL),
        adapter=UpstreamAdapter(),
        reflection_lm=reflect,
        max_metric_calls=budget,
        seed=0,
        run_dir=str(run_dir),
        skip_perfect_score=True,
    )
    log = json.loads((run_dir / "run_log.json").read_text())
    return result, log, GEPAState.load(str(run_dir)).i + 1


class ReefTrials(TrialRunner):
    """The proposer's minibatch rollouts: a tree answers with the token its rules text and the task encode."""

    def run(self, tree: HarnessTree, tasks) -> tuple[ScoredRollout, ...]:
        text = tree.files["rules.md"]
        return tuple(
            ScoredRollout(RolloutScore(score_of(output_for(text, task))), None, reply=output_for(text, task))
            for task in tasks
        )


def rollout_scores(tasks: tuple[str, ...], values: tuple[float, ...]) -> list[dict[str, Any]]:
    """The evaluation's JSON form of one scored rollout per task."""
    return RolloutScores.by_task(tasks, [RolloutScore(value) for value in values]).to_json()


class Reflector(ModelBinding):
    def chat(self, messages, **params):
        return reflect(messages[0]["content"])


def run_reef(iterations: int) -> tuple[Archive, dict[int, dict[str, Any]]]:
    """``iterations`` steps the way the backend runs them, and the plan the driver drew for each iteration.

    Each step's driver draws the iteration's plan from the committed method
    state and the latest outcome, as the AIME driver does, and sends the
    planned training problems as traffic; the proposer draws the same plan.
    """
    proposer = GEPAProposer(
        GEPASettings(
            minibatch_size=3,
            rng_seed=0,
            skip_perfect_score=True,
            perfect_score=1.0,
            max_metric_calls=None,
            kinds=(TreeKind.RULES,),
            trainset_size=len(TRAIN),
        ),
        feedback=ReefFeedback(),
    )
    selection = GEPASelection(None)
    served = ModelBinding("http://model.test", "m", api_key="k")
    models = ModelBindings(served=served, named={"reflection": Reflector("http://model.test", "reflection")})
    validation = tuple(example["input"] for example in VAL)
    text = SEED_TEXT
    method_state: Mapping[str, Any] = {}
    last_outcome: StepOutcome | None = None
    plans: dict[int, dict[str, Any]] = {}
    for step in range(1, iterations + 1):
        planned = archive_for_step(method_state, last_outcome)
        sync(planned, {"rules.md": text}, len(VAL))
        plan = planned.plan(len(TRAIN), len(VAL), 3)
        plans[plan["iteration"]] = plan
        samples = tuple(
            recorded_trajectory(
                f"r{step}-{index}",
                {
                    "messages": [{"role": "user", "content": TRAIN[index]["input"]}],
                    "response": {"message": {"role": "assistant", "content": output_for(text, TRAIN[index]["input"])}},
                },
                score_of(output_for(text, TRAIN[index]["input"])),
            )
            for index in plan["minibatch"]
        )
        tree = HarnessTree({"rules.md": text})
        proposal = proposer.propose(
            ProposalContext(
                step=step,
                tree=tree,
                descriptor=get_adapter("pi"),
                samples=samples,
                sources=tuple(SampleSource(None, "untagged") for _ in samples),
                request=None,
                evaluation_tasks=validation,
                models=models,
                method_state=method_state,
                last_outcome=last_outcome,
                trials=ReefTrials(),
            )
        )
        assert proposal.method_state is not None
        method_state = proposal.method_state
        if not proposal.patch.changes:
            continue
        child = str(proposal.patch.changes["rules.md"])
        candidate_scores = tuple(score_of(output_for(child, task)) for task in validation)
        current_scores = tuple(score_of(output_for(text, task)) for task in validation)
        decision = selection.decide(
            HarnessCandidate(
                candidate_id=f"c{step}",
                candidate_tree=tree.apply(proposal.patch),
                current_tree=tree,
                patch=proposal.patch,
                evaluation_tasks=validation,
                method_state=method_state,
            ),
            EvaluationResult(
                evaluator="test",
                evaluator_version="1",
                metrics={
                    "candidate_scores": rollout_scores(validation, candidate_scores),
                    "current_scores": rollout_scores(validation, current_scores),
                },
            ),
        )
        last_outcome = StepOutcome(
            step, proposal.patch, decision.selected, decision.reason, candidate_scores, current_scores
        )
        if decision.selected:
            text = child
    # The last step's validation folds in as the next step would fold it.
    return archive_for_step(method_state, last_outcome), plans


def test_the_method_walks_upstreams_exact_trajectory(tmp_path: Path) -> None:
    result, log, iterations = run_upstream(tmp_path / "upstream", budget=400)
    archive, plans = run_reef(iterations)

    # The same candidates, in the same order, from the same parents, with
    # the same validation means.
    assert [c.texts["rules.md"] for c in archive.candidates] == [c["rules"] for c in result.candidates]
    assert [c.parent for c in archive.candidates] == [None, *(p[0] for p in result.parents[1:])]
    assert [round(archive.mean_val(i), 6) for i in range(len(archive.candidates))] == [
        round(s, 6) for s in result.val_aggregate_scores
    ]
    # Every accepted iteration showed the reflection model the same problems
    # from the same parent.
    for entry in log:
        plan = plans[entry["i"]]
        assert plan["minibatch"] == list(entry["subsample_ids"]), entry["i"]
        assert plan["parent"] == entry["selected_program_candidate"], entry["i"]
    assert len(log) >= 3, "the synthetic task must accept several candidates for this check to mean anything"
