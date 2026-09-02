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
from pathlib import Path

import pytest

gepa = pytest.importorskip("gepa")

from gepa.core.adapter import EvaluationBatch  # noqa: E402
from gepa.core.state import GEPAState  # noqa: E402

from recipes.gepa.archive import Archive  # noqa: E402
from recipes.gepa.method import GEPAProposer, GEPASelector  # noqa: E402
from reef.harness.adapters import get_adapter  # noqa: E402
from reef.harness.episode import EpisodeResult  # noqa: E402
from reef.harness.model_binding import ModelBinding, ModelBindings  # noqa: E402
from reef.train.cordis_backend.strategies import resolve_episode_scorer  # noqa: E402
from reef.train.evaluation.contracts import EvaluationResult, UpdateCandidate  # noqa: E402
from reef.train.types import TraceSample  # noqa: E402

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


class ReefEpisodes:
    def __call__(self, descriptor, files, prompt, *, binary=None, timeout=600.0):
        text = files["pi-agent/AGENTS.md"].rstrip("\n")
        return EpisodeResult(0, "", "", ({"role": "assistant", "content": output_for(text, prompt)},), ())


class Reflector:
    def chat(self, messages, **params):
        return reflect(messages[0]["content"])


def evaluate(task: str, result: EpisodeResult) -> float:
    return score_of(result.trajectory[-1]["content"])


def run_reef(tmp_path: Path, iterations: int) -> Archive:
    archive = Archive(tmp_path / "archive.json")
    proposer = GEPAProposer(
        archive=archive,
        descriptor=get_adapter("pi"),
        binary=None,
        score_episode=resolve_episode_scorer(evaluate),
        feedback=feedback,
        minibatch_size=3,
        rng_seed=0,
        skip_perfect_score=True,
        perfect_score=1.0,
        max_metric_calls=None,
        kinds=("rules",),
        valset_size=len(VAL),
        episode_runner=ReefEpisodes(),
    )
    selector = GEPASelector(archive)
    served = ModelBinding("http://model.test", "m", api_key="k")
    models = ModelBindings(served=served, named={"reflection": Reflector()})
    text = SEED_TEXT
    for iteration in range(iterations):
        plan = archive.plan(len(TRAIN), len(VAL), 3)
        samples = tuple(
            TraceSample(
                f"r{iteration}-{index}",
                {
                    "messages": [{"role": "user", "content": TRAIN[index]["input"]}],
                    "response": {"message": {"role": "assistant", "content": output_for(text, TRAIN[index]["input"])}},
                },
                score_of(output_for(text, TRAIN[index]["input"])),
            )
            for index in plan["minibatch"]
        )
        mutations = proposer((("rules", {"text": text}),), samples, models)
        if not mutations:
            continue
        child = str(mutations[0].options["config"]["text"])
        evaluation = EvaluationResult(
            evaluator="test",
            evaluator_version="1",
            metrics={
                "candidate_scores": tuple(score_of(output_for(child, v["input"])) for v in VAL),
                "current_scores": tuple(score_of(output_for(text, v["input"])) for v in VAL),
            },
        )
        if selector.decide(UpdateCandidate(candidate_id=f"c{iteration}"), evaluation).selected:
            text = child
    return archive


def test_the_method_walks_upstreams_exact_trajectory(tmp_path: Path) -> None:
    result, log, iterations = run_upstream(tmp_path / "upstream", budget=400)
    archive = run_reef(tmp_path / "reef", iterations)

    # The same candidates, in the same order, from the same parents, with
    # the same validation means.
    assert [c.texts["rules"] for c in archive.candidates] == [c["rules"] for c in result.candidates]
    assert [c.parent for c in archive.candidates] == [None, *(p[0] for p in result.parents[1:])]
    assert [round(archive.mean_val(i), 6) for i in range(len(archive.candidates))] == [
        round(s, 6) for s in result.val_aggregate_scores
    ]
    # Every accepted iteration showed the reflection model the same problems
    # from the same parent.
    for entry in log:
        plan = archive.plans[str(entry["i"])]
        assert plan["minibatch"] == list(entry["subsample_ids"]), entry["i"]
        assert plan["parent"] == entry["selected_program_candidate"], entry["i"]
    assert len(log) >= 3, "the synthetic task must accept several candidates for this check to mean anything"
