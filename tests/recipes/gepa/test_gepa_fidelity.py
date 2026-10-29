"""Differential check of recipes/gepa against upstream GEPA, iteration by iteration.

Upstream's ``gepa.optimize`` and the Reef method run the same synthetic task
with deterministic scoring and a deterministic reflection model, from the
same seed. The Reef side runs whole steps on the real backend: a fake pi
binary answers each problem with a token that its rules and the problem
encode, the step's evaluation scores the validation set, and the compare
policy on the mean decides. Every draw upstream makes must land in the same
place here: which parent it picks, which training problems it shows the
reflection model, whether the child is accepted, and what the archive ends
up holding. Runs only where the upstream package is installed; it is a
development check, not a runtime dependency.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from tests.support._trajectories import recorded_trajectory
from tests.support.harness_service import backend_step
from tests.support.harness_steps import COMPONENT, TreeReleases, harness_recipe

gepa = pytest.importorskip("gepa")

from gepa.core.adapter import EvaluationBatch
from gepa.core.state import GEPAState
from recipes.gepa import GEPA, Feedback, GEPASettings
from recipes.gepa.archive import Archive
from recipes.gepa.method import ARCHIVE_FILE, archive_for_step
from reef.evaluation.harness import ComparePolicy
from reef.harness.rollouts.model_binding import ModelBinding
from reef.harness.tree import HarnessTree
from reef.train.backend import ComponentReleases
from reef.train.harness_backend.state import HarnessState
from reef.train.types import JobData

TRAIN = [{"input": f"train problem {i}", "answer": "### 1"} for i in range(45)]
VAL = [{"input": f"validation problem {i}", "answer": "### 1"} for i in range(45)]
SEED_TEXT = "Answer the question."

#: A fake pi that answers a problem as ``output_for`` does, from the rules the tree holds. Rendering ends the rules
#: file with a newline, which the tree's text does not have.
PI_FAKE = (
    f"#!{sys.executable}\n"
    """\
import hashlib, json, os, sys
from pathlib import Path

prompt = sys.argv[sys.argv.index("-p") + 1]
rules = (Path(os.environ["PI_CODING_AGENT_DIR"]) / "AGENTS.md").read_text().removesuffix("\\n")
token = hashlib.sha256(f"{rules}|{prompt}".encode()).hexdigest()[:8]
sessions = Path(os.environ["PI_CODING_AGENT_SESSION_DIR"])
sessions.mkdir(parents=True, exist_ok=True)
event = {"type": "message", "role": "assistant", "content": f"work... ### {token}"}
(sessions / "session.jsonl").write_text(json.dumps(event) + "\\n")
"""
)


def output_for(text: str, task: str) -> str:
    """What a composition answers: a token that encodes (text, task)."""
    return f"work... ### {hashlib.sha256(f'{text}|{task}'.encode()).hexdigest()[:8]}"


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


class Reflector(ModelBinding):
    def chat(self, messages, **params):
        return reflect(messages[0]["content"])


def run_reef(tmp_path: Path, iterations: int) -> tuple[Archive, dict[int, dict[str, Any]]]:
    """``iterations`` steps on the backend, and the plan the driver drew for each iteration.

    Before each step the driver draws the iteration's plan from the committed
    archive and the history, as the AIME driver does, and sends the planned
    training problems as the served tree's traffic; the method draws the same
    plan in the step.
    """
    binary = tmp_path / "token-pi"
    binary.write_text(PI_FAKE)
    binary.chmod(0o755)
    settings = GEPASettings(
        minibatch_size=3,
        seed=0,
        skip_perfect_score=True,
        perfect_score=1.0,
        components=("rules",),
        trainset_size=len(TRAIN),
    )
    validation = tuple(example["input"] for example in VAL)
    recipe = harness_recipe(
        tmp_path,
        GEPA(settings, ReefFeedback()),
        seed=HarnessTree({"rules.md": SEED_TEXT}),
        score=lambda task, result: score_of(result.trajectory[-1]["content"]),
        tasks=validation,
        binary=str(binary),
    )
    recipe = dataclasses.replace(recipe, models={"reflection": Reflector("http://model.test", "reflection")})
    releases = TreeReleases(recipe.seed)
    backend = recipe.build_backend("fidelity", ComponentReleases(releases, COMPONENT), None)
    policy = ComparePolicy(backend, statistic="mean")

    def committed_archive(state: dict[str, Any], text: str) -> Archive:
        archive = HarnessState.from_json(state).method_files.get(ARCHIVE_FILE)
        return archive_for_step(
            archive, backend.history, validation, {"rules.md": text}, repeats=recipe.evaluation.repeats
        )

    state = dict(backend.initial_state())
    plans: dict[int, dict[str, Any]] = {}
    try:
        for step in range(1, iterations + 1):
            text = releases.head(COMPONENT).files["rules.md"]
            plan = committed_archive(state, text).plan(len(TRAIN), len(VAL), 3)
            plans[plan["iteration"]] = plan
            samples = tuple(
                recorded_trajectory(
                    f"r{step}-{index}",
                    {
                        "messages": [{"role": "user", "content": TRAIN[index]["input"]}],
                        "response": {
                            "message": {"role": "assistant", "content": output_for(text, TRAIN[index]["input"])}
                        },
                    },
                    score_of(output_for(text, TRAIN[index]["input"])),
                )
                for index in plan["minibatch"]
            )
            result = backend_step(backend, JobData(f"fidelity:{step}", samples), state, policy)
            if result.artifact is not None:
                # What the committer does with a selected candidate: it becomes the head release.
                releases.publish(f"r{step}", HarnessTree.read(result.artifact.local_path))
                backend.commit_applied(result.state)
            state = dict(result.state)
        # The last step's validation folds in as the next step would fold it.
        return committed_archive(state, releases.head(COMPONENT).files["rules.md"]), plans
    finally:
        backend.close()


def test_the_method_walks_upstreams_exact_trajectory(tmp_path: Path) -> None:
    result, log, iterations = run_upstream(tmp_path / "upstream", budget=400)
    archive, plans = run_reef(tmp_path, iterations)

    # The same candidates, in the same order, from the same parents, with
    # the same validation means.
    assert [c.texts["rules.md"] for c in archive.candidates] == [c["rules"] for c in result.candidates]
    assert [c.parent for c in archive.candidates] == [None, *(p[0] for p in result.parents[1:])]
    assert [round(archive.mean_val(i), 6) for i in range(len(archive.candidates))] == [
        round(s, 6) for s in result.val_aggregate_scores
    ]
    # Every iteration showed the reflection model the same problems
    # from the same parent.
    for entry in log:
        plan = plans[entry["i"]]
        assert plan["minibatch"] == list(entry["subsample_ids"]), entry["i"]
        assert plan["parent"] == entry["selected_program_candidate"], entry["i"]
    assert len(log) >= 3, "the synthetic task must accept several candidates for this check to mean anything"
