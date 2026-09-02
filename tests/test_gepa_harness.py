"""Deterministic coverage for the GEPA harness-evolution reproduction.

Nothing here calls a model: fake Pi episodes drive the real pinned upstream
optimizer, and the driver's paid paths are stubbed at the adapter boundary.
"""

from __future__ import annotations

import importlib
import json
import shutil
import sys
import threading
from pathlib import Path

import pytest

pytest.importorskip("gepa")  # the example's pin, installed by CI; absent from a bare checkout
from gepa.core.adapter import EvaluationBatch
from gepa.core.result import GEPAResult

from reef.artifact import ArtifactRef
from reef.harness import EpisodeError, EpisodeResult
from reef.harness.adapters import get_adapter
from reef.harness.model_binding import ModelBinding

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = REPO_ROOT / "recipes" / "gepa" / "examples" / "aime_harness_evolve"
NO_USAGE = {"requests": 0, "input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
BINDING = ModelBinding("http://model.test", "task-model", api_key="transient")


@pytest.fixture
def load(monkeypatch):
    """Import an example module fresh. Sibling examples share the ``harness``
    package name, so a cached import from another suite must not win."""
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    for name in [name for name in sys.modules if name in {"harness", "run"} or name.startswith("harness.")]:
        del sys.modules[name]
    return importlib.import_module


def pi_episode(answer: str) -> EpisodeResult:
    trajectory = ({"message": {"role": "assistant", "content": [{"type": "text", "text": answer}]}},)
    return EpisodeResult(0, "", "", trajectory, ())


def gepa_result(*, candidates, scores, fronts) -> GEPAResult:
    return GEPAResult(
        candidates=candidates,
        parents=[[None], *[[0] for _ in candidates[1:]]],
        val_aggregate_scores=scores,
        val_subscores=[{} for _ in candidates],
        per_val_instance_best_candidates=fronts,
        discovery_eval_counts=list(range(len(candidates))),
    )


class Usage:
    def snapshot(self):
        return dict(NO_USAGE)


class FakeAdapter:
    """Scores by a lookup on the example input; every output is a clean episode."""

    usage = Usage()

    def __init__(self, scores, *, exit_code=0):
        self.scores = scores
        self.exit_code = exit_code

    def evaluate(self, batch, candidate, capture_traces=False):
        return EvaluationBatch(
            outputs=[{"exit_code": self.exit_code, "residue": []} for _ in batch],
            scores=[self.scores(item["input"]) for item in batch],
        )


# Pins ---------------------------------------------------------------------


def test_pins_are_exact_and_shared_with_packaging_and_ci(load):
    config = load("harness.config")

    assert config.GEPA_VERSION == "0.1.2"
    assert config.PI_VERSION == "0.84.2"
    assert config.TASK_MODEL == "gpt-4.1-mini-2025-04-14"
    assert config.REFLECTION_MODEL == "gpt-5-2025-08-07"
    assert config.SEARCH_BUDGET == 150
    assert config.SEEDS == (0, 1, 2)
    assert config.PI_TASK_MAX_TOKENS == 16_384
    assert config.AIME_SPLIT_SIZES == {"train": 45, "validation": 45, "test": 150}
    assert config.AIME_DATASET_SHA256 == "74e81306a9a1debadd64c49a4ab3588615f7bb698b695a59c17c65dd3b895185"
    pin = f"gepa=={config.GEPA_VERSION}"
    assert pin in (EXAMPLE_DIR / "pyproject.toml").read_text(encoding="utf-8")
    assert pin in (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")


# The Reef adapter ---------------------------------------------------------


def test_reef_adapter_renders_the_binding_and_scores_a_pi_trace(load):
    adapter_module = load("harness.adapter")
    calls, events = [], []

    class Cap:
        def before_call(self):
            events.append("before")

        def record_call(self, cost_usd):
            events.append(("record", cost_usd))

    def run(descriptor, files, prompt, **kwargs):
        calls.append((files, prompt, kwargs))
        return pi_episode("Reasoning\n### 42")

    adapter = adapter_module.ReefAdapter(
        descriptor=get_adapter("pi"), task_model=BINDING, binary="fake-pi", episode_runner=run, spend_cap=Cap()
    )
    evaluated = adapter.evaluate(
        [{"input": "What is six times seven?", "answer": "### 42"}],
        {"rules": "Solve carefully and end with ### <answer>."},
        capture_traces=True,
    )

    assert evaluated.scores == [1.0]
    assert evaluated.num_metric_calls == 1
    assert evaluated.outputs[0]["assistant_response"] == "Reasoning\n### 42"
    assert evaluated.trajectories[0]["expected_answer"] == "### 42"
    files, prompt, kwargs = calls[0]
    assert prompt == "What is six times seven?"
    assert kwargs == {"binary": "fake-pi", "timeout": 600.0}
    assert files["pi-agent/AGENTS.md"] == "Solve carefully and end with ### <answer>.\n"
    models = json.loads(files["pi-agent/models.json"])["providers"]["reef"]
    assert models["baseUrl"] == "http://model.test/v1"
    assert events == ["before", ("record", 0.0)]


def test_reef_adapter_scores_a_failed_episode_zero_and_explains_it(load):
    adapter_module = load("harness.adapter")

    def run(*args, **kwargs):
        raise EpisodeError("pi exited early")

    adapter = adapter_module.ReefAdapter(descriptor=get_adapter("pi"), task_model=BINDING, episode_runner=run)
    evaluated = adapter.evaluate([{"input": "q", "answer": "### 1"}], {"rules": "seed"}, capture_traces=True)

    assert evaluated.scores == [0.0]
    assert evaluated.outputs[0]["exit_code"] == -1
    assert "pi exited early" in evaluated.trajectories[0]["feedback"]


def test_reef_adapter_evaluates_a_batch_concurrently(load):
    adapter_module = load("harness.adapter")
    barrier = threading.Barrier(2)

    def run(descriptor, files, prompt, **kwargs):
        barrier.wait(timeout=5)  # both episodes must be in flight at once
        return pi_episode("### 1" if prompt == "first" else "### 2")

    adapter = adapter_module.ReefAdapter(
        descriptor=get_adapter("pi"), task_model=BINDING, max_workers=2, episode_runner=run
    )
    evaluated = adapter.evaluate(
        [{"input": "first", "answer": "### 1"}, {"input": "second", "answer": "### 2"}],
        {"rules": "seed"},
        capture_traces=True,
    )

    assert evaluated.scores == [1.0, 1.0]
    assert [trajectory["input"] for trajectory in evaluated.trajectories] == ["first", "second"]


def test_quickstart_episode_gives_pi_the_rendered_rules_as_its_system_prompt(load, monkeypatch):
    adapter_module = load("harness.adapter")
    observed = {}

    def fake_run(descriptor, files, prompt, *, binary, timeout):
        observed.update(descriptor=descriptor, files=files, prompt=prompt, binary=binary, timeout=timeout)
        return EpisodeResult(0, "", "", (), ())

    monkeypatch.setattr(adapter_module, "run_episode", fake_run)
    files = {"pi-agent/AGENTS.md": "same official prompt\n"}
    adapter_module.run_quickstart_episode(get_adapter("pi"), files, "problem", binary="fake-pi", timeout=12)

    argv = observed["descriptor"].argv
    assert argv[argv.index("--system-prompt") + 1] == "same official prompt"
    assert {"--no-tools", "--no-context-files", "--no-skills"}.issubset(argv)
    assert observed["files"]["pi-agent/AGENTS.md"] == files["pi-agent/AGENTS.md"]
    extension = observed["files"][adapter_module.QUICKSTART_EXTENSION]
    assert 'systemPrompt: "same official prompt"' in extension
    assert 'pi.on("before_provider_request"' in extension
    assert (observed["prompt"], observed["binary"], observed["timeout"]) == ("problem", "fake-pi", 12)


def test_reef_adapter_makes_the_pi_output_limit_explicit(load):
    adapter_module = load("harness.adapter")
    adapter = adapter_module.ReefAdapter(descriptor=get_adapter("pi"), task_model=BINDING, task_max_tokens=16_384)

    rendered = adapter.render_episode_candidate({"rules": "prompt"})

    model = json.loads(rendered["pi-agent/models.json"])["providers"]["reef"]["models"][0]
    assert model == {"id": "task-model", "maxTokens": 16_384}


@pytest.mark.parametrize(
    ("components", "candidate", "update", "role"),
    [
        ("RULES_ONLY", {"rules": "seed"}, "rules", "global rules loaded for every episode"),
        ("MULTI_NODE", {"rules": "seed", "skill": "seed skill"}, "skill", "skill node 'aime-solver'"),
    ],
)
def test_reflection_records_name_the_selected_component_role(load, components, candidate, update, role):
    adapter_module = load("harness.adapter")
    adapter = adapter_module.ReefAdapter(
        descriptor=get_adapter("pi"), task_model=BINDING, components=getattr(adapter_module, components)
    )
    trajectory = {
        "input": "task",
        "expected_answer": "### 2",
        "assistant_response": "### 1",
        "feedback": "expected 2",
        "exit_code": 0,
        "stderr": "",
        "residue": [],
        "events": [{"role": "assistant", "content": "### 1"}],
        "usage": NO_USAGE,
    }
    batch = EvaluationBatch(outputs=[{}], scores=[0.0], trajectories=[trajectory])

    reflective = adapter.make_reflective_dataset(candidate, batch, [update])

    assert reflective == {
        update: [
            {
                "Inputs": "task",
                "Generated Outputs": "### 1",
                "Feedback": "expected 2",
                "Component role": role,
                "Harness trajectory": [{"role": "assistant", "content": "### 1"}],
            }
        ]
    }


def test_multi_node_candidate_renders_rules_and_skill_without_the_binding(load):
    adapter_module = load("harness.adapter")
    adapter = adapter_module.ReefAdapter(
        descriptor=get_adapter("pi"), task_model=BINDING, components=adapter_module.MULTI_NODE
    )

    files = adapter.render_candidate({"rules": "Global reasoning rules.", "skill": "# AIME skill\n\nProcedure."})

    assert files["pi-agent/AGENTS.md"] == "Global reasoning rules.\n"
    assert files["pi-agent/skills/aime-solver/SKILL.md"] == "# AIME skill\n\nProcedure.\n"
    assert "providers" not in json.loads(files["pi-agent/models.json"])


def test_pinned_gepa_round_robin_evolves_both_nodes_in_the_complete_tree(load):
    from gepa import optimize

    adapter_module = load("harness.adapter")
    seen = []

    def run(descriptor, files, prompt, **kwargs):
        rules = files.get("pi-agent/AGENTS.md", "")
        skill = files.get("pi-agent/skills/aime-solver/SKILL.md", "")
        seen.append((rules, skill))
        correct = "improved rules" in rules if prompt == "rules task" else "improved skill" in skill
        return pi_episode("### 1" if correct else "### 0")

    adapter = adapter_module.ReefAdapter(
        descriptor=get_adapter("pi"), task_model=BINDING, components=adapter_module.MULTI_NODE, episode_runner=run
    )
    dataset = [{"input": "rules task", "answer": "### 1"}, {"input": "skill task", "answer": "### 1"}]

    result = optimize(
        seed_candidate={"rules": "seed rules", "skill": "seed skill"},
        trainset=dataset,
        valset=dataset,
        adapter=adapter,
        reflection_lm=None,
        custom_candidate_proposer=lambda candidate, reflective, components: {
            components[0]: f"improved {components[0]}"
        },
        candidate_selection_strategy="current_best",
        module_selector="round_robin",
        reflection_minibatch_size=2,
        max_metric_calls=14,
        seed=0,
        skip_perfect_score=False,
    )

    assert result.num_candidates == 3
    assert result.best_candidate == {"rules": "improved rules", "skill": "improved skill"}
    assert result.val_aggregate_scores == [0.0, 0.5, 1.0]
    assert any("improved rules" in rules and "seed skill" in skill for rules, skill in seen)
    assert any("improved rules" in rules and "improved skill" in skill for rules, skill in seen)


@pytest.mark.parametrize(
    ("response", "score"),
    [("work\n### 17", 1.0), ("### 17\nmore text", 1.0), ("The answer is 17", 0.0), ("### 16", 0.0)],
)
def test_aime_scorer_matches_the_upstream_expected_string(load, response, score):
    assert load("harness.adapter").score_aime_answer("### 17", response) == score


# Data and the upstream reference -----------------------------------------


def test_dataset_loader_rejects_same_size_content_drift(load, monkeypatch):
    import datasets

    def load_dataset(name, *args, **kwargs):
        if name == "AI-MO/aimo-validation-aime":
            return [{"problem": f"source-{index}", "solution": "solution", "answer": 1} for index in range(90)]
        return [{"problem": f"test-{index}", "answer": 1} for index in range(30)]

    monkeypatch.setattr(datasets, "load_dataset", load_dataset)

    with pytest.raises(RuntimeError, match="content changed"):
        load("harness.data").load_aime_splits()


def test_official_adapter_keeps_the_upstream_evaluator_and_its_feedback(load):
    reference = load("harness.reference")
    responses = iter(["work\n### 42", "The answer is 42", "The answer is 41"])

    class Model:
        usage = Usage()

        def batch_complete(self, messages, max_workers):
            return [next(responses) for _ in messages]

    adapter = reference.OfficialAIMEAdapter(Model())
    with_context = {"input": "problem", "answer": "### 42", "additional_context": {"solution": "full solution"}}
    without_context = {"input": "problem", "answer": "### 42"}

    correct = adapter.evaluate([with_context], reference.OFFICIAL_SEED_CANDIDATE)
    incorrect = adapter.evaluate([with_context], reference.OFFICIAL_SEED_CANDIDATE, capture_traces=True)
    # AIME 2025 rows carry no context; upstream would raise while building
    # feedback and lose the response text, so the runner normalizes it.
    heldout = adapter.evaluate([without_context], reference.OFFICIAL_SEED_CANDIDATE, capture_traces=True)

    assert correct.scores == [1.0]
    assert incorrect.scores == [0.0]
    assert "solution: full solution" in incorrect.trajectories[0]["feedback"]
    assert heldout.scores == [0.0]
    assert heldout.outputs == [{"full_assistant_response": "The answer is 41"}]
    assert "correct answer is '### 42'" in heldout.trajectories[0]["feedback"]


# Search, promotion, and sealing -------------------------------------------


def test_pareto_specialists_are_retained_without_bypassing_the_promotion_gate(load):
    search = load("harness.search")
    result = gepa_result(
        candidates=[{"rules": "seed"}, {"rules": "algebra"}, {"rules": "geometry"}],
        scores=[0.5, 0.5, 0.5],
        fronts={"algebra": {1}, "geometry": {2}},
    )

    assert search.pareto_candidate_indices(result) == (1, 2)
    decision = search.decide_promotion(result)
    assert decision.selected is False
    assert decision.candidate_idx == 0
    assert "no candidate strictly improved" in decision.reason


def test_promotion_gate_selects_only_a_strict_validation_improvement(load):
    search = load("harness.search")
    result = gepa_result(candidates=[{"rules": "seed"}, {"rules": "better"}], scores=[0.25, 0.75], fronts={0: {1}})

    decision = search.decide_promotion(result)

    assert (decision.selected, decision.candidate_idx, decision.seed_score, decision.candidate_score) == (
        True,
        1,
        0.25,
        0.75,
    )


def test_test_split_is_unsealed_only_after_upstream_search_returns(load, monkeypatch, tmp_path):
    search = load("harness.search")
    events = []
    testset = [{"input": "sealed-test", "answer": "### 3"}]

    def optimize(**kwargs):
        events.append(("optimize", kwargs))
        assert all(example["input"] != "sealed-test" for example in kwargs["trainset"] + kwargs["valset"])
        return gepa_result(candidates=[{"rules": "seed"}, {"rules": "better"}], scores=[0.0, 1.0], fronts={0: {1}})

    class Adapter:
        def evaluate(self, batch, candidate, capture_traces=False):
            events.append(("evaluate", batch, candidate))
            return EvaluationBatch(outputs=[{}], scores=[1.0 if candidate["rules"] == "better" else 0.0])

    monkeypatch.setattr(search.gepa, "optimize", optimize)
    outcome = search.run_sealed_search(
        seed_candidate={"rules": "seed"},
        trainset=[{"input": "train", "answer": "### 1"}],
        valset=[{"input": "validation", "answer": "### 2"}],
        testset=testset,
        adapter=Adapter(),
        reflection_lm=None,
        max_metric_calls=8,
        seed=0,
        run_dir=tmp_path / "run",
    )

    assert [event[0] for event in events] == ["optimize", "evaluate", "evaluate"]
    assert events[1][1] == testset and events[2][1] == testset
    quickstart = events[0][1]
    assert (quickstart["candidate_selection_strategy"], quickstart["frontier_type"]) == ("pareto", "instance")
    assert (quickstart["cache_evaluation"], quickstart["skip_perfect_score"]) == (False, True)
    assert outcome.promotion.selected is True
    assert (outcome.frozen_test_score, outcome.selected_test_score) == (0.0, 1.0)


def test_smoke_policy_reflects_on_a_perfect_training_minibatch(load, tmp_path):
    search = load("harness.search")
    prompts = []

    class PerfectAdapter:
        propose_new_texts = None

        def evaluate(self, batch, candidate, capture_traces=False):
            return EvaluationBatch(
                outputs=[{} for _ in batch],
                scores=[1.0 for _ in batch],
                trajectories=[{"input": item["input"]} for item in batch] if capture_traces else None,
                objective_scores=[{"score": 1.0} for _ in batch],
                num_metric_calls=len(batch),
            )

        def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
            return {component: [{"Feedback": "refine anyway"}] for component in components_to_update}

    def reflection_lm(prompt):
        prompts.append(prompt)
        return "refined rules"

    outcome = search.run_sealed_search(
        seed_candidate={"rules": "seed rules"},
        trainset=[{"input": f"train-{index}", "answer": "### 1"} for index in range(3)],
        valset=[{"input": "validation", "answer": "### 1"}],
        testset=[{"input": "test", "answer": "### 1"}],
        adapter=PerfectAdapter(),
        reflection_lm=reflection_lm,
        max_metric_calls=5,
        seed=0,
        run_dir=tmp_path / "smoke",
        skip_perfect_score=False,
    )

    assert prompts
    assert outcome.result.candidates == [{"rules": "seed rules"}]


def test_pinned_gepa_checkpoint_resumes_without_replaying_work(tmp_path):
    from gepa import optimize

    class DeterministicAdapter:
        propose_new_texts = None

        def evaluate(self, batch, candidate, capture_traces=False):
            improved = candidate["rules"] == "improved"
            return EvaluationBatch(
                outputs=[{"improved": improved} for _ in batch],
                scores=[1.0 if improved else 0.0 for _ in batch],
                trajectories=[{"input": item["input"]} for item in batch] if capture_traces else None,
                num_metric_calls=len(batch),
            )

        def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
            return {component: [{"Feedback": "use improved"}] for component in components_to_update}

    settings = {
        "seed_candidate": {"rules": "seed"},
        "trainset": [{"input": "deterministic"}],
        "valset": [{"input": "deterministic"}],
        "adapter": DeterministicAdapter(),
        "reflection_lm": None,
        "custom_candidate_proposer": lambda candidate, reflective, components: {components[0]: "improved"},
        "reflection_minibatch_size": 1,
        "run_dir": str(tmp_path / "resume"),
        "seed": 0,
        "skip_perfect_score": False,
    }

    first = optimize(**settings, max_metric_calls=4)
    resumed = optimize(**settings, max_metric_calls=0)

    assert first.num_candidates == 2
    assert first.best_candidate == {"rules": "improved"}
    assert (resumed.candidates, resumed.parents) == (first.candidates, first.parents)
    assert resumed.total_metric_calls == first.total_metric_calls


# Held-out checkpoints -----------------------------------------------------


def test_heldout_checkpoints_resume_without_repeating_completed_examples(load, tmp_path):
    heldout = load("harness.heldout")
    calls = []

    class InterruptingAdapter:
        interrupted = False

        def evaluate(self, batch, candidate, capture_traces=False):
            calls.append(batch[0]["input"])
            if batch[0]["input"] == "two" and not self.interrupted:
                self.interrupted = True
                raise RuntimeError("simulated interruption")
            return EvaluationBatch(outputs=[{"input": batch[0]["input"]}], scores=[float(batch[0]["answer"])])

    batch = [{"input": "one", "answer": "1"}, {"input": "two", "answer": "0"}, {"input": "three", "answer": "1"}]
    evaluator = heldout.CheckpointedEvaluator(InterruptingAdapter(), tmp_path / "checkpoints")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        evaluator.evaluate("selected", batch, {"rules": "candidate"})
    resumed = evaluator.evaluate("selected", batch, {"rules": "candidate"})

    assert calls == ["one", "two", "two", "three"]
    assert resumed.scores == [1.0, 0.0, 1.0]
    assert evaluator.evaluate("selected", batch, {"rules": "candidate"}).scores == resumed.scores
    assert len(calls) == 4
    with pytest.raises(RuntimeError, match="identity changed"):
        evaluator.evaluate("selected", batch, {"rules": "different candidate"})


def test_heldout_checkpoints_record_failures_only_when_asked_and_never_the_spend_cap(load, tmp_path):
    heldout = load("harness.heldout")
    models = load("harness.models")

    class Adapter:
        def evaluate(self, batch, candidate, capture_traces=False):
            if any(item["input"] == "fails" for item in batch):
                raise RuntimeError("provider failure")
            return EvaluationBatch(outputs=[{"answer": "1"} for _ in batch], scores=[1.0 for _ in batch])

    batch = [{"input": "works", "answer": "1"}, {"input": "fails", "answer": "1"}]
    strict = heldout.CheckpointedEvaluator(Adapter(), tmp_path / "strict", batch_size=2)
    with pytest.raises(RuntimeError, match="provider failure"):
        strict.evaluate("official", batch, {"system_prompt": "prompt"})

    lenient = heldout.CheckpointedEvaluator(Adapter(), tmp_path / "lenient", batch_size=2, failure_score=0.0)
    result = lenient.evaluate("official", batch, {"system_prompt": "prompt"})
    assert result.scores == [1.0, 0.0]
    assert result.outputs[1] == {"error": "RuntimeError"}

    class StopAdapter:
        def evaluate(self, batch, candidate, capture_traces=False):
            raise models.SpendCapReached("stop")

    stopped = heldout.CheckpointedEvaluator(StopAdapter(), tmp_path / "stopped", failure_score=0.0)
    with pytest.raises(models.SpendCapReached, match="stop"):
        stopped.evaluate("official", batch[:1], {"system_prompt": "prompt"})
    assert not (tmp_path / "stopped" / "official" / "example-0000.json").exists()


def test_heldout_checkpoints_batch_pending_examples(load, tmp_path):
    heldout = load("harness.heldout")
    calls = []

    class Adapter:
        def evaluate(self, batch, candidate, capture_traces=False):
            calls.append([example["input"] for example in batch])
            return EvaluationBatch(
                outputs=[{"input": example["input"]} for example in batch],
                scores=[float(example["answer"]) for example in batch],
            )

    batch = [{"input": f"example-{index}", "answer": str(index % 2)} for index in range(5)]
    evaluator = heldout.CheckpointedEvaluator(Adapter(), tmp_path / "checkpoints", batch_size=2)

    result = evaluator.evaluate("batched", batch, {"rules": "candidate"})

    assert calls == [["example-0", "example-1"], ["example-2", "example-3"], ["example-4"]]
    assert result.scores == [0.0, 1.0, 0.0, 1.0, 0.0]
    assert evaluator.evaluate("batched", batch, {"rules": "candidate"}).scores == result.scores
    assert len(calls) == 3


# Accounting ---------------------------------------------------------------


def test_pi_usage_and_price_estimate_include_cached_and_reasoning_tokens(load):
    models = load("harness.models")
    trajectory = [
        {"message": {"role": "assistant", "usage": {"input": 60, "cacheRead": 40, "output": 20, "reasoning": 5}}}
    ]

    usage = models.trajectory_usage(trajectory)
    price = models.ModelPrice(1.0, 0.5, 2.0, "test")

    assert usage == {
        "requests": 1,
        "input_tokens": 100,
        "cached_input_tokens": 40,
        "output_tokens": 20,
        "reasoning_tokens": 5,
    }
    assert price.estimate(usage) == pytest.approx(0.00012)


def test_tracked_chat_model_records_usage_and_spend(load):
    models = load("harness.models")
    costs = []

    class Cap:
        def before_call(self):
            costs.append("before")

        def record_call(self, cost_usd):
            costs.append(cost_usd)

    class Binding:
        def complete(self, body):
            assert body == {"messages": [{"role": "user", "content": "prompt"}]}
            return {
                "choices": [{"message": {"content": "answer"}}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "prompt_tokens_details": {"cached_tokens": 3},
                    "completion_tokens_details": {"reasoning_tokens": 2},
                },
            }

    model = models.TrackedChatModel(Binding(), price=models.REFLECTION_MODEL_PRICE, spend_cap=Cap())

    assert model("prompt") == "answer"
    usage = model.usage.snapshot()
    assert usage == {
        "requests": 1,
        "input_tokens": 10,
        "cached_input_tokens": 3,
        "output_tokens": 4,
        "reasoning_tokens": 2,
    }
    assert costs == ["before", pytest.approx(models.REFLECTION_MODEL_PRICE.estimate(usage))]


def test_usage_ledger_persists_across_restarts(load, tmp_path):
    models = load("harness.models")
    usage = {"requests": 1, "input_tokens": 10, "cached_input_tokens": 3, "output_tokens": 4, "reasoning_tokens": 2}

    models.UsageLedger(models.TASK_MODEL_PRICE, tmp_path / "task-usage.json").add(usage)
    resumed = models.UsageLedger(models.TASK_MODEL_PRICE, tmp_path / "task-usage.json")
    resumed.add(usage)

    assert resumed.snapshot()["input_tokens"] == 20
    assert (
        json.loads((tmp_path / "task-usage.json").read_text())["pricing"]["source"] == models.TASK_MODEL_PRICE.source
    )


def test_spend_cap_persists_and_stops_before_the_next_call(load, tmp_path):
    models = load("harness.models")

    cap = models.SpendCap(tmp_path / "observed-cost.json", 1.0)
    cap.before_call()
    cap.record_call(0.6)
    resumed = models.SpendCap(tmp_path / "observed-cost.json", 1.0)
    resumed.before_call()
    resumed.record_call(0.5)

    assert resumed.observed_usd == pytest.approx(1.1)
    assert resumed.completed_calls == 2
    with pytest.raises(models.SpendCapReached, match="no new model call"):
        resumed.before_call()


# The baseline gate --------------------------------------------------------


def test_baseline_gate_accepts_a_stochastic_but_aligned_reef_arm(load, tmp_path):
    baseline = load("harness.baseline")
    valset = [{"input": "one", "answer": "### 1"}, {"input": "two", "answer": "### 2"}]
    valset.extend({"input": f"same-{index}", "answer": "### 0"} for index in range(43))

    result = baseline.run_baseline_alignment(
        official=FakeAdapter(lambda task: {"one": 1.0}.get(task, 0.0)),
        reef=FakeAdapter(lambda task: {"two": 1.0}.get(task, 0.0)),
        valset=valset,
        output_dir=tmp_path,
        workers=1,
    )

    assert result["baseline_aligned"] is True
    assert result["evaluations_per_arm"] == 450
    assert result["reef_minus_official_score"] == 0.0
    assert result["lower_confidence_bound"] > -0.10
    assert json.loads((tmp_path / "result.json").read_text())["baseline_aligned"] is True


def test_baseline_gate_stops_before_search_on_a_material_gap(load, tmp_path):
    baseline = load("harness.baseline")
    valset = [{"input": f"problem-{index}", "answer": "### answer"} for index in range(45)]

    def correct_below(limit):
        return lambda task: float(int(task.split("-")[-1]) < limit)

    with pytest.raises(RuntimeError, match="no GEPA search was started"):
        baseline.run_baseline_alignment(
            official=FakeAdapter(correct_below(20)),
            reef=FakeAdapter(correct_below(13)),
            valset=valset,
            output_dir=tmp_path,
            workers=1,
        )

    result = json.loads((tmp_path / "result.json").read_text())
    assert result["baseline_aligned"] is False
    assert result["official_score"] == pytest.approx(20 / 45)
    assert result["reef_score"] == pytest.approx(13 / 45)
    assert result["lower_confidence_bound"] == pytest.approx(-0.247361, abs=1e-6)


def test_baseline_gate_rejects_a_shared_zero_from_a_failed_reef_episode(load, tmp_path):
    baseline = load("harness.baseline")

    with pytest.raises(RuntimeError, match="10 Reef episodes failed"):
        baseline.run_baseline_alignment(
            official=FakeAdapter(lambda task: 0.0),
            reef=FakeAdapter(lambda task: 0.0, exit_code=-1),
            valset=[{"input": "one", "answer": "### 1"}],
            output_dir=tmp_path,
            workers=1,
        )

    result = json.loads((tmp_path / "result.json").read_text())
    assert result["failed_reef_indices"] == list(range(10))
    assert result["baseline_aligned"] is False


# Publication and reports --------------------------------------------------


def test_publication_excludes_the_model_binding_and_records_release_identities(load, monkeypatch, tmp_path):
    adapter_module = load("harness.adapter")
    publication = load("harness.publication")
    observed = {}

    class Backend:
        def __init__(self, scenario, repository, **kwargs):
            observed["scenario"] = scenario

        def fork(self, metadata):
            return ArtifactRef("parent-content", "parent-release", None)

        def publish(self, artifact, expected_parent):
            observed["expected_parent"] = expected_parent.release_id
            observed["models"] = (artifact.local_path / "pi-agent" / "models.json").read_text()
            observed["metadata"] = dict(artifact.metadata)
            return ArtifactRef("selected-content", "selected-release", expected_parent.release_id)

    monkeypatch.setattr(publication, "GitLFSRepositoryBackend", Backend)
    adapter = adapter_module.ReefAdapter(
        descriptor=get_adapter("pi"),
        task_model=ModelBinding("http://model.test", "task-model", api_key="must-not-publish"),
        components=adapter_module.MULTI_NODE,
    )

    published = publication.publish_candidate(
        adapter=adapter,
        candidate={"rules": "rules", "skill": "skill"},
        output_dir=tmp_path / "result",
        scenario="gepa-test",
        metadata={"score": 1.0},
    )

    assert (published["content_id"], published["release_id"]) == ("selected-content", "selected-release")
    assert published["parent_release_id"] == "parent-release"
    assert observed["expected_parent"] == "parent-release"
    assert json.loads(observed["models"]) == {}
    tree = tmp_path / "result" / "published-composition"
    assert "must-not-publish" not in "".join(path.read_text() for path in tree.rglob("*") if path.is_file())
    assert observed["metadata"] == {"score": 1.0, "method": "gepa", "scenario": "gepa-test"}
    assert (
        publication.publish_candidate(
            adapter=adapter, candidate={}, output_dir=tmp_path / "result", scenario="gepa-test", metadata={}
        )
        == published
    )


def test_real_reef_git_lfs_publication_smoke(load, tmp_path):
    if shutil.which("git-lfs") is None:
        pytest.skip("git-lfs is required for the durable Reef artifact smoke test")
    adapter = load("harness.adapter").ReefAdapter(descriptor=get_adapter("pi"), task_model=BINDING)
    publication = load("harness.publication")

    published = publication.publish_candidate(
        adapter=adapter,
        candidate={"rules": "published rules"},
        output_dir=tmp_path / "durable",
        scenario="gepa-durable-smoke",
        metadata={"kind": "test"},
    )

    assert published["content_id"].startswith("content:")
    assert len(published["release_id"]) == 40
    assert len(published["parent_release_id"]) == 40
    assert Path(published["repository"]).is_dir()
    assert (tmp_path / "durable" / "publication.json").is_file()


def test_aggregate_report_keeps_negative_deltas(load, tmp_path):
    reporting = load("harness.reporting")
    for seed, frozen, selected, promoted, cost in [
        (0, 0.2, 0.4, True, 1.5),
        (1, 0.5, 0.3, True, 2.5),
        (2, 0.1, 0.1, False, 1.0),
    ]:
        run_dir = tmp_path / "rules" / f"seed-{seed}"
        run_dir.mkdir(parents=True)
        summary = {
            "frozen_test_score": frozen,
            "selected_test_score": selected,
            "promotion": {"selected": promoted},
            "estimated_cost_usd": {"total": cost},
            "wall_time_s": 10 + seed,
        }
        (run_dir / "summary.json").write_text(json.dumps(summary))

    reporting.write_aggregate_report(output_dir=tmp_path, cells=("rules",), seeds=(0, 1, 2))

    rules = json.loads((tmp_path / "results.json").read_text())["cells"]["rules"]
    assert rules["frozen_test_score_mean"] == pytest.approx(0.8 / 3)
    assert rules["test_delta_mean"] == pytest.approx(0.0)
    assert rules["promotion_rate"] == pytest.approx(2 / 3)
    assert rules["estimated_cost_usd_total"] == pytest.approx(5.0)
    assert rules["runs"][1]["test_delta"] == pytest.approx(-0.2)


# The driver ---------------------------------------------------------------


def session(run, **overrides):
    settings = {
        "api_key": "dummy",
        "pi_binary": "fake-pi",
        "spend_cap": None,
        "budget": 8,
        "smoke": False,
        "workers": {"reference": 10, "reef": 128, "heldout": 64},
    }
    return run.Session(**{**settings, **overrides})


def test_run_identity_refuses_an_incompatible_resume(load, tmp_path):
    files = load("harness.files")
    path = tmp_path / "run-identity.json"

    files.write_once(path, {"smoke": False, "workers": {"reef": 32}}, "run identity")
    files.write_once(path, {"smoke": False, "workers": {"reef": 32}}, "run identity")
    with pytest.raises(RuntimeError, match="does not match"):
        files.write_once(path, {"smoke": True, "workers": {"reef": 32}}, "run identity")


def test_dry_run_prints_the_plan_without_loading_the_dataset(load, monkeypatch, capsys):
    run = load("run")
    pins = {"gepa_version": "g", "pi_version": "p", "reef_commit": "r", "reef_dirty": False}
    monkeypatch.setattr(run, "verify_pins", lambda binary: pins)
    monkeypatch.setattr(run, "load_aime_splits", lambda: pytest.fail("a dry run must not load the dataset"))
    monkeypatch.setattr(sys, "argv", ["run.py", "--dry-run", "--cell", "rules", "--seeds", "0"])

    run.main()

    plan = json.loads(capsys.readouterr().out)
    assert plan["planned_task_evaluations"] == 150 + 2 * 150 + 2 * 10 * 45
    assert plan["workers"] == {"reference": 10, "reef": 32, "heldout": 16}
    assert plan["pins"] == pins


def test_verify_pins_refuses_another_pi_version(load, monkeypatch):
    run = load("run")
    monkeypatch.setattr(run, "_command", lambda command, cwd: "0.0.1" if command[0] == "fake-pi" else "")

    with pytest.raises(SystemExit, match="Pi version is"):
        run.verify_pins("fake-pi")


def test_reference_cell_uses_the_quickstart_settings(load, monkeypatch, tmp_path):
    run = load("run")
    observed = {}

    class Model:
        usage = Usage()

        def __init__(self, model, **kwargs):
            observed["task_model"] = (model, kwargs)

    class Official:
        def __init__(self, model, max_workers):
            self.usage = model.usage
            observed["official_workers"] = max_workers

    monkeypatch.setattr(run, "TrackedGEPALM", Model)
    monkeypatch.setattr(run, "OfficialAIMEAdapter", Official)

    class Reflection:
        usage = Usage()

        def __init__(self, binding, **kwargs):
            observed["reflection"] = binding

    monkeypatch.setattr(run, "TrackedChatModel", Reflection)
    monkeypatch.setattr(run, "run_sealed_search", lambda **kwargs: observed.setdefault("search", kwargs))
    monkeypatch.setattr(run, "write_search_report", lambda **kwargs: observed.setdefault("report", kwargs))
    dataset = [{"input": "example", "answer": "### 1"}]

    run.run_reference(session(run), 0, tmp_path, dataset, dataset, dataset)

    search = observed["search"]
    assert search["seed_candidate"] == run.OFFICIAL_SEED_CANDIDATE
    assert (search["max_metric_calls"], search["seed"], search["skip_perfect_score"]) == (8, 0, True)
    assert search["heldout_evaluator"].batch_size == 64
    assert search["heldout_evaluator"].failure_score == 0.0
    model, kwargs = observed["task_model"]
    assert model == run.TASK_MODEL
    assert kwargs["max_completion_tokens"] is None
    assert "temperature" not in kwargs
    assert observed["official_workers"] == 10
    assert observed["reflection"].model == run.REFLECTION_MODEL
    assert observed["report"]["config"]["pi_task_max_tokens"] is None


def test_reef_search_cell_persists_usage_before_search(load, monkeypatch, tmp_path):
    run = load("run")

    class SearchReached(RuntimeError):
        pass

    def stop_before_search(**kwargs):
        assert kwargs["adapter"].usage.path == tmp_path / "task-usage.json"
        assert kwargs["adapter"].max_workers == 128
        assert kwargs["heldout_evaluator"].batch_size == 64
        assert kwargs["skip_perfect_score"] is False
        raise SearchReached

    monkeypatch.setattr(run, "run_sealed_search", stop_before_search)
    dataset = [{"input": "example", "answer": "### 1"}]

    with pytest.raises(SearchReached):
        run.run_reef_search(session(run, smoke=True), "rules", 0, tmp_path, dataset, dataset, dataset)
    assert json.loads((tmp_path / "task-usage.json").read_text())["usage"] == NO_USAGE
