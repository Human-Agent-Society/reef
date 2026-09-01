"""Deterministic coverage for the Meta-Harness reproduction example."""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "recipes" / "meta_harness" / "examples" / "terminal_bench"


@pytest.fixture
def config_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    for name in [name for name in sys.modules if name == "harness" or name.startswith("harness.")]:
        del sys.modules[name]
    return importlib.import_module("harness.config")


@pytest.fixture
def meta_modules(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    for name in [name for name in sys.modules if name == "harness" or name.startswith("harness.")]:
        del sys.modules[name]
    return SimpleNamespace(
        budget=importlib.import_module("harness.budget"),
        codex=importlib.import_module("harness.codex"),
        composition=importlib.import_module("harness.composition"),
        evaluation=importlib.import_module("harness.evaluation"),
        harbor_eval=importlib.import_module("harness.harbor_eval"),
        pi_bridge=importlib.import_module("harness.pi_bridge"),
        publication=importlib.import_module("harness.publication"),
        search=importlib.import_module("harness.search"),
        workspace=importlib.import_module("harness.workspace"),
    )


@pytest.fixture
def runner_modules(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    sys.modules.pop("run", None)
    for name in [name for name in sys.modules if name == "harness" or name.startswith("harness.")]:
        del sys.modules[name]
    run = importlib.import_module("run")
    return SimpleNamespace(
        run=run,
        harbor_eval=importlib.import_module("harness.harbor_eval"),
        search=importlib.import_module("harness.search"),
    )


def test_reproduction_defaults_are_exact_and_secret_free(config_module):
    config = config_module.ExperimentConfig()

    assert config_module.REEF_COMMIT == "6b0d9a1345191a0df1a7e324fa09875e8581a83f"
    assert config_module.META_HARNESS_COMMIT == "95175f70c758dd1145b395edfe8b67e6f9d80fbd"
    assert config_module.TERMINAL_BENCH_COMMIT == "1a6ffa9674b571da0ed040c470cb40c4d85f9b9b"
    assert config_module.TERMINAL_BENCH_DATASET_URL == ("https://github.com/laude-institute/terminal-bench-2.git")
    assert config_module.TERMINAL_BENCH_DATASET_COMMIT == "69671fbaac6d67a7ef0dfec016cc38a64ef7a77c"
    assert config_module.HARBOR_VERSION == "0.3.0"
    assert config_module.TERMINAL_BENCH_VERSION == "0.2.18"
    assert config_module.PI_VERSION == "0.84.2"
    assert config_module.TARGET_MODEL == "gpt-5.4-mini-2026-03-17"
    assert config_module.TARGET_MODEL_PRICING.to_jsonable() == {
        "model": "gpt-5.4-mini-2026-03-17",
        "input_usd_per_million": 0.75,
        "cached_input_usd_per_million": 0.075,
        "output_usd_per_million": 4.5,
        "source_url": "https://developers.openai.com/api/docs/models/gpt-5.4-mini",
        "observed_at": "2026-08-30",
    }
    assert config_module.PROPOSER_MODEL == "gpt-5.5"
    assert config_module.PROPOSER_REASONING_EFFORT == "high"
    assert len(config_module.HARD_TASKS) == 30
    assert len(set(config_module.HARD_TASKS)) == 30
    assert config_module.TRAIN_TASKS + config_module.DEV_TASKS + config_module.TEST_TASKS == (config_module.HARD_TASKS)
    assert {len(config_module.TRAIN_TASKS), len(config_module.DEV_TASKS), len(config_module.TEST_TASKS)} == {10}
    assert config_module.TRAIN_TASKS[:1] == config_module.SMOKE_TRAIN_TASKS
    assert config_module.DEV_TASKS[:1] == config_module.SMOKE_DEV_TASKS
    assert config_module.TEST_TASKS[:1] == config_module.SMOKE_TEST_TASKS
    assert set(asdict(config)) == {
        "target_model",
        "proposer_model",
        "proposer_reasoning_effort",
        "rounds",
        "trials_per_task",
        "target_api_key_env",
        "target_base_url",
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"rounds": 1}, "at least two rounds"),
        ({"trials_per_task": 1}, "at least two trials"),
        ({"target_base_url": "api.openai.com"}, "HTTP"),
        ({"target_model": "gpt-unpriced"}, "pricing snapshot"),
        ({"proposer_reasoning_effort": "ultra"}, "reasoning effort"),
    ],
)
def test_reproduction_config_rejects_nonconforming_inputs(config_module, overrides, message):
    with pytest.raises(ValueError, match=message):
        config_module.ExperimentConfig(**overrides)


def _evaluation(modules, split, reward, task="task"):
    trial = modules.evaluation.TrialEvidence(
        task_id=task,
        trial=0,
        reward=reward,
        trajectory=({"role": "assistant", "content": "observed"},),
        usage={"input_tokens": 10, "output_tokens": 2},
        estimated_cost_usd=0.01,
        wall_time_s=1.5,
    )
    return modules.evaluation.EvaluationResult(split=split, trials=(trial,))


def test_composition_is_content_addressed_validated_and_provider_free(meta_modules):
    from reef.harness.adapters import get_adapter

    candidate = meta_modules.composition.genesis_composition()
    same = meta_modules.composition.CompositionCandidate.from_value(candidate.composition)
    rendered = candidate.render(get_adapter("pi"))

    assert candidate.content_hash == same.content_hash
    assert len(candidate.content_hash) == 64
    assert "pi-agent/AGENTS.md" in rendered
    assert "pi-agent/skills/terminal-task/SKILL.md" in rendered
    assert json.loads(rendered["pi-agent/models.json"]) == {}

    for kind, config in (
        ("config", {"data": {"skills": ["/host/path"]}}),
        ("code_extension", {"name": "leaky", "code": "export default () => {};"}),
    ):
        with pytest.raises(ValueError, match="sealed experiment"):
            meta_modules.composition.CompositionCandidate.from_value({"nodes": [{"kind": kind, "config": config}]})

    with pytest.raises(ValueError, match="unknown kind"):
        meta_modules.composition.CompositionCandidate.from_value(
            {
                "nodes": [
                    {
                        "kind": "not-a-node",
                        "config": {},
                    }
                ]
            }
        )


def test_workspace_retains_search_evidence_but_rejects_test_data(meta_modules, tmp_path):
    from reef.harness.adapters import get_adapter

    workspace = meta_modules.workspace.EvolutionWorkspace(tmp_path / "run", get_adapter("pi"))
    candidate = meta_modules.composition.genesis_composition()
    workspace.record_candidate(candidate, _evaluation(meta_modules, "train", 0.5), round_index=0, role="genesis")
    workspace.record_candidate(candidate, _evaluation(meta_modules, "dev", 0.25), round_index=0, role="genesis")

    candidate_dir = workspace.history_dir / candidate.content_hash
    assert (candidate_dir / "candidate.json").is_file()
    assert (candidate_dir / "evaluations" / "round-0000-train.json").is_file()
    assert (candidate_dir / "evaluations" / "round-0000-dev.json").is_file()
    assert "observed" in (candidate_dir / "evaluations" / "round-0000-train.json").read_text()
    assert "observed" not in (candidate_dir / "evaluations" / "round-0000-dev.json").read_text()
    with pytest.raises(ValueError, match="sealed"):
        workspace.record_candidate(
            candidate,
            _evaluation(meta_modules, "test", 1.0),
            round_index=2,
            role="selected",
        )


def test_full_history_proposal_can_branch_from_a_non_incumbent(meta_modules, tmp_path):
    from reef.harness.adapters import get_adapter

    workspace = meta_modules.workspace.EvolutionWorkspace(tmp_path / "run", get_adapter("pi"))
    genesis = meta_modules.composition.genesis_composition()
    specialist_value = genesis.composition
    specialist_value["nodes"][0]["config"]["text"] = "A retained non-incumbent specialist."
    specialist = meta_modules.composition.CompositionCandidate.from_value(
        specialist_value,
        parent_hashes=(genesis.content_hash,),
    )
    workspace.write_population(
        [
            {"candidate_hash": genesis.content_hash, "alias": "genesis", "score": 0.8},
            {"candidate_hash": specialist.content_hash, "alias": "specialist", "score": 0.4},
        ],
        incumbent_hash=genesis.content_hash,
    )
    proposal_dir = workspace.begin_round(1) / "branch-specialist"
    proposal_dir.mkdir()
    proposal_value = specialist.composition
    proposal_value["nodes"][1]["config"]["text"] += "\n\nVerify long-running commands."
    (proposal_dir / "composition.json").write_text(json.dumps(proposal_value))
    (workspace.root / "pending_eval.json").write_text(
        json.dumps(
            {
                "iteration": 1,
                "candidates": [
                    {
                        "name": "branch-specialist",
                        "base_candidates": ["specialist"],
                        "hypothesis": "The retained specialist has useful terminal discipline.",
                        "changes": "Add verification guidance without changing its other nodes.",
                    }
                ],
            }
        )
    )

    proposals = workspace.collect_proposals(1)

    assert len(proposals) == 1
    assert proposals[0].primary_parent_hash == specialist.content_hash
    assert proposals[0].primary_parent_hash != genesis.content_hash
    assert proposals[0].source_path.is_file()


def test_workspace_detects_history_mutation_and_recovers_completed_rounds(meta_modules, tmp_path):
    from reef.harness.adapters import get_adapter

    workspace = meta_modules.workspace.EvolutionWorkspace(tmp_path / "run", get_adapter("pi"))
    candidate = meta_modules.composition.genesis_composition()
    workspace.record_candidate(candidate, _evaluation(meta_modules, "train", 0.0), round_index=0, role="genesis")
    workspace.begin_round(1)
    workspace.transition_round(1, "proposing")
    workspace.transition_round(1, "proposed")
    workspace.transition_round(1, "validating")
    workspace.transition_round(1, "evaluating")
    workspace.transition_round(1, "committed")
    assert workspace.completed_rounds() == (1,)

    before = workspace.readonly_snapshot(writable_round=2)
    candidate_path = workspace.history_dir / candidate.content_hash / "candidate.json"
    candidate_path.write_text("{}\n")
    with pytest.raises(meta_modules.workspace.WorkspaceIntegrityError, match="read-only state"):
        workspace.assert_readonly_unchanged(before, writable_round=2)


class _DeterministicEvaluator:
    def __init__(self, modules):
        self.modules = modules
        self.calls = []

    def evaluate(self, candidate, *, split, round_index):
        self.calls.append((candidate.content_hash, split, round_index))
        text = candidate.canonical_json
        reward = 0.9 if "branch-improvement" in text else (0.4 if "specialist" in text else 0.8)
        return _evaluation(self.modules, split, reward, task=f"{split}-task")


class _ScriptedProposer:
    def __init__(self, modules, *, branch_from_specialist=True):
        self.modules = modules
        self.branch_from_specialist = branch_from_specialist
        self.surfaces = []

    def propose(self, *, surface, prompt, session_dir, round_index):
        self.surfaces.append(surface)
        catalog = json.loads((surface / "candidate-catalog.json").read_text())
        if round_index == 1:
            base = "genesis"
            source_hash = catalog["incumbent_hash"]
            marker = "specialist"
            name = "specialist"
        else:
            specialist = next(row for row in catalog["candidates"] if row["alias"] == "specialist")
            base = "specialist" if self.branch_from_specialist else catalog["incumbent_hash"]
            source_hash = specialist["candidate_hash"] if self.branch_from_specialist else catalog["incumbent_hash"]
            marker = "branch-improvement"
            name = "branch-specialist"
            assert (surface / "history" / specialist["candidate_hash"]).is_dir()

        value = json.loads((surface / "history" / source_hash / "candidate.json").read_text())["composition"]
        value["nodes"][0]["config"]["text"] += f" {marker}"
        proposal_dir = surface / "proposals" / f"round-{round_index:04d}" / name
        proposal_dir.mkdir(parents=True, exist_ok=True)
        (proposal_dir / "composition.json").write_text(json.dumps(value))
        (surface / "pending_eval.json").write_text(
            json.dumps(
                {
                    "iteration": round_index,
                    "candidates": [
                        {
                            "name": name,
                            "base_candidates": [base],
                            "hypothesis": f"Try {marker} guidance.",
                            "changes": f"Add the {marker} marker.",
                        }
                    ],
                }
            )
        )
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "events.jsonl").write_text('{"type":"complete"}\n')
        return self.modules.search.ProposerSession(session_id=f"session-{round_index}")


def test_full_history_search_branches_from_retained_candidate_and_resumes(meta_modules, tmp_path):
    from reef.harness.adapters import get_adapter

    workspace = meta_modules.workspace.EvolutionWorkspace(tmp_path / "run", get_adapter("pi"))
    evaluator = _DeterministicEvaluator(meta_modules)
    proposer = _ScriptedProposer(meta_modules)
    genesis = meta_modules.composition.genesis_composition()

    first = meta_modules.search.MetaHarnessSearch(
        workspace=workspace,
        evaluator=evaluator,
        proposer=proposer,
        mode="full_history",
        rounds=1,
    ).run(genesis)
    resumed = meta_modules.search.MetaHarnessSearch(
        workspace=workspace,
        evaluator=evaluator,
        proposer=proposer,
        mode="full_history",
        rounds=2,
    ).run(genesis)

    assert first.selected.content_hash == genesis.content_hash
    assert resumed.resumed is True
    assert len(resumed.population) == 3
    specialist, branch = resumed.population[1:]
    assert specialist.outcome == "retained"
    assert branch.candidate.parent_hashes == (specialist.candidate.content_hash,)
    assert resumed.selected.content_hash == branch.candidate.content_hash
    assert [call[2] for call in evaluator.calls].count(0) == 2
    assert {call[1] for call in evaluator.calls} == {"train", "dev"}
    assert not any(call[1] == "test" for call in evaluator.calls)
    state = workspace.read_search_state()
    assert state["completed_rounds"] == 2
    assert len(state["proposer_sessions"]) == 2


def test_search_resumes_frozen_proposal_after_evaluation_interruption(meta_modules, tmp_path):
    from reef.harness.adapters import get_adapter

    executed_trials = []

    def execute_trial(candidate, candidate_path, task_id, trial_index, round_index, trial_dir):
        executed_trials.append((candidate.content_hash, task_id, trial_index, round_index))
        reward = 0.4 if "specialist" in candidate.canonical_json else 0.8
        return meta_modules.harbor_eval.HarborTrialOutcome(
            reward=reward,
            trajectory=({"task": task_id, "trial": trial_index},),
            usage={"input_tokens": 10, "output_tokens": 2},
            estimated_cost_usd=0.01,
            wall_time_s=0.01,
            verifier={"rewards": {"reward": reward}, "task_checksum": f"checksum-{task_id}"},
        )

    evaluation_root = tmp_path / "private-evaluations"

    def evaluator_for(ledger):
        return meta_modules.harbor_eval.HarborEvaluator(
            splits={"train": ("train-task",), "dev": ("dev-task",), "test": ("test-task",)},
            trials_per_task=2,
            output_dir=evaluation_root,
            target_model="gpt-test",
            target_base_url="https://example.invalid",
            target_api_key_env="META_TEST_OPENAI_KEY",
            pi_binary=tmp_path / "pi",
            trial_executor=execute_trial,
            before_trial=ledger.before_trial,
            after_trial=ledger.record_trial,
            budget_namespace="full-history",
        )

    workspace = meta_modules.workspace.EvolutionWorkspace(tmp_path / "run", get_adapter("pi"))
    proposer = _ScriptedProposer(meta_modules)
    genesis = meta_modules.composition.genesis_composition()
    ledger_path = tmp_path / "observed-cost.json"
    first_ledger = meta_modules.budget.ObservedCostLedger(ledger_path, 0.05)

    with pytest.raises(meta_modules.budget.SpendCapReached, match="no new trial"):
        meta_modules.search.MetaHarnessSearch(
            workspace=workspace,
            evaluator=evaluator_for(first_ledger),
            proposer=proposer,
            mode="full_history",
            rounds=1,
        ).run(genesis)

    interrupted_state = workspace.read_search_state()
    assert interrupted_state["completed_rounds"] == 0
    assert len(interrupted_state["proposer_sessions"]) == 1
    assert workspace.round_state(1) == "failed"
    assert workspace.resumable_proposal(1) is not None
    assert len(executed_trials) == 5

    resumed_ledger = meta_modules.budget.ObservedCostLedger(ledger_path, 0.20)
    resumed = meta_modules.search.MetaHarnessSearch(
        workspace=workspace,
        evaluator=evaluator_for(resumed_ledger),
        proposer=proposer,
        mode="full_history",
        rounds=1,
    ).run(genesis)

    assert resumed.resumed is True
    assert resumed.selected.content_hash == genesis.content_hash
    assert len(resumed.population) == 2
    assert len(proposer.surfaces) == 1
    assert len(executed_trials) == 8
    assert resumed_ledger.observed_cost_usd == pytest.approx(0.08)
    assert workspace.round_state(1) == "committed"
    final_state = workspace.read_search_state()
    assert final_state["completed_rounds"] == 1
    assert len(final_state["proposer_sessions"]) == 1
    events = [json.loads(line) for line in (workspace.root / "proposal-events.jsonl").read_text().splitlines()]
    assert [event["status"] for event in events] == ["failed:SpendCapReached", "rejected"]


def test_search_retries_failed_proposer_without_consuming_round(meta_modules, tmp_path):
    from reef.harness.adapters import get_adapter

    scripted = _ScriptedProposer(meta_modules)

    class FailOnceProposer:
        def __init__(self):
            self.calls = 0

        def propose(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                surface = kwargs["surface"]
                partial = surface / "proposals" / "round-0001" / "partial"
                partial.mkdir(parents=True)
                (partial / "composition.json").write_text("{}\n")
                raise RuntimeError("temporary proposer failure")
            return scripted.propose(**kwargs)

    workspace = meta_modules.workspace.EvolutionWorkspace(tmp_path / "run", get_adapter("pi"))
    proposer = FailOnceProposer()
    genesis = meta_modules.composition.genesis_composition()

    with pytest.raises(RuntimeError, match="temporary proposer failure"):
        meta_modules.search.MetaHarnessSearch(
            workspace=workspace,
            evaluator=_DeterministicEvaluator(meta_modules),
            proposer=proposer,
            mode="full_history",
            rounds=1,
        ).run(genesis)

    assert workspace.round_state(1) == "proposer_failed"
    assert workspace.read_search_state()["completed_rounds"] == 0
    resumed = meta_modules.search.MetaHarnessSearch(
        workspace=workspace,
        evaluator=_DeterministicEvaluator(meta_modules),
        proposer=proposer,
        mode="full_history",
        rounds=1,
    ).run(genesis)

    assert resumed.resumed is True
    assert proposer.calls == 2
    assert len(resumed.proposer_sessions) == 1
    assert workspace.round_state(1) == "committed"
    assert not (workspace.proposals_dir / "round-0001" / "partial").exists()


@pytest.mark.parametrize("interrupted_state", ["proposed", "validating"])
def test_search_resumes_proposal_validation_without_rerunning_proposer(meta_modules, tmp_path, interrupted_state):
    from reef.harness.adapters import get_adapter

    workspace = meta_modules.workspace.EvolutionWorkspace(tmp_path / "run", get_adapter("pi"))
    proposer = _ScriptedProposer(meta_modules)
    genesis = meta_modules.composition.genesis_composition()
    original_transition = workspace.transition_round

    def interrupt_after_transition(round_index, state, **details):
        original_transition(round_index, state, **details)
        if state == interrupted_state:
            raise KeyboardInterrupt("simulated process interruption")

    workspace.transition_round = interrupt_after_transition
    with pytest.raises(KeyboardInterrupt, match="simulated process interruption"):
        meta_modules.search.MetaHarnessSearch(
            workspace=workspace,
            evaluator=_DeterministicEvaluator(meta_modules),
            proposer=proposer,
            mode="full_history",
            rounds=1,
        ).run(genesis)

    assert workspace.round_state(1) == interrupted_state
    workspace.transition_round = original_transition
    resumed = meta_modules.search.MetaHarnessSearch(
        workspace=workspace,
        evaluator=_DeterministicEvaluator(meta_modules),
        proposer=proposer,
        mode="full_history",
        rounds=1,
    ).run(genesis)

    assert resumed.resumed is True
    assert len(proposer.surfaces) == 1
    assert len(resumed.proposer_sessions) == 1
    assert workspace.round_state(1) == "committed"


def test_attempt_log_retains_invalid_rejected_and_selected_proposals(meta_modules, tmp_path):
    from reef.harness.adapters import get_adapter

    class AttemptProposer:
        def propose(self, *, surface, prompt, session_dir, round_index):
            catalog = json.loads((surface / "candidate-catalog.json").read_text())
            if round_index == 1:
                source_hash = catalog["incumbent_hash"]
                base = "unknown-candidate"
                marker = "invalid"
                name = "invalid-parent"
            elif round_index == 2:
                source_hash = catalog["incumbent_hash"]
                base = "genesis"
                marker = "specialist"
                name = "specialist"
            else:
                specialist = next(row for row in catalog["candidates"] if row["alias"] == "specialist")
                source_hash = specialist["candidate_hash"]
                base = "specialist"
                marker = "branch-improvement"
                name = "selected-branch"
            value = json.loads((surface / "history" / source_hash / "candidate.json").read_text())["composition"]
            value["nodes"][0]["config"]["text"] += f" {marker}"
            proposal_dir = surface / "proposals" / f"round-{round_index:04d}" / name
            proposal_dir.mkdir(parents=True, exist_ok=True)
            (proposal_dir / "composition.json").write_text(json.dumps(value))
            (surface / "pending_eval.json").write_text(
                json.dumps(
                    {
                        "iteration": round_index,
                        "candidates": [
                            {
                                "name": name,
                                "base_candidates": [base],
                                "hypothesis": f"Exercise the {marker} audit path.",
                                "changes": f"Add the {marker} marker.",
                            }
                        ],
                    }
                )
            )
            return meta_modules.search.ProposerSession(session_id=f"audit-{round_index}")

    workspace = meta_modules.workspace.EvolutionWorkspace(tmp_path / "run", get_adapter("pi"))
    outcome = meta_modules.search.MetaHarnessSearch(
        workspace=workspace,
        evaluator=_DeterministicEvaluator(meta_modules),
        proposer=AttemptProposer(),
        mode="full_history",
        rounds=3,
    ).run(meta_modules.composition.genesis_composition())

    events = [json.loads(line) for line in (workspace.root / "proposal-events.jsonl").read_text().splitlines()]
    statuses = [event["status"] for event in events]
    assert len(outcome.population) == 3
    assert statuses == ["invalid:ValueError", "rejected", "selected"]
    assert events[1]["retention_status"] == "retained"
    assert events[1]["incumbent_before"] == events[1]["incumbent_after"]
    assert events[1]["candidate_hash"] != events[2]["candidate_hash"]
    assert events[2]["incumbent_before"] == outcome.population[0].candidate.content_hash


def test_full_history_search_rejects_undeclared_surface_outputs(meta_modules, tmp_path):
    from reef.harness.adapters import get_adapter

    class RogueProposer:
        def propose(self, *, surface, prompt, session_dir, round_index):
            (surface / "rogue.txt").write_text("not a declared proposer output")
            return meta_modules.search.ProposerSession(session_id="rogue")

    workspace = meta_modules.workspace.EvolutionWorkspace(tmp_path / "run", get_adapter("pi"))
    search = meta_modules.search.MetaHarnessSearch(
        workspace=workspace,
        evaluator=_DeterministicEvaluator(meta_modules),
        proposer=RogueProposer(),
        mode="full_history",
        rounds=1,
    )

    with pytest.raises(meta_modules.workspace.WorkspaceIntegrityError, match="read-only surface"):
        search.run(meta_modules.composition.genesis_composition())


def test_incumbent_only_surface_omits_raw_history_and_rejects_other_parents(meta_modules, tmp_path):
    from reef.harness.adapters import get_adapter

    workspace = meta_modules.workspace.EvolutionWorkspace(tmp_path / "run", get_adapter("pi"))
    evaluator = _DeterministicEvaluator(meta_modules)
    proposer = _ScriptedProposer(meta_modules)
    genesis = meta_modules.composition.genesis_composition()
    search = meta_modules.search.MetaHarnessSearch(
        workspace=workspace,
        evaluator=evaluator,
        proposer=proposer,
        mode="incumbent_only",
        rounds=1,
    )
    population = search._initialize(genesis)
    surface = search._surface(population, 1)

    assert not (surface / "history").exists()
    assert (surface / "incumbent" / "composition.json").is_file()
    proposal = meta_modules.workspace.WorkspaceProposal(
        name="invalid-parent",
        proposal_id="round-0001:000:invalid-parent",
        candidate=meta_modules.composition.CompositionCandidate.from_value(
            genesis.composition,
            parent_hashes=("f" * 64,),
        ),
        hypothesis="invalid",
        changes="invalid",
        source_path=surface / "composition.json",
    )
    with pytest.raises(ValueError, match="retained population"):
        search._validate_parent(proposal, population)


def test_codex_proposer_is_pinned_isolated_and_audited(meta_modules, tmp_path, monkeypatch):
    captured = {}

    def run_process(command, environment, timeout_s):
        captured.update(command=list(command), environment=dict(environment), timeout_s=timeout_s)
        events = [
            {"type": "thread.started", "thread_id": "thread-123"},
            {
                "type": "item.completed",
                "item": {
                    "type": "command_execution",
                    "command": "find history -type f",
                    "status": "completed",
                    "exit_code": 0,
                },
            },
            {
                "type": "item.completed",
                "item": {
                    "type": "command_execution",
                    "command": "sed -n '1,40p' history/abc/candidate.json",
                    "status": "completed",
                    "exit_code": 0,
                },
            },
            {
                "type": "item.completed",
                "item": {
                    "type": "file_change",
                    "status": "completed",
                    "changes": [{"path": "proposals/round-0001/candidate/composition.json"}],
                },
            },
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 100, "cached_input_tokens": 20, "output_tokens": 30},
            },
        ]
        return meta_modules.codex.ProcessResult(
            returncode=0,
            stdout="".join(json.dumps(event) + "\n" for event in events),
            stderr="",
            wall_time_s=2.5,
        )

    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-codex")
    monkeypatch.setenv("UNRELATED_SECRET", "also-withheld")
    surface = tmp_path / "surface"
    surface.mkdir()
    proposer = meta_modules.codex.CodexProposer(
        model="gpt-5.5",
        reasoning_effort="high",
        codex_path=tmp_path / "codex",
        timeout_s=60,
        process_runner=run_process,
        version_reader=lambda path: "codex-cli test-version",
    )

    session = proposer.propose(
        surface=surface,
        prompt="Inspect retained history and write one proposal.",
        session_dir=tmp_path / "sessions" / "round-0001",
        round_index=1,
    )

    command = captured["command"]
    assert command[:3] == [str((tmp_path / "codex").resolve()), "exec", "--json"]
    assert command[command.index("--model") + 1] == "gpt-5.5"
    assert 'model_reasoning_effort="high"' in command
    assert "sandbox_workspace_write.network_access=false" in command
    assert 'default_permissions="meta_harness"' in command
    assert any(value.startswith("permissions.meta_harness.filesystem=") for value in command)
    assert "permissions.meta_harness.network={enabled=false}" in command
    assert {"--ignore-user-config", "--ignore-rules", "--ephemeral"} <= set(command)
    assert captured["environment"].get("OPENAI_API_KEY") is None
    assert captured["environment"].get("UNRELATED_SECRET") is None
    assert session.session_id == "thread-123"
    assert (session.input_tokens, session.output_tokens) == (100, 30)
    metadata = json.loads((tmp_path / str(session.artifact_dir) / "metadata.json").read_text())
    assert metadata["codex_version"] == "codex-cli test-version"
    assert metadata["evidence"]["completed_turns"] == 1
    assert metadata["evidence"]["tool_evidence"][0]["command"] == "find history -type f"
    assert metadata["evidence"]["history_accesses"] == [
        {
            "event_type": "item.completed",
            "item_type": "command_execution",
            "command": "sed -n '1,40p' history/abc/candidate.json",
        }
    ]
    assert "must-not-reach-codex" not in json.dumps(metadata)


def test_history_read_audit_requires_reader_to_target_history(meta_modules):
    completed = {"status": "completed", "exit_code": 0}

    assert meta_modules.codex._is_completed_history_content_read(
        {**completed, "command": "sed -n '1,40p' history/abc/candidate.json"}
    )
    assert not meta_modules.codex._is_completed_history_content_read(
        {**completed, "command": "cat unrelated.txt && echo history/abc/candidate.json"}
    )


def test_codex_permission_profile_preflight_is_sealed(meta_modules, tmp_path, monkeypatch):
    observed = {}

    def run(command, **kwargs):
        observed["command"] = list(command)
        surface = Path(command[command.index("--cd") + 1])
        (surface / "writable.txt").write_text("writable")
        return SimpleNamespace(returncode=0)

    binary = tmp_path / "codex"
    binary.write_text("placeholder")
    monkeypatch.setattr(meta_modules.codex.subprocess, "run", run)

    meta_modules.codex.verify_permission_profile(binary)

    assert observed["command"][1] == "sandbox"
    assert observed["command"][observed["command"].index("--permission-profile") + 1] == "meta_harness"
    assert any(value.startswith("permissions.meta_harness.filesystem=") for value in observed["command"])


@pytest.mark.parametrize(
    ("returncode", "events", "message"),
    [
        (1, [{"type": "thread.started", "thread_id": "failed"}], "status 1"),
        (
            0,
            [
                {"type": "thread.started", "thread_id": "empty"},
                {"type": "turn.completed", "usage": {}},
            ],
            "non-zero token usage",
        ),
    ],
)
def test_codex_proposer_fails_loudly_on_incomplete_runs(meta_modules, tmp_path, returncode, events, message):
    def run_process(command, environment, timeout_s):
        return meta_modules.codex.ProcessResult(
            returncode=returncode,
            stdout="".join(json.dumps(event) + "\n" for event in events),
            stderr="failure",
            wall_time_s=0.1,
        )

    surface = tmp_path / "surface"
    surface.mkdir()
    proposer = meta_modules.codex.CodexProposer(
        model="gpt-5.5",
        reasoning_effort="high",
        codex_path=tmp_path / "codex",
        process_runner=run_process,
        version_reader=lambda path: "codex-cli test-version",
    )

    with pytest.raises(meta_modules.codex.CodexProposerError, match=message):
        proposer.propose(
            surface=surface,
            prompt="write proposal",
            session_dir=tmp_path / "sessions",
            round_index=1,
        )


def test_codex_proposer_audits_process_exceptions(meta_modules, tmp_path):
    surface = tmp_path / "surface"
    surface.mkdir()

    def fail_process(command, environment, timeout_s):
        raise meta_modules.codex.CodexProposerError("timed out")

    proposer = meta_modules.codex.CodexProposer(
        model="gpt-5.5",
        reasoning_effort="high",
        codex_path=tmp_path / "codex",
        process_runner=fail_process,
        version_reader=lambda path: "codex-cli test-version",
    )
    session_root = tmp_path / "sessions"

    with pytest.raises(meta_modules.codex.CodexProposerError, match="timed out"):
        proposer.propose(
            surface=surface,
            prompt="write proposal",
            session_dir=session_root,
            round_index=1,
        )

    artifact_dir = next(session_root.iterdir())
    assert json.loads((artifact_dir / "metadata.json").read_text())["status"] == "process_exception"


def test_harbor_exec_bridge_forwards_authenticated_commands(meta_modules):
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    calls = []

    async def execute(*, command, timeout_sec):
        calls.append((command, timeout_sec))
        return SimpleNamespace(stdout="remote-out", stderr="remote-err", return_code=7)

    def post(url, token):
        request = Request(
            f"{url}/exec",
            data=json.dumps({"command": "pwd", "timeout_sec": 12}).encode(),
            headers={"authorization": f"Bearer {token}", "content-type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=2) as response:
            return json.load(response)

    async def scenario():
        loop = asyncio.get_running_loop()
        with meta_modules.pi_bridge.HarborExecBridge(execute, loop) as bridge:
            result = await asyncio.to_thread(post, bridge.url, bridge.token)
            with pytest.raises(HTTPError) as denied:
                await asyncio.to_thread(post, bridge.url, "wrong-token")
            assert denied.value.code == 401
        return result

    result = asyncio.run(scenario())

    assert result == {"stdout": "remote-out", "stderr": "remote-err", "return_code": 7}
    assert calls == [("pwd", 12)]


def test_pi_episode_loads_bridge_last_and_collects_native_usage(meta_modules, tmp_path):
    from reef.harness.adapters import get_adapter

    captured = {}

    def run_process(command, cwd, environment, timeout_s):
        captured.update(command=list(command), cwd=cwd, environment=dict(environment), timeout_s=timeout_s)
        bridge_index = command.index(str(cwd.parent / meta_modules.pi_bridge.BRIDGE_EXTENSION_PATH))
        candidate_index = command.index(str(cwd.parent / "pi-agent/extensions/candidate.ts"))
        assert bridge_index > candidate_index
        assert "createBashToolDefinition" in (cwd.parent / meta_modules.pi_bridge.BRIDGE_EXTENSION_PATH).read_text()
        sessions = Path(environment["PI_CODING_AGENT_SESSION_DIR"])
        sessions.mkdir(parents=True)
        event = {
            "type": "message",
            "message": {
                "role": "assistant",
                "usage": {
                    "input": 20,
                    "cacheRead": 5,
                    "output": 4,
                    "cost": {"total": 0.0},
                },
            },
        }
        (sessions / "session.jsonl").write_text(json.dumps(event) + "\n")
        return meta_modules.pi_bridge.PiProcessResult(0, '{"type":"agent_end"}\n', "", 1.25)

    runner = meta_modules.pi_bridge.PiEpisodeRunner(
        get_adapter("pi"),
        binary=tmp_path / "pi",
        timeout_s=90,
        process_runner=run_process,
    )
    result = runner.run(
        {
            "pi-agent/settings.json": "{}\n",
            "pi-agent/models.json": "{}\n",
            "pi-agent/extensions/candidate.ts": "export default () => {};\n",
        },
        "solve the task",
        bridge_url="http://127.0.0.1:1234",
        bridge_token="bridge-secret",
    )

    assert {"--no-extensions", "--tools", "bash", "--print"} <= set(captured["command"])
    assert captured["environment"]["REEF_HARBOR_BRIDGE_TOKEN"] == "bridge-secret"
    assert result.usage == {"input_tokens": 20, "cached_input_tokens": 5, "output_tokens": 4}
    assert result.estimated_cost_usd == pytest.approx((20 * 0.75 + 5 * 0.075 + 4 * 4.50) / 1_000_000)
    assert result.provider_reported_cost_usd == 0
    assert len(result.trajectory) == 1
    assert not captured["cwd"].parent.exists()


def test_harbor_agent_injects_ephemeral_binding_and_reports_pi_evidence(
    meta_modules,
    tmp_path,
    monkeypatch,
):
    class BaseAgent:
        def __init__(self, logs_dir, model_name=None, *args, **kwargs):
            self.logs_dir = Path(logs_dir)
            self.model_name = model_name

    class BaseEnvironment:
        pass

    class AgentContext:
        pass

    packages = {
        "harbor": ModuleType("harbor"),
        "harbor.agents": ModuleType("harbor.agents"),
        "harbor.agents.base": ModuleType("harbor.agents.base"),
        "harbor.environments": ModuleType("harbor.environments"),
        "harbor.environments.base": ModuleType("harbor.environments.base"),
        "harbor.models": ModuleType("harbor.models"),
        "harbor.models.agent": ModuleType("harbor.models.agent"),
        "harbor.models.agent.context": ModuleType("harbor.models.agent.context"),
    }
    packages["harbor.agents.base"].BaseAgent = BaseAgent
    packages["harbor.environments.base"].BaseEnvironment = BaseEnvironment
    packages["harbor.models.agent.context"].AgentContext = AgentContext
    for name, module in packages.items():
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules.pop("harness.harbor_agent", None)
    harbor_agent = importlib.import_module("harness.harbor_agent")

    candidate = meta_modules.composition.genesis_composition()
    candidate_path = tmp_path / "composition.json"
    candidate_path.write_text(candidate.canonical_json + "\n")
    original = candidate_path.read_text()
    captured = {}

    class FakeRunner:
        def run(self, files, instruction, *, bridge_url, bridge_token):
            from urllib.request import Request, urlopen

            models = json.loads(files["pi-agent/models.json"])
            captured["api_key"] = models["providers"]["reef"]["apiKey"]
            request = Request(
                f"{bridge_url}/exec",
                data=json.dumps({"command": "echo bridged", "timeout_sec": 8}).encode(),
                headers={"authorization": f"Bearer {bridge_token}", "content-type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=2) as response:
                captured["bridge_response"] = json.load(response)
            return meta_modules.pi_bridge.PiEpisodeResult(
                exit_code=0,
                stdout="",
                stderr="provider rejected ephemeral-test-key",
                trajectory=({"type": "message", "message": {"role": "assistant"}},),
                usage={"input_tokens": 20, "cached_input_tokens": 5, "output_tokens": 4},
                estimated_cost_usd=0.03,
                provider_reported_cost_usd=0.029,
                wall_time_s=1.0,
            )

    class FakeEnvironment:
        def __init__(self):
            self.calls = []

        async def exec(self, *, command, timeout_sec):
            self.calls.append((command, timeout_sec))
            return SimpleNamespace(stdout="inside-container", stderr="", return_code=0)

    monkeypatch.setenv("META_TEST_OPENAI_KEY", "ephemeral-test-key")
    logs_dir = tmp_path / "trial" / "agent"
    agent = harbor_agent.HarborAgent(
        logs_dir=logs_dir,
        model_name="gpt-test",
        composition_path=str(candidate_path),
        target_base_url="https://example.invalid",
        target_api_key_env="META_TEST_OPENAI_KEY",
        pi_binary=str(tmp_path / "pi"),
        pi_runner=FakeRunner(),
    )
    environment = FakeEnvironment()
    context = SimpleNamespace(
        n_input_tokens=None,
        n_cache_tokens=None,
        n_output_tokens=None,
        cost_usd=None,
        metadata=None,
    )

    asyncio.run(agent.run("solve", environment, context))

    assert captured["api_key"] == "ephemeral-test-key"
    assert captured["bridge_response"]["stdout"] == "inside-container"
    assert environment.calls == [("echo bridged", 8)]
    assert candidate_path.read_text() == original
    assert "ephemeral-test-key" not in candidate_path.read_text()
    assert (context.n_input_tokens, context.n_cache_tokens, context.n_output_tokens) == (25, 5, 4)
    assert context.cost_usd == pytest.approx(0.03)
    assert context.metadata["reef_meta_harness"]["provider_reported_cost_usd"] == pytest.approx(0.029)
    assert "ephemeral-test-key" not in context.metadata["reef_meta_harness"]["pi_stderr"]
    trajectory_path = Path(context.metadata["reef_meta_harness"]["trajectory_path"])
    assert trajectory_path.is_file()
    assert json.loads(trajectory_path.read_text())[0]["type"] == "message"


def test_harbor_evaluator_runs_exact_split_trials_and_retains_evidence(meta_modules, tmp_path):
    calls = []
    budget_identities = []

    def run_trial(candidate, candidate_path, task_id, trial_index, round_index, trial_dir):
        calls.append((task_id, trial_index, round_index, trial_dir))
        assert json.loads(candidate_path.read_text()) == candidate.composition
        reward = 1.0 if task_id == "train-a" and trial_index == 1 else 0.0
        return meta_modules.harbor_eval.HarborTrialOutcome(
            reward=reward,
            trajectory=({"task": task_id, "trial": trial_index},),
            usage={"input_tokens": 10, "output_tokens": 2},
            estimated_cost_usd=0.01,
            wall_time_s=1.0,
            verifier={"rewards": {"reward": reward}, "task_checksum": f"checksum-{task_id}"},
        )

    output_dir = tmp_path / "private-evaluations"
    evaluator = meta_modules.harbor_eval.HarborEvaluator(
        splits={
            "train": ("train-a", "train-b"),
            "dev": ("dev-a",),
            "test": ("test-a",),
        },
        trials_per_task=2,
        output_dir=output_dir,
        target_model="gpt-test",
        target_base_url="https://example.invalid",
        target_api_key_env="META_TEST_OPENAI_KEY",
        pi_binary=tmp_path / "pi",
        trial_executor=run_trial,
        before_trial=budget_identities.append,
        budget_namespace="cell-a",
    )
    candidate = meta_modules.composition.genesis_composition()

    result = evaluator.evaluate(candidate, split="train", round_index=3)
    resumed = evaluator.evaluate(candidate, split="train", round_index=3)

    assert result.score == pytest.approx(0.25)
    assert resumed.to_jsonable() == result.to_jsonable()
    assert len(result.trials) == 4
    assert len(calls) == 4
    assert all(identity.startswith("cell-a:") for identity in budget_identities)
    assert {call[0] for call in calls} == {"train-a", "train-b"}
    assert not any(call[0].startswith("dev") or call[0].startswith("test") for call in calls)
    assert result.to_jsonable()["usage"] == {"input_tokens": 40, "output_tokens": 8}
    evidence_paths = sorted(output_dir.rglob("evidence.json"))
    assert len(evidence_paths) == 4
    evidence = json.loads(evidence_paths[0].read_text())
    assert evidence["trajectory"][0]["task"] in {"train-a", "train-b"}
    assert evidence["verifier"]["rewards"]["reward"] in {0.0, 1.0}
    persisted = output_dir / "candidates" / candidate.content_hash / "composition.json"
    assert json.loads(persisted.read_text()) == candidate.composition


def test_harbor_evaluator_retries_infrastructure_failures_without_scoring_them(meta_modules, tmp_path):
    attempts = 0

    def run_trial(candidate, candidate_path, task_id, trial_index, round_index, trial_dir):
        nonlocal attempts
        attempts += 1
        return meta_modules.harbor_eval.HarborTrialOutcome(
            reward=0.0 if attempts == 1 else 1.0,
            trajectory=() if attempts == 1 else ({"type": "message"},),
            estimated_cost_usd=0.02 if attempts == 1 else 0.01,
            status="trajectory_missing" if attempts == 1 else "completed",
            verifier={"rewards": {"reward": 1.0}, "task_checksum": "checksum-train"},
        )

    ledger = meta_modules.budget.ObservedCostLedger(tmp_path / "cost.json", 1.0)
    evaluator = meta_modules.harbor_eval.HarborEvaluator(
        splits={"train": ("train-a",), "dev": ("dev-a",), "test": ("test-a",)},
        trials_per_task=1,
        output_dir=tmp_path / "evaluations",
        target_model="gpt-test",
        target_base_url="https://example.invalid",
        target_api_key_env="META_TEST_OPENAI_KEY",
        pi_binary=tmp_path / "pi",
        trial_executor=run_trial,
        before_trial=ledger.before_trial,
        after_trial=ledger.record_trial,
    )
    candidate = meta_modules.composition.genesis_composition()

    with pytest.raises(meta_modules.harbor_eval.HarborInfrastructureError, match="infrastructure status"):
        evaluator.evaluate(candidate, split="train", round_index=0)
    result = evaluator.evaluate(candidate, split="train", round_index=0)

    assert attempts == 2
    assert result.score == 1.0
    assert ledger.observed_cost_usd == pytest.approx(0.03)
    assert len(tuple((tmp_path / "evaluations").rglob("infrastructure-attempt-01.json"))) == 1
    assert len(tuple((tmp_path / "evaluations").rglob("evidence.json"))) == 1


def test_harbor_evaluator_rejects_overlapping_splits(meta_modules, tmp_path):
    with pytest.raises(ValueError, match="disjoint"):
        meta_modules.harbor_eval.HarborEvaluator(
            splits={"train": ("same",), "dev": ("same",), "test": ("held-out",)},
            trials_per_task=2,
            output_dir=tmp_path,
            target_model="gpt-test",
            target_base_url="https://example.invalid",
            target_api_key_env="META_TEST_OPENAI_KEY",
            pi_binary=tmp_path / "pi",
        )


def test_terminal_bench_registry_pin_rejects_drift(meta_modules):
    class FakeDatasetConfig:
        def __init__(self, *, task_names, **kwargs):
            self.task_names = task_names

        async def get_task_configs(self):
            return [
                SimpleNamespace(
                    path=Path(task_id),
                    git_url=meta_modules.harbor_eval.TERMINAL_BENCH_DATASET_URL,
                    git_commit_id=(
                        "0" * 40 if task_id == "task-b" else meta_modules.harbor_eval.TERMINAL_BENCH_DATASET_COMMIT
                    ),
                )
                for task_id in self.task_names
            ]

    with pytest.raises(RuntimeError, match="unexpected commit"):
        asyncio.run(
            meta_modules.harbor_eval.verify_dataset_registry_pin(
                ("task-a", "task-b"),
                dataset_config_factory=FakeDatasetConfig,
            )
        )


def test_harbor_trial_parser_retains_registry_checksum(meta_modules, tmp_path):
    trajectory_path = tmp_path / "trajectory.json"
    trajectory_path.write_text('[{"type":"message"}]\n')
    started = datetime.now(timezone.utc)
    result = SimpleNamespace(
        verifier_result=SimpleNamespace(rewards={"reward": 1.0}),
        agent_result=SimpleNamespace(
            metadata={
                "reef_meta_harness": {
                    "pi_exit_code": 0,
                    "trajectory_path": str(trajectory_path),
                }
            },
            n_input_tokens=20,
            n_cache_tokens=5,
            n_output_tokens=4,
            cost_usd=0.03,
        ),
        exception_info=None,
        task_checksum="task-checksum-123",
        source="terminal-bench",
        started_at=started,
        finished_at=started + timedelta(seconds=2),
    )

    outcome = meta_modules.harbor_eval._outcome_from_trial_result(result)

    assert outcome.reward == 1.0
    assert outcome.status == "completed"
    assert outcome.verifier["task_checksum"] == "task-checksum-123"
    assert outcome.wall_time_s == 2.0


def test_observed_cost_ledger_is_persistent_and_idempotent(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    budget = importlib.import_module("harness.budget")
    ledger = budget.ObservedCostLedger(tmp_path / "cost.json", 1.0)

    ledger.before_trial("trial-a")
    ledger.record_trial("trial-a", 0.6)
    ledger.record_trial("trial-a", 0.6)
    resumed = budget.ObservedCostLedger(tmp_path / "cost.json", 1.0)
    resumed.before_trial("trial-b")
    resumed.record_trial("trial-b", 0.5)

    assert resumed.observed_cost_usd == pytest.approx(1.1)
    with pytest.raises(budget.SpendCapReached, match="no new trial"):
        resumed.before_trial("trial-c")

    for invalid_cap in (float("nan"), float("inf"), float("-inf"), 0.0):
        with pytest.raises(ValueError, match="finite and positive"):
            budget.ObservedCostLedger(tmp_path / f"cost-{invalid_cap}.json", invalid_cap)
    for invalid_cost in (float("nan"), float("inf"), float("-inf"), -0.01):
        with pytest.raises(ValueError, match="finite and non-negative"):
            ledger.record_trial(f"invalid-{invalid_cost}", invalid_cost)


def test_meta_harness_plan_uses_equal_target_episode_budgets(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    sys.modules.pop("run", None)
    run = importlib.import_module("run")
    monkeypatch.setattr(run, "_command_output", lambda command: "test-version")
    config = run.ExperimentConfig()

    plan = run.build_plan(
        config,
        run.CELLS,
        tmp_path / "outputs",
        "/path/to/pi",
        Path("/path/to/codex"),
    )

    assert run.planned_target_episodes(config) == 140
    assert plan["planned_target_episodes_per_cell"] == 140
    assert plan["planned_target_episodes_total"] == 420
    assert plan["planned_proposer_turns"] == 4
    assert plan["target_model_pricing"]["model"] == config.target_model
    assert plan["target_model_pricing"]["output_usd_per_million"] == 4.5
    assert plan["pins"]["terminal_bench_dataset_commit"] == ("69671fbaac6d67a7ef0dfec016cc38a64ef7a77c")
    assert "bn-fit-modify" not in json.dumps(plan)

    smoke_splits = run.selected_task_splits(True)
    smoke_plan = run.build_plan(
        config,
        run.CELLS,
        tmp_path / "smoke-outputs",
        "/path/to/pi",
        Path("/path/to/codex"),
        splits=smoke_splits,
        smoke=True,
    )
    assert run.planned_target_episodes(config, smoke_splits) == 14
    assert smoke_plan["smoke"] is True
    assert smoke_plan["split_sizes"] == {"train": 1, "dev": 1, "test": 1}
    assert smoke_plan["split_identity_sha256"] != plan["split_identity_sha256"]
    assert smoke_plan["planned_target_episodes_per_cell"] == 14
    assert smoke_plan["planned_target_episodes_total"] == 42
    assert smoke_plan["planned_proposer_turns"] == 4


def test_run_identity_allows_staged_cells_but_rejects_configuration_drift(runner_modules, tmp_path):
    run = runner_modules.run
    plan = {
        "cells": ("frozen",),
        "smoke": False,
        "output_dir": str(tmp_path),
        "config": {"rounds": 2, "trials_per_task": 2},
        "target_model_pricing": {"model": "gpt-test"},
        "pins": {"reef_source_commit": "a", "reef_source_dirty": False, "pi_version": "1"},
        "binaries": {"codex_version": "codex 1"},
        "split_sizes": {"train": 10, "dev": 10, "test": 10},
        "split_identity_sha256": "split",
    }
    path = tmp_path / "run-identity.json"
    digest = run.ensure_run_identity(path, plan)

    assert run.ensure_run_identity(path, {**plan, "cells": ("frozen", "full_history")}) == digest
    changed = {**plan, "config": {**plan["config"], "rounds": 3}}
    with pytest.raises(RuntimeError, match="does not match"):
        run.ensure_run_identity(path, changed)


def test_done_marker_detects_evidence_changes(runner_modules, tmp_path):
    run = runner_modules.run
    output_dir = tmp_path / "full_history"
    output_dir.mkdir()
    (output_dir / "summary.json").write_text('{"cell":"full_history"}\n')
    repository = output_dir / "artifacts.git"
    repository.mkdir()
    (output_dir / "publication.json").write_text(json.dumps({"repository": str(repository)}) + "\n")
    run.mark_done(output_dir, "full_history", "identity")

    assert run.completed_cell(output_dir, "full_history", "identity")
    (output_dir / "summary.json").write_text('{"cell":"tampered"}\n')
    with pytest.raises(RuntimeError, match="evidence changed"):
        run.completed_cell(output_dir, "full_history", "identity")


@pytest.mark.parametrize("smoke", [False, True])
def test_three_cell_runner_retains_equal_real_evaluator_budgets_and_reports(
    runner_modules,
    tmp_path,
    monkeypatch,
    smoke,
):
    from reef.harness.adapters import get_adapter

    run = runner_modules.run
    real_evaluator = runner_modules.harbor_eval.HarborEvaluator

    def trial_executor(candidate, candidate_path, task_id, trial_index, round_index, trial_dir):
        text = candidate.canonical_json
        if "full-selected" in text:
            reward = 0.9
        elif "incumbent-selected" in text:
            reward = 0.8
        elif "retained-specialist" in text:
            reward = 0.2
        else:
            reward = 0.5
        return runner_modules.harbor_eval.HarborTrialOutcome(
            reward=reward,
            trajectory=({"task": task_id, "trial": trial_index, "round": round_index},),
            usage={"input_tokens": 10, "output_tokens": 2},
            estimated_cost_usd=0.001,
            wall_time_s=0.01,
            verifier={"rewards": {"reward": reward}, "task_checksum": f"checksum-{task_id}"},
        )

    def evaluator_factory(**kwargs):
        return real_evaluator(**kwargs, trial_executor=trial_executor)

    class ScriptedCodex:
        def __init__(self, **kwargs):
            pass

        def propose(self, *, surface, prompt, session_dir, round_index):
            full_history = (surface / "history").is_dir()
            if full_history:
                catalog = json.loads((surface / "candidate-catalog.json").read_text())
                if round_index == 1:
                    base_row = next(row for row in catalog["candidates"] if row["alias"] == "genesis")
                    marker = "retained-specialist"
                    name = "retained-specialist"
                else:
                    base_row = next(row for row in catalog["candidates"] if row["alias"] == "retained-specialist")
                    marker = "full-selected"
                    name = "full-selected"
                base = base_row["alias"]
                value = json.loads((surface / "history" / base_row["candidate_hash"] / "candidate.json").read_text())[
                    "composition"
                ]
            else:
                summary = json.loads((surface / "search-summary.json").read_text())
                base = summary["incumbent_hash"]
                value = json.loads((surface / "incumbent" / "composition.json").read_text())
                marker = "retained-specialist" if round_index == 1 else "incumbent-selected"
                name = marker
            value["nodes"][0]["config"]["text"] += f" {marker}"
            proposal_dir = surface / "proposals" / f"round-{round_index:04d}" / name
            proposal_dir.mkdir(parents=True, exist_ok=True)
            (proposal_dir / "composition.json").write_text(json.dumps(value))
            (surface / "pending_eval.json").write_text(
                json.dumps(
                    {
                        "iteration": round_index,
                        "candidates": [
                            {
                                "name": name,
                                "base_candidates": [base],
                                "hypothesis": f"Evaluate {marker}.",
                                "changes": f"Add {marker}.",
                            }
                        ],
                    }
                )
            )
            artifact_dir = session_dir / "scripted"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            history_accesses = (
                [{"command": "sed -n '1,80p' history/candidate.json"}] if full_history and round_index > 1 else []
            )
            (artifact_dir / "metadata.json").write_text(
                json.dumps({"evidence": {"history_accesses": history_accesses}})
            )
            return runner_modules.search.ProposerSession(
                session_id=f"scripted-{round_index}",
                input_tokens=10,
                output_tokens=2,
                wall_time_s=0.01,
                artifact_dir=artifact_dir.relative_to(session_dir.parent.parent).as_posix(),
            )

    publications = []

    def publish(candidate, *, descriptor, output_dir, scenario, metadata):
        publications.append((scenario, candidate.content_hash))
        (output_dir / "publication.json").write_text(
            json.dumps({"candidate_hash": candidate.content_hash, "scenario": scenario})
        )
        return {"candidate_hash": candidate.content_hash}

    monkeypatch.setattr(run, "HarborEvaluator", evaluator_factory)
    monkeypatch.setattr(run, "CodexProposer", ScriptedCodex)
    monkeypatch.setattr(run, "publish_candidate", publish)
    config = run.ExperimentConfig()
    ledger = run.ObservedCostLedger(tmp_path / "observed-cost.json", 10.0)
    descriptor = get_adapter("pi")
    splits = run.selected_task_splits(smoke)
    for cell in run.CELLS:
        run.run_cell(
            cell,
            tmp_path / cell,
            config=config,
            descriptor=descriptor,
            pi_binary=tmp_path / "pi",
            codex_binary=tmp_path / "codex",
            ledger=ledger,
            splits=splits,
        )
    run.write_aggregate_report(tmp_path, run.CELLS)

    results = json.loads((tmp_path / "results.json").read_text())
    assert results["equal_target_episode_budget"] is True
    assert results["equal_task_checksums"] is True
    expected_episode_count = 14 if smoke else 140
    expected_task_count = 3 if smoke else 30
    assert results["target_episode_counts"] == dict.fromkeys(run.CELLS, expected_episode_count)
    assert results["cells"]["frozen"]["test_score"] == pytest.approx(0.5)
    assert results["cells"]["incumbent_only"]["test_score"] == pytest.approx(0.8)
    assert results["cells"]["full_history"]["test_score"] == pytest.approx(0.9)
    assert results["cells"]["full_history"]["later_round_history_read"] is True
    assert len(results["task_checksums"]) == expected_task_count
    assert len(publications) == 3
    assert all((tmp_path / cell / "done.json").is_file() for cell in run.CELLS)


def test_publication_tree_is_provider_free_and_resumable(meta_modules, tmp_path, monkeypatch):
    from reef.harness.adapters import get_adapter

    class FakeBackend:
        def __init__(self, scenario, repository_path, **kwargs):
            self.repository_path = repository_path

        def fork(self, metadata):
            return SimpleNamespace(
                content_id="parent-content",
                release_id="parent",
                parent_release_id=None,
            )

        def metadata(self):
            return None

        def publish(self, artifact, expected_parent):
            return SimpleNamespace(
                content_id="published-content",
                release_id="published",
                parent_release_id=expected_parent.release_id,
            )

    monkeypatch.setattr(meta_modules.publication, "GitLFSRepositoryBackend", FakeBackend)
    candidate = meta_modules.composition.genesis_composition()

    first = meta_modules.publication.publish_candidate(
        candidate,
        descriptor=get_adapter("pi"),
        output_dir=tmp_path,
        scenario="test-meta-harness",
        metadata={"cell": "test"},
    )
    resumed = meta_modules.publication.publish_candidate(
        candidate,
        descriptor=get_adapter("pi"),
        output_dir=tmp_path,
        scenario="test-meta-harness",
        metadata={"cell": "test"},
    )

    assert resumed == first
    assert first["candidate_hash"] == candidate.content_hash
    assert first["repository"] == "artifacts.git"
    assert json.loads((tmp_path / "published-composition/pi-agent/models.json").read_text()) == {}
    assert "apiKey" not in (tmp_path / "published-composition/pi-agent/settings.json").read_text()


def test_publication_recovers_from_pending_fork(meta_modules, tmp_path, monkeypatch):
    from reef.harness.adapters import get_adapter

    state = {"metadata": None, "current": None, "publish_calls": 0}

    class FakeBackend:
        def __init__(self, scenario, repository_path, **kwargs):
            Path(repository_path).mkdir(parents=True, exist_ok=True)

        def metadata(self):
            return state["metadata"]

        def current(self):
            return state["current"]

        def fork(self, metadata):
            state["metadata"] = dict(metadata)
            state["current"] = SimpleNamespace(
                content_id="pending-content",
                release_id="pending",
                parent_release_id="initial",
            )
            return state["current"]

        def publish(self, artifact, expected_parent):
            state["publish_calls"] += 1
            if state["publish_calls"] == 1:
                raise RuntimeError("interrupted after fork")
            state["metadata"] = dict(artifact.metadata)
            state["current"] = SimpleNamespace(
                content_id="published-content",
                release_id="published",
                parent_release_id=expected_parent.release_id,
            )
            return state["current"]

    monkeypatch.setattr(meta_modules.publication, "GitLFSRepositoryBackend", FakeBackend)
    candidate = meta_modules.composition.genesis_composition()

    with pytest.raises(RuntimeError, match="interrupted after fork"):
        meta_modules.publication.publish_candidate(
            candidate,
            descriptor=get_adapter("pi"),
            output_dir=tmp_path,
            scenario="test-meta-harness",
            metadata={"cell": "test"},
        )
    recovered = meta_modules.publication.publish_candidate(
        candidate,
        descriptor=get_adapter("pi"),
        output_dir=tmp_path,
        scenario="test-meta-harness",
        metadata={"cell": "test"},
    )

    assert recovered["release_id"] == "published"
    assert state["publish_calls"] == 2
    assert state["metadata"]["publication_state"] == "published"
