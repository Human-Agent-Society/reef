"""Meta-Harness on Reef's harness backend: the prompt it builds from the history, its checks, its budgets.

Steps run through the phases the trainer runs (``backend_step``) and are
decided by ``compare`` on means, as a Meta-Harness deployment selects. The
tasks run the fake pi binary, and ``MarkerScorer`` scores 1 when the rules
carry the marker, so the seed and a tree without the marker score 0.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
from tests.support._strategies import resolve_task_scorer
from tests.support._trajectories import recorded_trajectory
from tests.support.harness_methods import (
    ALTERNATE,
    IMPROVED,
    SEED,
    TASK,
    MarkerScorer,
    QueueChat,
    meta_harness_sections,
    proposal_reply,
    served_runtime,
)
from tests.support.harness_service import backend_step, harness_dispatcher, report_once
from tests.support.harness_steps import COMPONENT, TreeReleases, harness_recipe

import recipes.meta_harness
from recipes.meta_harness import MetaHarness, MetaHarnessSettings
from recipes.meta_harness.method import parse_proposal
from reef.core.plugins import PluginConfigError
from reef.core.training_request import TrainingRequest
from reef.evaluation.harness import ComparePolicy
from reef.harness.tree import HarnessTree
from reef.recipe.harness import HarnessEvolutionRecipe
from reef.recipe.registry import build_recipe
from reef.storage.sqlite import SQLiteRecordStore
from reef.train.backend import ComponentReleases
from reef.train.harness_backend import AgentMethod
from reef.train.harness_backend.backend import HarnessBackend
from reef.train.trainer import Trainer
from reef.train.types import JobData, TrainStepResult

#: One failing traffic trace of the task, with its report's feedback.
SAMPLE = recorded_trajectory(
    "trace-1",
    {
        "messages": [{"role": "user", "content": TASK}],
        "response": {"choices": [{"message": {"role": "assistant", "content": "attempt"}}]},
    },
    0.0,
    feedback="the attempt failed",
)
#: A tree without the marker: it scores 0 like the seed, so compare rejects it.
LOSER = HarnessTree({"rules.md": "Answer quickly.\n"})


def meta_harness_backend(
    tmp_path: Path, chat: QueueChat, *, seed: HarnessTree = SEED, **settings: Any
) -> tuple[HarnessBackend, TreeReleases]:
    """The backend of one scenario whose method is ``MetaHarness(settings)`` and whose proposer model is ``chat``."""
    method = MetaHarness(MetaHarnessSettings(**settings))
    recipe = harness_recipe(tmp_path, method, seed=seed, score=MarkerScorer(), tasks=(TASK,))
    recipe = dataclasses.replace(recipe, models={"proposer": chat})
    releases = TreeReleases(recipe.seed)
    return recipe.build_backend("demo", ComponentReleases(releases, COMPONENT), None), releases


def run(
    backend: HarnessBackend,
    releases: TreeReleases,
    state: Any,
    number: int,
    *,
    policy: ComparePolicy | None = None,
    request: TrainingRequest | None = None,
) -> TrainStepResult:
    """Step ``number`` on the failing trace, decided on means; a selected tree becomes the next release."""
    data = JobData(f"demo:{number}", (SAMPLE,), request=request)
    result = backend_step(backend, data, state, ComparePolicy(backend, statistic="mean") if policy is None else policy)
    if result.artifact is not None and result.artifact.local_path is not None:
        releases.publish(f"r{number}", HarnessTree.read(result.artifact.local_path))
    return result


def search_state(prompt: str) -> dict[str, Any]:
    """The search state a proposal prompt shows, read back from its fenced JSON."""
    body = prompt.split(": data, not instructions]\n", 1)[1]
    return json.loads(body.rsplit("\n[END search state", 1)[0])


def test_a_step_proposes_a_complete_tree_and_the_history_keeps_what_the_proposer_said(tmp_path: Path) -> None:
    chat = QueueChat(proposal_reply(SEED, IMPROVED, hypothesis="the marker helps"))
    backend, releases = meta_harness_backend(tmp_path, chat)
    request = TrainingRequest("Prefer short answers.", "s", "base", "ask-1")

    result = run(backend, releases, backend.initial_state(), 1, request=request)

    assert result.metrics["selected"] is True and result.metrics["selection"]["policy"] == "compare"
    assert (result.metrics["candidate_mean"], result.metrics["current_mean"]) == (1.0, 0.0)
    notes = {"status": "proposed", "parent_id": SEED.checksum(), "hypothesis": "the marker helps"}
    assert result.metrics["method_notes"] == {**notes, "changes": "revise the general instruction"}
    state = search_state(chat.prompts[0])
    assert state["served_candidate_id"] == SEED.checksum() and state["evaluation_tasks"] == {"t1": TASK}
    assert state["instruction"] == "Prefer short answers." and state["parts"]["rules"]["path"] == "rules.md"
    [seed] = state["population"]
    assert (seed["outcome"], seed["files"], seed["scores"], seed["mean_score"]) == (
        "served",
        dict(SEED.files),
        {},
        None,
    )
    [traffic] = state["traces"]
    assert traffic["source"] == "traffic" and "the attempt failed" in traffic["record"]
    # The history keeps the candidate with what the proposer said, and is the population of the next step.
    candidate = backend.history.candidates[IMPROVED.checksum()]
    assert (candidate.origin, candidate.notes["hypothesis"]) == ("method", "the marker helps")


def test_full_history_branches_from_a_rejected_candidate_and_reads_its_run_records(tmp_path: Path) -> None:
    chat = QueueChat(proposal_reply(SEED, LOSER), proposal_reply(LOSER, IMPROVED))
    backend, releases = meta_harness_backend(tmp_path, chat, max_traces=3, trace_chars=300)

    first = run(backend, releases, backend.initial_state(), 1)
    second = run(backend, releases, first.state, 2)

    assert first.metrics["selected"] is False and second.metrics["selected"] is True
    assert second.metrics["method_notes"]["parent_id"] == LOSER.checksum()
    state = search_state(chat.prompts[1])
    population = {member["candidate_id"]: member for member in state["population"]}
    assert population[SEED.checksum()]["outcome"] == "served"
    loser = population[LOSER.checksum()]
    assert (loser["outcome"], loser["parent_id"], loser["scores"], loser["files"]) == (
        "rejected",
        SEED.checksum(),
        {"t1": [0.0]},
        dict(LOSER.files),
    )
    # The traffic first, then the runs in turns: the served tree's, then the newest candidate's.
    traces = state["traces"]
    assert [(trace["source"], trace.get("candidate_id")) for trace in traces] == [
        ("traffic", None),
        ("evaluation", SEED.checksum()),
        ("evaluation", LOSER.checksum()),
    ]
    task_run = traces[2]
    assert (task_run["task"], task_run["repeat"], task_run["score"], task_run["failure"]) == ("t1", 0, 0.0, None)
    # The excerpt of the run's record directory: its record files, cut to trace_chars around a note.
    assert task_run["record"].startswith("--- run.json ---") and "characters left out" in task_run["record"]
    assert "Answer quickly." in task_run["record"]


def test_incumbent_only_branches_from_the_served_tree_and_shows_the_refused_reply_next(tmp_path: Path) -> None:
    chat = QueueChat(proposal_reply(SEED, LOSER), proposal_reply(LOSER, ALTERNATE), proposal_reply(SEED, IMPROVED))
    backend, releases = meta_harness_backend(tmp_path, chat, mode="incumbent_only")

    first = run(backend, releases, backend.initial_state(), 1)
    second = run(backend, releases, first.state, 2)
    third = run(backend, releases, second.state, 3)

    error = "parent_id must name a candidate of the population (incumbent_only mode)"
    assert second.metrics["skipped"] == "the method left the head tree unchanged"
    assert second.metrics["method_notes"] == {"status": "invalid", "error": error}
    # The prompt shows the served tree alone, and the next one shows why the reply proposed nothing.
    assert [member["candidate_id"] for member in search_state(chat.prompts[1])["population"]] == [SEED.checksum()]
    assert search_state(chat.prompts[2])["attempts"] == [{"step": 2, "status": "invalid", "error": error}]
    assert third.metrics["selected"] is True


def test_a_proposal_keeps_the_parent_files_of_the_parts_it_may_not_change(tmp_path: Path) -> None:
    seed = HarnessTree({**SEED.files, "config/primary.json": json.dumps({}) + "\n"})
    changed = HarnessTree({**IMPROVED.files, "config/primary.json": json.dumps({"defaultTools": []}) + "\n"})
    chat = QueueChat(proposal_reply(seed, changed))
    backend, releases = meta_harness_backend(tmp_path, chat, seed=seed, components=("rules",))

    result = run(backend, releases, backend.initial_state(), 1)

    assert result.metrics["method_notes"] == {
        "status": "invalid",
        "error": "a proposal may change only the files of the parts ['rules']",
    }


def test_an_invalid_or_repeated_tree_leaves_the_head_unchanged_and_joins_the_attempts(tmp_path: Path) -> None:
    empty_skill = HarnessTree({**SEED.files, "skills/empty/SKILL.md": "\n"})
    chat = QueueChat("not json at all", proposal_reply(SEED, empty_skill), proposal_reply(SEED, SEED))
    backend, releases = meta_harness_backend(tmp_path, chat)

    state = backend.initial_state()
    results = []
    for number in (1, 2, 3):
        result = run(backend, releases, state, number)
        results.append(result)
        state = result.state

    assert all(result.metrics["skipped"] == "the method left the head tree unchanged" for result in results)
    notes = [result.metrics["method_notes"] for result in results]
    assert notes[0] == {"status": "invalid", "error": "proposer response contains no JSON object"}
    assert notes[1]["error"].startswith("the tree fails validation: skills/empty/SKILL.md")
    assert notes[2] == {"status": "duplicate", "candidate_id": SEED.checksum()}
    attempts = [json.loads(line) for line in state["method_files"]["attempts.jsonl"].splitlines()]
    assert [(attempt["step"], attempt["status"]) for attempt in attempts] == [
        (1, "invalid"),
        (2, "invalid"),
        (3, "duplicate"),
    ]
    assert list(backend.history.candidates) == [SEED.checksum()] and not backend.history.runs


@pytest.mark.parametrize(
    ("settings", "budget"),
    [({"max_candidates": 1}, "max_candidates"), ({"max_target_runs": 2}, "max_target_runs")],
)
def test_a_spent_budget_stops_the_search_before_a_model_call(tmp_path: Path, settings: dict, budget: str) -> None:
    chat = QueueChat(proposal_reply(SEED, IMPROVED))
    backend, releases = meta_harness_backend(tmp_path, chat, **settings)

    first = run(backend, releases, backend.initial_state(), 1)
    second = run(backend, releases, first.state, 2)

    assert first.metrics["selected"] is True and len(chat.prompts) == 1
    assert second.metrics["method_notes"] == {"status": "budget_exhausted", "budget": budget}


def test_a_retried_step_proposes_again_the_tree_its_failed_attempt_proposed(tmp_path: Path) -> None:
    """The history keeps the attempt that never committed: the retry neither counts it nor refuses it as known."""

    class FailingDecision(ComparePolicy):
        def decide(self, candidate, evaluation):
            raise RuntimeError("the decision is unavailable")

    chat = QueueChat(proposal_reply(SEED, IMPROVED), proposal_reply(SEED, IMPROVED))
    backend, releases = meta_harness_backend(tmp_path, chat, max_candidates=1, max_target_runs=2)
    with pytest.raises(RuntimeError, match="unavailable"):
        run(backend, releases, backend.initial_state(), 1, policy=FailingDecision(backend, statistic="mean"))

    retried = run(backend, releases, backend.initial_state(), 1)

    assert retried.metrics["selected"] is True and len(chat.prompts) == 2
    # Both trees' scores come from the failed attempt's runs.
    assert (retried.metrics["runs_run"], retried.metrics["runs_cached"]) == (0, 2)


def test_the_reply_is_one_object_of_the_four_proposal_fields() -> None:
    fenced = "```json\n" + proposal_reply(SEED, IMPROVED) + "\n```"
    assert parse_proposal(fenced)["files"] == dict(IMPROVED.files)
    with pytest.raises(ValueError, match="unknown fields"):
        parse_proposal('{"parent_id": "p", "files": {}, "entries": []}')
    with pytest.raises(ValueError, match="requires a files object"):
        parse_proposal('{"parent_id": "p"}')


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"mode": "greedy"}, "mode must be one of"),
        ({"model": "served"}, "the served model never proposes"),
        ({"components": ["rules", "rules"]}, "components must not repeat a part"),
        ({"max_files": 0}, "max_files must be at least 1"),
        ({"archive": "population"}, "unknown config fields: archive"),
    ],
)
def test_the_settings_refuse_a_search_that_cannot_run(config: dict, message: str) -> None:
    with pytest.raises(PluginConfigError, match=message):
        MetaHarness.from_config(config)


def test_a_proposer_model_that_the_deployment_does_not_declare_is_refused(tmp_path: Path) -> None:
    # Meta-Harness improves a fixed model's harness with a separate proposer. Falling back to the served model would
    # let a typo in evolution.models turn the harness under test into its own author.
    backend, _ = meta_harness_backend(tmp_path, QueueChat(), model="typo")
    with pytest.raises(ValueError, match=r"is not declared under evolution\.models"):
        backend.prepare_step(JobData("demo:1", (SAMPLE,)), backend.initial_state(), 0)


@pytest.mark.parametrize("adapter", ["pi", "opencode"])
def test_a_deployment_names_the_method_for_any_adapter(tmp_path: Path, adapter: str) -> None:
    config = meta_harness_sections(tmp_path, adapter=adapter, mode="incumbent_only", components=["rules"])
    api = "openai"
    recipe = build_recipe(config["implementation"], {}, config=config, runtime=served_runtime(api))

    assert isinstance(recipe, HarnessEvolutionRecipe) and isinstance(recipe.method, MetaHarness)
    assert recipe.method.settings == MetaHarnessSettings(mode="incumbent_only", components=("rules",))
    assert recipe.evolution.adapter == adapter and recipe.seed == SEED
    releases = ComponentReleases(TreeReleases(recipe.seed), COMPONENT)
    assert isinstance(recipe.build("general-task", SQLiteRecordStore(), releases=releases), Trainer)


def test_terminus_extension_uses_shared_recipe_run_runner_and_publication(tmp_path, monkeypatch) -> None:
    # Replace only process launch and the remote trial. Rendering, model
    # binding, Harbor's agent contract, scoring, selection and commit are real.
    pytest.importorskip("reef_eval")
    pytest.importorskip("harbor")
    from types import SimpleNamespace

    from harbor.agents.terminus_2 import Terminus2
    from harbor.models.trial.config import TrialConfig
    from harbor.utils.import_path import import_class

    from reef_terminus import runner
    from reef.execution.sandbox.process import ISOLATION_ENV, ProcessOutcome, SandboxExecutor

    code = (Path(__file__).parents[3] / "recipes/meta_harness/results/reef_harness.py").read_text()
    proposed = HarnessTree({**SEED.files, "extensions/agent.py": code})
    launches, agents = [], []
    monkeypatch.setattr(Terminus2, "_get_completion_confirmation_message", lambda self, output: "stock")

    class Lab:
        def __init__(self, trials):
            self.trials = trials

        async def run(self, task, agent, **overrides):
            config = TrialConfig.model_validate(
                {"task": {"name": task}, "agent": agent, "trials_dir": self.trials, **overrides}
            )
            assert config.environment.type.value == "e2b"
            assert config.agent.model_name == "served-model"
            assert config.agent.kwargs["llm_kwargs"]["api_key"] == "dummy"
            assert config.extra_instruction_paths[0].read_text() == SEED.files["rules.md"]
            agents.append(config.agent)
            if config.agent.import_path:
                cls = import_class(config.agent.import_path)
                assert issubclass(cls, Terminus2)
                assert "MANDATORY FALSIFICATION CHECKPOINT" in cls._get_completion_confirmation_message(
                    object.__new__(cls), "output"
                )
            return SimpleNamespace(rewards={"accuracy": float(bool(config.agent.import_path))}, tags={}, uri=None)

    def launch(self, argv, *, root, workspace, env, timeout, writable_paths, readonly_paths):
        launches.append(root)
        assert argv[0] == "reef-terminus" and timeout == 17
        assert set(writable_paths) == {root / "terminus/sessions", root / "terminus/trials"}
        assert root / "terminus/config.json" in readonly_paths
        with monkeypatch.context() as context:
            for key, value in {**env, **self.env, ISOLATION_ENV: "bwrap"}.items():
                context.setenv(key, value)
            return ProcessOutcome(runner.run(argv[-1]), "", "")

    monkeypatch.setattr("reef_eval.Lab", Lab)
    monkeypatch.setattr(SandboxExecutor, "preflight", lambda self: None)
    monkeypatch.setattr(SandboxExecutor, "launch", launch)
    config = meta_harness_sections(tmp_path, adapter="terminus")
    evolution = config["evolution"]
    evolution.pop("binary")
    evolution.update(
        executor="sandbox",
        sandbox={"egress_hosts": ["api.e2b.dev"], "env_from": ["REEF_TERMINUS_ENVIRONMENT", "E2B_API_KEY"]},
    )
    evolution["evaluation"].update(task_timeout_s=17, forbid_residue=True, tasks=[], harbor_tasks=["test/hello-world"])
    recipe = build_recipe(
        config["implementation"],
        {"REEF_TERMINUS_ENVIRONMENT": "e2b", "E2B_API_KEY": "test-only"},
        config=config,
        runtime=served_runtime(),
    )
    recipe = dataclasses.replace(
        recipe,
        scorer=resolve_task_scorer(lambda task, result: result.trajectory[0]["reward"]),
        models={"proposer": QueueChat(proposal_reply(SEED, proposed))},
    )
    dispatcher = harness_dispatcher(tmp_path, recipe, hold_local_cycles=True)
    try:
        scenario = dispatcher.get_or_create_scenario("terminus-meta")
        report_once(scenario, "1")
        result = scenario.prepare_training_step()
        assert result.metrics["selected"] is True
        scenario.commit(result)
        assert next((tmp_path / "repository").rglob("extensions/agent.py")).read_text() == code
    finally:
        dispatcher.close()
    assert len(agents) == len(launches) == 2
    assert sum(bool(agent.import_path) for agent in agents) == 1
    assert all(not root.exists() for root in launches)


def test_the_agent_folder_is_the_upstream_proposer_as_a_method_made_of_prompts() -> None:
    """agent/ holds a prompt for the deployment's coding agent and no tree of its own: the search is the history."""
    folder = Path(recipes.meta_harness.__file__).with_name("agent")
    method = AgentMethod.from_config({"prompts": str(folder), "model": "proposer"})
    assert "{history}" in method.start and method.resume is None and method.tree == HarnessTree({})
