"""Meta-Harness method contracts on Reef's real evolution lifecycle."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from recipes.meta_harness.backend import POPULATION_STATE_KEY, MetaHarnessBackend
from recipes.meta_harness.method import MetaHarnessProposer, MetaHarnessSelector, mutations_between
from recipes.meta_harness.population import Population, PopulationStore, content_id
from recipes.meta_harness.recipe import MetaHarnessRecipe, scenario_population_path
from reef.artifact import InMemoryRepositoryBackend
from reef.core import AgentRecord, RequestType
from reef.dispatcher import Dispatcher
from reef.harness.adapters import get_adapter
from reef.harness.episode import EpisodeResult
from reef.harness.model_binding import ModelBinding, ModelBindings
from reef.recipe import RecipeConfigError
from reef.recipe.registry import build_recipe
from reef.records import RecordStore
from reef.runtime.adapters.inference_proxy import InferenceProxyRuntime
from reef.train.cordis_backend.strategies import Mutation
from reef.train.evaluation.contracts import EvaluationResult, UpdateCandidate
from reef.train.trainer import Trainer
from reef.train.types import TraceSample


PI_FAKE = """\
#!/usr/bin/env python3
import json, os, sys
from pathlib import Path

agent_dir = Path(os.environ["PI_CODING_AGENT_DIR"])
session_dir = Path(os.environ["PI_CODING_AGENT_SESSION_DIR"])
session_dir.mkdir(parents=True, exist_ok=True)
rules_path = agent_dir / "AGENTS.md"
event = {"type": "agent_end", "rules": rules_path.read_text() if rules_path.exists() else ""}
(session_dir / "session.jsonl").write_text(json.dumps(event) + "\\n")
"""

MARKER = "META-HARNESS-WINNER"
SEED = ({"id": "rules", "name": "rules", "config": {"text": "Answer carefully."}},)
IMPROVED = ({"id": "rules", "name": "rules", "config": {"text": f"Answer carefully. {MARKER}"}},)
ALTERNATE = ({"id": "rules", "name": "rules", "config": {"text": f"Answer carefully. {MARKER} Check twice."}},)
TASK = "Solve the supplied task."
SAMPLE = TraceSample(
    "trace-1",
    {
        "messages": [{"role": "user", "content": TASK}],
        "response": {"choices": [{"message": {"role": "assistant", "content": "attempt"}}]},
    },
    0.0,
    feedback="the attempt failed",
)


class ServedModel(ModelBinding):
    def __init__(self) -> None:
        super().__init__(base_url="http://127.0.0.1:9", model="served-model", api_key="dummy")


class QueueChat:
    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def chat(self, messages, **params):
        self.prompts.append(messages[-1]["content"])
        return self.replies.pop(0)


def reply(parent_id: str, entries: tuple[dict[str, Any], ...], *, hypothesis: str = "improve rules") -> str:
    return json.dumps(
        {
            "parent_id": parent_id,
            "hypothesis": hypothesis,
            "changes": "revise the general instruction",
            "entries": entries,
        }
    )


def evaluation(candidate=(1.0,), current=(0.0,)) -> EvaluationResult:
    return EvaluationResult(
        evaluator="harness_episode_pairs",
        evaluator_version="1",
        metrics={"candidate_scores": candidate, "current_scores": current},
    )


def runtime(api: str = "openai") -> InferenceProxyRuntime:
    return InferenceProxyRuntime(
        model_path="served-model",
        base_url="http://127.0.0.1:9",
        api_key="dummy",
        api=api,
    )


def score_rules(task: str, result: EpisodeResult) -> float:
    del task
    last = result.trajectory[-1] if result.trajectory else {}
    return 1.0 if MARKER in str(last.get("rules", "")) else 0.0


def non_finite_score(task: str, result: EpisodeResult) -> float:
    del task, result
    return float("nan")


def make_binary(tmp_path: Path) -> Path:
    binary = tmp_path / "fake-pi"
    binary.write_text(PI_FAKE)
    binary.chmod(0o755)
    return binary


def sections(tmp_path: Path, *, scorer=score_rules, adapter: str = "pi", **method: Any) -> dict[str, Any]:
    block = {"archive": str(tmp_path / "meta-harness")}
    block.update(method)
    return {
        "implementation": "recipes.meta_harness.recipe:MetaHarnessRecipe",
        "model": {"path": "served-model"},
        "evolution": {
            "adapter": adapter,
            "binary": str(tmp_path / "fake-pi"),
            "evaluate": scorer,
            "tasks": [TASK],
            "seed": list(SEED),
            "meta_harness": block,
        },
        "data": {"batch_size": 1},
    }


def _report_once(scenario, name: str, suffix: str) -> None:
    scenario.records.append_result(
        AgentRecord.create(
            scenario=name,
            request_type=RequestType.INFERENCE,
            payload=SAMPLE.payload,
            agent_record_id=f"i{suffix}",
        )
    )
    scenario.records.append_result(
        AgentRecord.create(
            scenario=name,
            request_type=RequestType.REPORT,
            payload={"score": 0.0, "feedback": SAMPLE.feedback, "references": [f"i{suffix}"]},
            agent_record_id=f"r{suffix}",
        )
    )


# -- composition translation ------------------------------------------------


def test_complete_compositions_use_minimal_mutations_when_order_is_stable() -> None:
    target = (
        {"id": "rules", "name": "rules", "config": {"text": "better"}},
        {"id": "notes", "name": "skill", "config": {"name": "notes", "text": "read first"}},
    )

    assert mutations_between(SEED, target) == (
        Mutation("update", "rules", {"config": {"text": "better"}}),
        Mutation(
            "create",
            "notes",
            {"name": "skill", "config": {"name": "notes", "text": "read first"}},
        ),
    )


def test_a_reordered_composition_is_replaced_atomically_in_target_order() -> None:
    before = (
        {"id": "a", "name": "rules", "config": {"text": "a"}},
        {"id": "b", "name": "rules", "config": {"text": "b"}},
    )
    after = tuple(reversed(before))

    assert mutations_between(before, after) == (
        Mutation("remove", "a"),
        Mutation("remove", "b"),
        Mutation("create", "b", {"name": "rules", "config": {"text": "b"}}),
        Mutation("create", "a", {"name": "rules", "config": {"text": "a"}}),
    )


# -- population and history-aware proposal ----------------------------------


def test_population_retains_a_losing_candidate_as_a_future_parent() -> None:
    population = Population()
    genesis = population.sync_served(SEED, step=0)
    losing, is_new = population.stage_candidate(IMPROVED, parent_id=genesis.candidate_id, step=1)
    population.record_decision(candidate_scores=(0.0,), current_scores=(1.0,), selected=False)

    assert is_new
    assert population.served_id == genesis.candidate_id
    assert population.by_id(losing.candidate_id).outcome == "retained"
    assert population.by_id(losing.candidate_id).scores == (0.0,)


def test_store_refuses_to_adopt_state_that_never_reached_commit_applied(tmp_path: Path) -> None:
    store = PopulationStore(tmp_path / "mirror.json")
    speculative = Population()
    speculative.sync_served(SEED, step=0)

    with pytest.raises(RuntimeError, match="differs from the last applied commit"):
        store.begin(speculative.to_dict())

    assert store.committed.candidates == []


def test_population_refuses_a_served_composition_that_disagrees_with_committed_state() -> None:
    population = Population()
    population.sync_served(SEED, step=0)

    with pytest.raises(ValueError, match="does not match Reef's served composition"):
        population.sync_served(IMPROVED, step=1)


def test_full_history_proposer_can_branch_from_a_retained_candidate(tmp_path: Path) -> None:
    store = PopulationStore(tmp_path / "mirror.json")
    population = Population()
    genesis = population.sync_served(SEED, step=0)
    retained, _ = population.stage_candidate(IMPROVED, parent_id=genesis.candidate_id, step=1)
    population.record_decision(candidate_scores=(0.0,), current_scores=(0.5,), selected=False)
    store.restore_committed(population.to_dict())
    store.begin(population.to_dict())
    chat = QueueChat(reply(retained.candidate_id, ALTERNATE))
    proposer = MetaHarnessProposer(
        store=store,
        descriptor=get_adapter("pi"),
        tasks=(TASK,),
        episode_repeats=1,
    )

    mutations = proposer(SEED, (SAMPLE,), ModelBindings(served=ServedModel(), named={"proposer": chat}))

    assert mutations == (Mutation("update", "rules", {"config": ALTERNATE[0]["config"]}),)
    assert store.active.pending.parent_id == retained.candidate_id
    assert genesis.candidate_id in chat.prompts[0]
    assert retained.candidate_id in chat.prompts[0]
    assert TASK in chat.prompts[0]


def test_an_undeclared_proposer_model_is_refused_rather_than_silently_served(tmp_path: Path) -> None:
    # Meta-Harness improves a fixed model's harness with a separate proposer.
    # Falling back to the served model would let a typo in evolution.models
    # turn the harness under test into its own author, measuring a different
    # experiment with nothing in the record to say so.
    store = PopulationStore(tmp_path / "mirror.json")
    population = Population()
    population.sync_served(SEED, step=0)
    store.restore_committed(population.to_dict())
    store.begin(population.to_dict())
    proposer = MetaHarnessProposer(
        store=store,
        descriptor=get_adapter("pi"),
        tasks=(TASK,),
        episode_repeats=1,
        model="typo",
    )

    with pytest.raises(ValueError, match=r"is not declared under evolution\.models"):
        proposer(SEED, (SAMPLE,), ModelBindings(served=ServedModel(), named={"proposer": QueueChat()}))


def test_incumbent_only_mode_rejects_a_historical_parent_without_losing_the_attempt(tmp_path: Path) -> None:
    store = PopulationStore(tmp_path / "mirror.json")
    population = Population()
    genesis = population.sync_served(SEED, step=0)
    retained, _ = population.stage_candidate(IMPROVED, parent_id=genesis.candidate_id, step=1)
    population.record_decision(candidate_scores=(0.0,), current_scores=(1.0,), selected=False)
    store.restore_committed(population.to_dict())
    store.begin(population.to_dict())
    chat = QueueChat(reply(retained.candidate_id, ALTERNATE))
    proposer = MetaHarnessProposer(
        store=store,
        descriptor=get_adapter("pi"),
        tasks=(TASK,),
        episode_repeats=1,
        mode="incumbent_only",
    )

    assert proposer(SEED, (SAMPLE,), ModelBindings(served=ServedModel(), named={"proposer": chat})) is None
    assert store.active.attempts[-1]["status"] == "invalid"
    assert store.active.pending_id is None


def test_component_scope_keeps_non_evolvable_parent_nodes_fixed(tmp_path: Path) -> None:
    parent = (
        *SEED,
        {"id": "notes", "name": "skill", "config": {"name": "notes", "text": "fixed"}},
    )
    changed_fixed_node = (
        *IMPROVED,
        {"id": "notes", "name": "skill", "config": {"name": "notes", "text": "changed"}},
    )
    store = PopulationStore(tmp_path / "mirror.json")
    population = Population()
    genesis = population.sync_served(parent, step=0)
    store.restore_committed(population.to_dict())
    store.begin(population.to_dict())
    chat = QueueChat(reply(genesis.candidate_id, changed_fixed_node))
    proposer = MetaHarnessProposer(
        store=store,
        descriptor=get_adapter("pi"),
        tasks=(TASK,),
        episode_repeats=1,
        kinds=("rules",),
    )

    assert proposer(parent, (SAMPLE,), ModelBindings(served=ServedModel(), named={"proposer": chat})) is None
    assert store.active.attempts[-1]["status"] == "invalid"
    assert "may change only configured components" in store.active.attempts[-1]["error"]


def test_candidate_and_episode_budgets_stop_before_another_model_call(tmp_path: Path) -> None:
    store = PopulationStore(tmp_path / "mirror.json")
    population = Population()
    genesis = population.sync_served(SEED, step=0)
    candidate, _ = population.stage_candidate(IMPROVED, parent_id=genesis.candidate_id, step=1)
    population.record_decision(candidate_scores=(1.0,), current_scores=(0.0,), selected=True)
    assert candidate.outcome == "selected"
    store.restore_committed(population.to_dict())
    store.begin(population.to_dict())
    chat = QueueChat(reply(candidate.candidate_id, ALTERNATE))
    proposer = MetaHarnessProposer(
        store=store,
        descriptor=get_adapter("pi"),
        tasks=(TASK,),
        episode_repeats=1,
        max_candidates=1,
    )

    assert proposer(IMPROVED, (SAMPLE,), ModelBindings(served=ServedModel(), named={"proposer": chat})) is None
    assert chat.prompts == []
    assert store.active.attempts[-1] == {
        "step": 1,
        "status": "budget_exhausted",
        "budget": "max_candidates",
    }


def test_selector_judges_against_the_frontier_the_incumbent_was_admitted_on(tmp_path: Path) -> None:
    # Upstream keeps a high-water frontier and never re-runs the incumbent, so
    # a candidate must beat the score the incumbent was admitted on, not the
    # one it happens to post in this batch. Reproducing the search means
    # reproducing that, including its bias against a lucky incumbent.
    store = PopulationStore(tmp_path / "mirror.json")
    population = Population()
    genesis = population.sync_served(SEED, step=0)
    genesis.scores = (0.8,)
    proposed, _ = population.stage_candidate(IMPROVED, parent_id=genesis.candidate_id, step=1)
    store.restore_committed(population.to_dict())
    store.begin(population.to_dict())

    decision = MetaHarnessSelector(store).decide(
        UpdateCandidate("backend-candidate"),
        evaluation(candidate=(0.7,), current=(0.1,)),
    )

    assert not decision.selected
    assert decision.metrics["incumbent_mean"] == 0.8
    # The frontier does not drift down to this batch's measurement.
    assert store.active.by_id(genesis.candidate_id).scores == (0.8,)
    assert store.active.by_id(proposed.candidate_id).outcome == "retained"
    assert store.active.episode_calls == 2


def test_selector_retains_a_non_winner_as_a_future_parent(tmp_path: Path) -> None:
    store = PopulationStore(tmp_path / "mirror.json")
    population = Population()
    genesis = population.sync_served(SEED, step=0)
    proposed, _ = population.stage_candidate(IMPROVED, parent_id=genesis.candidate_id, step=1)
    store.restore_committed(population.to_dict())
    store.begin(population.to_dict())

    decision = MetaHarnessSelector(store).decide(
        UpdateCandidate("backend-candidate"),
        evaluation(candidate=(0.2,), current=(0.6,)),
    )

    assert not decision.selected
    assert store.active.by_id(proposed.candidate_id).outcome == "retained"
    assert store.active.served_id == genesis.candidate_id


# -- recipe and transactional lifecycle ------------------------------------


@pytest.mark.parametrize("adapter", ["pi", "codex"])
def test_recipe_is_adapter_agnostic(tmp_path: Path, adapter: str) -> None:
    config = sections(tmp_path, adapter=adapter)
    built = build_recipe(
        config["implementation"],
        {},
        config=config,
        runtime=runtime("responses" if adapter == "codex" else "openai"),
    )

    assert isinstance(built, MetaHarnessRecipe)
    assert built.adapter == adapter
    assert built.mode == "full_history"
    assert isinstance(built.build("general-task", RecordStore()), Trainer)


def test_recipe_rejects_a_selection_override_that_would_split_population_from_serving(tmp_path: Path) -> None:
    config = sections(tmp_path)
    config["evolution"]["selection"] = "always"

    with pytest.raises(RecipeConfigError, match=r"owns evolution\.selection"):
        build_recipe(config["implementation"], {}, config=config, runtime=runtime())


def test_scenario_population_path_cannot_escape_its_directory(tmp_path: Path) -> None:
    directory = tmp_path / "meta-harness"
    traversal = scenario_population_path(directory, "../outside")
    absolute = scenario_population_path(directory, "/tmp/outside")

    assert traversal.parent == directory
    assert absolute.parent == directory
    assert traversal != absolute
    assert "/" not in traversal.name and ".." not in traversal.name


def test_one_step_commits_population_and_composition_together(tmp_path: Path) -> None:
    make_binary(tmp_path)
    config = sections(tmp_path)
    recipe = build_recipe(config["implementation"], {}, config=config, runtime=runtime())
    chat = QueueChat(reply(content_id(SEED), IMPROVED))
    recipe = dataclasses.replace(recipe, models={"proposer": chat})
    initial = tmp_path / "initial"
    initial.mkdir()
    dispatcher = Dispatcher(
        recipe,
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        agent_record_dir=tmp_path / "records",
    )
    scenario_name = "../general-meta-harness"
    try:
        scenario = dispatcher.get_or_create_scenario(scenario_name)
        assert scenario is not None
        _report_once(scenario, scenario_name, "1")
        result = scenario.prepare_training_step()
        assert result is not None and result.state is not None
        backend = scenario.trainer._training_backend
        assert isinstance(backend, MetaHarnessBackend)
        with pytest.raises(RuntimeError, match="no active step"):
            _ = backend._population_store.active
        mirror = scenario_population_path(tmp_path / "meta-harness", scenario_name)
        assert not mirror.exists()
        assert result.metrics["selected"] is True
        assert result.metrics["selection"]["policy"] == "meta_harness"
        assert result.state[POPULATION_STATE_KEY]["served_id"] == content_id(IMPROVED)
        scenario.commit(result)
        committed = scenario.trainer.state[POPULATION_STATE_KEY]
    finally:
        dispatcher.close()

    assert json.loads(mirror.read_text()) == committed
    assert committed["pending_id"] is None
    assert committed["episode_calls"] == 2
    assert [row["outcome"] for row in committed["candidates"]] == ["selected", "selected"]


def test_failed_evaluation_restores_population_and_writes_no_mirror(tmp_path: Path) -> None:
    make_binary(tmp_path)
    config = sections(tmp_path, scorer=non_finite_score)
    recipe = build_recipe(config["implementation"], {}, config=config, runtime=runtime())
    recipe = dataclasses.replace(recipe, models={"proposer": QueueChat(reply(content_id(SEED), IMPROVED))})
    initial = tmp_path / "initial"
    initial.mkdir()
    dispatcher = Dispatcher(
        recipe,
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        agent_record_dir=tmp_path / "records",
    )
    try:
        scenario = dispatcher.get_or_create_scenario("failed-evaluation")
        assert scenario is not None
        initial_state = scenario.trainer.state
        _report_once(scenario, "failed-evaluation", "1")
        with pytest.raises(ValueError, match="non-finite score"):
            scenario.prepare_training_step()
        assert scenario.trainer.state == initial_state
        assert not scenario_population_path(tmp_path / "meta-harness", "failed-evaluation").exists()
    finally:
        dispatcher.close()


def test_failed_commit_keeps_mirror_at_previous_population_and_restart_heals_stale_mirror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_binary(tmp_path)
    config = sections(tmp_path)
    recipe = build_recipe(config["implementation"], {}, config=config, runtime=runtime())
    chat = QueueChat(
        reply(content_id(SEED), IMPROVED),
        reply(content_id(IMPROVED), ALTERNATE),
    )
    recipe = dataclasses.replace(recipe, models={"proposer": chat})
    initial = tmp_path / "initial"
    initial.mkdir()
    factory = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    records = tmp_path / "records"
    dispatcher = Dispatcher(recipe, factory, agent_record_dir=records)
    mirror = scenario_population_path(tmp_path / "meta-harness", "commit-failure")
    try:
        scenario = dispatcher.get_or_create_scenario("commit-failure")
        assert scenario is not None
        _report_once(scenario, "commit-failure", "1")
        first = scenario.prepare_training_step()
        assert first is not None
        scenario.commit(first)
        committed = json.loads(mirror.read_text())

        _report_once(scenario, "commit-failure", "2")
        second = scenario.prepare_training_step()
        assert second is not None and second.artifact is None
        backend = scenario.trainer._training_backend
        assert isinstance(backend, MetaHarnessBackend)
        with pytest.raises(RuntimeError, match="no active step"):
            _ = backend._population_store.active
        commit_log = scenario.commit_log
        assert commit_log is not None
        monkeypatch.setattr(commit_log, "append", lambda record: (_ for _ in ()).throw(RuntimeError("offline")))
        with pytest.raises(RuntimeError, match="offline"):
            scenario.commit(second)
        assert json.loads(mirror.read_text()) == committed
    finally:
        dispatcher.close()

    # A stale/corrupt mirror is never loaded as search state.  Recovery gets
    # the prior durable commit and rewrites the mirror from that value.
    mirror.write_text('{"stale": true}\n')
    restarted = Dispatcher(recipe, factory, agent_record_dir=records)
    try:
        recovered = restarted.get_or_create_scenario("commit-failure")
        assert recovered is not None
        assert recovered.scenario_step == 1
        assert recovered.trainer.state[POPULATION_STATE_KEY] == committed
        assert json.loads(mirror.read_text()) == committed
    finally:
        restarted.close()
