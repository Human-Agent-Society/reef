"""Meta-Harness method contracts on Reef's real evolution lifecycle."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest
from tests.support.harness_methods import (
    ALTERNATE,
    IMPROVED,
    SAMPLE,
    SEED,
    TASK,
    QueueChat,
    meta_harness_sections,
    proposal_reply,
    report_once,
    served_runtime,
)
from tests.support.harness_steps import TreeReleases, make_pi_binary

from recipes.meta_harness.population import POPULATION_KEY, Population, population_for_step
from recipes.meta_harness.proposer import MetaHarnessProposer, MetaHarnessSettings, parse_proposal
from recipes.meta_harness.recipe import MetaHarnessRecipe
from recipes.meta_harness.selection import MetaHarnessSelection
from reef.artifact import InMemoryRepositoryBackend
from reef.core.evaluation import EvaluationResult
from reef.dispatcher import Dispatcher
from reef.evaluation.harness import HarnessCandidate
from reef.harness.adapters import get_adapter
from reef.harness.episodes.model_binding import ModelBinding, ModelBindings
from reef.harness.episodes.run import EpisodeResult
from reef.harness.tree import HarnessTree, TreeKind, TreePatch
from reef.recipe import RecipeConfigError
from reef.recipe.registry import build_recipe
from reef.storage.commit_history import CommitHistoryScenarioStore
from reef.storage.sqlite import SQLiteRecordStore, SQLiteScenarioStorage
from reef.train.backend import ComponentReleases
from reef.train.harness_backend.contract import ProposalContext, SampleSource, StepOutcome, TreeProposal
from reef.train.harness_backend.state import HarnessState
from reef.train.trainer import Trainer


class ServedModel(ModelBinding):
    def __init__(self) -> None:
        super().__init__(base_url="http://127.0.0.1:9", model="served-model", api_key="dummy")


def evaluation(candidate=(1.0,), current=(0.0,)) -> EvaluationResult:
    return EvaluationResult(
        evaluator="harness_episode_pairs",
        evaluator_version="1",
        metrics={"candidate_scores": candidate, "current_scores": current},
    )


def outcome(step: int, selected: bool, candidate=(1.0,), current=(0.0,)) -> StepOutcome:
    return StepOutcome(step, TreePatch({"rules.md": "x"}), selected, "decided", candidate, current)


def non_finite_score(task: str, result: EpisodeResult) -> float:
    del task, result
    return float("nan")


def proposer(**settings: Any) -> MetaHarnessProposer:
    return MetaHarnessProposer(MetaHarnessSettings(evaluation_cost=2, **settings))


def context(population: Population, chat: ModelBinding, *, tree: HarnessTree = SEED, step: int = 2) -> ProposalContext:
    """A step over ``tree`` with ``population`` as the committed method state and ``chat`` as the proposer model."""
    return ProposalContext(
        step=step,
        tree=tree,
        descriptor=get_adapter("pi"),
        samples=(SAMPLE,),
        sources=(SampleSource("trace-1", "untagged"),),
        request=None,
        evaluation_tasks=(TASK,),
        models=ModelBindings(served=ServedModel(), named={"proposer": chat}),
        method_state={POPULATION_KEY: population.to_dict()},
    )


def population_of(proposal: TreeProposal) -> Population:
    assert proposal.method_state is not None
    return Population.from_dict(proposal.method_state[POPULATION_KEY])


def retained_population() -> tuple[Population, str, str]:
    """A population whose served seed beat the one candidate it was offered: the seed's id and the candidate's."""
    population = Population()
    genesis = population.sync_served(SEED, step=1)
    losing, _ = population.stage_candidate(IMPROVED, parent_id=genesis.candidate_id, step=1)
    population.settle(outcome(1, False, candidate=(0.0,), current=(0.5,)))
    return population, genesis.candidate_id, losing.candidate_id


# -- population ------------------------------------------------------------------


def test_population_retains_a_losing_candidate_as_a_future_parent() -> None:
    population, genesis, losing = retained_population()

    assert population.served_id == genesis
    assert population.by_id(losing).outcome == "retained"
    assert population.by_id(losing).scores == (0.0,)
    # The incumbent takes its first scores from the step that measured it beside the candidate.
    assert population.by_id(genesis).scores == (0.5,)
    assert population.episode_calls == 2 and population.pending_id is None


def test_a_served_tree_the_population_does_not_hold_joins_it_as_a_new_root() -> None:
    """A rollback or an operator's edit serves a tree the search never proposed: it becomes a parentless member."""
    population = Population()
    population.sync_served(SEED, step=1)
    rolled = population.sync_served(IMPROVED, step=2)

    assert rolled.parent_id is None and rolled.outcome == "selected"
    assert population.served_id == IMPROVED.checksum() and len(population.candidates) == 2


def test_a_step_folds_in_only_the_outcome_of_the_step_that_proposed_its_pending_candidate() -> None:
    population = Population()
    genesis = population.sync_served(SEED, step=1)
    population.stage_candidate(IMPROVED, parent_id=genesis.candidate_id, step=3)
    state = {POPULATION_KEY: population.to_dict()}

    settled = population_for_step(state, outcome(3, True))
    assert settled.served_id == IMPROVED.checksum() and settled.by_id(IMPROVED.checksum()).scores == (1.0,)
    # The proposing step never reached an evaluation: the candidate is dropped, unscored.
    for last in (outcome(2, True), None):
        dropped = population_for_step(state, last)
        assert dropped.pending_id is None and [candidate.candidate_id for candidate in dropped.candidates] == [
            SEED.checksum()
        ]


def test_the_population_state_round_trips_and_checks_its_members() -> None:
    population, _, _ = retained_population()
    assert Population.from_dict(population.to_dict()) == population
    corrupted = population.to_dict()
    corrupted["candidates"][1]["files"] = {"rules.md": "another tree"}
    with pytest.raises(ValueError, match="candidate id does not match its tree"):
        Population.from_dict(corrupted)
    with pytest.raises(ValueError, match="schema_version"):
        Population.from_dict({**population.to_dict(), "schema_version": 1})


# -- proposer -----------------------------------------------------------------


def test_full_history_proposer_can_branch_from_a_retained_candidate() -> None:
    population, genesis, retained = retained_population()
    chat = QueueChat(proposal_reply(IMPROVED, ALTERNATE))

    proposal = proposer().propose(context(population, chat))

    # The patch is stated against the served tree, whatever parent the tree came from.
    assert proposal.patch == SEED.diff(ALTERNATE)
    proposed = population_of(proposal)
    assert proposed.pending.parent_id == retained and proposed.pending_step == 2
    assert proposal.notes["status"] == "proposed" and proposal.notes["parent_id"] == retained
    assert genesis in chat.prompts[0] and retained in chat.prompts[0]
    assert TASK in chat.prompts[0] and "the attempt failed" in chat.prompts[0]


def test_an_undeclared_proposer_model_is_refused_rather_than_silently_served() -> None:
    # Meta-Harness improves a fixed model's harness with a separate proposer.
    # Falling back to the served model would let a typo in evolution.models
    # turn the harness under test into its own author, measuring a different
    # experiment with nothing in the record to say so.
    population = Population()
    population.sync_served(SEED, step=1)

    with pytest.raises(ValueError, match=r"is not declared under evolution\.models"):
        proposer(model="typo").propose(context(population, QueueChat()))


def test_incumbent_only_mode_rejects_a_historical_parent_without_losing_the_attempt() -> None:
    population, _, retained = retained_population()
    chat = QueueChat(proposal_reply(IMPROVED, ALTERNATE))

    proposal = proposer(mode="incumbent_only").propose(context(population, chat))

    assert proposal.patch == TreePatch({}) and proposal.notes["status"] == "invalid"
    kept = population_of(proposal)
    assert kept.attempts[-1]["status"] == "invalid" and kept.pending_id is None
    assert kept.proposer_calls == 1  # the attempt is spent and recorded
    assert retained not in chat.prompts[0]  # the prompt shows the incumbent alone


def test_component_scope_keeps_non_evolvable_parent_files_fixed() -> None:
    parent = HarnessTree({**SEED.files, "skills/notes/SKILL.md": "fixed"})
    changed = HarnessTree({**IMPROVED.files, "skills/notes/SKILL.md": "changed"})
    population = Population()
    population.sync_served(parent, step=1)

    proposal = proposer(kinds=(TreeKind.RULES,)).propose(
        context(population, QueueChat(proposal_reply(parent, changed)), tree=parent)
    )

    assert proposal.patch == TreePatch({})
    error = population_of(proposal).attempts[-1]["error"]
    assert "may change only the files of kinds ['rules']" in error


def test_a_proposal_that_fails_validation_or_repeats_a_candidate_is_recorded_and_skipped() -> None:
    population, genesis, retained = retained_population()
    empty_skill = HarnessTree({**SEED.files, "skills/empty/SKILL.md": "\n"})
    duplicate = proposer().propose(context(population, QueueChat(proposal_reply(SEED, IMPROVED))))
    assert duplicate.patch == TreePatch({}) and duplicate.notes == {"status": "duplicate", "candidate_id": retained}
    invalid = proposer().propose(context(population, QueueChat(proposal_reply(SEED, empty_skill))))
    assert invalid.patch == TreePatch({})
    assert "proposal fails validation: skills/empty/SKILL.md" in invalid.notes["error"]
    unknown = proposer().propose(context(population, QueueChat("not json at all")))
    assert unknown.notes == {"status": "invalid", "error": "proposer response contains no JSON object"}
    assert population_of(unknown).served_id == genesis


def test_candidate_and_episode_budgets_stop_before_another_model_call() -> None:
    population = Population()
    genesis = population.sync_served(SEED, step=1)
    population.stage_candidate(IMPROVED, parent_id=genesis.candidate_id, step=1)
    population.settle(outcome(1, True))
    chat = QueueChat(proposal_reply(IMPROVED, ALTERNATE))

    exhausted = proposer(max_candidates=1).propose(context(population, chat, tree=IMPROVED))
    assert exhausted.patch == TreePatch({}) and chat.prompts == []
    assert population_of(exhausted).attempts[-1] == {
        "step": 1,
        "status": "budget_exhausted",
        "budget": "max_candidates",
    }
    # Two episodes spent and two more for the next evaluation reach a budget of three.
    episodes = proposer(max_target_episodes=3).propose(context(population, chat, tree=IMPROVED))
    assert episodes.notes == {"status": "budget_exhausted", "budget": "max_target_episodes"} and chat.prompts == []


def test_the_reply_is_one_object_of_the_four_proposal_fields() -> None:
    fenced = "```json\n" + proposal_reply(SEED, IMPROVED) + "\n```"
    assert parse_proposal(fenced)["files"] == dict(IMPROVED.files)
    with pytest.raises(ValueError, match="unknown fields"):
        parse_proposal('{"parent_id": "p", "files": {}, "entries": []}')
    with pytest.raises(ValueError, match="requires a files object"):
        parse_proposal('{"parent_id": "p"}')


# -- selection -------------------------------------------------------------------


def harness_candidate(population: Population, proposal_id: str | None = None) -> HarnessCandidate:
    return HarnessCandidate(
        candidate_id="backend-candidate",
        candidate_tree=IMPROVED,
        current_tree=SEED,
        patch=SEED.diff(IMPROVED),
        evaluation_tasks=(TASK,),
        proposal_id=proposal_id,
        method_state={POPULATION_KEY: population.to_dict()},
    )


def test_selection_judges_against_the_frontier_the_incumbent_was_admitted_on() -> None:
    # Upstream keeps a high-water frontier and never re-runs the incumbent, so
    # a candidate must beat the score the incumbent was admitted on, not the
    # one it happens to post in this batch. Reproducing the search means
    # reproducing that, including its bias against a lucky incumbent.
    population = Population()
    genesis = population.sync_served(SEED, step=1)
    genesis.scores = (0.8,)
    proposed, _ = population.stage_candidate(IMPROVED, parent_id=genesis.candidate_id, step=1)

    decision = MetaHarnessSelection(None).decide(
        harness_candidate(population), evaluation(candidate=(0.7,), current=(0.1,))
    )

    assert not decision.selected and decision.policy == "meta_harness"
    assert decision.metrics["incumbent_mean"] == 0.8
    assert decision.metrics["population_candidate_id"] == proposed.candidate_id
    assert decision.metrics["target_episode_calls"] == 2
    # The next step records the result; the frontier does not drift down to this batch's measurement.
    population.settle(outcome(1, False, candidate=(0.7,), current=(0.1,)))
    assert population.by_id(genesis.candidate_id).scores == (0.8,)
    assert population.by_id(proposed.candidate_id).outcome == "retained"
    assert population.served_id == genesis.candidate_id


def test_the_first_selection_compares_against_the_incumbents_own_run() -> None:
    population = Population()
    genesis = population.sync_served(SEED, step=1)
    population.stage_candidate(IMPROVED, parent_id=genesis.candidate_id, step=1)

    decision = MetaHarnessSelection(None).decide(harness_candidate(population), evaluation((0.6,), (0.2,)))

    assert decision.selected and decision.metrics["incumbent_mean"] == 0.2


def test_selection_refuses_scores_it_cannot_compare() -> None:
    population = Population()
    genesis = population.sync_served(SEED, step=1)
    population.stage_candidate(IMPROVED, parent_id=genesis.candidate_id, step=1)
    selection = MetaHarnessSelection(None)
    with pytest.raises(ValueError, match="only finite values"):
        selection.decide(harness_candidate(population), evaluation((float("nan"),), (0.0,)))
    with pytest.raises(ValueError, match="equal lengths"):
        selection.decide(harness_candidate(population), evaluation((1.0, 1.0), (0.0,)))


def test_a_client_proposal_from_the_inbox_is_judged_against_the_incumbent() -> None:
    population = Population()
    population.sync_served(SEED, step=1)
    decision = MetaHarnessSelection(None).decide(harness_candidate(population, proposal_id="p1"), evaluation())
    assert decision.selected


# -- recipe and the committed lifecycle ----------------------------------------------


@pytest.mark.parametrize("adapter", ["pi", "codex"])
def test_recipe_is_adapter_agnostic(tmp_path: Path, adapter: str) -> None:
    config = meta_harness_sections(tmp_path, adapter=adapter)
    built = build_recipe(
        config["implementation"],
        {},
        config=config,
        runtime=served_runtime("responses" if adapter == "codex" else "openai"),
    )

    assert isinstance(built, MetaHarnessRecipe)
    assert built.adapter == adapter and built.seed == SEED
    assert isinstance(built.propose, MetaHarnessProposer) and built.propose.settings.mode == "full_history"
    # Both sides of the one task: the cost the episode budget counts per evaluation.
    assert built.propose.settings.evaluation_cost == 2
    assert built.selection.plugin_class is MetaHarnessSelection
    releases = ComponentReleases(TreeReleases(built.seed), "harness")
    assert isinstance(built.build("general-task", SQLiteRecordStore(), releases=releases), Trainer)


def test_recipe_rejects_a_selection_override_that_would_split_population_from_serving(tmp_path: Path) -> None:
    config = meta_harness_sections(tmp_path)
    config["evolution"]["selection"] = "always"

    with pytest.raises(RecipeConfigError, match=r"owns evolution\.selection"):
        build_recipe(config["implementation"], {}, config=config, runtime=served_runtime())


@pytest.mark.parametrize(
    ("block", "message"),
    [
        ({"archive": "meta-harness"}, r"unknown key\(s\) archive"),
        ({"mode": "greedy"}, "mode must be one of"),
        ({"components": "rules"}, "must be a list of tree kinds"),
        ({"max_files": 0}, "max_files must be a positive integer"),
    ],
)
def test_recipe_rejects_an_invalid_meta_harness_block(tmp_path: Path, block: dict[str, Any], message: str) -> None:
    config = meta_harness_sections(tmp_path, **block)
    with pytest.raises(RecipeConfigError, match=message):
        build_recipe(config["implementation"], {}, config=config, runtime=served_runtime())


@pytest.mark.parametrize(
    "option,value",
    [("publish", "review"), ("review_kinds", ["rules"]), ("recheck_every", 1), ("promote_failures", True)],
)
def test_recipe_rejects_options_that_diverge_from_committed_search(tmp_path, option, value):
    config = meta_harness_sections(tmp_path)
    config["evolution"][option] = value

    with pytest.raises((ValueError, RecipeConfigError), match="Meta-Harness requires"):
        build_recipe(config["implementation"], {}, config=config, runtime=served_runtime())


def meta_harness_dispatcher(tmp_path: Path, recipe: MetaHarnessRecipe, storage: Path | None = None) -> Dispatcher:
    """A dispatcher whose base release is the recipe's seed, as a deployment's is."""
    initial = tmp_path / "initial"
    if not initial.exists():
        recipe.seed.write(initial)
    return Dispatcher(
        recipe,
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        scenario_storage=SQLiteScenarioStorage(storage),
    )


def built_recipe(tmp_path: Path, *replies: str, **sections: Any) -> MetaHarnessRecipe:
    make_pi_binary(tmp_path)
    config = meta_harness_sections(tmp_path, **sections)
    recipe = build_recipe(config["implementation"], {}, config=config, runtime=served_runtime())
    assert isinstance(recipe, MetaHarnessRecipe)
    return dataclasses.replace(recipe, models={"proposer": QueueChat(*replies)})


def test_one_step_commits_the_population_with_the_tree(tmp_path: Path) -> None:
    recipe = built_recipe(tmp_path, proposal_reply(SEED, IMPROVED))
    dispatcher = meta_harness_dispatcher(tmp_path, recipe, tmp_path / "records")
    try:
        scenario = dispatcher.get_or_create_scenario("general-meta-harness")
        assert scenario is not None
        report_once(scenario, "general-meta-harness", "1")
        result = scenario.prepare_training_step()
        assert result is not None and result.state is not None
        assert result.metrics["selected"] is True and result.metrics["patch"] == dict(IMPROVED.files)
        assert result.metrics["selection"]["policy"] == "meta_harness"
        assert result.metrics["population_candidate_id"] == IMPROVED.checksum()
        assert result.state["method_state"][POPULATION_KEY]["pending_id"] == IMPROVED.checksum()
        scenario.commit(result)
        committed = HarnessState.from_json(scenario.trainer.state)
    finally:
        dispatcher.close()

    # The next step folds the evaluation into the committed population.
    population = population_for_step(committed.method_state, committed.last_outcome)
    assert population.served_id == IMPROVED.checksum() and population.pending_id is None
    assert population.episode_calls == 2
    assert [candidate.outcome for candidate in population.candidates] == ["selected", "selected"]


def test_terminus_extension_uses_shared_recipe_episode_runner_and_publication(tmp_path, monkeypatch) -> None:
    # Replace only process launch and the remote trial. Rendering, model
    # binding, Harbor's agent contract, scoring, selection and commit are real.
    pytest.importorskip("reef_eval")
    pytest.importorskip("harbor")
    from types import SimpleNamespace

    from harbor.agents.terminus_2 import Terminus2
    from harbor.models.trial.config import TrialConfig
    from harbor.utils.import_path import import_class

    from reef.harness.runners.terminus import runner
    from reef.sandbox.process import ISOLATION_ENV, ProcessOutcome, SandboxExecutor

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
    config = meta_harness_sections(
        tmp_path, adapter="terminus", scorer=lambda task, result: result.trajectory[0]["reward"]
    )
    config["evolution"].pop("binary")
    config["evolution"].update(
        executor="sandbox",
        episode_timeout_s=17,
        forbid_residue=True,
        tasks=["test/hello-world"],
        sandbox={"egress_hosts": ["api.e2b.dev"], "env_from": ["REEF_TERMINUS_ENVIRONMENT", "E2B_API_KEY"]},
    )
    recipe = build_recipe(
        config["implementation"],
        {"REEF_TERMINUS_ENVIRONMENT": "e2b", "E2B_API_KEY": "test-only"},
        config=config,
        runtime=served_runtime(),
    )
    recipe = dataclasses.replace(recipe, models={"proposer": QueueChat(proposal_reply(SEED, proposed))})
    dispatcher = meta_harness_dispatcher(tmp_path, recipe)
    try:
        scenario = dispatcher.get_or_create_scenario("terminus-meta")
        report_once(scenario, "terminus-meta", "1")
        result = scenario.prepare_training_step()
        assert result.metrics["selected"] is True
        scenario.commit(result)
        committed = HarnessState.from_json(scenario.trainer.state)
        assert population_for_step(committed.method_state, committed.last_outcome).served_id == proposed.checksum()
        assert next((tmp_path / "repository").rglob("extensions/agent.py")).read_text() == code
    finally:
        dispatcher.close()
    assert len(agents) == len(launches) == 2
    assert sum(bool(agent.import_path) for agent in agents) == 1
    assert all(not root.exists() for root in launches)


def test_failed_evaluation_restores_the_committed_state(tmp_path: Path) -> None:
    recipe = built_recipe(tmp_path, proposal_reply(SEED, IMPROVED), scorer=non_finite_score)
    dispatcher = meta_harness_dispatcher(tmp_path, recipe, tmp_path / "records")
    try:
        scenario = dispatcher.get_or_create_scenario("failed-evaluation")
        assert scenario is not None
        initial_state = scenario.trainer.state
        report_once(scenario, "failed-evaluation", "1")
        with pytest.raises(ValueError, match="finite"):
            scenario.prepare_training_step()
        assert scenario.trainer.state == initial_state
    finally:
        dispatcher.close()


@pytest.mark.parametrize("failure", ["activation", "publication", "commit"])
def test_failed_publication_does_not_advance_population_or_head(tmp_path, monkeypatch, failure):
    recipe = built_recipe(tmp_path, proposal_reply(SEED, IMPROVED))
    dispatcher = meta_harness_dispatcher(tmp_path, recipe, tmp_path / "records")
    try:
        scenario = dispatcher.get_or_create_scenario("publish-failure")
        previous = dict(scenario.trainer.state)
        report_once(scenario, "publish-failure", "1")
        result = scenario.prepare_training_step()
        assert result.artifact is not None
        backend = scenario.trainer.candidate_backend
        assert HarnessTree(backend.releases.head().files) == SEED
        head = scenario.current_artifact_ref()
        checkpoint = scenario.repository.require_checkpoint_artifact()
        durable_head = scenario.repository.backend.current()

        def fail(*args, **kwargs):
            raise RuntimeError("unavailable")

        if failure == "publication":
            monkeypatch.setattr(scenario.repository, "publish", fail)
        elif failure == "commit":
            assert isinstance(scenario.store, CommitHistoryScenarioStore)
            monkeypatch.setattr(scenario.store.commit_history, "append", fail)
        else:
            monkeypatch.setattr(scenario._committer, "_activate", fail)
        with pytest.raises(RuntimeError, match="unavailable"):
            scenario.commit(result)
        assert scenario.trainer.state == previous
        assert HarnessTree(backend.releases.head().files) == SEED
        assert scenario.current_artifact_ref() == head
        assert scenario.repository.require_checkpoint_artifact() == checkpoint
        assert scenario.repository.backend.current() == durable_head
    finally:
        dispatcher.close()
