"""Guarantees of the harness_evolve cookbook recipe and its evolution backend,
hermetic: episodes run a fake pi binary through the real adapter path."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

import reef.train.cordis_backend as reef_cordis_backend
from reef.artifact import InMemoryRepositoryBackend
from reef.core import AgentRecord, RequestType
from reef.core.reports import ScoredRolloutReport
from reef.dispatcher import Dispatcher
from reef.harness.adapters import get_adapter
from reef.harness.episode import EpisodeResult
from reef.harness.model_binding import ModelBinding, ModelBindingError, ModelBindings
from reef.recipe import RecipeConfigError
from reef.recipe.registry import RecipeRegistry, recipe_class_for
from reef.records import RecordStore
from reef.runtime.adapters.inference_proxy import InferenceProxyRuntime
from reef.train.cordis_backend import CordisBackend, CordisRecipe, Mutation, MutationError, ScoreComparisonSelector
from reef.train.cordis_backend.strategies import resolve_episode_scorer, resolve_proposer
from reef.train.evaluation import DefaultCandidateEvaluationPlugin
from reef.train.trainer import Trainer
from reef.train.types import NoArtifactPublication, SavedArtifactPublication, TraceBatch, TraceSample, TrainStepResult

# The fake harness scores itself: its trajectory carries the rules text, so
# the episode scorer can prefer compositions containing the marker.
PI_FAKE = """\
#!/usr/bin/env python3
import json, os, sys
from pathlib import Path

prompt = sys.argv[sys.argv.index("-p") + 1]
agent_dir = Path(os.environ["PI_CODING_AGENT_DIR"])
session_dir = Path(os.environ["PI_CODING_AGENT_SESSION_DIR"])
session_dir.mkdir(parents=True, exist_ok=True)
rules_path = agent_dir / "AGENTS.md"
event = {"type": "agent_end", "rules": rules_path.read_text() if rules_path.exists() else ""}
(session_dir / "session.jsonl").write_text(json.dumps(event) + "\\n")
"""

RULES = {"name": "rules", "config": {"text": "Answer briefly."}}
SKILL = {"name": "skill", "config": {"name": "notes", "text": "# notes"}}

# The b200 first-boot baseline shape: without these nodes the tree renders
# an empty models.json, no episode reaches a model, and every comparison ties.
# The seed carries no credential; admission refuses key-named fields (#476)
# and the episode binding is injected at render time.
SEED_MODELS = {
    "id": "models",
    "name": "config",
    "config": {
        "target": "models",
        "data": {
            "providers": {
                "qwen": {
                    "api": "openai-completions",
                    "baseUrl": "http://localhost:8000/v1",
                    "models": [{"id": "qwen3-8b"}],
                }
            }
        },
    },
}
SEED_SETTINGS = {
    "id": "settings",
    "name": "config",
    "config": {"data": {"defaultModel": "qwen/qwen3-8b", "defaultProvider": "qwen"}},
}


def evaluate(task: str, result: EpisodeResult) -> float:
    del task
    return 1.0 if "marker" in result.trajectory[-1]["rules"] else 0.0


def batch() -> TraceBatch:
    return TraceBatch("demo:trace:1", (TraceSample("a1", {"messages": []}, 0.0),))


# The deployment's model binding: where episodes and proposals reach a model.
# It is rendered into episodes at run time and never enters a seed or a tree.
MODEL = ModelBinding(base_url="http://localhost:8000", model="qwen3-8b", api_key="dummy")


def runtime() -> InferenceProxyRuntime:
    return InferenceProxyRuntime(model_path=MODEL.model, base_url=MODEL.base_url, api_key=MODEL.api_key)


def recipe(tmp_path: Path, propose, seed: tuple = ()) -> CordisRecipe:
    return CordisRecipe(
        resolve_proposer(propose),
        resolve_episode_scorer(evaluate),
        ("task one",),
        binary=str(make_binary(tmp_path)),
        seed=seed,
        runtime=runtime(),
    )


def make_binary(tmp_path: Path) -> Path:
    binary = tmp_path / "fake-pi"
    binary.write_text(PI_FAKE)
    binary.chmod(0o755)
    return binary


def backend(tmp_path: Path, propose, seed: tuple = ()) -> CordisBackend:
    return CordisBackend(
        descriptor=get_adapter("pi"),
        propose=resolve_proposer(propose),
        score_episode=resolve_episode_scorer(evaluate),
        tasks=("task one",),
        models=MODEL,
        binary=str(make_binary(tmp_path)),
        seed=seed,
    )


def run_backend_step(
    backend: CordisBackend,
    trace_batch: TraceBatch,
    state,
) -> TrainStepResult:
    """Exercise the backend phases directly; production runs them in Trainer."""
    prepared = backend.prepare_step(trace_batch, state, 0)
    if prepared.outcome == "skip":
        return TrainStepResult(prepared.state, prepared.metrics)
    candidate = prepared.candidate
    assert candidate is not None
    try:
        evaluator = DefaultCandidateEvaluationPlugin(backend, ScoreComparisonSelector())
        evaluation = evaluator.evaluate(candidate)
        decision = evaluator.decide(candidate, evaluation)
        return backend.settle_step(prepared, decision)
    except BaseException:
        backend.abort_step(prepared)
        raise


# -- recipe resolution ----------------------------------------------------


def test_recipe_resolves_by_dotted_reference() -> None:
    assert recipe_class_for("reef.train.cordis_backend.recipe:CordisRecipe") is CordisRecipe
    assert recipe_class_for("harness_evolve") is None


def test_yaml_config_boots_the_recipe_through_dotted_references(tmp_path: Path, monkeypatch) -> None:
    module = tmp_path / "demo_evolution.py"
    module.write_text(
        "def propose(nodes, samples, model):\n    return None\n\ndef evaluate(task, result):\n    return 0.0\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    built = CordisRecipe.from_environment(
        {},
        config={
            "evolution": {
                "propose": "demo_evolution:propose",
                "evaluate": "demo_evolution:evaluate",
                "tasks": ["task one"],
                "adapter": "pi",
            }
        },
        runtime=runtime(),
    )
    assert built.adapter == "pi"
    assert built.tasks == ("task one",)
    assert built.propose((), (), MODEL) is None


def test_config_without_evolution_section_is_rejected() -> None:
    with pytest.raises(RecipeConfigError, match="'evolution' config section"):
        CordisRecipe.from_environment({}, config={})


def test_build_returns_a_trainer_over_the_evolution_backend(tmp_path: Path) -> None:
    built = recipe(tmp_path, lambda nodes, samples, model: None)
    trainer = built.build("demo", RecordStore())
    assert isinstance(trainer, Trainer)
    assert trainer.report_type is ScoredRolloutReport


# -- mutation shape validation -------------------------------------------


def test_mutation_op_must_be_valid() -> None:
    with pytest.raises(MutationError, match="op must be one of"):
        Mutation("rename", "r1", RULES)


def test_mutation_id_must_be_root_level() -> None:
    with pytest.raises(MutationError, match="root-level id"):
        Mutation("remove", "group:child")


def test_create_requires_options() -> None:
    with pytest.raises(MutationError, match="requires options"):
        Mutation("create", "r1")


def test_remove_takes_no_options() -> None:
    with pytest.raises(MutationError, match="takes no options"):
        Mutation("remove", "r1", RULES)


# -- backend step behavior -----------------------------------------------


def test_winning_mutation_publishes_an_artifact(tmp_path: Path) -> None:
    def propose(nodes, samples, model):
        del nodes, samples
        return Mutation("create", "r1", {"name": "rules", "config": {"text": "marker rules"}})

    b = backend(tmp_path, propose)
    result = run_backend_step(b, batch(), b.initial_state())
    assert result.metrics["published"] is True
    assert result.metrics["selected"] is True
    assert result.metrics["selection"]["outcome"] == "select"
    assert result.metrics["selection"]["policy"] == "score_comparison"
    assert result.metrics["selection"]["candidate_id"] == "demo:trace:1:candidate"
    assert result.metrics["selection"]["metrics"] == {"wins": 1, "losses": 0, "ties": 0}
    assert "wins" not in result.metrics["selection"]["evaluation"]["metrics"]
    assert result.metrics["wins"] == 1
    assert isinstance(result.publication, SavedArtifactPublication)
    assert result.artifact is not None
    assert result.artifact.local_path is not None
    state = result.state
    assert state["steps"] == 1
    assert len(state["entries"]) == 1


def test_losing_mutation_reverts_and_publishes_nothing(tmp_path: Path) -> None:
    def propose(nodes, samples, model):
        del nodes, samples
        return Mutation("create", "r1", {"name": "rules", "config": {"text": "no help"}})

    b = backend(tmp_path, propose)
    result = run_backend_step(b, batch(), b.initial_state())
    assert result.metrics["published"] is False
    assert result.metrics["selected"] is False
    assert result.metrics["selection"]["outcome"] == "reject"
    assert isinstance(result.publication, NoArtifactPublication)
    assert result.state["entries"] == []


def test_rejected_proposal_is_skipped_with_no_artifact(tmp_path: Path) -> None:
    def propose(nodes, samples, model):
        del nodes, samples
        return Mutation("create", "bad", {"name": "rules", "config": {}})

    b = backend(tmp_path, propose)
    result = run_backend_step(b, batch(), b.initial_state())
    assert "rejected" in result.metrics["skipped"]
    assert isinstance(result.publication, NoArtifactPublication)
    assert result.state["entries"] == []


def test_no_proposal_skips_the_step(tmp_path: Path) -> None:
    b = backend(tmp_path, lambda nodes, samples, model: None)
    result = run_backend_step(b, batch(), b.initial_state())
    assert result.metrics["skipped"] == "no proposal"


def test_empty_sequence_is_no_proposal(tmp_path: Path) -> None:
    b = backend(tmp_path, lambda nodes, samples, model: ())
    result = run_backend_step(b, batch(), b.initial_state())
    assert result.metrics["skipped"] == "no proposal"


def test_composite_proposal_settles_under_one_selection_decision(tmp_path: Path) -> None:
    """A sequence is one proposal: both mutations publish together under a
    single verdict, and the ledger lists the whole set."""

    def propose(nodes, samples, model):
        del nodes, samples
        return [
            Mutation("create", "r1", {"name": "rules", "config": {"text": "marker rules"}}),
            Mutation("create", "r2", {"name": "rules", "config": {"text": "more rules"}}),
        ]

    b = backend(tmp_path, propose)
    result = run_backend_step(b, batch(), b.initial_state())
    assert result.metrics["published"] is True
    assert result.metrics["mutations"] == [{"op": "create", "id": "r1"}, {"op": "create", "id": "r2"}]
    assert "mutation" not in result.metrics
    assert isinstance(result.publication, SavedArtifactPublication)
    assert [entry["id"] for entry in result.state["entries"]] == ["r1", "r2"]


def test_composite_proposal_reverts_atomically_when_rejected(tmp_path: Path) -> None:
    def propose(nodes, samples, model):
        del nodes, samples
        return [
            Mutation("create", "r1", {"name": "rules", "config": {"text": "no help"}}),
            Mutation("create", "r2", {"name": "rules", "config": {"text": "still no help"}}),
        ]

    b = backend(tmp_path, propose)
    result = run_backend_step(b, batch(), b.initial_state())
    assert result.metrics["published"] is False
    assert isinstance(result.publication, NoArtifactPublication)
    assert result.state["entries"] == []


def test_composite_proposal_with_a_failing_step_reverts_the_applied_prefix(tmp_path: Path) -> None:
    """One snapshot covers the whole sequence: a failure in the second
    mutation also unwinds the first, and the step is a skip."""

    def propose(nodes, samples, model):
        del nodes, samples
        return [
            Mutation("create", "r1", {"name": "rules", "config": {"text": "marker rules"}}),
            Mutation("update", "missing", {"config": {"text": "x"}}),
        ]

    b = backend(tmp_path, propose)
    result = run_backend_step(b, batch(), b.initial_state())
    assert "skipped" in result.metrics
    assert result.state["entries"] == []
    # The tree itself is restored too, not just the recorded state.
    assert b._nodes() == ()


def test_single_mutation_metrics_are_unchanged_by_the_composite_seam(tmp_path: Path) -> None:
    def propose(nodes, samples, model):
        del nodes, samples
        return Mutation("create", "r1", {"name": "rules", "config": {"text": "marker rules"}})

    b = backend(tmp_path, propose)
    result = run_backend_step(b, batch(), b.initial_state())
    assert result.metrics["mutation"] == {"op": "create", "id": "r1"}
    assert "mutations" not in result.metrics


def test_composition_state_recovers_from_algorithm_state(tmp_path: Path) -> None:
    """A second backend resumes the composition from the first backend's
    algorithm state — the entries list carries the tree through recovery."""

    def propose(nodes, samples, model):
        del nodes, samples
        return Mutation("create", "r1", {"name": "rules", "config": {"text": "marker rules"}})

    first = backend(tmp_path, propose)
    result = run_backend_step(first, batch(), first.initial_state())

    second = backend(tmp_path, lambda nodes, samples, model: None)
    run_backend_step(second, batch(), result.state)
    nodes = second._nodes()
    assert nodes == (("rules", {"text": "marker rules"}),)


def test_episode_scorer_failure_reverts_before_it_propagates(tmp_path: Path) -> None:
    proposals = iter([Mutation("create", "r1", {"name": "rules", "config": {"text": "hi"}})])

    def propose(nodes, samples, model):
        del nodes, samples
        return next(proposals, None)

    def broken_score(task, result):
        raise RuntimeError("episode scorer bug")

    b = CordisBackend(
        descriptor=get_adapter("pi"),
        propose=resolve_proposer(propose),
        score_episode=resolve_episode_scorer(broken_score),
        tasks=("task one",),
        models=MODEL,
        binary=str(make_binary(tmp_path)),
    )
    with pytest.raises(RuntimeError, match="episode scorer bug"):
        run_backend_step(b, batch(), b.initial_state())
    assert b._nodes() == ()
    follow_up = run_backend_step(b, batch(), b.initial_state())
    assert follow_up.metrics["skipped"] == "no proposal"


# -- tree mutation mechanics ----------------------------------------------


def test_create_duplicate_id_is_rejected(tmp_path: Path) -> None:
    b = backend(tmp_path, lambda n, s, m: None)
    run_backend_step(b, batch(), b.initial_state())
    b._apply(Mutation("create", "r1", RULES))
    with pytest.raises(MutationError, match="already exists"):
        b._apply(Mutation("create", "r1", SKILL))


def test_update_missing_id_is_rejected(tmp_path: Path) -> None:
    b = backend(tmp_path, lambda n, s, m: None)
    run_backend_step(b, batch(), b.initial_state())
    with pytest.raises(MutationError, match="cannot resolve"):
        b._apply(Mutation("update", "x", RULES))


def test_update_merges_and_disabled_hides(tmp_path: Path) -> None:
    b = backend(tmp_path, lambda n, s, m: None)
    run_backend_step(b, batch(), b.initial_state())
    b._apply(Mutation("create", "r1", RULES))
    b._apply(Mutation("update", "r1", {"config": {"text": "Be verbose."}}))
    assert b._nodes() == (("rules", {"text": "Be verbose."}),)
    b._apply(Mutation("update", "r1", {"disabled": True}))
    assert b._nodes() == ()


def test_remove_deletes_entry(tmp_path: Path) -> None:
    b = backend(tmp_path, lambda n, s, m: None)
    run_backend_step(b, batch(), b.initial_state())
    b._apply(Mutation("create", "r1", RULES))
    b._apply(Mutation("create", "s1", SKILL))
    b._apply(Mutation("remove", "r1"))
    assert b._nodes() == (("skill", SKILL["config"]),)


def test_entries_and_load_round_trip(tmp_path: Path) -> None:
    b = backend(tmp_path, lambda n, s, m: None)
    run_backend_step(b, batch(), b.initial_state())
    b._apply(Mutation("create", "r1", RULES))
    b._apply(Mutation("create", "s1", SKILL))
    serialized = b._entries()

    fresh = backend(tmp_path, lambda n, s, m: None)
    run_backend_step(fresh, batch(), fresh.initial_state())
    fresh._loader.root.update(serialized)
    assert fresh._nodes() == b._nodes()


# -- revert exactness and ledger hygiene ----------------------------------


def seeded_state() -> dict:
    return {"steps": 1, "entries": [{"id": "r1", "name": "rules", "config": {"text": "Answer briefly."}}]}


def test_losing_update_revert_deletes_introduced_keys(tmp_path: Path) -> None:
    # The update changes the text and introduces a key; the revert must
    # restore the old text and delete the key, never leave disabled=False.
    upd = Mutation("update", "r1", {"config": {"text": "Be verbose."}, "disabled": True})
    b = backend(tmp_path, lambda nodes, samples, model: upd)
    result = run_backend_step(b, batch(), seeded_state())
    assert result.metrics["published"] is False
    assert result.state["entries"] == seeded_state()["entries"]


def test_losing_remove_revert_restores_the_entry_at_its_position(tmp_path: Path) -> None:
    b = backend(tmp_path, lambda nodes, samples, model: Mutation("remove", "r1"))
    state = seeded_state()
    state["entries"].append({"id": "s1", "name": "skill", "config": {"name": "notes", "text": "# notes"}})
    result = run_backend_step(b, batch(), state)
    assert result.metrics["published"] is False
    assert [entry["id"] for entry in result.state["entries"]] == ["r1", "s1"]


def test_failed_episodes_are_counted_never_scored(tmp_path: Path) -> None:
    # An unlaunchable episode loses its pairing but must not put -inf into
    # the metrics: the commit log serializes them as JSON, which has no
    # -Infinity, and a non-JSON reader would decode a wrong finite number.
    import json

    b = CordisBackend(
        descriptor=get_adapter("pi"),
        propose=resolve_proposer(lambda nodes, samples, model: Mutation("create", "r1", RULES)),
        score_episode=resolve_episode_scorer(evaluate),
        tasks=("task one",),
        models=MODEL,
        binary=str(tmp_path / "no-such-binary"),
    )
    result = run_backend_step(b, batch(), b.initial_state())
    assert result.metrics["episode_failures"] == 2  # both sides, one task
    assert result.metrics["published"] is False  # both failed: a tie, reverted
    json.loads(json.dumps(result.metrics, allow_nan=False))  # ledger-legal


def test_non_finite_episode_score_raises_and_reverts(tmp_path: Path) -> None:
    b = CordisBackend(
        descriptor=get_adapter("pi"),
        propose=resolve_proposer(lambda nodes, samples, model: Mutation("create", "r1", RULES)),
        score_episode=resolve_episode_scorer(lambda task, result: float("nan")),
        tasks=("task one",),
        models=MODEL,
        binary=str(make_binary(tmp_path)),
    )
    with pytest.raises(ValueError, match="episode scorer returned a non-finite score"):
        run_backend_step(b, batch(), b.initial_state())
    assert b._nodes() == ()  # the snapshot came back before the raise


# -- seed composition ------------------------------------------------------


def test_seed_boots_the_composition_tree(tmp_path: Path) -> None:
    """A seeded recipe builds a backend whose first step proposes over the
    seed nodes - the baseline the first mutation is measured against."""
    seen = []

    def propose(nodes, samples, model):
        del samples
        seen.append(nodes)
        return

    trainer = recipe(tmp_path, propose, seed=(SEED_MODELS, SEED_SETTINGS)).build("demo", RecordStore())
    b = trainer.training_backend
    assert isinstance(b, CordisBackend)
    state = b.initial_state()
    assert state == {"steps": 0, "entries": [SEED_MODELS, SEED_SETTINGS]}
    run_backend_step(b, batch(), state)
    assert seen == [(("config", SEED_MODELS["config"]), ("config", SEED_SETTINGS["config"]))]


def test_recovered_state_wins_over_the_seed(tmp_path: Path) -> None:
    seen = []

    def propose(nodes, samples, model):
        del samples
        seen.append(nodes)
        return

    b = backend(tmp_path, propose, seed=(SEED_MODELS, SEED_SETTINGS))
    run_backend_step(b, batch(), seeded_state())
    assert seen == [(("rules", {"text": "Answer briefly."}),)]


def test_invalid_seed_refuses_boot_naming_the_entry(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"seed entry 'r1' rejected: .*'text'"):
        backend(tmp_path, lambda n, s, m: None, seed=({"id": "r1", "name": "rules", "config": {}},))
    with pytest.raises(ValueError, match="non-empty string 'id'"):
        backend(tmp_path, lambda n, s, m: None, seed=({"name": "rules", "config": {"text": "hi"}},))


def test_seed_with_an_inline_key_refuses_boot(tmp_path: Path) -> None:
    """Issue #476: tree state persists into every commit record, the snapshot
    metadata, and the published artifact, so a credential-bearing config node
    never passes admission. The refusal names the sanctioned channels."""
    keyed = {
        "id": "models",
        "name": "config",
        "config": {
            "target": "models",
            "data": {"providers": {"qwen": {"apiKey": "sk-476-inline", "baseUrl": "http://localhost:8000/v1"}}},
        },
    }
    with pytest.raises(ValueError, match=r"seed entry 'models' rejected: .*reef\.upstream_api_key"):
        backend(tmp_path, lambda n, s, m: None, seed=(keyed,))


def _report_once(scenario, name: str, suffix: str) -> None:
    """One scored sample through the record store: enough to wake one step."""
    scenario.records.append_result(
        AgentRecord.create(
            scenario=name,
            request_type=RequestType.INFERENCE,
            payload={"messages": [{"role": "user", "content": "q"}]},
            agent_record_id=f"i{suffix}",
        )
    )
    scenario.records.append_result(
        AgentRecord.create(
            scenario=name,
            request_type=RequestType.REPORT,
            payload={"score": 0.0, "references": [f"i{suffix}"]},
            agent_record_id=f"r{suffix}",
        )
    )


def test_keyed_proposal_leaves_no_key_material_in_commit_records(tmp_path: Path) -> None:
    """The decisive #476 run against the real commit protocol: a proposal
    smuggling an inline key lands as a rejected mutation, the stored commit
    record bytes carry no key material, and a process restart recovers the
    committed state and keeps stepping."""
    key = "sk-476-DECISIVE-KEY-0123456789abcdef"
    leak = {
        "name": "config",
        "config": {"target": "models", "data": {"providers": {"leak": {"apiKey": key}}}},
    }
    proposals = iter((Mutation("create", "leak", leak),))
    built = recipe(tmp_path, lambda nodes, samples, model: next(proposals, None), seed=(SEED_MODELS, SEED_SETTINGS))
    initial = tmp_path / "initial"
    initial.mkdir()
    factory = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    agent_record_dir = tmp_path / "agent-record"

    dispatcher = Dispatcher(RecipeRegistry({built.name: built}), factory, agent_record_dir=agent_record_dir)
    try:
        scenario = dispatcher.get_or_create_scenario("leak-476", built.name)
        assert scenario is not None
        _report_once(scenario, "leak-476", "1")
        result = scenario.prepare_training_step()
        assert result is not None
        scenario.commit(result)
    finally:
        dispatcher.close()

    raw = (agent_record_dir / f"{hashlib.sha256(b'leak-476').hexdigest()}.commits.jsonl").read_bytes()
    assert key.encode() not in raw
    record = json.loads(raw.splitlines()[0])
    assert "inline credential" in record["metrics"]["skipped"]
    assert record["algorithm_state"]["entries"] == [SEED_MODELS, SEED_SETTINGS]

    restarted = Dispatcher(RecipeRegistry({built.name: built}), factory, agent_record_dir=agent_record_dir)
    try:
        recovered = restarted.get_or_create_scenario("leak-476", built.name)
        assert recovered is not None
        assert recovered.trainer.state["entries"] == [SEED_MODELS, SEED_SETTINGS]
        _report_once(recovered, "leak-476", "2")
        follow_up = recovered.prepare_training_step()
        assert follow_up is not None
        assert follow_up.metrics["skipped"] == "no proposal"
        recovered.commit(follow_up)
    finally:
        restarted.close()


def test_disabled_seed_with_an_inline_key_refuses_boot(tmp_path: Path) -> None:
    """Issue #476 follow-up: a disabled entry builds no fiber, so admission
    validates its config directly. Disabled is a serving state, never a
    validation bypass, because state persists disabled entries verbatim."""
    keyed = {
        "id": "models",
        "name": "config",
        "disabled": True,
        "config": {"target": "models", "data": {"providers": {"qwen": {"apiKey": "sk-476-disabled-seed"}}}},
    }
    with pytest.raises(ValueError, match=r"seed entry 'models' rejected: .*inline credential"):
        backend(tmp_path, lambda n, s, m: None, seed=(keyed,))


def test_admission_refuses_plural_and_list_valued_key_fields(tmp_path: Path) -> None:
    """The tripwire matches plural credential names and list values too."""
    for data in ({"apiKeys": ["sk-476-list"]}, {"providers": {"a": {"tokens": ["sk-476-plural"]}}}):
        keyed = {"id": "models", "name": "config", "config": {"target": "models", "data": data}}
        with pytest.raises(ValueError, match="inline credential"):
            backend(tmp_path, lambda n, s, m: None, seed=(keyed,))


def test_disabled_keyed_proposal_leaves_no_key_material_in_commit_records(tmp_path: Path) -> None:
    """Verify probe for #476: a proposal smuggling an inline key under
    ``disabled: True`` never builds a fiber, so the gate must not depend on
    one. The mutation lands rejected and the stored commit record bytes
    carry no key material."""
    key = "sk-476-DISABLED-KEY-0123456789abcdef"
    leak = {
        "name": "config",
        "disabled": True,
        "config": {"target": "models", "data": {"providers": {"leak": {"apiKey": key}}}},
    }
    proposals = iter((Mutation("create", "leak", leak),))
    built = recipe(tmp_path, lambda nodes, samples, model: next(proposals, None), seed=(SEED_MODELS, SEED_SETTINGS))
    initial = tmp_path / "initial"
    initial.mkdir()
    factory = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    agent_record_dir = tmp_path / "agent-record"

    dispatcher = Dispatcher(RecipeRegistry({built.name: built}), factory, agent_record_dir=agent_record_dir)
    try:
        scenario = dispatcher.get_or_create_scenario("leak-476-disabled", built.name)
        assert scenario is not None
        _report_once(scenario, "leak-476-disabled", "1")
        result = scenario.prepare_training_step()
        assert result is not None
        scenario.commit(result)
    finally:
        dispatcher.close()

    raw = (agent_record_dir / f"{hashlib.sha256(b'leak-476-disabled').hexdigest()}.commits.jsonl").read_bytes()
    assert key.encode() not in raw
    record = json.loads(raw.splitlines()[0])
    assert "inline credential" in record["metrics"]["skipped"]
    assert record["algorithm_state"]["entries"] == [SEED_MODELS, SEED_SETTINGS]


def test_pregate_recovered_state_refuses_the_step_and_writes_nothing(tmp_path: Path) -> None:
    """Verify probe for #476: a workdir committed before the gate holds an
    inline key in its recorded state. The restarted scenario refuses the
    next step naming the field and the rotation duty, and no newly written
    file carries the key bytes."""
    key = "sk-476-PREGATE-KEY-fedcba9876543210"
    built = recipe(tmp_path, lambda nodes, samples, model: None, seed=(SEED_MODELS, SEED_SETTINGS))
    initial = tmp_path / "initial"
    initial.mkdir()
    factory = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    agent_record_dir = tmp_path / "agent-record"

    dispatcher = Dispatcher(RecipeRegistry({built.name: built}), factory, agent_record_dir=agent_record_dir)
    try:
        scenario = dispatcher.get_or_create_scenario("pregate-476", built.name)
        assert scenario is not None
        _report_once(scenario, "pregate-476", "1")
        result = scenario.prepare_training_step()
        assert result is not None
        scenario.commit(result)
    finally:
        dispatcher.close()

    # Rewrite the committed record as a pre-gate workdir holds it: the seed
    # entry names the key inline. This file is the only one with key bytes.
    log_path = agent_record_dir / f"{hashlib.sha256(b'pregate-476').hexdigest()}.commits.jsonl"
    record = json.loads(log_path.read_bytes().splitlines()[-1])
    record["algorithm_state"]["entries"][0]["config"]["data"]["providers"]["qwen"]["apiKey"] = key
    log_path.write_bytes(json.dumps(record, separators=(",", ":"), sort_keys=True).encode() + b"\n")
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    restarted = Dispatcher(RecipeRegistry({built.name: built}), factory, agent_record_dir=agent_record_dir)
    try:
        recovered = restarted.get_or_create_scenario("pregate-476", built.name)
        assert recovered is not None
        _report_once(recovered, "pregate-476", "2")
        with pytest.raises(ValueError, match=r"apiKey.*inline credential.*rotate"):
            recovered.prepare_training_step()
    finally:
        restarted.close()

    for path in tmp_path.rglob("*"):
        if path.is_file() and before.get(path) != path.read_bytes():
            assert key.encode() not in path.read_bytes(), path


def test_yaml_seed_must_be_a_list_of_mappings(tmp_path: Path, monkeypatch) -> None:
    module = tmp_path / "demo_evolution.py"
    module.write_text(
        "def propose(nodes, samples, model):\n    return None\n\ndef evaluate(task, result):\n    return 0.0\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    def config(seed):
        return {
            "evolution": {
                "propose": "demo_evolution:propose",
                "evaluate": "demo_evolution:evaluate",
                "tasks": ["task one"],
                "seed": seed,
            }
        }

    built = CordisRecipe.from_environment({}, config=config([SEED_MODELS]))
    assert built.seed == (SEED_MODELS,)
    with pytest.raises(RecipeConfigError, match=r"evolution\.seed must be a list"):
        CordisRecipe.from_environment({}, config=config("models"))
    with pytest.raises(RecipeConfigError, match=r"evolution\.seed entries must be"):
        CordisRecipe.from_environment({}, config=config(["models"]))


def test_recipe_rejects_removed_acceptance_and_raw_selection_callable(tmp_path, monkeypatch) -> None:
    module = tmp_path / "demo_method.py"
    module.write_text(
        "def propose(nodes, samples, model):\n    return None\n\n"
        "def evaluate(task, result):\n    return 0.0\n\n"
        "def accept(candidate, current):\n    return True\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    def config(**evolution):
        return {
            "evolution": {
                "propose": "demo_method:propose",
                "evaluate": "demo_method:evaluate",
                "tasks": ["task one"],
                **evolution,
            }
        }

    with pytest.raises(RecipeConfigError, match=r"evolution\.acceptance was removed"):
        CordisRecipe.from_environment({}, config=config(acceptance="always"))
    with pytest.raises(RecipeConfigError, match=r"must provide decide"):
        CordisRecipe.from_environment({}, config=config(selection="demo_method:accept"))


def test_recipe_resolves_candidate_selector(tmp_path, monkeypatch) -> None:
    module = tmp_path / "demo_selection.py"
    module.write_text(
        "def propose(nodes, samples, model):\n    return None\n\n"
        "def evaluate(task, result):\n    return 0.0\n\n"
        "class Policy:\n"
        "    def decide(self, candidate, evaluation):\n"
        "        from reef.train.evaluation import SelectionDecision\n"
        "        return SelectionDecision('select', 'demo', '1', 'selected by demo', evaluation)\n\n"
        "policy = Policy()\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    def config(**evolution):
        return {
            "evolution": {
                "propose": "demo_selection:propose",
                "evaluate": "demo_selection:evaluate",
                "tasks": ["task one"],
                **evolution,
            }
        }

    compared = CordisRecipe.from_environment({}, config=config())
    assert compared.candidate_selector is not None
    assert type(compared.candidate_selector).__name__ == "ScoreComparisonSelector"

    named = CordisRecipe.from_environment({}, config=config(selection="always"))
    assert named.candidate_selector is not None
    assert type(named.candidate_selector).__name__ == "AlwaysSelect"

    dotted = CordisRecipe.from_environment({}, config=config(selection="demo_selection:policy"))
    assert dotted.candidate_selector is not None
    assert type(dotted.candidate_selector).__name__ == "Policy"

    with pytest.raises(RecipeConfigError, match="acceptance was removed"):
        CordisRecipe.from_environment(
            {},
            config=config(selection="always", acceptance="pairwise"),
        )


def test_model_binding_reaches_episodes_but_never_the_published_tree(tmp_path: Path, monkeypatch) -> None:
    """The deployment's endpoint renders into each evaluation episode through
    the adapter's model_binding template, and the published artifact carries
    no provider: a client points its own harness at Reef."""
    seen: list[dict[str, str]] = []
    original = reef_cordis_backend.run_episode

    def spy(descriptor, files, prompt, **kwargs):
        seen.append(dict(files))
        return original(descriptor, files, prompt, **kwargs)

    monkeypatch.setattr(reef_cordis_backend, "run_episode", spy)
    b = backend(tmp_path, lambda n, s, m: Mutation("create", "r1", {"name": "rules", "config": {"text": "marker"}}))
    result = run_backend_step(b, batch(), b.initial_state())

    assert seen, "evaluation ran episodes"
    models = json.loads(seen[0]["pi-agent/models.json"])
    assert models["providers"]["reef"]["baseUrl"] == "http://localhost:8000/v1"
    assert models["providers"]["reef"]["apiKey"] == "dummy"
    assert json.loads(seen[0]["pi-agent/settings.json"])["defaultModel"] == "reef/qwen3-8b"

    assert result.artifact is not None
    published = Path(result.artifact.local_path)
    assert json.loads((published / "pi-agent/models.json").read_text()) == {}
    assert "dummy" not in (published / "pi-agent/settings.json").read_text()


def test_propose_receives_the_model_binding(tmp_path: Path) -> None:
    received: list[ModelBindings] = []

    def propose(nodes, samples, model):
        received.append(model)

    b = backend(tmp_path, propose)
    run_backend_step(b, batch(), b.initial_state())
    assert [models.served for models in received] == [MODEL]
    assert list(received[0]) == ["served"]


def test_recipe_without_a_runtime_refuses_to_build(tmp_path: Path) -> None:
    built = CordisRecipe(
        resolve_proposer(lambda n, s, m: None),
        resolve_episode_scorer(evaluate),
        ("task one",),
        binary=str(make_binary(tmp_path)),
    )
    with pytest.raises(RecipeConfigError, match=r"reef\.upstream_url"):
        built.build("demo", RecordStore())


def test_adapter_without_model_binding_refuses_boot(tmp_path: Path) -> None:
    descriptor = dataclasses.replace(get_adapter("pi"), model_binding={})
    with pytest.raises(ModelBindingError, match="declares no model_binding"):
        CordisBackend(
            descriptor=descriptor,
            propose=resolve_proposer(lambda n, s, m: None),
            score_episode=resolve_episode_scorer(evaluate),
            tasks=("task one",),
            models=MODEL,
            binary=str(make_binary(tmp_path)),
        )


def test_gate_knobs_reach_the_episodes_and_interleave_the_sides(tmp_path: Path, monkeypatch) -> None:
    """episode_timeout_s reaches every run_episode call, a repeat is one more
    pairing of the same task, and episodes alternate candidate and current
    inside each pairing instead of running batched by side."""
    sides: list[str] = []
    timeouts: list[float] = []
    original = reef_cordis_backend.run_episode

    def spy(descriptor, files, prompt, **kwargs):
        timeouts.append(kwargs["timeout"])
        sides.append("candidate" if "marker" in files.get("pi-agent/AGENTS.md", "") else "current")
        return original(descriptor, files, prompt, **kwargs)

    monkeypatch.setattr(reef_cordis_backend, "run_episode", spy)
    b = CordisBackend(
        descriptor=get_adapter("pi"),
        propose=resolve_proposer(
            lambda n, s, m: Mutation("create", "r1", {"name": "rules", "config": {"text": "marker"}})
        ),
        score_episode=resolve_episode_scorer(evaluate),
        tasks=("task one",),
        models=MODEL,
        binary=str(make_binary(tmp_path)),
        episode_timeout_s=5.0,
        episode_repeats=2,
    )
    result = run_backend_step(b, batch(), b.initial_state())

    assert timeouts == [5.0, 5.0, 5.0, 5.0]
    assert sides == ["candidate", "current", "candidate", "current"]
    assert result.metrics["episode_repeats"] == 2
    assert len(result.metrics["selection"]["evaluation"]["metrics"]["candidate_scores"]) == 2


def test_littering_episodes_are_counted_and_forbid_residue_fails_them(tmp_path: Path) -> None:
    """Files an episode leaves outside the cleanup whitelist are counted into
    the evaluation; with forbid_residue a littering episode scores as one
    that could not run, so it cannot win."""
    litterer = tmp_path / "fake-pi-littering"
    litterer.write_text(
        PI_FAKE.replace(
            '(session_dir / "session.jsonl").write_text(json.dumps(event) + "\\n")',
            '(session_dir / "session.jsonl").write_text(json.dumps(event) + "\\n")\n'
            '(agent_dir.parent / "junk.txt").write_text("litter")',
        )
    )
    litterer.chmod(0o755)

    def build(forbid: bool) -> CordisBackend:
        return CordisBackend(
            descriptor=get_adapter("pi"),
            propose=resolve_proposer(
                lambda n, s, m: Mutation("create", "r1", {"name": "rules", "config": {"text": "marker"}})
            ),
            score_episode=resolve_episode_scorer(evaluate),
            tasks=("task one",),
            models=MODEL,
            binary=str(litterer),
            forbid_residue=forbid,
        )

    counted = run_backend_step(build(False), batch(), build(False).initial_state())
    assert counted.metrics["candidate_residue"] == 1
    assert counted.metrics["current_residue"] == 1
    assert counted.metrics["episode_failures"] == 0

    forbidden = run_backend_step(build(True), batch(), build(True).initial_state())
    assert forbidden.metrics["episode_failures"] == 2
    assert forbidden.metrics["selected"] is False
    evaluation = forbidden.metrics["selection"]["evaluation"]["metrics"]
    assert {f["stage"] for f in evaluation["candidate_failures"]} == {"residue"}
    assert {f["stage"] for f in evaluation["current_failures"]} == {"residue"}


def test_recipe_parses_and_validates_the_episode_gate_knobs(tmp_path: Path, monkeypatch) -> None:
    module = tmp_path / "demo_gate_knobs.py"
    module.write_text(
        "def propose(nodes, samples, model):\n    return None\n\ndef evaluate(task, result):\n    return 0.0\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    def config(**evolution):
        return {
            "evolution": {
                "propose": "demo_gate_knobs:propose",
                "evaluate": "demo_gate_knobs:evaluate",
                "tasks": ["task one"],
                **evolution,
            }
        }

    built = CordisRecipe.from_environment(
        {}, config=config(episode_timeout_s=5, episode_repeats=2, forbid_residue=True)
    )
    assert built.episode_timeout_s == 5.0
    assert built.episode_repeats == 2
    assert built.forbid_residue is True

    defaults = CordisRecipe.from_environment({}, config=config())
    assert defaults.episode_timeout_s == 600.0
    assert defaults.episode_repeats == 1
    assert defaults.forbid_residue is False

    with pytest.raises(RecipeConfigError, match=r"episode_timeout_s must be a positive number"):
        CordisRecipe.from_environment({}, config=config(episode_timeout_s=0))
    with pytest.raises(RecipeConfigError, match=r"episode_repeats must be an integer"):
        CordisRecipe.from_environment({}, config=config(episode_repeats=0))
    with pytest.raises(RecipeConfigError, match=r"forbid_residue must be a boolean"):
        CordisRecipe.from_environment({}, config=config(forbid_residue="yes"))


def test_backend_rejects_invalid_gate_knobs(tmp_path: Path) -> None:
    def build(**kwargs) -> CordisBackend:
        return CordisBackend(
            descriptor=get_adapter("pi"),
            propose=resolve_proposer(lambda n, s, m: None),
            score_episode=resolve_episode_scorer(evaluate),
            tasks=("task one",),
            models=MODEL,
            binary=str(make_binary(tmp_path)),
            **kwargs,
        )

    with pytest.raises(ValueError, match="episode_timeout_s must be a positive number"):
        build(episode_timeout_s=0)
    with pytest.raises(ValueError, match="episode_repeats must be an integer"):
        build(episode_repeats=0)
    with pytest.raises(ValueError, match="forbid_residue must be a boolean"):
        build(forbid_residue="no")


def test_recipe_selects_the_record_driven_processor(tmp_path: Path, monkeypatch) -> None:
    """data.batch_policy records swaps the processor; the default stays the
    reported window."""
    module = tmp_path / "demo_batch_policy.py"
    module.write_text(
        "def propose(nodes, samples, model):\n    return None\n\ndef evaluate(task, result):\n    return 0.0\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    def config(policy: str | None):
        data = {} if policy is None else {"batch_policy": policy}
        return {
            "data": data,
            "evolution": {
                "propose": "demo_batch_policy:propose",
                "evaluate": "demo_batch_policy:evaluate",
                "tasks": ["task one"],
                "binary": str(make_binary(tmp_path)),
            },
        }

    records = CordisRecipe.from_environment(
        {}, config=config("records"), runtime=InferenceProxyRuntime(model_path="m", base_url="http://localhost:8000")
    )
    assert records.batch_policy == "records"
    trainer = records.build("demo", RecordStore())
    assert type(trainer.processor).__name__ == "RecordDrivenTraceProcessor"

    reported = CordisRecipe.from_environment(
        {}, config=config(None), runtime=InferenceProxyRuntime(model_path="m", base_url="http://localhost:8000")
    )
    assert reported.batch_policy == "reports"
    trainer = reported.build("demo", RecordStore())
    assert type(trainer.processor).__name__ == "CordisProcessor"

    with pytest.raises(RecipeConfigError, match="batch_policy must be 'reports' or 'records'"):
        CordisRecipe.from_environment({}, config=config("sometimes"))
