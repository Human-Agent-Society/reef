"""Guarantees of the recipes/gepa method package, hermetic: no reflection
model is ever reached, the proposer's episodes run through an injected fake
runner, and the one end-to-end step drives a fake pi binary through the real
adapter, backend, and commit path."""

from __future__ import annotations

import dataclasses
import json
import random
from pathlib import Path
from typing import Any

import pytest

from recipes.gepa import components, reflection
from recipes.gepa.archive import Archive, Candidate
from recipes.gepa.method import GEPAProposer, GEPASelector, default_feedback
from recipes.gepa.recipe import GEPARecipe
from reef.artifact import InMemoryRepositoryBackend
from reef.core import AgentRecord, RequestType
from reef.dispatcher import Dispatcher
from reef.harness.adapters import get_adapter
from reef.harness.episode import EpisodeError, EpisodeResult
from reef.harness.model_binding import ModelBinding, ModelBindings
from reef.recipe import RecipeConfigError
from reef.recipe.registry import RecipeRegistry, build_recipe
from reef.records import RecordStore
from reef.runtime.adapters.inference_proxy import InferenceProxyRuntime
from reef.train.cordis_backend.strategies import Mutation, resolve_episode_scorer
from reef.train.evaluation.contracts import EvaluationResult, UpdateCandidate
from reef.train.trainer import Trainer
from reef.train.types import TraceSample

# The fake harness scores itself: its trajectory carries the rules text, so
# the scorer can prefer any composition whose instructions carry the marker.
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

MARKER = "SHOW-YOUR-WORK"
SEED_TEXT = "Answer briefly."
REFLECTED = f"Answer briefly. {MARKER}"
FENCED_REPLY = f"Here you go:\n```\n{REFLECTED}\n```\n"

#: The served composition as the backend passes it: (kind, config) pairs with
#: entry ids stripped. The skill is here to prove ``kinds`` narrows what GEPA
#: is allowed to rewrite.
NODES = (
    ("rules", {"text": SEED_TEXT}),
    ("skill", {"name": "notes", "text": "# notes"}),
)
SEED = ({"id": "rules", "name": "rules", "config": {"text": SEED_TEXT}},)

TASK = "what is 2+2?"
SAMPLE = TraceSample(
    "a1",
    {
        "messages": [{"role": "user", "content": TASK}],
        "response": {"choices": [{"message": {"role": "assistant", "content": "4"}}]},
    },
    0.0,
)
#: The same exchange as the proxy records a streamed reply: the message it
#: assembled from the chunks sits beside the raw stream body, no ``choices``.
STREAMED_SAMPLE = TraceSample(
    "a2",
    {
        "messages": [{"role": "user", "content": TASK}],
        "response": {"stream": True, "body": "data: ...", "message": {"role": "assistant", "content": "4"}},
    },
    0.0,
)


def test_recorded_traffic_reads_buffered_and_streamed_replies() -> None:
    from recipes.gepa.method import _traffic

    assert [texts for _, texts in _traffic((SAMPLE, STREAMED_SAMPLE))] == [(TASK, "4"), (TASK, "4")]


def score_rules(task: str, result: EpisodeResult) -> float:
    """1.0 when the composition that ran carried the marker instruction."""
    last = result.trajectory[-1] if result.trajectory else {}
    return 1.0 if MARKER in str(last.get("content", last.get("rules", ""))) else 0.0


class FakeEpisodes:
    """An episode runner that answers with the rules text it was rendered."""

    def __init__(self, failure: Exception | None = None) -> None:
        self.prompts: list[str] = []
        self.rules: list[str] = []
        self.failure = failure

    def __call__(self, descriptor, files, prompt, *, binary=None, timeout=600.0):
        self.prompts.append(prompt)
        text = next((value for path, value in files.items() if path.endswith("AGENTS.md")), "")
        self.rules.append(text)
        if self.failure is not None:
            raise self.failure
        return EpisodeResult(
            exit_code=0,
            stdout="",
            stderr="",
            trajectory=({"role": "assistant", "content": text},),
            residue=(),
        )


class FakeChat:
    """A model binding stand-in that answers one canned reflection reply."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def chat(self, messages, **params):
        self.prompts.append(messages[0]["content"])
        return self.reply


class ServedModel(ModelBinding):
    """The model under test: a real binding whose ``chat`` is canned.

    Episodes render this binding into the tree, so it has to be a genuine
    ``ModelBinding``; the canned ``chat`` is what the reflection fallback
    reaches when no ``evolution.models.reflection`` is declared.
    """

    def __init__(self, reply: str = "") -> None:
        super().__init__(base_url="http://127.0.0.1:9", model="served-model", api_key="dummy")
        object.__setattr__(self, "reply", reply)
        object.__setattr__(self, "prompts", [])

    def chat(self, messages, **params):
        self.prompts.append(messages[0]["content"])
        return self.reply


def bindings(reply: str = FENCED_REPLY, *, declared: bool = True) -> tuple[ModelBindings, Any]:
    """The models a proposer receives, and the object its reflection lands on."""
    if declared:
        reflector = FakeChat(reply)
        return ModelBindings(served=ServedModel(), named={"reflection": reflector}), reflector
    served = ServedModel(reply)
    return ModelBindings(served=served), served


def runtime() -> InferenceProxyRuntime:
    return InferenceProxyRuntime(model_path="served-model", base_url="http://127.0.0.1:9", api_key="dummy")


def make_binary(tmp_path: Path) -> Path:
    binary = tmp_path / "fake-pi"
    binary.write_text(PI_FAKE)
    binary.chmod(0o755)
    return binary


def proposer(
    archive: Archive,
    episodes: FakeEpisodes,
    **overrides: Any,
) -> GEPAProposer:
    settings: dict[str, Any] = {
        "minibatch_size": 3,
        "rng_seed": 0,
        "skip_perfect_score": True,
        "perfect_score": 1.0,
        "max_metric_calls": None,
        "kinds": ("rules",),
        "valset_size": 2,
    }
    settings.update(overrides)
    return GEPAProposer(
        archive=archive,
        descriptor=get_adapter("pi"),
        binary=None,
        score_episode=resolve_episode_scorer(score_rules),
        feedback=default_feedback,
        episode_runner=episodes,
        **settings,
    )


# -- components: the composition as a GEPA candidate ------------------------


def test_texts_of_keys_nodes_by_kind_and_name() -> None:
    assert components.texts_of(NODES, ("rules", "skill")) == {"rules": SEED_TEXT, "skill:notes": "# notes"}
    assert components.texts_of(NODES, ("rules",)) == {"rules": SEED_TEXT}
    assert components.component_key("config", {"target": "primary"}) is None


def test_mutations_address_the_entry_the_component_composes_under() -> None:
    served = {"rules": SEED_TEXT, "skill:notes": "# notes"}
    child = {"rules": REFLECTED, "skill:notes": "# better", "skill:absent": "unreachable"}
    assert components.mutations(served, child) == (
        Mutation("update", "rules", {"name": "rules", "config": {"text": REFLECTED}}),
        Mutation("update", "notes", {"name": "skill", "config": {"name": "notes", "text": "# better"}}),
    )
    assert components.mutations(served, served) == ()


# -- reflection: the copied prompt and extractor ----------------------------


def test_the_prompt_carries_the_current_text_and_the_records() -> None:
    prompt = reflection.render_prompt(
        SEED_TEXT,
        [{"Inputs": TASK, "Generated Outputs": "4", "Feedback": "correct"}],
    )
    assert "<curr_param>" not in prompt and "<side_info>" not in prompt
    assert SEED_TEXT in prompt
    assert "# Example 1\n## Inputs\nwhat is 2+2?\n\n## Generated Outputs\n4\n\n## Feedback\ncorrect\n\n" in prompt
    assert prompt.endswith("Provide the new instructions within ``` blocks.")


def test_the_prompt_nests_structured_feedback_as_markdown_headers() -> None:
    prompt = reflection.render_prompt("x", [{"Feedback": {"why": {"detail": "off by one"}}}])
    assert "## Feedback\n### why\n#### detail\noff by one" in prompt


def test_a_template_missing_a_placeholder_is_refused() -> None:
    with pytest.raises(ValueError, match="<side_info>"):
        reflection.render_prompt("x", [], template="only <curr_param>")


def test_the_extractor_reads_fenced_and_unfenced_replies() -> None:
    assert reflection.extract_new_text(FENCED_REPLY) == REFLECTED
    assert reflection.extract_new_text(f"```text\n{REFLECTED}\n```") == REFLECTED
    assert reflection.extract_new_text(REFLECTED) == REFLECTED
    assert reflection.extract_new_text(f"```\n{REFLECTED}") == REFLECTED  # never closed
    assert reflection.extract_new_text(f"{REFLECTED}\n```") == REFLECTED  # never opened


# -- archive: fronts, parent sampling, resume -------------------------------


def archive_with(tmp_path: Path, *scores: list[float] | None) -> Archive:
    archive = Archive(tmp_path / "archive.json")
    for index, score in enumerate(scores):
        if index == 0:
            archive.seed({"rules": f"text {index}"})
        else:
            archive.add({"rules": f"text {index}"}, 0, [0.0])
        if score is not None:
            archive.record_validation(index, score)
    archive.clear_pending()
    return archive


def test_a_strict_win_replaces_a_front_and_an_equal_score_joins_it(tmp_path: Path) -> None:
    archive = archive_with(tmp_path, [0.0, 1.0], [1.0, 1.0], [0.0, 0.0])
    assert archive.fronts() == {0: {1}, 1: {0, 1}}
    assert archive.best() == 1
    assert archive.mean_val(2) == 0.0


def test_an_unvalidated_candidate_is_on_no_front(tmp_path: Path) -> None:
    archive = archive_with(tmp_path, [1.0, 1.0], None)
    assert archive.fronts() == {0: {0}, 1: {0}}


class RecordingRandom(random.Random):
    """Captures the frequency-weighted sampling list GEPA builds."""

    seen: list[int] = []

    def choice(self, seq):
        self.seen = list(seq)
        return seq[0]


def test_parent_sampling_weights_survivors_by_how_many_fronts_they_hold(tmp_path: Path) -> None:
    """Two specialists, neither dominated: the one holding two of the three
    instance fronts is drawn twice as often as the one holding one."""
    archive = archive_with(tmp_path, [1.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    rng = RecordingRandom()
    assert archive.select_parent(rng) == 0
    assert rng.seen == [0, 0, 1]


def test_parent_sampling_drops_candidates_another_candidate_covers(tmp_path: Path) -> None:
    """Candidate 2 is on every front that 0 and 1 are on, so only it survives."""
    archive = archive_with(tmp_path, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0])
    rng = RecordingRandom()
    assert archive.select_parent(rng) == 2
    assert rng.seen == [2, 2, 2]


def test_parent_sampling_falls_back_to_the_served_candidate(tmp_path: Path) -> None:
    archive = Archive(tmp_path / "archive.json")
    archive.seed({"rules": SEED_TEXT})
    assert archive.select_parent(random.Random(0)) == 0


def test_the_round_robin_cursor_walks_the_components_and_children_inherit_it(tmp_path: Path) -> None:
    archive = Archive(tmp_path / "archive.json")
    archive.seed({"rules": "r", "skill:notes": "n"})
    keys = ["rules", "skill:notes"]
    assert [archive.next_component(0, keys) for _ in range(3)] == ["rules", "skill:notes", "rules"]
    child = archive.add({"rules": "r2", "skill:notes": "n"}, 0, [1.0])
    assert archive.next_component(child, keys) == "skill:notes"


def test_the_archive_resumes_from_disk(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "demo.json"
    archive = Archive(path)
    archive.seed({"rules": SEED_TEXT})
    archive.charge(4)
    archive.add({"rules": REFLECTED}, 0, [1.0])
    archive.record_validation(1, [1.0, 0.0])
    archive.reject()

    resumed = Archive(path)
    assert resumed.to_dict() == archive.to_dict()
    assert (resumed.metric_calls, resumed.steps, resumed.rejections) == (4, 2, 1)
    assert resumed.candidates[1] == Candidate(
        texts={"rules": REFLECTED},
        parent=0,
        val_scores=[1.0, 0.0],
        minibatch_scores=[1.0],
        cursor=0,
        discovered_at=4,
    )


# -- proposer: minibatch, reflection, acceptance ----------------------------


def test_a_served_parent_uses_recorded_traffic_and_runs_no_parent_episodes(tmp_path: Path) -> None:
    """The served composition's minibatch is free: the mechanism already
    batched its real traffic, so only the child costs episodes."""
    archive = Archive(tmp_path / "archive.json")
    episodes = FakeEpisodes()
    models, reflector = bindings()

    proposal = proposer(archive, episodes)(NODES, (SAMPLE,), models)

    assert proposal == (Mutation("update", "rules", {"name": "rules", "config": {"text": REFLECTED}}),)
    assert episodes.prompts == [TASK]  # the child only
    # GEPA's count: the seed's validation (2 tasks), the served parent's
    # minibatch (1, from traffic), and the child's minibatch (1).
    assert archive.metric_calls == 4
    assert archive.pending == 1
    assert archive.candidates[1].parent == 0
    assert archive.candidates[1].minibatch_scores == [1.0]
    assert archive.candidates[0].texts == {"rules": SEED_TEXT}  # the skill stayed out of ``kinds``
    # The reflective record is the recorded exchange, graded by the hook.
    (prompt,) = reflector.prompts
    assert f"## Inputs\n{TASK}" in prompt
    assert "## Generated Outputs\n4" in prompt
    assert "## Feedback\nThis response scored 0.000." in prompt


def test_a_parent_that_never_served_is_re_run_on_the_minibatch(tmp_path: Path) -> None:
    """No traffic was ever served from an archive candidate, so its minibatch
    scores and its outputs have to be produced by episodes."""
    archive = Archive(tmp_path / "archive.json")
    archive.seed({"rules": SEED_TEXT})
    archive.add({"rules": "Parent rules."}, 0, [1.0])
    archive.record_validation(0, [0.0, 0.0])
    archive.record_validation(1, [1.0, 1.0])  # dominates, so it is always the parent
    archive.clear_pending()
    episodes = FakeEpisodes()
    models, reflector = bindings()

    proposal = proposer(archive, episodes)(NODES, (SAMPLE,), models)

    assert proposal == (Mutation("update", "rules", {"name": "rules", "config": {"text": REFLECTED}}),)
    assert episodes.prompts == [TASK, TASK]  # parent then child
    assert archive.metric_calls == 2
    assert archive.candidates[2].parent == 1
    (prompt,) = reflector.prompts
    assert "Parent rules." in prompt  # reflection reads the parent's own text and output


def test_a_child_that_does_not_beat_its_parent_is_rejected(tmp_path: Path) -> None:
    archive = Archive(tmp_path / "archive.json")
    episodes = FakeEpisodes()
    models, _ = bindings("```\nNo marker here.\n```")

    assert proposer(archive, episodes)(NODES, (SAMPLE,), models) is None
    assert (archive.rejections, archive.steps, archive.pending) == (1, 1, None)
    assert len(archive.candidates) == 1


def test_an_episode_that_cannot_run_scores_zero_and_reports_its_error(tmp_path: Path) -> None:
    archive = Archive(tmp_path / "archive.json")
    archive.seed({"rules": SEED_TEXT})
    archive.add({"rules": "Parent rules."}, 0, [1.0])
    archive.record_validation(0, [0.0])
    archive.record_validation(1, [1.0])
    archive.clear_pending()
    episodes = FakeEpisodes(failure=EpisodeError("harness binary 'pi' not found"))
    models, reflector = bindings()

    assert proposer(archive, episodes)(NODES, (SAMPLE,), models) is None  # child ties the parent at 0.0
    (prompt,) = reflector.prompts
    assert "## Feedback\nharness binary 'pi' not found" in prompt


def test_the_budget_short_circuits_before_any_model_call(tmp_path: Path) -> None:
    archive = Archive(tmp_path / "archive.json")
    archive.charge(150)
    episodes = FakeEpisodes()
    models, reflector = bindings()

    assert proposer(archive, episodes, max_metric_calls=150)(NODES, (SAMPLE,), models) is None
    assert (episodes.prompts, reflector.prompts) == ([], [])


def test_a_perfect_minibatch_short_circuits_before_any_model_call(tmp_path: Path) -> None:
    archive = Archive(tmp_path / "archive.json")
    episodes = FakeEpisodes()
    models, reflector = bindings()
    solved = TraceSample("a1", dict(SAMPLE.payload), 1.0)

    assert proposer(archive, episodes)(NODES, (solved,), models) is None
    assert (episodes.prompts, reflector.prompts) == ([], [])
    # The same minibatch is worth reflecting on once the bar is above it.
    proposer(archive, episodes, perfect_score=2.0)(NODES, (solved,), models)
    assert len(reflector.prompts) == 1


def test_reflection_falls_back_to_the_model_under_test(tmp_path: Path) -> None:
    archive = Archive(tmp_path / "archive.json")
    episodes = FakeEpisodes()
    models, served = bindings(declared=False)

    assert proposer(archive, episodes)(NODES, (SAMPLE,), models) is not None
    assert len(served.prompts) == 1


def test_a_tree_the_archive_does_not_recognise_is_reseeded(tmp_path: Path) -> None:
    """An operator edit or a rollback is a new root, never a descendant."""
    archive = Archive(tmp_path / "archive.json")
    archive.seed({"rules": "something else"})
    episodes = FakeEpisodes()
    models, _ = bindings()

    proposer(archive, episodes)(NODES, (SAMPLE,), models)

    assert archive.served == 1
    assert archive.candidates[1].texts == {"rules": SEED_TEXT}
    assert archive.candidates[1].parent is None


# -- selector: the validation pass and the Pareto update --------------------


def evaluation(candidate: tuple, current: tuple) -> EvaluationResult:
    return EvaluationResult(
        evaluator="harness_episode_pairs",
        evaluator_version="1",
        metrics={"candidate_scores": candidate, "current_scores": current},
    )


def pending_archive(tmp_path: Path) -> Archive:
    archive = Archive(tmp_path / "archive.json")
    archive.seed({"rules": SEED_TEXT})
    archive.add({"rules": REFLECTED}, 0, [1.0])
    return archive


def test_the_first_decision_records_both_sides_and_charges_both(tmp_path: Path) -> None:
    archive = pending_archive(tmp_path)
    decision = GEPASelector(archive).decide(UpdateCandidate("c1"), evaluation((1.0, None), (0.0, 0.0)))

    assert decision.selected
    assert (decision.policy, decision.policy_version) == ("gepa", "1")
    assert archive.candidates[1].val_scores == [1.0, 0.0]  # a failed episode is a zero, not a gap
    assert archive.candidates[0].val_scores == [0.0, 0.0]  # the seed's own first validation
    # The candidate's validation (2); the seed's was charged when it was seeded.
    assert (archive.served, archive.pending, archive.metric_calls) == (1, None, 2)
    assert decision.metrics == {
        "candidate_val_mean": 0.5,
        "served_val_mean": 0.0,
        "parent": 0,
        "archive_size": 2,
        "front_size": 2,  # the seed still holds the task both compositions failed
        "metric_calls": 4,
    }


def test_a_candidate_that_does_not_beat_the_served_mean_is_rejected(tmp_path: Path) -> None:
    archive = pending_archive(tmp_path)
    decision = GEPASelector(archive).decide(UpdateCandidate("c1"), evaluation((1.0, 0.0), (0.0, 1.0)))

    assert not decision.selected  # equal means are not an improvement
    assert (archive.served, archive.pending) == (0, None)
    assert archive.candidates[1].val_scores == [1.0, 0.0]  # still on a front, still a parent
    assert archive.fronts() == {0: {1}, 1: {0}}


def test_the_seed_is_validated_once(tmp_path: Path) -> None:
    archive = pending_archive(tmp_path)
    selector = GEPASelector(archive)
    selector.decide(UpdateCandidate("c1"), evaluation((0.0, 0.0), (1.0, 1.0)))
    archive.add({"rules": "third"}, 0, [1.0])
    selector.decide(UpdateCandidate("c2"), evaluation((0.0, 0.0), (1.0, 1.0)))

    assert archive.metric_calls == 4  # 2 per validated candidate; the seed's own pass was charged at seeding
    assert archive.candidates[0].val_scores == [1.0, 1.0]


# -- recipe: config boot and the per-scenario binding -----------------------


def feedback_hook(task: str, output: str, score: float) -> str:
    return f"{task} answered {output} at {score}"


def sections(tmp_path: Path, **gepa: Any) -> dict[str, Any]:
    block = {
        "archive": str(tmp_path / "gepa"),
        "minibatch_size": 2,
        "seed": 7,
        "skip_perfect_score": False,
        "perfect_score": 0.5,
        "max_metric_calls": 150,
        "components": ["rules"],
    }
    block.update(gepa)
    return {
        "implementation": "recipes.gepa.recipe:GEPARecipe",
        "model": {"path": "served-model"},
        "evolution": {
            "adapter": "pi",
            "binary": str(tmp_path / "fake-pi"),
            "evaluate": score_rules,
            "feedback": feedback_hook,
            "tasks": ["check one", "check two"],
            "seed": list(SEED),
            "gepa": block,
        },
        "data": {"batch_size": 1},
    }


def test_the_config_boots_the_recipe_and_binds_both_seams(tmp_path: Path) -> None:
    config = sections(tmp_path)
    built = build_recipe(str(config["implementation"]), {}, config=config, runtime=runtime())

    assert isinstance(built, GEPARecipe)
    assert built.name == "gepa"
    assert built.archive_dir == tmp_path / "gepa"
    assert (built.minibatch_size, built.rng_seed, built.max_metric_calls) == (2, 7, 150)
    assert (built.skip_perfect_score, built.perfect_score, built.kinds) == (False, 0.5, ("rules",))
    assert built.feedback is feedback_hook
    # Unbound until build: the archive is per scenario, so the seams cannot
    # be filled at config time.
    assert isinstance(built.build("demo", RecordStore()), Trainer)


def test_a_seed_that_breaks_the_id_convention_is_refused(tmp_path: Path) -> None:
    config = sections(tmp_path)
    config["evolution"]["seed"] = [{"id": "wrong", "name": "skill", "config": {"name": "notes", "text": "# n"}}]
    with pytest.raises(RecipeConfigError, match="must have id equal to its config name"):
        build_recipe(str(config["implementation"]), {}, config=config, runtime=runtime())

    config["evolution"]["seed"] = [{"id": "house-rules", "name": "rules", "config": {"text": SEED_TEXT}}]
    with pytest.raises(RecipeConfigError, match="entry id 'rules'"):
        build_recipe(str(config["implementation"]), {}, config=config, runtime=runtime())


def test_an_unknown_evolvable_kind_is_refused(tmp_path: Path) -> None:
    config = sections(tmp_path, components=["config"])
    with pytest.raises(RecipeConfigError, match="components must name kinds"):
        build_recipe(str(config["implementation"]), {}, config=config, runtime=runtime())


def test_a_missing_gepa_block_is_refused(tmp_path: Path) -> None:
    config = sections(tmp_path)
    config["evolution"].pop("gepa")
    with pytest.raises(RecipeConfigError, match=r"evolution\.gepa"):
        build_recipe(str(config["implementation"]), {}, config=config, runtime=runtime())


def test_a_configured_selection_object_is_left_alone(tmp_path: Path) -> None:
    """Only the empty seams are filled: an operator who names a policy keeps it."""
    from reef.train.evaluation import AlwaysSelect

    config = sections(tmp_path)
    config["evolution"]["selection"] = AlwaysSelect()
    built = build_recipe(str(config["implementation"]), {}, config=config, runtime=runtime())
    assert isinstance(built.candidate_selector, AlwaysSelect)


# -- one full step through the real backend and commit path -----------------


def _report_once(scenario, name: str, suffix: str) -> None:
    """One scored sample through the record store: enough to wake one step."""
    scenario.records.append_result(
        AgentRecord.create(
            scenario=name,
            request_type=RequestType.INFERENCE,
            payload={"messages": [{"role": "user", "content": TASK}]},
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


def test_one_step_publishes_and_the_gate_carries_the_gepa_metrics(tmp_path: Path) -> None:
    """The whole loop on the real mechanism: reflection rewrites the rules
    node, the child wins its minibatch, the mechanism scores both trees over
    the validation set, and the archive's own numbers ride the release gate."""
    make_binary(tmp_path)
    config = sections(tmp_path)
    booted = build_recipe(str(config["implementation"]), {}, config=config, runtime=runtime())
    # The declared reflection endpoint is the one thing this run may not
    # reach; everything else is the production path.
    built = dataclasses.replace(booted, models={"reflection": FakeChat(FENCED_REPLY)})
    initial = tmp_path / "initial"
    initial.mkdir()
    factory = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    dispatcher = Dispatcher(
        RecipeRegistry({built.name: built}),
        factory,
        agent_record_dir=tmp_path / "agent-record",
    )
    try:
        scenario = dispatcher.get_or_create_scenario("gepa-demo", built.name)
        assert scenario is not None
        _report_once(scenario, "gepa-demo", "1")
        result = scenario.prepare_training_step()
        assert result is not None
        scenario.commit(result)
    finally:
        dispatcher.close()

    assert result.metrics["published"] is True
    assert result.metrics["mutation"] == {"op": "update", "id": "rules"}
    gate = result.metrics
    assert (gate["candidate_val_mean"], gate["served_val_mean"], gate["parent"]) == (1.0, 0.0, 0)
    assert (gate["archive_size"], gate["front_size"]) == (2, 1)
    # The seed's validation (2), the served minibatch (1), the child's (1),
    # then the candidate's validation (2): the mechanism's re-run of the
    # served side is not a metric call.
    assert gate["metric_calls"] == 6
    assert gate["selection"]["policy"] == "gepa"

    archive = json.loads((tmp_path / "gepa" / "gepa-demo.json").read_text())
    assert archive["served"] == 1
    assert archive["pending"] is None
    assert archive["candidates"][1]["texts"] == {"rules": REFLECTED}
    assert archive["candidates"][1]["val_scores"] == [1.0, 1.0]


def test_every_reflection_is_logged_with_its_prompt_reply_and_verdict(tmp_path: Path) -> None:
    """The archive keeps what the reflection model saw and wrote, accepted or
    not, so a search outcome can be traced to the exact text rather than
    inferred from the candidates it left behind."""
    archive = Archive(tmp_path / "archive.json")
    models, reflector = bindings()

    proposer(archive, FakeEpisodes())(NODES, (SAMPLE,), models)

    (row,) = archive.proposals
    assert (row["parent"], row["component"], row["accepted"], row["candidate"]) == (0, "rules", True, 1)
    assert row["prompt"] == reflector.prompts[0]
    assert row["reply"] == FENCED_REPLY
    assert row["minibatch"][0]["Inputs"] == TASK
    assert (row["parent_scores"], row["child_scores"]) == ([0.0], [1.0])
    assert Archive(tmp_path / "archive.json").proposals == archive.proposals  # persisted
