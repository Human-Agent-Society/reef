"""Guarantees of the recipes/gepa method package, hermetic: no reflection
model is ever reached, the proposer's minibatch episodes run through a fake
trial runner, and the full steps drive a fake pi binary through the real
adapter, backend, and commit path."""

from __future__ import annotations

import dataclasses
import json
import random
from pathlib import Path
from typing import Any

import pytest
from tests.support._strategies import resolve_episode_scorer, resolve_proposer
from tests.support._trajectories import recorded_trajectory
from tests.support.harness_steps import TreeReleases

from recipes.gepa import components, reflection
from recipes.gepa.archive import Archive, Candidate
from recipes.gepa.proposer import (
    ARCHIVE_KEY,
    Feedback,
    GEPAProposer,
    GEPASettings,
    ScoreFeedback,
    archive_for_step,
    traffic,
)
from recipes.gepa.recipe import GEPARecipe
from recipes.gepa.selection import GEPASelection
from reef.artifact import InMemoryRepositoryBackend
from reef.core import AgentRecord, RequestType
from reef.core.evaluation import EvaluationResult
from reef.core.trajectories import recorded_payload
from reef.dispatcher import Dispatcher
from reef.evaluation.harness import FailureObservation, HarnessCandidate, ScoredEpisode
from reef.harness.adapters import get_adapter
from reef.harness.episodes.model_binding import ModelBinding, ModelBindings
from reef.harness.episodes.run import EpisodeResult
from reef.harness.tree import HarnessTree, TreeKind, TreePatch
from reef.inference.http import InferenceProxyRuntime
from reef.recipe import RecipeConfigError
from reef.recipe.registry import build_recipe
from reef.storage.sqlite import SQLiteRecordStore, SQLiteScenarioStorage
from reef.train.backend import ComponentReleases
from reef.train.harness_backend.contract import ProposalContext, SampleSource, StepOutcome, TreeProposal, TrialRunner
from reef.train.trainer import Trainer

# The fake harness scores itself: its trajectory carries the rules text, so
# the scorer can prefer any tree whose instructions carry the marker.
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

#: The served tree as the backend passes it. The skill is here to prove ``kinds`` narrows what GEPA is allowed to
#: rewrite, and the config file that GEPA never rewrites structure.
TREE = HarnessTree({"config/primary.json": "{}\n", "rules.md": SEED_TEXT, "skills/notes/SKILL.md": "# notes"})
#: The validation set the step's evaluation runs.
EVALUATION_TASKS = ("check one", "check two")

TASK = "what is 2+2?"
SAMPLE = recorded_trajectory(
    "a1",
    {
        "messages": [{"role": "user", "content": TASK}],
        "response": {"choices": [{"message": {"role": "assistant", "content": "4"}}]},
    },
    0.0,
)
#: The same exchange as the proxy records a streamed reply: the message it
#: assembled from the chunks sits beside the raw stream body, no ``choices``.
STREAMED_SAMPLE = recorded_trajectory(
    "a2",
    {
        "messages": [{"role": "user", "content": TASK}],
        "response": {"stream": True, "body": "data: ...", "message": {"role": "assistant", "content": "4"}},
    },
    0.0,
)


def test_recorded_traffic_reads_buffered_and_streamed_replies() -> None:
    assert [texts for _, texts in traffic((SAMPLE, STREAMED_SAMPLE))] == [(TASK, "4"), (TASK, "4")]


def score_rules(task: str, result: EpisodeResult) -> float:
    """1.0 when the tree that ran carried the marker instruction."""
    last = result.trajectory[-1] if result.trajectory else {}
    return 1.0 if MARKER in str(last.get("content", last.get("rules", ""))) else 0.0


class FakeTrials(TrialRunner):
    """Trial episodes that answer with the rules text the tree carries; ``failure`` fails every episode."""

    def __init__(self, failure: str | None = None) -> None:
        self.runs: list[tuple[str, tuple[str, ...]]] = []
        self.failure = failure

    def run(self, tree: HarnessTree, tasks) -> tuple[ScoredEpisode, ...]:
        rules = tree.files.get("rules.md", "")
        self.runs.append((rules, tuple(tasks)))
        if self.failure is not None:
            return tuple(ScoredEpisode(None, FailureObservation(task, "launch", self.failure)) for task in tasks)
        return tuple(ScoredEpisode(1.0 if MARKER in rules else 0.0, None, reply=rules) for task in tasks)


class FakeChat(ModelBinding):
    """A reflection model that answers one canned reply and keeps the prompts it was asked."""

    def __init__(self, reply: str = FENCED_REPLY) -> None:
        super().__init__(base_url="http://127.0.0.1:9", model="reflection", api_key="dummy")
        object.__setattr__(self, "reply", reply)
        object.__setattr__(self, "prompts", [])

    def chat(self, messages, **params):
        self.prompts.append(messages[0]["content"])
        return self.reply


def bindings(reply: str = FENCED_REPLY, *, declared: bool = True) -> tuple[ModelBindings, FakeChat]:
    """The models a proposer receives, and the binding its reflection lands on."""
    reflector = FakeChat(reply)
    if declared:
        return ModelBindings(served=FakeChat(""), named={"reflection": reflector}), reflector
    return ModelBindings(served=reflector), reflector


def proposer(**overrides: Any) -> GEPAProposer:
    settings: dict[str, Any] = {
        "minibatch_size": 3,
        "rng_seed": 0,
        "skip_perfect_score": True,
        "perfect_score": 1.0,
        "max_metric_calls": None,
        "kinds": (TreeKind.RULES,),
    }
    settings.update(overrides)
    return GEPAProposer(GEPASettings(**settings))


def context(
    models: ModelBindings,
    trials: TrialRunner | None,
    *,
    samples: tuple = (SAMPLE,),
    archive: Archive | None = None,
    last_outcome: StepOutcome | None = None,
    step: int = 1,
) -> ProposalContext:
    """One automatic step over ``TREE``, with ``archive`` as the committed method state."""
    return ProposalContext(
        step=step,
        tree=TREE,
        descriptor=get_adapter("pi"),
        samples=samples,
        sources=tuple(SampleSource(None, "untagged") for _ in samples),
        request=None,
        evaluation_tasks=EVALUATION_TASKS,
        models=models,
        method_state={} if archive is None else {ARCHIVE_KEY: archive.to_state()},
        last_outcome=last_outcome,
        trials=trials,
    )


def archive_of(proposal: TreeProposal) -> Archive:
    assert proposal.method_state is not None
    return Archive.from_state(proposal.method_state[ARCHIVE_KEY])


# -- components: the tree as a GEPA candidate ----------------------------------


def test_texts_of_keys_the_evolvable_files_by_path() -> None:
    assert components.texts_of(TREE, (TreeKind.RULES, TreeKind.SKILL)) == {
        "rules.md": SEED_TEXT,
        "skills/notes/SKILL.md": "# notes",
    }
    assert components.texts_of(TREE, (TreeKind.RULES,)) == {"rules.md": SEED_TEXT}


def test_the_patch_writes_the_components_the_child_changed_against_the_served_tree() -> None:
    served = {"rules.md": SEED_TEXT, "skills/notes/SKILL.md": "# notes"}
    child = {"rules.md": REFLECTED, "skills/notes/SKILL.md": "# better", "skills/absent/SKILL.md": "unreachable"}
    assert components.patch_to(served, child) == TreePatch(
        {"rules.md": REFLECTED, "skills/notes/SKILL.md": "# better"}
    )
    assert components.patch_to(served, served) == TreePatch({})
    # The tree a minibatch episode runs: the served tree with the candidate's texts, every other file kept.
    assert components.candidate_tree(TREE, {"rules.md": REFLECTED}) == HarnessTree(
        {**TREE.files, "rules.md": REFLECTED}
    )


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


# -- archive: fronts, parent sampling, the committed state ------------------


def archive_with(*scores: list[float] | None) -> Archive:
    archive = Archive()
    for index, score in enumerate(scores):
        if index == 0:
            archive.seed({"rules.md": f"text {index}"})
        else:
            archive.add({"rules.md": f"text {index}"}, 0, [0.0], step=index)
        if score is not None:
            archive.record_validation(index, score)
    archive.clear_pending()
    return archive


def test_a_strict_win_replaces_a_front_and_an_equal_score_joins_it() -> None:
    archive = archive_with([0.0, 1.0], [1.0, 1.0], [0.0, 0.0])
    assert archive.fronts() == {0: {1}, 1: {0, 1}}
    assert archive.best() == 1
    assert archive.mean_val(2) == 0.0


def test_an_unvalidated_candidate_is_on_no_front() -> None:
    archive = archive_with([1.0, 1.0], None)
    assert archive.fronts() == {0: {0}, 1: {0}}


class RecordingRandom(random.Random):
    """Captures the frequency-weighted sampling list GEPA builds."""

    seen: list[int] = []

    def choice(self, seq):
        self.seen = list(seq)
        return seq[0]


def test_parent_sampling_weights_survivors_by_how_many_fronts_they_hold() -> None:
    """Two specialists, neither dominated: the one holding two of the three
    instance fronts is drawn twice as often as the one holding one."""
    archive = archive_with([1.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    rng = RecordingRandom()
    assert archive.select_parent(rng) == 0
    assert rng.seen == [0, 0, 1]


def test_parent_sampling_drops_candidates_another_candidate_covers() -> None:
    """Candidate 2 is on every front that 0 and 1 are on, so only it survives."""
    archive = archive_with([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0])
    rng = RecordingRandom()
    assert archive.select_parent(rng) == 2
    assert rng.seen == [2, 2, 2]


def test_parent_sampling_falls_back_to_the_served_candidate() -> None:
    archive = Archive()
    archive.seed({"rules.md": SEED_TEXT})
    assert archive.select_parent(random.Random(0)) == 0


def test_the_round_robin_cursor_walks_the_components_and_children_inherit_it() -> None:
    archive = Archive()
    archive.seed({"rules.md": "r", "skills/notes/SKILL.md": "n"})
    keys = ["rules.md", "skills/notes/SKILL.md"]
    assert [archive.next_component(0, keys) for _ in range(3)] == ["rules.md", "skills/notes/SKILL.md", "rules.md"]
    child = archive.add({"rules.md": "r2", "skills/notes/SKILL.md": "n"}, 0, [1.0], step=1)
    assert archive.next_component(child, keys) == "skills/notes/SKILL.md"


def test_the_archive_state_round_trips() -> None:
    archive = Archive(rng_seed=3)
    archive.seed({"rules.md": SEED_TEXT})
    archive.charge(4)
    archive.add({"rules.md": REFLECTED}, 0, [1.0], step=2)
    archive.record_validation(1, [1.0, 0.0])
    archive.reject()

    resumed = Archive.from_state(json.loads(json.dumps(archive.to_state())))
    assert resumed == archive
    assert (resumed.metric_calls, resumed.steps, resumed.rejections, resumed.pending_step) == (4, 2, 1, 2)
    assert resumed.candidates[1] == Candidate(
        texts={"rules.md": REFLECTED},
        parent=0,
        val_scores=[1.0, 0.0],
        minibatch_scores=[1.0],
        cursor=0,
        discovered_at=4,
    )
    with pytest.raises(ValueError, match="must be a mapping"):
        Archive.from_state(["not", "a", "state"])


def test_the_plan_is_a_function_of_the_committed_state_and_walks_epochs() -> None:
    """A driver that reads the committed state draws the plan the proposer draws, and an epoch shows every
    training problem once before the order is shuffled again."""
    archive = Archive()
    archive.seed({"rules.md": SEED_TEXT})
    state = archive.to_state()
    first = Archive.from_state(state).plan(trainset_size=4, valset_size=2, minibatch_size=2)
    assert first == archive.plan(trainset_size=4, valset_size=2, minibatch_size=2)
    assert first["iteration"] == 0 and first["parent"] == 0
    second = archive.plan(trainset_size=4, valset_size=2, minibatch_size=2)
    assert second["iteration"] == 1 and sorted(first["minibatch"] + second["minibatch"]) == [0, 1, 2, 3]
    assert archive.iteration == 2


def test_settling_a_step_records_both_sides_and_serves_a_selected_candidate() -> None:
    archive = Archive()
    archive.seed({"rules.md": SEED_TEXT})
    archive.add({"rules.md": REFLECTED}, 0, [1.0], step=1)
    archive.settle([1.0, 0.0], [0.0, 0.0], selected=True)
    assert archive.candidates[1].val_scores == [1.0, 0.0]
    # The seed's first validation arrives with the first evaluated step; it was charged when it was seeded.
    assert archive.candidates[0].val_scores == [0.0, 0.0]
    assert (archive.served, archive.pending, archive.pending_step, archive.metric_calls) == (1, None, None, 2)
    # A rejected candidate keeps its scores, and with them its fronts: it can still parent proposals.
    archive.add({"rules.md": "third"}, 1, [1.0], step=2)
    archive.settle([0.0, 1.0], [1.0, 0.0], selected=False)
    assert (archive.served, archive.candidates[2].val_scores, archive.candidates[1].val_scores) == (
        1,
        [0.0, 1.0],
        [1.0, 0.0],
    )


def test_a_step_folds_in_only_the_outcome_of_the_step_that_proposed_its_pending_candidate() -> None:
    archive = Archive()
    archive.seed({"rules.md": SEED_TEXT})
    archive.add({"rules.md": REFLECTED}, 0, [1.0], step=3)
    state = {ARCHIVE_KEY: archive.to_state()}

    def outcome(step: int) -> StepOutcome:
        return StepOutcome(step, TreePatch({"rules.md": REFLECTED}), True, "won", (1.0, None), (0.0, 0.0))

    settled = archive_for_step(state, outcome(3))
    # A failed episode is a zero, not a gap.
    assert settled.candidates[1].val_scores == [1.0, 0.0] and settled.served == 1
    # The step that proposed it never reached an evaluation: the candidate is dropped unvalidated.
    for last in (outcome(2), None):
        dropped = archive_for_step(state, last)
        assert dropped.pending is None and dropped.served == 0 and dropped.candidates[1].val_scores is None


# -- proposer: minibatch, reflection, acceptance ----------------------------


def test_a_served_parent_uses_recorded_traffic_and_runs_no_parent_episodes() -> None:
    """The served tree's minibatch is free: the step already batched its real
    traffic, so only the child costs episodes."""
    trials = FakeTrials()
    models, reflector = bindings()

    proposal = proposer().propose(context(models, trials))

    assert proposal.patch == TreePatch({"rules.md": REFLECTED})
    assert trials.runs == [(REFLECTED, (TASK,))]  # the child only
    archive = archive_of(proposal)
    # GEPA's count: the seed's validation (2 tasks), the served parent's
    # minibatch (1, from traffic), and the child's minibatch (1).
    assert archive.metric_calls == 4
    assert (archive.pending, archive.pending_step) == (1, 1)
    assert archive.candidates[1].parent == 0
    assert archive.candidates[1].minibatch_scores == [1.0]
    assert archive.candidates[0].texts == {"rules.md": SEED_TEXT}  # the skill stayed out of ``kinds``
    # The reflective record is the recorded exchange, graded by the hook.
    (prompt,) = reflector.prompts
    assert f"## Inputs\n{TASK}" in prompt
    assert "## Generated Outputs\n4" in prompt
    assert "## Feedback\nThis response scored 0.000." in prompt
    # The notes keep the iteration without the prompt and the minibatch, which the archive keeps in full.
    assert proposal.notes["accepted"] is True and "prompt" not in proposal.notes and "minibatch" not in proposal.notes


def committed_parent() -> Archive:
    """An archive whose validated candidate 1 dominates the served seed, so it is always the parent."""
    archive = Archive()
    archive.seed({"rules.md": SEED_TEXT})
    archive.add({"rules.md": "Parent rules."}, 0, [1.0], step=1)
    archive.record_validation(0, [0.0, 0.0])
    archive.record_validation(1, [1.0, 1.0])
    archive.clear_pending()
    return archive


def test_a_parent_that_never_served_is_re_run_on_the_minibatch() -> None:
    """No traffic was ever served from an archive candidate, so its minibatch
    scores and its outputs have to be produced by episodes."""
    trials = FakeTrials()
    models, reflector = bindings()

    proposal = proposer().propose(context(models, trials, archive=committed_parent()))

    assert proposal.patch == TreePatch({"rules.md": REFLECTED})
    assert trials.runs == [("Parent rules.", (TASK,)), (REFLECTED, (TASK,))]  # parent then child
    archive = archive_of(proposal)
    assert archive.metric_calls == 2
    assert archive.candidates[2].parent == 1
    (prompt,) = reflector.prompts
    assert "Parent rules." in prompt  # reflection reads the parent's own text and output


def test_a_child_that_does_not_beat_its_parent_is_rejected() -> None:
    models, _ = bindings("```\nNo marker here.\n```")

    proposal = proposer().propose(context(models, FakeTrials()))

    assert proposal.patch == TreePatch({}) and proposal.notes["accepted"] is False
    archive = archive_of(proposal)
    assert (archive.rejections, archive.steps, archive.pending) == (1, 1, None)
    assert len(archive.candidates) == 1


def test_an_episode_that_cannot_run_scores_zero_and_reports_its_error() -> None:
    models, reflector = bindings()

    proposal = proposer().propose(
        context(models, FakeTrials(failure="harness binary 'pi' not found"), archive=committed_parent())
    )

    assert proposal.patch == TreePatch({})  # the child ties the parent at 0.0
    (prompt,) = reflector.prompts
    assert "## Feedback\nharness binary 'pi' not found" in prompt


def test_a_step_without_trial_episodes_is_refused() -> None:
    models, _ = bindings()
    with pytest.raises(ValueError, match="trial episodes"):
        proposer().propose(context(models, None))


def test_the_budget_short_circuits_before_any_model_call() -> None:
    archive = Archive()
    archive.seed({"rules.md": SEED_TEXT})
    archive.charge(150)
    trials = FakeTrials()
    models, reflector = bindings()

    proposal = proposer(max_metric_calls=150).propose(context(models, trials, archive=archive))

    assert proposal.patch == TreePatch({}) and "150 metric calls" in proposal.notes["stopped"]
    assert (trials.runs, reflector.prompts) == ([], [])


def test_a_perfect_minibatch_short_circuits_before_any_model_call() -> None:
    trials = FakeTrials()
    models, reflector = bindings()
    solved = recorded_trajectory("a1", dict(recorded_payload(SAMPLE)), 1.0)

    proposal = proposer().propose(context(models, trials, samples=(solved,)))
    assert proposal.patch == TreePatch({}) and proposal.notes == {
        "skipped": "the parent scored perfectly on the minibatch"
    }
    assert (trials.runs, reflector.prompts) == ([], [])
    # The same minibatch is worth reflecting on once the bar is above it.
    proposer(perfect_score=2.0).propose(context(models, trials, samples=(solved,)))
    assert len(reflector.prompts) == 1


def test_reflection_falls_back_to_the_model_under_test() -> None:
    models, served = bindings(declared=False)

    assert proposer().propose(context(models, FakeTrials())).patch.changes
    assert len(served.prompts) == 1


def test_a_tree_the_archive_does_not_recognise_is_reseeded() -> None:
    """An operator edit or a rollback is a new root, never a descendant."""
    archive = Archive()
    archive.seed({"rules.md": "something else"})
    models, _ = bindings()

    reseeded = archive_of(proposer().propose(context(models, FakeTrials(), archive=archive)))

    assert reseeded.candidates[1].texts == {"rules.md": SEED_TEXT}
    assert reseeded.candidates[1].parent is None
    assert reseeded.candidates[2].parent == 1  # the step's child descends from the new root


def test_every_reflection_is_logged_with_its_prompt_reply_and_result() -> None:
    """The archive keeps what the reflection model saw and wrote, accepted or
    not, so a search outcome can be traced to the exact text rather than
    inferred from the candidates it left behind."""
    models, reflector = bindings()

    archive = archive_of(proposer().propose(context(models, FakeTrials())))

    (row,) = archive.proposals
    assert (row["parent"], row["component"], row["accepted"], row["candidate"]) == (0, "rules.md", True, 1)
    assert row["prompt"] == reflector.prompts[0]
    assert row["reply"] == FENCED_REPLY
    assert row["minibatch"][0]["Inputs"] == TASK
    assert (row["parent_scores"], row["child_scores"]) == ([0.0], [1.0])


# -- selection: the stored validation means ----------------------------------


def evaluation(candidate: tuple, current: tuple) -> EvaluationResult:
    return EvaluationResult(
        evaluator="harness_episode_pairs",
        evaluator_version="1",
        metrics={"candidate_scores": candidate, "current_scores": current},
    )


def pending_archive() -> Archive:
    archive = Archive()
    archive.seed({"rules.md": SEED_TEXT})
    archive.add({"rules.md": REFLECTED}, 0, [1.0], step=1)
    return archive


def harness_candidate(archive: Archive) -> HarnessCandidate:
    return HarnessCandidate(
        candidate_id="c1",
        candidate_tree=HarnessTree({"rules.md": REFLECTED}),
        current_tree=HarnessTree({"rules.md": SEED_TEXT}),
        patch=TreePatch({"rules.md": REFLECTED}),
        evaluation_tasks=EVALUATION_TASKS,
        method_state={ARCHIVE_KEY: archive.to_state()},
    )


def test_the_first_decision_compares_against_the_seeds_own_run() -> None:
    decision = GEPASelection(None).decide(harness_candidate(pending_archive()), evaluation((1.0, None), (0.0, 0.0)))

    assert decision.selected and decision.policy == "gepa"
    metrics = decision.metrics
    assert (metrics["candidate_val_mean"], metrics["served_val_mean"], metrics["parent"]) == (0.5, 0.0, 0)
    # The candidate's validation (2) counts already; the seed's was charged when it was seeded.
    assert (metrics["archive_size"], metrics["metric_calls"]) == (2, 2)


def test_a_candidate_that_does_not_beat_the_stored_served_mean_is_rejected() -> None:
    """Once the served candidate has a stored validation, the step's fresh run of it does not count."""
    archive = pending_archive()
    archive.record_validation(0, [1.0, 0.0])
    decision = GEPASelection(None).decide(harness_candidate(archive), evaluation((1.0, 0.0), (0.0, 0.0)))
    assert not decision.selected  # equal means are not an improvement
    assert decision.metrics["served_val_mean"] == 0.5


def test_the_reported_front_holds_the_candidates_the_steps_validation_puts_on_it() -> None:
    decision = GEPASelection(None).decide(harness_candidate(pending_archive()), evaluation((1.0, 0.0), (0.0, 0.0)))
    assert decision.metrics["front_size"] == 2  # the candidate holds task 0, the seed ties on task 1


def test_the_selection_takes_no_config_and_decides_harness_candidates_only() -> None:
    with pytest.raises(ValueError, match="takes no config"):
        GEPASelection.from_config({"margin": 1}, candidate_backend=None)
    from reef.core.evaluation import UpdateCandidate

    with pytest.raises(TypeError, match="HarnessCandidate"):
        GEPASelection(None).decide(UpdateCandidate("c1"), evaluation((1.0,), (0.0,)))


# -- recipe: config boot ------------------------------------------------------


class FeedbackHook(Feedback):
    def feedback(self, task: str, output: str, score: float) -> str:
        return f"{task} answered {output} at {score}"


feedback_hook = FeedbackHook()


def runtime() -> InferenceProxyRuntime:
    return InferenceProxyRuntime(model_path="served-model", base_url="http://127.0.0.1:9", api_key="dummy")


def make_binary(tmp_path: Path) -> Path:
    binary = tmp_path / "fake-pi"
    binary.write_text(PI_FAKE)
    binary.chmod(0o755)
    return binary


def sections(tmp_path: Path, **gepa: Any) -> dict[str, Any]:
    seed = tmp_path / "seed"
    if not seed.exists():
        HarnessTree({"rules.md": SEED_TEXT}).write(seed)
    block = {
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
            "evaluate": resolve_episode_scorer(score_rules),
            "feedback": feedback_hook,
            "tasks": list(EVALUATION_TASKS),
            "seed": str(seed),
            "gepa": block,
            "batch_size": 1,
        },
    }


def test_the_config_boots_the_recipe_with_the_proposer_and_the_selection(tmp_path: Path) -> None:
    config = sections(tmp_path)
    built = build_recipe(str(config["implementation"]), {}, config=config, runtime=runtime())

    assert isinstance(built, GEPARecipe) and built.name == "gepa"
    assert isinstance(built.propose, GEPAProposer)
    assert built.propose.settings == GEPASettings(
        minibatch_size=2,
        rng_seed=7,
        skip_perfect_score=False,
        perfect_score=0.5,
        max_metric_calls=150,
        kinds=(TreeKind.RULES,),
    )
    assert built.propose.feedback is feedback_hook
    assert built.selection.plugin_class is GEPASelection
    assert built.seed == HarnessTree({"rules.md": SEED_TEXT})
    releases = ComponentReleases(TreeReleases(built.seed), "harness")
    assert isinstance(built.build("demo", SQLiteRecordStore(), releases=releases), Trainer)
    # Without a feedback hook the reflection reads the score restated.
    config["evolution"].pop("feedback")
    default = build_recipe(str(config["implementation"]), {}, config=config, runtime=runtime())
    assert isinstance(default.propose.feedback, ScoreFeedback)


@pytest.mark.parametrize(
    ("gepa", "message"),
    [
        ({"components": ["config"]}, "components must name kinds"),
        ({"components": ["bogus"]}, "bogus"),
        ({"components": []}, "non-empty list"),
        ({"archive": "gepa"}, r"unknown key\(s\) archive"),
        ({"minibatch_size": 0}, "minibatch_size must be positive"),
        ({"seed": "zero"}, "seed must be an integer"),
    ],
)
def test_an_invalid_gepa_block_is_refused(tmp_path: Path, gepa: dict[str, Any], message: str) -> None:
    config = sections(tmp_path, **gepa)
    with pytest.raises(RecipeConfigError, match=message):
        build_recipe(str(config["implementation"]), {}, config=config, runtime=runtime())


def test_a_missing_gepa_block_is_refused(tmp_path: Path) -> None:
    config = sections(tmp_path)
    config["evolution"].pop("gepa")
    with pytest.raises(RecipeConfigError, match=r"evolution\.gepa"):
        build_recipe(str(config["implementation"]), {}, config=config, runtime=runtime())


def test_a_configured_selection_and_proposer_are_left_alone(tmp_path: Path) -> None:
    """Only the empty seams are filled: an operator who names a policy or a proposer keeps it."""
    from reef.evaluation.evaluators import BackendAlwaysSelectPlugin

    own = resolve_proposer(lambda context: None)
    config = sections(tmp_path)
    config["evolution"].update(selection="always", propose=own)
    built = build_recipe(str(config["implementation"]), {}, config=config, runtime=runtime())
    assert built.selection.plugin_class is BackendAlwaysSelectPlugin
    assert built.propose is own


# -- full steps through the real backend and commit path -----------------------


def report_once(scenario, name: str, suffix: str) -> None:
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


def test_a_step_publishes_and_the_next_step_folds_its_validation_into_the_committed_archive(tmp_path: Path) -> None:
    """The whole loop on the real backend: reflection rewrites the rules,
    the child wins its minibatch, the evaluation scores both trees over the
    validation set, the selection's numbers ride the step's metrics, and the
    archive travels as the committed method state, settled by the next step."""
    make_binary(tmp_path)
    config = sections(tmp_path)
    booted = build_recipe(str(config["implementation"]), {}, config=config, runtime=runtime())
    # The declared reflection endpoint is the one thing this run may not
    # reach; everything else is the production path.
    built = dataclasses.replace(booted, models={"reflection": FakeChat(FENCED_REPLY)})
    initial = tmp_path / "initial"
    built.seed.write(initial)
    factory = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    dispatcher = Dispatcher(built, factory, scenario_storage=SQLiteScenarioStorage(tmp_path / "agent-record"))
    try:
        scenario = dispatcher.get_or_create_scenario("gepa-demo")
        assert scenario is not None
        report_once(scenario, "gepa-demo", "1")
        result = scenario.prepare_training_step()
        assert result is not None and result.state is not None
        assert result.state["method_state"][ARCHIVE_KEY]["pending"] == 1
        scenario.commit(result)

        report_once(scenario, "gepa-demo", "2")
        second = scenario.prepare_training_step()
        assert second is not None and second.state is not None
    finally:
        dispatcher.close()

    metrics = result.metrics
    assert metrics["selected"] is True and metrics["patch"] == {"rules.md": REFLECTED}
    assert (metrics["candidate_val_mean"], metrics["served_val_mean"], metrics["parent"]) == (1.0, 0.0, 0)
    assert metrics["archive_size"] == 2 and metrics["selection"]["policy"] == "gepa"
    # The seed's validation (2), the served minibatch (1), the child's (1),
    # then the candidate's validation (2): the evaluation's re-run of the
    # served side is not a metric call.
    assert metrics["metric_calls"] == 6
    assert metrics["proposal_notes"]["candidate"] == 1

    archive = Archive.from_state(second.state["method_state"][ARCHIVE_KEY])
    assert archive.served == 1
    assert archive.candidates[1].texts == {"rules.md": REFLECTED}
    assert archive.candidates[1].val_scores == [1.0, 1.0] and archive.candidates[0].val_scores == [0.0, 0.0]
