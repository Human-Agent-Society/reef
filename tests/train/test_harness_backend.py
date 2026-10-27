"""The harness backend's step: from the head tree and one proposal to a published, rejected or skipped step.

Each step runs through the phases the trainer runs (``backend_step``) on an
in-memory commit history whose base release is the pi seed; the test
records a published tree as the next release, as the scenario committer
does. Episodes run the fake pi binary, whose score prefers rules that hold
the marker.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.support._trajectories import recorded_trajectory
from tests.support.harness_service import backend_step
from tests.support.harness_steps import PI_SEED, TreeReleases, changed_rules, harness_backend

from reef.core.training_request import TrainingRequest
from reef.harness.tree import HarnessTree, TreeKind, TreePatch
from reef.train.harness_backend.contract import RejectedProposal, SampleSource, StepOutcome, TreeProposal
from reef.train.harness_backend.inbox import ProposalInbox
from reef.train.harness_backend.state import HarnessState
from reef.train.types import JobData, TrainStepResult

MARKER = TreePatch(changed_rules("marker rules\n"))
PLAIN = TreePatch(changed_rules("plain rules\n"))
REQUEST = TrainingRequest("tighten the rules", "s", "base", "ask-1")


def batch(step: int, *samples, request: TrainingRequest | None = None) -> JobData:
    return JobData(f"demo:{step}", tuple(samples), request=request)


def published(releases: TreeReleases, release_id: str, result: TrainStepResult) -> HarnessTree:
    """Record ``result``'s tree as release ``release_id``, as the committer's publish does."""
    if result.artifact is None or result.artifact.local_path is None:
        raise AssertionError(f"the step published nothing: {result.metrics.get('selection')}")
    tree = HarnessTree.read(result.artifact.local_path)
    releases.publish(release_id, tree)
    return tree


def state_of(result: TrainStepResult) -> HarnessState:
    return HarnessState.from_json(result.state)


def test_a_winning_patch_publishes_the_head_tree_with_the_patch_applied(tmp_path: Path) -> None:
    backend, releases = harness_backend(tmp_path, lambda context: MARKER)
    result = backend_step(backend, batch(1), backend.initial_state())
    assert result.metrics["selected"] is True and result.pending is False
    assert result.metrics["head_release_id"] == "base" and result.metrics["patch"] == MARKER.to_json()
    assert HarnessTree.read(result.artifact.local_path) == PI_SEED.apply(MARKER)
    selection = result.metrics["selection"]
    assert (selection["selected"], selection["policy"], selection["candidate_id"]) == (
        True,
        "score_comparison",
        "demo:1:candidate",
    )
    assert selection["metrics"] == {"wins": 1, "losses": 0, "ties": 0} and result.metrics["wins"] == 1
    reason = selection["reason"]
    state = state_of(result)
    assert state.steps == 1 and state.failure_streak == 0 and state.rejected == ()
    assert state.last_outcome == StepOutcome(1, MARKER, True, reason, (1.0,), (0.0,))
    # The committer publishes; the backend's own tree goes once the commit is durable.
    assert releases.head("harness").release_id == "base"
    backend.commit_applied(result.state)
    assert not result.artifact.local_path.exists()


def test_a_losing_patch_keeps_the_head_and_records_the_rejected_patch(tmp_path: Path) -> None:
    contexts = []
    backend, releases = harness_backend(
        tmp_path, lambda context: contexts.append(context) or PLAIN, max_rejected_history=1
    )
    first = backend_step(backend, batch(1), backend.initial_state())
    assert first.metrics["selected"] is False and first.artifact is None
    assert releases.head("harness").release_id == "base"
    rejected = RejectedProposal(1, PLAIN, first.metrics["selection"]["reason"])
    assert state_of(first).rejected == (rejected,) and state_of(first).failure_streak == 1
    assert state_of(first).last_outcome.selected is False and state_of(first).last_outcome.patch == PLAIN
    second = backend_step(backend, batch(2), first.state)
    # The next proposal sees the rejection; the state keeps the latest max_rejected_history of them.
    assert contexts[1].rejected == (rejected,) and contexts[1].tree == PI_SEED
    assert [proposal.step for proposal in state_of(second).rejected] == [2]
    assert state_of(second).failure_streak == 2


def test_no_proposal_skips_the_step_with_nothing_to_record(tmp_path: Path) -> None:
    backend, _ = harness_backend(tmp_path, lambda context: None)
    result = backend_step(backend, batch(1), backend.initial_state())
    assert result.metrics["skipped"] == "no proposal" and result.artifact is None
    assert result.state == HarnessState(steps=1).to_json()
    assert backend.step_progress is None


def test_an_empty_patch_skips_the_step_but_commits_the_method_state_and_notes(tmp_path: Path) -> None:
    contexts = []

    def propose(context):
        contexts.append(context)
        return TreeProposal(TreePatch({}), notes={"plan": "wait"}, method_state={"round": context.step})

    backend, _ = harness_backend(tmp_path, propose)
    first = backend_step(backend, batch(1), backend.initial_state())
    assert first.metrics["skipped"] == "the proposal changes nothing" and first.artifact is None
    assert first.metrics["proposal_notes"] == {"plan": "wait"}
    assert state_of(first).method_state == {"round": 1}
    backend_step(backend, batch(2), first.state)
    assert [dict(context.method_state) for context in contexts] == [{}, {"round": 1}]


def test_an_invalid_patch_is_asked_for_once_more_with_its_validation_problems(tmp_path: Path) -> None:
    problem = "notes.txt: not a path of the harness layout (see reef.harness.tree.layout)"
    answers = iter((TreePatch({"notes.txt": "x"}), MARKER))
    seen: list[tuple[str, ...]] = []
    backend, _ = harness_backend(tmp_path, lambda context: seen.append(context.problems) or next(answers))
    result = backend_step(backend, batch(1), backend.initial_state())
    assert seen == [(), (problem,)] and result.metrics["selected"] is True

    # A retry that fails again skips the step with the problems; the proposer is asked twice only.
    asked: list[tuple[str, ...]] = []
    (tmp_path / "again").mkdir()
    again, _ = harness_backend(
        tmp_path / "again", lambda context: asked.append(context.problems) or TreePatch({"notes.txt": "x"})
    )
    result = backend_step(again, batch(1), again.initial_state())
    assert asked == [(), (problem,)]
    assert result.metrics["skipped"] == problem and "selected" not in result.metrics


def test_an_automatic_step_takes_the_oldest_proposal_before_it_asks_the_proposer(tmp_path: Path) -> None:
    asked: list[int] = []
    backend, releases = harness_backend(tmp_path, lambda context: asked.append(context.step), reads_requests=True)
    inbox = backend.proposals
    assert inbox is not None

    def submit(patch: TreePatch) -> str:
        proposal_id = ProposalInbox.new_id()
        body = {"patch": patch.to_json(), "reason": "r", "session": "s", "release_id": "base"}
        assert inbox.submit(proposal_id, body) is None
        return proposal_id

    first_id, stale_id = submit(MARKER), submit(MARKER)
    first = backend_step(backend, batch(1), backend.initial_state())
    proposal = first.metrics["proposal"]
    assert proposal == {"id": first_id, "session": "s", "release_id": "base", "reason": "r"}
    assert first.metrics["selected"] is True
    # A client's proposal is no outcome of the method's: the outcome the method folds in stays as it was.
    assert state_of(first).last_outcome is None
    settled = json.loads((inbox.directory / "settled" / f"{first_id}.json").read_text(encoding="utf-8"))
    assert settled["result"] == {"step": 1, "selected": True, "reason": first.metrics["selection"]["reason"]}
    published(releases, "r1", first)

    # The step validates the claimed patch again, against the head it builds on.
    second = backend_step(backend, batch(2), first.state)
    assert second.metrics["proposal"]["id"] == stale_id and second.metrics["skipped"] == "the patch changes nothing"
    refused = json.loads((inbox.directory / "refused" / f"{stale_id}.json").read_text(encoding="utf-8"))
    assert refused["refused"] == "the patch changes nothing"

    # An instruction step answers its request and leaves the inbox to the next automatic step.
    waiting = submit(PLAIN)
    third = backend_step(backend, batch(3, request=REQUEST), second.state)
    assert asked == [3] and third.metrics["skipped"] == "no proposal" and inbox.pending() == [waiting]
    fourth = backend_step(backend, batch(4), third.state)
    assert fourth.metrics["proposal"]["id"] == waiting and asked == [3] and inbox.pending() == []


def test_an_aborted_step_refuses_its_claimed_proposal(tmp_path: Path) -> None:
    backend, _ = harness_backend(tmp_path, lambda context: None)
    proposal_id = ProposalInbox.new_id()
    backend.proposals.submit(proposal_id, {"patch": MARKER.to_json(), "reason": "", "session": "", "release_id": ""})
    prepared = backend.prepare_step(batch(1), backend.initial_state(), 0)
    backend.abort_step(prepared)
    refused = json.loads((backend.proposals.directory / "refused" / f"{proposal_id}.json").read_text(encoding="utf-8"))
    assert refused["refused"] == "step aborted before a result"
    assert backend.failed_step_metrics() == {"failed_stage": "evaluating", "patch": MARKER.to_json()}


@pytest.mark.parametrize(
    ("fields", "patch", "pending"),
    [
        ({"publish": "review"}, MARKER, True),
        (
            {"review_kinds": (TreeKind.CODE_EXTENSION,)},
            TreePatch({**MARKER.changes, "extensions/notify.ts": "export default function (pi) {}\n"}),
            True,
        ),
        ({"review_kinds": (TreeKind.CODE_EXTENSION,)}, MARKER, False),
    ],
    ids=["publish-review", "reviewed-kind", "other-kind"],
)
def test_a_win_waits_for_review_under_publish_review_or_when_it_changes_a_reviewed_kind(
    tmp_path: Path, fields: dict, patch: TreePatch, pending: bool
) -> None:
    backend, _ = harness_backend(tmp_path, lambda context: patch, **fields)
    result = backend_step(backend, batch(1), backend.initial_state())
    assert result.metrics["selected"] is True and result.artifact is not None and result.pending is pending


def test_a_native_loop_change_always_waits_for_review(tmp_path: Path) -> None:
    backend, _ = harness_backend(tmp_path, lambda context: None)
    assert backend.held_for_review(TreePatch({"native/loops/main.py": "NAME = 'main'\n"}))
    assert not backend.held_for_review(MARKER)


def test_a_recheck_evaluates_the_replaced_tree_against_the_head_and_keeps_the_head_when_it_loses(
    tmp_path: Path,
) -> None:
    asked: list[int] = []
    backend, releases = harness_backend(
        tmp_path, lambda context: asked.append(context.step) or MARKER, recheck_every=2
    )
    first = backend_step(backend, batch(1), backend.initial_state())
    assert state_of(first).rollback_release_id == "base"
    assert state_of(first).rollback_context == first.metrics["evaluation_context"]
    published(releases, "r1", first)

    second = backend_step(backend, batch(2), first.state)
    assert asked == [1]
    assert second.metrics["recheck"] is True and second.metrics["recheck_reason"] == "cadence"
    assert second.metrics["recheck_release_id"] == "base" and second.metrics["proposer_calls"] == 0
    # The candidate is the replaced tree: the patch from the head back to it.
    assert second.metrics["head_release_id"] == "r1" and second.metrics["patch"] == changed_rules("Answer briefly.\n")
    assert second.metrics["rolled_back"] is False and second.artifact is None
    state = state_of(second)
    # A recheck is no proposal: the streak, the rejections and the method's latest outcome stay as they were.
    assert state.failure_streak == 0 and state.rejected == () and state.last_outcome == state_of(first).last_outcome
    assert state.rollback_release_id == "base"


def test_a_recheck_rolls_back_when_the_replaced_tree_wins(tmp_path: Path) -> None:
    prefer_marker = [True]

    def score(task, result):
        return 1.0 if ("marker" in result.trajectory[-1]["rules"]) == prefer_marker[0] else 0.0

    backend, releases = harness_backend(
        tmp_path,
        lambda context: MARKER,
        score=score,
        recheck_every=2,
        review_kinds=(TreeKind.RULES,),
    )
    first = backend_step(backend, batch(1), backend.initial_state())
    assert first.metrics["selected"] is True and first.pending is True
    published(releases, "r1", first)
    prefer_marker[0] = False
    second = backend_step(backend, batch(2), first.state)
    assert second.metrics["recheck"] is True and second.metrics["rolled_back"] is True
    assert HarnessTree.read(second.artifact.local_path) == PI_SEED
    # The rollback serves a tree published before, so only publish: review holds it.
    assert second.pending is False
    state = state_of(second)
    assert state.rollback_release_id is None and state.rollback_context is None
    assert state.failure_streak == 0 and state.last_outcome == state_of(first).last_outcome


def test_a_recheck_runs_off_cadence_when_the_evaluation_context_drifted(tmp_path: Path) -> None:
    backend, releases = harness_backend(tmp_path, lambda context: None, recheck_every=2)
    releases.publish("r1", PI_SEED.apply(MARKER))
    stale = {**backend.evaluation_context(), "model": "an-older-model"}
    state = HarnessState(steps=2, rollback_release_id="base", rollback_context=stale).to_json()
    result = backend_step(backend, batch(3), state)
    assert result.metrics["recheck"] is True and result.metrics["recheck_reason"] == "drift"
    assert result.metrics["rolled_back"] is False


def test_the_step_record_keeps_the_proposer_calls_and_the_patch(tmp_path: Path) -> None:
    backend, _ = harness_backend(tmp_path, lambda context: MARKER, step_record_dir=str(tmp_path / "steps"))
    result = backend_step(backend, batch(1), backend.initial_state())
    step_dir = Path(result.metrics["step_record"])
    assert step_dir == (tmp_path / "steps" / "demo" / "1").resolve()
    assert json.loads((step_dir / "proposer.json").read_text(encoding="utf-8")) == []
    assert json.loads((step_dir / "patch.json").read_text(encoding="utf-8")) == MARKER.to_json()
    listed = {entry["path"] for entry in backend.read_step_records(str(step_dir), None)["files"]}
    assert {"patch.json", "proposer.json"} <= listed


def test_the_progress_follows_the_step_through_its_phases(tmp_path: Path) -> None:
    seen: list[tuple[str, str | None, int | None]] = []
    holder: dict = {}

    def propose(context):
        progress = holder["backend"].step_progress
        seen.append((progress.phase, progress.request_id, progress.episodes_total))
        return MARKER

    def score(task, result):
        progress = holder["backend"].step_progress
        seen.append((progress.phase, progress.request_id, progress.episodes_total))
        return 1.0

    backend, _ = harness_backend(tmp_path, propose, score=score, reads_requests=True)
    holder["backend"] = backend
    backend_step(backend, batch(1, request=REQUEST), backend.initial_state())
    assert seen == [("proposing", "ask-1", None), ("evaluating", "ask-1", 2), ("evaluating", "ask-1", 2)]
    assert backend.step_progress is None


def test_the_step_and_failure_streak_limits_skip_automatic_steps_only(tmp_path: Path) -> None:
    backend, _ = harness_backend(tmp_path, lambda context: MARKER, reads_requests=True, max_steps=1)
    first = backend_step(backend, batch(1), backend.initial_state())
    second = backend_step(backend, batch(2), first.state)
    assert second.metrics["skipped"] == "step budget of 1 exhausted" and state_of(second).steps == 2
    answered = backend_step(backend, batch(3, request=REQUEST), second.state)
    assert "skipped" not in answered.metrics

    (tmp_path / "streak").mkdir()
    streak, _ = harness_backend(tmp_path / "streak", lambda context: PLAIN, max_failure_streak=1)
    first = backend_step(streak, batch(1), streak.initial_state())
    assert state_of(first).failure_streak == 1
    second = backend_step(streak, batch(2), first.state)
    assert second.metrics["skipped"] == "failure streak of 1 rejections"


def test_the_context_carries_the_step_the_head_the_samples_and_the_methods_state(tmp_path: Path) -> None:
    contexts = []

    def propose(context):
        contexts.append(context)
        return TreeProposal(MARKER if context.step == 1 else PLAIN, method_state={"seen": context.step})

    backend, releases = harness_backend(tmp_path, propose)
    sample = recorded_trajectory("i1", {"messages": [{"role": "user", "content": "q"}]}, 0.5)
    first = backend_step(backend, batch(1, sample), backend.initial_state())
    head = published(releases, "r1", first)
    backend_step(backend, batch(2), first.state)

    one, two = contexts
    assert (one.step, one.tree, one.descriptor.name, one.request) == (1, PI_SEED, "pi", None)
    assert one.evaluation_tasks == ("task one",) and one.samples == (sample,)
    assert one.sources == (SampleSource("i1", "untagged"),)
    assert one.last_outcome is None and dict(one.method_state) == {} and one.manifest is None
    assert one.models.served.model == "qwen3-8b"
    assert (two.step, two.tree, two.samples) == (2, head, ())
    assert two.last_outcome == state_of(first).last_outcome and two.last_outcome.step == 1
    assert dict(two.method_state) == {"seen": 1} and two.manifest is not None


def test_a_trial_scores_a_tree_on_the_evaluations_episodes(tmp_path: Path) -> None:
    scores: list[float | None] = []

    def propose(context):
        scores.extend(episode.score for episode in context.trials.run(PI_SEED.apply(MARKER), ("one", "two")))
        return

    backend, _ = harness_backend(tmp_path, propose)
    backend_step(backend, batch(1), backend.initial_state())
    assert scores == [1.0, 1.0]


def test_an_instruction_step_needs_a_proposer_that_reads_requests(tmp_path: Path) -> None:
    backend, _ = harness_backend(tmp_path, lambda context: MARKER)
    with pytest.raises(ValueError, match="reads_requests is true"):
        backend_step(backend, batch(1, request=REQUEST), backend.initial_state())
