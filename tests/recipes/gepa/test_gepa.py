"""Guarantees of the recipes/gepa method, hermetic: GEPA's archive rules, its reflection prompt, and whole steps.

No reflection model is reached: a fake binding answers. A step runs through
the phases the trainer runs (``backend_step``), decided by the ``compare``
policy on the mean as a GEPA deployment decides it. Its tasks run a fake
pi binary through the real adapter path: the binary answers with the rules it
was given, and a prompt that holds ``CRASH`` leaves a session that cannot be
read, so that run has no score. The scorer prefers rules with the marker.
"""

from __future__ import annotations

import dataclasses
import json
import random
import sys
from pathlib import Path
from typing import Any

import pytest
from tests.support._trajectories import recorded_trajectory
from tests.support.harness_service import backend_step
from tests.support.harness_steps import COMPONENT, TreeReleases, harness_recipe

from recipes.gepa import GEPA, GEPASettings, ScoreFeedback, components, reflection
from recipes.gepa.archive import Archive, Candidate
from recipes.gepa.method import ARCHIVE_FILE, archive_for_step
from reef.core.evaluation import RolloutScore
from reef.core.job_data import TrajectorySample
from reef.core.plugins import PluginConfigError
from reef.evaluation.harness import ComparePolicy, DecisionRecord, FailureObservation, History, RunRecord
from reef.harness.adapters import get_adapter
from reef.harness.tasks.model_binding import ModelBinding, ModelBindings
from reef.harness.tasks.run import TaskResult
from reef.harness.tree import HarnessTree
from reef.train.backend import ComponentReleases
from reef.train.harness_backend.backend import HarnessBackend
from reef.train.harness_backend.state import HarnessState
from reef.train.types import JobData, TrainStepResult

MARKER = "SHOW-YOUR-WORK"
SEED_TEXT = "Answer briefly."
REFLECTED = f"Answer briefly. {MARKER}"
FENCED_REPLY = f"Here you go:\n```\n{REFLECTED}\n```\n"
NO_MARKER_REPLY = "```\nNo marker here.\n```"

#: The seed tree. The skill proves that ``components`` narrows what GEPA may rewrite, and the config file that GEPA
#: never rewrites structure.
TREE = HarnessTree({"config/primary.json": "{}\n", "rules.md": SEED_TEXT, "skills/notes/SKILL.md": "# notes\n"})
#: The step's evaluation tasks, GEPA's validation set. The second cannot run on any tree.
TASKS = ("check one", "CRASH check two")
TASK = "what is 2+2?"

#: A fake pi whose answer is the rules it was given. A prompt with CRASH writes a line that is not JSON before the
#: answer, so the session cannot be read.
PI_FAKE = (
    f"#!{sys.executable}\n"
    """\
import json, os, sys
from pathlib import Path

prompt = sys.argv[sys.argv.index("-p") + 1]
rules = (Path(sys.argv[sys.argv.index("-e") + 1]) / "rules.md").read_text()
sessions = Path(os.environ["PI_CODING_AGENT_SESSION_DIR"])
sessions.mkdir(parents=True, exist_ok=True)
event = {"type": "message", "role": "assistant", "content": rules}
(sessions / "session.jsonl").write_text(("not json\\n" if "CRASH" in prompt else "") + json.dumps(event) + "\\n")
"""
)


def score_rules(task: str, result: TaskResult) -> float:
    """1.0 when the tree that ran carried the marker."""
    return 1.0 if MARKER in result.trajectory[-1]["content"] else 0.0


class FakeChat(ModelBinding):
    """A reflection model that answers one reply and keeps the prompts it was asked."""

    def __init__(self, reply: str = FENCED_REPLY) -> None:
        super().__init__(base_url="http://127.0.0.1:9", model="reflection", api_key="dummy")
        object.__setattr__(self, "reply", reply)
        object.__setattr__(self, "prompts", [])

    def chat(self, messages, **params):
        self.prompts.append(messages[0]["content"])
        return self.reply


def gepa_backend(
    tmp_path: Path, settings: GEPASettings, *, reflector: ModelBinding | None = None
) -> tuple[HarnessBackend, TreeReleases]:
    """The backend of a pi scenario that GEPA evolves from ``TREE``, its reflection model ``reflector``."""
    binary = tmp_path / "gepa-pi"
    binary.write_text(PI_FAKE)
    binary.chmod(0o755)
    recipe = harness_recipe(tmp_path, GEPA(settings), seed=TREE, score=score_rules, tasks=TASKS, binary=str(binary))
    recipe = dataclasses.replace(recipe, models={"reflection": FakeChat() if reflector is None else reflector})
    releases = TreeReleases(recipe.seed)
    return recipe.build_backend("demo", ComponentReleases(releases, COMPONENT), None), releases


def step(backend: HarnessBackend, number: int, state: dict[str, Any], *samples: TrajectorySample) -> TrainStepResult:
    """Step ``number`` on ``samples``, decided by the compare policy on the mean."""
    policy = ComparePolicy(backend, statistic="mean")
    return backend_step(backend, JobData(f"demo:{number}", samples), state, policy, scenario_step=number - 1)


def sample(name: str, score: float, prompt: str = TASK) -> TrajectorySample:
    """One served exchange: ``prompt``, the answer ``4``, and the reported ``score``."""
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "response": {"choices": [{"message": {"role": "assistant", "content": "4"}}]},
    }
    return recorded_trajectory(name, payload, score)


def committed(archive: Archive) -> dict[str, Any]:
    """The backend state of a step whose method committed ``archive``."""
    return HarnessState(method_files={ARCHIVE_FILE: json.dumps(archive.to_state())}).to_json()


def archive_of(result: TrainStepResult) -> Archive:
    """The archive a step committed."""
    return Archive.from_state(json.loads(HarnessState.from_json(result.state).method_files[ARCHIVE_FILE]))


RULES_ONLY = GEPASettings(components=("rules",))
PI_LAYOUT = get_adapter("pi").layout


# -- components: the tree as a GEPA candidate ----------------------------------


def test_texts_of_keys_the_evolvable_files_by_path_and_refuses_a_part_that_is_not_text() -> None:
    assert components.texts_of(TREE, PI_LAYOUT, ("rules", "skill")) == {
        "rules.md": SEED_TEXT,
        "skills/notes/SKILL.md": "# notes\n",
    }
    assert components.texts_of(TREE, PI_LAYOUT, ("rules",)) == {"rules.md": SEED_TEXT}
    with pytest.raises(ValueError, match="GEPA components must name text parts of the tree: rules, skill, command"):
        components.texts_of(TREE, PI_LAYOUT, ("extension",))


def test_a_candidate_tree_is_the_served_tree_with_the_candidates_texts() -> None:
    texts = {"rules.md": REFLECTED, "skills/notes/SKILL.md": "# better\n", "skills/absent/SKILL.md": "unreachable"}
    # Every other file stays, and a component the served tree does not hold is never invented.
    assert components.candidate_tree(TREE, texts) == HarnessTree(
        {**TREE.files, "rules.md": REFLECTED, "skills/notes/SKILL.md": "# better\n"}
    )
    assert components.candidate_tree(TREE, components.texts_of(TREE, PI_LAYOUT, ("rules",))) == TREE


# -- reflection: the copied prompt and extractor --------------------------------


def test_the_prompt_carries_the_current_text_and_the_records() -> None:
    prompt = reflection.render_prompt(SEED_TEXT, [{"Inputs": TASK, "Generated Outputs": "4", "Feedback": "correct"}])
    assert "<curr_param>" not in prompt and "<side_info>" not in prompt
    assert SEED_TEXT in prompt
    assert "# Example 1\n## Inputs\nwhat is 2+2?\n\n## Generated Outputs\n4\n\n## Feedback\ncorrect\n\n" in prompt
    assert prompt.endswith("Provide the new instructions within ``` blocks.")


def test_the_prompt_nests_structured_feedback_as_markdown_headers() -> None:
    prompt = reflection.render_prompt("x", [{"Feedback": {"why": {"detail": "off by one"}}}])
    assert "## Feedback\n### why\n#### detail\noff by one" in prompt


def test_the_extractor_reads_fenced_and_unfenced_replies() -> None:
    assert reflection.extract_new_text(FENCED_REPLY) == REFLECTED
    assert reflection.extract_new_text(f"```text\n{REFLECTED}\n```") == REFLECTED
    assert reflection.extract_new_text(REFLECTED) == REFLECTED
    assert reflection.extract_new_text(f"```\n{REFLECTED}") == REFLECTED  # never closed
    assert reflection.extract_new_text(f"{REFLECTED}\n```") == REFLECTED  # never opened


# -- archive: fronts, parent sampling, the committed state ----------------------


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
    # A state that breaks the archive's invariants is refused when it is read.
    with pytest.raises(ValueError, match="must index one of the 2 candidates"):
        Archive.from_state({**archive.to_state(), "served": 2})
    with pytest.raises(ValueError, match="set exactly when"):
        Archive.from_state({**archive.to_state(), "pending_step": None})


def test_the_plan_is_a_function_of_the_committed_state_and_walks_epochs() -> None:
    """A driver that reads the committed state draws the plan the method draws, and an epoch shows every
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
    archive.settle([1.0, 0.0], [0.0, 0.0], selected=True, repeats=1)
    assert archive.candidates[1].val_scores == [1.0, 0.0]
    # The seed's first validation arrives with the first evaluated step; it was charged when it was seeded.
    assert archive.candidates[0].val_scores == [0.0, 0.0]
    assert (archive.served, archive.pending, archive.pending_step, archive.metric_calls) == (1, None, None, 2)
    # A rejected candidate keeps its scores, and with them its fronts: it can still parent proposals.
    archive.add({"rules.md": "third"}, 1, [1.0], step=2)
    archive.settle([0.0, 1.0], [1.0, 0.0], selected=False, repeats=1)
    assert (archive.served, archive.candidates[2].val_scores, archive.candidates[1].val_scores) == (
        1,
        [0.0, 1.0],
        [1.0, 0.0],
    )


def test_with_repeats_a_tasks_validation_is_the_mean_of_its_runs_and_each_run_is_charged(
    tmp_path: Path,
) -> None:
    """Two runs per task and tree: a task's validation score is their mean, with a run that could not run
    as a zero, and the seed's and the candidate's validation passes each cost one metric call per run."""
    seed, child = HarnessTree({"rules.md": SEED_TEXT}), HarnessTree({"rules.md": REFLECTED})
    history = History.open(tmp_path / "history")
    # The scores task by task and repeat by repeat; one run of the child could not run.
    for tree, parent, values in ((seed, None, (0.0, 0.0, 1.0, 1.0)), (child, seed, (1.0, 0.0, None, 1.0))):
        if parent is None:
            history.add_candidate(tree, step=1, origin="seed")
        else:
            history.add_candidate(tree, step=1, origin="method", parent=parent.checksum())
        runs = [(task, repeat) for task in TASKS for repeat in range(2)]
        for (task, repeat), score in zip(runs, values, strict=True):
            failure = None if score is not None else FailureObservation(task, "launch", "the binary is missing")
            task_run, _ = history.new_run()
            record = RunRecord(
                task_run, tree.checksum(), task, task, task, repeat, "ctx", 1, RolloutScore(score), failure
            )
            history.add_run(record)
    history.add_decision(DecisionRecord(1, child.checksum(), seed.checksum(), True, "compare", "a higher mean"))

    seeded = archive_for_step(None, history, TASKS, seed.files, repeats=2)
    seeded.add(child.files, 0, [1.0], step=1)
    settled = archive_for_step(json.dumps(seeded.to_state()), history, TASKS, child.files, repeats=2)
    assert (settled.candidates[1].val_scores, settled.candidates[0].val_scores) == ([0.5, 0.5], [0.0, 1.0])
    # Each validation pass ran 2 tasks with 2 runs each: the seed's, then the candidate's.
    assert (seeded.metric_calls, settled.metric_calls, settled.served) == (4, 8, 1)


# -- settings ---------------------------------------------------------------------


def test_the_config_parses_into_settings_and_builds_the_feedback() -> None:
    config = {
        "minibatch_size": 2,
        "seed": 7,
        "skip_perfect_score": False,
        "perfect_score": 0.5,
        "max_metric_calls": 150,
        "components": ["rules"],
        "trainset_size": 45,
        "feedback": "recipes.gepa:ScoreFeedback",
    }
    method = GEPA.from_config(config)
    assert method.settings == GEPASettings(**{**config, "components": ("rules",)})
    assert isinstance(method.feedback, ScoreFeedback)
    # Without a feedback setting the reflection reads the score restated.
    assert isinstance(GEPA.from_config({}).feedback, ScoreFeedback)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"components": ["rules", "rules"]}, "components must name one or more distinct parts"),
        ({"components": []}, "components must name one or more distinct parts"),
        ({"archive": "gepa"}, "unknown config fields: archive"),
        ({"minibatch_size": 0}, "minibatch_size must be positive"),
        ({"seed": "zero"}, "GEPA.seed must be a valid int"),
        ({"feedback": "reef.evaluation.harness:VerifierRewardScorer"}, "inherits Feedback"),
    ],
)
def test_an_invalid_config_is_refused(config: dict[str, Any], message: str) -> None:
    with pytest.raises(PluginConfigError, match=message):
        GEPA.from_config(config)


# -- whole steps through the real backend ------------------------------------------


def committed_parent(text: str = "Parent rules.") -> Archive:
    """An archive whose validated candidate 1 dominates the served seed, so it is always the parent."""
    archive = Archive()
    archive.seed({"rules.md": SEED_TEXT})
    archive.add({"rules.md": text}, 0, [1.0], step=1)
    archive.record_validation(0, [0.0, 0.0])
    archive.record_validation(1, [1.0, 1.0])
    archive.clear_pending()
    return archive


def test_a_step_reflects_on_the_served_trees_traces_and_the_next_step_settles_its_validation(
    tmp_path: Path,
) -> None:
    reflector = FakeChat()
    backend, releases = gepa_backend(tmp_path, RULES_ONLY, reflector=reflector)
    first = step(backend, 1, backend.initial_state(), sample("a1", 0.0))

    # The served tree's traces are the parent's minibatch, graded by the feedback hook: only the child ran.
    assert first.metrics["method_runs"] == 1
    (prompt,) = reflector.prompts
    assert f"## Inputs\n{TASK}" in prompt and "## Generated Outputs\n4" in prompt
    assert "## Feedback\nThis response scored 0.000." in prompt
    notes = first.metrics["method_notes"]
    assert (notes["parent"], notes["component"], notes["parent_scores"], notes["child_scores"]) == (
        0,
        "rules.md",
        [0.0],
        [1.0],
    )
    assert notes["accepted"] is True and notes["candidate"] == 1
    # GEPA's count: the seed's validation (2 tasks), the served minibatch (1, from the traces) and the child's (1).
    assert notes["metric_calls"] == 4
    # The child is the step's candidate, and its validation mean beats the seed's.
    assert first.metrics["selected"] is True and first.metrics["patch"] == {"rules.md": REFLECTED}
    archive = archive_of(first)
    assert (archive.pending, archive.pending_step) == (1, 1)
    assert archive.candidates[0].texts == {"rules.md": SEED_TEXT}  # the skill stayed out of ``components``
    # The archive and the notes keep the same record of the reflection. The step's calls.json keeps the prompt and
    # the reply, so the archive, which every step commits whole, leaves them out.
    (logged,) = archive.proposals
    assert logged == {"step": 1, **{key: value for key, value in notes.items() if key != "metric_calls"}}
    (call,) = json.loads((Path(first.metrics["step_record"]) / "calls.json").read_text())
    assert (call["messages"][0]["content"], call["reply"]) == (prompt, FENCED_REPLY)

    releases.publish("r1", HarnessTree.read(first.artifact.local_path))
    second = step(backend, 2, first.state, sample("a2", 1.0))
    # The next step folds the validation in from the history; the task that could not run is a zero, not a gap.
    settled = archive_of(second)
    assert (settled.served, settled.pending) == (1, None)
    assert (settled.candidates[1].val_scores, settled.candidates[0].val_scores) == ([1.0, 0.0], [0.0, 0.0])
    # The candidate's validation (2), then the new parent's minibatch (1), which scored perfectly.
    assert second.metrics["method_notes"] == {
        "skipped": "the parent scored perfectly on the minibatch",
        "metric_calls": 7,
    }


def test_a_parent_that_never_served_runs_on_the_minibatch(tmp_path: Path) -> None:
    reflector = FakeChat()
    backend, _ = gepa_backend(tmp_path, RULES_ONLY, reflector=reflector)
    result = step(backend, 1, committed(committed_parent()), sample("a1", 0.0))

    # No traffic was served from candidate 1, so its run is a run, and the child's another.
    assert result.metrics["method_runs"] == 2
    (prompt,) = reflector.prompts
    assert "```\nParent rules.\n```" in prompt and "## Generated Outputs\nParent rules." in prompt
    notes = result.metrics["method_notes"]
    assert (notes["parent"], notes["candidate"], notes["metric_calls"]) == (1, 2, 2)
    assert archive_of(result).candidates[2].parent == 1


def test_a_child_that_does_not_beat_its_parent_is_rejected(tmp_path: Path) -> None:
    backend, _ = gepa_backend(tmp_path, RULES_ONLY, reflector=FakeChat(NO_MARKER_REPLY))
    result = step(backend, 1, backend.initial_state(), sample("a1", 0.0))

    assert result.metrics["skipped"] == "the method left the head tree unchanged"
    assert result.metrics["method_summary"] == "the child scored 0 on the minibatch and its parent 0"
    notes = result.metrics["method_notes"]
    assert (notes["accepted"], notes["child_scores"]) == (False, [0.0])
    archive = archive_of(result)
    assert (archive.rejections, archive.steps, archive.pending, len(archive.candidates)) == (1, 1, None, 1)


def test_a_child_that_fails_validation_ends_the_iteration_without_an_run(tmp_path: Path) -> None:
    # A reflection that writes a credential-shaped literal makes a tree that Reef never evaluates or publishes.
    reply = "```\nAnswer briefly with the key sk-" + "a" * 24 + "\n```"
    backend, _ = gepa_backend(tmp_path, RULES_ONLY, reflector=FakeChat(reply))
    result = step(backend, 1, backend.initial_state(), sample("a1", 0.0))

    assert result.metrics["skipped"] == "the method left the head tree unchanged"
    assert result.metrics["method_runs"] == 0 and "inline credential" in result.metrics["method_summary"]
    notes = result.metrics["method_notes"]
    assert (notes["accepted"], notes["child_scores"], len(notes["problems"])) == (False, None, 1)
    # No child ran, so GEPA spent no iteration on it.
    assert (archive_of(result).steps, archive_of(result).rejections) == (0, 0)


def test_an_run_that_cannot_run_scores_zero_and_its_error_is_the_feedback(tmp_path: Path) -> None:
    reflector = FakeChat()
    backend, _ = gepa_backend(tmp_path, RULES_ONLY, reflector=reflector)
    result = step(backend, 1, committed(committed_parent()), sample("a1", 0.0, prompt=f"CRASH {TASK}"))

    (prompt,) = reflector.prompts
    assert "corrupt event at line 1" in prompt.split("## Feedback\n", 1)[1]
    # The child cannot run the task either, so it ties its parent at zero.
    notes = result.metrics["method_notes"]
    assert (notes["parent_scores"], notes["child_scores"], notes["accepted"]) == ([0.0], [0.0], False)


def test_a_spent_budget_stops_the_search_before_any_run_or_model_call(tmp_path: Path) -> None:
    reflector = FakeChat()
    backend, _ = gepa_backend(tmp_path, GEPASettings(components=("rules",), max_metric_calls=150), reflector=reflector)
    spent = Archive()
    spent.seed({"rules.md": SEED_TEXT})
    spent.charge(150)
    result = step(backend, 1, committed(spent), sample("a1", 0.0))

    assert result.metrics["method_notes"] == {
        "stopped": "the budget of 150 metric calls is spent",
        "metric_calls": 150,
    }
    assert (result.metrics["method_runs"], reflector.prompts) == (0, [])


@pytest.mark.parametrize(("perfect_score", "reflections"), [(1.0, 0), (2.0, 1)])
def test_a_perfect_minibatch_ends_the_iteration_before_any_model_call(
    tmp_path: Path, perfect_score: float, reflections: int
) -> None:
    reflector = FakeChat()
    settings = GEPASettings(components=("rules",), perfect_score=perfect_score)
    backend, _ = gepa_backend(tmp_path, settings, reflector=reflector)
    step(backend, 1, backend.initial_state(), sample("a1", 1.0))
    # The same minibatch is worth reflecting on once the bar is above it.
    assert len(reflector.prompts) == reflections


def test_reflection_falls_back_to_the_model_under_test(tmp_path: Path) -> None:
    backend, _ = gepa_backend(tmp_path, RULES_ONLY)
    served = FakeChat()
    # A deployment that declares no reflection model gives the method the model under test alone.
    backend.models = ModelBindings(served=served)
    result = step(backend, 1, backend.initial_state(), sample("a1", 0.0))
    assert len(served.prompts) == 1 and result.metrics["method_notes"]["accepted"] is True


def test_a_tree_the_archive_does_not_hold_becomes_a_new_root(tmp_path: Path) -> None:
    """An operator edit or a rollback is a new root, never a descendant."""
    elsewhere = Archive()
    elsewhere.seed({"rules.md": "something else"})
    backend, _ = gepa_backend(tmp_path, RULES_ONLY)
    archive = archive_of(step(backend, 1, committed(elsewhere), sample("a1", 0.0)))

    assert (archive.candidates[1].texts, archive.candidates[1].parent) == ({"rules.md": SEED_TEXT}, None)
    assert archive.candidates[2].parent == 1  # the step's child descends from the new root


def test_a_pending_candidate_whose_step_reached_no_decision_is_dropped(tmp_path: Path) -> None:
    pending = Archive()
    pending.seed({"rules.md": SEED_TEXT})
    pending.add({"rules.md": REFLECTED}, 0, [1.0], step=3)
    backend, _ = gepa_backend(tmp_path, RULES_ONLY, reflector=FakeChat(NO_MARKER_REPLY))
    # Step 3 left a candidate pending, and the history holds no decision on it: the backend skipped that step.
    archive = archive_of(step(backend, 4, committed(pending), sample("a1", 0.0)))
    assert (archive.pending, archive.served, archive.candidates[1].val_scores) == (None, 0, None)
