"""Guarantees of the evolve step record, hermetic: the proposer's model calls,
the proposed patch, and every evaluation episode's trajectory land where a
reader can rebuild the decision, and nothing lands when the record is off. A
fake native binary writes a session tree shaped like the native loop's, so
the stage path metric is checked against a kept file; the served model is a
stub endpoint behind ``urllib.request.urlopen``."""

from __future__ import annotations

import io
import json
import tarfile
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import pytest
from tests.support._strategies import resolve_episode_scorer, resolve_proposer
from tests.support.harness_service import backend_step
from tests.support.harness_steps import (
    COMPONENT,
    MODEL,
    TreeReleases,
    changed_rules,
    harness_backend,
    harness_recipe,
    make_pi_binary,
    runtime,
)

from reef.evaluation.harness import EpisodeEvaluationWorker
from reef.harness.adapters import get_adapter
from reef.harness.episodes.run import EpisodeError, TrajectoryKeepError, run_episode
from reef.harness.tree import HarnessTree, TreePatch, render_tree
from reef.harness.tree.checks import RECORD_TEXT_CAP
from reef.recipe import RecipeConfigError
from reef.recipe.harness import HarnessEvolutionRecipe
from reef.runtime.executor.config import ExecutorSettings
from reef.sandbox.process import LocalExecutor
from reef.storage.sqlite import SQLiteRecordStore
from reef.train.backend import ComponentReleases
from reef.train.harness_backend.backend import HarnessBackend
from reef.train.harness_backend.records import StepRecord
from reef.train.types import JobData, TrainStepResult

# A harness shaped like the native loop: the root session, one delegated checker
# turn under agents/, and the rendered rules echoed into the header for the scorer.
NATIVE_FAKE = """\
#!/usr/bin/env python3
import json, os
from pathlib import Path

root = Path(os.environ["REEF_NATIVE_DIR"])
sessions = Path(os.environ["REEF_NATIVE_SESSION_DIR"])
(sessions / "agents").mkdir(parents=True, exist_ok=True)
rules = (root / "RULES.md").read_text() if (root / "RULES.md").exists() else ""


def event(type_, data):
    return json.dumps({"type": type_, "seq": 0, "time": 0, "data": data}) + "\\n"


checker = [
    event("session", {"agent": "checker", "turn": 2, "parent": "root"}),
    event("stage/enter", {"stage": "verify", "kind": "verify"}),
    event("stage/exit", {"stage": "verify", "outcome": "pass", "to": "done"}),
    event("turn/end", {"turn": 2, "reason": {"kind": "gave_up"}}),
]
(sessions / "agents" / "002-checker.jsonl").write_text("".join(checker))
root_turn = [
    event("session", {"agent": "root", "turn": 1, "rules": rules}),
    event("stage/enter", {"stage": "think", "kind": "model"}),
    event("stage/exit", {"stage": "think", "outcome": "tool_calls", "to": "act"}),
    event("stage/enter", {"stage": "act", "kind": "tools"}),
    event("stage/exit", {"stage": "act", "outcome": "done", "to": "think"}),
    event("stage/enter", {"stage": "think", "kind": "model"}),
    event("stage/exit", {"stage": "think", "outcome": "text", "to": "done"}),
    event("turn/end", {"turn": 1, "reason": {"kind": "completed"}}),
]
(sessions / "session.jsonl").write_text("".join(root_turn))
"""

MARKER = TreePatch(changed_rules("marker rules\n"))
NATIVE_SEED = HarnessTree({"rules.md": "Answer briefly.\n"})
STAGE_PATH = {"stages": ["think", "act", "think"], "reason": "completed"}


def make_native_binary(tmp_path: Path) -> Path:
    binary = tmp_path / "fake-native"
    binary.write_text(NATIVE_FAKE)
    binary.chmod(0o755)
    return binary


def native_score(task: str, result) -> float:
    return (
        1.0 if any("marker" in str((event.get("data") or {}).get("rules", "")) for event in result.trajectory) else 0.0
    )


def native_backend(
    tmp_path: Path, propose=lambda context: MARKER, *, binary: str | None = None, **fields
) -> tuple[HarnessBackend, TreeReleases]:
    """The native backend over the fake native binary, and its in-memory history whose base is the native seed."""
    recipe = replace(
        harness_recipe(tmp_path, propose, seed=NATIVE_SEED, score=native_score, **fields),
        adapter="native",
        binary=binary or str(make_native_binary(tmp_path)),
    )
    releases = TreeReleases(recipe.seed)
    return recipe.build_backend("demo", ComponentReleases(releases, COMPONENT), None), releases


def step(backend: HarnessBackend, state=None) -> TrainStepResult:
    return backend_step(backend, JobData("demo:1"), backend.initial_state() if state is None else state)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def served_model(monkeypatch):
    """The served model's endpoint: each call answers the next reply, the last one again once they run out.

    A reply is a string, or an exception the call raises; ``usage`` rides every answer when set.
    """
    endpoint = {"replies": ["ok"], "usage": None}

    def urlopen(request, timeout=None):
        replies = endpoint["replies"]
        reply = replies.pop(0) if len(replies) > 1 else replies[0]
        if isinstance(reply, Exception):
            raise reply
        if reply == "echo":
            reply = "raw " + json.loads(request.data)["messages"][-1]["content"]
        answer = {"choices": [{"message": {"role": "assistant", "content": reply}}]}
        if endpoint["usage"] is not None:
            answer["usage"] = endpoint["usage"]
        return io.BytesIO(json.dumps(answer).encode())

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    return endpoint


def test_remote_worker_returns_records_without_touching_driver_path(tmp_path):
    blocker = tmp_path / "driver-only"
    blocker.write_text("this path cannot be a directory on the worker")
    worker = EpisodeEvaluationWorker(
        get_adapter("native"),
        resolve_episode_scorer(native_score),
        str(make_native_binary(tmp_path)),
        10,
        LocalExecutor(),
        False,
        transfer_records=True,
    )
    scored = worker.run({}, "one", blocker / "episode")
    assert scored.record_archive is not None
    assert blocker.read_text() == "this path cannot be a directory on the worker"
    with tarfile.open(fileobj=BytesIO(scored.record_archive), mode="r:gz") as archive:
        assert {"./session.jsonl", "./agents/002-checker.jsonl", "./episode.json"} <= set(archive.getnames())


# -- the proposer's calls ---------------------------------------------------


def test_record_records_every_proposer_call_through_the_binding_seam(tmp_path: Path, served_model) -> None:
    """Without a call budget the proposer still sees a recording binding, and
    every chat lands with its model, request, reply and time, long text cut
    at the cap with a marker naming what was dropped."""
    long_prompt = "x" * (RECORD_TEXT_CAP + 5)

    def propose(context):
        context.models.served.chat([{"role": "user", "content": "first"}], timeout_s=5.0, max_tokens=8)
        context.models.served.chat([{"role": "system", "content": long_prompt}, {"role": "user", "content": "second"}])
        return MARKER

    backend, _ = harness_backend(tmp_path, propose, step_record_dir=str(tmp_path / "record"))
    result = step(backend)

    assert result.metrics["proposer_calls"] == 2
    assert result.metrics["proposer_seconds"] >= 0.0
    # An answer that reports no usage counts zero tokens rather than none.
    assert result.metrics["proposer_input_tokens"] == 0 and result.metrics["proposer_output_tokens"] == 0
    record = (tmp_path / "record" / "demo" / "1").resolve()
    assert result.metrics["step_record"] == str(record)
    calls = read_json(record / "proposer.json")
    assert [call["model"] for call in calls] == [MODEL.model, MODEL.model]
    assert calls[0]["messages"] == [{"role": "user", "content": "first"}]
    assert calls[0]["params"] == {"max_tokens": 8, "timeout_s": 5.0}
    assert calls[0]["reply"] == "ok" and calls[0]["seconds"] >= 0.0
    assert calls[1]["params"] == {} and calls[1]["messages"][1] == {"role": "user", "content": "second"}
    clipped = calls[1]["messages"][0]["content"]
    assert clipped.startswith("x" * RECORD_TEXT_CAP) and clipped.endswith("... [clipped 5 chars]")
    assert read_json(record / "patch.json") == MARKER.to_json()

    def wordy(context):
        context.models.served.chat([{"role": "user", "content": "go"}], stop=["z" * (RECORD_TEXT_CAP + 2)])
        return MARKER

    (tmp_path / "params").mkdir()
    backend, _ = harness_backend(tmp_path / "params", wordy, step_record_dir=str(tmp_path / "params" / "record"))
    step(backend)
    stop = read_json(tmp_path / "params" / "record" / "demo" / "1" / "proposer.json")[0]["params"]["stop"][0]
    assert stop.startswith("z" * RECORD_TEXT_CAP) and stop.endswith("... [clipped 2 chars]")


def test_record_keeps_a_failed_call_and_the_calls_before_a_budget_stop(tmp_path: Path, served_model) -> None:
    served_model["replies"] = [RuntimeError("endpoint down")]

    def propose(context):
        try:
            context.models.served.chat([{"role": "user", "content": "hi"}])
        except RuntimeError:
            return

    backend, _ = harness_backend(tmp_path, propose, step_record_dir=str(tmp_path / "broken"))
    result = step(backend)
    assert result.metrics["skipped"] == "no proposal" and result.metrics["proposer_calls"] == 1
    record = tmp_path / "broken" / "demo" / "1"
    calls = read_json(record / "proposer.json")
    assert calls[0]["error"] == "RuntimeError: endpoint down" and "reply" not in calls[0]
    assert calls[0]["seconds"] >= 0.0
    # No proposal, no patch on file.
    assert not (record / "patch.json").exists()

    served_model["replies"] = ["ok"]

    def greedy(context):
        while True:
            context.models.served.chat([{"role": "user", "content": "more"}])

    (tmp_path / "greedy").mkdir()
    backend, _ = harness_backend(
        tmp_path / "greedy", greedy, max_model_calls_per_step=2, step_record_dir=str(tmp_path / "budget")
    )
    with pytest.raises(RuntimeError, match="model call budget of 2"):
        step(backend)
    # The stop raised out of propose, and the two calls before it are on file.
    assert len(read_json(tmp_path / "budget" / "demo" / "1" / "proposer.json")) == 2
    assert not (tmp_path / "budget" / "demo" / "1" / "patch.json").exists()


def test_a_rejected_removal_keeps_its_null_in_the_record_the_metrics_and_the_state(tmp_path: Path) -> None:
    removal = TreePatch({"rules.md": None})
    backend, _ = harness_backend(tmp_path, lambda context: removal, step_record_dir=str(tmp_path / "record"))
    result = step(backend)
    assert result.metrics["selected"] is False
    assert result.metrics["patch"] == {"rules.md": None}
    assert result.state["rejected_proposals"][0]["patch"] == {"rules.md": None}
    assert read_json(tmp_path / "record" / "demo" / "1" / "patch.json") == {"rules.md": None}


# -- the step's files and the stage path -----------------------------------


@pytest.mark.parametrize("workers", [1, 2])
def test_step_record_writes_the_patch_and_every_episode_and_the_path_matches_the_kept_trajectory(
    tmp_path: Path,
    workers: int,
) -> None:
    backend, _ = native_backend(
        tmp_path,
        step_record_dir=str(tmp_path / "record"),
        episode_repeats=2,
        worker_executor=ExecutorSettings(workers=workers),
    )
    result = step(backend)
    backend.close()
    assert result.metrics["selected"] is True

    record = tmp_path / "record" / "demo" / "1"
    assert read_json(record / "proposer.json") == []
    assert read_json(record / "patch.json") == MARKER.to_json()
    episodes = record / "episodes"
    names = ["candidate-0", "candidate-0-1", "current-0", "current-0-1"]
    assert sorted(path.name for path in episodes.iterdir()) == names
    for name in names:
        assert (episodes / name / "session.jsonl").is_file()
        assert (episodes / name / "agents" / "002-checker.jsonl").is_file()

    # The path metric is what the kept file says: the root's stage/exit names and how its turn ended.
    kept = [json.loads(line) for line in (episodes / "candidate-0" / "session.jsonl").read_text().splitlines()]
    stages = [event["data"]["stage"] for event in kept if event["type"] == "stage/exit"]
    reasons = [event["data"]["reason"]["kind"] for event in kept if event["type"] == "turn/end"]
    assert stages == STAGE_PATH["stages"] and reasons == ["completed"]
    assert result.metrics["candidate_paths"] == (STAGE_PATH, STAGE_PATH)
    assert result.metrics["current_paths"] == (STAGE_PATH, STAGE_PATH)
    # The checker's own verify stage and gave_up reason stay out of the root's path.
    checker = [
        json.loads(line)
        for line in (episodes / "candidate-0" / "agents" / "002-checker.jsonl").read_text().splitlines()
    ]
    assert [event["data"]["stage"] for event in checker if event["type"] == "stage/exit"] == ["verify"]


def test_a_retried_step_claims_a_new_directory_and_a_record_file_is_never_replaced(tmp_path: Path) -> None:
    backend, _ = native_backend(tmp_path, step_record_dir=str(tmp_path / "record"))
    step(backend)
    root = (tmp_path / "record" / "demo").resolve()
    first = root / "1"
    before = {path.relative_to(first): path.stat().st_mtime_ns for path in first.rglob("*")}

    # The same step number again (a crash between prepare and commit) lands beside the first attempt.
    result = step(backend)
    second = root / "1-2"
    assert result.metrics["step_record"] == str(second)
    assert {path.relative_to(first): path.stat().st_mtime_ns for path in first.rglob("*")} == before
    assert {path.relative_to(second) for path in second.rglob("*")} == set(before)
    step(backend)
    assert sorted(path.name for path in root.iterdir()) == ["1", "1-2", "1-3"]

    with pytest.raises(FileExistsError):
        StepRecord.write(first, "proposer.json", [])
    assert read_json(first / "proposer.json") == []


def test_a_recheck_step_writes_episodes_only_and_asks_the_proposer_nothing(tmp_path: Path) -> None:
    backend, releases = native_backend(tmp_path, step_record_dir=str(tmp_path / "record"), recheck_every=1)
    first = step(backend)
    assert first.metrics["selected"] is True
    releases.publish("r1", HarnessTree.read(first.artifact.local_path))
    second = step(backend, first.state)
    assert second.metrics["recheck"] is True and second.metrics["proposer_calls"] == 0
    record = (tmp_path / "record" / "demo" / "2").resolve()
    assert second.metrics["step_record"] == str(record)
    assert sorted(path.name for path in record.iterdir()) == ["episodes"]
    assert sorted(path.name for path in (record / "episodes").iterdir()) == ["candidate-0", "current-0"]


def test_a_record_copy_that_fails_aborts_the_step_instead_of_scoring_it(tmp_path: Path) -> None:
    backend, _ = native_backend(tmp_path, step_record_dir=str(tmp_path / "record"))
    prepared = backend.prepare_step(JobData("demo:1"), backend.initial_state(), 0)
    candidate = prepared.candidate
    assert candidate is not None and candidate.record_dir == (tmp_path / "record" / "demo" / "1").resolve()
    # A file where the episodes directory goes stands in for a full or unwritable record volume.
    (candidate.record_dir / "episodes").write_text("in the way")
    with pytest.raises(TrajectoryKeepError, match="cannot keep episode trajectory"):
        backend.evaluate(candidate)
    backend.abort_step(prepared)
    assert backend.failed_step_metrics()["failed_stage"] == "evaluating"

    # The failure is not an episode failure: run_episode raises it past the launch mapping, and the episode
    # root is still removed on the way out.
    from reef.harness.episodes import run as episode_module

    removed: list[Path] = []
    remove = episode_module._remove_episode_root

    def observed(root: Path) -> None:
        removed.append(root)
        remove(root)

    descriptor = get_adapter("pi")
    files = render_tree(HarnessTree({"rules.md": "keep me\n"}), descriptor, MODEL.config_overlays(descriptor))
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(episode_module, "_remove_episode_root", observed)
        with pytest.raises(TrajectoryKeepError) as caught:
            run_episode(descriptor, files, "task", binary=str(make_pi_binary(tmp_path)), keep_dir=blocker / "kept")
    assert not isinstance(caught.value, EpisodeError)
    assert len(removed) == 1 and not removed[0].exists()


def test_nothing_is_written_when_the_step_record_is_off(tmp_path: Path) -> None:
    backend, _ = native_backend(tmp_path)
    before = set(tmp_path.rglob("*"))
    result = step(backend)
    assert set(tmp_path.rglob("*")) == before
    assert "step_record" not in result.metrics
    # The counts and the path ride the commit record either way.
    assert result.metrics["proposer_calls"] == 0
    assert result.metrics["candidate_paths"] == (STAGE_PATH,)


def test_an_episode_that_could_not_run_has_no_path_and_a_pi_session_has_no_stages(tmp_path: Path) -> None:
    missing, _ = native_backend(tmp_path, binary=str(tmp_path / "no-such-binary"))
    result = step(missing)
    assert result.metrics["episode_failures"] == 2
    assert result.metrics["candidate_paths"] == (None,) and result.metrics["current_paths"] == (None,)

    (tmp_path / "pi").mkdir()
    pi, _ = harness_backend(tmp_path / "pi", lambda context: MARKER)
    result = step(pi)
    assert result.metrics["candidate_paths"] == ({"stages": [], "reason": None},)


# -- the episode seam -------------------------------------------------------


def test_run_episode_keeps_the_trajectory_before_the_root_is_removed(tmp_path: Path) -> None:
    descriptor = get_adapter("pi")
    files = render_tree(HarnessTree({"rules.md": "keep me\n"}), descriptor, MODEL.config_overlays(descriptor))
    keep = tmp_path / "kept"
    result = run_episode(descriptor, files, "task", binary=str(make_pi_binary(tmp_path)), keep_dir=keep)
    kept = [json.loads(line) for line in (keep / "session.jsonl").read_text().splitlines()]
    assert kept == list(result.trajectory) and kept[-1]["rules"] == "keep me\n"

    # A binary that never writes a session keeps nothing and raises nothing.
    silent = tmp_path / "silent"
    silent.write_text("#!/bin/sh\nexit 0\n")
    silent.chmod(0o755)
    empty = run_episode(descriptor, files, "task", binary=str(silent), keep_dir=tmp_path / "none")
    assert empty.trajectory == () and not (tmp_path / "none").exists()

    # A link an evolved tool left in the session directory is kept as a link: the outside file never enters.
    outside = tmp_path / "outside.txt"
    outside.write_text("not for the record")
    linker = tmp_path / "linker"
    linker.write_text(
        f'#!/bin/sh\nmkdir -p "$PI_CODING_AGENT_SESSION_DIR"\nln -s {outside} "$PI_CODING_AGENT_SESSION_DIR/pulled"\n'
    )
    linker.chmod(0o755)
    run_episode(descriptor, files, "task", binary=str(linker), keep_dir=tmp_path / "linked")
    assert (tmp_path / "linked" / "pulled").is_symlink()

    # A kept record is never merged over: an existing episode directory refuses the copy.
    taken = tmp_path / "taken"
    taken.mkdir()
    (taken / "session.jsonl").write_text("earlier record\n")
    with pytest.raises(TrajectoryKeepError, match="cannot keep episode trajectory"):
        run_episode(descriptor, files, "task", binary=str(make_pi_binary(tmp_path)), keep_dir=taken)
    assert (taken / "session.jsonl").read_text() == "earlier record\n"


# -- the setting ------------------------------------------------------------


def recipe_config(**evolution) -> dict:
    return {
        "evolution": {
            "propose": resolve_proposer(lambda context: None),
            "evaluate": resolve_episode_scorer(native_score),
            "tasks": ["one"],
            **evolution,
        }
    }


def test_recipe_parses_step_record_dir_and_the_backend_refuses_an_unwritable_one(tmp_path: Path) -> None:
    binary = str(make_pi_binary(tmp_path))
    on = HarnessEvolutionRecipe.from_environment(
        {}, config=recipe_config(step_record_dir=str(tmp_path / "record"), binary=binary), runtime=runtime()
    )
    assert on.step_record_dir == str(tmp_path / "record")
    assert HarnessEvolutionRecipe.from_environment({}, config=recipe_config()).step_record_dir is None
    for bad in (42, "", "  "):
        with pytest.raises(RecipeConfigError, match="step_record_dir must be a non-empty path"):
            HarnessEvolutionRecipe.from_environment({}, config=recipe_config(step_record_dir=bad))

    # One recipe serves many scenarios; each scenario's steps record under their own directory.
    records = SQLiteRecordStore()
    for scenario in ("demo", "other"):
        releases = ComponentReleases(TreeReleases(on.seed), COMPONENT)
        trainer = on.build(scenario, records, releases=releases)
        assert trainer.candidate_backend.records.root == (tmp_path / "record" / scenario).resolve()
        trainer.close()
    records.close()
    assert sorted(path.name for path in (tmp_path / "record").iterdir()) == ["demo", "other"]

    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    with pytest.raises(ValueError, match=r"step_record_dir .* cannot be created"):
        native_backend(tmp_path, step_record_dir=str(blocker / "record"))
    with pytest.raises(ValueError, match="step_record_dir must be a non-empty path"):
        native_backend(tmp_path, step_record_dir="")


def test_a_relative_record_dir_is_made_absolute_at_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    config = recipe_config(step_record_dir="rel", binary=str(make_pi_binary(tmp_path)))
    recipe = HarnessEvolutionRecipe.from_environment({}, config=config, runtime=runtime())
    backend = recipe.build_backend("demo", ComponentReleases(TreeReleases(recipe.seed), COMPONENT), None)
    assert backend.records.root == (tmp_path / "rel" / "demo").resolve() and backend.records.root.is_absolute()


# -- what the boundary never saw, the raw request seam, and the episode record ---------------------


def test_record_redacts_credential_shaped_text_from_the_proposer_and_the_refused_patch(
    tmp_path: Path, served_model
) -> None:
    key = "sk-" + "a" * 24
    served_model["replies"] = [f"use {key} for auth"]

    def propose(context):
        context.models.served.chat([{"role": "user", "content": f"the key is {key}"}])
        return TreePatch(changed_rules(f"always send {key}\n"))

    backend, _ = harness_backend(tmp_path, propose, step_record_dir=str(tmp_path / "record"))
    result = step(backend)
    assert "inline credential" in str(result.metrics.get("skipped"))  # the tree validation refused it
    record = tmp_path / "record" / "demo" / "1"
    written = (record / "proposer.json").read_text() + (record / "patch.json").read_text()
    # Two proposals (the first and its retry), each with the request, the reply and the response on file.
    assert key not in written and written.count("[redacted credential]") == 7


def test_a_methods_complete_call_shares_the_budget_and_the_record(tmp_path: Path, served_model) -> None:
    served_model["replies"] = ["echo"]
    served_model["usage"] = {"prompt_tokens": 40, "completion_tokens": 6}

    def propose(context):
        context.models.served.complete({"messages": [{"role": "user", "content": "one"}]}, timeout_s=3.0)
        with pytest.raises(RuntimeError, match="budget"):
            context.models.served.complete({"messages": [{"role": "user", "content": "two"}]})
        return MARKER

    backend, _ = harness_backend(
        tmp_path, propose, step_record_dir=str(tmp_path / "record"), max_model_calls_per_step=1
    )
    result = step(backend)
    assert result.metrics["proposer_calls"] == 1
    calls = read_json(tmp_path / "record" / "demo" / "1" / "proposer.json")
    assert calls[0]["body"] == {"messages": [{"role": "user", "content": "one"}]}
    assert calls[0]["params"] == {"timeout_s": 3.0} and "reply" not in calls[0]
    assert calls[0]["response"]["choices"][0]["message"]["content"] == "raw one"
    # The tokens the call reported land in the record and, summed, in the step's metrics: recorded, never charged.
    assert calls[0]["usage"] == {"input_tokens": 40, "output_tokens": 6}
    assert result.metrics["proposer_input_tokens"] == 40 and result.metrics["proposer_output_tokens"] == 6


def test_a_long_reply_is_clipped_with_the_marker(tmp_path: Path, served_model) -> None:
    served_model["replies"] = ["y" * (RECORD_TEXT_CAP + 3)]

    def propose(context):
        context.models.served.chat([{"role": "user", "content": "go"}])
        return MARKER

    backend, _ = harness_backend(tmp_path, propose, step_record_dir=str(tmp_path / "record"))
    step(backend)
    reply = read_json(tmp_path / "record" / "demo" / "1" / "proposer.json")[0]["reply"]
    assert reply.startswith("y" * RECORD_TEXT_CAP) and reply.endswith("... [clipped 3 chars]")


def test_every_episode_leaves_a_record_the_scorer_can_be_replayed_from(tmp_path: Path) -> None:
    backend, _ = native_backend(tmp_path, step_record_dir=str(tmp_path / "record"))
    result = step(backend)
    record = read_json(tmp_path / "record" / "demo" / "1" / "episodes" / "candidate-0" / "episode.json")
    assert record["task"] == "task one" and record["exit_code"] == 0 and record["failure"] is None
    assert record["score"] == 1.0 and result.metrics["selected"] is True
    assert record["path"] == result.metrics["candidate_paths"][0]
    assert record["residue"] == [] and isinstance(record["stdout"], str) and isinstance(record["stderr"], str)

    # An episode that could not run still leaves its record: the failure, and no output to show.
    broken, _ = native_backend(tmp_path, binary=str(tmp_path / "missing"), step_record_dir=str(tmp_path / "broken"))
    step(broken)
    record = read_json(tmp_path / "broken" / "demo" / "1" / "episodes" / "candidate-0" / "episode.json")
    assert record["score"] is None and record["failure"]["stage"] == "launch"
    assert record["exit_code"] is None and record["stdout"] is None and record["residue"] is None


def test_agent_work_sums_the_tokens_each_agents_steps_reported() -> None:
    """Per agent, the usage on the assistant messages and compactions of a
    native trajectory adds up beside the turn and tool counters, and the
    side's sum keeps every counter."""
    from reef.harness.episodes.trajectory import agent_work, sum_agents

    trajectory = [
        {"type": "session", "data": {"agent": "root"}},
        {"type": "step/start", "data": {"step": 1}},
        {"type": "assistant/message", "data": {"step": 1, "usage": {"input_tokens": 100, "output_tokens": 10}}},
        {"type": "context/compacted", "data": {"step": 1, "usage": {"input_tokens": 30, "output_tokens": 5}}},
        {"type": "step/start", "data": {"step": 2}},
        {"type": "assistant/message", "data": {"step": 2}},
        {"type": "session", "data": {"agent": "checker"}},
        {"type": "step/start", "data": {"step": 1}},
        {"type": "assistant/message", "data": {"step": 1, "usage": {"input_tokens": 7, "output_tokens": 1}}},
    ]
    work = agent_work(trajectory)
    assert work == {
        "root": {"turns": 1, "steps": 2, "tool_calls": 0, "tool_errors": 0, "input_tokens": 130, "output_tokens": 15},
        "checker": {"turns": 1, "steps": 1, "tool_calls": 0, "tool_errors": 0, "input_tokens": 7, "output_tokens": 1},
    }
    assert sum_agents([work, {"root": {"turns": 1, "steps": 1, "tool_calls": 2, "tool_errors": 0}}]) == {
        "checker": work["checker"],
        "root": {"turns": 2, "steps": 3, "tool_calls": 2, "tool_errors": 0, "input_tokens": 130, "output_tokens": 15},
    }
