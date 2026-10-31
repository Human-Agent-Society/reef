"""Guarantees of the evolve step record and the run records, hermetic: the method's model calls, its result
and its candidate land in the step's record directory, and every evaluation run's trajectory lands in the
scenario's history, where a reader can rebuild the decision. A fake native binary writes a session tree shaped
like the native loop's, so the stage path metric is checked against a kept file; the served model is a stub
endpoint behind ``urllib.request.urlopen``."""

from __future__ import annotations

import io
import json
import tarfile
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import pytest
from tests.support._strategies import resolve_task_scorer
from tests.support.harness_service import backend_step
from tests.support.harness_steps import (
    COMPONENT,
    MODEL,
    PI_SEED,
    UNCHANGED,
    TreeReleases,
    build_harness_backend,
    changed_rules,
    harness_backend,
    harness_recipe,
    head_of,
    make_pi_binary,
)

from reef.evaluation.harness import TaskEvaluationWorker
from reef.evaluation.harness.runner import TaskRun
from reef.execution.sandbox.process import LocalExecutor
from reef.harness.adapters import get_adapter
from reef.harness.tasks.model_binding import ModelBindings
from reef.harness.tasks.run import TaskRunError, TrajectoryKeepError, run_task
from reef.harness.tree import HarnessTree, TreePatch, render_tree
from reef.harness.tree.checks import RECORD_TEXT_CAP
from reef.runtime.executor.config import ExecutorSettings
from reef.train.harness_backend.backend import HarnessBackend
from reef.train.harness_backend.records import StepRecord
from reef.train.harness_backend.state import HarnessState
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
    tmp_path: Path,
    method=lambda workspace, services: MARKER,
    *,
    binary: str | None = None,
    workers: int = 1,
    **fields,
) -> tuple[HarnessBackend, TreeReleases]:
    """The native backend over the fake native binary, and its in-memory history whose base is the native seed."""
    recipe = harness_recipe(
        tmp_path,
        method,
        seed=NATIVE_SEED,
        score=native_score,
        adapter="native",
        binary=binary or str(make_native_binary(tmp_path)),
        **fields,
    )
    recipe = replace(recipe, worker_executor=ExecutorSettings(workers=workers))
    releases = TreeReleases(recipe.seed)
    return build_harness_backend(recipe, "demo", releases), releases


def steps_dir(tmp_path: Path) -> Path:
    """Where ``harness_recipe`` keeps the step records of the ``demo`` scenario."""
    return (tmp_path / "harness" / "demo" / COMPONENT / "steps").resolve()


def step(backend: HarnessBackend, state=None, scenario_step: int | None = None) -> TrainStepResult:
    state = HarnessState().to_json() if state is None else state
    return backend_step(backend, JobData("demo:1"), state, scenario_step=scenario_step)


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
    worker = TaskEvaluationWorker(
        get_adapter("native"),
        resolve_task_scorer(native_score),
        str(make_native_binary(tmp_path)),
        10,
        LocalExecutor(),
        False,
        transfer_records=True,
    )
    scored = worker.run(TaskRun({}, "one", blocker / "run"), ModelBindings(served=MODEL))
    assert scored.record_archive is not None
    assert blocker.read_text() == "this path cannot be a directory on the worker"
    with tarfile.open(fileobj=BytesIO(scored.record_archive), mode="r:gz") as archive:
        assert {"./session.jsonl", "./agents/002-checker.jsonl", "./run.json"} <= set(archive.getnames())


# -- the method's calls -----------------------------------------------------


def test_the_record_keeps_every_model_call_through_the_binding_seam(tmp_path: Path, served_model) -> None:
    """Without a call budget the method still sees a recording binding, and every chat lands with its model,
    request, reply and time, long text cut at the cap with a marker naming what was dropped."""
    long_prompt = "x" * (RECORD_TEXT_CAP + 5)

    def method(workspace, services):
        services.models.served.chat([{"role": "user", "content": "first"}], timeout_s=5.0, max_tokens=8)
        services.models.served.chat(
            [{"role": "system", "content": long_prompt}, {"role": "user", "content": "second"}]
        )
        return MARKER

    backend, _ = harness_backend(tmp_path, method)
    result = step(backend)

    assert result.metrics["model_calls"] == 2
    assert result.metrics["model_seconds"] >= 0.0
    # An answer that reports no usage counts zero tokens rather than none.
    assert result.metrics["model_input_tokens"] == 0 and result.metrics["model_output_tokens"] == 0
    record = steps_dir(tmp_path) / "1"
    assert result.metrics["step_record"] == str(record)
    calls = read_json(record / "calls.json")
    assert [call["model"] for call in calls] == [MODEL.model, MODEL.model]
    assert calls[0]["messages"] == [{"role": "user", "content": "first"}]
    assert calls[0]["params"] == {"max_tokens": 8, "timeout_s": 5.0}
    assert calls[0]["reply"] == "ok" and calls[0]["seconds"] >= 0.0
    assert calls[1]["params"] == {} and calls[1]["messages"][1] == {"role": "user", "content": "second"}
    clipped = calls[1]["messages"][0]["content"]
    assert clipped.startswith("x" * RECORD_TEXT_CAP) and clipped.endswith("... [clipped 5 chars]")
    # The record keeps the candidate the method left; a method that said nothing leaves output/ empty.
    assert HarnessTree.read(record / "harness") == PI_SEED.apply(MARKER)
    assert list((record / "output").iterdir()) == []

    def wordy(workspace, services):
        services.models.served.chat([{"role": "user", "content": "go"}], stop=["z" * (RECORD_TEXT_CAP + 2)])
        return MARKER

    (tmp_path / "params").mkdir()
    backend, _ = harness_backend(tmp_path / "params", wordy)
    step(backend)
    stop = read_json(steps_dir(tmp_path / "params") / "1" / "calls.json")[0]["params"]["stop"][0]
    assert stop.startswith("z" * RECORD_TEXT_CAP) and stop.endswith("... [clipped 2 chars]")


def test_the_record_keeps_a_failed_call_and_the_calls_before_a_budget_stop(tmp_path: Path, served_model) -> None:
    served_model["replies"] = [RuntimeError("endpoint down")]

    def method(workspace, services):
        try:
            services.models.served.chat([{"role": "user", "content": "hi"}])
        except RuntimeError:
            return

    backend, _ = harness_backend(tmp_path, method)
    result = step(backend)
    assert result.metrics["skipped"] == UNCHANGED and result.metrics["model_calls"] == 1
    calls = read_json(steps_dir(tmp_path) / "1" / "calls.json")
    assert calls[0]["error"] == "RuntimeError: endpoint down" and "reply" not in calls[0]
    assert calls[0]["seconds"] >= 0.0

    served_model["replies"] = ["ok"]

    def greedy(workspace, services):
        while True:
            services.models.served.chat([{"role": "user", "content": "more"}])

    (tmp_path / "greedy").mkdir()
    backend, _ = harness_backend(tmp_path / "greedy", greedy, max_model_calls_per_step=2)
    with pytest.raises(RuntimeError, match="model call budget of 2"):
        step(backend)
    # The stop raised out of the method, and the two calls before it are on file.
    record = steps_dir(tmp_path / "greedy") / "1"
    assert len(read_json(record / "calls.json")) == 2 and list((record / "output").iterdir()) == []


def test_a_rejected_removal_keeps_its_null_in_the_metrics(tmp_path: Path) -> None:
    removal = TreePatch({"rules.md": None})
    backend, _ = harness_backend(tmp_path, lambda workspace, services: removal)
    result = step(backend)
    assert result.metrics["selected"] is False
    assert result.metrics["patch"] == {"rules.md": None}
    assert "rules.md" not in HarnessTree.read(steps_dir(tmp_path) / "1" / "harness").files


# -- the run records and the stage path ---------------------------------


@pytest.mark.parametrize("workers", [1, 2])
def test_the_history_keeps_every_run_and_the_path_matches_the_kept_trajectory(
    tmp_path: Path,
    workers: int,
) -> None:
    backend, _ = native_backend(tmp_path, workers=workers, evaluation={"repeats": 2})
    result = step(backend)
    backend.close()
    assert result.metrics["selected"] is True
    assert read_json(steps_dir(tmp_path) / "1" / "calls.json") == []

    history = backend.history
    candidate = NATIVE_SEED.apply(MARKER).checksum()
    assert sorted((record.tree == candidate, record.repeat) for record in history.runs) == [
        (False, 0),
        (False, 1),
        (True, 0),
        (True, 1),
    ]
    for record in history.runs:
        kept = history.run_dir(record.run_id)
        assert (kept / "session.jsonl").is_file() and (kept / "agents" / "002-checker.jsonl").is_file()

    # The path metric is what the kept file says: the root's stage/exit names and how its turn ended.
    first = history.run_dir(next(record.run_id for record in history.runs if record.tree == candidate))
    kept = [json.loads(line) for line in (first / "session.jsonl").read_text().splitlines()]
    stages = [event["data"]["stage"] for event in kept if event["type"] == "stage/exit"]
    reasons = [event["data"]["reason"]["kind"] for event in kept if event["type"] == "turn/end"]
    assert stages == STAGE_PATH["stages"] and reasons == ["completed"]
    assert result.metrics["candidate_paths"] == (STAGE_PATH, STAGE_PATH)
    assert result.metrics["current_paths"] == (STAGE_PATH, STAGE_PATH)
    # The checker's own verify stage and gave_up reason stay out of the root's path.
    checker = [json.loads(line) for line in (first / "agents" / "002-checker.jsonl").read_text().splitlines()]
    assert [event["data"]["stage"] for event in checker if event["type"] == "stage/exit"] == ["verify"]


def test_a_retried_step_claims_a_new_directory_and_a_record_file_is_never_replaced(tmp_path: Path) -> None:
    backend, _ = native_backend(tmp_path)
    step(backend)
    root = steps_dir(tmp_path)
    first = root / "1"
    before = {path.relative_to(first): path.stat().st_mtime_ns for path in first.rglob("*")}

    # The same step number again (a crash between prepare and commit) lands beside the first attempt.
    result = step(backend, scenario_step=0)
    second = root / "1-2"
    assert result.metrics["step_record"] == str(second)
    assert {path.relative_to(first): path.stat().st_mtime_ns for path in first.rglob("*")} == before
    assert {path.relative_to(second) for path in second.rglob("*")} == set(before)
    step(backend, scenario_step=0)
    assert sorted(path.name for path in root.iterdir()) == ["1", "1-2", "1-3"]

    with pytest.raises(FileExistsError):
        StepRecord.write(first, "calls.json", [])
    assert read_json(first / "calls.json") == []


def test_an_run_record_that_cannot_be_kept_aborts_the_step_instead_of_scoring_it(tmp_path: Path) -> None:
    backend, _ = native_backend(tmp_path)
    prepared = backend.prepare_step(JobData("demo:1"), HarnessState().to_json(), 0, head=head_of(backend))
    assert prepared.candidate is not None
    # A file where the run records go stands in for a full or unwritable history volume.
    (backend.history.root / "runs").write_text("in the way")
    with pytest.raises(TrajectoryKeepError, match="cannot keep run trajectory"):
        backend.evaluate(prepared.candidate)
    backend.abort_step(prepared)
    assert backend.failed_step_metrics()["failed_stage"] == "evaluating"

    # The failure is not a run failure: run_task raises it past the launch mapping, and the run
    # root is still removed on the way out.
    from reef.harness.tasks import run as run_module

    removed: list[Path] = []
    remove = run_module._remove_run_root

    def observed(root: Path) -> None:
        removed.append(root)
        remove(root)

    descriptor = get_adapter("pi")
    files = render_tree(HarnessTree({"rules.md": "keep me\n"}), descriptor, MODEL.config_overlays(descriptor))
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(run_module, "_remove_run_root", observed)
        with pytest.raises(TrajectoryKeepError) as caught:
            run_task(descriptor, files, "task", binary=str(make_pi_binary(tmp_path)), keep_dir=blocker / "kept")
    assert not isinstance(caught.value, TaskRunError)
    assert len(removed) == 1 and not removed[0].exists()


def test_a_run_that_could_not_run_has_no_path_and_a_pi_session_has_no_stages(tmp_path: Path) -> None:
    missing, _ = native_backend(tmp_path, binary=str(tmp_path / "no-such-binary"))
    result = step(missing)
    assert result.metrics["run_failures"] == 2
    assert result.metrics["candidate_paths"] == (None,) and result.metrics["current_paths"] == (None,)

    (tmp_path / "pi").mkdir()
    pi, _ = harness_backend(tmp_path / "pi", lambda workspace, services: MARKER)
    result = step(pi)
    assert result.metrics["candidate_paths"] == ({"stages": [], "reason": None},)


# -- the run seam -------------------------------------------------------


def test_run_task_keeps_the_trajectory_before_the_root_is_removed(tmp_path: Path) -> None:
    descriptor = get_adapter("pi")
    files = render_tree(HarnessTree({"rules.md": "keep me\n"}), descriptor, MODEL.config_overlays(descriptor))
    keep = tmp_path / "kept"
    result = run_task(descriptor, files, "task", binary=str(make_pi_binary(tmp_path)), keep_dir=keep)
    kept = [json.loads(line) for line in (keep / "session.jsonl").read_text().splitlines()]
    assert kept == list(result.trajectory) and kept[-1]["rules"] == "keep me\n"

    # A binary that never writes a session keeps nothing and raises nothing.
    silent = tmp_path / "silent"
    silent.write_text("#!/bin/sh\nexit 0\n")
    silent.chmod(0o755)
    empty = run_task(descriptor, files, "task", binary=str(silent), keep_dir=tmp_path / "none")
    assert empty.trajectory == () and not (tmp_path / "none").exists()

    # A link an evolved tool left in the session directory is kept as a link: the outside file never enters.
    outside = tmp_path / "outside.txt"
    outside.write_text("not for the record")
    linker = tmp_path / "linker"
    linker.write_text(
        f'#!/bin/sh\nmkdir -p "$PI_CODING_AGENT_SESSION_DIR"\nln -s {outside} "$PI_CODING_AGENT_SESSION_DIR/pulled"\n'
    )
    linker.chmod(0o755)
    run_task(descriptor, files, "task", binary=str(linker), keep_dir=tmp_path / "linked")
    assert (tmp_path / "linked" / "pulled").is_symlink()

    # A kept record is never merged over: an existing run directory refuses the copy.
    taken = tmp_path / "taken"
    taken.mkdir()
    (taken / "session.jsonl").write_text("earlier record\n")
    with pytest.raises(TrajectoryKeepError, match="cannot keep run trajectory"):
        run_task(descriptor, files, "task", binary=str(make_pi_binary(tmp_path)), keep_dir=taken)
    assert (taken / "session.jsonl").read_text() == "earlier record\n"


# -- what the checks never saw, the raw request seam, and the run record ------------------------


def test_the_record_redacts_credential_shaped_text_from_the_model_calls(tmp_path: Path, served_model) -> None:
    key = "sk-" + "a" * 24
    served_model["replies"] = [f"use {key} for auth"]

    def method(workspace, services):
        services.models.served.chat([{"role": "user", "content": f"the key is {key}"}])
        # An agent's gateway records the calls it relays itself.
        services.calls.record({"source": "agent", "reply": f"the agent used {key}"})
        workspace.write_output(f"wrote {key} into the rules")

    backend, _ = harness_backend(tmp_path, method)
    result = step(backend)
    assert result.metrics["skipped"] == UNCHANGED
    record = steps_dir(tmp_path) / "1"
    written = (record / "calls.json").read_text() + (record / "output" / "summary.md").read_text()
    # The request, the reply, the provider's response, the gateway's entry and the method's summary, each on file
    # without the key.
    assert key not in written and written.count("[redacted credential]") == 5


def test_a_methods_complete_call_shares_the_budget_and_the_record(tmp_path: Path, served_model) -> None:
    served_model["replies"] = ["echo"]
    served_model["usage"] = {"prompt_tokens": 40, "completion_tokens": 6}

    def method(workspace, services):
        services.models.served.complete({"messages": [{"role": "user", "content": "one"}]}, timeout_s=3.0)
        with pytest.raises(RuntimeError, match="budget"):
            services.models.served.complete({"messages": [{"role": "user", "content": "two"}]})
        return MARKER

    backend, _ = harness_backend(tmp_path, method, max_model_calls_per_step=1)
    result = step(backend)
    assert result.metrics["model_calls"] == 1
    calls = read_json(steps_dir(tmp_path) / "1" / "calls.json")
    assert calls[0]["body"] == {"messages": [{"role": "user", "content": "one"}]}
    assert calls[0]["params"] == {"timeout_s": 3.0} and "reply" not in calls[0]
    assert calls[0]["response"]["choices"][0]["message"]["content"] == "raw one"
    # The tokens the call reported land in the record and, summed, in the step's metrics: recorded, never charged.
    assert calls[0]["usage"] == {"input_tokens": 40, "output_tokens": 6}
    assert result.metrics["model_input_tokens"] == 40 and result.metrics["model_output_tokens"] == 6


def test_a_long_reply_is_clipped_with_the_marker(tmp_path: Path, served_model) -> None:
    served_model["replies"] = ["y" * (RECORD_TEXT_CAP + 3)]

    def method(workspace, services):
        services.models.served.chat([{"role": "user", "content": "go"}])
        return MARKER

    backend, _ = harness_backend(tmp_path, method)
    step(backend)
    reply = read_json(steps_dir(tmp_path) / "1" / "calls.json")[0]["reply"]
    assert reply.startswith("y" * RECORD_TEXT_CAP) and reply.endswith("... [clipped 3 chars]")


def test_every_run_leaves_a_record_the_scorer_can_be_replayed_from(tmp_path: Path) -> None:
    backend, _ = native_backend(tmp_path)
    result = step(backend)
    candidate = NATIVE_SEED.apply(MARKER).checksum()
    first = next(record for record in backend.history.runs if record.tree == candidate)
    record = read_json(backend.history.run_dir(first.run_id) / "run.json")
    assert record["task"] == "task one" and record["exit_code"] == 0 and record["failure"] is None
    assert record["score"] == 1.0 and result.metrics["selected"] is True
    assert record["path"] == result.metrics["candidate_paths"][0]
    assert record["residue"] == [] and isinstance(record["stdout"], str) and isinstance(record["stderr"], str)

    # A run that could not run still leaves its record: the failure, and no output to show.
    (tmp_path / "broken").mkdir()
    broken, _ = native_backend(tmp_path / "broken", binary=str(tmp_path / "missing"))
    step(broken)
    first = next(record for record in broken.history.runs if record.tree == candidate)
    record = read_json(broken.history.run_dir(first.run_id) / "run.json")
    assert record["score"] == "failed" and record["failure"]["stage"] == "launch"
    assert record["exit_code"] is None and record["stdout"] is None and record["residue"] is None


def test_agent_work_sums_the_tokens_each_agents_steps_reported() -> None:
    """Per agent, the usage on the assistant messages and compactions of a
    native trajectory adds up beside the turn and tool counters, and the
    side's sum keeps every counter."""
    from reef.harness.tasks.trajectory import agent_work, sum_agents

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
