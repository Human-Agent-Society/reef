"""Guarantees of the evolve step record, hermetic: the proposer's model calls,
the parsed proposal with its options, and every gate episode's trajectory
land where a reader can rebuild the decision, and nothing lands when the
record is off. A fake native binary writes a session tree shaped like the
native loop's, so the stage path metric is checked against a kept file."""

from __future__ import annotations

import json
import tarfile
from io import BytesIO
from pathlib import Path

import pytest

from reef.harness.adapters import get_adapter
from reef.harness.episodes.executor import LocalExecutor
from reef.harness.episodes.model_binding import ModelBindings
from reef.harness.episodes.run import EpisodeError, TrajectoryKeepError, run_episode
from reef.harness.tree.render import render_composition
from reef.recipe import RecipeConfigError
from reef.train.cordis_backend import CordisBackend, CordisRecipe, Mutation
from reef.train.cordis_backend.backend import RECORD_TEXT_CAP, EpisodeEvaluationWorker
from reef.train.cordis_backend.strategies import resolve_episode_scorer, resolve_proposer

from .test_harness_recipe import (
    MODEL,
    _ChatBinding,
    batch,
    evaluate,
    make_binary,
    run_backend_step,
    runtime,
    seeded_state,
)

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

MARKER_RULES = Mutation("create", "r1", {"name": "rules", "config": {"text": "marker rules"}})
MARKER_RECORD = {"op": "create", "id": "r1", "options": {"name": "rules", "config": {"text": "marker rules"}}}


def make_native_binary(tmp_path: Path) -> Path:
    binary = tmp_path / "fake-native"
    binary.write_text(NATIVE_FAKE)
    binary.chmod(0o755)
    return binary


def native_score(task: str, result) -> float:
    del task
    return (
        1.0 if any("marker" in str((event.get("data") or {}).get("rules", "")) for event in result.trajectory) else 0.0
    )


def native_backend(tmp_path: Path, propose=lambda n, s, m: MARKER_RULES, **options) -> CordisBackend:
    return CordisBackend(
        descriptor=get_adapter("native"),
        propose=resolve_proposer(propose),
        score_episode=resolve_episode_scorer(native_score),
        tasks=("task one",),
        models=MODEL,
        binary=options.pop("binary", str(make_native_binary(tmp_path))),
        **options,
    )


def pi_backend(tmp_path: Path, propose, **options) -> CordisBackend:
    return CordisBackend(
        descriptor=get_adapter("pi"),
        propose=resolve_proposer(propose),
        score_episode=resolve_episode_scorer(evaluate),
        tasks=("task one",),
        models=options.pop("models", MODEL),
        binary=str(make_binary(tmp_path)),
        **options,
    )


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


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


def test_record_records_every_proposer_call_through_the_binding_seam(tmp_path: Path) -> None:
    """Without a call budget the proposer still sees a recording binding, and
    every chat lands with its model, request, reply and time, long text cut
    at the cap with a marker naming what was dropped."""
    long_prompt = "x" * (RECORD_TEXT_CAP + 5)

    def propose(nodes, samples, models):
        models.served.chat([{"role": "user", "content": "first"}], timeout_s=5.0, max_tokens=8)
        models.served.chat([{"role": "system", "content": long_prompt}, {"role": "user", "content": "second"}])
        return MARKER_RULES

    b = pi_backend(tmp_path, propose, models=ModelBindings(served=_ChatBinding()), step_record_dir=tmp_path / "record")
    result = run_backend_step(b, batch(), b.initial_state())

    assert result.metrics["proposer_calls"] == 2
    assert result.metrics["proposer_seconds"] >= 0.0
    assert result.metrics["step_record"] == str(tmp_path / "record" / "1")
    calls = read_json(tmp_path / "record" / "1" / "proposer.json")
    assert [call["model"] for call in calls] == [MODEL.model, MODEL.model]
    assert calls[0]["messages"] == [{"role": "user", "content": "first"}]
    assert calls[0]["params"] == {"max_tokens": 8, "timeout_s": 5.0}
    assert calls[0]["reply"] == "ok" and calls[0]["seconds"] >= 0.0
    assert calls[1]["params"] == {} and calls[1]["messages"][1] == {"role": "user", "content": "second"}
    clipped = calls[1]["messages"][0]["content"]
    assert clipped.startswith("x" * RECORD_TEXT_CAP) and clipped.endswith("... [clipped 5 chars]")
    assert read_json(tmp_path / "record" / "1" / "mutations.json") == [MARKER_RECORD]

    def wordy(nodes, samples, models):
        models.served.chat([{"role": "user", "content": "go"}], stop=["z" * (RECORD_TEXT_CAP + 2)])
        return MARKER_RULES

    w = pi_backend(tmp_path, wordy, models=ModelBindings(served=_ChatBinding()), step_record_dir=tmp_path / "params")
    run_backend_step(w, batch(), w.initial_state())
    stop = read_json(tmp_path / "params" / "1" / "proposer.json")[0]["params"]["stop"][0]
    assert stop.startswith("z" * RECORD_TEXT_CAP) and stop.endswith("... [clipped 2 chars]")


def test_record_keeps_a_failed_call_and_the_calls_before_a_budget_stop(tmp_path: Path) -> None:
    class Broken(_ChatBinding):
        def chat(self, *args, **kwargs) -> str:
            raise RuntimeError("endpoint down")

    def propose(nodes, samples, models):
        try:
            models.served.chat([{"role": "user", "content": "hi"}])
        except RuntimeError:
            return

    b = pi_backend(tmp_path, propose, models=ModelBindings(served=Broken()), step_record_dir=tmp_path / "broken")
    result = run_backend_step(b, batch(), b.initial_state())
    assert result.metrics["skipped"] == "no proposal" and result.metrics["proposer_calls"] == 1
    calls = read_json(tmp_path / "broken" / "1" / "proposer.json")
    assert calls[0]["error"] == "RuntimeError: endpoint down" and "reply" not in calls[0]
    assert calls[0]["seconds"] >= 0.0
    assert read_json(tmp_path / "broken" / "1" / "mutations.json") == []

    def greedy(nodes, samples, models):
        while True:
            models.served.chat([{"role": "user", "content": "more"}])

    g = pi_backend(
        tmp_path,
        greedy,
        models=ModelBindings(served=_ChatBinding()),
        max_model_calls_per_step=2,
        step_record_dir=tmp_path / "budget",
    )
    with pytest.raises(RuntimeError, match="model call budget of 2"):
        run_backend_step(g, batch(), g.initial_state())
    # The stop raised out of propose, and the two calls before it are on file.
    assert len(read_json(tmp_path / "budget" / "1" / "proposer.json")) == 2
    assert not (tmp_path / "budget" / "1" / "mutations.json").exists()


# -- rejected proposals keep their content ----------------------------------


def test_rejected_proposal_keeps_its_options_in_state_metrics_and_the_proposer_view(tmp_path: Path) -> None:
    options = {"name": "rules", "config": {"text": "no help"}}
    seen: list[tuple] = []

    def propose(nodes, samples, models, rejected=()):
        seen.append(tuple(rejected))
        return Mutation("create", f"r{len(seen)}", options)

    b = pi_backend(tmp_path, propose, max_rejected_history=3)
    first = run_backend_step(b, batch(), b.initial_state())
    assert first.metrics["selected"] is False
    record = {"op": "create", "id": "r1", "options": options}
    assert first.metrics["mutation"] == record
    assert first.state["rejected_proposals"][0]["mutations"] == [record]
    second = run_backend_step(b, batch(), first.state)
    assert seen[1][0]["mutations"] == [record]
    assert second.state["rejected_proposals"][-1]["mutations"] == [{**record, "id": "r2"}]


def test_a_rejected_remove_records_its_absent_options(tmp_path: Path) -> None:
    b = pi_backend(tmp_path, lambda n, s, m: Mutation("remove", "r1"))
    result = run_backend_step(b, batch(), seeded_state())
    assert result.metrics["selected"] is False
    assert result.metrics["mutation"] == {"op": "remove", "id": "r1", "options": None}
    assert result.state["rejected_proposals"][0]["mutations"] == [{"op": "remove", "id": "r1", "options": None}]


# -- the step's files and the stage path -----------------------------------


@pytest.mark.parametrize("workers", [1, 2])
def test_step_record_writes_the_proposal_and_every_episode_and_the_path_matches_the_kept_trajectory(
    tmp_path: Path,
    workers: int,
) -> None:
    from reef.runtime.executor.config import ExecutorSettings

    b = native_backend(
        tmp_path,
        step_record_dir=tmp_path / "record",
        episode_repeats=2,
        worker_executor=ExecutorSettings(workers=workers),
    )
    result = run_backend_step(b, batch(), b.initial_state())
    b.close()
    assert result.metrics["selected"] is True

    step = tmp_path / "record" / "1"
    assert read_json(step / "proposer.json") == []
    assert read_json(step / "mutations.json") == [MARKER_RECORD]
    episodes = step / "episodes"
    names = ["candidate-0", "candidate-0-1", "current-0", "current-0-1"]
    assert sorted(path.name for path in episodes.iterdir()) == names
    for name in names:
        assert (episodes / name / "session.jsonl").is_file()
        assert (episodes / name / "agents" / "002-checker.jsonl").is_file()

    # The path metric is what the kept file says: the root's stage/exit names and how its turn ended.
    kept = [json.loads(line) for line in (episodes / "candidate-0" / "session.jsonl").read_text().splitlines()]
    stages = [event["data"]["stage"] for event in kept if event["type"] == "stage/exit"]
    reasons = [event["data"]["reason"]["kind"] for event in kept if event["type"] == "turn/end"]
    assert stages == ["think", "act", "think"] and reasons == ["completed"]
    path = {"stages": stages, "reason": "completed"}
    assert result.metrics["candidate_paths"] == (path, path)
    assert result.metrics["current_paths"] == (path, path)
    # The checker's own verify stage and gave_up reason stay out of the root's path.
    checker = [
        json.loads(line)
        for line in (episodes / "candidate-0" / "agents" / "002-checker.jsonl").read_text().splitlines()
    ]
    assert [event["data"]["stage"] for event in checker if event["type"] == "stage/exit"] == ["verify"]


def test_a_retried_step_claims_a_new_directory_and_a_record_file_is_never_replaced(tmp_path: Path) -> None:
    b = native_backend(tmp_path, step_record_dir=tmp_path / "record")
    run_backend_step(b, batch(), b.initial_state())
    first = tmp_path / "record" / "1"
    before = {path.relative_to(first): path.stat().st_mtime_ns for path in first.rglob("*")}

    # The same step number again (a crash between prepare and commit) lands beside the first attempt.
    result = run_backend_step(b, batch(), b.initial_state())
    second = tmp_path / "record" / "1-2"
    assert result.metrics["step_record"] == str(second)
    assert {path.relative_to(first): path.stat().st_mtime_ns for path in first.rglob("*")} == before
    assert {path.relative_to(second) for path in second.rglob("*")} == set(before)
    run_backend_step(b, batch(), b.initial_state())
    assert sorted(path.name for path in (tmp_path / "record").iterdir()) == ["1", "1-2", "1-3"]

    with pytest.raises(FileExistsError):
        CordisBackend._write_record(first, "proposer.json", [])
    assert read_json(first / "proposer.json") == []


def test_a_recheck_step_writes_episodes_only_and_counts_no_proposer_calls(tmp_path: Path) -> None:
    b = native_backend(tmp_path, step_record_dir=tmp_path / "record", recheck_every=1)
    first = run_backend_step(b, batch(), b.initial_state())
    assert first.metrics["selected"] is True
    second = run_backend_step(b, batch(), first.state)
    assert second.metrics["recheck"] is True
    assert second.metrics["proposer_calls"] == 0 and second.metrics["proposer_seconds"] == 0.0
    step = tmp_path / "record" / "2"
    assert second.metrics["step_record"] == str(step)
    assert sorted(path.name for path in step.iterdir()) == ["episodes"]
    assert sorted(path.name for path in (step / "episodes").iterdir()) == ["candidate-0", "current-0"]


def test_a_record_copy_that_fails_aborts_the_step_instead_of_scoring_it(tmp_path: Path) -> None:
    b = native_backend(tmp_path, step_record_dir=tmp_path / "record")
    prepared = b.prepare_step(batch(), b.initial_state(), 0)
    candidate = prepared.candidate
    assert candidate is not None and candidate.record_dir == tmp_path / "record" / "1"
    # A file where the episodes directory goes stands in for a full or unwritable record volume.
    (candidate.record_dir / "episodes").write_text("in the way")
    with pytest.raises(TrajectoryKeepError, match="cannot keep episode trajectory"):
        b.evaluate(candidate)
    b.abort_step(prepared)
    assert b.initial_state()["entries"] == []

    # The failure is not an episode failure: run_episode raises it past the launch mapping, and the episode
    # root is still removed on the way out.
    from reef.harness.episodes import run as episode_module

    removed: list[Path] = []
    remove = episode_module._remove_episode_root

    def observed(root: Path) -> None:
        removed.append(root)
        remove(root)

    descriptor = get_adapter("pi")
    files = render_composition([("rules", {"text": "keep me"}), *MODEL.compose_nodes(descriptor)], descriptor)
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(episode_module, "_remove_episode_root", observed)
        with pytest.raises(TrajectoryKeepError) as caught:
            run_episode(descriptor, files, "task", binary=str(make_binary(tmp_path)), keep_dir=blocker / "kept")
    assert not isinstance(caught.value, EpisodeError)
    assert len(removed) == 1 and not removed[0].exists()


def test_nothing_is_written_when_the_step_record_is_off(tmp_path: Path) -> None:
    b = native_backend(tmp_path)
    before = set(tmp_path.rglob("*"))
    result = run_backend_step(b, batch(), b.initial_state())
    assert set(tmp_path.rglob("*")) == before
    assert "step_record" not in result.metrics
    # The counts and the path ride the commit record either way.
    assert result.metrics["proposer_calls"] == 0
    assert result.metrics["candidate_paths"] == ({"stages": ["think", "act", "think"], "reason": "completed"},)


def test_an_episode_that_could_not_run_has_no_path_and_a_pi_session_has_no_stages(tmp_path: Path) -> None:
    missing = native_backend(tmp_path, binary=str(tmp_path / "no-such-binary"))
    result = run_backend_step(missing, batch(), missing.initial_state())
    assert result.metrics["episode_failures"] == 2
    assert result.metrics["candidate_paths"] == (None,) and result.metrics["current_paths"] == (None,)

    pi = pi_backend(tmp_path, lambda n, s, m: MARKER_RULES)
    result = run_backend_step(pi, batch(), pi.initial_state())
    assert result.metrics["candidate_paths"] == ({"stages": [], "reason": None},)


# -- the episode seam -------------------------------------------------------


def test_run_episode_keeps_the_trajectory_before_the_root_is_removed(tmp_path: Path) -> None:
    descriptor = get_adapter("pi")
    files = render_composition([("rules", {"text": "keep me"}), *MODEL.compose_nodes(descriptor)], descriptor)
    keep = tmp_path / "kept"
    result = run_episode(descriptor, files, "task", binary=str(make_binary(tmp_path)), keep_dir=keep)
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
        run_episode(descriptor, files, "task", binary=str(make_binary(tmp_path)), keep_dir=taken)
    assert (taken / "session.jsonl").read_text() == "earlier record\n"


# -- the setting ------------------------------------------------------------


def test_recipe_parses_step_record_dir_and_the_backend_refuses_an_unwritable_one(tmp_path: Path) -> None:
    def config(**evolution):
        return {"evolution": {"propose": lambda n, s, m: None, "evaluate": evaluate, "tasks": ["one"], **evolution}}

    on = CordisRecipe.from_environment({}, config=config(step_record_dir=str(tmp_path / "record")), runtime=runtime())
    assert on.step_record_dir == str(tmp_path / "record")
    assert CordisRecipe.from_environment({}, config=config()).step_record_dir is None
    for bad in (42, "", "  "):
        with pytest.raises(RecipeConfigError, match="step_record_dir must be a non-empty path"):
            CordisRecipe.from_environment({}, config=config(step_record_dir=bad))

    from reef.records import RecordStore

    # One recipe serves many scenarios; each scenario's steps record under its own directory.
    backend = on.build("demo", RecordStore()).training_backend
    assert isinstance(backend, CordisBackend) and backend._step_record_dir == tmp_path / "record" / "demo"
    other = on.build("other", RecordStore()).training_backend
    assert isinstance(other, CordisBackend) and other._step_record_dir == tmp_path / "record" / "other"
    assert sorted(path.name for path in (tmp_path / "record").iterdir()) == ["demo", "other"]

    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    with pytest.raises(ValueError, match=r"step_record_dir .* cannot be created"):
        native_backend(tmp_path, step_record_dir=blocker / "record")
    with pytest.raises(ValueError, match="step_record_dir must be a non-empty path"):
        native_backend(tmp_path, step_record_dir="")


def test_a_relative_record_dir_is_made_absolute_at_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from reef.records import RecordStore

    monkeypatch.chdir(tmp_path)
    config = {
        "evolution": {
            "propose": lambda n, s, m: None,
            "evaluate": evaluate,
            "tasks": ["one"],
            "step_record_dir": "rel",
        }
    }
    recipe = CordisRecipe.from_environment({}, config=config, runtime=runtime())
    backend = recipe.build("demo", RecordStore()).training_backend
    assert isinstance(backend, CordisBackend)
    assert backend._step_record_dir == (tmp_path / "rel" / "demo").resolve() and backend._step_record_dir.is_absolute()


# -- what the boundary never saw, the raw request seam, and the episode record ---------------------


def test_record_redacts_credential_shaped_text_from_the_proposer_and_the_refused_proposal(tmp_path: Path) -> None:
    key = "sk-" + "a" * 24

    class Leaky(_ChatBinding):
        def chat(self, *args, **kwargs) -> str:
            return f"use {key} for auth"

    def propose(nodes, samples, models):
        models.served.chat([{"role": "user", "content": f"the key is {key}"}])
        return Mutation("create", "leak", {"name": "rules", "config": {"text": f"always send {key}"}})

    b = pi_backend(tmp_path, propose, models=ModelBindings(served=Leaky()), step_record_dir=tmp_path / "record")
    result = run_backend_step(b, batch(), b.initial_state())
    assert "inline credential" in str(result.metrics.get("skipped"))  # the tree boundary refused it
    step = tmp_path / "record" / "1"
    written = (step / "proposer.json").read_text() + (step / "mutations.json").read_text()
    assert key not in written and written.count("[redacted credential]") == 3


def test_a_methods_complete_call_shares_the_budget_and_the_record(tmp_path: Path) -> None:
    class Raw(_ChatBinding):
        def complete(self, body, *, timeout_s=None):
            content = "raw " + body["messages"][-1]["content"]
            return {"choices": [{"message": {"role": "assistant", "content": content}}]}

    def propose(nodes, samples, models):
        models.served.complete({"messages": [{"role": "user", "content": "one"}]}, timeout_s=3.0)
        with pytest.raises(RuntimeError, match="budget"):
            models.served.complete({"messages": [{"role": "user", "content": "two"}]})
        return MARKER_RULES

    b = pi_backend(
        tmp_path,
        propose,
        models=ModelBindings(served=Raw()),
        step_record_dir=tmp_path / "record",
        max_model_calls_per_step=1,
    )
    result = run_backend_step(b, batch(), b.initial_state())
    assert result.metrics["proposer_calls"] == 1
    calls = read_json(tmp_path / "record" / "1" / "proposer.json")
    assert calls[0]["body"] == {"messages": [{"role": "user", "content": "one"}]}
    assert calls[0]["params"] == {"timeout_s": 3.0} and "reply" not in calls[0]
    assert calls[0]["response"]["choices"][0]["message"]["content"] == "raw one"


def test_a_long_reply_is_clipped_with_the_marker(tmp_path: Path) -> None:
    class Verbose(_ChatBinding):
        def chat(self, *args, **kwargs) -> str:
            return "y" * (RECORD_TEXT_CAP + 3)

    def propose(nodes, samples, models):
        models.served.chat([{"role": "user", "content": "go"}])
        return MARKER_RULES

    b = pi_backend(tmp_path, propose, models=ModelBindings(served=Verbose()), step_record_dir=tmp_path / "record")
    run_backend_step(b, batch(), b.initial_state())
    reply = read_json(tmp_path / "record" / "1" / "proposer.json")[0]["reply"]
    assert reply.startswith("y" * RECORD_TEXT_CAP) and reply.endswith("... [clipped 3 chars]")


def test_every_episode_leaves_a_record_the_scorer_can_be_replayed_from(tmp_path: Path) -> None:
    b = native_backend(tmp_path, step_record_dir=tmp_path / "record")
    result = run_backend_step(b, batch(), b.initial_state())
    record = read_json(tmp_path / "record" / "1" / "episodes" / "candidate-0" / "episode.json")
    assert record["task"] == "task one" and record["exit_code"] == 0 and record["failure"] is None
    assert record["score"] == 1.0 and result.metrics["selected"] is True
    assert record["path"] == result.metrics["candidate_paths"][0]
    assert record["residue"] == [] and isinstance(record["stdout"], str) and isinstance(record["stderr"], str)

    # An episode that could not run still leaves its record: the failure, and no output to show.
    broken = native_backend(tmp_path, binary=str(tmp_path / "missing"), step_record_dir=tmp_path / "broken")
    run_backend_step(broken, batch(), broken.initial_state())
    record = read_json(tmp_path / "broken" / "1" / "episodes" / "candidate-0" / "episode.json")
    assert record["score"] is None and record["failure"]["stage"] == "launch"
    assert record["exit_code"] is None and record["stdout"] is None and record["residue"] is None
