"""Guarantees of the ``tutorials/evolve-your-harness`` runner, hermetic: no service, no model, no episodes.

The recipe batches valid scored reports -- passing ones included -- and ``batch_size: 1`` makes each report its
own batch, so every report the runner submits triggers one evolve step whatever its score was. The number of
failing reports is therefore not the number of steps the run triggers, and an all-passing run is not a run that
batched nothing: it waits for three verdicts like any other. A run whose steps all ended rejected or skipped ends
on those verdicts instead of waiting on a head change that cannot come.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TUTORIAL = REPO_ROOT / "tutorials" / "evolve-your-harness"
#: The three tasks serve.yaml carries; the leading key is what the tutorial grader's answer table reads.
TASKS = (
    "[sieve] how many primes are below 100000?",
    "[fib] compute fib(90) exactly",
    "[csv] compute the median of the value column",
)

#: The verdicts the issue's all-passing run recorded: one rejection on a 0 win / 0 loss / 3 tie gate, then two
#: steps the proposer gave no proposal. None of them publishes, so the head stays on the seed.
REJECTED = {"wins": 0, "losses": 0, "ties": 3, "candidate_score": 3.0, "current_score": 3.0}
SKIPPED = {"skipped": "no proposal"}
NO_PUBLISH = (REJECTED, SKIPPED, SKIPPED)
PUBLISHED = {"published": True, "wins": 1, "losses": 0, "ties": 2, "candidate_score": 3.0, "current_score": 2.0}
#: The release a fresh scenario serves before any step publishes, and the one a publishing step leaves behind.
SEED_RELEASE = "seed-release"
PUBLISHED_RELEASE = "published-release"


class StubReefClient:
    """A ``ReefClient`` stand-in: it records every report the run submits and every route the runner reads.

    The catalog answers with one training row per report already submitted, in order: a step's row exists only
    once the report that triggered it has been accepted, which is what the service commits. The rows carry the
    verdicts the caller handed in, so a run can end with every step complete and none of them published.

    The head moves at the commit, and the runner learns about that commit from the catalog: until a catalog read
    has shown a published row, ``/reef/harness`` serves the seed. That is what makes a runner that reads the head
    before it waits for the verdicts see the seed and no further.

    ``reads`` is every route read the runner makes, in order, as ``("catalog", {"rows": n, "published": bool})``
    for ``/reef/harness/releases`` and ``("manifest", release_id)`` for ``/reef/harness``. It says whether the
    manifest was read because a verdict was on the record, or polled before there was one to read.
    """

    def __init__(
        self, *args: object, verdicts: tuple[dict, ...] = (), head: str = PUBLISHED_RELEASE, **kwargs: object
    ) -> None:
        self.verdicts = list(verdicts)
        self.head = head
        self.reports: list[dict] = []
        self.requested: list[str] = []
        self.reads: list[tuple[str, dict | str]] = []
        self.turns: list[str] = []
        self.mounts: list[str] = []
        self.published_seen = False

    def _step_rows(self) -> list[dict]:
        return [
            {"release_id": f"release-{index}", "operation": "training", "metrics": {"steps": index, **verdict}}
            for index, verdict in enumerate(self.verdicts[: len(self.reports)], start=1)
        ]

    def inference_with_record(self, scenario: str, path: str, payload: dict, **kwargs: object) -> tuple[dict, str]:
        return {"choices": [{"message": {"content": "9592"}}]}, "receipt-1"

    def report(self, scenario: str, payload: dict, **kwargs: object) -> dict:
        self.reports.append(payload)
        return {"agent_record_id": payload["agent_record_id"], "scenario": scenario, "request_type": "report"}

    def get(self, path: str, **kwargs: object) -> dict:
        self.requested.append(path)
        if path == "/reef/harness/releases":
            rows = self._step_rows()
            published = any(row["metrics"].get("published") for row in rows)
            self.published_seen = self.published_seen or published
            self.reads.append(("catalog", {"rows": len(rows), "published": published}))
            return {"releases": rows}
        if path == "/reef/harness":
            release = self.head if self.published_seen else SEED_RELEASE
            self.reads.append(("manifest", release))
            return {
                "release_id": release,
                "parent_release_id": SEED_RELEASE,
                "evaluation": {},
                "files": {},
            }
        return {}


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """``tutorials/evolve-your-harness/run.py`` as a module: it is a script, so it is loaded from its path."""
    # run.py imports the tutorial's own `harness` package, which sits beside it and is on no import path.
    monkeypatch.syspath_prepend(str(TUTORIAL))
    # Other examples also import a package named harness; the tutorial path must win over their cached imports.
    for name in [name for name in sys.modules if name == "harness" or name.startswith("harness.")]:
        sys.modules.pop(name)
    spec = importlib.util.spec_from_file_location("harness_evolve_run", TUTORIAL / "run.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("run.py could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stage(
    runner: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    scores: tuple[float, ...],
    verdicts: tuple[dict, ...],
    head: str = PUBLISHED_RELEASE,
) -> StubReefClient:
    """Point the runner at ``scores``, one per task, and at ``verdicts`` for the steps those reports trigger."""
    tasks_file = tmp_path / "tasks.json"
    tasks_file.write_text(json.dumps(list(TASKS[: len(scores)])), encoding="utf-8")
    monkeypatch.setattr(runner, "TASKS_FILE", tasks_file)
    client = StubReefClient(verdicts=verdicts, head=head)
    monkeypatch.setattr(runner, "ReefClient", lambda *args, **kwargs: client)
    graded = iter(scores)

    def grade(task, text):
        # The native form grades a second pass after the run's own tasks, which the scores do not cover.
        return next(graded, scores[-1])

    monkeypatch.setattr(runner.evolution, "grade_text", grade)
    return client


def _record_waits(runner: ModuleType, monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Replace the wait with a recorder of the step count the runner asks it to wait for."""
    waits: list[int] = []

    def wait(client, before, expected, deadline):
        waits.append(expected)
        return []

    monkeypatch.setattr(runner, "_wait_for_steps", wait)
    return waits


def _stage_native(runner: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, client: StubReefClient) -> None:
    """The native form's surroundings: a pulled seed tree, and the turn and wrapper calls stubbed out.

    The wrapper's ``report`` command posts the score outside the client, so the stub records it the same way:
    the step rows the catalog answers with appear once the run has reported a turn, as they do in the service.
    """
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / runner.RELEASE_FILE).write_text(json.dumps({"release_id": SEED_RELEASE}), encoding="utf-8")
    monkeypatch.setattr(runner, "TREE_DIR", tree)
    monkeypatch.setattr(runner, "SCRATCH_DIR", tmp_path / "scratch")
    monkeypatch.setattr(runner, "report", lambda score, feedback: client.reports.append({"score": score}))

    def turn(prompt, session=None):
        client.turns.append(prompt)
        return [], {"text": "9592", "session": "session-1", "exit": 0}

    def mount_events(release_id):
        client.mounts.append(release_id)
        return [("native/sessions/serve.jsonl", {"release_id": release_id})]

    monkeypatch.setattr(runner, "turn", turn)
    monkeypatch.setattr(runner, "_mount_events", mount_events)


def test_an_all_passing_run_waits_for_the_step_every_report_it_submitted_batches(
    runner: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Three passing reports batch three steps: the runner waits for three, and does not return as batched-nothing."""
    client = _stage(runner, monkeypatch, tmp_path, (1.0, 1.0, 1.0), NO_PUBLISH)
    waits = _record_waits(runner, monkeypatch)

    runner.main()

    printed = capsys.readouterr().out
    assert len(client.reports) == 3, "three tasks go through the runner, so three reports are submitted"
    assert waits == [3], f"three accepted reports batch three steps; the runner asked for {waits}"
    assert "nothing batched" not in printed and "no evolve step runs" not in printed
    assert "3 report(s) batched (0 of them failing)" in printed


def test_the_expected_step_count_is_the_reports_submitted_not_the_failures(
    runner: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """One failing report beside two passing ones still batches three steps: a failure count is not a step count."""
    client = _stage(runner, monkeypatch, tmp_path, (1.0, 0.0, 1.0), NO_PUBLISH)
    waits = _record_waits(runner, monkeypatch)

    runner.main()

    printed = capsys.readouterr().out
    assert len(client.reports) == 3
    assert waits == [3], f"every accepted report batches a step; the runner asked for {waits}, not 1"
    assert "3 report(s) batched (1 of them failing)" in printed


def test_completed_steps_that_publish_nothing_end_the_run_without_reading_the_head(
    runner: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An all-passing run's three steps can all end rejected or skipped: the run ends on their verdicts.

    Nothing publishes, so there is no head change to wait for: the runner must neither sit out the publication
    deadline nor pull the manifest a win would have left behind.
    """
    client = _stage(runner, monkeypatch, tmp_path, (1.0, 1.0, 1.0), NO_PUBLISH)

    started = time.monotonic()
    runner.main()
    elapsed = time.monotonic() - started
    printed = capsys.readouterr().out

    assert elapsed < 30.0, f"the run waited {elapsed:.1f}s for a head change no step could publish"
    assert "nothing batched" not in printed and "no evolve step runs" not in printed
    assert "step 1: rejected" in printed, "the rejected gate is on the record before the run ends"
    assert "step 2: no proposal" in printed and "step 3: no proposal" in printed
    assert "no skill mutation won a gate" in printed
    assert "/reef/harness" not in client.requested, "no step published, so the run must not pull the head"


def test_a_publishing_step_still_reaches_the_manifest(
    runner: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The same wait ends at the manifest when one of this run's steps did publish: the pull path still runs."""
    client = _stage(runner, monkeypatch, tmp_path, (1.0, 1.0, 1.0), (PUBLISHED, SKIPPED, SKIPPED))

    runner.main()

    printed = capsys.readouterr().out
    assert "/reef/harness" in client.requested, "a step published, so the run reads the head it left behind"
    assert "published: artifact published-release" in printed


def test_the_native_run_stops_on_the_verdicts_of_steps_that_publish_nothing(
    runner: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The native form waited on the head alone, so a run whose steps publish nothing sat out the whole deadline."""
    client = _stage(runner, monkeypatch, tmp_path, (1.0, 0.0, 1.0), NO_PUBLISH)
    _stage_native(runner, monkeypatch, tmp_path, client)
    monkeypatch.setattr(runner, "PULL_TIMEOUT_S", 5.0)

    started = time.monotonic()
    runner.native_main()
    elapsed = time.monotonic() - started
    printed = capsys.readouterr().out

    assert elapsed < 2.0, f"the native run waited {elapsed:.1f}s on a head change no step could publish"
    assert "nothing batched" not in printed and "no evolve step runs" not in printed
    assert "3 report(s) batched (1 of them failing)" in printed
    assert "no mutation won a gate" in printed
    assert "/reef/harness" not in client.requested, "no step published, so the run must not follow the head"


def test_the_native_run_follows_a_published_release_through_to_the_second_pass(
    runner: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A native run that does publish: the verdicts come first, then the head, the mount and the second pass.

    The manifest the service serves is a genuinely new release, not the seed, so the run has to reach the
    published-release path on the strength of this run's step verdicts rather than on a head that never moved.

    The contract is an ordering one, so it is asserted on the recorded route reads, not on a clock: the catalog
    has to expose the published row before ``/reef/harness`` is read at all, and the head is then read once. A
    runner that polls the head waiting for it to move reads the manifest first, and reads it repeatedly, which
    this ordering rules out without measuring how long the run took. ``PULL_TIMEOUT_S`` is shortened only so
    such a regression fails here instead of blocking on the real deadline.
    """
    client = _stage(
        runner, monkeypatch, tmp_path, (1.0, 0.0, 1.0), (PUBLISHED, SKIPPED, SKIPPED), head=PUBLISHED_RELEASE
    )
    _stage_native(runner, monkeypatch, tmp_path, client)
    monkeypatch.setattr(runner, "PULL_TIMEOUT_S", 5.0)

    runner.native_main()
    printed = capsys.readouterr().out

    catalog_reads = [(index, read) for index, read in enumerate(client.reads) if read[0] == "catalog"]
    manifest_reads = [(index, read) for index, read in enumerate(client.reads) if read[0] == "manifest"]
    published_at = next((index for index, read in catalog_reads if read[1]["published"]), None)
    assert published_at is not None, f"the run never read a catalog row that published: {client.reads}"
    assert manifest_reads, f"the run never read the head: {client.reads}"
    first_manifest_at, first_manifest = manifest_reads[0]
    assert published_at < first_manifest_at, (
        f"the published row was read at {published_at}, the head at {first_manifest_at}: "
        f"the run polled the head before it had a verdict ({client.reads})"
    )
    assert len(manifest_reads) == 1, f"the head is read once, after the verdicts, not polled: {client.reads}"
    assert (
        first_manifest[1] == PUBLISHED_RELEASE
    ), f"the head the run read is {first_manifest[1]}, not the published one"

    assert len(client.reports) == 3, "three turns, three scores reported through the wrapper"
    assert "step 1: published" in printed, "the catalog returned this run's steps, one of them published"
    assert f"published: artifact {PUBLISHED_RELEASE}" in printed, "the head is a release other than the seed"
    assert set(client.mounts) == {PUBLISHED_RELEASE}, f"the process mounted the published release, saw {client.mounts}"
    assert len(client.turns) == 4, f"the second pass runs after the run's three turns, saw {len(client.turns)}"
    assert "second pass" in printed
    assert "no mutation won a gate" not in printed, "a step published, so the run must not take the no-publish path"
    assert "nothing batched" not in printed and "no evolve step runs" not in printed


def test_the_native_run_waits_for_every_step_of_an_all_passing_run(
    runner: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Three native turns, all passing: three reports batch three steps, and the wait covers all three."""
    client = _stage(runner, monkeypatch, tmp_path, (1.0, 1.0, 1.0), NO_PUBLISH)
    _stage_native(runner, monkeypatch, tmp_path, client)
    waits = _record_waits(runner, monkeypatch)

    runner.native_main()

    printed = capsys.readouterr().out
    assert len(client.reports) == 3, "three turns, three scores reported through the wrapper"
    assert waits == [3], f"three accepted reports batch three steps; the native runner asked for {waits}"
    assert "nothing batched" not in printed and "no evolve step runs" not in printed
    assert "3 report(s) batched (0 of them failing)" in printed
