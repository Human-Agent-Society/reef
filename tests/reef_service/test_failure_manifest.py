"""Guard tests for the failure manifest (issue #475): the state schema is
pinned field by field, fingerprints are stable across processes and temp
directories, the diff classes follow the streak rules, every pre-manifest
proposer shape keeps working, and the manifest rides the commit protocol
from one step's settlement to the next step's propose."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reef.artifact import LiveWeightArtifactRef
from reef.harness.adapters import get_adapter
from reef.scenario.commit_log import CommitLog, CommitRecord
from reef.train.cordis_backend import CordisBackend, FailureManifest, FailureRecord, Mutation
from reef.train.cordis_backend.manifest import MANIFEST_KIND, FailureObservation, advance, fingerprint, normalize_cause
from reef.train.cordis_backend.strategies import Proposer, resolve_episode_scorer, resolve_proposer

from .test_harness_recipe import MODEL, RULES, batch, evaluate, make_binary, run_backend_step

LAUNCH_CAUSE = "harness binary '/tmp/pytest-123/no-such-binary' not found"
LAUNCH_FINGERPRINT = "75b82de293902cee"


def backend(binary: str, propose) -> CordisBackend:
    return CordisBackend(
        descriptor=get_adapter("pi"),
        propose=resolve_proposer(propose),
        score_episode=resolve_episode_scorer(evaluate),
        tasks=("task one",),
        models=MODEL,
        binary=binary,
    )


# -- schema pin ------------------------------------------------------------


def test_state_schema_is_pinned_field_by_field() -> None:
    manifest = advance(None, 3, (FailureObservation("task one", "launch", LAUNCH_CAUSE),))
    assert manifest.to_state() == {
        "kind": "reef-failure-manifest/1",
        "step": 3,
        "entries": [
            {
                "fingerprint": LAUNCH_FINGERPRINT,
                "task": "task one",
                "stage": "launch",
                "cause": "harness binary '<path>' not found",
                "count": 1,
                "first_seen_step": 3,
                "last_seen_step": 3,
            }
        ],
        "fixed": [],
    }
    assert FailureManifest.from_state(manifest.to_state()) == manifest


def test_state_survives_the_commit_log_json_round_trip_byte_for_byte() -> None:
    previous = advance(None, 1, (FailureObservation("task one", "launch", LAUNCH_CAUSE),))
    manifest = advance(previous, 2, (FailureObservation("task one", "exit", "exit 2: boom"),))
    line = json.dumps(manifest.to_state(), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    decoded = json.loads(line)
    assert json.dumps(decoded, ensure_ascii=False, separators=(",", ":"), sort_keys=True) == line
    assert FailureManifest.from_state(decoded) == manifest


def test_unknown_state_kind_is_rejected() -> None:
    state = advance(None, 1, ()).to_state()
    state["kind"] = "reef-failure-manifest/0"
    with pytest.raises(ValueError, match="reef-failure-manifest/0"):
        FailureManifest.from_state(state)


# -- fingerprint pin -------------------------------------------------------


def test_fingerprint_digest_is_pinned() -> None:
    assert fingerprint("task one", "launch", LAUNCH_CAUSE) == LAUNCH_FINGERPRINT


def test_normalization_collapses_volatile_paths_and_numbers() -> None:
    a = "pi session /tmp/reef-episode-pi-abc12/session.jsonl has a corrupt event at line 3"
    b = "pi session /var/folders/xy/reef-episode-pi-zz9/session.jsonl has a corrupt event at line 41"
    assert normalize_cause(a) == "pi session <path> has a corrupt event at line <n>"
    assert normalize_cause(a) == normalize_cause(b)
    assert fingerprint("task one", "trajectory", a) == fingerprint("task one", "trajectory", b)


def test_fingerprint_separates_task_and_stage() -> None:
    assert fingerprint("task one", "launch", LAUNCH_CAUSE) != fingerprint("task two", "launch", LAUNCH_CAUSE)
    assert fingerprint("task one", "launch", LAUNCH_CAUSE) != fingerprint("task one", "exit", LAUNCH_CAUSE)


# -- diff classes ----------------------------------------------------------


def observation(task: str) -> FailureObservation:
    return FailureObservation(task, "launch", LAUNCH_CAUSE)


def test_diff_classes_and_streaks() -> None:
    first = advance(None, 1, (observation("a"), observation("b")))
    assert {record.task for record in first.new} == {"a", "b"}
    assert first.persisting == ()
    assert first.fixed == ()

    second = advance(first, 2, (observation("a"), observation("c")))
    assert {record.task for record in second.new} == {"c"}
    assert {record.task for record in second.persisting} == {"a"}
    assert {record.task for record in second.fixed} == {"b"}
    persisting = next(record for record in second.entries if record.task == "a")
    assert (persisting.count, persisting.first_seen_step, persisting.last_seen_step) == (2, 1, 2)
    fixed = next(iter(second.fixed))
    assert (fixed.count, fixed.first_seen_step, fixed.last_seen_step) == (1, 1, 1)

    # A fingerprint that reappears after a fixed step restarts as new.
    third = advance(second, 3, (observation("b"),))
    reappeared = next(iter(third.entries))
    assert reappeared.task == "b"
    assert (reappeared.count, reappeared.first_seen_step, reappeared.last_seen_step) == (1, 3, 3)
    assert {record.task for record in third.fixed} == {"a", "c"}


def test_repeated_observation_in_one_step_accumulates_count() -> None:
    manifest = advance(None, 1, (observation("a"), observation("a")))
    assert len(manifest.entries) == 1
    assert manifest.entries[0].count == 2


def test_entries_serialize_sorted_by_fingerprint() -> None:
    manifest = advance(None, 1, (observation("b"), observation("a"), observation("c")))
    fingerprints = [entry["fingerprint"] for entry in manifest.to_state()["entries"]]
    assert fingerprints == sorted(fingerprints)


# -- proposer compatibility ------------------------------------------------


def manifest_state() -> dict:
    return advance(None, 1, (observation("task one"),)).to_state()


def test_three_argument_function_and_lambda_run_against_manifest_state(tmp_path: Path) -> None:
    def propose(nodes, samples, models):
        return None

    for proposer in (propose, lambda nodes, samples, models: None):
        b = backend(str(make_binary(tmp_path)), proposer)
        result = run_backend_step(b, batch(), {"steps": 1, "entries": [], "failure_manifest": manifest_state()})
        assert result.metrics["skipped"] == "no proposal"  # the callable ran
        assert result.state["failure_manifest"] == manifest_state()  # a skip carries the manifest untouched


def test_pre_manifest_proposer_subclass_runs_against_manifest_state(tmp_path: Path) -> None:
    calls = []

    class OldStyleProposer(Proposer):
        def __call__(self, nodes, samples, models):  # type: ignore[override]  # deliberate pre-manifest signature
            calls.append(nodes)
            return

    b = backend(str(make_binary(tmp_path)), OldStyleProposer())
    result = run_backend_step(b, batch(), {"steps": 1, "entries": [], "failure_manifest": manifest_state()})
    assert calls == [()]
    assert result.metrics["skipped"] == "no proposal"


def test_manifest_keyword_receives_none_when_the_state_has_no_key(tmp_path: Path) -> None:
    received = []

    def propose(nodes, samples, models, *, manifest=None):
        received.append(manifest)
        return

    b = backend(str(make_binary(tmp_path)), propose)
    result = run_backend_step(b, batch(), b.initial_state())
    assert received == [None]
    assert "failure_manifest" not in result.state  # absence stays unknown, never an empty manifest


def test_var_keyword_proposer_receives_the_manifest(tmp_path: Path) -> None:
    received = []

    def propose(nodes, samples, models, **kwargs):
        received.append(kwargs["manifest"])
        return

    b = backend(str(make_binary(tmp_path)), propose)
    run_backend_step(b, batch(), {"steps": 1, "entries": [], "failure_manifest": manifest_state()})
    assert received == [FailureManifest.from_state(manifest_state())]


# -- the manifest through steps, the commit protocol, and recovery ---------


def failing_step(binary: str, received: list[FailureManifest | None], state: dict):
    def propose(nodes, samples, models, *, manifest=None):
        received.append(manifest)
        return Mutation("create", f"r{len(received)}", RULES)

    return run_backend_step(backend(binary, propose), batch(), state)


def test_manifest_settles_persists_and_fixes_across_real_steps(tmp_path: Path) -> None:
    received: list[FailureManifest | None] = []
    missing = str(tmp_path / "no-such-binary")

    # Step 1: both sides fail at launch, the step is rejected, and the
    # retained current side's failure becomes the manifest.
    first = failing_step(missing, received, {"steps": 0, "entries": []})
    assert received == [None]
    assert first.metrics["failures"] == {"new": 1, "persisting": 0, "fixed": 0}
    entry = FailureManifest.from_state(first.state["failure_manifest"]).entries[0]
    assert entry.task == "task one"
    assert entry.stage == "launch"
    assert entry.cause == "harness binary '<path>' not found"
    assert (entry.count, entry.first_seen_step, entry.last_seen_step) == (1, 1, 1)

    # Step 2 on the committed state: propose observes the failure, and the
    # unchanged failure persists with its streak extended.
    second = failing_step(missing, received, dict(first.state))
    assert isinstance(received[1], FailureManifest)
    assert received[1].entries[0].count == 1
    assert second.metrics["failures"] == {"new": 0, "persisting": 1, "fixed": 0}
    persisted = FailureManifest.from_state(second.state["failure_manifest"]).entries[0]
    assert (persisted.count, persisted.first_seen_step, persisted.last_seen_step) == (2, 1, 2)

    # Step 3 with a working binary: no episode fails, so the entry is fixed.
    third = failing_step(str(make_binary(tmp_path)), received, dict(second.state))
    assert received[2] == FailureManifest.from_state(second.state["failure_manifest"])
    assert third.metrics["failures"] == {"new": 0, "persisting": 0, "fixed": 1}
    settled = FailureManifest.from_state(third.state["failure_manifest"])
    assert settled.entries == ()
    assert settled.fixed[0].count == 2


def test_manifest_recovers_through_the_commit_log(tmp_path: Path) -> None:
    received: list[FailureManifest | None] = []
    first = failing_step(str(tmp_path / "no-such-binary"), received, {"steps": 0, "entries": []})

    log = CommitLog(tmp_path / "commit.jsonl")
    log.append(
        CommitRecord(
            scenario="demo",
            step=1,
            artifact_ref=LiveWeightArtifactRef(
                content_id="live:1", release_id="live:demo:w1:1", parent_release_id="base", runtime_load_id="w1"
            ),
            checkpoint=False,
            algorithm_state=dict(first.state),
            high_water_sequence=1,
            high_water_offset=1,
        )
    )
    recovered = log.records()[-1].algorithm_state
    assert recovered is not None
    assert recovered["failure_manifest"] == first.state["failure_manifest"]

    failing_step(str(tmp_path / "no-such-binary"), received, dict(recovered))
    assert received[1] == FailureManifest.from_state(first.state["failure_manifest"])


def test_commit_record_without_failure_manifest_still_drives_a_step(tmp_path: Path) -> None:
    # Algorithm state may carry only the lineage keys. It must parse and step,
    # delivering no previous failure manifest.
    line = json.dumps(
        {
            "record": "reef-commit/5",
            "scenario": "demo",
            "step": 1,
            "artifact_ref": {
                "kind": "live_weights",
                "content_id": "live:1",
                "release_id": "live:demo:w1:1",
                "parent_release_id": "base",
                "runtime_load_id": "w1",
            },
            "checkpoint": False,
            "algorithm_state": {"steps": 1, "entries": []},
            "record_progress": {
                "high_water_sequence": 1,
                "high_water_offset": 1,
                "compacted_ids": [],
                "consumed_ids": [],
            },
            "recorded_at": 1700000000.0,
        },
        sort_keys=True,
    )
    record = CommitRecord.from_dict(json.loads(line))
    assert record.algorithm_state is not None
    received: list[FailureManifest | None] = []
    result = failing_step(str(make_binary(tmp_path)), received, dict(record.algorithm_state))
    assert received == [None]
    assert isinstance(FailureManifest.from_state(result.state["failure_manifest"]), FailureManifest)


def test_manifest_entry_type_is_exported() -> None:
    record = FailureRecord(
        fingerprint=LAUNCH_FINGERPRINT,
        task="task one",
        stage="launch",
        cause="harness binary '<path>' not found",
        count=1,
        first_seen_step=1,
        last_seen_step=1,
    )
    assert FailureRecord.from_dict(record.to_dict()) == record
    assert MANIFEST_KIND == "reef-failure-manifest/1"


def test_normalize_cause_masks_opaque_identifier_runs() -> None:
    cause = normalize_cause("refused key sk-live-Abc123XyzLongSecretRun447 request 8f3a2bc190de77aa41c2")
    assert "<id>" in cause
    assert "Abc123" not in cause and "8f3a2bc1" not in cause
    assert normalize_cause("harness binary not found: pi").endswith("pi")
