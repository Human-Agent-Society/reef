"""Guard tests for the failure manifest (issue #475): the state schema is
pinned field by field, fingerprints are stable across processes and temp
directories, the diff classes follow the streak rules, the proposer reads the
committed manifest from its context, and the manifest rides the committer
from one step's settlement to the next step's proposal."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from tests.support._trajectories import recorded_trajectory
from tests.support.harness_service import backend_step
from tests.support.harness_steps import COMPONENT, TreeReleases, changed_rules, harness_recipe

from reef.artifact import ArtifactRef
from reef.evaluation.harness import FailureObservation
from reef.harness.tree import TreePatch
from reef.storage.commits import CommitRecord
from reef.storage.sqlite import SQLiteRecordStore
from reef.train.backend import ComponentReleases
from reef.train.harness_backend import FailureManifest, FailureRecord
from reef.train.harness_backend.backend import HarnessBackend
from reef.train.harness_backend.manifest import MANIFEST_KIND, advance, fingerprint, normalize_cause
from reef.train.types import TrainingBatch, TrainStepResult

LAUNCH_CAUSE = "harness binary '/tmp/pytest-123/no-such-binary' not found"
LAUNCH_FINGERPRINT = "75b82de293902cee"


def backend(tmp_path: Path, propose, binary: str | None = None) -> HarnessBackend:
    """A pi backend over the fake binary, or over ``binary`` when given, whose head is the seed tree."""
    recipe = harness_recipe(tmp_path, propose)
    if binary is not None:
        recipe = replace(recipe, binary=binary)
    return recipe.build_backend("demo", ComponentReleases(TreeReleases(recipe.seed), COMPONENT), None)


def batch() -> TrainingBatch:
    return TrainingBatch("demo:trace:1", (recorded_trajectory("a1", {"messages": []}, 0.0),))


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


# -- the proposer's view ---------------------------------------------------


def manifest_state() -> dict:
    return advance(None, 1, (observation("task one"),)).to_state()


def test_the_proposer_reads_the_committed_manifest_from_its_context(tmp_path: Path) -> None:
    received = []

    def propose(context):
        received.append(context.manifest)
        return

    b = backend(tmp_path, propose)
    first = backend_step(b, batch(), b.initial_state())
    assert received == [None]
    assert first.state["failure_manifest"] is None  # absence stays unknown, never an empty manifest
    result = backend_step(b, batch(), {"steps": 1, "failure_manifest": manifest_state()})
    assert received[1] == FailureManifest.from_state(manifest_state())
    assert result.metrics["skipped"] == "no proposal"
    assert result.state["failure_manifest"] == manifest_state()  # a skip carries the manifest untouched


# -- the manifest through steps, the committer, and recovery ---------


def failing_step(
    tmp_path: Path, binary: str | None, received: list[FailureManifest | None], state: dict
) -> TrainStepResult:
    """One step whose proposal rewrites the rules, over ``binary`` (the fake pi when ``None``)."""

    def propose(context):
        received.append(context.manifest)
        return TreePatch(changed_rules(f"rules {len(received)}\n"))

    return backend_step(backend(tmp_path, propose, binary), batch(), state)


def test_manifest_settles_persists_and_fixes_across_real_steps(tmp_path: Path) -> None:
    received: list[FailureManifest | None] = []
    missing = str(tmp_path / "no-such-binary")

    # Step 1: both sides fail at launch, the step is rejected, and the
    # retained current side's failure becomes the manifest.
    first = failing_step(tmp_path, missing, received, {"steps": 0})
    assert received == [None]
    assert first.metrics["failures"] == {"new": 1, "persisting": 0, "fixed": 0}
    entry = FailureManifest.from_state(first.state["failure_manifest"]).entries[0]
    assert entry.task == "task one"
    assert entry.stage == "launch"
    assert entry.cause == "harness binary '<path>' not found"
    assert (entry.count, entry.first_seen_step, entry.last_seen_step) == (1, 1, 1)

    # Step 2 on the committed state: propose observes the failure, and the
    # unchanged failure persists with its streak extended.
    second = failing_step(tmp_path, missing, received, dict(first.state))
    assert isinstance(received[1], FailureManifest)
    assert received[1].entries[0].count == 1
    assert second.metrics["failures"] == {"new": 0, "persisting": 1, "fixed": 0}
    persisted = FailureManifest.from_state(second.state["failure_manifest"]).entries[0]
    assert (persisted.count, persisted.first_seen_step, persisted.last_seen_step) == (2, 1, 2)

    # Step 3 with a working binary: no episode fails, so the entry is fixed.
    third = failing_step(tmp_path, None, received, dict(second.state))
    assert received[2] == FailureManifest.from_state(second.state["failure_manifest"])
    assert third.metrics["failures"] == {"new": 0, "persisting": 0, "fixed": 1}
    settled = FailureManifest.from_state(third.state["failure_manifest"])
    assert settled.entries == ()
    assert settled.fixed[0].count == 2


def test_manifest_recovers_through_the_commit_history(tmp_path: Path) -> None:
    received: list[FailureManifest | None] = []
    first = failing_step(tmp_path, str(tmp_path / "no-such-binary"), received, {"steps": 0})

    with SQLiteRecordStore(tmp_path / "records.sqlite3") as records:
        log = records.commit_history()
        log.append(
            CommitRecord(
                scenario="demo",
                step=1,
                artifact_ref=ArtifactRef(
                    content_id="content:1", release_id="release:demo:1", parent_release_id="base"
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

    failing_step(tmp_path, str(tmp_path / "no-such-binary"), received, dict(recovered))
    assert received[1] == FailureManifest.from_state(first.state["failure_manifest"])


def test_commit_record_without_failure_manifest_still_drives_a_step(tmp_path: Path) -> None:
    # Algorithm state may carry only the step counter. It must parse and step,
    # delivering no previous failure manifest.
    line = json.dumps(
        {
            "record": "reef-commit/5",
            "scenario": "demo",
            "step": 1,
            "artifact_ref": {
                "kind": "artifact",
                "content_id": "content:1",
                "release_id": "release:demo:1",
                "parent_release_id": "base",
            },
            "checkpoint": False,
            "algorithm_state": {"steps": 1},
            "record_progress": {
                "high_water_sequence": 1,
                "high_water_offset": 1,
                "consumed_ids": [],
            },
            "recorded_at": 1700000000.0,
        },
        sort_keys=True,
    )
    record = CommitRecord.from_dict(json.loads(line))
    assert record.algorithm_state is not None
    received: list[FailureManifest | None] = []
    result = failing_step(tmp_path, None, received, dict(record.algorithm_state))
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
