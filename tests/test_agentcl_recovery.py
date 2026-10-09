"""Reviewed AgentCL recovery fixtures; no model, service, or container calls."""

from __future__ import annotations

import asyncio
import copy
import hashlib
from pathlib import Path
from types import ModuleType

import pytest
from tests import test_sdft_agentcl as sdft_workflow
from tests import test_sdpo_agentcl as sdpo_workflow

from recipes.sdft.examples.agentcl.report import JsonObject


@pytest.fixture(params=[sdft_workflow, sdpo_workflow], ids=["sdft", "sdpo"])
def workflow(request: pytest.FixtureRequest) -> ModuleType:
    return request.param


def prepare(workflow: ModuleType, tmp_path: Path, kind: str) -> tuple[
    sdft_workflow.run.Campaign | sdpo_workflow.run.Campaign,
    JsonObject,
    JsonObject,
    JsonObject,
    Path,
    JsonObject,
    Path,
]:
    arguments = workflow.arguments_fixture(tmp_path)
    manifest = workflow.manifest_fixture(arguments.data_root)
    api = workflow.FakeApi(arguments.attempts)

    class NoReplay(workflow.run.EpisodeBackend):
        async def run(self, task: JsonObject, episode_id: str, phase: str, release: JsonObject) -> JsonObject:
            raise AssertionError("reviewed recovery must not call a model backend")

        def recover(self, episode_id: str) -> JsonObject | None:
            raise AssertionError("reviewed recovery must not query backend recovery")

    campaign = workflow.run.Campaign(arguments, api, NoReplay(), manifest)
    task = {**manifest["training"][0], "campaign_position": 0}
    identifier = workflow.report.stable_id(arguments.run_id, "baseline", task["category"], task["id"], 0, "episode")
    release = copy.deepcopy(api.current)
    recovered = {
        "episode_id": identifier,
        "phase": "baseline",
        "category": task["category"],
        "task_id": task["id"],
        "position": task["position"],
        "campaign_position": 0,
        "attempt": 0,
        "report_id": workflow.report.stable_id(
            arguments.run_id, "baseline", task["category"], task["id"], 0, "report"
        ),
        "release_id": release["release_id"],
        "runtime_load_id": release["runtime_load_id"],
        "outcome": "completed",
        "fault": None,
        "score": 1.0,
        "references": ["native-receipt"],
        "turns": [{"receipt": "native-receipt", "release_id": "base", "runtime_load_id": "runtime-base"}],
        "messages": [{"role": "assistant", "content": "FINAL\n```python\nx = 1\n```"}],
        "trajectory": {"schema_version": "ATIF-v1.7", "session_id": identifier},
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "elapsed_seconds": 0.1,
    }
    original = copy.deepcopy(recovered)
    original.update(outcome="fault", score=None, fault="missing verifier reward or Harbor infrastructure error")
    if kind == "confirmed_zero_call_retry":
        original.update(references=[], turns=[], messages=[], trajectory={}, prompt_tokens=0, completion_tokens=0)
    root = arguments.run_root
    original_path = root / "recovery-originals" / f"{identifier}.json"
    workflow.report.write_object(original_path, original)
    proof = root / "proofs" / "result.json"
    workflow.report.write_object(proof, {"reward": 1.0, "status": "passed", "parent_checked": True})
    answer = root / "proofs" / "answer.py"
    answer.write_text("x = 1\n")
    record = {
        "reviewed": True,
        "recovery_kind": kind,
        "original_artifact": original_path.relative_to(root).as_posix(),
        "original_sha256": hashlib.sha256(original_path.read_bytes()).hexdigest(),
        "sampled_release": release,
        "recovered_episode": recovered,
        "terminal_final_validated": True,
        "answer_blob": {"artifact": "proofs/answer.py", "sha256": hashlib.sha256(answer.read_bytes()).hexdigest()},
        "verifier_result": {
            "artifact": "proofs/result.json",
            "sha256": hashlib.sha256(proof.read_bytes()).hexdigest(),
        },
        "new_trial_row": {"artifact": "proofs/result.json", "sha256": hashlib.sha256(proof.read_bytes()).hexdigest()},
        "harbor_key": f"{identifier}-reviewed-retry",
    }
    recovery_path = root / "episode-recoveries" / f"{identifier}.json"
    workflow.report.write_object(recovery_path, record)
    state = {"started_episodes": [identifier]}
    return campaign, task, release, state, recovery_path, record, original_path


@pytest.mark.parametrize("kind", ["verifier_only_saved_final", "confirmed_zero_call_retry"])
def test_reviewed_recovery_is_canonical_without_backend_replay(
    workflow: ModuleType, tmp_path: Path, kind: str
) -> None:
    campaign, task, release, state, recovery_path, record, original_path = prepare(workflow, tmp_path, kind)
    original_bytes = original_path.read_bytes()
    recovery_bytes = recovery_path.read_bytes()
    result = asyncio.run(campaign.episode(task, "baseline", 0, release, state))
    canonical = campaign.arguments.run_root / "episodes" / f"{result['episode_id']}.json"
    assert result == record["recovered_episode"]
    assert workflow.report.read_object(canonical) == result
    assert asyncio.run(campaign.episode(task, "baseline", 0, release, state)) == result
    assert original_path.read_bytes() == original_bytes
    assert recovery_path.read_bytes() == recovery_bytes
    assert state == {"started_episodes": [result["episode_id"]]}
    assert not campaign.api.reports


@pytest.mark.parametrize("score", [0.0, 1.0])
def test_zero_call_retry_retains_bounded_evaluation_truncation(
    workflow: ModuleType, tmp_path: Path, score: float
) -> None:
    campaign, task, release, state, recovery_path, record, _ = prepare(workflow, tmp_path, "confirmed_zero_call_retry")
    record["recovered_episode"].update(
        outcome="truncated", score=score, fault="maximum student turns reached without a final submission"
    )
    workflow.report.write_object(recovery_path, record)
    if score:
        with pytest.raises(RuntimeError):
            asyncio.run(campaign.episode(task, "baseline", 0, release, state))
    else:
        result = asyncio.run(campaign.episode(task, "baseline", 0, release, state))
        assert result["outcome"] == "truncated" and result["score"] == 0
        assert not campaign.api.reports


def test_cached_fault_is_preserved_and_requires_review(workflow: ModuleType, tmp_path: Path) -> None:
    campaign, task, release, state, recovery_path, record, original_path = prepare(
        workflow, tmp_path, "verifier_only_saved_final"
    )
    canonical = campaign.arguments.run_root / "episodes" / original_path.name
    canonical.parent.mkdir()
    canonical.write_bytes(original_path.read_bytes())
    original_bytes = canonical.read_bytes()
    recovery_path.unlink()
    assert asyncio.run(campaign.episode(task, "baseline", 0, release, state))["outcome"] == "fault"
    workflow.report.write_object(recovery_path, record)
    assert asyncio.run(campaign.episode(task, "baseline", 0, release, state)) == record["recovered_episode"]
    assert canonical.read_bytes() == original_bytes
    completed = copy.deepcopy(record["recovered_episode"])
    completed["score"] = 0.0
    workflow.report.write_object(canonical, completed)
    completed_bytes = canonical.read_bytes()
    with pytest.raises(RuntimeError, match="canonical episode"):
        asyncio.run(campaign.episode(task, "baseline", 0, release, state))
    assert canonical.read_bytes() == completed_bytes


def test_review_hash_identity_receipts_and_runtime_fail_closed(workflow: ModuleType, tmp_path: Path) -> None:
    campaign, task, release, state, recovery_path, record, original_path = prepare(
        workflow, tmp_path, "verifier_only_saved_final"
    )
    changes = [
        ("reviewed", None),
        ("reviewed", 1),
        ("recovery_kind", "automatic_retry"),
        ("original_sha256", "0" * 64),
        ("sampled_release", {**release, "operation": "training"}),
        ("terminal_final_validated", False),
    ]
    for key, value in changes:
        changed = copy.deepcopy(record)
        changed[key] = value
        workflow.report.write_object(recovery_path, changed)
        with pytest.raises(RuntimeError):
            asyncio.run(campaign.episode(task, "baseline", 0, release, state))
    for key in ("episode_id", "task_id", "phase", "attempt", "report_id", "release_id", "runtime_load_id"):
        changed = copy.deepcopy(record)
        changed["recovered_episode"][key] = "changed"
        workflow.report.write_object(recovery_path, changed)
        with pytest.raises(RuntimeError):
            asyncio.run(campaign.episode(task, "baseline", 0, release, state))
    for key in ("verifier_result", "answer_blob"):
        for missing in (True, False):
            changed = copy.deepcopy(record)
            if missing:
                changed.pop(key)
            else:
                changed[key]["sha256"] = "0" * 64
            workflow.report.write_object(recovery_path, changed)
            with pytest.raises((RuntimeError, KeyError)):
                asyncio.run(campaign.episode(task, "baseline", 0, release, state))
    for key, value in (
        ("references", ["native-receipt", "native-receipt"]),
        ("score", 0.5),
        ("score", True),
        ("outcome", "fault"),
    ):
        changed = copy.deepcopy(record)
        changed["recovered_episode"][key] = value
        workflow.report.write_object(recovery_path, changed)
        with pytest.raises(RuntimeError):
            asyncio.run(campaign.episode(task, "baseline", 0, release, state))
    changed = copy.deepcopy(record)
    changed["recovered_episode"]["turns"][0]["runtime_load_id"] = "another-runtime"
    workflow.report.write_object(recovery_path, changed)
    with pytest.raises(RuntimeError, match="runtime binding"):
        asyncio.run(campaign.episode(task, "baseline", 0, release, state))
    workflow.report.write_object(recovery_path, record)
    original_path.write_text(original_path.read_text() + "\n")
    with pytest.raises(RuntimeError, match="checksum"):
        asyncio.run(campaign.episode(task, "baseline", 0, release, state))
    assert not (campaign.arguments.run_root / "episodes").exists()


def test_retry_requires_zero_calls_and_new_physical_key(workflow: ModuleType, tmp_path: Path) -> None:
    campaign, task, release, state, recovery_path, record, original_path = prepare(
        workflow, tmp_path, "confirmed_zero_call_retry"
    )
    original = workflow.report.read_object(original_path)
    for key, value in (
        ("references", ["old-receipt"]),
        ("turns", [{"receipt": "old-receipt"}]),
        ("prompt_tokens", 1),
        ("completion_tokens", 1),
    ):
        workflow.report.write_object(original_path, {**original, key: value})
        changed = {**record, "original_sha256": hashlib.sha256(original_path.read_bytes()).hexdigest()}
        workflow.report.write_object(recovery_path, changed)
        with pytest.raises(RuntimeError, match="zero-call"):
            asyncio.run(campaign.episode(task, "baseline", 0, release, state))
    workflow.report.write_object(original_path, original)
    record["original_sha256"] = hashlib.sha256(original_path.read_bytes()).hexdigest()
    for key, value in (
        ("harbor_key", record["recovered_episode"]["episode_id"]),
        ("new_trial_row", {"artifact": "proofs/result.json", "sha256": "bad"}),
    ):
        workflow.report.write_object(recovery_path, {**record, key: value})
        with pytest.raises(RuntimeError):
            asyncio.run(campaign.episode(task, "baseline", 0, release, state))


def test_recovery_paths_cannot_escape_or_use_symlinks(workflow: ModuleType, tmp_path: Path) -> None:
    campaign, task, release, state, recovery_path, record, original_path = prepare(
        workflow, tmp_path, "verifier_only_saved_final"
    )
    for relative in (str(original_path), "../outside.json", "proofs/result.json"):
        changed = {**record, "original_artifact": relative}
        if relative == "proofs/result.json":
            changed["original_sha256"] = record["verifier_result"]["sha256"]
        workflow.report.write_object(recovery_path, changed)
        with pytest.raises(RuntimeError):
            asyncio.run(campaign.episode(task, "baseline", 0, release, state))
    linked = campaign.arguments.run_root / "proofs" / "linked.json"
    linked.symlink_to(original_path)
    changed = copy.deepcopy(record)
    changed["verifier_result"] = {"artifact": "proofs/linked.json", "sha256": record["original_sha256"]}
    workflow.report.write_object(recovery_path, changed)
    with pytest.raises(RuntimeError, match="symlinks"):
        asyncio.run(campaign.episode(task, "baseline", 0, release, state))
    recovery_path.unlink()
    recovery_path.symlink_to(original_path)
    with pytest.raises(RuntimeError, match="symlinks"):
        asyncio.run(campaign.episode(task, "baseline", 0, release, state))


def prepare_saved_episode(
    workflow: ModuleType, tmp_path: Path, outcome: str = "completed", phase: str = "baseline"
) -> tuple[
    sdft_workflow.run.Campaign | sdpo_workflow.run.Campaign,
    JsonObject,
    JsonObject,
    JsonObject,
    Path,
    JsonObject,
    Path,
]:
    campaign, task, release, state, recovery_path, record, original_path = prepare(
        workflow, tmp_path, "verifier_only_saved_episode"
    )
    root = campaign.arguments.run_root
    source = "def solve():\n    return 1\n"
    result = record["recovered_episode"]
    result["messages"] = [{"role": "assistant", "content": f"FINAL\n```python\n{source}```"}]
    identifier = workflow.report.stable_id(
        campaign.arguments.run_id, phase, task["category"], task["id"], 0, "episode"
    )
    result.update(
        episode_id=identifier,
        phase=phase,
        report_id=workflow.report.stable_id(
            campaign.arguments.run_id, phase, task["category"], task["id"], 0, "report"
        ),
        outcome=outcome,
        score=1.0 if outcome == "completed" else 0.0,
        fault=None if outcome == "completed" else "maximum student turns reached without a final submission",
        metadata={"original": "unchanged"},
    )
    if outcome == "truncated":
        result["messages"] = [{"role": "assistant", "content": "```python\nx = 1\n```"}]
    result["trajectory"]["session_id"] = identifier
    original = copy.deepcopy(result)
    original.update(outcome="fault", score=None, fault="missing verifier reward or Harbor infrastructure error")
    workflow.report.write_object(original_path, original)
    record["original_sha256"] = hashlib.sha256(original_path.read_bytes()).hexdigest()
    snapshot = copy.deepcopy(result)
    snapshot.pop("score")
    snapshot_path = root / "recovery-originals" / "saved-harbor-episode.json"
    workflow.report.write_object(snapshot_path, snapshot)
    record["original_snapshot"] = {
        "artifact": snapshot_path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
    }
    proof_path = root / "proofs" / "result.json"
    record["harbor_key"] = identifier + "-fresh-cpu-verifier"
    workflow.report.write_object(
        proof_path,
        {
            "key": record["harbor_key"],
            "rewards": {"reward": result["score"]},
            "tags": {
                "episode_id": identifier,
                "recovery_kind": record["recovery_kind"],
                "parent_checked": True,
                "fresh_verifier": True,
            },
        },
    )
    record["verifier_result"]["sha256"] = hashlib.sha256(proof_path.read_bytes()).hexdigest()
    answer_path = root / "proofs" / "answer.py"
    answer_path.write_text(source)
    record["answer_blob"]["sha256"] = hashlib.sha256(answer_path.read_bytes()).hexdigest()
    if outcome == "truncated":
        record["answer_blob"] = None
    recovery_path.unlink()
    recovery_path = root / "episode-recoveries" / f"{identifier}.json"
    workflow.report.write_object(recovery_path, record)
    state["started_episodes"] = [identifier]
    return campaign, task, release, state, recovery_path, record, original_path


@pytest.mark.parametrize("phase", ["baseline", "baseline-independent", "frozen-repeat", "independent"])
@pytest.mark.parametrize("outcome", ["completed", "truncated"])
def test_saved_episode_preserves_original_and_never_calls_backend(
    workflow: ModuleType, tmp_path: Path, phase: str, outcome: str
) -> None:
    campaign, task, release, state, recovery_path, record, original_path = prepare_saved_episode(
        workflow, tmp_path, outcome, phase
    )
    root = campaign.arguments.run_root
    snapshot_path = root / record["original_snapshot"]["artifact"]
    saved_bytes = (original_path.read_bytes(), snapshot_path.read_bytes(), recovery_path.read_bytes())
    canonical = root / "episodes" / f"{record['recovered_episode']['episode_id']}.json"
    canonical.parent.mkdir()
    canonical.write_bytes(original_path.read_bytes())
    result = asyncio.run(campaign.episode(task, phase, 0, release, state))
    assert result == record["recovered_episode"]
    assert asyncio.run(campaign.episode(task, phase, 0, release, state)) == result
    assert workflow.metrics.summarize([result])["accuracy"] == (1.0 if outcome == "completed" else 0.0)
    assert workflow.metrics.summarize([result])["valid"] == 1
    assert result["fault"] == (
        None if outcome == "completed" else "maximum student turns reached without a final submission"
    )
    assert (original_path.read_bytes(), snapshot_path.read_bytes(), recovery_path.read_bytes()) == saved_bytes
    assert canonical.read_bytes() == saved_bytes[0]
    assert not campaign.api.reports


@pytest.mark.parametrize("target", ["snapshot", "result"])
@pytest.mark.parametrize(
    "key",
    [
        "episode_id",
        "phase",
        "release_id",
        "runtime_load_id",
        "references",
        "turns",
        "messages",
        "trajectory",
        "prompt_tokens",
        "completion_tokens",
        "elapsed_seconds",
        "metadata",
    ],
)
def test_saved_episode_rejects_immutable_mutation(workflow: ModuleType, tmp_path: Path, target: str, key: str) -> None:
    campaign, task, release, state, recovery_path, record, _ = prepare_saved_episode(workflow, tmp_path)
    if target == "snapshot":
        path = campaign.arguments.run_root / record["original_snapshot"]["artifact"]
        changed = workflow.report.read_object(path)
    else:
        changed = record["recovered_episode"]
    changed[key] = "changed"
    if target == "snapshot":
        workflow.report.write_object(path, changed)
        record["original_snapshot"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    workflow.report.write_object(recovery_path, record)
    with pytest.raises(RuntimeError):
        asyncio.run(campaign.episode(task, "baseline", 0, release, state))


@pytest.mark.parametrize(
    "reason",
    [
        "student response or episode token window was exhausted",
        "student code tool timed out",
        "maximum student turns reached without a final submission",
    ],
)
def test_saved_truncated_episode_preserves_model_reason(workflow: ModuleType, tmp_path: Path, reason: str) -> None:
    campaign, task, release, state, recovery_path, record, _ = prepare_saved_episode(workflow, tmp_path, "truncated")
    snapshot_path = campaign.arguments.run_root / record["original_snapshot"]["artifact"]
    snapshot = workflow.report.read_object(snapshot_path)
    snapshot["fault"] = reason
    workflow.report.write_object(snapshot_path, snapshot)
    record["original_snapshot"]["sha256"] = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    record["recovered_episode"]["fault"] = reason
    workflow.report.write_object(recovery_path, record)
    assert asyncio.run(campaign.episode(task, "baseline", 0, release, state))["fault"] == reason
    record["recovered_episode"]["fault"] = (
        "maximum student turns reached without a final submission"
        if reason != ("maximum student turns reached without a final submission")
        else "student code tool timed out"
    )
    workflow.report.write_object(recovery_path, record)
    with pytest.raises(RuntimeError, match="original completion or bounded truncation"):
        asyncio.run(campaign.episode(task, "baseline", 0, release, state))


@pytest.mark.parametrize("outcome", ["truncated", "completed"])
def test_saved_truncated_episode_cannot_become_success(workflow: ModuleType, tmp_path: Path, outcome: str) -> None:
    campaign, task, release, state, recovery_path, record, _ = prepare_saved_episode(workflow, tmp_path, "truncated")
    record["recovered_episode"].update(outcome=outcome, score=1.0)
    proof_path = campaign.arguments.run_root / record["verifier_result"]["artifact"]
    proof = workflow.report.read_object(proof_path)
    proof["rewards"]["reward"] = 1.0
    workflow.report.write_object(proof_path, proof)
    record["verifier_result"]["sha256"] = hashlib.sha256(proof_path.read_bytes()).hexdigest()
    workflow.report.write_object(recovery_path, record)
    with pytest.raises(RuntimeError):
        asyncio.run(campaign.episode(task, "baseline", 0, release, state))


@pytest.mark.parametrize("key", ["reviewed", "original_snapshot", "original_sha256", "verifier_result", "answer_blob"])
def test_saved_completed_episode_requires_review_hashes_and_proof(
    workflow: ModuleType, tmp_path: Path, key: str
) -> None:
    campaign, task, release, state, recovery_path, record, _ = prepare_saved_episode(workflow, tmp_path)
    record.pop(key)
    workflow.report.write_object(recovery_path, record)
    with pytest.raises((RuntimeError, KeyError)):
        asyncio.run(campaign.episode(task, "baseline", 0, release, state))


@pytest.mark.parametrize("key", ["original_snapshot", "verifier_result", "answer_blob"])
def test_saved_episode_rejects_changed_artifact_hash(workflow: ModuleType, tmp_path: Path, key: str) -> None:
    campaign, task, release, state, recovery_path, record, _ = prepare_saved_episode(workflow, tmp_path)
    record[key]["sha256"] = "0" * 64
    workflow.report.write_object(recovery_path, record)
    with pytest.raises(RuntimeError, match="checksum"):
        asyncio.run(campaign.episode(task, "baseline", 0, release, state))


@pytest.mark.parametrize("mutation", ["score", "error", "episode", "kind", "review", "fresh", "key", "same-key"])
def test_saved_episode_validates_proof_contents(workflow: ModuleType, tmp_path: Path, mutation: str) -> None:
    campaign, task, release, state, recovery_path, record, _ = prepare_saved_episode(workflow, tmp_path)
    proof_path = campaign.arguments.run_root / record["verifier_result"]["artifact"]
    proof = workflow.report.read_object(proof_path)
    if mutation == "score":
        proof["rewards"]["reward"] = 0.0
    elif mutation == "key":
        proof["key"] = "another-physical-key"
    elif mutation == "same-key":
        proof["key"] = record["recovered_episode"]["episode_id"]
        record["harbor_key"] = proof["key"]
    else:
        tag = {
            "error": "error",
            "episode": "episode_id",
            "kind": "recovery_kind",
            "review": "parent_checked",
            "fresh": "fresh_verifier",
        }[mutation]
        proof["tags"][tag] = "changed"
    workflow.report.write_object(proof_path, proof)
    record["verifier_result"]["sha256"] = hashlib.sha256(proof_path.read_bytes()).hexdigest()
    workflow.report.write_object(recovery_path, record)
    with pytest.raises(RuntimeError, match="verifier proof"):
        asyncio.run(campaign.episode(task, "baseline", 0, release, state))


@pytest.mark.parametrize(
    "source", ["def solve():\n    return 2\n", "x = 1\n", "import os\ndef solve():\n    return 1\n"]
)
def test_saved_completed_episode_rejects_fabricated_answer(workflow: ModuleType, tmp_path: Path, source: str) -> None:
    campaign, task, release, state, recovery_path, record, _ = prepare_saved_episode(workflow, tmp_path)
    answer_path = campaign.arguments.run_root / record["answer_blob"]["artifact"]
    answer_path.write_text(source)
    record["answer_blob"]["sha256"] = hashlib.sha256(answer_path.read_bytes()).hexdigest()
    workflow.report.write_object(recovery_path, record)
    with pytest.raises(RuntimeError, match="original model output"):
        asyncio.run(campaign.episode(task, "baseline", 0, release, state))


@pytest.mark.parametrize("answer_present", [False, True])
def test_saved_truncated_episode_accepts_only_missing_answer_descriptor(
    workflow: ModuleType, tmp_path: Path, answer_present: bool
) -> None:
    campaign, task, release, state, recovery_path, record, _ = prepare_saved_episode(workflow, tmp_path, "truncated")
    missing_path = campaign.arguments.run_root / "proofs" / "missing-answer.json"
    workflow.report.write_object(
        missing_path, {"answer_present": answer_present, "reason": "original_episode_did_not_submit"}
    )
    record["answer_blob"] = {
        "artifact": "proofs/missing-answer.json",
        "sha256": hashlib.sha256(missing_path.read_bytes()).hexdigest(),
    }
    workflow.report.write_object(recovery_path, record)
    if answer_present:
        with pytest.raises(RuntimeError, match="fabricate"):
            asyncio.run(campaign.episode(task, "baseline", 0, release, state))
    else:
        assert asyncio.run(campaign.episode(task, "baseline", 0, release, state))["score"] == 0


def test_saved_episode_rejects_training(workflow: ModuleType, tmp_path: Path) -> None:
    campaign, task, release, state, _, _, _ = prepare_saved_episode(workflow, tmp_path, phase="train")
    with pytest.raises(RuntimeError, match="evaluation-only"):
        asyncio.run(campaign.episode(task, "train", 0, release, state))


def test_saved_final_uses_original_single_python_block_parser(workflow: ModuleType, tmp_path: Path) -> None:
    campaign, task, release, state, recovery_path, record, original_path = prepare_saved_episode(workflow, tmp_path)
    snapshot_path = campaign.arguments.run_root / record["original_snapshot"]["artifact"]
    content = record["recovered_episode"]["messages"][-1]["content"] + "\n```"
    for path in (original_path, snapshot_path):
        episode = workflow.report.read_object(path)
        episode["messages"][-1]["content"] = content
        workflow.report.write_object(path, episode)
    record["recovered_episode"]["messages"][-1]["content"] = content
    record["original_sha256"] = hashlib.sha256(original_path.read_bytes()).hexdigest()
    record["original_snapshot"]["sha256"] = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    workflow.report.write_object(recovery_path, record)
    result = asyncio.run(campaign.episode(task, "baseline", 0, release, state))
    assert result["messages"][-1]["content"] == content


@pytest.mark.parametrize(
    "content",
    [
        "```python\ndef solve():\n    return 1\n```",
        "FINAL\n```python\nx = 1\n```",
        "FINAL\n```python\nimport os\ndef solve():\n    return 1\n```",
        "FINAL\n```python\ndef solve():\n    return 1\n```\n```python\nx = 1\n```",
    ],
)
def test_saved_completed_episode_validates_original_final(workflow: ModuleType, tmp_path: Path, content: str) -> None:
    campaign, task, release, state, recovery_path, record, original_path = prepare_saved_episode(workflow, tmp_path)
    snapshot_path = campaign.arguments.run_root / record["original_snapshot"]["artifact"]
    for path in (original_path, snapshot_path):
        episode = workflow.report.read_object(path)
        episode["messages"] = [{"role": "assistant", "content": content}]
        workflow.report.write_object(path, episode)
    record["recovered_episode"]["messages"] = [{"role": "assistant", "content": content}]
    record["original_sha256"] = hashlib.sha256(original_path.read_bytes()).hexdigest()
    record["original_snapshot"]["sha256"] = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    workflow.report.write_object(recovery_path, record)
    with pytest.raises(RuntimeError, match="FINAL"):
        asyncio.run(campaign.episode(task, "baseline", 0, release, state))
