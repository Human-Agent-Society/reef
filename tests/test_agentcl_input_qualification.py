"""CPU-only native-input qualification regressions with synthetic captures and campaign commits."""

from __future__ import annotations

import argparse
import asyncio
import copy
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest
from tests import test_sdft_agentcl as sdft_workflow
from tests import test_sdpo_agentcl as sdpo_workflow

from recipes.sdft.examples.agentcl import qualification as sdft_qualification
from recipes.sdft.examples.agentcl import report as sample_report
from recipes.sdft.examples.agentcl import run as sdft_run
from recipes.sdpo.examples.agentcl import qualification as sdpo_qualification
from recipes.sdpo.examples.agentcl import run as sdpo_run


@pytest.fixture(params=[sdft_qualification, sdpo_qualification], ids=["sdft", "sdpo"])
def qualification(request: pytest.FixtureRequest) -> ModuleType:
    return request.param


@pytest.fixture(params=[sdft_workflow, sdpo_workflow], ids=["sdft", "sdpo"])
def workflow(request: pytest.FixtureRequest) -> ModuleType:
    return request.param


def native_sample(
    turn_count: int,
    report_id: str = "report",
    release_id: str = "release",
    runtime_load_id: str = "runtime",
) -> tuple[sample_report.JsonObject, sample_report.JsonObject]:
    tokens = [10, 11]
    mask: list[int] = []
    log_probs: list[float] = []
    turns: list[sample_report.JsonObject] = []
    canonical_turns: list[sample_report.JsonObject] = []
    references: list[str] = []
    for index in range(turn_count):
        if index > 0:
            tokens.append(30 + index)
            mask.append(0)
            log_probs.append(0.0)
        tokens.extend([20 + index * 2, 21 + index * 2])
        mask.extend([1, 1])
        log_probs.extend([-0.2, -0.3])
        receipt = f"{report_id}-turn-{index}"
        references.append(receipt)
        training: sample_report.JsonObject = {
            "tokens": list(tokens),
            "loss_mask": [1, 1],
            "rollout_log_probs": [-0.2, -0.3],
            "runtime_load_id": runtime_load_id,
        }
        turns.append({"receipt": receipt, **training})
        canonical_turns.append(
            {
                "receipt": receipt,
                "release_id": release_id,
                "record": {
                    "payload": {"response": {"training": copy.deepcopy(training)}},
                    "artifact_ref": {"release_id": release_id},
                },
            }
        )
    capture: sample_report.JsonObject = {
        "report_id": report_id,
        "references": references,
        "turns": turns,
        "tokens": tokens,
        "loss_mask": mask,
        "rollout_log_probs": log_probs,
        "teacher_tokens": [100, 101, *tokens[2:]],
        "runtime_load_id": runtime_load_id,
        "distill_sample_weight": 1.0,
        "capture_stage": "native_processor_output_before_optimizer",
    }
    episode: sample_report.JsonObject = {
        "report_id": report_id,
        "references": list(references),
        "turns": canonical_turns,
        "release_id": release_id,
        "phase": "train",
        "outcome": "completed",
        "score": 1.0,
    }
    return capture, episode


def save_sample(run_root: Path, capture: sample_report.JsonObject, episode: sample_report.JsonObject) -> None:
    sample_report.write_object(run_root / "episodes" / f"{episode['report_id']}.json", episode)
    sample_report.write_object(run_root / "teacher-records" / f"{episode['report_id']}.json", capture)


@pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf"), True])
def test_invalid_sample_weight_cannot_hide_behind_an_active_episode(qualification, tmp_path, weight):
    capture, episode = native_sample(2, report_id="invalid")
    capture["distill_sample_weight"] = weight
    with pytest.raises(ValueError, match="finite and non-negative"):
        qualification.check_native_sample(capture, episode)
    if weight == -1.0:
        active_capture, active_episode = native_sample(2, report_id="active")
        save_sample(tmp_path, capture, episode)
        save_sample(tmp_path, active_capture, active_episode)
        with pytest.raises(ValueError, match="finite and non-negative"):
            qualification.qualify_inputs(tmp_path)


def test_valid_one_turn_retains_every_assistant_token(qualification: ModuleType) -> None:
    capture, episode = native_sample(1)
    checked = qualification.check_native_sample(capture, episode)
    assert checked["turn_count"] == 1
    assert checked["assistant_token_count"] == 2
    assert checked["masked_context_token_count"] == 0
    assert checked["active"] is True
    assert checked["all_assistant_tokens_selected"] is True
    assert checked["teacher_suffix_identity"] is True


def test_explicit_multi_turn_diagnostic_remains_available(qualification: ModuleType) -> None:
    capture, episode = native_sample(1)
    with pytest.raises(ValueError, match="multi-turn"):
        qualification.check_native_sample(capture, episode, require_multi_turn=True)
    capture, episode = native_sample(2)
    checked = qualification.check_native_sample(capture, episode, require_multi_turn=True)
    assert checked["turn_count"] == 2
    assert checked["assistant_token_count"] == 4
    assert checked["masked_context_token_count"] == 1
    assert checked["tool_context_zero_loss"] is True


@pytest.mark.parametrize("source", ["capture", "episode"])
@pytest.mark.parametrize("change", ["missing", "empty"])
def test_missing_or_empty_turns_fail(qualification: ModuleType, source: str, change: str) -> None:
    capture, episode = native_sample(1)
    changed = capture if source == "capture" else episode
    if change == "missing":
        changed.pop("turns")
    else:
        changed["turns"] = []
    with pytest.raises((ValueError, KeyError)):
        qualification.check_native_sample(capture, episode, require_multi_turn=False)


@pytest.mark.parametrize("turn_count", [1, 2])
@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("report", "terminal report"),
        ("receipt", "ordered unique receipt"),
        ("assistant-mask", "changed tokens or trained tool/context"),
        ("turn-mask", "every sampled assistant token"),
        ("teacher-suffix", "full suffixes differ"),
        ("nonfinite-probability", "non-finite rollout"),
        ("selected-probability", "exact log probabilities"),
        ("canonical-record", "canonical authenticated inference record"),
        ("release", "canonical turn release"),
        ("runtime", "mixed runtime load IDs"),
        ("weight", "weight must be finite"),
    ],
)
def test_tensor_receipt_and_release_checks_remain_strict(
    qualification: ModuleType, turn_count: int, change: str, message: str
) -> None:
    capture, episode = native_sample(turn_count)
    turns = cast(list[sample_report.JsonObject], capture["turns"])
    mask = cast(list[int], capture["loss_mask"])
    if change == "report":
        capture["report_id"] = "different-report"
    elif change == "receipt":
        turns[0]["receipt"] = "different-receipt"
    elif change == "assistant-mask":
        mask[0] = 0
    elif change == "turn-mask":
        turns[0]["loss_mask"] = [1, 0]
    elif change == "teacher-suffix":
        cast(list[int], capture["teacher_tokens"])[-1] = 999
    elif change == "nonfinite-probability":
        cast(list[float], capture["rollout_log_probs"])[0] = float("nan")
    elif change == "selected-probability":
        cast(list[float], capture["rollout_log_probs"])[0] = -0.4
    elif change == "canonical-record":
        canonical_turn = cast(list[sample_report.JsonObject], episode["turns"])[0]
        record = cast(sample_report.JsonObject, canonical_turn["record"])
        payload = cast(sample_report.JsonObject, record["payload"])
        response = cast(sample_report.JsonObject, payload["response"])
        cast(sample_report.JsonObject, response["training"])["tokens"] = [10, 11, 20, 999]
    elif change == "release":
        episode["release_id"] = "different-release"
    elif change == "runtime":
        capture["runtime_load_id"] = "different-runtime"
    else:
        capture["distill_sample_weight"] = float("inf")
    with pytest.raises(ValueError, match=message):
        qualification.check_native_sample(capture, episode)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("history", "histories drifted"),
        ("context-mask", "trained tool/context"),
        ("mixed-runtime", "mixed runtime load IDs"),
        ("receipt-order", "ordered unique receipt"),
    ],
)
def test_multi_turn_history_and_context_are_strict(qualification: ModuleType, change: str, message: str) -> None:
    capture, episode = native_sample(2)
    turns = cast(list[sample_report.JsonObject], capture["turns"])
    if change == "history":
        cast(list[int], turns[1]["tokens"])[0] = 999
    elif change == "context-mask":
        cast(list[int], capture["loss_mask"])[2] = 1
    elif change == "mixed-runtime":
        turns[1]["runtime_load_id"] = "different-runtime"
    else:
        turns.reverse()
    with pytest.raises(ValueError, match=message):
        qualification.check_native_sample(capture, episode)


def test_final_qualification_accepts_mixed_one_and_two_turn_episodes(
    qualification: ModuleType, tmp_path: Path
) -> None:
    for turn_count in (1, 2):
        capture, episode = native_sample(turn_count, f"report-{turn_count}")
        save_sample(tmp_path, capture, episode)
    result = qualification.qualify_inputs(tmp_path)
    checks = cast(list[sample_report.JsonObject], result["episodes"])
    assert result["native_input_checks_passed"] is True
    assert [checked["turn_count"] for checked in checks] == [1, 2]
    assert all(checked["active"] is True for checked in checks)
    assert result["optimizer_execution_verified"] is False


@pytest.mark.parametrize("extra_episode", ["one-turn", "inactive-multi-turn", "evaluation-multi-turn"])
def test_final_qualification_requires_active_training_multi_turn_coverage(
    qualification: ModuleType, tmp_path: Path, extra_episode: str
) -> None:
    capture, episode = native_sample(1, "active-one-turn")
    save_sample(tmp_path, capture, episode)
    if extra_episode == "one-turn":
        capture, episode = native_sample(1, "another-one-turn")
    else:
        capture, episode = native_sample(2, "multi-turn")
        if extra_episode == "inactive-multi-turn":
            capture["distill_sample_weight"] = 0.0
        else:
            episode["phase"] = "baseline"
    save_sample(tmp_path, capture, episode)
    with pytest.raises(ValueError, match="multi-turn"):
        qualification.qualify_inputs(tmp_path)


@pytest.mark.parametrize("weight", [0.0, 1.0])
def test_final_qualification_checks_every_episode_even_with_valid_multi_turn_coverage(
    qualification: ModuleType, tmp_path: Path, weight: float
) -> None:
    capture, episode = native_sample(2, "valid-multi-turn")
    save_sample(tmp_path, capture, episode)
    capture, episode = native_sample(1, "corrupt-one-turn")
    capture["distill_sample_weight"] = weight
    cast(list[int], capture["loss_mask"])[0] = 0
    save_sample(tmp_path, capture, episode)
    with pytest.raises(ValueError, match="trained tool/context"):
        qualification.qualify_inputs(tmp_path)


def test_final_qualification_still_requires_effective_signal(qualification: ModuleType, tmp_path: Path) -> None:
    capture, episode = native_sample(2)
    capture["distill_sample_weight"] = 0.0
    save_sample(tmp_path, capture, episode)
    with pytest.raises(ValueError, match="no effective distillation signal"):
        qualification.qualify_inputs(tmp_path)


class CapturedEpisodes(sdft_run.EpisodeBackend, sdpo_run.EpisodeBackend):
    def __init__(self, arguments: argparse.Namespace, first_grid_change: str = "valid"):
        self.arguments = arguments
        self.first_grid_change = first_grid_change
        self.calls: list[tuple[str, str]] = []
        self.results: dict[str, sample_report.JsonObject] = {}

    async def run(
        self,
        task: sample_report.JsonObject,
        episode_id: str,
        phase: str,
        release: sample_report.JsonObject,
    ) -> sample_report.JsonObject:
        attempt = len(self.calls) % self.arguments.attempts
        turn_count = 1 if task["campaign_position"] == 0 else 2
        if task["campaign_position"] > 0:
            first_checks = sample_report.read_object(self.arguments.run_root / "native-first-task-checks.json")
            assert first_checks["passed"] is True
            assert len(cast(list[str], first_checks["checked_reports"])) == self.arguments.attempts
        report_id = sample_report.stable_id(
            self.arguments.run_id, phase, str(task["category"]), str(task["id"]), attempt, "report"
        )
        capture, episode = native_sample(
            turn_count, report_id, str(release["release_id"]), str(release["runtime_load_id"])
        )
        if task["campaign_position"] == 0:
            if self.first_grid_change == "inactive":
                capture["distill_sample_weight"] = 0.0
            elif self.first_grid_change == "corrupt-last" and attempt == self.arguments.attempts - 1:
                cast(list[int], capture["loss_mask"])[0] = 0
        episode.update(
            episode_id=episode_id,
            messages=[],
            trajectory={},
            prompt_tokens=2,
            completion_tokens=turn_count * 2,
            elapsed_seconds=0.1,
        )
        sample_report.write_object(self.arguments.run_root / "teacher-records" / f"{report_id}.json", capture)
        self.calls.append((str(task["category"]), str(release["release_id"])))
        self.results[episode_id] = copy.deepcopy(episode)
        return episode

    def recover(self, episode_id: str) -> sample_report.JsonObject | None:
        return copy.deepcopy(self.results.get(episode_id))


def test_first_task_one_turn_grid_advances_to_paired_multi_turn_task(workflow: ModuleType, tmp_path: Path) -> None:
    arguments = workflow.arguments_fixture(tmp_path)
    arguments.native_input_checks = True
    manifest = workflow.manifest_fixture(arguments.data_root)
    api = workflow.FakeApi(attempts=arguments.attempts)
    backend = CapturedEpisodes(arguments)
    campaign = workflow.run.Campaign(arguments, api, backend, manifest)
    asyncio.run(campaign.execute_phase("train"))
    assert backend.calls == [("raw", "base")] * arguments.attempts + [("new", "release-1")] * arguments.attempts
    assert campaign.cursor["training_step"] == 2
    assert len(api.history) == 2
    assert len(api.reports) == 2 * arguments.attempts
    result = workflow.run.qualify_inputs(arguments.run_root)
    checks = cast(list[sample_report.JsonObject], result["episodes"])
    assert sorted(checked["turn_count"] for checked in checks) == [1] * arguments.attempts + [2] * arguments.attempts
    assert result["native_input_checks_passed"] is True
    assert result["optimizer_execution_verified"] is False
    resumed = workflow.run.Campaign(arguments, api, backend, manifest)
    asyncio.run(resumed.execute_phase("train"))
    assert len(backend.calls) == 2 * arguments.attempts


@pytest.mark.parametrize("change", ["inactive", "corrupt-last"])
def test_first_task_gate_still_checks_active_signal_and_every_sample(
    workflow: ModuleType, tmp_path: Path, change: str
) -> None:
    arguments = workflow.arguments_fixture(tmp_path)
    arguments.native_input_checks = True
    manifest = workflow.manifest_fixture(arguments.data_root)
    api = workflow.FakeApi(attempts=arguments.attempts)
    backend = CapturedEpisodes(arguments, change)
    campaign = workflow.run.Campaign(arguments, api, backend, manifest)
    if change == "inactive":
        with pytest.raises(RuntimeError, match="no effective distillation signal"):
            asyncio.run(campaign.execute_phase("train"))
    else:
        with pytest.raises(ValueError, match="trained tool/context"):
            asyncio.run(campaign.execute_phase("train"))
    assert backend.calls == [("raw", "base")] * arguments.attempts
    assert campaign.cursor["training_step"] == 1
    assert len(api.history) == 1
    assert not (arguments.run_root / "native-first-task-checks.json").exists()
