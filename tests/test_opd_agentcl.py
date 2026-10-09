"""Direct CPU checks for the checkpoint-backed OPD AgentCL campaign."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import cast

import pytest

from recipes.opd.examples.agentcl import native, qualification, report, run, taskexport
from reef.core import AgentRecord, RequestType
from reef.core.artifact_ref import LiveWeightArtifactRef
from reef.core.reports import TeacherContextReport
from reef.train.types import ProcessorContext


def manifest_fixture(root: Path) -> report.JsonObject:
    training: list[report.JsonValue] = []
    independent: list[report.JsonValue] = []
    reference_bytes = b"Reference qualification only; never a teacher prompt"
    test_code = "assert candidate() == 1\n"
    for role, count, rows in (("training", 96, training), ("independent", 120, independent)):
        for position in range(count):
            if role == "training":
                category = "raw" if position < 48 else "new"
                identifier = str(position if position < 48 else (position - 47) % 48)
            else:
                category = "new"
                identifier = f"independent-{position}"
            reference = f"privileged/{role}/{position:03d}.py"
            task_relative = f"tasks/{role}/{position:03d}"
            task_path = root / task_relative
            files = {
                "instruction.md": b"Implement candidate().",
                "environment/Dockerfile": b"FROM python:3.12-slim\n",
                "tests/task.json": json.dumps({"test_code": test_code}).encode(),
            }
            for relative, content in files.items():
                path = task_path / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            rows.append(
                {
                    "role": role,
                    "category": category,
                    "id": identifier,
                    "pair_id": identifier,
                    "position": position,
                    "task_path": task_relative,
                    "reference_path": reference,
                    "reference_sha256": hashlib.sha256(reference_bytes).hexdigest(),
                    "test_sha256": hashlib.sha256(test_code.encode()).hexdigest(),
                    "files": {relative: hashlib.sha256(content).hexdigest() for relative, content in files.items()},
                }
            )
            path = root / reference
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(reference_bytes)
    manifest: report.JsonObject = {
        "schema_version": 1,
        "dataset": "osunlp/AgentCL",
        "revision": run.DATASET_REVISION,
        "source_files": dict(taskexport.SOURCE_FILES),
        "overlay_version": "fixture",
        "answer_contract": "fixture",
        "corrections_sha256": hashlib.sha256(b"fixture").hexdigest(),
        "training": training,
        "independent": independent,
    }
    report.write_object(root / "manifest.json", manifest)
    manifest_checksum = hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
    results = []
    for value in training + independent:
        row = cast(report.JsonObject, value)
        results.append(
            {
                **row,
                "task_files": dict(cast(report.JsonObject, row["files"])),
                "reference_key": taskexport.reference_key(manifest_checksum, row),
                "reward": 1,
                "error": None,
            }
        )
    report.write_object(
        root / "reference-verification.json",
        {
            "schema_version": 2,
            "revision": run.DATASET_REVISION,
            "manifest_sha256": manifest_checksum,
            "status": "passed",
            "total": 216,
            "passed": 216,
            "results": results,
        },
    )
    return manifest


def arguments_fixture(root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        run_root=root / "run",
        data_root=root / "data",
        cache_dir=root / "cache",
        run_id="opd-fixture",
        wandb_project=None,
        wandb_entity=None,
        profile="smoke",
        model_path="student-checkpoint",
        teacher_checkpoint="frozen-compatible-teacher",
        seed=42,
        scenario="opd-fixture",
        service_url="http://fixture",
        steps=2,
        attempts=1,
        max_turns=8,
        native_input_checks=False,
        commit_timeout_seconds=1,
        poll_seconds=0.001,
    )


class FakeApi(run.ReefApi):
    def __init__(self) -> None:
        self.current: report.JsonObject = {
            "release_id": "base",
            "checkpoint": True,
            "runtime_load_id": "runtime-base",
            "operation": "creation",
            "pending": False,
        }
        self.history: list[report.JsonObject] = []
        self.reports: dict[str, report.JsonObject] = {}
        self.events: list[str] = []
        self.raise_after_accept = False
        self.publish = True

    def current_release(self) -> report.JsonObject:
        return copy.deepcopy(self.current)

    def commits(self, record_ids: list[str]) -> list[report.JsonObject]:
        return [
            copy.deepcopy(commit)
            for commit in self.history
            if set(record_ids).intersection(cast(list[str], commit["consumed_ids"]))
        ]

    def record(self, record_id: str) -> report.JsonObject | None:
        if record_id not in self.reports:
            return None
        return {"payload": self.reports[record_id]}

    def report(self, payload: report.JsonObject) -> None:
        identifier = str(payload["agent_record_id"])
        self.events.append("report:" + identifier)
        self.reports[identifier] = {key: value for key, value in payload.items() if key != "agent_record_id"}
        if self.publish:
            step = len(self.history) + 1
            parent = self.current["release_id"] if self.current["checkpoint"] else self.current["parent_release_id"]
            artifact: report.JsonObject = {
                "kind": "live_weights",
                "release_id": f"release-{step}",
                "parent_release_id": parent,
            }
            self.history.append(
                {
                    "step": step,
                    "operation": "training",
                    "operation_verified": True,
                    "pending": False,
                    "artifact_ref": artifact,
                    "consumed_ids": [identifier, *cast(list[str], payload["references"])],
                }
            )
            self.current = {
                "release_id": artifact["release_id"],
                "parent_release_id": parent,
                "checkpoint": False,
                "runtime_load_id": f"runtime-{step}",
                "operation": "training",
                "pending": False,
            }
            self.events.append(f"commit:{step}")
        if self.raise_after_accept:
            self.raise_after_accept = False
            raise TimeoutError("accepted report, response lost")


class FakeEpisodes(run.EpisodeBackend):
    def __init__(self, api: FakeApi) -> None:
        self.api = api
        self.calls: list[tuple[str, str, str]] = []
        self.seeds: list[int] = []
        self.results: dict[str, report.JsonObject] = {}
        self.fail = False
        self.change_release = False
        self.outcome = "completed"

    async def run(
        self, task: report.JsonObject, episode_id: str, phase: str, release: report.JsonObject
    ) -> report.JsonObject:
        self.calls.append((str(task["id"]), phase, str(release["release_id"])))
        self.seeds.append(int(cast(int, task["sampling_seed"])))
        self.api.events.append(f"episode:{task['category']}:{task['id']}")
        if self.fail:
            raise TimeoutError("unknown remote inference outcome")
        references = [f"receipt-{episode_id}-{index}" for index in range(2)]
        result: report.JsonObject = {
            "episode_id": episode_id,
            "release_id": release["release_id"],
            "runtime_load_id": release["runtime_load_id"],
            "outcome": self.outcome,
            "score": 0.0,
            "references": references,
            "turns": [
                {
                    "receipt": receipt,
                    "release_id": release["release_id"],
                    "runtime_load_id": release["runtime_load_id"],
                }
                for receipt in references
            ],
            "messages": [{"role": "user", "content": "task"}, {"role": "assistant", "content": "FINAL"}],
            "trajectory": {},
            "prompt_tokens": 4,
            "completion_tokens": 2,
            "elapsed_seconds": 0.1,
        }
        self.results[episode_id] = copy.deepcopy(result)
        if self.change_release:
            self.api.current["release_id"] = "unexpected-release"
        return result

    def recover(self, episode_id: str) -> report.JsonObject | None:
        return copy.deepcopy(self.results.get(episode_id))


def parse_fixture(monkeypatch: pytest.MonkeyPatch, *arguments: str) -> argparse.Namespace:
    monkeypatch.setattr(sys, "argv", ["run.py", *arguments])
    return run.parse_arguments()


@pytest.mark.parametrize("command", ["train", "baseline", "evaluate"])
@pytest.mark.parametrize("checkpoint", [None, "", "   "])
def test_campaign_parse_requires_nonempty_teacher(monkeypatch, command, checkpoint):
    monkeypatch.delenv("AGENTCL_TEACHER_CHECKPOINT", raising=False)
    arguments = [command]
    if checkpoint is not None:
        arguments.extend(["--teacher-checkpoint", checkpoint])
    with pytest.raises(SystemExit, match="2"):
        parse_fixture(monkeypatch, *arguments)


def test_teacher_environment_and_cli_override(monkeypatch):
    monkeypatch.setenv("AGENTCL_TEACHER_CHECKPOINT", "/env-teacher")
    arguments = parse_fixture(monkeypatch, "train", "--profile", "smoke")
    assert arguments.teacher_checkpoint == "/env-teacher"
    assert arguments.steps == 2 and arguments.attempts == 1
    arguments = parse_fixture(monkeypatch, "train", "--teacher-checkpoint", "/cli-teacher")
    assert arguments.teacher_checkpoint == "/cli-teacher"
    assert arguments.steps == 96 and arguments.attempts == 1


def test_export_needs_no_teacher_but_service_dry_run_does(monkeypatch):
    monkeypatch.delenv("AGENTCL_TEACHER_CHECKPOINT", raising=False)
    assert parse_fixture(monkeypatch, "export", "--dry-run").teacher_checkpoint == ""
    with pytest.raises(SystemExit, match="2"):
        parse_fixture(monkeypatch, "verify", "--dry-run")


def test_turn_budget_is_fixed(monkeypatch):
    with pytest.raises(SystemExit, match="2"):
        parse_fixture(monkeypatch, "train", "--teacher-checkpoint", "/teacher", "--max-turns", "4")


def test_actual_dry_run_parses_config_without_submission(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "train",
            "--dry-run",
            "--profile",
            "smoke",
            "--data-root",
            str(tmp_path / "data"),
            "--run-root",
            str(tmp_path / "run"),
            "--teacher-checkpoint",
            "/frozen teacher",
            "--model-path",
            "/student checkpoint",
        ],
    )
    asyncio.run(run.main())
    output = capsys.readouterr().out
    specification = json.loads(output.split("Service proposal", 1)[0])
    assert specification["submits"] == 0
    assert specification["base_commit"] == "daeaf4d42fb2f9e20445df6618afdc2a4097bdeb"
    assert specification["opd_implementation_revision"] == "a06415d7c417dc0f180b3301e64d256aa91dcb40"
    assert specification["opd_implementation_branch"] == "origin/codex/opd-slime-reproduction"
    assert specification["teacher_checkpoint"] == "/frozen teacher"
    assert specification["ema"] == 0 and specification["loss"] == "reverse KL"
    assert specification["top_k"] == 1 and specification["importance_sampling_cap"] == 0
    assert specification["student_window_tokens"] == 8192 and specification["teacher_window_tokens"] == 16384
    assert specification["response_tokens"] == 2048 and specification["max_turns"] == 8
    assert specification["temperature"] == 0.7
    assert "AGENTCL_TEACHER_CHECKPOINT='/frozen teacher'" in output
    assert "AGENTCL_MODEL_PATH='/student checkpoint'" in output
    assert not (tmp_path / "run").exists() and not (tmp_path / "data").exists()


def test_native_service_has_frozen_teacher_settings(tmp_path, monkeypatch):
    from reef.service.deploy.config_utils import load_config
    from reef.service.deploy.service_config import service_config_from_mapping

    arguments = arguments_fixture(tmp_path)
    for key, value in {
        "REEF_PORT": "28902",
        "REEF_TOKEN": "",
        "AGENTCL_ROUTER_PORT": "23002",
        "AGENTCL_RUN_ROOT": str(tmp_path),
        "AGENTCL_MODEL_PATH": arguments.model_path,
        "AGENTCL_TEACHER_CHECKPOINT": arguments.teacher_checkpoint,
        "AGENTCL_BATCH_SIZE": "1",
        "AGENTCL_STEPS": "2",
        "AGENTCL_WARMUP": "1",
    }.items():
        monkeypatch.setenv(key, value)
    configuration = load_config(run.HERE / "serve.yaml")
    service = service_config_from_mapping(configuration)
    run.validate_service_proposal(arguments)
    assert configuration["recipe"]["implementation"] == "recipes.opd.examples.agentcl.native:CapturedOPDRecipe"
    assert "tokenizer-path" not in configuration["recipe"]["config"]
    assert configuration["recipe"]["config"]["accept-multi-turn-policy-samples"] is True
    options = service.training_backend_options
    for key, value in run.OPD_TRAINING_OPTIONS.items():
        assert options[key] == value
    assert options["opd-teacher-checkpoint"] == arguments.teacher_checkpoint
    assert service.colocate and service.inference_num_gpus == 4 and service.tensor_parallel_size == 1
    assert options["tensor-model-parallel-size"] == "4"
    assert not any(key.startswith(("sdft-", "sdpo-")) for key in options)


@pytest.mark.parametrize("change", ["teacher", "ema", "topology", "tokenizer"])
def test_service_proposal_rejects_incompatible_settings(tmp_path, monkeypatch, change):
    from reef.service.deploy import config_utils

    configuration = config_utils.load_config(run.HERE / "serve.yaml", interpolate_env=False)
    if change == "teacher":
        configuration["training"]["options"]["opd-teacher"] = "self"
    elif change == "ema":
        configuration["training"]["options"]["opd-teacher-update-rate"] = "0.01"
    elif change == "topology":
        configuration["training"]["colocate"] = False
    else:
        configuration["recipe"]["config"]["tokenizer-path"] = "/tokenizer"
    monkeypatch.setattr(config_utils, "load_config", lambda *args, **kwargs: configuration)
    with pytest.raises(ValueError):
        run.validate_service_proposal(arguments_fixture(tmp_path))


def test_complete_episode_report_commit_order_and_no_reference_reads(tmp_path):
    arguments = arguments_fixture(tmp_path)
    manifest = manifest_fixture(arguments.data_root)
    run.validate_reference_verification(arguments.data_root, manifest)
    shutil.rmtree(arguments.data_root / "privileged")
    manifest = run.load_manifest(arguments.data_root)
    api = FakeApi()
    backend = FakeEpisodes(api)
    campaign = run.Campaign(arguments, api, backend, manifest)
    for phase in ("baseline", "baseline-independent", "train", "frozen-repeat", "independent"):
        asyncio.run(campaign.execute_phase(phase))
    assert backend.calls[4:6] == [("0", "train", "base"), ("0", "train", "release-1")]
    assert run.selected_training_positions(arguments, manifest) == [0, 95]
    assert len(api.reports) == len(api.history) == 2
    training_events = api.events[4:]
    assert training_events[0] == "episode:raw:0"
    assert training_events[1].startswith("report:") and training_events[2] == "commit:1"
    assert training_events[3] == "episode:new:0"
    assert training_events[4].startswith("report:") and training_events[5] == "commit:2"
    assert all(payload["metadata"] == {"teacher_context": ""} for payload in api.reports.values())
    for commit in api.history:
        assert len(commit["consumed_ids"]) == 3
        assert commit["artifact_ref"]["parent_release_id"] == "base"
    assert all(call[2] == "base" for call in backend.calls[:4])
    assert all(call[2] == "release-2" for call in backend.calls[6:])
    assert backend.seeds[:2] == backend.seeds[4:6] == backend.seeds[6:8]
    before = copy.deepcopy(api.reports)
    resumed = run.Campaign(arguments, api, backend, manifest)
    for phase in ("train", "frozen-repeat", "independent"):
        asyncio.run(resumed.execute_phase(phase))
    assert len(backend.calls) == 10 and api.reports == before
    assert resumed.cursor["training_step"] == 2


@pytest.mark.parametrize("setting", ["teacher_checkpoint", "model_path", "seed", "native_options", "manifest"])
def test_immutable_specification_requires_fresh_root(tmp_path, monkeypatch, setting):
    arguments = arguments_fixture(tmp_path)
    manifest = manifest_fixture(arguments.data_root)
    api = FakeApi()
    backend = FakeEpisodes(api)
    run.Campaign(arguments, api, backend, manifest)
    if setting == "teacher_checkpoint":
        arguments.teacher_checkpoint = "different-frozen-teacher"
    elif setting == "model_path":
        arguments.model_path = "different-student"
    elif setting == "seed":
        arguments.seed = 43
    elif setting == "native_options":
        monkeypatch.setitem(run.OPD_TRAINING_OPTIONS, "opd-top-k", "2")
    else:
        manifest["training"][0]["reference_sha256"] = "changed"
    with pytest.raises(RuntimeError, match="fresh run root"):
        run.Campaign(arguments, api, backend, manifest)
    assert not backend.calls and not api.reports


@pytest.mark.parametrize("change", ["missing", "partial", "duplicate", "checksum"])
def test_reference_qualification_blocks_training_before_inference(tmp_path, change):
    arguments = arguments_fixture(tmp_path)
    manifest = manifest_fixture(arguments.data_root)
    path = arguments.data_root / "reference-verification.json"
    if change == "missing":
        path.unlink()
    else:
        checked = report.read_object(path)
        if change == "partial":
            checked["passed"] = 215
        elif change == "duplicate":
            checked["results"][-1] = checked["results"][0]
        else:
            checked["results"][0]["reference_sha256"] = "wrong"
        report.write_object(path, checked)
    api = FakeApi()
    backend = FakeEpisodes(api)
    campaign = run.Campaign(arguments, api, backend, manifest)
    with pytest.raises(RuntimeError, match="reference"):
        asyncio.run(campaign.execute_phase("train"))
    assert not backend.calls and not api.reports


def test_report_response_lost_resumes_consumption_without_replay(tmp_path):
    arguments = arguments_fixture(tmp_path)
    manifest = manifest_fixture(arguments.data_root)
    api = FakeApi()
    api.raise_after_accept = True
    backend = FakeEpisodes(api)
    campaign = run.Campaign(arguments, api, backend, manifest)
    with pytest.raises(TimeoutError):
        asyncio.run(campaign.execute_phase("train"))
    resumed = run.Campaign(arguments, api, backend, manifest)
    asyncio.run(resumed.execute_phase("train"))
    assert len(backend.calls) == 2 and len(api.reports) == 2
    assert resumed.cursor["training_step"] == 2


def test_unknown_inference_is_not_replayed(tmp_path):
    arguments = arguments_fixture(tmp_path)
    manifest = manifest_fixture(arguments.data_root)
    api = FakeApi()
    backend = FakeEpisodes(api)
    backend.fail = True
    campaign = run.Campaign(arguments, api, backend, manifest)
    with pytest.raises(TimeoutError):
        asyncio.run(campaign.execute_phase("train"))
    resumed = run.Campaign(arguments, api, backend, manifest)
    with pytest.raises(RuntimeError, match="never replay"):
        asyncio.run(resumed.execute_phase("train"))
    assert len(backend.calls) == 1 and not api.reports


def test_pending_commit_does_not_advance_or_resubmit(tmp_path):
    arguments = arguments_fixture(tmp_path)
    arguments.commit_timeout_seconds = 0.01
    manifest = manifest_fixture(arguments.data_root)
    api = FakeApi()
    api.publish = False
    backend = FakeEpisodes(api)
    campaign = run.Campaign(arguments, api, backend, manifest)
    with pytest.raises(TimeoutError, match="commit deadline"):
        asyncio.run(campaign.execute_phase("train"))
    resumed = run.Campaign(arguments, api, backend, manifest)
    with pytest.raises(TimeoutError, match="commit deadline"):
        asyncio.run(resumed.execute_phase("train"))
    assert resumed.cursor["training_step"] == 0
    assert len(backend.calls) == len(api.reports) == 1
    assert sum(event.startswith("report:") for event in api.events) == 1


def test_unknown_report_outcome_never_resubmits(tmp_path):
    arguments = arguments_fixture(tmp_path)
    manifest = manifest_fixture(arguments.data_root)
    api = FakeApi()
    api.publish = False
    api.raise_after_accept = True
    backend = FakeEpisodes(api)
    campaign = run.Campaign(arguments, api, backend, manifest)
    with pytest.raises(TimeoutError):
        asyncio.run(campaign.execute_phase("train"))
    api.reports.clear()
    resumed = run.Campaign(arguments, api, backend, manifest)
    with pytest.raises(RuntimeError, match="unknown report outcome"):
        asyncio.run(resumed.execute_phase("train"))
    assert len(backend.calls) == 1
    assert sum(event.startswith("report:") for event in api.events) == 1
    assert resumed.cursor["training_step"] == 0


def test_persisted_report_cannot_add_teacher_context_on_resume(tmp_path):
    arguments = arguments_fixture(tmp_path)
    arguments.commit_timeout_seconds = 0.01
    manifest = manifest_fixture(arguments.data_root)
    api = FakeApi()
    api.publish = False
    backend = FakeEpisodes(api)
    campaign = run.Campaign(arguments, api, backend, manifest)
    with pytest.raises(TimeoutError, match="commit deadline"):
        asyncio.run(campaign.execute_phase("train"))
    path = next((arguments.run_root / "reports").glob("*.json"))
    payload = report.read_object(path)
    payload["metadata"] = {"teacher_context": "verifier feedback"}
    report.write_object(path, payload)
    resumed = run.Campaign(arguments, api, backend, manifest)
    with pytest.raises(RuntimeError, match="persisted report differs"):
        asyncio.run(resumed.execute_phase("train"))
    assert len(backend.calls) == 1 and resumed.cursor["training_step"] == 0


@pytest.mark.parametrize("outcome", ["fault", "truncated"])
def test_training_fault_or_truncation_never_reports(tmp_path, outcome):
    arguments = arguments_fixture(tmp_path)
    manifest = manifest_fixture(arguments.data_root)
    api = FakeApi()
    backend = FakeEpisodes(api)
    backend.outcome = outcome
    campaign = run.Campaign(arguments, api, backend, manifest)
    with pytest.raises(RuntimeError, match="fault or truncation"):
        asyncio.run(campaign.execute_phase("train"))
    assert len(backend.calls) == 1 and not api.reports and campaign.cursor["training_step"] == 0


def test_frozen_evaluation_rejects_release_change_without_reporting(tmp_path):
    arguments = arguments_fixture(tmp_path)
    manifest = manifest_fixture(arguments.data_root)
    api = FakeApi()
    backend = FakeEpisodes(api)
    campaign = run.Campaign(arguments, api, backend, manifest)
    asyncio.run(campaign.execute_phase("train"))
    previous = copy.deepcopy(api.reports)
    backend.change_release = True
    with pytest.raises(RuntimeError, match="frozen evaluation release changed"):
        asyncio.run(campaign.execute_phase("independent"))
    assert api.reports == previous


@pytest.mark.parametrize(
    "change", ["partial", "extra", "multiple", "pending", "unverified", "wrong_parent", "wrong_step", "not_served"]
)
def test_commit_requires_exact_verified_consumption(change):
    api = FakeApi()
    api.current["release_id"] = "next"
    commit: report.JsonObject = {
        "step": 1,
        "operation": "training",
        "operation_verified": True,
        "pending": False,
        "artifact_ref": {"release_id": "next", "parent_release_id": "base"},
        "consumed_ids": ["report", "receipt-0", "receipt-1"],
    }
    api.history = [commit]
    if change == "partial":
        commit["consumed_ids"] = ["report", "receipt-0"]
    elif change == "extra":
        commit["consumed_ids"].append("unrelated")
    elif change == "multiple":
        api.history.append(copy.deepcopy(commit))
    elif change == "pending":
        commit["pending"] = True
        assert run.matched_commit(api, ["report"], ["receipt-0", "receipt-1"], "base", 1) is None
        return
    elif change == "unverified":
        commit["operation_verified"] = False
    elif change == "wrong_parent":
        commit["artifact_ref"]["parent_release_id"] = "other"
    elif change == "wrong_step":
        commit["step"] = 2
    else:
        api.current["release_id"] = "not-served"
    with pytest.raises(RuntimeError):
        run.matched_commit(api, ["report"], ["receipt-0", "receipt-1"], "base", 1)


def test_opd_report_empty_context_contract():
    episode: report.JsonObject = {"phase": "train", "references": ["a", "b"], "score": 0.0, "report_id": "r"}
    payload = report.terminal_report("opd", episode, "", 1)
    assert payload == {
        "agent_record_id": "r",
        "references": ["a", "b"],
        "score": 0.0,
        "metadata": {"teacher_context": ""},
    }
    for context in ("demonstration", "The submitted solution did not pass the verifier."):
        with pytest.raises(ValueError, match="empty"):
            report.terminal_report("opd", episode, context, 1)
    episode["phase"] = "independent"
    with pytest.raises(ValueError, match="evaluation"):
        report.terminal_report("opd", episode, "", 1)
    episode.update(phase="train", references=["a", "a"])
    with pytest.raises(ValueError, match="repeat"):
        report.terminal_report("opd", episode, "", 1)


def native_episode_fixture(processor: native.CapturedOPDProcessor, context: str = "") -> report.JsonObject:
    messages = [{"role": "user", "content": "task"}]
    tool_messages = [
        *messages,
        {"role": "assistant", "content": "tool call"},
        {"role": "user", "content": "tool output"},
    ]
    episode: report.JsonObject = {
        "report_id": "terminal-report",
        "references": [],
        "turns": [],
        "release_id": "release",
    }
    for index, (request_messages, tokens) in enumerate(
        ((messages, [10, 11, 20]), (tool_messages, [10, 11, 20, 30, 21]))
    ):
        receipt = f"turn-{index}"
        training = {
            "tokens": tokens,
            "loss_mask": [1],
            "rollout_log_probs": [-0.2],
            "runtime_load_id": "runtime",
            "request_messages": request_messages,
            "request_tools": None,
        }
        response_text = "tool call" if index == 0 else "FINAL"
        payload = {
            "messages": request_messages,
            "response": {
                "training": training,
                "choices": [{"message": {"role": "assistant", "content": response_text}}],
            },
        }
        processor.ingest(
            AgentRecord.create(
                scenario="opd-fixture",
                request_type=RequestType.INFERENCE,
                agent_record_id=receipt,
                payload=payload,
                artifact_ref=LiveWeightArtifactRef("content", "release", "base", "runtime"),
            )
        )
        episode["references"].append(receipt)
        episode["turns"].append(
            {"receipt": receipt, "record": {"payload": payload, "artifact_ref": {"release_id": "release"}}}
        )
    body = TeacherContextReport(teacher_context=context, score=0.0).to_dict(references=episode["references"])
    processor.ingest(
        AgentRecord.create(
            scenario="opd-fixture",
            request_type=RequestType.REPORT,
            agent_record_id="terminal-report",
            payload=body,
            references=tuple(episode["references"]),
        )
    )
    return episode


def native_context_fixture(output: Path) -> ProcessorContext:
    return ProcessorContext(
        "opd-fixture",
        {
            "native_sample_dir": str(output),
            "accept_multi_turn_policy_samples": True,
            "batch_size": 1,
            "max_teacher_tokens": 16384,
        },
        TeacherContextReport,
    )


@pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf"), True])
def test_native_qualification_rejects_invalid_weights_even_with_active_episode(tmp_path, monkeypatch, weight):
    monkeypatch.setitem(sys.modules, "transformers", None)
    processor = native.CapturedOPDProcessor(native_context_fixture(tmp_path / "teacher-records"))
    episode = native_episode_fixture(processor)
    processor.build_batch()
    capture = report.read_object(tmp_path / "teacher-records/terminal-report.json")
    valid = copy.deepcopy(capture)
    invalid_episode = copy.deepcopy(episode)
    invalid_episode["report_id"] = "invalid-report"
    invalid_episode["phase"] = "train"
    episode["phase"] = "train"
    capture["report_id"] = "invalid-report"
    capture["distill_sample_weight"] = weight
    with pytest.raises(ValueError, match="finite and non-negative"):
        qualification.check_native_sample(capture, invalid_episode)
    if weight == -1.0:
        report.write_object(tmp_path / "episodes/valid.json", episode)
        report.write_object(tmp_path / "episodes/invalid.json", invalid_episode)
        report.write_object(tmp_path / "teacher-records/terminal-report.json", valid)
        report.write_object(tmp_path / "teacher-records/invalid-report.json", capture)
        with pytest.raises(ValueError, match="finite and non-negative"):
            qualification.qualify_inputs(tmp_path)


def test_native_empty_context_uses_same_ids_and_masks_without_tokenizer(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    output = tmp_path / "teacher-records"
    processor = native.CapturedOPDProcessor(native_context_fixture(output))
    episode = native_episode_fixture(processor)
    batch = processor.build_batch()
    assert len(batch.items) == 1
    sample = batch.items[0]
    capture = report.read_object(output / "terminal-report.json")
    assert capture["tokens"] == capture["teacher_tokens"] == list(sample.training["tokens"]) == [10, 11, 20, 30, 21]
    assert capture["loss_mask"] == [1, 0, 1]
    assert capture["source_agent_record_ids"] == ["turn-0", "turn-1", "terminal-report"]
    assert capture["teacher_input"]["messages"] == [{"role": "user", "content": "task"}]
    assert capture["teacher_input"]["prompt_text_decoded_from_captured_tokens"] is None
    assert "no teacher prompt rendering" in capture["teacher_input"]["source"]
    checked = qualification.check_native_sample(capture, episode, require_multi_turn=True)
    assert checked["assistant_token_count"] == 2 and checked["masked_context_token_count"] == 1
    assert checked["teacher_student_sequence_identity"] is True and checked["active"] is True
    assert output.stat().st_mode & 0o777 == 0o700
    assert (output / "terminal-report.json").stat().st_mode & 0o777 == 0o600
    capture["teacher_tokens"][0] = 999
    with pytest.raises(ValueError, match="exact recorded student"):
        qualification.check_native_sample(capture, episode)


def test_native_algorithm_parses_proposed_teacher_settings(tmp_path):
    from recipes.opd.slime import OpdAlgorithm

    arguments = arguments_fixture(tmp_path)
    options = {**run.OPD_TRAINING_OPTIONS, "opd-teacher-checkpoint": arguments.teacher_checkpoint}
    flags = [value for key, setting in options.items() for value in ("--" + key, setting)]
    settings, remaining = OpdAlgorithm().parse_specific_options(flags)
    assert not remaining
    assert settings.teacher == "separate" and settings.teacher_checkpoint == arguments.teacher_checkpoint
    assert settings.divergence == "reverse" and settings.top_k == 1
    assert settings.teacher_update_rate == 0 and settings.importance_sampling_cap == 0
    assert settings.skip_response_tokens == 0


def test_native_teacher_window_overflow_blocks_sample(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    context = native_context_fixture(tmp_path)
    context.config["max_teacher_tokens"] = 4
    processor = native.CapturedOPDProcessor(context)
    with pytest.raises(ValueError, match="exceeds max_teacher_tokens"):
        native_episode_fixture(processor)
    assert not list(tmp_path.glob("*.json"))


def test_native_rejects_demonstration_context(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    processor = native.CapturedOPDProcessor(native_context_fixture(tmp_path))
    with pytest.raises(ValueError, match="teacher_context must be empty"):
        native_episode_fixture(processor, "privileged demonstration")
        processor.build_batch()
