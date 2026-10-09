"""CPU fixtures for AgentCL workflow; no GPU or provider calls."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path
from typing import cast

import pytest

from recipes.sdft.examples.agentcl import metrics, render_traces, report, run, taskexport, verification


def manifest_fixture(root: Path) -> report.JsonObject:
    training: list[report.JsonValue] = []
    independent: list[report.JsonValue] = []
    reference_bytes = b"Verified demonstration"
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
        run_id="fixture",
        wandb_project=None,
        wandb_entity=None,
        profile="smoke",
        model_path="Qwen2.5-7B-Instruct",
        seed=42,
        scenario="fixture",
        service_url="http://fixture",
        steps=2,
        attempts=1,
        max_turns=8,
        native_input_checks=False,
        commit_timeout_seconds=1,
        poll_seconds=0.001,
    )


class FakeApi(run.ReefApi):
    def __init__(self, attempts: int = 1):
        self.current = {
            "release_id": "base",
            "checkpoint": True,
            "runtime_load_id": "runtime-base",
            "operation": "creation",
            "pending": False,
        }
        self.history = []
        self.reports = {}
        self.waiting = []
        self.attempts = attempts
        self.polls = 0
        self.raise_after_accept = False

    def current_release(self):
        return copy.deepcopy(self.current)

    def commits(self, record_ids):
        self.polls += 1
        return [
            copy.deepcopy(commit) for commit in self.history if set(record_ids).intersection(commit["consumed_ids"])
        ]

    def record(self, record_id):
        return None if record_id not in self.reports else {"payload": self.reports[record_id]}

    def report(self, payload):
        identifier = payload["agent_record_id"]
        if identifier in self.reports:
            return
        self.reports[identifier] = {key: value for key, value in payload.items() if key != "agent_record_id"}
        self.waiting.append(payload)
        if len(self.waiting) == self.attempts:
            step = len(self.history) + 1
            artifact = {
                "kind": "artifact",
                "release_id": f"release-{step}",
                "parent_release_id": self.current["release_id"],
            }
            consumed = [row["agent_record_id"] for row in self.waiting]
            consumed += [reference for row in self.waiting for reference in row["references"]]
            self.history.append(
                {
                    "step": step,
                    "operation": "training",
                    "operation_verified": True,
                    "pending": False,
                    "artifact_ref": artifact,
                    "consumed_ids": consumed,
                    "metrics": {"loss": 0.2},
                }
            )
            self.current = {
                "release_id": artifact["release_id"],
                "checkpoint": True,
                "runtime_load_id": f"runtime-{step}",
                "operation": "training",
                "pending": False,
            }
            self.waiting.clear()
        if self.raise_after_accept:
            self.raise_after_accept = False
            raise TimeoutError("accepted, response lost")


class FakeEpisodes(run.EpisodeBackend):
    def __init__(self):
        self.calls = []
        self.results = {}
        self.seeds = []
        self.fail = False
        self.outcome = "completed"

    async def run(self, task, episode_id, phase, release):
        self.calls.append((task["id"], phase, release["release_id"]))
        self.seeds.append(task["sampling_seed"])
        if self.fail:
            raise TimeoutError("unknown remote inference result")
        receipt = f"receipt-{episode_id}"
        result = {
            "episode_id": episode_id,
            "release_id": release["release_id"],
            "outcome": self.outcome,
            "score": 1.0,
            "references": [receipt],
            "turns": [{"receipt": receipt, "release_id": release["release_id"]}],
            "messages": [],
            "trajectory": {},
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "elapsed_seconds": 0.1,
        }
        self.results[episode_id] = copy.deepcopy(result)
        return result

    def recover(self, episode_id):
        return copy.deepcopy(self.results.get(episode_id))


def test_smoke_pair_is_found_by_identity(tmp_path):
    arguments = arguments_fixture(tmp_path)
    manifest = manifest_fixture(arguments.data_root)
    assert run.selected_training_positions(arguments, manifest) == [0, 95]


def test_campaign_waits_commits_and_frozen_evaluation_never_reports(tmp_path):
    arguments = arguments_fixture(tmp_path)
    manifest = manifest_fixture(arguments.data_root)
    api = FakeApi()
    backend = FakeEpisodes()
    campaign = run.Campaign(arguments, api, backend, manifest)
    asyncio.run(campaign.execute_phase("train"))
    assert backend.calls == [("0", "train", "base"), ("0", "train", "release-1")]
    assert len(api.history) == 2
    assert api.polls >= 4
    before = copy.deepcopy(api.reports)
    for task in manifest["training"] + manifest["independent"]:
        task["reference_path"] = "missing-private-reference"
    asyncio.run(campaign.execute_phase("frozen-repeat"))
    asyncio.run(campaign.execute_phase("independent"))
    assert api.reports == before
    assert all(call[2] == "release-2" for call in backend.calls[2:])
    again = run.Campaign(arguments, api, backend, manifest_fixture(arguments.data_root))
    asyncio.run(again.execute_phase("train"))
    assert len(backend.calls) == 6


def test_report_response_lost_reconciles_consumption_without_replay(tmp_path):
    arguments = arguments_fixture(tmp_path)
    manifest = manifest_fixture(arguments.data_root)
    api = FakeApi()
    api.raise_after_accept = True
    backend = FakeEpisodes()
    campaign = run.Campaign(arguments, api, backend, manifest)
    with pytest.raises(TimeoutError):
        asyncio.run(campaign.execute_phase("train"))
    resumed = run.Campaign(arguments, api, backend, manifest)
    asyncio.run(resumed.execute_phase("train"))
    assert len(backend.calls) == 2
    assert len(api.reports) == 2
    assert resumed.cursor["training_step"] == 2


def test_unknown_inference_outcome_never_replayed(tmp_path):
    arguments = arguments_fixture(tmp_path)
    manifest = manifest_fixture(arguments.data_root)
    api = FakeApi()
    backend = FakeEpisodes()
    backend.fail = True
    campaign = run.Campaign(arguments, api, backend, manifest)
    with pytest.raises(TimeoutError):
        asyncio.run(campaign.execute_phase("train"))
    resumed = run.Campaign(arguments, api, backend, manifest)
    with pytest.raises(RuntimeError, match="never replay"):
        asyncio.run(resumed.execute_phase("train"))
    assert len(backend.calls) == 1
    assert not api.reports


@pytest.mark.parametrize(
    "change", ["partial", "multiple", "pending", "rollback", "same_release", "wrong_parent", "wrong_step"]
)
def test_exact_commit_predicate(change):
    api = FakeApi()
    api.current = {"release_id": "next", "operation": "training"}
    commit = {
        "step": 1,
        "operation": "training",
        "operation_verified": True,
        "pending": False,
        "artifact_ref": {"release_id": "next", "parent_release_id": "base"},
        "consumed_ids": ["report", "receipt"],
    }
    api.history = [commit]
    if change == "partial":
        commit["consumed_ids"] = ["report"]
    elif change == "multiple":
        api.history.append(copy.deepcopy(commit))
    elif change == "pending":
        commit["pending"] = True
        assert run.matched_commit(api, ["report"], ["receipt"], "base", 1) is None
        return
    elif change == "rollback":
        commit["operation"] = "rollback"
    elif change == "same_release":
        commit["artifact_ref"]["release_id"] = "base"
    elif change == "wrong_parent":
        commit["artifact_ref"]["parent_release_id"] = "other"
    else:
        commit["step"] = 2
    with pytest.raises(RuntimeError):
        run.matched_commit(api, ["report"], ["receipt"], "base", 1)


def test_terminal_reports_reject_evaluation_and_duplicate_receipts():
    record = {"phase": "baseline", "references": ["a"], "score": 0.0, "report_id": "r", "attempt": 0}
    with pytest.raises(ValueError, match="evaluation"):
        report.terminal_report("sdft", record, "private", 1)
    record.update(phase="train", references=["a", "a"])
    with pytest.raises(ValueError, match="repeat"):
        report.terminal_report("sdft", record, "private", 1)
    assert report.stable_id("a", "train", "raw", "1", 0, "report") != report.stable_id(
        "a", "train", "new", "1", 0, "report"
    )


def test_metrics_compare_one_attempt_and_mean_k_separately():
    records = []
    for phase, attempt, score in [
        ("baseline", 0, 0.0),
        ("train", 0, 1.0),
        ("train", 1, 0.0),
        ("frozen-repeat", 0, 1.0),
        ("baseline-independent", 0, 0.0),
        ("independent", 0, 1.0),
    ]:
        records.append(
            {
                "phase": phase,
                "attempt": attempt,
                "score": score,
                "outcome": "completed",
                "category": "new",
                "task_id": "same",
                "position": 0,
                "release_id": "r",
            }
        )
    summary = metrics.build_summary(records)
    assert summary["gains"] == {"PG": 100.0, "SG": 0.0, "GG": 100.0}
    assert summary["phases"]["train"]["one_attempt"]["accuracy"] == 1.0
    assert summary["phases"]["train"]["mean_at_k"]["accuracy"] == 0.5


def test_trace_json_is_not_executable_markup(tmp_path):
    record = {
        "report_id": "r",
        "episode_id": "e",
        "phase": "train",
        "task_id": "<script>alert(1)</script>",
        "category": "new",
        "attempt": 0,
        "turns": [{"text": "</script><script>alert(1)</script>"}],
        "outcome": "fault",
    }
    report.write_object(tmp_path / "episodes/e.json", record)
    output = render_traces.render_traces(tmp_path).read_text()
    assert "</script><script>alert" not in output
    assert "\\u003cscript" in output
    encoded = output.split('type="application/json">', 1)[1].split("</script>", 1)[0]
    assert json.loads(encoded)[0]["task_id"] == record["task_id"]


def test_cpu_config_and_zero_submission_dry_run(tmp_path, monkeypatch):
    from reef.service.deploy.config_utils import load_config
    from reef.service.deploy.service_config import service_config_from_mapping

    for key, value in {
        "REEF_PORT": "28902",
        "REEF_TOKEN": "",
        "AGENTCL_ROUTER_PORT": "23002",
        "AGENTCL_RUN_ROOT": str(tmp_path),
        "AGENTCL_MODEL_PATH": "/model",
        "AGENTCL_BATCH_SIZE": "1",
        "AGENTCL_STEPS": "96",
        "AGENTCL_WARMUP": "10",
    }.items():
        monkeypatch.setenv(key, value)
    settings = load_config(run.HERE / "serve.yaml")
    service = service_config_from_mapping(settings)
    assert service.model_path == "/model"
    assert service.training_backend_options["sdft-skip-response-tokens"] == "0"
    assert service.training_backend_options["sdft-teacher-update-rate"] == "0.01"
    assert service.inference_num_gpus == 4
    spec = run.run_specification(arguments_fixture(tmp_path), None)
    assert spec["student_window_tokens"] == 8192
    assert spec["teacher_window_tokens"] == 16384


def test_independent_verification_rejects_reordered_and_wrong_release(tmp_path):
    arguments = arguments_fixture(tmp_path)
    manifest = manifest_fixture(arguments.data_root)
    api = FakeApi(attempts=arguments.attempts)
    backend = FakeEpisodes()
    campaign = run.Campaign(arguments, api, backend, manifest)
    for phase in ("baseline", "baseline-independent", "train", "frozen-repeat", "independent"):
        asyncio.run(campaign.execute_phase(phase))
    records = [report.read_object(path) for path in (arguments.run_root / "episodes").glob("*.json")]
    report.write_object(arguments.run_root / "summary.json", metrics.build_summary(records))
    initial = verification.verify_run(arguments.run_root)
    assert initial["local_checks_passed"] is True
    assert initial["complete"] is False
    for path in (arguments.run_root / "episodes").glob("*.json"):
        record = report.read_object(path)
        if record["phase"] == "train" and record["campaign_position"] == 1:
            record["release_id"] = "base"
            record["turns"][0]["release_id"] = "base"
            record["references"] = ["different-receipt"]
            record["campaign_position"] = 0
            report.write_object(path, record)
            break
    result = verification.verify_run(arguments.run_root)
    assert result["local_checks_passed"] is False
    assert any("supplied order" in value for value in result["failures"])
    assert any("prior committed release" in value for value in result["failures"])
    assert any("turn receipts" in value for value in result["failures"])


def test_trace_large_integers_order_and_offline_csp(tmp_path):
    for position in (1, 0):
        record = {
            "report_id": str(position),
            "episode_id": str(position),
            "phase": "train",
            "category": "new",
            "task_id": str(position),
            "attempt": 0,
            "position": position,
            "turns": [],
            "outcome": "completed",
            "large_integer": 2**60 + 1,
        }
        report.write_object(tmp_path / f"episodes/{1 - position}.json", record)
    output = render_traces.render_traces(tmp_path).read_text()
    encoded = output.split('type="application/json">', 1)[1].split("</script>", 1)[0]
    rows = json.loads(encoded)
    assert [row["position"] for row in rows] == [0, 1]
    assert rows[0]["large_integer"] == str(2**60 + 1)
    assert "connect-src 'none'" in output
    assert "script-src 'sha256-" in output
    assert "Reset filters" in output
    assert "Download filtered JSON" in output


def test_http_adapter_uses_actual_native_history_envelope(tmp_path):
    from aiohttp.test_utils import TestClient, TestServer

    from reef.core import AgentRecord, RequestType
    from reef.dispatcher import build_default_dispatcher
    from reef.service.app import create_app
    from reef.storage.commits import CommitRecord
    from reef.storage.sqlite import SQLiteScenarioStorage

    async def check():
        dispatcher = build_default_dispatcher(
            agent_record_dir=tmp_path / "records", scenario_storage=SQLiteScenarioStorage(tmp_path / "records")
        )
        scenario = dispatcher.get_or_create_scenario("fixture")
        scenario.records.append(
            AgentRecord.create(
                scenario="fixture",
                request_type=RequestType.INFERENCE,
                agent_record_id="receipt",
                payload={"messages": [{"role": "user", "content": "question"}]},
            )
        )
        scenario.store.commit_log.append(
            CommitRecord(
                scenario="fixture",
                step=1,
                artifact_ref=scenario.current_artifact_ref(),
                checkpoint=False,
                algorithm_state={},
                high_water_sequence=1,
                high_water_offset=1,
                consumed_ids=frozenset({"receipt", "report"}),
            )
        )
        client = TestClient(TestServer(create_app(dispatcher, tokens="fixture-token")))
        await client.start_server()
        api = run.HttpReefApi(str(client.make_url("/")).rstrip("/"), "fixture", "fixture-token")
        try:
            current = await asyncio.to_thread(api.current_release)
            assert isinstance(current["release_id"], str)
            commits = await asyncio.to_thread(api.commits, ["report"])
            assert commits[0]["operation"] == "training"
            assert commits[0]["operation_verified"] is True
            assert commits[0]["pending"] is False
            assert commits[0]["consumed_ids"] == ["receipt", "report"]
            assert "parent_release_id" in commits[0]["artifact_ref"]
            record = await asyncio.to_thread(api.record, "receipt")
            assert record["payload"]["messages"][0]["content"] == "question"
            assert await asyncio.to_thread(api.record, "missing") is None
        finally:
            await client.close()
            dispatcher.close()

    asyncio.run(check())


def test_live_commit_parent_is_last_durable_checkpoint():
    api = FakeApi()
    api.current = {"release_id": "live-2", "operation": "training"}
    api.history = [
        {
            "step": 2,
            "operation": "training",
            "operation_verified": True,
            "pending": False,
            "artifact_ref": {"kind": "live_weights", "release_id": "live-2", "parent_release_id": "base"},
            "consumed_ids": ["report", "receipt"],
        }
    ]
    assert run.matched_commit(api, ["report"], ["receipt"], "live-1", 2, "base")["step"] == 2


def test_native_capture_preserves_actual_loss_inputs(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from recipes.sdft.examples.agentcl import native, qualification
    from reef.core import AgentRecord, RequestType
    from reef.core.artifact_ref import LiveWeightArtifactRef
    from reef.core.reports import TeacherContextReport
    from reef.train.types import ProcessorContext

    class Tokenizer:
        def from_pretrained(self, path, **kwargs):
            return self

        def apply_chat_template(self, messages, **kwargs):
            return [100, 101]

        def decode(self, tokens, **kwargs):
            return "captured:" + ",".join(str(token) for token in tokens)

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=Tokenizer()))
    output = tmp_path / "teacher-records"
    context = ProcessorContext(
        "fixture",
        {
            "tokenizer_path": "fixture-tokenizer",
            "native_sample_dir": str(output),
            "accept_multi_turn_policy_samples": True,
            "batch_size": 1,
            "groups_per_step": 1,
            "rollouts_per_group": 2,
            "max_teacher_tokens": 16384,
            "max_teacher_prompt_tokens": 16384,
            "include_environment_feedback": True,
        },
        TeacherContextReport,
    )
    processor = native.CapturedSDFTProcessor(context)
    messages = [{"role": "user", "content": "fixture task"}]
    first_response = "```python\nprint(1)\n```"
    second_messages = [
        *messages,
        {"role": "assistant", "content": first_response},
        {"role": "user", "content": "tool output: 1"},
    ]
    episodes = []
    for attempt in range(1):
        turn_specs = [
            (messages, [10, 11, 20], first_response),
            (second_messages, [10, 11, 20, 30, 21], "FINAL\n```python\ndef solution(): return 1\n```"),
        ]
        turns = []
        references = []
        for index, (request_messages, tokens, response_text) in enumerate(turn_specs):
            receipt = f"turn-{attempt}-{index}"
            references.append(receipt)
            training = {
                "tokens": tokens,
                "loss_mask": [1],
                "rollout_log_probs": [-0.2],
                "runtime_load_id": "runtime",
                "request_messages": request_messages,
                "request_tools": None,
            }
            payload = {
                "messages": request_messages,
                "response": {
                    "choices": [{"message": {"role": "assistant", "content": response_text}}],
                    "training": training,
                },
            }
            record = AgentRecord.create(
                scenario="fixture",
                request_type=RequestType.INFERENCE,
                agent_record_id=receipt,
                payload=payload,
                artifact_ref=LiveWeightArtifactRef("content", "release", "base", "runtime"),
            )
            processor.ingest(record)
            turns.append(
                {"receipt": receipt, "record": {"payload": payload, "artifact_ref": {"release_id": "release"}}}
            )
        report_id = f"report-{attempt}"
        body = TeacherContextReport(teacher_context="fixture demonstration", score=1.0).to_dict(references=references)
        processor.ingest(
            AgentRecord.create(
                scenario="fixture",
                request_type=RequestType.REPORT,
                agent_record_id=report_id,
                payload=body,
                references=tuple(references),
            )
        )
        episodes.append({"report_id": report_id, "references": references, "turns": turns, "release_id": "release"})
    batch = processor.build_batch()
    assert len(batch.items) == 1
    for sample, episode in zip(batch.items, episodes, strict=True):
        capture = report.read_object(output / f"{episode['report_id']}.json")
        assert capture["tokens"] == list(sample.training["tokens"])
        assert capture["teacher_tokens"] == list(sample.training["teacher_tokens"])
        assert capture["trajectory"] == dict(sample.trajectory)
        assert capture["group_id"] == sample.group_id
        assert capture["source_agent_record_ids"] == list(sample.source_agent_record_ids)
        assert output.stat().st_mode & 0o777 == 0o700
        checked = qualification.check_native_sample(capture, episode)
        assert checked["assistant_token_count"] == 2
        assert checked["masked_context_token_count"] == 1
        assert capture["teacher_input"]["prompt_text_decoded_from_captured_tokens"] == "captured:100,101"
        assert (output / f"{episode['report_id']}.json").stat().st_mode & 0o777 == 0o600
        capture["loss_mask"][1] = 1
        with pytest.raises(ValueError, match="tool/context"):
            qualification.check_native_sample(capture, episode)


def test_native_diagnostic_secret_fields_are_removed():
    from recipes.sdft.examples.agentcl.native import diagnostic_json

    assert diagnostic_json(
        {
            "authorization": "private",
            "api-key": "private",
            "records": [{"tokens": [1, 2], "token": "private", "messages": [{"content": "exact"}]}],
        }
    ) == {"records": [{"tokens": [1, 2], "messages": [{"content": "exact"}]}]}
