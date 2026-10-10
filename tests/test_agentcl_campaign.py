"""Minimal three-method campaign contracts; fixtures make no model or container calls."""

import asyncio
import copy
from unittest.mock import patch

import pytest
from tests.test_agentcl_tasks import manifest_fixture

from recipes.agentcl import run


class Api(run.ReefApi):
    def __init__(self, attempts):
        self.attempts = attempts
        self.release = {"release_id": "base", "parent_release_id": None, "checkpoint": True}
        self.history = []
        self.reports = []
        self.pending = []
        self.events = []
        self.fail_report = False

    def current_release(self):
        return copy.deepcopy(self.release)

    def commits(self, record_ids):
        return [
            copy.deepcopy(row) for row in self.history if not record_ids or set(record_ids) & set(row["consumed_ids"])
        ]

    def record(self, record_id):
        return None

    def report(self, payload):
        self.events.append(("report", payload["agent_record_id"]))
        self.reports.append(copy.deepcopy(payload))
        if self.fail_report:
            raise TimeoutError("ambiguous report acknowledgement")
        self.pending.append(payload)
        if len(self.pending) == self.attempts:
            step = len(self.history) + 1
            consumed = [row["agent_record_id"] for row in self.pending]
            consumed.extend(receipt for row in self.pending for receipt in row["references"])
            release = f"release-{step}"
            self.history.append(
                {
                    "step": step,
                    "operation": "training",
                    "operation_verified": True,
                    "pending": False,
                    "consumed_ids": consumed,
                    "artifact_ref": {"release_id": release, "parent_release_id": "base"},
                }
            )
            self.release = {"release_id": release, "parent_release_id": "base", "checkpoint": False}
            self.pending.clear()


class Episodes(run.EpisodeBackend):
    def __init__(self, api):
        self.api = api
        self.calls = []
        self.outcome = "completed"
        self.score = 1.0
        self.change_release = False
        self.fail = False

    def recover(self, episode_id):
        return None

    async def run(self, task, episode_id, phase, release):
        self.api.events.append(("episode", task["id"]))
        self.calls.append((task["id"], phase, release["release_id"]))
        if self.fail:
            raise TimeoutError("ambiguous inference outcome")
        if self.change_release:
            self.api.release["release_id"] = "other-writer"
        references = [episode_id + "-1", episode_id + "-2"]
        return {
            "episode_id": episode_id,
            "release_id": release["release_id"],
            "runtime_load_id": "load-" + release["release_id"],
            "outcome": self.outcome,
            "score": self.score,
            "completion_tokens": 4,
            "references": references,
            "turns": [
                {
                    "receipt": receipt,
                    "release_id": release["release_id"],
                    "runtime_load_id": "load-" + release["release_id"],
                }
                for receipt in references
            ],
        }


def campaign(tmp_path, method):
    with patch(
        "sys.argv",
        [
            "run.py",
            "train",
            "--method",
            method,
            "--profile",
            "smoke",
            "--teacher-checkpoint",
            "/teacher",
            "--run-root",
            str(tmp_path / "run"),
            "--data-root",
            str(tmp_path / "data"),
        ],
    ):
        args = run.parse_arguments()
    manifest = manifest_fixture(args.data_root)
    api = Api(args.attempts)
    backend = Episodes(api)
    return args, manifest, api, backend, run.Campaign(args, api, backend, manifest)


@pytest.mark.parametrize("method", ["opd", "sdft", "sdpo"])
def test_all_methods_train_complete_groups_then_evaluate_without_reports(tmp_path, method):
    args, _, api, backend, driver = campaign(tmp_path, method)
    for phase in ("baseline", "baseline-independent", "train", "frozen-repeat", "independent"):
        result = asyncio.run(driver.execute_phase(phase))
        assert result["accuracy"] == 1.0
    assert len(api.history) == 2
    assert len(api.reports) == 2 * args.attempts
    assert driver.state["training_step"] == 2
    assert all(call[2] == "release-2" for call in backend.calls if call[1] in ("frozen-repeat", "independent"))
    training_events = api.events[4 : 4 + 4 * args.attempts]
    for index in (0, 2 * args.attempts):
        assert [name for name, _ in training_events[index : index + args.attempts]] == ["episode"] * args.attempts
        assert [name for name, _ in training_events[index + args.attempts : index + 2 * args.attempts]] == [
            "report"
        ] * args.attempts
    for payload in api.reports:
        if method == "opd":
            assert payload["metadata"]["teacher_context"] == ""
        elif method == "sdft":
            assert payload["metadata"]["teacher_context"] == "Verified demonstration"
        else:
            assert payload["metadata"]["group"] == 0
            assert payload["metadata"]["rollout"] in range(args.attempts)
    episodes = list((args.run_root / "episodes").glob("*.json"))
    assert len(episodes) == 8 + 2 * args.attempts
    assert len(list((args.run_root / "commits").glob("*.json"))) == 2
    with pytest.raises(RuntimeError, match="already started"):
        asyncio.run(driver.execute_phase("train"))


@pytest.mark.parametrize("failure", ["inference", "report", "fault", "truncation", "release"])
def test_unknown_or_invalid_work_is_terminal_and_never_replayed(tmp_path, failure):
    args, manifest, api, backend, driver = campaign(tmp_path, "sdft")
    backend.fail = failure == "inference"
    api.fail_report = failure == "report"
    if failure in ("fault", "truncation"):
        backend.outcome = "fault" if failure == "fault" else "truncated"
        backend.score = None if failure == "fault" else 0.0
    backend.change_release = failure == "release"
    with pytest.raises((TimeoutError, RuntimeError)):
        asyncio.run(driver.execute_phase("train"))
    previous = len(backend.calls)
    restarted = run.Campaign(args, api, backend, manifest)
    with pytest.raises(RuntimeError, match="already started"):
        asyncio.run(restarted.execute_phase("train"))
    assert len(backend.calls) == previous
    assert not api.history


@pytest.mark.parametrize("field", ["method", "teacher_checkpoint", "seed"])
def test_changed_settings_require_new_run(tmp_path, field):
    args, manifest, api, backend, _ = campaign(tmp_path, "opd")
    if field == "method":
        args.method = "sdft"
    elif field == "teacher_checkpoint":
        args.teacher_checkpoint = "/changed-teacher"
    else:
        args.seed += 1
    with pytest.raises(RuntimeError, match="settings changed"):
        run.Campaign(args, api, backend, manifest)
    assert not backend.calls


def test_unqualified_training_never_starts_an_episode(tmp_path):
    args, _, _, backend, driver = campaign(tmp_path, "sdpo")
    (args.data_root / "reference-verification.json").unlink()
    with pytest.raises(RuntimeError, match="216 isolated"):
        asyncio.run(driver.execute_phase("train"))
    assert not backend.calls


def test_frozen_evaluation_waits_for_all_commits(tmp_path):
    _, _, api, backend, driver = campaign(tmp_path, "opd")
    with pytest.raises(RuntimeError, match="every configured"):
        asyncio.run(driver.execute_phase("independent"))
    assert not backend.calls and not api.reports


@pytest.mark.parametrize("change", ["operation", "consumed", "step", "parent"])
def test_commit_barrier_rejects_nonmatching_consumption(change):
    api = Api(1)
    api.report({"agent_record_id": "report", "references": ["turn1", "turn2"]})
    if change == "operation":
        api.history[0]["operation_verified"] = False
    elif change == "consumed":
        api.history[0]["consumed_ids"].append("other")
    elif change == "step":
        api.history[0]["step"] = 2
    else:
        api.history[0]["artifact_ref"]["parent_release_id"] = "other"
    with pytest.raises(RuntimeError):
        run.matched_commit(api, ["report"], ["turn1", "turn2"], "base", 1)


@pytest.mark.parametrize("method", ["opd", "sdft", "sdpo"])
def test_dry_run_prints_native_config_and_upload_override_without_requests(tmp_path, capsys, method):
    argv = [
        "run.py",
        "train",
        "--method",
        method,
        "--profile",
        "smoke",
        "--teacher-checkpoint",
        "/teacher",
        "--run-root",
        str(tmp_path),
        "--dry-run",
        "--wandb-project",
        "reef-agentcl",
        "--wandb-entity",
        "recsys",
    ]
    with patch("sys.argv", argv):
        asyncio.run(run.main())
    output = capsys.readouterr().out
    assert f"serve-{method}.yaml" in output
    assert "--observability.wandb" in output
    assert '"submits": 0' in output
    assert not (tmp_path / "cursor.json").exists()
