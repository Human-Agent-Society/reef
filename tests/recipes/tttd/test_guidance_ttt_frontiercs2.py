"""FrontierCS 2 tasks for Guidance-TTT: task files, the judge adapter, and the executor key file."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

from recipes.tttd.examples.guidance_ttt.harness.config import EXAMPLE_DIR, FRONTIERCS2_TASKS, RunConfig
from recipes.tttd.examples.guidance_ttt.harness.contract import TaskContract
from recipes.tttd.examples.guidance_ttt.harness.execution import openai_compatible_backend, read_api_key
from recipes.tttd.examples.guidance_ttt.harness.prompts import build_execution_prompt
from recipes.tttd.examples.guidance_ttt.harness.state import LibraryEntry

JUDGES = EXAMPLE_DIR / "judges"


def load_judge_module():
    spec = importlib.util.spec_from_file_location("frontiercs2_server", JUDGES / "frontiercs2_server.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("task", FRONTIERCS2_TASKS)
def test_task_directory_matches_the_judge_registry(task):
    registry = json.loads((JUDGES / "frontiercs2_tasks.json").read_text())
    contract = TaskContract.load(
        EXAMPLE_DIR / "harbor" / task / "contract.json",
        problem_prompt=(EXAMPLE_DIR / "harbor" / task / "instruction.md").read_text(),
    )
    assert set(registry) == set(FRONTIERCS2_TASKS)
    assert contract.judge_problem_id == task
    assert contract.solution_language == registry[task]["language"]
    assert registry[task]["editable_file"] in contract.problem_prompt
    assert f"{task}:eval" in (EXAMPLE_DIR / "harbor" / task / "tests/grade.py").read_text()


def test_run_config_accepts_frontiercs2_tasks(monkeypatch, tmp_path):
    monkeypatch.setenv("GUIDANCE_TASK", "polar_express")
    monkeypatch.setenv("GUIDANCE_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("GUIDANCE_EXECUTOR", "openai")
    monkeypatch.setenv("GUIDANCE_EXECUTOR_MODEL", "provider/model")
    monkeypatch.setenv("GUIDANCE_EXECUTOR_URL", "http://127.0.0.1:1/v1")
    config = RunConfig.load()
    assert config.verifier_timeout_s > 1_800
    assert config.backend.model == "provider/model"
    assert config.contract().execution_context == ""
    (tmp_path / "execution_context.md").write_text("## `helper.py`\n")
    assert config.contract().execution_context == "## `helper.py`\n"


def test_execution_prompt_carries_repository_context_only_when_given():
    entry = LibraryEntry(
        id="e", parent_id=None, problem_id="t", timestep=0, guidance="", execution_thinking="", solution="x = 1",
        verifier_reward=1.0, verifier_raw_score=1.0, verifier_status="valid", verifier_message="ok",
        summary="s", reusable_idea="s", failure_mode=None,
    )  # fmt: skip
    common = {
        "problem_prompt": "problem",
        "selected_entry": entry,
        "guidance": "g",
        "solution_language": "python",
        "solution_contract": "c",
        "score_direction": "max",
        "raw_score_label": "score",
    }
    assert "<repository_context>" not in build_execution_prompt(**common).user
    prompt = build_execution_prompt(**common, execution_context="def helper(): ...")
    assert "<repository_context>\ndef helper(): ...\n</repository_context>" in prompt.user


def test_api_key_file_is_read_on_each_call(tmp_path):
    key_file = tmp_path / ".env"
    key_file.write_text("OTHER=1\nPROXY_KEY=first\n")
    backend = openai_compatible_backend(
        model="m", base_url="http://127.0.0.1:1/v1", api_key_env="PROXY_KEY", api_key_file=str(key_file)
    )
    assert read_api_key(backend) == "first"
    key_file.write_text("PROXY_KEY='second'\n")
    assert read_api_key(backend) == "second"
    assert "second" not in json.dumps(backend.safe_dict())


def test_patch_applies_to_the_pristine_file(tmp_path):
    judge = load_judge_module()
    repo = tmp_path / "agent-base/frontier/task/repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg/policy.py").write_text("def f():\n    return 1\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    task = judge.Task("t", tmp_path, "pkg/policy.py", "python", tmp_path / "t.sif", 2, 60.0)
    for code in ("def f():\n    return 2\n", "def f():\n    return 1\n"):
        (tmp_path / "solution.patch").write_text(judge.make_patch(task, code))
        subprocess.run(["git", "-C", str(repo), "apply", "--check", str(tmp_path / "solution.patch")], check=True)
    assert judge.make_patch(task, "def f():\n    return 1").count("+# unchanged") == 1


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ({"status": "passed", "raw_score": 4.5, "metrics": {"score_unbounded": 4.5}, "summary": "ok"}, (True, 4.5)),
        ({"status": "passed", "raw_score": 0.0, "metrics": {}, "summary": "no workload"}, (False, 0)),
        ({"status": "failed", "summary": "patch rejected", "failure": {"code": "invalid_submission"}}, (False, 0)),
    ],
)
def test_verifier_result_maps_to_the_judge_protocol(result, expected):
    judge = load_judge_module()
    task = judge.Task("t", Path("."), "f.py", "python", Path("t.sif"), 2, 60.0)
    payload = judge.FrontierCS2Judge._read(task, "train", result, 1.0)
    assert (payload["valid"], payload["score"]) == expected


def test_verifier_configuration_error_is_infrastructure():
    judge = load_judge_module()
    task = judge.Task("t", Path("."), "f.py", "python", Path("t.sif"), 2, 60.0)
    failed = {"status": "failed", "summary": "bad", "failure": {"code": "verifier_configuration_error"}}
    with pytest.raises(RuntimeError):
        judge.FrontierCS2Judge._read(task, "train", failed, 1.0)
