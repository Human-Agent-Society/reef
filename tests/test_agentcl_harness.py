"""Deterministic CPU fixtures for the multi-turn Harbor student harness."""

from __future__ import annotations

import ast
import asyncio
import contextlib
import copy
import importlib
import json
import os
import pty
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from harbor.environments.base import ExecResult
from harbor.models.trajectories.trajectory import Trajectory


@pytest.fixture
def module():
    return importlib.import_module("recipes.agentcl.harness.episode")


def fixture_model(module, *, changes=None):
    class FixtureModel(module.RecordedModel):
        def __init__(self):
            self.requests = []
            self.previous_tokens = []
            self.records = {}

        def read_record(self, receipt):
            return copy.deepcopy(self.records[receipt])

        def complete(self, request, release_id):
            index = len(self.requests)
            self.requests.append(copy.deepcopy(request))
            if index == 0:
                text = "```python\nvalue = 41\nprint(value)\n```"
            elif index == 1:
                text = "```python\nprint(value + 1)\n```"
            else:
                text = "FINAL\n```python\ndef answer():\n    return 42\n```"
            prompt = [*self.previous_tokens, 100 + index]
            output = [200 + index, 300 + index]
            self.previous_tokens = prompt + output
            response = {
                "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(output)},
                "training": {
                    "tokens": self.previous_tokens[:],
                    "loss_mask": [1, 1],
                    "rollout_log_probs": [-0.1, -0.2],
                    "runtime_load_id": "runtime-one",
                },
            }
            headers = {"x-reef-agent-record-id": f"receipt-{index}", "x-reef-release-id": release_id}
            if changes:
                changes(index, response, headers)
            self.records[f"receipt-{index}"] = {
                "agent_record_id": f"receipt-{index}",
                "request_type": "inference",
                "artifact_ref": {"release_id": release_id},
                "payload": {**copy.deepcopy(request), "response": copy.deepcopy(response)},
            }
            public_response = {key: value for key, value in response.items() if key != "training"}
            return public_response, headers

    return FixtureModel()


def fixture_sandbox(module):
    class FixtureSandbox(module.EpisodeSandbox):
        def __init__(self):
            self.actions = []
            self.answer = None
            self.started = 0
            self.closed = 0

        async def start(self):
            self.started += 1

        async def execute(self, code, timeout_seconds, output_limit):
            self.actions.append(code)
            return {"status": "ok", "output": "41\n" if len(self.actions) == 1 else "42\n"}

        async def submit(self, code):
            self.answer = code

        async def close(self):
            self.closed += 1

    return FixtureSandbox()


def test_three_turn_episode_records_all_receipts_exact_history_and_atif(module, tmp_path):
    model = fixture_model(module)
    sandbox = fixture_sandbox(module)
    settings = module.EpisodeSettings("run/task/attempt0", "fixture-model", "release-one")
    output = tmp_path / "host" / "episode.json"
    runner = module.EpisodeRunner(model, sandbox, settings, tmp_path / "agent", output)
    episode = asyncio.run(runner.run("Problem only."))
    assert episode["outcome"] == "completed"
    assert episode["references"] == ["receipt-0", "receipt-1", "receipt-2"]
    assert episode["completion_tokens"] == 6
    assert episode["runtime_load_id"] == "runtime-one"
    assert sandbox.started == sandbox.closed == 1
    assert len(sandbox.actions) == 2
    assert sandbox.answer == "def answer():\n    return 42\n"
    for index in (1, 2):
        previous = model.requests[index - 1]["messages"]
        current = model.requests[index]["messages"]
        assert current[: len(previous)] == previous
        assert current[-2] == episode["turns"][index - 1]["response"]["choices"][0]["message"]
        assert current[-1]["role"] == "user" and "Code execution result:" in current[-1]["content"]
        assert "literal uppercase line FINAL" in current[-1]["content"]
        assert "with no heading or explanation before it" in current[-1]["content"]
        assert "FINAL\n```python\n<your complete Python module>\n```" in current[-1]["content"]
        assert "Do not write 'Final solution:'." in current[-1]["content"]
        assert "A code block without uppercase FINAL runs another development action." in current[-1]["content"]
    assert json.loads(output.read_text())["references"] == episode["references"]
    records = [json.loads(line) for line in (tmp_path / "agent/turns.jsonl").read_text().splitlines()]
    assert [turn["receipt"] for turn in records] == episode["references"]
    trajectory = Trajectory.model_validate(episode["trajectory"])
    agent_steps = [step for step in trajectory.steps if step.source == "agent"]
    assert len(agent_steps) == 3
    assert sum(len(step.metrics.completion_token_ids) for step in agent_steps) == 6


@pytest.mark.parametrize("text", ["No code block.", "```python\nvalue = 1\n```\n```python\nvalue = 2\n```"])
def test_format_error_feedback_includes_literal_final_template(module, tmp_path, text):
    def change(index, response, headers):
        if index == 0:
            response["choices"][0]["message"]["content"] = text

    model = fixture_model(module, changes=change)
    sandbox = fixture_sandbox(module)
    runner = module.EpisodeRunner(
        model, sandbox, module.EpisodeSettings("format-feedback", "fixture", "release-one"), tmp_path
    )
    episode = asyncio.run(runner.run("Problem only."))
    feedback = model.requests[1]["messages"][-1]
    assert feedback["role"] == "user"
    assert feedback["content"].startswith("Return exactly one fenced python block.\n")
    assert "literal uppercase line FINAL" in feedback["content"]
    assert "with no heading or explanation before it" in feedback["content"]
    assert "FINAL\n```python\n<your complete Python module>\n```" in feedback["content"]
    assert "Replace the placeholder with your code." in feedback["content"]
    assert "Do not write 'Final solution:'." in feedback["content"]
    assert "A code block without uppercase FINAL runs another development action." in feedback["content"]
    assert model.requests[1]["messages"][-2] == {"role": "assistant", "content": text}
    assert episode["outcome"] == "completed"
    assert episode["references"] == ["receipt-0", "receipt-1", "receipt-2"]
    assert len(sandbox.actions) == 1


@pytest.mark.parametrize(
    ("prefix", "submitted"),
    [
        ("Final solution:", False),
        ("final", False),
        ("Final", False),
        ("Here is FINAL", False),
        ("FINAL", True),
        (" \nFINAL", True),
    ],
)
def test_final_parser_keeps_case_sensitive_start_marker(module, tmp_path, prefix, submitted):
    code = "def answer():\n    return 42\n"
    text = prefix + "\n\n```python\n" + code + "```"

    def change(index, response, headers):
        response["choices"][0]["message"]["content"] = text

    model = fixture_model(module, changes=change)
    sandbox = fixture_sandbox(module)
    runner = module.EpisodeRunner(
        model, sandbox, module.EpisodeSettings("final-marker", "fixture", "release-one", max_turns=1), tmp_path
    )
    episode = asyncio.run(runner.run("Problem only."))
    assert episode["references"] == ["receipt-0"]
    assert episode["messages"][1] == {"role": "assistant", "content": text}
    assert episode["turns"][0]["record"]["payload"]["response"]["training"]["loss_mask"] == [1, 1]
    if submitted:
        assert episode["outcome"] == "completed"
        assert sandbox.answer == code
        assert sandbox.actions == []
    else:
        assert episode["outcome"] == "truncated"
        assert episode["fault"] == "maximum student turns reached without a final submission"
        assert sandbox.answer is None
        assert sandbox.actions == [code]


def test_lowercase_final_exhausts_original_eight_turn_budget_without_submission(module, tmp_path):
    text = "Final solution:\n\n```python\ndef answer():\n    return 42\n```"

    def change(index, response, headers):
        response["choices"][0]["message"]["content"] = text

    model = fixture_model(module, changes=change)
    sandbox = fixture_sandbox(module)
    settings = module.EpisodeSettings("nonfinal-eight-turns", "fixture", "release-one")
    runner = module.EpisodeRunner(model, sandbox, settings, tmp_path)
    episode = asyncio.run(runner.run("Problem only."))
    assert settings.max_turns == 8
    assert settings.max_episode_tokens == 8192
    assert episode["outcome"] == "truncated"
    assert episode["fault"] == "maximum student turns reached without a final submission"
    assert episode["references"] == [f"receipt-{index}" for index in range(8)]
    assert len(sandbox.actions) == 8
    assert sandbox.answer is None
    assert episode["completion_tokens"] == 16
    assert [message["content"] for message in episode["messages"] if message["role"] == "assistant"] == [text] * 8
    for turn in episode["turns"]:
        assert turn["request"]["max_tokens"] == 2048
        assert turn["request"]["temperature"] == 0.7
        assert turn["record"]["payload"]["response"]["training"]["loss_mask"] == [1, 1]
    for index in range(1, 8):
        previous = model.requests[index - 1]["messages"]
        current = model.requests[index]["messages"]
        assert current[: len(previous)] == previous
        assert "FINAL\n```python\n<your complete Python module>\n```" in current[-1]["content"]
    saved = json.loads((tmp_path / "episode.json").read_text())
    assert saved["outcome"] == "truncated"
    assert saved["references"] == episode["references"]


@pytest.mark.parametrize("limit", ["response-length", "episode-window"])
def test_exact_final_cannot_bypass_training_truncation(module, tmp_path, limit):
    def change(index, response, headers):
        response["choices"][0]["message"]["content"] = "FINAL\n```python\ndef answer():\n    return 42\n```"
        if limit == "response-length":
            response["choices"][0]["finish_reason"] = "length"
        else:
            response["usage"]["prompt_tokens"] = 8192

    sandbox = fixture_sandbox(module)
    runner = module.EpisodeRunner(
        fixture_model(module, changes=change),
        sandbox,
        module.EpisodeSettings("truncated-final", "fixture", "release-one"),
        tmp_path,
    )
    episode = asyncio.run(runner.run("Problem only."))
    assert episode["outcome"] == "truncated"
    assert episode["fault"] == "student response or episode token window was exhausted"
    assert episode["references"] == ["receipt-0"]
    assert sandbox.answer is None
    assert sandbox.actions == []


@pytest.mark.parametrize("problem", ["drift", "version", "mask", "log_probs", "receipt"])
def test_strict_qualification_rejects_faults_without_dropping_recorded_turns(module, tmp_path, problem):
    def change(index, response, headers):
        if index != 1:
            return
        if problem == "drift":
            response["training"]["tokens"][0] = 999
        elif problem == "version":
            response["training"]["runtime_load_id"] = "runtime-other"
        elif problem == "mask":
            response["training"]["loss_mask"] = [1, 0]
        elif problem == "log_probs":
            response["training"]["rollout_log_probs"] = [-0.1]
        else:
            headers.pop("x-reef-agent-record-id")

    runner = module.EpisodeRunner(
        fixture_model(module, changes=change),
        fixture_sandbox(module),
        module.EpisodeSettings("fault-test", "fixture", "release-one"),
        tmp_path,
    )
    with pytest.raises(module.EpisodeFault):
        asyncio.run(runner.run("Problem only."))
    saved = json.loads((tmp_path / "episode.json").read_text())
    assert saved["outcome"] == "fault"
    assert len(saved["references"]) == (1 if problem == "receipt" else 2)
    assert saved["fault"]


def test_truncation_keeps_receipts_and_does_not_submit_answer(module, tmp_path):
    sandbox = fixture_sandbox(module)
    runner = module.EpisodeRunner(
        fixture_model(module),
        sandbox,
        module.EpisodeSettings("short", "fixture", "release-one", max_turns=1),
        tmp_path,
    )
    episode = asyncio.run(runner.run("Problem only."))
    assert episode["outcome"] == "truncated"
    assert episode["references"] == ["receipt-0"]
    assert sandbox.answer is None


def test_evaluation_needs_no_training_tensors_and_has_no_report_interface(module, tmp_path):
    def change(index, response, headers):
        response.pop("training")

    runner = module.EpisodeRunner(
        fixture_model(module, changes=change),
        fixture_sandbox(module),
        module.EpisodeSettings("eval", "fixture", "release-one", phase="independent"),
        tmp_path,
    )
    episode = asyncio.run(runner.run("Independent problem only."))
    assert episode["phase"] == "independent" and episode["outcome"] == "completed"
    assert all(turn["release_id"] == "release-one" for turn in episode["turns"])


def test_kernel_retains_state_and_resets_namespace_fixture_only(module):
    kernel = importlib.import_module("recipes.agentcl.harbor.environment.kernel")
    namespace = {"__name__": "fixture"}
    assert kernel.execute("value = 41", namespace, 1, 200)["status"] == "ok"
    assert kernel.execute("print(value + 1)", namespace, 1, 200)["output"] == "42\n"
    assert kernel.execute("print(value)", {"__name__": "new"}, 1, 200)["status"] == "error"
    assert len(kernel.execute("print('x' * 1000)", namespace, 1, 20)["output"]) == 20


def test_reef_adapter_does_not_rebind_scenario_to_current_policy_release(module):
    class FixtureClient:
        def post(self, path, scenario, request):
            assert path == "/v1/chat/completions"
            assert scenario == "fixture-scenario"
            return {}, {"x-reef-agent-record-id": "receipt"}

    adapter = module.ReefRecordedModel(FixtureClient(), "fixture-scenario")
    assert adapter.complete({}, "release-one")[1]["x-reef-agent-record-id"] == "receipt"


def test_student_final_is_validated_before_sandbox_submission_and_can_retry(module, tmp_path):
    def change(index, response, headers):
        if index == 0:
            response["choices"][0]["message"][
                "content"
            ] = "FINAL\n```python\nimport sys\ndef answer():\n    return 42\n```"

    model = fixture_model(module, changes=change)
    sandbox = fixture_sandbox(module)
    runner = module.EpisodeRunner(
        model, sandbox, module.EpisodeSettings("restricted-final", "fixture", "release-one"), tmp_path
    )
    episode = asyncio.run(runner.run("Problem only."))
    assert episode["outcome"] == "completed"
    assert len(episode["references"]) == 3
    assert model.requests[1]["messages"][-1]["content"].startswith("Final module rejected:")
    assert "import sys" not in sandbox.answer
    assert len(sandbox.actions) == 1


def test_rejected_main_guard_gets_explicit_feedback_and_valid_final_next_turn(module, tmp_path):
    valid_code = "import math\ndef answer():\n    return math.floor(42.5)\n"
    rejected_code = valid_code + "\nif __name__ == '__main__':\n    print(answer())\n"
    rejected_text = "FINAL\n```python\n" + rejected_code + "```"
    valid_text = "FINAL\n```python\n" + valid_code + "```"
    with pytest.raises(module.AnswerContractError) as captured:
        module.validate_final_module(rejected_code)
    original_error = str(captured.value)
    assert original_error == "Process control and interpreter introspection are not permitted."
    module.validate_final_module(valid_code)

    def change(index, response, headers):
        if index == 0:
            response["choices"][0]["message"]["content"] = rejected_text
        else:
            assert index == 1, "Recovery must not add model calls."
            feedback = model.requests[index]["messages"][-1]
            assert feedback["role"] == "user"
            assert feedback["content"].startswith("Final module rejected: " + original_error + "\n")
            assert "Remove all executable examples" in feedback["content"]
            assert "if __name__ == '__main__':" in feedback["content"]
            assert "only allowed imports and top-level function definitions" in feedback["content"]
            assert "FINAL\n```python\n<your complete Python module>\n```" in feedback["content"]
            response["choices"][0]["message"]["content"] = valid_text

    model = fixture_model(module, changes=change)
    sandbox = fixture_sandbox(module)
    settings = module.EpisodeSettings("main-guard-feedback", "fixture", "release-one")
    episode = asyncio.run(module.EpisodeRunner(model, sandbox, settings, tmp_path).run("Problem only."))
    assert episode["outcome"] == "completed" and episode["fault"] is None
    assert len(model.requests) == 2
    assert episode["references"] == ["receipt-0", "receipt-1"]
    assert episode["completion_tokens"] == 4
    assert sandbox.started == sandbox.closed == 1
    assert sandbox.actions == [] and sandbox.answer == valid_code
    assert settings.max_turns == 8 and settings.max_episode_tokens == 8192
    assert model.requests[0]["messages"] == [{"role": "user", "content": "Problem only."}]
    assert model.requests[1]["messages"][:-2] == model.requests[0]["messages"]
    assert model.requests[1]["messages"][-2] == {"role": "assistant", "content": rejected_text}
    assert episode["messages"] == model.requests[1]["messages"] + [{"role": "assistant", "content": valid_text}]
    for index, turn in enumerate(episode["turns"]):
        assert turn["request"] == model.requests[index]
        assert turn["request"]["max_tokens"] == 2048 and turn["request"]["temperature"] == 0.7
        assert turn["response"]["choices"][0]["message"]["content"] == (rejected_text, valid_text)[index]
        training = turn["record"]["payload"]["response"]["training"]
        assert training["loss_mask"] == [1, 1]
        assert training["rollout_log_probs"] == [-0.1, -0.2]
    first_tokens = episode["turns"][0]["record"]["payload"]["response"]["training"]["tokens"]
    second_tokens = episode["turns"][1]["record"]["payload"]["response"]["training"]["tokens"]
    assert second_tokens[: len(first_tokens)] == first_tokens
    trajectory = Trajectory.model_validate(episode["trajectory"])
    agent_steps = [step for step in trajectory.steps if step.source == "agent"]
    assert len(agent_steps) == 2 and sum(step.llm_call_count for step in agent_steps) == 2
    assert sum(len(step.metrics.completion_token_ids) for step in agent_steps) == 4
    with pytest.raises(module.AnswerContractError, match=original_error):
        module.validate_final_module(rejected_code)


@pytest.mark.parametrize("capture", ["case {name}", "case [*{name}]", "case {{**{name}}}"])
@pytest.mark.parametrize("name", ["all", "len", "globals", "__private"])
def test_final_contract_rejects_forbidden_pattern_bindings(module, capture, name):
    source = "def answer():\n    return 0\n\nmatch None:\n    " + capture.format(name=name) + ":\n        pass\n"
    with pytest.raises(module.AnswerContractError):
        module.validate_final_module(source)


@pytest.mark.parametrize("capture", ["case value", "case [*values]", "case {**remaining}"])
def test_final_contract_preserves_ordinary_pattern_bindings(module, capture):
    source = "def answer():\n    match None:\n        " + capture + ":\n            return 0\n"
    module.validate_final_module(source)


def test_pattern_binding_cannot_turn_incorrect_answer_into_verifier_pass(module, tmp_path):
    contract = importlib.import_module(module.__package__ + ".answer_contract")
    verifier = importlib.import_module("recipes.agentcl.harbor.tests.verify")
    source = "def answer():\n    return 0\n\nmatch lambda values: True:\n    case all:\n        pass\n"
    assert contract.CONTRACT_VERSION == "agentcl-final-module-v2"
    with pytest.raises(module.AnswerContractError, match="Builtin names"):
        module.validate_final_module(source)
    task = tmp_path / "task.json"
    answer = tmp_path / "answer.py"
    task.write_text(json.dumps({"test_code": "assert all([answer() == 42])\n"}))
    answer.write_text(source)
    result = verifier.grade(task, answer, 2, isolate_user=False)
    assert result["reward"] == 0 and result["status"] == "invalid"
    answer.write_text("def answer():\n    return 0\n")
    assert verifier.grade(task, answer, 2, isolate_user=False)["status"] == "failed"


@pytest.mark.parametrize("primary_fault", [False, True])
@pytest.mark.parametrize("cleanup_error", [RuntimeError, OSError, ValueError, asyncio.CancelledError])
def test_episode_cleanup_fault_is_persisted_and_keeps_primary_error(
    module, tmp_path, monkeypatch, primary_fault, cleanup_error
):
    sandbox = fixture_sandbox(module)
    monkeypatch.setattr(sandbox, "close", AsyncMock(side_effect=cleanup_error("cleanup failed")))
    primary_error = module.EpisodeFault("submission failed")
    if primary_fault:
        monkeypatch.setattr(sandbox, "submit", AsyncMock(side_effect=primary_error))
    output = tmp_path / "host" / "episode.json"
    runner = module.EpisodeRunner(
        fixture_model(module),
        sandbox,
        module.EpisodeSettings("cleanup-fault", "fixture", "release-one"),
        tmp_path / "logs",
        output,
    )
    expected_error = module.EpisodeFault if primary_fault else cleanup_error
    with pytest.raises(expected_error) as captured:
        asyncio.run(runner.run("Problem only."))
    saved = json.loads(output.read_text())
    assert saved["outcome"] == "fault"
    assert saved["references"] == ["receipt-0", "receipt-1", "receipt-2"]
    assert saved["fault"] == ("submission failed" if primary_fault else "cleanup failed")
    assert json.loads((tmp_path / "logs" / "episode.json").read_text()) == saved
    if primary_fault:
        assert captured.value is primary_error
        assert primary_error.__notes__ == ["isolated Python kernel cleanup also failed: cleanup failed"]


def test_episode_cancellation_remains_the_primary_fault_when_cleanup_fails(module, tmp_path, monkeypatch):
    sandbox = fixture_sandbox(module)
    primary_error = asyncio.CancelledError("student cancelled")
    monkeypatch.setattr(sandbox, "execute", AsyncMock(side_effect=primary_error))
    monkeypatch.setattr(sandbox, "close", AsyncMock(side_effect=RuntimeError("cleanup failed")))
    runner = module.EpisodeRunner(
        fixture_model(module), sandbox, module.EpisodeSettings("cancelled", "fixture", "release-one"), tmp_path
    )
    with pytest.raises(asyncio.CancelledError) as captured:
        asyncio.run(runner.run("Problem only."))
    assert captured.value is primary_error
    saved = json.loads((tmp_path / "episode.json").read_text())
    assert saved["outcome"] == "fault" and saved["fault"] == "student cancelled"
    assert saved["references"] == ["receipt-0"]
    assert primary_error.__notes__ == ["isolated Python kernel cleanup also failed: cleanup failed"]


def harness_adapter(module):
    return importlib.import_module(module.__name__.rsplit(".", 1)[0] + ".agent")


def test_kernel_launch_survives_controlling_terminal_exit(module, tmp_path):
    adapter = harness_adapter(module)
    environment = SimpleNamespace(exec=AsyncMock(return_value=ExecResult(return_code=0, stdout="42")))
    sandbox = adapter.HarborSandbox(environment)
    asyncio.run(sandbox.start())
    assert environment.exec.await_count == 1
    assert environment.exec.await_args.kwargs == {"user": "student", "timeout_sec": 10}
    command = environment.exec.await_args.args[0]
    assert command.startswith(adapter.SAFE_COMMAND_PREFIX + "python3 -I -S -c ")
    launch_script = shlex.split(command)[-1]
    assert "deadline = time.monotonic() + 2.0" in launch_script
    assert "process.poll() is None" in launch_script
    assert "process.kill()" in launch_script and "process.wait()" in launch_script
    assert "test -S" not in command
    source = Path(importlib.import_module("recipes.agentcl.harbor.environment.kernel").__file__)
    socket_path = tmp_path / "kernel.sock"
    log_path = tmp_path / "kernel.log"
    kernel_path = tmp_path / "kernel.py"
    pid_path = tmp_path / "kernel.pid"
    receipt_path = tmp_path / "launch.json"
    kernel_path.write_text(
        source.read_text()
        .replace('"/tmp/agentcl-kernel.sock"', repr(str(socket_path)))
        .replace("def serve() -> None:\n", "def serve() -> None:\n    import time\n    time.sleep(0.2)\n")
    )
    launch_script = (
        launch_script.replace("/tmp/agentcl-kernel.sock", str(socket_path))
        .replace("/workspace/answer.py", str(tmp_path / "answer.py"))
        .replace("/tmp/agentcl-kernel.log", str(log_path))
        .replace("/opt/agentcl/kernel.py", str(kernel_path))
        .replace("cwd='/workspace'", "cwd=" + repr(str(tmp_path)))
        .replace(sandbox.launch_receipt_path, str(receipt_path))
        .replace(
            "print(process.pid, flush=True)",
            "pathlib.Path("
            + repr(str(pid_path))
            + ").write_text(str(process.pid))\n            print(process.pid, flush=True)",
        )
    )
    started = time.monotonic()
    terminal_pid, terminal_fd = pty.fork()
    if terminal_pid == 0:
        os.execvp("bash", ["bash", "-c", shlex.quote(sys.executable) + " -I -S -c " + shlex.quote(launch_script)])
    kernel_pid = None
    terminal_reaped = False
    try:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            finished, status = os.waitpid(terminal_pid, os.WNOHANG)
            if finished:
                terminal_reaped = True
                assert os.waitstatus_to_exitcode(status) == 0
                break
            time.sleep(0.01)
        assert terminal_reaped
        kernel_pid = int(pid_path.read_text())
        assert os.read(terminal_fd, 1024).decode().strip() == str(kernel_pid)
        os.close(terminal_fd)
        terminal_fd = None
        assert time.monotonic() - started >= 0.2
        assert socket_path.is_socket()
        receipt = json.loads(receipt_path.read_text())
        assert receipt == {
            "pid": kernel_pid,
            "socket": str(socket_path),
            "nonce": Path(sandbox.launch_receipt_path).stem[15:],
        }
        assert log_path.exists()
        assert os.getsid(kernel_pid) == kernel_pid
    finally:
        if kernel_pid is None and pid_path.exists():
            kernel_pid = int(pid_path.read_text())
        if kernel_pid is not None:
            with contextlib.suppress(ProcessLookupError):
                os.kill(kernel_pid, signal.SIGKILL)
        if terminal_fd is not None:
            os.close(terminal_fd)
        if not terminal_reaped:
            os.kill(terminal_pid, signal.SIGKILL)
            os.waitpid(terminal_pid, 0)


@pytest.mark.parametrize("stdout", [None, "", "not-a-pid", "0", "1", "-1"])
def test_kernel_launch_rejects_invalid_pid_without_attempting_cleanup(module, stdout):
    adapter = harness_adapter(module)
    environment = SimpleNamespace(exec=AsyncMock(return_value=ExecResult(return_code=0, stdout=stdout)))
    sandbox = adapter.HarborSandbox(environment)
    with pytest.raises(module.EpisodeFault, match="invalid process ID"):
        asyncio.run(sandbox.start())
    asyncio.run(sandbox.close())
    assert sandbox.kernel_pid is None
    assert environment.exec.await_count == 1


def test_kernel_launch_failure_does_not_attempt_socket_cleanup(module):
    adapter = harness_adapter(module)
    environment = SimpleNamespace(
        exec=AsyncMock(return_value=ExecResult(return_code=1, stdout="launch stdout", stderr="launch stderr"))
    )
    sandbox = adapter.HarborSandbox(environment)
    with pytest.raises(module.EpisodeFault, match=r"launch stdout.*launch stderr"):
        asyncio.run(sandbox.start())
    asyncio.run(sandbox.close())
    assert environment.exec.await_count == 1


@pytest.mark.parametrize("failure", ["early-exit", "deadline", "popen"])
def test_kernel_failed_local_launch_leaves_no_child_or_ready_receipt(module, tmp_path, failure):
    adapter = harness_adapter(module)
    environment = SimpleNamespace(exec=AsyncMock(return_value=ExecResult(return_code=0, stdout="42")))
    sandbox = adapter.HarborSandbox(environment)
    asyncio.run(sandbox.start())
    script = shlex.split(environment.exec.await_args.args[0])[-1]
    socket_path = tmp_path / "kernel.sock"
    receipt_path = tmp_path / "launch.json"
    child_path = tmp_path / "child.py"
    pid_path = tmp_path / "child.pid"
    child_path.write_text("import time\ntime.sleep(60)\n" if failure == "deadline" else "raise SystemExit(7)\n")
    child_command = "['/missing-agentcl-python']" if failure == "popen" else repr([sys.executable, str(child_path)])
    script = (
        script.replace("[sys.executable, '/opt/agentcl/kernel.py', '--serve']", child_command)
        .replace("/tmp/agentcl-kernel.sock", str(socket_path))
        .replace("/workspace/answer.py", str(tmp_path / "answer.py"))
        .replace("/tmp/agentcl-kernel.log", str(tmp_path / "kernel.log"))
        .replace(sandbox.launch_receipt_path, str(receipt_path))
        .replace("cwd='/workspace'", "cwd=" + repr(str(tmp_path)))
        .replace("ready = False", f"pathlib.Path({str(pid_path)!r}).write_text(str(process.pid))\nready = False")
    )
    started = time.monotonic()
    result = subprocess.run([sys.executable, "-I", "-S", "-c", script], capture_output=True, text=True, timeout=4)
    assert result.returncode != 0
    assert result.stdout == ""
    assert not socket_path.exists() and not receipt_path.exists()
    assert time.monotonic() - started < 3
    if failure == "popen":
        assert "FileNotFoundError" in result.stderr
        assert not pid_path.exists()
    else:
        expected = "did not become ready" if failure == "deadline" else "exited before readiness: 7"
        assert expected in result.stderr
        with pytest.raises(ProcessLookupError):
            os.kill(int(pid_path.read_text()), 0)


def test_kernel_launch_transport_timeout_is_not_retried_or_suppressed(module):
    adapter = harness_adapter(module)
    error = RuntimeError("command timed out after 10 seconds")
    environment = SimpleNamespace(exec=AsyncMock(side_effect=error), download_file=AsyncMock())
    sandbox = adapter.HarborSandbox(environment)
    with pytest.raises(RuntimeError) as captured:
        asyncio.run(sandbox.start())
    assert captured.value is error
    assert sandbox.kernel_pid is None
    assert sandbox.launch_receipt_path.startswith("/tmp/agentcl-launch-")
    assert environment.exec.await_count == 1
    environment.download_file.assert_not_awaited()


def test_kernel_launch_lost_exit_file_does_not_trust_a_potentially_stale_receipt(module):
    adapter = harness_adapter(module)
    error = "Error: open /var/lib/containers/storage/vfs-containers/container/userdata/exec/exit/container: no such file or directory"
    environment = SimpleNamespace(
        exec=AsyncMock(return_value=ExecResult(return_code=255, stdout="42", stderr=error)), download_file=AsyncMock()
    )
    sandbox = adapter.HarborSandbox(environment)
    with pytest.raises(module.EpisodeFault, match=r"exit=255.*stdout=42.*no such file"):
        asyncio.run(sandbox.start())
    assert sandbox.kernel_pid is None
    assert environment.exec.await_count == 1
    environment.download_file.assert_not_awaited()


@pytest.mark.parametrize("cleanup_failure", [ExecResult(return_code=1), RuntimeError("transport failure")])
def test_kernel_cleanup_failure_remains_visible_without_primary_fault(module, cleanup_failure):
    adapter = harness_adapter(module)
    environment = SimpleNamespace(
        exec=AsyncMock(side_effect=[ExecResult(return_code=0, stdout="42"), cleanup_failure])
    )
    sandbox = adapter.HarborSandbox(environment)
    asyncio.run(sandbox.start())
    with pytest.raises(RuntimeError):
        asyncio.run(sandbox.close())
    assert sandbox.kernel_pid == 42


def test_primary_submit_fault_is_retained_when_owned_cleanup_also_fails(module):
    adapter = harness_adapter(module)
    environment = SimpleNamespace(
        exec=AsyncMock(
            side_effect=[
                ExecResult(return_code=0, stdout="42"),
                ExecResult(return_code=1, stdout="primary write failure"),
                ExecResult(return_code=1, stdout="cleanup failure"),
            ]
        )
    )
    sandbox = adapter.HarborSandbox(environment)

    async def _run():
        try:
            await sandbox.start()
            await sandbox.submit("def answer(): return 42\n")
        finally:
            await sandbox.close()

    with pytest.raises(module.EpisodeFault, match="primary write failure") as captured:
        asyncio.run(_run())
    assert captured.value.__notes__ == ["isolated Python kernel cleanup also failed; Harbor owns container removal"]
    assert sandbox.kernel_pid == 42


def cleanup_receipt_from_command(command):
    script = shlex.split(command)[-1]
    call = ast.parse(script).body[-1].value
    receipt_path = ast.literal_eval(call.func.value.args[0])
    receipt = ast.literal_eval(call.args[0].args[0])
    return receipt_path, receipt


@pytest.mark.parametrize("failure", [None, "wrong", "missing"])
def test_cleanup_timeout_accepts_only_exact_independent_result_file(module, failure):
    adapter = harness_adapter(module)
    environment = SimpleNamespace(exec=AsyncMock(side_effect=RuntimeError("Command timed out after 5 seconds")))

    async def download(source, target):
        expected_path, receipt = cleanup_receipt_from_command(environment.exec.await_args.args[0])
        assert source == expected_path
        if failure == "missing":
            raise FileNotFoundError(source)
        if failure == "wrong":
            receipt["receipt_id"] = "stale"
        Path(target).write_text(json.dumps(receipt))

    environment.download_file = AsyncMock(side_effect=download)
    sandbox = adapter.HarborSandbox(environment)
    sandbox.kernel_pid = 42
    if failure:
        with pytest.raises(module.EpisodeFault):
            asyncio.run(sandbox.close())
        assert sandbox.kernel_pid == 42
    else:
        asyncio.run(sandbox.close())
        assert sandbox.kernel_pid is None
    assert environment.exec.await_count == 1


def test_lost_podman_exit_file_requires_exact_cleanup_result_file(module):
    adapter = harness_adapter(module)
    error = "Error: open /var/lib/containers/storage/vfs-containers/container/userdata/exec/exit/container: no such file or directory"
    environment = SimpleNamespace(exec=AsyncMock(return_value=ExecResult(return_code=255, stdout=error)))

    async def download(source, target):
        expected_path, receipt = cleanup_receipt_from_command(environment.exec.await_args.args[0])
        assert source == expected_path
        Path(target).write_text(json.dumps(receipt))

    environment.download_file = AsyncMock(side_effect=download)
    sandbox = adapter.HarborSandbox(environment)
    sandbox.kernel_pid = 42
    asyncio.run(sandbox.close())
    assert sandbox.kernel_pid is None
    assert environment.exec.await_count == 1
    assert environment.download_file.await_count == 1


@pytest.mark.parametrize("failure", ["missing", "wrong-receipt", "wrong-pid", "incomplete"])
def test_lost_exit_file_with_missing_or_invalid_result_still_fails(module, failure):
    adapter = harness_adapter(module)
    error = "Error: open /var/lib/containers/storage/vfs-containers/container/userdata/exec/exit/container: no such file or directory"
    environment = SimpleNamespace(exec=AsyncMock(return_value=ExecResult(return_code=255, stdout=error)))

    async def download(source, target):
        _, receipt = cleanup_receipt_from_command(environment.exec.await_args.args[0])
        if failure == "missing":
            raise FileNotFoundError(source)
        if failure == "wrong-receipt":
            receipt["receipt_id"] = "stale"
        elif failure == "wrong-pid":
            receipt["kernel_pid"] += 1
        else:
            receipt["socket_removed"] = False
        Path(target).write_text(json.dumps(receipt))

    environment.download_file = AsyncMock(side_effect=download)
    sandbox = adapter.HarborSandbox(environment)
    sandbox.kernel_pid = 42
    with pytest.raises(module.EpisodeFault):
        asyncio.run(sandbox.close())
    assert sandbox.kernel_pid == 42


def test_other_cleanup_errors_do_not_consult_result_file(module):
    adapter = harness_adapter(module)
    environment = SimpleNamespace(
        exec=AsyncMock(return_value=ExecResult(return_code=1, stdout="PermissionError")), download_file=AsyncMock()
    )
    sandbox = adapter.HarborSandbox(environment)
    sandbox.kernel_pid = 42
    with pytest.raises(module.EpisodeFault, match="PermissionError"):
        asyncio.run(sandbox.close())
    environment.download_file.assert_not_awaited()


def test_kernel_owned_pid_cleanup_is_idempotent(module):
    adapter = harness_adapter(module)
    environment = SimpleNamespace(
        exec=AsyncMock(side_effect=[ExecResult(return_code=0, stdout="42"), ExecResult(return_code=0)])
    )
    sandbox = adapter.HarborSandbox(environment)
    asyncio.run(sandbox.start())
    asyncio.run(sandbox.close())
    asyncio.run(sandbox.close())
    assert sandbox.kernel_pid is None
    assert environment.exec.await_count == 2


@pytest.mark.parametrize("answer", ["correct", "wrong", "missing"])
def test_submit_lost_exit_file_requires_exact_independent_answer_copy(module, answer):
    adapter = harness_adapter(module)
    code = "def answer():\n    return 'exact π'\n"
    error = "Error: open /var/lib/containers/storage/vfs-containers/container/userdata/exec/exit/container: no such file or directory"
    environment = SimpleNamespace(
        exec=AsyncMock(return_value=ExecResult(return_code=255, stdout="write stdout", stderr=error))
    )

    async def download(source, target):
        assert source == "/workspace/answer.py"
        if answer == "missing":
            raise FileNotFoundError(source)
        Path(target).write_bytes((code if answer == "correct" else code + "# wrong\n").encode())

    environment.download_file = AsyncMock(side_effect=download)
    sandbox = adapter.HarborSandbox(environment)
    if answer == "correct":
        asyncio.run(sandbox.submit(code))
    else:
        with pytest.raises(module.EpisodeFault, match=r"exit=255.*write stdout.*no such file"):
            asyncio.run(sandbox.submit(code))
    assert environment.exec.await_count == environment.download_file.await_count == 1
    assert environment.exec.await_args.kwargs == {"user": "student", "timeout_sec": 10}


@pytest.mark.parametrize(
    "result",
    [
        ExecResult(return_code=1, stdout="write stdout", stderr="PermissionError"),
        ExecResult(return_code=255, stdout="unrelated Podman error"),
        RuntimeError("write timeout"),
    ],
)
def test_submit_other_errors_remain_visible_without_copy_or_retry(module, result):
    adapter = harness_adapter(module)
    environment = SimpleNamespace(exec=AsyncMock(side_effect=[result]), download_file=AsyncMock())
    sandbox = adapter.HarborSandbox(environment)
    with pytest.raises(RuntimeError) as captured:
        asyncio.run(sandbox.submit("def answer(): return 42\n"))
    if isinstance(result, RuntimeError):
        assert captured.value is result
    else:
        assert result.stdout in str(captured.value)
        assert (result.stderr or "") in str(captured.value)
    assert environment.exec.await_count == 1
    environment.download_file.assert_not_awaited()


def test_submit_preserves_bounded_stdout_and_stderr(module):
    adapter = harness_adapter(module)
    environment = SimpleNamespace(
        exec=AsyncMock(
            return_value=ExecResult(return_code=1, stdout="discarded" + "x" * 2000, stderr="discarded" + "y" * 2000)
        )
    )
    with pytest.raises(module.EpisodeFault) as captured:
        asyncio.run(adapter.HarborSandbox(environment).submit("def answer(): return 42\n"))
    message = str(captured.value)
    assert "discarded" not in message
    assert "x" * 2000 in message and "y" * 2000 in message
    assert len(message) < 4200
