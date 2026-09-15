"""The openenv kind: the reply parsed, the package checked, the task written, the served log scored."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from recipes.beta.spade import (
    DesignerReplyError,
    GeneratedOpenEnvTask,
    OpenEnvReply,
    openenv_check,
    openenv_task,
    parse_openenv_reply,
)
from recipes.beta.spade.openenv import REPLAY_SCRIPT, app_text, openenv_models, reply_errors
from reef.core.tasks import read_harbor_task, write_harbor_task

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

MODELS = """from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class GuessAction(Action):
    guess: int = Field(..., description="A number from 1 to 3")


class GuessObservation(Observation):
    message: str = Field("", description="What the environment said")
"""
ENVIRONMENT = """import random
import uuid

from openenv.core.env_server import Environment
from openenv.core.env_server.types import State

from openenv_task.models import GuessAction, GuessObservation


class GuessEnvironment(Environment):
    def __init__(self):
        self._state = State(episode_id=str(uuid.uuid4()), step_count=0)
        self.target = 0

    def reset(self, seed=None, **kwargs):
        self.target = random.Random(seed).randint(1, 3)
        self._state = State(episode_id=str(uuid.uuid4()), step_count=0)
        return GuessObservation(message="Guess a number from 1 to 3.", reward=0.0, done=False)

    def step(self, action: GuessAction):
        self._state.step_count += 1
        if action.guess == self.target:
            return GuessObservation(message="Right.", reward=1.0, done=True)
        return GuessObservation(message="Wrong.", reward=0.0, done=False)

    @property
    def state(self):
        return self._state
"""
DOCUMENT = {
    "instruction": "A hidden number between 1 and 3 was drawn. Find it with as few guesses as you can; each guess tells you only whether it was right.",
    "models": MODELS,
    "environment": ENVIRONMENT,
    "action_example": {"guess": 2},
    "hint": "There are only three candidates; a wrong guess rules one out.",
}
REPLY_TEXT = "Here it is.\n\n```json\n" + json.dumps(DOCUMENT, indent=2) + "\n```\n"


def reply(**changes: object) -> OpenEnvReply:
    document = dict(DOCUMENT)
    document.update(changes)
    return parse_openenv_reply("```json\n" + json.dumps(document) + "\n```")


def generated(**overrides: object) -> GeneratedOpenEnvTask:
    fields: dict[str, object] = {
        "reply": reply(),
        "skill": "deduction",
        "generation": 12,
        "index": 3,
        "source_record_id": "rec-designer-12",
        "step": 48,
        "seed": 7,
    }
    fields.update(overrides)
    return GeneratedOpenEnvTask(**fields)  # type: ignore[arg-type]


# ----------------------------------------------------------------------------------------------- the reply


def test_the_reply_yields_the_goal_the_modules_the_example_action_and_the_hint() -> None:
    parsed = parse_openenv_reply(REPLY_TEXT)
    assert parsed.instruction == DOCUMENT["instruction"] + "\n"
    assert parsed.models == MODELS and parsed.environment == ENVIRONMENT
    assert parsed.action_example == {"guess": 2} and parsed.hint == DOCUMENT["hint"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: d.pop("models"), "models must be non-empty"),
        (lambda d: d.__setitem__("action_example", {}), "action_example must be a non-empty object"),
        (lambda d: d.__setitem__("action_example", [1]), "action_example must be a non-empty object"),
        (lambda d: d.__setitem__("extra", 1), "keys the task has no place for: extra"),
    ],
)
def test_an_unusable_openenv_reply_is_refused(change, message: str) -> None:
    document = json.loads(json.dumps(DOCUMENT))
    change(document)
    with pytest.raises(DesignerReplyError, match=message):
        parse_openenv_reply("```json\n" + json.dumps(document) + "\n```")


# ----------------------------------------------------------------------------------------------- the package


def test_a_usable_package_has_no_errors_and_names_its_classes() -> None:
    assert reply_errors(reply()) == []
    models = openenv_models(reply())
    assert (models.action, models.observation, models.environment) == (
        "GuessAction",
        "GuessObservation",
        "GuessEnvironment",
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"instruction": "Guess."}, "fewer than 80 characters"),
        ({"models": "x = (\n"}, "models is not valid Python"),
        (
            {"models": MODELS.replace("class GuessObservation(Observation):", "class GuessObservation:")},
            "exactly one Observation subclass, found 0",
        ),
        ({"models": MODELS + "\nclass OtherAction(Action):\n    pass\n"}, "exactly one Action subclass, found 2"),
        (
            {
                "environment": ENVIRONMENT.replace(
                    "    @property\n    def state(self):\n        return self._state\n", ""
                )
            },
            "lacks state",
        ),
        (
            {"environment": ENVIRONMENT.replace("class GuessEnvironment(Environment):", "class GuessEnvironment:")},
            "exactly one Environment subclass, found 0",
        ),
        ({"environment": "import os\n" + ENVIRONMENT}, "imports 'os', which the rules forbid"),
        (
            {"environment": "import numpy\n" + ENVIRONMENT},
            "imports 'numpy'; only the standard library, openenv and pydantic",
        ),
        (
            {"environment": ENVIRONMENT.replace("from openenv_task.models import", "from ..models import")},
            "relative import",
        ),
        (
            {"environment": ENVIRONMENT.replace("        self.target = 0\n", "        self.target = eval('0')\n")},
            "calls eval()",
        ),
    ],
)
def test_a_package_that_breaks_the_contract_is_refused(changes: dict[str, str], message: str) -> None:
    errors = reply_errors(reply(**changes))
    assert any(message in error for error in errors), errors
    with pytest.raises(ValueError, match="not a usable environment package"):
        openenv_task(generated(reply=reply(**changes)))


# ----------------------------------------------------------------------------------------------- the task


def test_the_task_serves_the_package_behind_a_root_only_command(tmp_path: Path) -> None:
    task = openenv_task(generated())
    assert task.name == "openenv-00012-003-deduction"
    root = write_harbor_task(task, tmp_path)
    assert sorted(p.name for p in (root / "environment").iterdir()) == [
        "Dockerfile",
        "app.py",
        "environment.py",
        "models.py",
        "serve",
        "serve.py",
    ]
    assert (root / "environment" / "models.py").read_text() == MODELS
    assert (root / "environment" / "environment.py").read_text() == ENVIRONMENT
    dockerfile = (root / "environment" / "Dockerfile").read_text()
    assert "pip install --no-cache-dir 'openenv>=0.4,<0.5'" in dockerfile and "curl" in dockerfile
    assert "useradd --create-home --shell /bin/bash agent" in dockerfile
    assert "chmod -R go-rwx /opt/env" in dockerfile
    assert "agent ALL=(root) NOPASSWD: /usr/local/bin/python3 /opt/env/serve.py" in dockerfile
    assert (
        root / "environment" / "serve"
    ).read_text() == "#!/bin/sh\nexec sudo -n /usr/local/bin/python3 /opt/env/serve.py\n"
    app = (root / "environment" / "app.py").read_text()
    assert "from openenv_task.models import GuessAction, GuessObservation" in app
    assert "class LoggedEnvironment(GuessEnvironment):" in app and "MAX_TURNS = 30" in app and "SEED = 7" in app
    assert 'create_app(LoggedEnvironment, GuessAction, GuessObservation, env_name="openenv_task")' in app
    instruction = (root / "instruction.md").read_text()
    assert instruction.startswith(DOCUMENT["instruction"])
    assert (
        "`serve`" in instruction and "localhost:8000/step" in instruction and '{"action": {"guess": 2}}' in instruction
    )
    assert "after 30 steps" in instruction and "code" in instruction
    assert (
        (root / "tests" / "test.sh")
        .read_text()
        .endswith("python3 -S /tests/replay.py /var/env/steps.jsonl /logs/verifier/reward.txt\n")
    )
    assert (root / "solution" / "hint.txt").read_text() == DOCUMENT["hint"] + "\n"
    document = tomllib.loads((root / "task.toml").read_text())
    assert document["agent"] == {"timeout_sec": 900, "user": "agent"}
    assert (
        document["metadata"]["kind"] == "openenv"
        and document["metadata"]["max_turns"] == 30
        and document["metadata"]["seed"] == 7
    )
    assert document["metadata"]["models"] == {
        "action": "GuessAction",
        "observation": "GuessObservation",
        "environment": "GuessEnvironment",
    }
    assert read_harbor_task(root) == task


def test_the_app_text_is_valid_python_for_any_class_names() -> None:
    import ast

    text = app_text(openenv_models(reply()), max_turns=5, seed=3)
    ast.parse(text)
    assert "MAX_TURNS = 5" in text and "SEED = 3" in text


# ----------------------------------------------------------------------------------------------- the verifier


def scored(tmp_path: Path, events: list[dict[str, object]]) -> float:
    log_path = tmp_path / "steps.jsonl"
    log_path.write_text("".join(json.dumps(event) + "\n" for event in events))
    script = tmp_path / "replay.py"
    script.write_text(REPLAY_SCRIPT)
    reward_path = tmp_path / "reward.txt"
    subprocess.run([sys.executable, "-S", str(script), str(log_path), str(reward_path)], check=True, timeout=30)
    return float(reward_path.read_text())


def test_the_verifier_scores_the_first_episode_that_ended(tmp_path: Path) -> None:
    win = [
        {"event": "reset"},
        {"event": "step", "reward": 0.0, "done": False},
        {"event": "step", "reward": 1.0, "done": True},
    ]
    assert scored(tmp_path, win) == 1.0
    truncated = [{"event": "reset"}, {"event": "step", "reward": 0.0, "done": False, "truncated": True}]
    assert scored(tmp_path, truncated) == 0.0
    later_reset = [*win, {"event": "reset"}, {"event": "step", "reward": 0.0, "done": True}]
    assert scored(tmp_path, later_reset) == 1.0
    unfinished = [{"event": "reset"}, {"event": "step", "reward": 0.5, "done": False}]
    assert scored(tmp_path, unfinished) == 0.0
    assert scored(tmp_path, []) == 0.0
    clipped = [{"event": "reset"}, {"event": "step", "reward": 7.0, "done": True}]
    assert scored(tmp_path, clipped) == 1.0


def test_the_verifier_survives_a_missing_or_torn_log(tmp_path: Path) -> None:
    script = tmp_path / "replay.py"
    script.write_text(REPLAY_SCRIPT)
    reward_path = tmp_path / "reward.txt"
    subprocess.run(
        [sys.executable, "-S", str(script), str(tmp_path / "missing.jsonl"), str(reward_path)], check=True, timeout=30
    )
    assert reward_path.read_text() == "0.0\n"
    (tmp_path / "torn.jsonl").write_text('{"event": "reset"}\n{"event": "st')
    subprocess.run(
        [sys.executable, "-S", str(script), str(tmp_path / "torn.jsonl"), str(reward_path)], check=True, timeout=30
    )
    assert reward_path.read_text() == "0.0\n"


# ----------------------------------------------------------------------------------------------- the check


FAKE_DOCKER = '''#!{python}
"""A stand in for docker: records the calls and answers build, run, exec and rm the way the check expects."""
import json
import sys
from pathlib import Path

calls_path = Path(__file__).with_suffix(".calls")
script = json.loads(Path(__file__).with_suffix(".json").read_text())
arguments = sys.argv[1:]
with calls_path.open("a") as handle:
    handle.write(json.dumps(arguments) + "\\n")
verb = arguments[0]
if verb == "build":
    if script.get("build_fails"):
        sys.stderr.write("Dockerfile parse error: unknown instruction: [database]\\n"); sys.exit(1)
    print("sha256:abc")
elif verb == "run":
    print("cid123")
elif verb == "exec":
    if "/usr/local/bin/serve" in arguments:
        if script.get("serve_fails"):
            sys.stderr.write("the environment server exited; see /var/env/server.log\\n"); sys.exit(1)
        print("the environment server is up")
    elif "cat" in arguments:
        print("ModuleNotFoundError: No module named 'openenv_task.models'")
    elif "/reset" in " ".join(arguments):
        print(json.dumps(script.get("reset", {{"observation": {{"message": "Guess a number from 1 to 3."}}, "reward": 0.0, "done": False}})))
    elif "/step" in " ".join(arguments):
        if script.get("step_fails"):
            sys.stderr.write("curl: (22) The requested URL returned error: 422\\n"); sys.exit(22)
        print(json.dumps(script.get("step", {{"observation": {{"message": "Wrong."}}, "reward": 0.0, "done": False}})))
elif verb == "rm":
    pass
'''


def fake_docker(tmp_path: Path, monkeypatch, **script: object) -> Path:
    directory = tmp_path / "bin"
    directory.mkdir(exist_ok=True)
    path = directory / "docker"
    path.write_text(FAKE_DOCKER.format(python=sys.executable))
    path.with_suffix(".json").write_text(json.dumps(script))
    path.chmod(0o755)
    monkeypatch.setenv("PATH", f"{directory}{os.pathsep}{os.environ['PATH']}")
    return path.with_suffix(".calls")


def test_the_check_builds_starts_resets_steps_and_removes_the_container(tmp_path: Path, monkeypatch) -> None:
    calls_path = fake_docker(tmp_path, monkeypatch)
    root = write_harbor_task(openenv_task(generated()), tmp_path / "tasks")
    result = openenv_check(root, action_example={"guess": 2})
    assert result.is_serving and result.reason == ""
    assert result.first_observation == '{"message": "Guess a number from 1 to 3."}'
    calls = [json.loads(line) for line in calls_path.read_text().splitlines()]
    assert calls[0][:3] == ["build", "-q", "-t"] and calls[0][-1] == str(root / "environment")
    assert calls[1][:5] == ["run", "-d", "--rm", "--network", "none"]
    assert calls[2] == ["exec", "cid123", "/usr/local/bin/serve"]
    assert calls[3][:4] == ["exec", "-u", "agent", "cid123"] and "localhost:8000/reset" in calls[3]
    assert "localhost:8000/step" in calls[4] and '{"action": {"guess": 2}}' in calls[4]
    assert calls[-1] == ["rm", "-f", "cid123"]


@pytest.mark.parametrize(
    ("script", "message"),
    [
        ({"build_fails": True}, "the image did not build: Dockerfile parse error"),
        ({"serve_fails": True}, "serve failed: the environment server exited"),
        ({"step_fails": True}, "the example action failed: curl: (22)"),
        ({"step": {"reward": 0.0}}, "step answered without an observation"),
    ],
)
def test_a_package_that_does_not_serve_is_refused_with_the_reason(
    tmp_path: Path, monkeypatch, script, message: str
) -> None:
    calls_path = fake_docker(tmp_path, monkeypatch, **script)
    root = write_harbor_task(openenv_task(generated()), tmp_path / "tasks")
    result = openenv_check(root, action_example={"guess": 2})
    assert not result.is_serving and message in result.reason
    calls = [json.loads(line) for line in calls_path.read_text().splitlines()]
    if not script.get("build_fails"):
        assert calls[-1] == ["rm", "-f", "cid123"]


def test_a_missing_docker_is_reported_not_raised(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    root = write_harbor_task(openenv_task(generated()), tmp_path / "tasks")
    result = openenv_check(root, action_example={"guess": 2})
    assert not result.is_serving and "docker is not installed" in result.reason
