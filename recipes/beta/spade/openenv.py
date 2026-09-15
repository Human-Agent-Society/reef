"""A generated environment of the openenv kind: an OpenEnv environment package as a Harbor task, and its checks.

The Designer's ``openenv`` kind is an OpenEnv environment (huggingface/OpenEnv): a ``models.py`` with one
``Action`` and one ``Observation`` model, and an ``Environment`` subclass with ``reset``, ``step`` and
``state``. The task keeps the package from the agent while letting it act: the container installs
``openenv``, the package lives under ``/opt/env`` readable by root only, a sudo rule lets the agent run
``serve``, which starts the environment's HTTP server as root on the loopback interface, and the agent
acts with ``curl`` against ``/reset``, ``/step``, ``/schema`` and ``/state``. The server wraps the
Designer's class so every reset and step lands in a root held log, the episode starts from the task's
seed whatever the agent sends, and the turn limit ends it; the verifier reads that log and writes the
episode return::

    openenv-00012-003-planning/
      task.toml                       metadata: kind, skill, generation, step, index, difficulty, document, max_turns, seed, models
      instruction.md                  the goal and how to act through the server
      environment/Dockerfile          python:3.12-slim plus openenv, curl, sudo, the agent user, the package under /opt/env
      environment/models.py           the Designer's models, unchanged
      environment/environment.py      the Designer's class, unchanged
      environment/app.py              the server: the class wrapped with the log and the turn limit
      environment/serve.py            starts the server as root, once
      environment/serve               the command on the agent's PATH: sudo, then serve.py
      tests/replay.py                 reads /var/env/steps.jsonl and writes the episode return
      tests/test.sh
      solution/hint.txt               the privileged hint (Harbor never mounts solution/ for the agent)

``openenv_check`` builds the image, starts the server in a container without network, and plays a reset
and one step with the Designer's example action, so a package that does not serve never becomes a task.
"""

from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass

from recipes.beta.spade.designer import FORBIDDEN_CALLS, FORBIDDEN_MODULES, OpenEnvReply
from recipes.beta.spade.tasks import AGENT_USER, SKILL_PATTERN
from reef.core.tasks import HarborTask

DEFAULT_IMAGE = "python:3.12-slim"
DEFAULT_MAX_TURNS = 30
OPENENV_REQUIREMENT = "openenv>=0.4,<0.5"
ENVIRONMENT_DIRECTORY = "/opt/env"
PACKAGE = "openenv_task"
LOG_PATH = "/var/env/steps.jsonl"
PORT = 8000
MIN_INSTRUCTION_CHARS = 80
ALLOWED_TOP_MODULES = frozenset({"openenv", "pydantic", PACKAGE})
# An OpenEnv state carries an episode id, so uuid is allowed here although the gym rules forbid it.
FORBIDDEN = FORBIDDEN_MODULES - {"uuid"}
CHECK_TIMEOUT_S = 600.0
SERVER_WAIT_S = 60.0
CONTAINER_NAME_PATTERN = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class GeneratedOpenEnvTask:
    """One OpenEnv environment as the Designer emitted it, with where it came from."""

    reply: OpenEnvReply
    skill: str
    generation: int
    index: int
    source_record_id: str
    step: int = 0
    difficulty: str | None = None
    document_id: str | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.reply, OpenEnvReply):
            raise ValueError("reply must be an OpenEnvReply")
        if not isinstance(self.skill, str) or not SKILL_PATTERN.fullmatch(self.skill):
            raise ValueError(f"skill {self.skill!r} must match {SKILL_PATTERN.pattern}")
        for label, number in (
            ("generation", self.generation),
            ("index", self.index),
            ("step", self.step),
            ("seed", self.seed),
        ):
            if isinstance(number, bool) or not isinstance(number, int) or number < 0:
                raise ValueError(f"{label} must be a non-negative integer")
        if not isinstance(self.source_record_id, str) or not self.source_record_id:
            raise ValueError("source_record_id must name the Designer's generation record")
        for label, text in (("difficulty", self.difficulty), ("document_id", self.document_id)):
            if text is not None and (not isinstance(text, str) or not text):
                raise ValueError(f"{label} must be a non-empty string when set")

    @property
    def name(self) -> str:
        """The task directory name: unique per generation and index, readable by skill."""
        return f"openenv-{self.generation:05d}-{self.index:03d}-{self.skill}"


@dataclass(frozen=True)
class OpenEnvModels:
    """The class names a reply's code defines: the action, the observation and the environment."""

    action: str
    observation: str
    environment: str


@dataclass(frozen=True)
class OpenEnvCheck:
    """Whether the package served in its container: a reset and one step answered; ``reason`` names the first break."""

    is_serving: bool
    reason: str
    first_observation: str = ""


def class_names(code: str, base: str) -> list[str]:
    """Top level classes of ``code`` whose bases name ``base``."""
    tree = ast.parse(code)
    names = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            bases = {
                b.id if isinstance(b, ast.Name) else b.attr if isinstance(b, ast.Attribute) else "" for b in node.bases
            }
            if base in bases:
                names.append(node.name)
    return names


def code_errors(code: str, label: str) -> list[str]:
    """Imports and calls the rules forbid, plus imports outside the standard library, openenv and pydantic."""
    errors: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"{label} is not valid Python: {exc}"]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module is not None and node.level == 0:
            modules = [node.module.split(".")[0]]
        else:
            modules = []
        for module in modules:
            if module in FORBIDDEN:
                errors.append(f"{label} imports {module!r}, which the rules forbid")
            elif module not in ALLOWED_TOP_MODULES and module not in sys.stdlib_module_names:
                errors.append(
                    f"{label} imports {module!r}; only the standard library, openenv and pydantic are available"
                )
        if isinstance(node, ast.ImportFrom) and node.level > 0:
            errors.append(f"{label} uses a relative import; import the models as {PACKAGE}.models")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            errors.append(f"{label} calls {node.func.id}(), which the rules forbid")
    return errors


def reply_errors(reply: OpenEnvReply) -> list[str]:
    """Why an openenv reply is not a usable environment package; empty when it is one."""
    errors: list[str] = []
    if len(reply.instruction.strip()) < MIN_INSTRUCTION_CHARS:
        errors.append(f"the instruction has fewer than {MIN_INSTRUCTION_CHARS} characters")
    errors.extend(code_errors(reply.models, "models"))
    errors.extend(code_errors(reply.environment, "environment"))
    if errors:
        return errors
    actions = class_names(reply.models, "Action")
    observations = class_names(reply.models, "Observation")
    environments = class_names(reply.environment, "Environment")
    if len(actions) != 1:
        errors.append(f"models must define exactly one Action subclass, found {len(actions)}")
    if len(observations) != 1:
        errors.append(f"models must define exactly one Observation subclass, found {len(observations)}")
    if len(environments) != 1:
        errors.append(f"environment must define exactly one Environment subclass, found {len(environments)}")
    if environments:
        tree = ast.parse(reply.environment)
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == environments[0]:
                methods = {item.name for item in node.body if isinstance(item, ast.FunctionDef)}
                missing = sorted({"reset", "step", "state"} - methods)
                if missing:
                    errors.append(f"the environment class lacks {', '.join(missing)}")
    if not isinstance(reply.action_example, dict) or not reply.action_example:
        errors.append("action_example must be a non-empty object")
    return errors


def openenv_models(reply: OpenEnvReply) -> OpenEnvModels:
    """The three class names of a reply that passed ``reply_errors``."""
    return OpenEnvModels(
        action=class_names(reply.models, "Action")[0],
        observation=class_names(reply.models, "Observation")[0],
        environment=class_names(reply.environment, "Environment")[0],
    )


def openenv_task(
    task: GeneratedOpenEnvTask, *, max_turns: int = DEFAULT_MAX_TURNS, image: str = DEFAULT_IMAGE
) -> HarborTask:
    """The Harbor task that holds ``task``: the served package, the verifier, the hint and the metadata."""
    if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns < 1:
        raise ValueError("max_turns must be a positive integer")
    errors = reply_errors(task.reply)
    if errors:
        raise ValueError("the reply is not a usable environment package: " + "; ".join(errors))
    models = openenv_models(task.reply)
    metadata: dict[str, object] = {
        "kind": "openenv",
        "skill": task.skill,
        "generation": task.generation,
        "step": task.step,
        "index": task.index,
        "max_turns": max_turns,
        "seed": task.seed,
        "models": {"action": models.action, "observation": models.observation, "environment": models.environment},
    }
    if task.difficulty is not None:
        metadata["difficulty"] = task.difficulty
    if task.document_id is not None:
        metadata["document"] = task.document_id
    return HarborTask(
        name=task.name,
        instruction=instruction_text(task.reply.instruction, task.reply.action_example, max_turns),
        tests={
            "test.sh": f"#!/bin/sh\nset -eu\nmkdir -p /logs/verifier\npython3 -S /tests/replay.py {LOG_PATH} /logs/verifier/reward.txt\n",
            "replay.py": REPLAY_SCRIPT,
        },
        environment={
            "Dockerfile": dockerfile_text(image),
            "models.py": task.reply.models,
            "environment.py": task.reply.environment,
            "app.py": app_text(models, max_turns=max_turns, seed=task.seed),
            "serve.py": SERVE_SCRIPT,
            "serve": SERVE_COMMAND,
        },
        config={
            "agent": {"timeout_sec": 900, "user": AGENT_USER},
            "verifier": {"timeout_sec": 120},
            "environment": {"cpus": 1, "memory_mb": 2048, "storage_mb": 2048, "gpus": 0, "network_mode": "no-network"},
        },
        metadata=metadata,
        solution={"hint.txt": task.reply.hint + "\n"},
        source_agent_record_ids=(task.source_record_id,),
    )


def instruction_text(goal: str, action_example: dict[str, object], max_turns: int) -> str:
    example = json.dumps({"action": action_example})
    return (
        f"{goal.strip()}\n"
        "\n"
        "The environment runs behind an HTTP server on this machine. Start it once with `serve`, then act with curl:\n"
        f"`curl -s localhost:{PORT}/schema` shows the action and observation shapes; "
        f"`curl -s -X POST localhost:{PORT}/reset` starts the episode and returns the first observation; "
        f"`curl -s -X POST localhost:{PORT}/step -H 'content-type: application/json' -d '{example}'` "
        "takes one action and returns the observation, the reward and whether the episode is done. "
        f"The episode ends when it is done or after {max_turns} steps; a reset after that changes nothing the "
        "verifier reads. You never see the environment's code; the verifier scores the last step of the first "
        "episode that ended.\n"
    )


def dockerfile_text(image: str) -> str:
    """The agent's image: openenv, curl, sudo, a non root agent user, and the package readable by root only."""
    return (
        f"FROM {image}\n"
        "RUN apt-get update && apt-get install -y --no-install-recommends sudo curl && rm -rf /var/lib/apt/lists/* \\\n"
        f" && pip install --no-cache-dir '{OPENENV_REQUIREMENT}' \\\n"
        f" && useradd --create-home --shell /bin/bash {AGENT_USER} \\\n"
        f" && mkdir -p {ENVIRONMENT_DIRECTORY}/{PACKAGE}/server /var/env /workspace \\\n"
        f" && chmod 700 {ENVIRONMENT_DIRECTORY} /var/env && chown {AGENT_USER}:{AGENT_USER} /workspace\n"
        f"COPY models.py {ENVIRONMENT_DIRECTORY}/{PACKAGE}/models.py\n"
        f"COPY environment.py {ENVIRONMENT_DIRECTORY}/{PACKAGE}/server/environment.py\n"
        f"COPY app.py {ENVIRONMENT_DIRECTORY}/{PACKAGE}/server/app.py\n"
        f"COPY serve.py {ENVIRONMENT_DIRECTORY}/serve.py\n"
        "COPY serve /usr/local/bin/serve\n"
        f"RUN touch {ENVIRONMENT_DIRECTORY}/{PACKAGE}/__init__.py {ENVIRONMENT_DIRECTORY}/{PACKAGE}/server/__init__.py \\\n"
        f" && chmod -R go-rwx {ENVIRONMENT_DIRECTORY} && chmod 755 /usr/local/bin/serve \\\n"
        f" && echo '{AGENT_USER} ALL=(root) NOPASSWD: /usr/local/bin/python3 {ENVIRONMENT_DIRECTORY}/serve.py' "
        "> /etc/sudoers.d/environment \\\n"
        " && chmod 440 /etc/sudoers.d/environment\n"
        "WORKDIR /workspace\n"
    )


SERVE_COMMAND = f"#!/bin/sh\nexec sudo -n /usr/local/bin/python3 {ENVIRONMENT_DIRECTORY}/serve.py\n"

#: Starts the server once as root, in the background, on the loopback interface; a second call reports it is up.
SERVE_SCRIPT = f'''"""serve: start the environment's HTTP server as root, once."""

import os
import subprocess
import sys
import time
import urllib.request

PID_PATH = "/var/env/server.pid"
URL = "http://127.0.0.1:{PORT}/health"


def is_up():
    try:
        with urllib.request.urlopen(URL, timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def main():
    if is_up():
        print("the environment server is up")
        return 0
    os.makedirs("/var/env", exist_ok=True)
    log = open("/var/env/server.log", "ab")
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "{PACKAGE}.server.app:app", "--host", "127.0.0.1", "--port", "{PORT}"],
        cwd="{ENVIRONMENT_DIRECTORY}",
        env={{**os.environ, "PYTHONPATH": "{ENVIRONMENT_DIRECTORY}", "ENABLE_WEB_INTERFACE": "false"}},
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=log,
        start_new_session=True,
    )
    with open(PID_PATH, "w", encoding="utf-8") as handle:
        handle.write(f"{{process.pid}}\\n")
    for _ in range(300):
        if is_up():
            print("the environment server is up")
            return 0
        if process.poll() is not None:
            print("the environment server exited; see /var/env/server.log", file=sys.stderr)
            return 1
        time.sleep(0.2)
    print("the environment server did not come up in 60 s", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
'''


def app_text(models: OpenEnvModels, *, max_turns: int, seed: int) -> str:
    """The server module: the Designer's class wrapped with the log, the fixed seed and the turn limit."""
    return f'''"""The environment served: one instance for the whole run, every reset and step logged, the seed fixed, the turn limit enforced."""

import inspect
import json
import threading
import time

from openenv.core.env_server import create_app

from {PACKAGE}.models import {models.action}, {models.observation}
from {PACKAGE}.server.environment import {models.environment}

LOG_PATH = "{LOG_PATH}"
MAX_TURNS = {max_turns}
SEED = {seed}


def log(event):
    event["time"] = time.time()
    with open(LOG_PATH, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\\n")


class LoggedEnvironment({models.environment}):
    """The Designer's environment with what the verifier needs around it.

    The request body reaches the Designer's code only as the action: the seed is the task's, and the
    other reset and step parameters of the OpenEnv server are dropped.
    """

    def __init__(self):
        super().__init__()
        self.lock = threading.Lock()
        self.turns = 0
        self.is_over = False
        self.last = None

    def reset(self, *args, **kwargs):
        with self.lock:
            if "seed" in inspect.signature(super().reset).parameters:
                observation = super().reset(seed=SEED)
            else:
                observation = super().reset()
            self.turns = 0
            self.is_over = False
            self.last = observation
            log({{"event": "reset", "seed": SEED}})
            return observation

    def step(self, action, *args, **kwargs):
        with self.lock:
            if self.is_over:
                log({{"event": "step_after_end"}})
                return self.last
            observation = super().step(action)
            self.turns += 1
            reward = float(observation.reward or 0.0)
            done = bool(observation.done)
            truncated = not done and self.turns >= MAX_TURNS
            if truncated:
                observation.done = True
            self.is_over = done or truncated
            self.last = observation
            log({{"event": "step", "turn": self.turns, "reward": reward, "done": done, "truncated": truncated}})
            return observation

    def close(self):
        return None


# The OpenEnv HTTP server builds an environment from its factory for every request and closes it after;
# the episode outlives requests, so the factory returns the one instance and close() above does nothing.
ENVIRONMENT = LoggedEnvironment()


def environment():
    return ENVIRONMENT


app = create_app(environment, {models.action}, {models.observation}, env_name="{PACKAGE}")
'''


#: The verifier: the first episode of the root held log, its last reward when it ended, else 0.
REPLAY_SCRIPT = '''"""Read the environment server's log and write the episode return of the first episode."""

import json
import math
import sys


def main(log_path, reward_path):
    value = 0.0
    try:
        with open(log_path, encoding="utf-8", errors="replace") as handle:
            events = [json.loads(line) for line in handle if line.strip()]
        started = False
        rewards = []
        done = False
        for event in events:
            kind = event.get("event")
            if kind == "reset" and not started:
                started = True
            elif kind == "step" and started:
                rewards.append(float(event.get("reward", 0.0)))
                if event.get("done") or event.get("truncated"):
                    done = bool(event.get("done"))
                    break
        if rewards and done and math.isfinite(rewards[-1]):
            value = max(-1.0, min(1.0, rewards[-1]))
    except (OSError, ValueError):
        value = 0.0
    with open(reward_path, "w", encoding="utf-8") as handle:
        handle.write(f"{value}\\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
'''


def docker_run(arguments: Sequence[str], *, timeout_s: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["docker", *arguments], capture_output=True, text=True, timeout=timeout_s, check=False)


def openenv_check(task_path, *, action_example: dict[str, object], timeout_s: float = CHECK_TIMEOUT_S) -> OpenEnvCheck:
    """Build the task's image, start the server in a container without network, reset and take one step."""
    if shutil.which("docker") is None:
        return OpenEnvCheck(is_serving=False, reason="docker is not installed")
    tag = "spade-openenv-" + CONTAINER_NAME_PATTERN.sub("-", str(task_path).lower()).strip("-")[-60:]
    built = docker_run(["build", "-q", "-t", tag, str(task_path / "environment")], timeout_s=timeout_s)
    if built.returncode != 0:
        return OpenEnvCheck(is_serving=False, reason=f"the image did not build: {built.stderr.strip()[-500:]}")
    started = docker_run(["run", "-d", "--rm", "--network", "none", tag, "sleep", "900"], timeout_s=60.0)
    if started.returncode != 0:
        return OpenEnvCheck(is_serving=False, reason=f"the container did not start: {started.stderr.strip()[-500:]}")
    container = started.stdout.strip()
    try:
        serve = docker_run(["exec", container, "/usr/local/bin/serve"], timeout_s=SERVER_WAIT_S + 30.0)
        if serve.returncode != 0:
            log = docker_run(["exec", container, "cat", "/var/env/server.log"], timeout_s=30.0)
            return OpenEnvCheck(is_serving=False, reason=f"serve failed: {(serve.stderr + log.stdout).strip()[-500:]}")
        reset = docker_run(
            ["exec", "-u", AGENT_USER, container, "curl", "-s", "-f", "-X", "POST", f"localhost:{PORT}/reset"],
            timeout_s=60.0,
        )
        if reset.returncode != 0 or not reset.stdout.strip():
            return OpenEnvCheck(is_serving=False, reason=f"reset failed: {reset.stderr.strip()[-300:] or 'no reply'}")
        try:
            first = json.loads(reset.stdout)
        except json.JSONDecodeError:
            return OpenEnvCheck(is_serving=False, reason=f"reset did not answer JSON: {reset.stdout[:200]!r}")
        body = json.dumps({"action": action_example})
        step = docker_run(
            [
                "exec",
                "-u",
                AGENT_USER,
                container,
                "curl",
                "-s",
                "-f",
                "-X",
                "POST",
                f"localhost:{PORT}/step",
                "-H",
                "content-type: application/json",
                "-d",
                body,
            ],
            timeout_s=60.0,
        )
        if step.returncode != 0 or not step.stdout.strip():
            return OpenEnvCheck(
                is_serving=False, reason=f"the example action failed: {step.stderr.strip()[-300:] or 'no reply'}"
            )
        try:
            answer = json.loads(step.stdout)
        except json.JSONDecodeError:
            return OpenEnvCheck(is_serving=False, reason=f"step did not answer JSON: {step.stdout[:200]!r}")
        if not isinstance(answer, dict) or "observation" not in answer:
            return OpenEnvCheck(
                is_serving=False, reason=f"step answered without an observation: {step.stdout[:200]!r}"
            )
        return OpenEnvCheck(
            is_serving=True, reason="", first_observation=json.dumps(first.get("observation", first))[:2000]
        )
    finally:
        docker_run(["rm", "-f", container], timeout_s=60.0)
        time.sleep(0)
