"""A prompt adapter's agent plays a Harbor task directory in the task's container, scored by the task's verifier."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest

from reef.evaluation.harness.evaluator import container_binding
from reef.evaluation.harness.scoring import verifier_reward
from reef.evaluation.harness.tasks import EvaluationTask
from reef.harness.adapters import get_adapter
from reef.harness.tasks.harbor_task import CONTAINER_ROOT, can_run_in_task_container, run_harbor_task
from reef.harness.tasks.model_binding import ModelBinding
from reef.harness.tasks.rollouts import RolloutRow, TaskLab
from reef.harness.tasks.run import TaskRunError
from reef.harness.tasks.trajectory import failed_trial_error

SESSION_EVENT = {"type": "message", "message": {"role": "assistant", "content": [{"type": "text", "text": "391"}]}}


class StandInLab(TaskLab):
    """A trial that writes what ``AdapterAgent`` leaves in the agent's log directory, or nothing when ``ran`` is
    false, and keeps the agent spec it got."""

    def __init__(self, trials: Path, rewards: Mapping[str, float], *, ran: bool = True, error: str = "") -> None:
        self.trials = trials
        self.rewards = rewards
        self.ran = ran
        self.error = error
        self.agents: list[Mapping[str, object]] = []

    async def run(self, task_path, agent, *, key, tags, overrides) -> RolloutRow:
        self.agents.append(agent)
        trial = self.trials / key
        if self.ran:
            sessions = trial / "agent" / "sessions"
            sessions.mkdir(parents=True)
            (sessions / "session.jsonl").write_text(json.dumps(SESSION_EVENT) + "\n")
            (trial / "agent" / "command.json").write_text(json.dumps({"exit_code": 0, "stdout": "", "stderr": ""}))
        else:
            trial.mkdir(parents=True)
        return RolloutRow(rewards=dict(self.rewards), error=self.error, trial_uri=trial.as_uri())


def harbor_task(root: Path) -> Path:
    task = root / "multiply"
    (task / "environment").mkdir(parents=True)
    (task / "task.toml").write_text('version = "1.0"\n')
    (task / "instruction.md").write_text("Write 17 times 23 to /app/answer.txt.\n")
    (task / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n")
    return task


def test_the_agent_plays_the_task_in_its_container_and_the_verifier_reward_scores_it(tmp_path: Path) -> None:
    pi = get_adapter("pi")
    task = harbor_task(tmp_path)
    lab = StandInLab(tmp_path / "trials", {"reward": 1.0})
    result = run_harbor_task(pi, {"pi-agent/AGENTS.md": "Be exact.\n"}, str(task), keep_dir=tmp_path / "kept", lab=lab)

    assert result.exit_code == 0
    assert result.trajectory[0]["type"] == "verifier" and result.trajectory[1:] == (SESSION_EVENT,)
    assert verifier_reward(EvaluationTask.of_directory(task), result) == 1.0
    assert (tmp_path / "kept" / "session.jsonl").is_file()
    (agent,) = lab.agents
    kwargs = agent["kwargs"]
    assert agent["import_path"] == "reef.harness.tasks.harbor_task:AdapterAgent"
    assert (kwargs["package"], kwargs["version"]) == (pi.install.package, pi.install.version)
    assert kwargs["argv"] == list(pi.argv) and kwargs["sessions"] == pi.trajectory_path
    # The run's environment points the agent at the root in the container, where the tree lands.
    assert kwargs["env"]["PI_CODING_AGENT_DIR"] == f"{CONTAINER_ROOT}/pi-agent"
    assert not Path(kwargs["files_dir"]).exists()  # the staged tree goes with the run


def test_a_trial_that_never_ran_the_agent_is_a_failed_trial_with_its_error(tmp_path: Path) -> None:
    lab = StandInLab(tmp_path / "trials", {}, ran=False, error="the image did not build")
    result = run_harbor_task(get_adapter("pi"), {}, str(harbor_task(tmp_path)), lab=lab)

    assert result.exit_code == 1 and result.stderr == "the image did not build"
    assert failed_trial_error(result.trajectory) == "the image did not build"


def test_only_an_adapter_with_an_npm_install_and_a_text_prompt_plays_a_task_in_a_container(tmp_path: Path) -> None:
    assert all(can_run_in_task_container(get_adapter(name)) for name in ("claude", "pi", "opencode", "codex"))
    # An adapter whose own command takes the task directory runs it itself; hermes installs from git.
    assert not can_run_in_task_container(replace(get_adapter("pi"), takes_task_directory=True))
    assert not can_run_in_task_container(get_adapter("hermes"))
    with pytest.raises(TaskRunError, match="'hermes' cannot play a Harbor task directory"):
        run_harbor_task(get_adapter("hermes"), {}, str(harbor_task(tmp_path)))
    with pytest.raises(TaskRunError, match="is not a Harbor task directory"):
        run_harbor_task(get_adapter("pi"), {}, str(tmp_path / "missing"))


def test_a_container_reaches_a_loopback_model_route_at_the_agent_host() -> None:
    route = ModelBinding(base_url="http://127.0.0.1:8901/reef/scenarios/s/evaluation", model="m")
    assert container_binding(route, "host.docker.internal").base_url == (
        "http://host.docker.internal:8901/reef/scenarios/s/evaluation"
    )
    public = ModelBinding(base_url="https://api.example.com", model="m")
    assert container_binding(public, "host.docker.internal") == public
