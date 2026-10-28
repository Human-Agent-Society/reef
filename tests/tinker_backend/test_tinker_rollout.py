"""The Tinker component's own rollouts: a task job's episodes sampled on Tinker, then trained."""

from __future__ import annotations

import asyncio
import json
import time
import urllib.request
from pathlib import Path

import pytest

from reef.core.evaluation import RolloutScore, RolloutScores
from reef.core.job_data import JobData, TaskSample, TrajectorySample
from reef.core.trajectories import trajectory_reward
from reef.harness.client.tasks import EpisodeRow, TaskLab
from reef.runtime.interfaces import TrainingMethod
from reef.runtime.jobs import TrainingJob
from reef.tinker.checkpoint import TinkerCheckpoint
from reef.tinker.sampling import SampleResult, TinkerSampler
from reef.train.tinker_backend import component as component_module
from reef.train.tinker_backend.client import TinkerClient
from reef.train.tinker_backend.component import ROLLOUTS_FILE, TinkerTrainingComponent
from reef.train.tinker_backend.config import TinkerConfig
from reef.train.tinker_backend.rollout import RolloutSettings, SamplerEndpoint, TaskRollouts, roll_out

MODEL = "Qwen/Qwen3-8B"
ROLLOUT = {"episodes_per_task": 2, "concurrency": 2}


class Sampler(TinkerSampler):
    """Renders every conversation to three prompt tokens and samples two tokens."""

    def __init__(self) -> None:
        self.paths: list[str | None] = []

    def render(self, messages, *, template_kwargs):
        return [1, 2, 3]

    def decode(self, tokens):
        return "done"

    def sample(self, sampler_path, prompt, params):
        self.paths.append(sampler_path)
        return SampleResult((7, 8), (-0.1, -0.2), "stop")

    def close(self):
        return


class AgentLab(TaskLab):
    """Calls the model once through the agent's api_base, then scores the episode with the task's reward."""

    def __init__(self, rewards: dict[str, float], failing: frozenset[str] = frozenset()) -> None:
        self.rewards = rewards
        self.failing = failing

    async def run(self, task_path, agent, *, key, tags, overrides) -> EpisodeRow:
        if task_path.name in self.failing:
            return EpisodeRow(rewards={}, error="the sandbox did not start")
        body = json.dumps({"model": "m", "messages": [{"role": "user", "content": f"solve {task_path.name}"}]})
        request = urllib.request.Request(
            f"{agent['kwargs']['api_base']}/chat/completions",
            data=body.encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        # A blocking agent: the sampler endpoint runs on its own threads.
        with urllib.request.urlopen(request, timeout=10) as response:
            response.read()
        return EpisodeRow(rewards={"reward": self.rewards[task_path.name]})


def harbor_task(root: Path, name: str) -> TaskSample:
    path = root / name
    path.mkdir(parents=True)
    (path / "task.toml").write_text('version = "1.0"\n')
    return TaskSample(path, metadata={"task": {"name": name}})


@pytest.mark.unit
def test_a_rollout_samples_each_task_on_the_start_sampler_and_groups_its_samples(tmp_path):
    tasks = (harbor_task(tmp_path, "add"), harbor_task(tmp_path, "sort"), harbor_task(tmp_path, "broken"))
    sampler = Sampler()

    rollouts = asyncio.run(
        roll_out(
            tasks,
            RolloutSettings.from_parameters({"rollout": ROLLOUT}),
            sampler,
            "tinker://start/sampler",
            model=MODEL,
            runtime_load_id="tinker:start",
            work_dir=tmp_path / "work",
            lab=AgentLab({"add": 1.0, "sort": 0.0}, failing=frozenset({"broken"})),
        )
    )

    assert sampler.paths == ["tinker://start/sampler"] * 4
    # Each task's two episodes form its group; the task whose sandbox failed has none.
    assert [sample.group_id for sample in rollouts.samples] == ["add", "add", "sort", "sort"]
    assert rollouts.rewards() == {"add": (1.0, 1.0), "sort": (0.0, 0.0), "broken": ()}
    # The sandbox of "broken" did not start: its rollouts failed, which is not the policy's failure.
    assert rollouts.scores.to_json() == [
        {"task": "add", "scores": [1.0, 1.0]},
        {"task": "sort", "scores": [0.0, 0.0]},
        {"task": "broken", "scores": ["failed", "failed"]},
    ]
    training = rollouts.samples[0].training
    assert (training["tokens"], training["rollout_log_probs"]) == ([1, 2, 3, 7, 8], [-0.1, -0.2])
    assert training["runtime_load_id"] == "tinker:start"
    assert trajectory_reward(rollouts.samples[0]) == 1.0


@pytest.mark.unit
def test_the_sampler_endpoint_refuses_a_request_that_tinker_cannot_serve():
    with SamplerEndpoint(Sampler(), None, model=MODEL, runtime_load_id="tinker:base") as endpoint:
        body = json.dumps({"messages": [{"role": "user", "content": "hi"}], "tools": [{"type": "function"}]})
        request = urllib.request.Request(f"{endpoint.url}/v1/chat/completions", data=body.encode(), method="POST")
        with pytest.raises(urllib.error.HTTPError) as refused:
            urllib.request.urlopen(request, timeout=10)
    assert refused.value.code == 400


@pytest.mark.unit
def test_the_sampler_endpoint_answers_a_bad_request_and_a_sampler_failure_with_their_status():
    class Failing(Sampler):
        def sample(self, sampler_path, prompt, params):
            raise ValueError("Tinker returned non-finite sampled log probabilities")

    def post(url: str, body: bytes) -> tuple[int, str]:
        request = urllib.request.Request(f"{url}/v1/chat/completions", data=body, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, response.read().decode()
        except urllib.error.HTTPError as error:
            return error.code, error.read().decode()

    chat = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()
    with SamplerEndpoint(Failing(), None, model=MODEL, runtime_load_id="tinker:base") as endpoint:
        assert post(endpoint.url, b"{not json")[0] == 400
        status, body = post(endpoint.url, chat)
    assert status == 502 and "non-finite" in body


@pytest.mark.unit
@pytest.mark.parametrize(
    ("rollout", "message"),
    [
        (None, "needs the rollout parameter"),
        ({"episodes_per_task": 1}, "episodes_per_task must be an int of at least 2"),
        ({"episodes_per_task": 2, "concurrency": 0}, "concurrency must be an int of at least 1"),
        ({"episodes_per_task": 2, "sandbox": "docker"}, r"takes no \['sandbox'\]"),
        ({}, "incomplete"),
    ],
)
def test_the_rollout_parameter_is_checked(rollout, message):
    with pytest.raises(ValueError, match=message):
        RolloutSettings.from_parameters({} if rollout is None else {"rollout": rollout})


class Client(TinkerClient):
    def __init__(self) -> None:
        self.trained: list[int] = []

    def initialize(self):
        return TinkerCheckpoint(MODEL, 32, "tinker://base/state", "tinker://base/sampler")

    def train(self, checkpoint, batches, loss, learning_rates, *, kl_coef, export_adapter):
        self.trained.append(sum(len(batch) for batch in batches))
        return TinkerCheckpoint(MODEL, 32, "tinker://update/state", "tinker://update/sampler"), {"loss": 0.5}

    def download(self, checkpoint, directory):
        raise AssertionError("no adapter export")

    def close(self):
        return


def finished(component, job_id):
    deadline = time.monotonic() + 10
    while (status := component.status(job_id)).state == "running":
        assert time.monotonic() < deadline
        time.sleep(0.01)
    return status


def grouped(tmp_path: Path) -> TaskRollouts:
    def sample(name: str, reward: float) -> TrajectorySample:
        training = {"tokens": [1, 2, 3, 7, 8], "loss_mask": [1, 1], "rollout_log_probs": [-0.1, -0.2]}
        trajectory = {
            "schema_version": "ATIF-v1.6",
            "agent": {"name": "a"},
            "steps": [{"step_id": 1}],
            "extra": {"reef": {"training": training, "reward": reward}},
        }
        return TrajectorySample(trajectory, group_id=name)

    scores = RolloutScores((("add", (RolloutScore.scored(1.0), RolloutScore.scored(0.0))),))
    return TaskRollouts((sample("add", 1.0), sample("add", 0.0)), scores)


@pytest.mark.unit
def test_a_task_job_of_two_tasks_with_one_name_is_refused(tmp_path):
    import reef.recipe.generated_tasks  # noqa: F401 - registers the task_group objective

    component = TinkerTrainingComponent(
        MODEL, TinkerConfig(state_dir=str(tmp_path / "state")), Client(), sampler=Sampler()
    )
    try:
        tasks = (harbor_task(tmp_path / "a", "add"), harbor_task(tmp_path / "b", "add"))
        parameters = {"method": TrainingMethod("task_group").to_dict(), "rollout": ROLLOUT, "batch_size": 2}
        with pytest.raises(ValueError, match="distinct names"):
            component.submit(TrainingJob("job-0", "tinker", None, JobData("d", tasks), parameters))
    finally:
        component.close()


@pytest.mark.unit
def test_a_task_job_trains_its_rollouts_and_a_resumed_job_trains_the_same_ones(tmp_path, monkeypatch):
    import reef.recipe.generated_tasks  # noqa: F401 - registers the task_group objective

    calls = []

    async def fake_roll_out(tasks, settings, sampler, sampler_path, **options):
        calls.append((tuple(task.task_path.name for task in tasks), sampler_path))
        return grouped(tmp_path)

    monkeypatch.setattr(component_module, "roll_out", fake_roll_out)
    client = Client()
    component = TinkerTrainingComponent(
        MODEL, TinkerConfig(state_dir=str(tmp_path / "state")), client, sampler=Sampler()
    )
    try:
        parameters = {"method": TrainingMethod("task_group").to_dict(), "rollout": ROLLOUT, "batch_size": 2}
        job = TrainingJob("job-0", "tinker", None, JobData("d", (harbor_task(tmp_path, "add"),)), parameters)
        component.submit(job)
        status = finished(component, "job-0")
        assert status.state == "finished" and len(status.checkpoints) == 1
        assert calls == [(("add",), "tinker://base/sampler")] and client.trained == [2]
        assert status.metrics["rollout_rewards"] == {"add": [1.0, 0.0]}
        # The rollouts are kept: running the job again trains them without new episodes.
        kept = json.loads((component.job_directory("job-0") / ROLLOUTS_FILE).read_text())
        assert kept["scores"] == [{"task": "add", "scores": [1.0, 0.0]}]
        settings = component.job_parameters(parameters).rollout
        assert component.task_rollouts(job, component.job_directory("job-0"), component._base, settings) == grouped(
            tmp_path
        )
        assert len(calls) == 1
    finally:
        component.close()


@pytest.mark.unit
def test_a_task_job_is_checked_at_submit(tmp_path):
    import reef.recipe.generated_tasks  # noqa: F401 - registers the task_group objective

    task = harbor_task(tmp_path, "add")
    method = {"method": TrainingMethod("task_group").to_dict()}
    trajectory = grouped(tmp_path).samples[0]
    without_sampler = TinkerTrainingComponent(MODEL, TinkerConfig(state_dir=str(tmp_path / "a")), Client())
    try:
        with pytest.raises(ValueError, match="no sampler"):
            without_sampler.submit(
                TrainingJob("j", "tinker", None, JobData("d", (task,)), {**method, "rollout": ROLLOUT})
            )
    finally:
        without_sampler.close()
    component = TinkerTrainingComponent(
        MODEL, TinkerConfig(state_dir=str(tmp_path / "b")), Client(), sampler=Sampler()
    )
    try:
        for items, parameters, message in (
            ((task,), method, "needs the rollout parameter"),
            ((task, trajectory), {**method, "rollout": ROLLOUT}, "either trajectories or tasks"),
            ((trajectory,), {**method, "rollout": ROLLOUT}, "applies only to a job of tasks"),
        ):
            with pytest.raises(ValueError, match=message):
                component.submit(TrainingJob("j", "tinker", None, JobData("d", items), parameters))
    finally:
        component.close()


@pytest.mark.unit
def test_an_evaluation_job_scores_its_start_and_trains_nothing(tmp_path, monkeypatch):
    import reef.recipe.generated_tasks  # noqa: F401 - registers the task_group objective

    async def fake_roll_out(tasks, settings, sampler, sampler_path, **options):
        return grouped(tmp_path)

    monkeypatch.setattr(component_module, "roll_out", fake_roll_out)
    client = Client()
    component = TinkerTrainingComponent(
        MODEL, TinkerConfig(state_dir=str(tmp_path / "state")), client, sampler=Sampler()
    )
    try:
        parameters = {
            "method": TrainingMethod("task_group").to_dict(),
            "rollout": ROLLOUT,
            "evaluation_of": "job-3",
        }
        task = harbor_task(tmp_path, "add")
        component.submit(TrainingJob("eval-0", "tinker", None, JobData("d", (task,)), parameters))
        status = finished(component, "eval-0")
        assert status.state == "finished" and status.checkpoints == () and client.trained == []
        assert status.metrics["evaluation_of"] == "job-3"
        assert status.metrics["evaluation_scores"] == [{"task": "add", "scores": [1.0, 0.0]}]
        assert status.metrics["evaluation_mean_score"] == 0.5
        trajectory = grouped(tmp_path).samples[0]
        with pytest.raises(ValueError, match="its data must be tasks"):
            component.submit(
                TrainingJob("j", "tinker", None, JobData("d", (trajectory,)), {**parameters, "rollout": None})
            )
    finally:
        component.close()
