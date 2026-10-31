"""Slime's task jobs: rollouts on the component's own rollout group, synced to the workers' weights, then trained."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("ray", reason="requires the optional Ray runtime")

from tests.slime_backend.test_slime_component import Stack, batch_for, parameters
from tests.support.slime_component import finished

from reef.core.evaluation import RolloutScore, RolloutScores
from reef.core.job_data import JobData, TaskSample
from reef.runtime.executor.placement import GpuBundles
from reef.runtime.interfaces import TrainingMethod
from reef.runtime.jobs import TrainingJob
from reef.train.slime_backend import component as slime_component
from reef.train.slime_backend import rollout_group as rollout_group_module
from reef.train.slime_backend.component import ROLLOUTS_FILE
from reef.train.slime_backend.rollout_group import EngineChat, SlimeRolloutGroup
from reef.train.task_rollouts import TaskJobRollouts

ROLLOUT = {"rollouts_per_task": 2}


class FakeRolloutGroup(SlimeRolloutGroup):
    """A rollout group that records when the engines take the workers' weights, release them, and close."""

    def __init__(self, events: list[str]) -> None:
        super().__init__(SimpleNamespace(manager=None), colocated=True, router_url="http://router:3000")
        self.events = events

    def sync(self, actor_group) -> None:
        self.events.append("sync")

    def release(self) -> None:
        self.events.append("release")

    def close(self) -> None:
        self.events.append("close")


def task(root: Path, name: str) -> TaskSample:
    path = root / name
    path.mkdir(parents=True)
    return TaskSample(path, metadata={"task": {"name": name}})


def grouped() -> TaskJobRollouts:
    """Two rollouts of the task ``add``, one that solved it, as their samples."""
    samples = tuple(replace(sample, group_id="add") for sample in batch_for(0, rewards=(1.0, 0.0)).samples)
    return TaskJobRollouts(samples, RolloutScores((("add", (RolloutScore.scored(1.0), RolloutScore.scored(0.0))),)))


@pytest.fixture
def task_stack(tmp_path, monkeypatch):
    """A Slime component with a fake rollout group; ``roll_out`` records the chat it answers with, in ``events``."""
    events: list[str] = []
    chats: list[EngineChat] = []

    def fake_roll_out(tasks, settings, chat, **options):
        events.append("roll_out")
        chats.append(chat)
        if "failing" in options["work_dir"].parts:
            raise RuntimeError("the sandbox did not start")
        return grouped()

    monkeypatch.setattr(slime_component, "roll_out", fake_roll_out)
    stack = Stack(tmp_path, rollout_group=FakeRolloutGroup(events), hf_checkpoint=str(tmp_path / "model"))
    yield stack, events, chats
    stack.component.close()


@pytest.mark.unit
def test_a_task_job_rolls_out_on_the_synced_engines_releases_them_and_trains_the_rollouts(tmp_path, task_stack):
    stack, events, chats = task_stack
    job = TrainingJob(
        "job-0",
        "slime",
        stack.component.base_start,
        JobData("tasks", (task(tmp_path / "tasks", "add"),)),
        {**parameters(TrainingMethod("sft")), "rollout": ROLLOUT},
        scenario="s",
    )

    stack.component.submit(job)
    status = finished(stack.component, "job-0")

    assert status.state == "finished" and len(status.checkpoints) == 1
    # The engines serve the workers' weights only from the sync to the release, around the rollouts.
    assert events == ["sync", "roll_out", "release"]
    assert chats[0].runtime_load_id == "slime:base"
    assert status.metrics["rollout_rewards"] == {"add": [1.0, 0.0]}
    assert status.metrics["rollout_samples"] == 2
    # The rollouts' two samples train as the job's two optimizer steps.
    assert stack.processor.calls[0]["rollout_ids"] == [0, 1]
    # The rollouts are kept: a resumed job trains them without rolling out again.
    directory = stack.component.job_directory("job-0")
    assert json.loads((directory / ROLLOUTS_FILE).read_text())["scores"] == [{"task": "add", "scores": [1.0, 0.0]}]
    settings = slime_component.RolloutSettings.from_parameters(job.parameters)
    assert stack.component.task_rollouts(job, directory, settings) == grouped()
    assert events == ["sync", "roll_out", "release"]


@pytest.mark.unit
def test_failed_rollouts_still_release_the_engines(tmp_path, task_stack):
    stack, events, _ = task_stack
    job = TrainingJob(
        "job-0",
        "slime",
        stack.component.base_start,
        JobData("tasks", (task(tmp_path / "tasks", "add"),)),
        {**parameters(TrainingMethod("sft")), "rollout": ROLLOUT},
        scenario="s",
    )
    (tmp_path / "failing").mkdir()
    with pytest.raises(RuntimeError, match="sandbox did not start"):
        stack.component.task_rollouts(
            job, tmp_path / "failing" / "job", slime_component.RolloutSettings.from_parameters(job.parameters)
        )
    assert events == ["sync", "roll_out", "release"]


@pytest.mark.unit
def test_a_task_job_is_checked_at_submit(tmp_path, task_stack):
    stack, events, _ = task_stack
    add = task(tmp_path / "a", "add")
    method = parameters(TrainingMethod("sft"))
    for items, job_parameters, message in (
        ((add,), method, "needs the rollout parameter"),
        ((add, *batch_for(0).samples), {**method, "rollout": ROLLOUT}, "either trajectories or tasks"),
        ((add, task(tmp_path / "b", "add")), {**method, "rollout": ROLLOUT}, "distinct names"),
        ((add,), {**method, "rollout": {"rollouts_per_task": 1}}, "rollouts_per_task must be an int of at least 2"),
    ):
        with pytest.raises(ValueError, match=message):
            stack.component.submit(
                TrainingJob(
                    "j", "slime", stack.component.base_start, JobData("d", items), job_parameters, scenario="s"
                )
            )
    assert events == []


class FakeManager:
    """The rollout manager's Ray handle: each remote call records its name."""

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def __getattr__(self, name: str) -> Any:
        manager = self

        class Method:
            def remote(self, *args: Any) -> str:
                manager.events.append(name)
                return name

        return Method()


class ActorGroup:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def update_weights(self) -> None:
        self.events.append("update_weights")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("colocated", "synced", "released"),
    [
        # Colocated engines take back their weights' memory, then their KV cache, around Slime's weight transfer.
        (True, ["onload_weights", "update_weights", "onload_kv"], ["offload"]),
        # Engines on their own GPUs keep their memory.
        (False, ["update_weights"], []),
    ],
)
def test_the_rollout_group_holds_gpu_memory_only_between_sync_and_release(monkeypatch, colocated, synced, released):
    events: list[str] = []
    monkeypatch.setattr(rollout_group_module.ray, "get", lambda value: value)
    group = SlimeRolloutGroup(
        SimpleNamespace(manager=FakeManager(events)), colocated=colocated, router_url="http://router:3000"
    )
    group.sync(ActorGroup(events))
    assert events == synced
    events.clear()
    group.release()
    assert events == released


@pytest.mark.unit
def test_closing_the_rollout_group_detaches_the_driver_and_keeps_the_session() -> None:
    closes: list[bool] = []
    startup = SimpleNamespace(
        manager=None,
        placements={"actor": ("pg", [0, 1], [0, 1]), "critic": None, "rollout": ("pg", [0, 1], [0, 1])},
        close=lambda *, failed: closes.append(failed),
    )
    group = SlimeRolloutGroup(startup, colocated=True, router_url="http://router:3000")
    # The workers train on the session's GPUs.
    assert group.placements["actor"] == GpuBundles("pg", [0, 1], [0, 1]) and group.placements["critic"] is None
    group.close()
    # Slime's failure cleanup detaches the driver; its success cleanup would dispose the session.
    assert closes == [True]


class Chat(EngineChat):
    """An engine chat whose template renders three prompt tokens and whose router answers ``answer``."""

    def __init__(self, answer: tuple[int, Any]) -> None:
        super().__init__("http://router:3000", model_path="/models/m", timeout_s=10.0, runtime_load_id="slime:base")
        self.reply = answer
        self.bodies: list[Any] = []

    def render(self, messages, template_kwargs):
        return [1, 2, 3]

    def generate(self, body):
        self.bodies.append(body)
        return self.reply


GENERATED = {
    "text": "done",
    "meta_info": {"output_token_logprobs": [[-0.1, 7, None], [-0.2, 8, None]], "finish_reason": {"type": "length"}},
}
CHAT = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2, "temperature": 0.7, "seed": 3}


@pytest.mark.unit
def test_the_engine_chat_answers_with_the_capture_of_the_engines_tokens():
    chat = Chat((200, GENERATED))
    status, body = chat.answer("/v1/chat/completions", CHAT)
    assert status == 200
    assert chat.bodies == [
        {
            "input_ids": [1, 2, 3],
            "sampling_params": {"max_new_tokens": 2, "temperature": 0.7, "sampling_seed": 3},
            "return_logprob": True,
        }
    ]
    assert body["choices"][0] == {
        "index": 0,
        "message": {"role": "assistant", "content": "done"},
        "finish_reason": "length",
    }
    training = body["training"]
    assert (training["tokens"], training["loss_mask"], training["rollout_log_probs"]) == (
        [1, 2, 3, 7, 8],
        [1, 1],
        [-0.1, -0.2],
    )
    assert (training["prompt_length"], training["runtime_load_id"]) == (3, "slime:base")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("path", "payload", "answer", "status"),
    [
        ("/v1/messages", CHAT, (200, GENERATED), 404),
        ("/v1/chat/completions", ["not", "an", "object"], (200, GENERATED), 400),
        ("/v1/chat/completions", {**CHAT, "tools": [{"type": "function"}]}, (200, GENERATED), 400),
        ("/v1/chat/completions", {**CHAT, "n": 2}, (200, GENERATED), 400),
        ("/v1/chat/completions", {**CHAT, "max_tokens": 0}, (200, GENERATED), 400),
        # An engine refusal keeps its status, so an agent can shrink a request that overflows the context.
        ("/v1/chat/completions", CHAT, (400, "the prompt is too long"), 400),
        ("/v1/chat/completions", CHAT, (200, {"text": "done", "meta_info": {}}), 502),
        (
            "/v1/chat/completions",
            CHAT,
            (200, {"text": "x", "meta_info": {"output_token_logprobs": [[float("nan"), 7, None]]}}),
            502,
        ),
    ],
)
def test_the_engine_chat_answers_each_call_with_its_status(path, payload, answer, status):
    assert Chat(answer).answer(path, payload)[0] == status
