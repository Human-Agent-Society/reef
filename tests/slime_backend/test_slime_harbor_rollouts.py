"""Harbor rollouts as the generate step of Slime's own loop, with fakes of Slime's ``Sample`` and of the engines.

These tests need no Ray, torch or Slime. ``test_slime_task_jobs`` checks the driver, and the end of this module checks
the fields against Slime's real ``Sample`` and ``BatchBuilder`` where Slime is installed.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import urllib.request
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from reef.harness.harbor.rollouts import RolloutRow, TaskLab
from reef.storage.durable_files import write_json
from reef.train.slime_backend import harbor_rollouts
from reef.train.slime_backend.dataset import DATASET_FILE
from reef.train.slime_backend.engine_chat import EngineChat
from reef.train.slime_backend.harbor_rollouts import (
    GENERATE_FUNCTION_PATH,
    HarborRolloutSession,
    SampleFields,
    TaskJobSource,
    TaskPrompts,
    TaskRolloutResult,
    apply_fields,
    check_rollout_can_train,
    check_task_job_options,
    generate_task_rollout,
    read_task_rollout_results,
    removed_fields,
    task_job_metrics,
)
from reef.train.task_rollouts import HarborSettings

DISTRIBUTION = {"temperature": 1.0, "top_p": 1.0, "top_k": -1}


class Chat(EngineChat):
    """An engine chat whose template renders ``len(messages)`` prompt tokens per message, with a fixed answer."""

    def __init__(self, answer: tuple[int, Any], *, topk_logprobs: int = 0, max_response_tokens: int | None = None):
        super().__init__(
            "http://router:3000",
            model_path="/models/m",
            timeout_s=10.0,
            runtime_load_id="slime:job",
            distribution=DISTRIBUTION,
            max_response_tokens=max_response_tokens,
            topk_logprobs=topk_logprobs,
        )
        self.reply = answer
        self.bodies: list[Any] = []

    def render(self, messages, template_kwargs):
        # A later call's prompt extends the earlier prompt and response: 1, 2, 3, 7, 8, then 9 for the new message.
        return [1, 2, 3] if len(messages) == 1 else [1, 2, 3, 7, 8, 9]

    def generate(self, body):
        self.bodies.append(body)
        return self.reply


GENERATED = {
    "text": "done",
    "meta_info": {
        "output_token_logprobs": [[-0.1, 7, None], [-0.2, 8, None]],
        "finish_reason": {"type": "stop"},
        "weight_version": "3",
    },
}
LIMITED = {**GENERATED, "meta_info": {**GENERATED["meta_info"], "finish_reason": {"type": "length"}}}
CHAT = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2, "temperature": 0.7, "seed": 3}


def post(agent: dict[str, Any], messages: list[dict[str, str]]) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{agent['kwargs']['api_base']}/chat/completions",
        data=json.dumps({"model": "m", "messages": messages, "temperature": 0.7}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


class AgentLab(TaskLab):
    """Calls the model ``turns`` times through the agent's api_base and gives the task's reward."""

    def __init__(self, *, reward: float = 1.0, error: str = "", turns: int = 1, build_error: str = "") -> None:
        self.reward = reward
        self.error = error
        self.turns = turns
        self.build_error = build_error
        self.builds: list[str] = []
        self.runs = 0

    async def run(self, task_path, agent, *, key, tags, overrides) -> RolloutRow:
        self.runs += 1
        if self.error:
            return RolloutRow(rewards={}, error=self.error)
        messages = [{"role": "user", "content": "first"}]
        for _ in range(self.turns):
            answer = post(agent, messages)
            messages += [answer["choices"][0]["message"], {"role": "user", "content": "next"}]
        return RolloutRow(rewards={"reward": self.reward})

    async def build_environment(self, task_path, *, overrides) -> str:
        self.builds.append(task_path.name)
        return self.build_error


def source_of(tmp_path: Path, **harbor: Any) -> TaskJobSource:
    task = tmp_path / "tasks" / "add"
    task.mkdir(parents=True, exist_ok=True)
    (task / "task.toml").write_text('version = "1.0"\n')
    directory = tmp_path / "job"
    directory.mkdir(exist_ok=True)
    return TaskJobSource("job", "token", (("add", str(task)),), HarborSettings(**harbor), str(directory), 10.0, 0)


def run_one(session: HarborRolloutSession, source: TaskJobSource, group_index: int = 0) -> SampleFields:
    ((name, path),) = source.tasks
    return asyncio.run(session.run(name, Path(path), group_index))


@pytest.fixture
def sessions():
    """Makes sessions over ``Chat``, and closes them."""
    made: list[HarborRolloutSession] = []

    def make(source: TaskJobSource, lab: TaskLab, *, answer=(200, GENERATED), top_k: int = 0) -> HarborRolloutSession:
        chat = Chat(answer, topk_logprobs=top_k)
        session = HarborRolloutSession(source, chat=chat, model="/models/m", top_k=top_k, lab=lab)
        made.append(session)
        return session

    yield make
    for session in made:
        session.close()


class FakeSample:
    """The fields of Slime's ``Sample`` that a generate function reads and writes."""

    class Status(Enum):
        PENDING = "pending"
        COMPLETED = "completed"
        TRUNCATED = "truncated"
        ABORTED = "aborted"

    def __init__(self, *, index: int = 5, group_index: int = 2, metadata: dict[str, Any] | None = None) -> None:
        self.index = index
        self.group_index = group_index
        self.metadata = metadata or {}
        self.status = FakeSample.Status.PENDING
        self.rollout_topk_token_ids = None
        self.rollout_topk_log_probs = None
        self.remove_sample = False


@pytest.mark.unit
def test_one_call_becomes_one_sample_with_the_verifiers_reward(tmp_path, sessions) -> None:
    source = source_of(tmp_path)
    fields = run_one(sessions(source, AgentLab(reward=0.5)), source)
    sample = FakeSample()
    apply_fields(sample, fields)
    assert sample.tokens == [1, 2, 3, 7, 8]
    assert (sample.response_length, sample.loss_mask, sample.rollout_log_probs) == (2, [1, 1], [-0.1, -0.2])
    assert (sample.reward, sample.status, sample.remove_sample) == (0.5, FakeSample.Status.COMPLETED, False)
    assert (sample.weight_versions, sample.response, sample.group_index) == (["3"], "done", 2)
    assert read_task_rollout_results(Path(source.job_directory)) == (
        TaskRolloutResult("token", "add", 0, 0.5, True, ""),
    )


@pytest.mark.unit
def test_several_calls_become_one_sample_whose_later_prompt_tokens_are_masked(tmp_path, sessions) -> None:
    source = source_of(tmp_path)
    fields = run_one(sessions(source, AgentLab(turns=2)), source)
    # The first prompt is the prompt. The second call's new prompt token, 9, is a response token with mask 0.
    assert fields.tokens == (1, 2, 3, 7, 8, 9, 7, 8)
    assert fields.response_length == 5
    assert fields.loss_mask == (1, 1, 0, 1, 1)
    assert fields.rollout_log_probs == (-0.1, -0.2, 0.0, -0.1, -0.2)
    assert fields.weight_versions == ("3", "3")


@pytest.mark.unit
def test_a_last_call_at_its_token_limit_truncates_the_sample(tmp_path, sessions) -> None:
    source = source_of(tmp_path)
    assert run_one(sessions(source, AgentLab(), answer=(200, LIMITED)), source).status == "truncated"


@pytest.mark.unit
def test_score_centering_gets_a_head_per_response_token_with_a_uniform_head_where_masked(tmp_path, sessions) -> None:
    heads = [[[-0.1, 7, None], [-2.5, 9, None]], [[-0.2, 8, None], [-1.9, 7, None]]]
    answer = {"text": "done", "meta_info": {**GENERATED["meta_info"], "output_top_logprobs": heads}}
    source = source_of(tmp_path)
    fields = run_one(sessions(source, AgentLab(turns=2), answer=(200, answer), top_k=2), source)
    assert fields.topk_token_ids == ((7, 9), (8, 7), (0, 1), (7, 9), (8, 7))
    assert fields.topk_log_probs[2] == pytest.approx((-0.6931471, -0.6931471))
    assert fields.topk_log_probs[0] == (-0.1, -2.5)
    # A removed sample has a head too: Slime validates the heads of every sample.
    removed = removed_fields("failed", 2)
    assert removed.topk_token_ids == ((0, 1),) and len(removed.topk_log_probs) == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    ("lab", "answer", "reward", "error"),
    [
        (AgentLab(error="the sandbox did not start"), (200, GENERATED), None, "the sandbox did not start"),
        # The verifier scored the rollout, but the agent made no call: the sample cannot train, and the reward reports.
        (AgentLab(reward=1.0, turns=0), (200, GENERATED), 1.0, "no model call"),
        # An agent that gives up when the engines fail.
        (AgentLab(), (500, "no engine"), None, "HTTPError 500"),
    ],
)
def test_a_failed_rollout_is_a_removed_sample_outside_its_prompt_group(
    tmp_path, sessions, lab, answer, reward, error
) -> None:
    source = source_of(tmp_path)
    fields = run_one(sessions(source, lab, answer=answer), source)
    sample = FakeSample(index=5, group_index=2)
    apply_fields(sample, fields)
    assert sample.remove_sample and sample.status == FakeSample.Status.ABORTED
    assert (sample.tokens, sample.response_length, sample.loss_mask, sample.reward) == ([0, 0], 1, [0], 0.0)
    # Slime keys a group's mean and deviation by group_index: this index of its own leaves the group's statistics.
    assert sample.group_index == -6
    assert error in sample.metadata["abort_reason"]
    (result,) = read_task_rollout_results(Path(source.job_directory))
    assert (result.reward, result.assembled) == (reward, False)


class RaisingLab(AgentLab):
    async def run(self, task_path, agent, *, key, tags, overrides) -> RolloutRow:
        raise RuntimeError("the Modal client failed")


@pytest.mark.unit
def test_a_rollout_that_raises_stops_only_itself(tmp_path, sessions) -> None:
    source = source_of(tmp_path)
    fields = run_one(sessions(source, RaisingLab()), source)
    assert fields.remove_sample and "the Modal client failed" in fields.error


class ForkLab(AgentLab):
    """The second call's prompt rewrites the first prompt, so the calls are no one linear rollout."""

    async def run(self, task_path, agent, *, key, tags, overrides) -> RolloutRow:
        post(agent, [{"role": "user", "content": "first"}])
        post(agent, [{"role": "system", "content": "s"}, {"role": "user", "content": "other"}])
        return RolloutRow(rewards={"reward": 1.0})


class ForkChat(Chat):
    def render(self, messages, template_kwargs):
        return [1, 2, 3] if len(messages) == 1 else [40, 41, 42]


@pytest.mark.unit
def test_calls_that_do_not_assemble_are_a_removed_sample(tmp_path) -> None:
    source = source_of(tmp_path, scaffold_tolerance=0)
    session = HarborRolloutSession(source, chat=ForkChat((200, GENERATED)), model="/models/m", top_k=0, lab=ForkLab())
    try:
        fields = run_one(session, source)
    finally:
        session.close()
    assert fields.remove_sample and "do not assemble" in fields.error
    (result,) = read_task_rollout_results(Path(source.job_directory))
    assert (result.reward, result.assembled) == (1.0, False)


class BlockingLab(AgentLab):
    """An agent that blocks its event loop while it runs, and records how many rollouts ran at once."""

    def __init__(self) -> None:
        super().__init__()
        self.lock = threading.Lock()
        self.running = 0
        self.most = 0

    async def run(self, task_path, agent, *, key, tags, overrides) -> RolloutRow:
        with self.lock:
            self.running += 1
            self.most = max(self.most, self.running)
        time.sleep(0.1)
        with self.lock:
            self.running -= 1
        return await super().run(task_path, agent, key=key, tags=tags, overrides=overrides)


@pytest.mark.unit
def test_a_blocking_agent_leaves_slimes_loop_free_and_concurrency_bounds_the_rollouts(tmp_path, sessions) -> None:
    source = source_of(tmp_path, concurrency=2)
    lab = BlockingLab()
    session = sessions(source, lab)
    ((name, path),) = source.tasks

    async def scenario() -> int:
        beats = 0

        async def heartbeat() -> None:
            nonlocal beats
            while True:
                beats += 1
                await asyncio.sleep(0.01)

        beating = asyncio.create_task(heartbeat())
        await asyncio.gather(*(session.run(name, Path(path), index) for index in range(5)))
        beating.cancel()
        return beats

    # Five rollouts of 0.1 s, two at a time, take at least 0.3 s: the loop beats throughout.
    assert asyncio.run(scenario()) > 10
    assert lab.most == 2 and lab.runs == 5


class HangingLab(AgentLab):
    """A rollout that waits until it is cancelled, and then needs time to clean up."""

    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.cleaned = threading.Event()

    async def run(self, task_path, agent, *, key, tags, overrides) -> RolloutRow:
        self.started.set()
        try:
            await asyncio.sleep(60)
        finally:
            await asyncio.sleep(0.05)
            self.cleaned.set()
        return RolloutRow(rewards={"reward": 1.0})


@pytest.mark.unit
def test_cancelling_a_rollout_cancels_its_own_loop_and_waits_for_its_cleanup(tmp_path, sessions) -> None:
    source = source_of(tmp_path)
    lab = HangingLab()
    session = sessions(source, lab)
    ((name, path),) = source.tasks

    async def scenario() -> None:
        running = asyncio.create_task(session.run(name, Path(path), 0))
        await asyncio.to_thread(lab.started.wait, 5)
        # Slime's abort cancels the task of a request-abortable generate function.
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running

    asyncio.run(scenario())
    assert lab.cleaned.is_set()
    # An aborted rollout is no result: Slime drops its group.
    assert read_task_rollout_results(Path(source.job_directory)) == ()


@pytest.mark.unit
def test_slime_cancels_this_generate_function_on_abort() -> None:
    assert generate_task_rollout.abort_mode == "request"
    assert GENERATE_FUNCTION_PATH.rpartition(".") == (harbor_rollouts.__name__, ".", "generate_task_rollout")


@pytest.mark.unit
def test_a_remote_environment_builds_once_for_the_tasks_rollouts(tmp_path, sessions) -> None:
    source = source_of(tmp_path, environment="modal", concurrency=2)
    lab = AgentLab()
    session = sessions(source, lab)
    ((name, path),) = source.tasks

    async def scenario() -> None:
        await asyncio.gather(*(session.run(name, Path(path), index) for index in range(3)))

    asyncio.run(scenario())
    assert lab.builds == ["add"] and lab.runs == 3


@pytest.mark.unit
def test_a_failed_build_removes_the_tasks_samples_after_a_second_try(tmp_path, sessions) -> None:
    source = source_of(tmp_path, environment="e2b")
    lab = AgentLab(build_error="no template")
    fields = run_one(sessions(source, lab), source)
    assert fields.remove_sample and "did not start: no template" in fields.error
    assert lab.builds == ["add", "add"] and lab.runs == 0


class Args(SimpleNamespace):
    """Slime's options that a task job reads, with the values that work."""

    def __init__(self, **overrides: Any) -> None:
        values = {
            "loss_family": "slime",
            "advantage_estimator": "grpo",
            "rewards_normalization": True,
            "n_samples_per_prompt": 2,
            "rollout_batch_size": 4,
            "over_sampling_batch_size": 4,
            "group_rm": False,
            "reward_key": None,
            "custom_generate_function_path": None,
            "data_source_path": "reef.train.slime_backend.dataset_source.ReefDatasetSource",
            "rollout_function_path": "slime.rollout.sglang_rollout.generate_rollout",
            "partial_rollout": False,
            "dynamic_sampling_filter_path": None,
            "rollout_top_p": 1.0,
            "use_rollout_routing_replay": False,
            "use_score_centering": False,
            "score_centering_top_k": 2,
        }
        super().__init__(**{**values, **overrides})


@pytest.mark.unit
def test_the_options_of_a_task_job_that_work_pass() -> None:
    check_task_job_options(Args())
    check_task_job_options(Args(over_sampling_batch_size=None, data_source_path=None))
    # Without a group baseline, one rollout per task can train.
    check_task_job_options(Args(n_samples_per_prompt=1, advantage_estimator="ppo"))
    check_task_job_options(Args(n_samples_per_prompt=1, rewards_normalization=False))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"loss_family": "sft"}, "'slime' loss family"),
        ({"group_rm": True}, "--group-rm"),
        ({"reward_key": "score"}, "--reward-key"),
        ({"custom_generate_function_path": "my.generate"}, "--custom-generate-function-path"),
        ({"data_source_path": "my.Source"}, "--data-source-path"),
        ({"rollout_function_path": "my.rollout"}, "--rollout-function-path"),
        ({"partial_rollout": True}, "--partial-rollout"),
        ({"over_sampling_batch_size": 8}, "--over-sampling-batch-size"),
        ({"dynamic_sampling_filter_path": "my.filter"}, "--dynamic-sampling-filter-path"),
        ({"rollout_top_p": 0.9}, "--rollout-top-p"),
        ({"use_rollout_routing_replay": True}, "--use-rollout-routing-replay"),
        ({"n_samples_per_prompt": 1}, "--n-samples-per-prompt of at least 2"),
    ],
)
def test_an_option_that_cannot_work_is_refused(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        check_task_job_options(Args(**overrides))


def result(group_index: int, *, assembled: bool = True) -> TaskRolloutResult:
    return TaskRolloutResult("t", "add", group_index, 1.0, assembled, "" if assembled else "failed")


@pytest.mark.unit
def test_a_rollout_trains_only_when_a_group_has_enough_samples() -> None:
    check_rollout_can_train([result(0), result(0), result(1, assembled=False)], group_size=2)
    check_rollout_can_train([result(0)], group_size=1)
    for results in ([result(0), result(1)], [result(0, assembled=False)], []):
        with pytest.raises(RuntimeError, match="no prompt group has 2 samples"):
            check_rollout_can_train(results, group_size=2)


@pytest.mark.unit
def test_the_jobs_metrics_report_each_tasks_rewards_and_the_removed_rollouts() -> None:
    results = [
        TaskRolloutResult("t", "add", 0, 1.0, True, ""),
        TaskRolloutResult("t", "add", 0, 0.5, False, "no assembly"),
        TaskRolloutResult("t", "add", 0, None, False, "no sandbox"),
    ]
    assert task_job_metrics(["add", "sort"], results) == {
        "rollout_rewards": {"add": [1.0, 0.5], "sort": []},
        "rollout_samples": 1,
        "removed_rollouts": 2,
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    ("result_fields", "message"),
    [
        (("t", "add", 0, None, True, ""), "has a reward"),
        (("t", "add", 0, 1.0, True, "failed"), "has a reward"),
        (("t", "add", 0, 1.0, False, ""), "needs its error"),
    ],
)
def test_a_result_that_contradicts_itself_is_refused(result_fields: tuple[Any, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        TaskRolloutResult(*result_fields)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"response_length": 0, "loss_mask": (), "rollout_log_probs": ()}, "at least one response token"),
        ({"tokens": (7, 8)}, "at least one response token"),
        ({"loss_mask": (1,)}, "one mask entry"),
        ({"rollout_log_probs": (-0.1,)}, "one mask entry"),
        ({"status": "pending"}, "terminal statuses"),
        ({"remove_sample": True}, "if and only if"),
        ({"status": "aborted", "remove_sample": True}, "no trained token"),
        ({"topk_token_ids": ((1,), (2,))}, "both sampler heads"),
    ],
)
def test_sample_fields_that_slime_could_not_train_are_refused(overrides: dict[str, Any], message: str) -> None:
    fields: dict[str, Any] = {
        "tokens": (1, 7, 8),
        "response_length": 2,
        "loss_mask": (1, 1),
        "rollout_log_probs": (-0.1, -0.2),
        "reward": 1.0,
        "status": "completed",
        "remove_sample": False,
    }
    with pytest.raises(ValueError, match=message):
        SampleFields(**{**fields, **overrides})


@pytest.mark.unit
def test_the_job_record_reads_back(tmp_path) -> None:
    source = source_of(
        tmp_path, environment="modal", concurrency=3, agent={"name": "pi"}, verifier_timeout_reward=None
    )
    assert TaskJobSource.from_record(json.loads(json.dumps(source.to_record()))) == source
    with pytest.raises(ValueError, match="distinct names"):
        TaskJobSource("j", "t", (("a", "/a"), ("a", "/b")), HarborSettings(), "/job", 1.0, 0)
    with pytest.raises(ValueError, match="absolute"):
        TaskJobSource("j", "t", (("a", "a"),), HarborSettings(), "/job", 1.0, 0)


@pytest.mark.unit
def test_task_prompts_shuffle_by_seed_and_epoch_only() -> None:
    prompts = TaskPrompts(list("abcdefgh"), seed=7)
    assert prompts.samples == list("abcdefgh") and len(prompts) == 8
    prompts.shuffle(3)
    third = list(prompts.samples)
    prompts.shuffle(0)
    other = TaskPrompts(list("abcdefgh"), seed=7)
    other.shuffle(3)
    # A resumed job shuffles to its epoch directly and sees the same order.
    assert other.samples == third and prompts.samples != third
    assert sorted(third) == list("abcdefgh")
    with pytest.raises(ValueError, match="at least one prompt"):
        TaskPrompts([], seed=7)


@pytest.mark.unit
def test_the_generate_function_runs_the_recorded_jobs_session_and_refuses_an_older_driver(
    tmp_path, monkeypatch
) -> None:
    source = source_of(tmp_path)
    write_json(tmp_path / "save" / DATASET_FILE, source.to_record())
    made: list[Any] = []

    class Session(HarborRolloutSession):
        def __init__(self, source, *, chat, model, top_k) -> None:
            made.append(chat)
            super().__init__(source, chat=Chat((200, GENERATED)), model=model, top_k=top_k, lab=AgentLab(reward=0.25))

    monkeypatch.setattr(harbor_rollouts, "HarborRolloutSession", Session)
    monkeypatch.setattr(harbor_rollouts, "SESSIONS", {})
    args = SimpleNamespace(
        save_hf=str(tmp_path / "save" / "hf" / "{rollout_id}"),
        hf_checkpoint="/models/m",
        sglang_router_ip="10.0.0.1",
        sglang_router_port=3000,
        rollout_temperature=0.9,
        rollout_top_p=1.0,
        rollout_top_k=-1,
        rollout_max_response_len=64,
        use_score_centering=False,
        score_centering_top_k=2,
    )
    ((name, path),) = source.tasks
    sample = FakeSample(metadata={"task": name, "path": path, "token": "token"})
    assert asyncio.run(generate_task_rollout(args, sample, {})) is sample
    assert sample.reward == 0.25 and sample.status == FakeSample.Status.COMPLETED
    # The session answers on Slime's router, with Slime's rollout distribution and response limit.
    assert made[0].router_url == "http://10.0.0.1:3000"
    assert made[0].distribution == {"temperature": 0.9, "top_p": 1.0, "top_k": -1}
    assert made[0].max_response_tokens == 64

    stale = FakeSample(metadata={"task": name, "path": path, "token": "older-driver"})
    asyncio.run(generate_task_rollout(args, stale, {}))
    assert stale.remove_sample and "newer driver" in stale.metadata["abort_reason"]
    assert len(made) == 1
    for session in harbor_rollouts.SESSIONS.values():
        session.close()


@pytest.mark.unit
def test_the_engine_chat_answers_with_the_capture_of_the_engines_tokens() -> None:
    chat = Chat((200, LIMITED))
    status, body = chat.answer("/v1/chat/completions", CHAT)
    assert status == 200
    # The agent asked for temperature 0.7, but the trainer reads Slime's rollout distribution.
    assert chat.bodies == [
        {
            "input_ids": [1, 2, 3],
            "sampling_params": {"max_new_tokens": 2, "sampling_seed": 3, **DISTRIBUTION},
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
    assert (training["prompt_length"], training["runtime_load_id"], training["weight_version"]) == (
        3,
        "slime:job",
        "3",
    )


@pytest.mark.unit
def test_slimes_response_limit_bounds_each_call() -> None:
    chat = Chat((200, GENERATED), max_response_tokens=1)
    assert chat.answer("/v1/chat/completions", CHAT)[0] == 200
    assert chat.bodies[0]["sampling_params"]["max_new_tokens"] == 1


HEADS = [[[-0.1, 7, None], [-2.5, 9, None]], [[-0.2, 8, None], [-1.9, 7, None]]]


@pytest.mark.unit
def test_the_engine_chat_records_the_engines_heads() -> None:
    answer = {"text": "done", "meta_info": {**GENERATED["meta_info"], "output_top_logprobs": HEADS}}
    chat = Chat((200, answer), topk_logprobs=2)
    status, body = chat.answer("/v1/chat/completions", CHAT)
    assert status == 200
    assert chat.bodies[0]["top_logprobs_num"] == 2
    assert (body["training"]["topk_indices"], body["training"]["topk_log_probs"]) == (
        [[7, 9], [8, 7]],
        [[-0.1, -2.5], [-0.2, -1.9]],
    )
    for heads in (None, HEADS[:1], [HEADS[0], HEADS[1][:1]], [HEADS[0], [[float("inf"), 8, None], HEADS[1][1]]]):
        meta = {**GENERATED["meta_info"], "output_top_logprobs": heads}
        status, _ = Chat((200, {"text": "done", "meta_info": meta}), topk_logprobs=2).answer(
            "/v1/chat/completions", CHAT
        )
        assert status == 502
    # Without score centering the engines record no head.
    status, body = Chat((200, GENERATED)).answer("/v1/chat/completions", CHAT)
    assert status == 200 and "topk_indices" not in body["training"]


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
def test_the_engine_chat_answers_each_call_with_its_status(path, payload, answer, status) -> None:
    assert Chat(answer).answer(path, payload)[0] == status


def slime_builder() -> Any:
    """Slime's real ``BatchBuilder``, or a skip where the Slime runtime is missing."""
    batch_builder = pytest.importorskip("slime.data.batch_builder", reason="requires the Slime runtime")
    args = SimpleNamespace(
        custom_reward_post_process_path=None,
        custom_convert_samples_to_train_data_path=None,
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=True,
        reward_key=None,
        n_samples_per_prompt=3,
        rollout_batch_size=1,
    )
    return batch_builder.BatchBuilder(args)


def slime_samples(fields: list[SampleFields]) -> list[Any]:
    types = pytest.importorskip("slime.utils.types", reason="requires the Slime runtime")
    samples = []
    for index, value in enumerate(fields):
        sample = types.Sample(group_index=0, index=index, metadata={"task": "add"})
        apply_fields(sample, value)
        samples.append(sample)
    return samples


def assembled(reward: float) -> SampleFields:
    return SampleFields((1, 2, 7, 3, 8), 3, (1, 0, 1), (-0.1, 0.0, -0.2), reward, "completed", False)


@pytest.mark.unit
def test_slime_centers_the_rollouts_that_ran_without_the_removed_sample() -> None:
    builder = slime_builder()
    samples = slime_samples([assembled(1.0), assembled(0.0), removed_fields("failed", 0)])
    train_data = builder.convert(samples)
    # The group's mean is 0.5 over the two rollouts that ran; the removed sample's reward 0 would make it 1/3.
    assert train_data["rewards"][0] == pytest.approx(-train_data["rewards"][1])
    assert train_data["rewards"][2] == 0.0
    assert train_data["loss_masks"] == [[1, 0, 1], [1, 0, 1], [0]]
    assert train_data["rollout_mask_sums"] == [2, 2, 0]
    assert train_data["rollout_log_probs"][0] == [-0.1, 0.0, -0.2]
    assert train_data["response_lengths"] == [3, 3, 1]


@pytest.mark.unit
def test_slime_gives_a_lone_rollout_of_a_group_no_advantage() -> None:
    builder = slime_builder()
    samples = slime_samples([assembled(1.0), removed_fields("a", 0), removed_fields("b", 0)])
    assert builder.convert(samples)["rewards"] == [0.0, 0.0, 0.0]
