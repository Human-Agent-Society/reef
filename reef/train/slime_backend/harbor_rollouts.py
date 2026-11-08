"""Harbor tasks as a source of samples of Slime's own training loop.

A task job runs as a dataset job (``reef.train.slime_backend.dataset``) whose generate step differs. The job's driver
records the job in the save root's ``DATASET_FILE`` (``TaskJobSource``), and ``ReefDatasetSource`` serves the job's
tasks as prompts. Each prompt names ``generate_task_rollout`` as its Slime generate function. That function runs in
Slime's rollout manager: it runs one Harbor rollout of the prompt's task, and the agent's model calls reach Slime's
own engines through ``EngineChat``. The assembled rollout becomes the fields of Slime's ``Sample``, with the verifier's
reward. Slime then computes the advantages and trains its loss as for any other sample. Nothing here depends on an
objective.

Each rollout runs on an event loop of its own, on a thread of the session, because Harbor runs in the calling process
and a blocking agent must not stop Slime's event loop. The session's thread count is ``harbor.concurrency``.

A rollout that fails in the infrastructure, or that does not assemble, becomes a removed sample (``removed_fields``).
This module imports no Ray, torch or Slime.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import random
import threading
from collections import Counter
from collections.abc import Coroutine, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

from reef.core.job_data import TrajectorySample
from reef.execution.sandbox.remote import REMOTE_TASK_ENVIRONMENTS
from reef.harness.harbor.rollouts import TaskLab, TaskRollout, TaskRolloutRunner
from reef.storage.durable_files import read_json
from reef.train.slime_backend.dataset import DATASET_FILE, save_root
from reef.train.slime_backend.engine_chat import EngineChat
from reef.train.slime_backend.native import DATASET_SOURCE_PATH, SLIME_DEFAULT_SOURCE_PATH
from reef.train.slime_backend.reef_adapters.arguments import SlimeArguments
from reef.train.task_rollouts import (
    ROLLOUT_SCENARIO,
    ChatEndpoint,
    HarborSettings,
    RolloutChat,
    rollout_assembly,
    rollout_sample,
)

logger = logging.getLogger(__name__)

#: The result of each rollout of a task job, one JSON object per line, in the job's directory.
TASK_ROLLOUTS_FILE = "task-rollouts.jsonl"
#: The Slime generate function of a task job's prompts.
GENERATE_FUNCTION_PATH = "reef.train.slime_backend.harbor_rollouts.generate_task_rollout"
#: The only rollout function that runs a prompt's generate function with these failure and abort rules.
SLIME_SYNC_ROLLOUT_PATH = "slime.rollout.sglang_rollout.generate_rollout"
#: The estimators for which Slime subtracts each prompt group's mean reward (``BatchBuilder._post_process_rewards``).
GROUP_BASELINE_ESTIMATORS = ("grpo", "gspo", "cispo", "reinforce_plus_plus_baseline")

Prompt = TypeVar("Prompt")
ResultT = TypeVar("ResultT")


def has_group_baseline(args: SlimeArguments) -> bool:
    """Whether Slime centers each reward on its prompt group, so a group of one sample has advantage 0."""
    return args.advantage_estimator in GROUP_BASELINE_ESTIMATORS and args.rewards_normalization


def head_count(args: SlimeArguments) -> int:
    """The sampler log-probs that Slime's score centering reads per response token, or 0 when it is off."""
    return args.score_centering_top_k if args.use_score_centering else 0


def check_task_job_options(args: SlimeArguments) -> None:
    """Raise ``ValueError`` for each option of Slime's parsed ``args`` that a task job cannot work with."""
    if args.loss_family != "slime":
        raise ValueError("a task job needs the 'slime' loss family: Slime computes its advantages and trains its loss")
    for refused, message in (
        (args.group_rm, "--group-rm: the verifier rewards each rollout"),
        (args.reward_key, "--reward-key: the verifier's reward is one number"),
        (args.custom_generate_function_path, "--custom-generate-function-path: a Harbor rollout is the generate step"),
        (
            args.data_source_path not in (None, SLIME_DEFAULT_SOURCE_PATH, DATASET_SOURCE_PATH),
            "--data-source-path: Reef's data source serves the tasks",
        ),
        (
            args.rollout_function_path != SLIME_SYNC_ROLLOUT_PATH,
            f"--rollout-function-path other than {SLIME_SYNC_ROLLOUT_PATH}",
        ),
        (args.partial_rollout, "--partial-rollout: a Harbor rollout cannot continue later"),
        (
            args.over_sampling_batch_size not in (None, args.rollout_batch_size),
            "--over-sampling-batch-size other than --rollout-batch-size: extra rollouts would run and be aborted",
        ),
        (args.dynamic_sampling_filter_path, "--dynamic-sampling-filter-path: a dropped group shifts the resume point"),
        (args.rollout_top_p != 1, "--rollout-top-p other than 1: the engines' answers carry no top-p replay"),
        (
            args.use_rollout_routing_replay,
            "--use-rollout-routing-replay: the engines' answers carry no routed experts",
        ),
    ):
        if refused:
            raise ValueError(f"a task job cannot run with {message}")
    if has_group_baseline(args) and args.n_samples_per_prompt < 2:
        raise ValueError(
            f"a task job with --advantage-estimator {args.advantage_estimator} needs --n-samples-per-prompt of at "
            "least 2: Slime centers each reward on its group, so a group of one sample trains nothing"
        )


@dataclass(frozen=True)
class TaskJobSource:
    """The task job that the save root's ``DATASET_FILE`` records.

    Values: ``job_id`` and ``token`` are non-empty strings, and each driver makes a new ``token``. ``tasks`` is a
    non-empty tuple of ``(name, path)`` pairs in the job's order, with distinct names and absolute paths.
    ``harbor`` is how to run the rollouts. ``job_directory`` is the job's absolute directory.
    ``timeout_seconds`` is a positive bound on one model call. ``skip_rollouts`` is a non-negative int, the job's
    Slime rollouts that trained before this driver. Purpose: the driver makes it; ``ReefDatasetSource`` and
    ``task_session`` read it from the record in the rollout manager's process.
    """

    job_id: str
    token: str
    tasks: tuple[tuple[str, str], ...]
    harbor: HarborSettings
    job_directory: str
    timeout_seconds: float
    skip_rollouts: int

    def __post_init__(self) -> None:
        if not self.job_id or not self.token:
            raise ValueError("TaskJobSource needs a job ID and a token")
        names = [name for name, _ in self.tasks]
        if not names or len(set(names)) != len(names):
            raise ValueError("TaskJobSource.tasks needs at least one task, with distinct names")
        if any(not Path(path).is_absolute() for _, path in self.tasks) or not Path(self.job_directory).is_absolute():
            raise ValueError("TaskJobSource needs absolute task paths and an absolute job directory")
        if not self.timeout_seconds > 0:
            raise ValueError("TaskJobSource.timeout_seconds must be positive")
        if self.skip_rollouts < 0:
            raise ValueError("TaskJobSource.skip_rollouts must not be negative")

    def to_record(self) -> dict[str, Any]:
        """This value as the JSON object of ``DATASET_FILE``, which ``from_record`` reads back."""
        return {**asdict(self), "tasks": [list(task) for task in self.tasks]}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> TaskJobSource:
        return cls(
            record["job_id"],
            record["token"],
            tuple((name, path) for name, path in record["tasks"]),
            HarborSettings(**record["harbor"]),
            record["job_directory"],
            record["timeout_seconds"],
            record["skip_rollouts"],
        )


class TaskPrompts(Generic[Prompt]):
    """The prompts of one task job, with the part of the interface of Slime's ``Dataset`` that its data source reads.

    Values: ``samples`` is a non-empty list of ``Prompt`` values, one per task, in the order of the current epoch.
    ``shuffle(epoch)`` orders them as Slime's ``Dataset.shuffle`` does: a function of ``seed`` and ``epoch`` only, so
    a resumed job sees the same order. Purpose: ``ReefDatasetSource`` makes it, and Slime's ``RolloutDataSource``
    reads ``samples``, ``len`` and ``shuffle``.
    """

    def __init__(self, samples: Sequence[Prompt], seed: int) -> None:
        if not samples:
            raise ValueError("TaskPrompts needs at least one prompt")
        self.origin_samples = list(samples)
        self.samples = list(samples)
        self.seed = seed
        self.epoch_id = -1

    def shuffle(self, epoch: int) -> None:
        if epoch == self.epoch_id:
            return
        order = list(range(len(self.origin_samples)))
        random.Random(self.seed + epoch).shuffle(order)
        self.samples = [self.origin_samples[index] for index in order]
        self.epoch_id = epoch

    def __len__(self) -> int:
        return len(self.samples)


@dataclass(frozen=True)
class SampleFields:
    """What one rollout writes into Slime's ``Sample``.

    Values: ``tokens`` is the prompt of the first model call and the ``response_length`` response tokens after it.
    ``response_length`` is positive. ``loss_mask`` and ``rollout_log_probs`` have one entry per response token; a
    mask of 1 marks a token that the model sampled, whose entry is its sampled log-prob, and a mask of 0 marks a
    prompt token of a later call, whose entry is 0.0. ``status`` is the value of Slime's ``Sample.Status``:
    ``completed``, ``truncated`` (the last call reached its token limit) or ``aborted``. ``remove_sample`` holds if
    and only if the status is ``aborted``, and then every mask entry is 0 and ``reward`` is 0.0. ``weight_versions``
    holds the engines' weight version of each model call that gave one. ``topk_token_ids`` and ``topk_log_probs`` are
    None, or both hold one row per response token of Slime's score centering width. ``error`` says why a removed
    sample ran no training rollout, and is "" otherwise. Purpose: ``HarborRolloutSession`` makes it, and
    ``apply_fields`` writes it.
    """

    tokens: tuple[int, ...]
    response_length: int
    loss_mask: tuple[int, ...]
    rollout_log_probs: tuple[float, ...]
    reward: float
    status: str
    remove_sample: bool
    response: str = ""
    weight_versions: tuple[str, ...] = ()
    topk_token_ids: tuple[tuple[int, ...], ...] | None = None
    topk_log_probs: tuple[tuple[float, ...], ...] | None = None
    error: str = ""

    def __post_init__(self) -> None:
        count = self.response_length
        if count < 1 or len(self.tokens) <= count:
            raise ValueError("SampleFields needs a prompt and at least one response token")
        if len(self.loss_mask) != count or len(self.rollout_log_probs) != count:
            raise ValueError("SampleFields needs one mask entry and one log-prob per response token")
        if self.status not in ("completed", "truncated", "aborted"):
            raise ValueError(f"SampleFields.status {self.status!r} is not one of Slime's terminal statuses")
        if self.remove_sample != (self.status == "aborted"):
            raise ValueError("SampleFields removes a sample if and only if it is aborted")
        if self.remove_sample and (any(self.loss_mask) or self.reward != 0.0):
            raise ValueError("a removed sample has no trained token and reward 0")
        heads = (self.topk_token_ids, self.topk_log_probs)
        if (heads[0] is None) != (heads[1] is None) or any(row is not None and len(row) != count for row in heads):
            raise ValueError("SampleFields needs both sampler heads, one row per response token, or neither")


def dummy_head(top_k: int) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """The head of a masked token, as Slime's ``Sample.append_response_tokens`` makes it: distinct IDs, uniform mass."""
    return tuple(range(top_k)), (-math.log(top_k),) * top_k


def removed_fields(error: str, top_k: int) -> SampleFields:
    """A sample that does not train, in the shape of Slime's ``examples/coding_agent_rl/generate.py``.

    ``BatchBuilder.convert`` zeroes the loss mask of a removed sample and clamps its denominator to 1. Score centering
    validates the heads of every sample, so a removed sample has the heads of a masked token.
    """
    heads = dummy_head(top_k) if top_k else None
    return SampleFields(
        tokens=(0, 0),
        response_length=1,
        loss_mask=(0,),
        rollout_log_probs=(0.0,),
        reward=0.0,
        status="aborted",
        remove_sample=True,
        topk_token_ids=None if heads is None else (heads[0],),
        topk_log_probs=None if heads is None else (heads[1],),
        error=error,
    )


def assembled_fields(rollout: TaskRollout, sample: TrajectorySample, top_k: int) -> SampleFields | None:
    """The Slime fields of a rollout's assembled ``sample``, or None when it has no trained token or lacks a head.

    The assembly already follows Slime's multi-turn shape: the prompt of the first call is the prompt, and each later
    call's new prompt tokens are response tokens with mask 0 and log-prob 0.0. A rollout that ran without a reward
    scores 0.
    """
    training = sample.training
    loss_mask = tuple(training["loss_mask"])
    log_probs = tuple(training["rollout_log_probs"])
    if not any(loss_mask) or len(log_probs) != len(loss_mask):
        return None
    topk_ids: tuple[tuple[int, ...], ...] | None = None
    topk_values: tuple[tuple[float, ...], ...] | None = None
    if top_k:
        ids, values = training.get("topk_indices", ()), training.get("topk_log_probs", ())
        if len(ids) != len(loss_mask):
            return None
        filler_ids, filler_values = dummy_head(top_k)
        topk_ids = tuple(tuple(row) if row else filler_ids for row in ids)
        topk_values = tuple(tuple(row) if row else filler_values for row in values)
    captures = [turn.response["training"] for turn in rollout.turns]
    return SampleFields(
        tokens=tuple(training["tokens"]),
        response_length=len(loss_mask),
        loss_mask=loss_mask,
        rollout_log_probs=log_probs,
        reward=0.0 if rollout.reward is None else rollout.reward,
        status="truncated" if captures[-1].get("finish_reason") == "length" else "completed",
        remove_sample=False,
        response=captures[-1]["response_message"]["content"],
        weight_versions=tuple(capture["weight_version"] for capture in captures if "weight_version" in capture),
        topk_token_ids=topk_ids,
        topk_log_probs=topk_values,
    )


def apply_fields(sample: Any, fields: SampleFields) -> None:
    """Write ``fields`` into Slime's ``Sample``.

    A removed sample leaves its prompt group: Slime keys the group's mean and deviation by ``group_index``, so a
    negative index of its own keeps its reward 0 out of the statistics of the rollouts that ran.
    """
    sample.tokens = list(fields.tokens)
    sample.response = fields.response
    sample.response_length = fields.response_length
    sample.loss_mask = list(fields.loss_mask)
    sample.rollout_log_probs = list(fields.rollout_log_probs)
    sample.reward = fields.reward
    sample.remove_sample = fields.remove_sample
    sample.weight_versions = list(fields.weight_versions)
    # The status is an enum member of Sample.Status, which this module does not import.
    sample.status = type(sample.status)(fields.status)
    if fields.topk_token_ids is not None and fields.topk_log_probs is not None:
        sample.rollout_topk_token_ids = [list(row) for row in fields.topk_token_ids]
        sample.rollout_topk_log_probs = [list(row) for row in fields.topk_log_probs]
    if fields.remove_sample:
        sample.group_index = -1 - sample.index
        sample.metadata = {**sample.metadata, "abort_reason": fields.error}


@dataclass(frozen=True)
class TaskRolloutResult:
    """What one rollout of a task job came to.

    Values: ``token`` names the driver whose rollout it was, ``task`` the task's name, and ``group_index`` Slime's
    prompt group. ``reward`` is the rollout's score, or None when the infrastructure failed. ``assembled`` holds if and
    only if the rollout's sample trains, which implies a reward. ``error`` is why the sample does not train, and "" if
    it does. Purpose: ``HarborRolloutSession`` appends it to ``TASK_ROLLOUTS_FILE``, and the driver reads it to check
    each Slime rollout and to report ``rollout_rewards``.
    """

    token: str
    task: str
    group_index: int
    reward: float | None
    assembled: bool
    error: str

    def __post_init__(self) -> None:
        if self.assembled and (self.reward is None or self.error):
            raise ValueError("a sample that trains has a reward and no error")
        if not self.assembled and not self.error:
            raise ValueError("a sample that does not train needs its error")


def read_task_rollout_results(directory: Path) -> tuple[TaskRolloutResult, ...]:
    """The results in the job directory's ``TASK_ROLLOUTS_FILE``, in the order the rollouts ended."""
    path = directory / TASK_ROLLOUTS_FILE
    if not path.exists():
        return ()
    return tuple(TaskRolloutResult(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines())


def check_rollout_can_train(results: Sequence[TaskRolloutResult], *, group_size: int) -> None:
    """Raise ``RuntimeError`` unless a prompt group of one Slime rollout holds ``group_size`` samples that train.

    Otherwise the workers would train zero loss and the job would save a checkpoint of nothing new.
    """
    counts = Counter(result.group_index for result in results if result.assembled)
    if not counts or max(counts.values()) < group_size:
        errors = [result.error for result in results if result.error]
        first = errors[0] if errors else "no rollout ended"
        raise RuntimeError(
            f"{len(errors)} of {len(results)} task rollouts did not give a sample, so no prompt group has "
            f"{group_size} samples to train; the first error: {first}"
        )


def task_job_metrics(task_names: Sequence[str], results: Sequence[TaskRolloutResult]) -> dict[str, Any]:
    """The job's metrics: each task's rewards of the rollouts that ran, the samples that trained, and the others."""
    rewards: dict[str, list[float]] = {name: [] for name in task_names}
    for result in results:
        if result.reward is not None:
            rewards[result.task].append(result.reward)
    samples = sum(result.assembled for result in results)
    return {"rollout_rewards": rewards, "rollout_samples": samples, "removed_rollouts": len(results) - samples}


class HarborRolloutSession:
    """The Harbor rollouts of one driver of one task job, in Slime's rollout manager.

    ``source`` is the recorded job, ``chat`` answers the agent's model calls, and ``model`` is the model name the agent
    sends. ``top_k`` is the sampler log-probs per response token, or 0. ``lab`` runs each rollout; None is reef-eval's
    Harbor lab. ``run`` gives the Slime fields of one rollout and appends its result; a failed rollout gives
    ``removed_fields``, and cancelling ``run`` cancels the rollout and waits for its cleanup. At most
    ``harbor.concurrency`` rollouts and environment builds run at a time. ``close`` closes the chat endpoint and lets
    the threads end when their work ends.
    """

    def __init__(
        self, source: TaskJobSource, *, chat: RolloutChat, model: str, top_k: int, lab: TaskLab | None = None
    ) -> None:
        settings = source.harbor
        self.source = source
        self.top_k = top_k
        self.assembly = rollout_assembly(settings)
        self.results_path = Path(source.job_directory) / TASK_ROLLOUTS_FILE
        self.results_lock = threading.Lock()
        self.threads = ThreadPoolExecutor(max_workers=settings.concurrency, thread_name_prefix="task-rollout")
        # One build of each task's remote environment; its rollouts wait for it. Keyed and read on Slime's loop.
        self.builds: dict[Path, asyncio.Future[str]] = {}
        self.endpoint = ChatEndpoint(chat)
        self.endpoint.__enter__()
        self.runner = TaskRolloutRunner(
            reef_url=self.endpoint.url,
            scenario=ROLLOUT_SCENARIO,
            model=model,
            work_dir=Path(source.job_directory) / "rollouts",
            agent=settings.agent,
            agent_host=settings.agent_host,
            environment=settings.environment,
            is_reporting=False,
            is_unrecorded=True,
            verifier_timeout_reward=settings.verifier_timeout_reward,
            lab=lab,
            concurrency=settings.concurrency,
        )

    def close(self) -> None:
        self.endpoint.__exit__()
        self.threads.shutdown(wait=False)

    async def run(self, task: str, task_path: Path, group_index: int) -> SampleFields:
        """One rollout of ``task``, called on Slime's event loop."""
        build_error = await self.built(task_path)
        if build_error:
            return self.finish(
                task,
                group_index,
                removed_fields(f"the environment of {task} did not start: {build_error}", self.top_k),
                None,
            )
        return await self.on_own_loop(self.rollout(task, task_path, group_index))

    async def built(self, task_path: Path) -> str:
        """Wait for the one build of the task's remote environment: "", or the error that stopped it.

        Two concurrent rollouts of one task must not build its environment together, which E2B refuses.
        """
        if self.runner.environment not in REMOTE_TASK_ENVIRONMENTS:
            return ""
        if task_path not in self.builds:
            self.builds[task_path] = asyncio.ensure_future(self.on_own_loop(self.build(task_path)))
        # The shield: an aborted rollout must not cancel the build that other rollouts wait for.
        return await asyncio.shield(self.builds[task_path])

    async def build(self, task_path: Path) -> str:
        try:
            return await self.runner.build_task_environment(task_path, asyncio.Semaphore(1))
        # Harbor and the provider's SDK raise their own types, and one task's build must not stop the other tasks.
        except Exception as exc:
            return str(exc) or type(exc).__name__

    async def rollout(self, task: str, task_path: Path, group_index: int) -> SampleFields:
        """Run and assemble one rollout on this thread's own loop."""
        try:
            rollout = await self.runner.run_async(task_path)
        # Harbor and the agent raise their own types, and one rollout must not stop the Slime rollout's other samples.
        except Exception as exc:
            logger.warning("a rollout of task %s failed", task, exc_info=True)
            return self.finish(task, group_index, removed_fields(f"the rollout raised: {exc!r}", self.top_k), None)
        if rollout.is_infrastructure_failure:
            return self.finish(task, group_index, removed_fields(rollout.error, self.top_k), None)
        reward = 0.0 if rollout.reward is None else rollout.reward
        sample = rollout_sample(rollout, task, self.assembly, self.source.job_id)
        fields = None if sample is None else assembled_fields(rollout, sample, self.top_k)
        if fields is None:
            error = rollout.error or "the rollout's model calls do not assemble into a sample"
            return self.finish(task, group_index, removed_fields(error, self.top_k), reward)
        return self.finish(task, group_index, fields, reward)

    def finish(self, task: str, group_index: int, fields: SampleFields, reward: float | None) -> SampleFields:
        """Append the rollout's result and return ``fields``. ``reward`` is None when the infrastructure failed."""
        result = TaskRolloutResult(
            self.source.token, task, group_index, reward, not fields.remove_sample, fields.error
        )
        with self.results_lock, open(self.results_path, "a", encoding="utf-8") as results:
            results.write(json.dumps(asdict(result)) + "\n")
        return fields

    async def on_own_loop(self, work: Coroutine[Any, Any, ResultT]) -> ResultT:
        """Await ``work`` on a new event loop on one of the session's threads, without blocking the calling loop.

        A cancel of this call cancels ``work`` and waits for its cleanup, also when ``work`` still waits for a thread.
        """
        loop = asyncio.new_event_loop()
        task = loop.create_task(work)

        def run() -> ResultT:
            try:
                return loop.run_until_complete(task)
            finally:
                # An agent or an SDK can leave tasks behind; they end with this rollout's loop.
                left = asyncio.all_tasks(loop)
                for leftover in left:
                    leftover.cancel()
                if left:
                    loop.run_until_complete(asyncio.gather(*left, return_exceptions=True))
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.close()

        finished = asyncio.get_running_loop().run_in_executor(self.threads, run)
        try:
            return await asyncio.shield(finished)
        except asyncio.CancelledError:
            # The loop ended meanwhile when the rollout has just finished, and there is nothing to cancel.
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(task.cancel)
            await asyncio.wait({finished})
            with contextlib.suppress(asyncio.CancelledError):
                finished.exception()
            raise


#: The session of the driver that the save root's record names, by its token: at most one, in the rollout manager.
SESSIONS: dict[str, HarborRolloutSession] = {}


def task_session(args: SlimeArguments, token: str) -> HarborRolloutSession | None:
    """The session of driver ``token``, or None when a newer driver replaced that driver's record.

    A new token closes the previous driver's session. Slime calls the generate function on one event loop, so this
    function has one caller at a time.
    """
    session = SESSIONS.get(token)
    if session is not None:
        return session
    record = read_json(save_root(args.save_hf) / DATASET_FILE)
    if record is None or record["token"] != token:
        return None
    for previous in SESSIONS.values():
        previous.close()
    SESSIONS.clear()
    source = TaskJobSource.from_record(record)
    top_k = head_count(args)
    chat = EngineChat(
        f"http://{args.sglang_router_ip}:{args.sglang_router_port}",
        model_path=args.hf_checkpoint,
        timeout_s=source.timeout_seconds,
        runtime_load_id=f"slime:{source.job_id}",
        distribution={
            "temperature": args.rollout_temperature,
            "top_p": args.rollout_top_p,
            "top_k": args.rollout_top_k,
        },
        max_response_tokens=args.rollout_max_response_len,
        topk_logprobs=top_k,
    )
    SESSIONS[token] = HarborRolloutSession(source, chat=chat, model=args.hf_checkpoint, top_k=top_k)
    return SESSIONS[token]


async def generate_task_rollout(args: SlimeArguments, sample: Any, sampling_params: dict[str, Any]) -> Any:
    """Slime's generate function of a task job's prompt: run one Harbor rollout of its task, and fill ``sample``.

    ``sampling_params`` are not read: ``EngineChat`` sends Slime's rollout distribution with each model call. The
    result is one ``Sample``, one rollout. Its reward is set, so Slime calls no reward model.
    """
    session = task_session(args, sample.metadata["token"])
    if session is None:
        fields = removed_fields("a newer driver of the job replaced this rollout's driver", head_count(args))
    else:
        fields = await session.run(sample.metadata["task"], Path(sample.metadata["path"]), sample.group_index)
    apply_fields(sample, fields)
    return sample


# Slime cancels the task of a request-abortable generate function on abort, instead of aborting the engines' requests:
# Harbor's agent would retry a request that the engines abort.
generate_task_rollout.abort_mode = "request"  # type: ignore[attr-defined]


__all__ = [
    "GENERATE_FUNCTION_PATH",
    "HarborRolloutSession",
    "SampleFields",
    "TaskJobSource",
    "TaskPrompts",
    "TaskRolloutResult",
    "check_rollout_can_train",
    "check_task_job_options",
    "generate_task_rollout",
    "read_task_rollout_results",
    "task_job_metrics",
]
