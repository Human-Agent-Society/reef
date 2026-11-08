"""Delayed samples in a Slime task job: the record, the generate function of the delayed groups and the checks.

These tests need no Ray, torch or Slime, except the last three, which run where Slime is installed: two run the
delayed groups through Slime's data source, its batch builder and its optimizer-step schedule at the pin, and one
checks Slime's TIS weight on the clipped loss.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from tests.support.job_data import delayed_sample

from reef.storage.durable_files import write_json
from reef.train.algos.group_advantages import GroupAdvantageRule
from reef.train.algos.signals import TrainingSignal
from reef.train.slime_backend import harbor_rollouts
from reef.train.slime_backend.dataset import DATASET_FILE
from reef.train.slime_backend.harbor_rollouts import (
    DELAYED_FUNCTION_PATH,
    DELAYED_POSITION_KEY,
    DELAYED_ROWS_FILE,
    SampleFields,
    TaskJobSource,
    apply_fields,
    check_delayed_task_job,
    delayed_group_count,
    generate_delayed_samples,
    read_delayed_rows,
    trainable_delayed,
    write_delayed_rows,
)
from reef.train.slime_backend.objectives import SlimeObjective, resolve_slime_objective
from reef.train.slime_backend.task_advantages import ADVANTAGE_RULE_KEY, DELAYED_ADVANTAGE_KEY, TASK_ADVANTAGES_PATH
from reef.train.task_rollouts import HarborSettings

SPADE = GroupAdvantageRule(divide_by_std=True, std="population", epsilon=1e-8)


class CarryingObjective(SlimeObjective):
    """A method on the slime family that centers by SPADE's rule and carries delayed samples."""

    name = "test-carrying"
    loss_family = "slime"
    rollout_advantages = SPADE
    carries_samples = True

    def prepare(self, data: Any, state: Mapping[str, Any]) -> TrainingSignal:
        raise AssertionError("a task job prepares nothing")


class RuleObjective(SlimeObjective):
    """A method on the slime family with its own rule that carries no samples."""

    name = "test-rule"
    loss_family = "slime"
    rollout_advantages = SPADE

    def prepare(self, data: Any, state: Mapping[str, Any]) -> TrainingSignal:
        raise AssertionError("a task job prepares nothing")


def source_of(tmp_path: Path, *, count: int = 3, carries: bool = True, skip: int = 0) -> TaskJobSource:
    directory = tmp_path / "job"
    directory.mkdir(exist_ok=True)
    return TaskJobSource(
        "job",
        "token",
        (("add", str(tmp_path / "add")),),
        HarborSettings(),
        str(directory),
        10.0,
        skip,
        carries_samples=carries,
        delayed_count=count,
        rollout_advantages=SPADE,
    )


class Args(SimpleNamespace):
    """Slime's options that a task job of an objective that carries samples reads, with values that work."""

    def __init__(self, **overrides: Any) -> None:
        values = {
            "advantage_estimator": "grpo",
            "n_samples_per_prompt": 2,
            "rollout_batch_size": 4,
            "global_batch_size": 7,
            "use_tis": True,
            "normalize_advantages": False,
            "custom_advantage_function_path": None,
            "use_dynamic_batch_size": True,
            "custom_tis_function_path": None,
            "use_rollout_logprobs": False,
            "rollout_temperature": 0.6,
            "rollout_top_p": 1.0,
            "rollout_top_k": -1,
            "use_score_centering": False,
            "score_centering_top_k": 2,
        }
        super().__init__(**{**values, **overrides})


class FakeSample:
    """The fields of Slime's ``Sample`` that the delayed generate function reads and writes."""

    class Status(Enum):
        PENDING = "pending"
        COMPLETED = "completed"
        TRUNCATED = "truncated"
        ABORTED = "aborted"

    def __init__(self, index: int, metadata: dict[str, Any], *, rollout_id: int | None = None) -> None:
        self.index = index
        self.group_index = 0
        self.rollout_id = rollout_id
        self.metadata = metadata
        self.status = FakeSample.Status.PENDING
        self.rollout_topk_token_ids = None
        self.rollout_topk_log_probs = None
        self.remove_sample = False


@pytest.mark.unit
def test_the_record_of_a_job_that_carries_samples_reads_back(tmp_path) -> None:
    source = source_of(tmp_path)
    assert TaskJobSource.from_record(json.loads(json.dumps(source.to_record()))) == source
    with pytest.raises(ValueError, match="carries samples"):
        TaskJobSource("j", "t", (("a", "/a"),), HarborSettings(), "/job", 1.0, 0, delayed_count=1)


@pytest.mark.unit
def test_the_delayed_groups_lead_only_the_first_rollout_and_shift_the_task_prompts(tmp_path) -> None:
    assert [delayed_group_count(count, 8) for count in (0, 1, 8, 9, 24)] == [1, 1, 1, 2, 3]
    # Nothing to carry still takes one group of padding rows, so the batch keeps its shape.
    assert source_of(tmp_path, count=0).delayed_groups(8) == 1
    assert source_of(tmp_path, count=3).delayed_groups(2) == 2
    assert source_of(tmp_path, count=0, carries=False).delayed_groups(8) == 0
    assert source_of(tmp_path, count=3).skipped_task_prompts(5, 2) == 0
    # Two rollouts of five prompts served two delayed groups and eight tasks.
    assert source_of(tmp_path, count=3, skip=2).skipped_task_prompts(5, 2) == 8
    assert source_of(tmp_path, count=0, carries=False, skip=2).skipped_task_prompts(5, 2) == 10


@pytest.mark.unit
def test_each_task_prompt_carries_the_objectives_rule(tmp_path) -> None:
    source = source_of(tmp_path)
    assert source.task_metadata("add", "/add") == {
        "task": "add",
        "path": "/add",
        "token": "token",
        ADVANTAGE_RULE_KEY: SPADE.to_dict(),
    }
    plain = TaskJobSource("j", "t", (("a", "/a"),), HarborSettings(), "/job", 1.0, 0)
    assert plain.task_metadata("a", "/a") == {"task": "a", "path": "/a", "token": "t"}
    assert source.delayed_metadata(2) == {
        DELAYED_POSITION_KEY: 2,
        "path": str(Path(source.job_directory) / DELAYED_ROWS_FILE),
        "token": "token",
    }


@pytest.mark.unit
def test_the_delayed_rows_keep_each_samples_tokens_and_advantage(tmp_path) -> None:
    write_delayed_rows(tmp_path, [delayed_sample(2.5), delayed_sample(-1.0, tokens=4)])
    rows = read_delayed_rows(tmp_path / DELAYED_ROWS_FILE)
    assert [advantage for _, advantage in rows] == [2.5, -1.0]
    fields = rows[1][0]
    assert (fields.tokens, fields.response_length, fields.loss_mask) == ((1, 2, 3, 4), 2, (1, 1))
    assert (fields.rollout_log_probs, fields.reward, fields.status) == ((-0.5, -1.25), 0.0, "completed")
    write_delayed_rows(tmp_path, [])
    assert read_delayed_rows(tmp_path / DELAYED_ROWS_FILE) == ()


@pytest.mark.unit
def test_a_delayed_entry_fills_its_sample_and_a_later_entry_pads_in_its_groups_rollout(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(harbor_rollouts, "DELAYED_ROWS", {})
    source = source_of(tmp_path, count=3)
    write_delayed_rows(Path(source.job_directory), [delayed_sample(2.5), delayed_sample(-1.0), delayed_sample(0.5)])
    args = Args()
    carried = FakeSample(2, source.delayed_metadata(2))
    padding = FakeSample(3, source.delayed_metadata(3), rollout_id=2)
    asyncio.run(generate_delayed_samples(args, carried, {}))
    asyncio.run(generate_delayed_samples(args, padding, {}))

    assert (carried.tokens, carried.loss_mask, carried.rollout_log_probs) == ([1, 2, 3, 4, 5], [1, 1], [-0.5, -1.25])
    assert (carried.reward, carried.status, carried.remove_sample) == (0.0, FakeSample.Status.COMPLETED, False)
    assert carried.metadata[DELAYED_ADVANTAGE_KEY] == 0.5
    assert padding.remove_sample and padding.loss_mask == [0] and padding.metadata[DELAYED_ADVANTAGE_KEY] == 0.0
    # The padding row stays in the rollout of its group's first entry, so it adds no rollout to Slime's step.
    assert padding.rollout_id == 2
    # A new driver's entries read its own rows.
    write_delayed_rows(Path(source.job_directory), [delayed_sample(7.0)])
    newer = FakeSample(4, {**source.delayed_metadata(0), "token": "newer"})
    asyncio.run(generate_delayed_samples(args, newer, {}))
    assert newer.metadata[DELAYED_ADVANTAGE_KEY] == 7.0 and list(harbor_rollouts.DELAYED_ROWS) == ["newer"]


@pytest.mark.unit
def test_a_task_job_that_carries_samples_passes_with_options_that_work() -> None:
    check_delayed_task_job(Args(), CarryingObjective())
    check_delayed_task_job(Args(use_tis=False), RuleObjective())
    # Slime's own objective reads none of these options.
    check_delayed_task_job(SimpleNamespace(), resolve_slime_objective("slime_dataset"))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("objective", "overrides", "message"),
    [
        ("rule", {"advantage_estimator": "ppo"}, "--advantage-estimator in grpo, gspo, cispo"),
        ("rule", {"n_samples_per_prompt": 1}, "at least 2"),
        ("rule", {"normalize_advantages": True}, "--normalize-advantages"),
        ("rule", {"custom_advantage_function_path": "my.advantages"}, "--custom-advantage-function-path"),
        ("carrying", {"rollout_batch_size": 1, "global_batch_size": 1}, "--rollout-batch-size of at least 2"),
        ("carrying", {"use_tis": False}, "without --use-tis"),
        ("carrying", {"custom_tis_function_path": "my.tis"}, "--custom-tis-function-path"),
        ("carrying", {"use_rollout_logprobs": True}, "--use-rollout-logprobs"),
        ("carrying", {"use_dynamic_batch_size": False}, "--use-dynamic-batch-size"),
        ("carrying", {"use_score_centering": True}, "--use-score-centering"),
        # A first rollout of 4 prompts in groups of 2 holds at least 3 * 2 + 1 = 7 rollouts.
        ("carrying", {"global_batch_size": 8}, "--global-batch-size of at most 7"),
    ],
)
def test_options_that_a_task_job_of_the_objective_cannot_run_with_are_refused(
    objective: str, overrides: dict[str, Any], message: str
) -> None:
    objectives = {"rule": RuleObjective(), "carrying": CarryingObjective()}
    with pytest.raises(ValueError, match=message):
        check_delayed_task_job(Args(**overrides), objectives[objective])


@pytest.mark.unit
def test_the_delayed_samples_that_cannot_train_are_left_out_so_the_tasks_still_train() -> None:
    samples = [delayed_sample(float(index)) for index in range(8)]
    other = delayed_sample(9.0, distribution={"temperature": 1.0, "top_p": 1.0, "top_k": -1})

    assert trainable_delayed(Args(), CarryingObjective(), samples[:3]) == (tuple(samples[:3]), {})
    assert trainable_delayed(Args(), resolve_slime_objective("slime_dataset"), samples[:2]) == ((), {"objective": 2})
    assert trainable_delayed(Args(), CarryingObjective(), [other, samples[0]]) == ((samples[0],), {"distribution": 1})
    # Groups of 2 in a rollout of 4 prompts: at most 3 groups (6 samples) leave a prompt for a task.
    assert trainable_delayed(Args(global_batch_size=7), CarryingObjective(), samples) == (
        tuple(samples[:6]),
        {"capacity": 2},
    )
    assert trainable_delayed(Args(global_batch_size=5), CarryingObjective(), samples[:6]) == (
        tuple(samples[:5]),
        {"capacity": 1},
    )


def slime_rollout(
    tmp_path: Path, *, carried: list[float], group_size: int, batch_size: int, task_rewards: list[list[float]]
) -> tuple[Any, list[Any]]:
    """One task-job rollout through Slime's data source, the delayed generate function and Slime's batch hook.

    Returns Slime's arguments and the rollout's flat samples in batch order. Each task's plays get the rewards of
    ``task_rewards``; the generate step of the plays is a stand-in for the Harbor rollout.
    """
    dataset_source = pytest.importorskip(
        "reef.train.slime_backend.dataset_source", reason="requires the Slime runtime"
    )
    base_types = pytest.importorskip("slime.rollout.base_types", reason="requires the Slime runtime")
    rollout_data_utils = pytest.importorskip("slime.observability.rollout_data_utils")
    harbor_rollouts.DELAYED_ROWS.clear()
    args = SimpleNamespace(
        prompt_data=None,
        buffer_filter_path=None,
        buffer_sort_by_staleness=False,
        save_hf=str(tmp_path / "save" / "hf" / "{rollout_id}"),
        rollout_seed=1,
        rollout_shuffle=False,
        rollout_batch_size=batch_size,
        n_samples_per_prompt=group_size,
        rollout_data_transport="object-store",
        rollout_sample_filter_path=None,
        use_score_centering=False,
        score_centering_top_k=0,
    )
    directory = tmp_path / "job"
    directory.mkdir()
    tasks = []
    for index in range(len(task_rewards)):
        (tmp_path / f"task-{index}").mkdir()
        tasks.append((f"task-{index}", str(tmp_path / f"task-{index}")))
    write_delayed_rows(directory, [delayed_sample(advantage) for advantage in carried])
    source = TaskJobSource(
        "job",
        "token",
        tuple(tasks),
        HarborSettings(),
        str(directory),
        10.0,
        0,
        carries_samples=True,
        delayed_count=len(carried),
        rollout_advantages=SPADE,
    )
    write_json(tmp_path / "save" / DATASET_FILE, source.to_record())
    groups = dataset_source.ReefDatasetSource(args).get_samples(batch_size)
    for group in groups:
        for sample in group:
            if sample.generate_function_path == DELAYED_FUNCTION_PATH:
                asyncio.run(generate_delayed_samples(args, sample, {}))
            else:
                reward = task_rewards[int(sample.metadata["task"].removeprefix("task-"))].pop(0)
                fields = SampleFields((1, 2, 7, 8), 2, (1, 1), (-0.1, -0.2), reward, "completed", False)
                apply_fields(sample, fields)
    # Slime's rollout function completes the groups in any order; its batch hook sorts them.
    groups.reverse()
    output = base_types.finalize_rollout_groups(args, 0, groups)
    data = output.samples
    rollout_data_utils.validate_rollout_id_annotated(data)
    return args, [sample for group in data for sample in group]


def train_step(
    args: Any, samples: list[Any], global_batch_size: int, *, dp_size: int = 1, dynamic: bool = False
) -> tuple[dict[str, Any], list[int]]:
    """Slime's conversion of the rollout with Reef's reward step, and the sample positions of its first step on the
    first of ``dp_size`` ranks, with fixed or dynamic micro-batches."""
    batch_builder = pytest.importorskip("slime.data.batch_builder", reason="requires the Slime runtime")
    dp_schedule = pytest.importorskip("slime.utils.dp_schedule", reason="requires the Slime runtime")
    args = SimpleNamespace(
        **vars(args),
        custom_reward_post_process_path=TASK_ADVANTAGES_PATH,
        custom_convert_samples_to_train_data_path=None,
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=True,
        reward_key=None,
        rollout_top_p=1.0,
        use_rollout_routing_replay=False,
        use_dynamic_batch_size=dynamic,
        micro_batch_size=1,
        max_tokens_per_gpu=4096 if dynamic else None,
        balance_data=False,
        balance_by_flops=False,
    )
    train_data = batch_builder.BatchBuilder(args).convert(samples)
    partitions, _, steps, _ = dp_schedule.build_dp_schedule(
        args,
        {"dp_size": dp_size, "cp_size": 1, "vpp_size": 1, "microbatch_group_size_per_vp_stage": 1},
        [len(tokens) for tokens in train_data["tokens"]],
        global_batch_size=global_batch_size,
        rollout_indices=train_data["rollout_ids"],
    )
    assert len(steps) == 1, "one rollout trains one optimizer step"
    return train_data, partitions[0]


@pytest.mark.unit
def test_slime_trains_the_delayed_samples_first_and_trims_only_the_last_task_rollouts(tmp_path) -> None:
    carried = [float(index) - 11.5 for index in range(24)]
    task_rewards = [[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0] for _ in range(24)]
    args, samples = slime_rollout(tmp_path, carried=carried, group_size=8, batch_size=27, task_rewards=task_rewards)
    train_data, step = train_step(args, samples, 192)

    # 24 delayed rollouts and 192 task rollouts: the step takes the delayed ones and the first 168 plays.
    assert len(set(train_data["rollout_ids"])) == 216
    assert train_data["rewards"][:24] == carried
    assert train_data["rewards"][24:32] == pytest.approx(
        [1.290994, -0.774597, -0.774597] + [1.290994] * 2 + [-0.774597] * 3
    )
    assert step == list(range(24 + 168))


@pytest.mark.unit
@pytest.mark.parametrize("carried", [0, 1, 17, 24])
def test_options_that_pass_the_check_fill_one_step_whatever_the_job_carries(tmp_path, carried: int) -> None:
    # The paper's options: 27 prompts of 8, a step of 192. A first rollout holds at least 26 * 8 + 1 = 209 rollouts.
    check_delayed_task_job(
        Args(rollout_batch_size=27, n_samples_per_prompt=8, global_batch_size=192), CarryingObjective()
    )
    tasks = 27 - delayed_group_count(carried, 8)
    task_rewards = [[1.0, 0.0] * 4 for _ in range(tasks)]
    advantages = [float(index) for index in range(carried)]
    args, samples = slime_rollout(tmp_path, carried=advantages, group_size=8, batch_size=27, task_rewards=task_rewards)

    train_data, _ = train_step(args, samples, 192, dp_size=2, dynamic=True)

    assert train_data["rewards"][:carried] == advantages


@pytest.mark.unit
def test_a_padding_row_adds_no_rollout_to_the_step(tmp_path) -> None:
    # Three delayed samples in two groups of two: the fourth entry pads in its group's first rollout.
    args, samples = slime_rollout(
        tmp_path, carried=[2.0, -2.0, 0.0], group_size=2, batch_size=5, task_rewards=[[1.0, 0.0] for _ in range(3)]
    )
    train_data, step = train_step(args, samples, 7)

    assert train_data["loss_masks"][3] == [0] and train_data["rewards"][3] == 0.0
    assert train_data["rollout_ids"][3] == train_data["rollout_ids"][2]
    # Three delayed rollouts and the first two tasks' plays train; the last task's two plays are trimmed.
    assert step == list(range(4 + 4))
    assert math.isclose(train_data["rewards"][4], 1.0, rel_tol=1e-6) and train_data["rewards"][:3] == [2.0, -2.0, 0.0]


@pytest.mark.unit
def test_tis_weights_the_clipped_loss_of_a_delayed_sample_by_its_clamped_sampling_ratio() -> None:
    torch = pytest.importorskip("torch")
    ppo_utils = pytest.importorskip("slime.utils.ppo_utils", reason="requires the Slime runtime")
    loss = pytest.importorskip("slime.backends.megatron_utils.loss", reason="requires the Slime runtime")
    # The workers' log-probs equal the old ones (p = p_old). The sampler's q gives the ratios 1, e, e^-1.5 and 1.
    log_probs = torch.tensor([-0.5, -1.0, -2.0, -0.1], requires_grad=True)
    old_log_probs = log_probs.detach().clone()
    rollout_log_probs = torch.tensor([-0.5, -2.0, -0.5, -0.1])
    advantages = torch.tensor([1.5, 1.5, -2.0, 0.0])
    pg_loss, _ = ppo_utils.compute_policy_loss(old_log_probs - log_probs, advantages, 0.2, 0.28)
    weighted, _, _ = loss.vanilla_tis_function(
        SimpleNamespace(tis_clip_low=0.0, tis_clip=2.0),
        pg_loss=pg_loss,
        train_log_probs=[old_log_probs],
        rollout_log_probs=[rollout_log_probs],
        loss_masks=[torch.ones(4)],
    )
    weighted.sum().backward()
    # Each token's loss is -A * clamp(p_old / q, 0, 2) * p / p_old, so its gradient in log p is -A * clamp(p_old / q, 0, 2).
    expected = -advantages * torch.exp(old_log_probs - rollout_log_probs).clamp(0.0, 2.0)
    assert torch.allclose(weighted.detach(), expected) and torch.allclose(log_probs.grad, expected)
    assert expected[1] == -3.0, "the ratio e is clamped to 2"
