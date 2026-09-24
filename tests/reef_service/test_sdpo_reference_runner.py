"""Keep the reference launcher on the paper's batch and comparator contracts."""

from argparse import Namespace
from pathlib import Path

import pytest

from recipes.sdpo.examples.paper.run_reference import build_command


def arguments(**changes):
    values = {
        "reference": Path("/tmp/sdpo-reference"),
        "output": Path("/tmp/reference-output"),
        "model_path": Path("/tmp/pinned-model"),
        "method": "sdpo",
        "dataset": "chemistry",
        "seed": 42,
        "gpus": 4,
        "minibatch": 32,
        "learning_rate": 1e-5,
        "steps": 2,
    }
    return Namespace(**(values | changes))


def overrides(args):
    return dict(value.split("=", 1) for value in build_command(args)[5:])


def test_sdpo_smoke_retains_paper_sampling_and_distillation_budget():
    command = build_command(arguments())
    assert command[1:5] == ["-m", "verl.trainer.main_ppo", "--config-name", "sdpo"]
    values = overrides(arguments())
    assert values["data.train_batch_size"] == "32"
    assert values["actor_rollout_ref.rollout.n"] == "8"
    assert values["actor_rollout_ref.rollout.val_kwargs.n"] == "16"
    assert values["actor_rollout_ref.actor.self_distillation.distillation_topk"] == "100"
    assert values["actor_rollout_ref.actor.self_distillation.include_environment_feedback"] == "False"
    assert values["trainer.total_training_steps"] == "2"
    assert values["trainer.resume_mode"] == "disable"


def test_full_budget_removes_only_smoke_step_ceiling_and_seeds_both_sampler_and_actor():
    values = overrides(arguments(steps=0, seed=44))
    assert "trainer.total_training_steps" not in values
    assert values["trainer.total_epochs"] == "30"
    assert values["data.seed"] == values["actor_rollout_ref.actor.fsdp_config.seed"] == "44"
    assert values["actor_rollout_ref.actor.ppo_mini_batch_size"] == "32"


@pytest.mark.parametrize("minibatch,learning_rate", [(8, 1e-5), (32, 1e-5), (8, 1e-6), (32, 1e-6)])
def test_grpo_comparators_keep_same_sampling_without_sdpo_overrides(minibatch, learning_rate):
    args = arguments(method="grpo", dataset="tooluse", minibatch=minibatch, learning_rate=learning_rate)
    command = build_command(args)
    assert command[4] == "baseline_grpo"
    values = overrides(args)
    assert values["vars.task"] == "datasets/tooluse"
    assert values["data.train_batch_size"] == "32"
    assert values["actor_rollout_ref.rollout.n"] == "8"
    assert values["actor_rollout_ref.actor.ppo_mini_batch_size"] == str(minibatch)
    assert values["actor_rollout_ref.actor.optim.lr"] == str(learning_rate)
    assert not any("self_distillation" in key for key in values)
