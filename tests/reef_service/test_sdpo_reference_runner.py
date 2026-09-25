"""Keep the reference launcher on the paper's batch and comparator contracts."""

import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from recipes.sdpo.examples.paper.run_reference import build_command, progress_record, run_training


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
        "evaluate_first": False,
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


def test_progress_uses_only_training_duration_from_the_pinned_logger():
    line = "\x1b[36m(TaskRunner pid=7)\x1b[0m step:5 - timing_s/step:120.5 - timing_s/testing:600 - val-core/chemistry/acc/mean@16:0.4"
    assert progress_record(line) == (5, 120.5, True)
    assert progress_record("Training Progress: 50%| 5/10") is None
    assert progress_record("step:5 - actor/loss:0.1") is None
    with pytest.raises(ValueError, match="invalid"):
        progress_record("step:5 - timing_s/step:-1")


def test_time_budget_deduplicates_steps_and_waits_for_validation(tmp_path):
    script = """
import time
print('step:1 - timing_s/step:1800', flush=True)
print('step:1 - timing_s/step:1800', flush=True)
print('step:2 - timing_s/step:2000', flush=True)
print('step:3 - timing_s/step:100 - val-core/chemistry/acc/mean@16:0.5', flush=True)
time.sleep(60)
"""
    run_training(
        [sys.executable, "-c", script], cwd=tmp_path, environment=os.environ.copy(), output=tmp_path, training_hours=1
    )
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["steps"] == 3
    assert progress["training_seconds"] == 3900
    assert progress["stopped_at_budget_boundary"] is True
    assert progress["validation_completed"] is True


def test_reference_failure_is_not_reported_as_budget_completion(tmp_path):
    with pytest.raises(subprocess.CalledProcessError):
        run_training(
            [sys.executable, "-c", "raise SystemExit(3)"],
            cwd=tmp_path,
            environment=os.environ.copy(),
            output=tmp_path,
            training_hours=0,
        )
    assert not (tmp_path / "progress.json").exists()


def test_natural_completion_is_successful_only_for_a_step_budget(tmp_path):
    command = [sys.executable, "-c", "print('step:1 - timing_s/step:1 - val-core/chemistry/acc/mean@16:0.5')"]
    run_training(command, cwd=tmp_path, environment=os.environ.copy(), output=tmp_path, training_hours=0)
    with pytest.raises(RuntimeError, match="ended before"):
        run_training(command, cwd=tmp_path, environment=os.environ.copy(), output=tmp_path, training_hours=1)


def test_untrained_evaluation_is_explicit_and_preserves_the_training_budget():
    values = overrides(arguments(steps=0, evaluate_first=True))
    assert values["trainer.val_before_train"] == "True"
    assert "trainer.total_training_steps" not in values
    assert "trainer.val_before_train" not in overrides(arguments())
