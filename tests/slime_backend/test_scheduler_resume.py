from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from reef.train.slime_backend.reef_adapters.megatron.scheduler_resume import restore_scheduler_progress


class _CosineScheduler:
    """Megatron's scheduler as far as a resume sees it: a sample counter that sets the lr."""

    def __init__(self, max_lr: float, min_lr: float, decay_samples: int) -> None:
        self.max_lr, self.min_lr, self.decay_samples = max_lr, min_lr, decay_samples
        self.num_steps = 0

    def step(self, increment: int) -> None:
        self.num_steps += increment

    @property
    def lr(self) -> float:
        ratio = min(self.num_steps / self.decay_samples, 1.0)
        return self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * ratio))


def resume_args(**overrides):
    values = {"no_load_optim": True, "finetune": False, "rollout_batch_size": 128, "n_samples_per_prompt": 1}
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.unit
def test_a_resume_without_optimizer_state_continues_the_schedule() -> None:
    # gold-run-s: cosine over 60 steps of 128, resumed after 24 of them.
    uninterrupted = _CosineScheduler(2e-6, 2e-7, 60 * 128)
    for _ in range(24):
        uninterrupted.step(128)
    resumed = _CosineScheduler(2e-6, 2e-7, 60 * 128)

    assert restore_scheduler_progress(resume_args(), 24, resumed) == 24 * 128
    assert resumed.lr == pytest.approx(uninterrupted.lr)
    assert resumed.lr < 1.4e-6


@pytest.mark.unit
@pytest.mark.parametrize(
    ("overrides", "start_rollout_id", "loaded_steps"),
    [
        # The optimizer, and with it the scheduler, came from the checkpoint.
        ({"no_load_optim": False}, 24, 0),
        # Fine-tuning starts a new schedule on purpose.
        ({"finetune": True}, 24, 0),
        ({}, 0, 0),
        # Something already set the scheduler; stepping again would double count.
        ({}, 24, 128),
    ],
)
def test_the_schedule_is_left_alone_when_there_is_nothing_to_restore(
    overrides, start_rollout_id, loaded_steps
) -> None:
    scheduler = _CosineScheduler(2e-6, 2e-7, 60 * 128)
    scheduler.num_steps = loaded_steps

    assert restore_scheduler_progress(resume_args(**overrides), start_rollout_id, scheduler) == 0
    assert scheduler.num_steps == loaded_steps
