"""Bring the learning-rate schedule back to where a resumed run left it.

Megatron saves the optimizer-parameter scheduler together with the optimizer,
so a checkpoint written with ``--no-save-optim`` holds neither, and a resume
with ``--no-load-optim`` starts the schedule from step zero: the learning rate
silently returns to its peak and the warmup/decay restarts. The checkpoint does
record its iteration, which is the run's rollout id, so the scheduler is
stepped forward by the samples those rollouts consumed. That is the same
per-rollout estimate Slime sizes the schedule with
(``rollout_batch_size * n_samples_per_prompt``), so the restored point lies on
the configured curve wherever the schedule itself does.

This module imports nothing heavy so it stays testable without Megatron.
"""

from __future__ import annotations

from typing import Any


def scheduler_samples_to_restore(args: Any, start_rollout_id: int, scheduler: Any) -> int:
    """Samples to step ``scheduler`` by after a resume; zero when there is nothing to restore."""
    if scheduler is None or not getattr(args, "no_load_optim", False) or getattr(args, "finetune", False):
        return 0
    if start_rollout_id <= 0 or getattr(scheduler, "num_steps", 0) != 0:
        return 0
    per_rollout = int(getattr(args, "rollout_batch_size", 0) or 0) * int(getattr(args, "n_samples_per_prompt", 1) or 1)
    return start_rollout_id * per_rollout


def restore_scheduler_progress(args: Any, start_rollout_id: int, scheduler: Any) -> int:
    """Step ``scheduler`` to the resumed rollout and return the samples it was advanced by."""
    samples = scheduler_samples_to_restore(args, start_rollout_id, scheduler)
    if samples:
        scheduler.step(increment=samples)
    return samples
