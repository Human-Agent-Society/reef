"""The teacher of a distillation family: whose weights it runs on, and the forward-only pass that scores the samples.

The weights are the student's own (the current ones, or a slow-moving copy
of them: the SDFT reference's ``ref_model`` that moves toward the policy by
the update rate after every step, a frozen snapshot at rate 0) or another
model's, loaded once from its checkpoint. They live in the actor's weight
backups (Slime's ``TensorBackuper``: bfloat16 tensors pinned on the host,
restored into the model by tag) beside the actor's own copy: a pass backs
the actor up, switches the teacher in, runs, and switches the actor back,
so the trained weights are what the step starts from. Loading and seeding
happen lazily in the first pre-train hook rather than at actor init: a
colocated actor sleeps between the two with its process groups retired,
and the backup walks the model through them.

The pass is Slime's own ``forward_only`` over the teacher sequences the
processor built (the teacher's prompt ids followed by the student's response
ids verbatim), packed under the token budget. Its callback keeps, per sample,
what the loss needs at every response position:

- the exact representation (``top_k == 0``): the teacher's log-probs over
  this rank's vocab shard, ``[R, V_local]`` rows normalized over the full
  vocabulary, in float16 on the host until the loss moves a micro-batch's
  rows back to the device;
- the top-K representation: the teacher's own top-K ids and its log-probs
  at them, ``[R, K]``, plus its log-prob at the sampled token, ``[R]``.

Indexing mirrors ``get_log_probs_and_entropy``'s cp1 branch: the packed
stream concatenates samples by ``total_length``, and the logits row
predicting response token ``j`` sits at
``offset + total_length - response_length - 1 + j``. Logits are divided by
``rollout_temperature`` as the loss divides the student's. Megatron and
Slime are imported where the pass runs, so the weights' lifecycle is
testable with CPU torch alone.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import torch
import torch.distributed as dist

from reef.train.slime_backend.distill.algorithm import DistillSettings, settings_from_args
from reef.train.slime_backend.distill.objective import gather_log_probs_at_ids, global_log_sum_exp, native_topk_ids

#: The actor's backup tags: Slime's own copy of the training weights, and the teacher's.
ACTOR_TAG = "actor"
TEACHER_TAG = "distill_teacher"


def mix_teacher_weights(
    teacher: Mapping[str, torch.Tensor], actor: Mapping[str, torch.Tensor], update_rate: float
) -> None:
    """Move the teacher's floating-point tensors toward the actor's: ``teacher = (1 - rate) * teacher + rate * actor``.

    The SDFT reference's ``ref_model_mixup_alpha`` update, applied in place.
    The teacher tensors are the higher-precision accumulator (the actor's may
    be bfloat16); integer buffers are left as they are.
    """
    if not 0 <= update_rate <= 1:
        raise ValueError(f"the teacher update rate must be in [0, 1], got {update_rate}")
    for name, target in teacher.items():
        if not target.is_floating_point():
            continue
        target.mul_(1.0 - update_rate).add_(actor[name].to(dtype=target.dtype), alpha=update_rate)


class TeacherWeights(ABC):
    """Which weights score the teacher sequences, and how they get into the model for the pass."""

    @abstractmethod
    def switch_in(self, actor: Any) -> None:
        """Put the teacher's weights into the model, keeping the actor's for :meth:`switch_out`."""

    @abstractmethod
    def switch_out(self, actor: Any) -> None:
        """Put the actor's weights back."""


class CurrentWeights(TeacherWeights):
    """The student's current weights: the pass runs on the model as it is."""

    def switch_in(self, actor: Any) -> None:
        return None

    def switch_out(self, actor: Any) -> None:
        return None


class MovingCopy(TeacherWeights):
    """A copy of the weights that moves toward the policy by ``update_rate`` before every pass.

    The copy is seeded from the actor's weights before its first pass: on a
    fresh start the base model's, the reference's starting teacher; on a
    restart from a Megatron checkpoint the resumed weights, since the copy
    an interrupted run had moved is not checkpointed. It lives in the backups
    under ``TEACHER_TAG`` with a float32 accumulator beside it, so the small
    updates the reference applies do not vanish in bfloat16 rounding; a rate
    of 0 keeps the seed as it is.
    """

    def __init__(self, update_rate: float) -> None:
        if not 0 <= update_rate < 1:
            raise ValueError(f"a moving copy needs an update rate in [0, 1), got {update_rate}")
        self._update_rate = update_rate
        self._accumulator: dict[str, torch.Tensor] = {}

    def switch_in(self, actor: Any) -> None:
        backuper = actor.weights_backuper
        if TEACHER_TAG not in backuper.backup_tags:
            backuper.backup(TEACHER_TAG)
            if self._update_rate > 0.0:
                for name, tensor in backuper.get(TEACHER_TAG).items():
                    self._accumulator[name] = tensor.float() if tensor.is_floating_point() else tensor.clone()
        backuper.backup(ACTOR_TAG)
        if self._update_rate > 0.0:
            mix_teacher_weights(self._accumulator, backuper.get(ACTOR_TAG), self._update_rate)
            teacher_backup = backuper.get(TEACHER_TAG)
            for name, accumulated in self._accumulator.items():
                teacher_backup[name].copy_(accumulated)
        actor._switch_model(TEACHER_TAG)

    def switch_out(self, actor: Any) -> None:
        actor._switch_model(ACTOR_TAG)


class SeparateCheckpoint(TeacherWeights):
    """Another model's weights, loaded once from its checkpoint beside the actor's.

    The checkpoint (Megatron, or a Hugging Face directory the bridge loads)
    must fit the actor's model and parallel layout: the pass runs it through
    the actor's own modules. Slime's tagged loader puts it into the model and
    backs it up under ``TEACHER_TAG``; the actor's backup taken just before
    restores the trained weights.
    """

    def __init__(self, checkpoint: str) -> None:
        if not checkpoint.strip():
            raise ValueError("a separate teacher needs its checkpoint path")
        self._checkpoint = checkpoint

    def switch_in(self, actor: Any) -> None:
        backuper = actor.weights_backuper
        backuper.backup(ACTOR_TAG)
        if TEACHER_TAG not in backuper.backup_tags:
            actor.load_other_checkpoint(TEACHER_TAG, self._checkpoint)
        else:
            actor._switch_model(TEACHER_TAG)

    def switch_out(self, actor: Any) -> None:
        actor._switch_model(ACTOR_TAG)


def teacher_weights(settings: DistillSettings) -> TeacherWeights:
    """The weights the settings' teacher runs on."""
    if settings.teacher == "separate":
        return SeparateCheckpoint(settings.teacher_checkpoint)
    if settings.teacher_update_rate >= 1.0:
        return CurrentWeights()
    return MovingCopy(settings.teacher_update_rate)


#: Log-probs are stored in float16; the floor keeps every stored value inside
#: its range while ``exp`` of it is still exactly zero in float32.
TEACHER_LOG_PROB_FLOOR = -1.0e4
#: The teacher of this process: built from the settings on the first pass.
_TEACHER: TeacherWeights | None = None


def pack_forward_schedule(lengths: list[int], budget: int) -> list[list[int]]:
    """Greedy contiguous packing of sample indices under a token budget.

    Mirrors dynamic batching's invariant (per-microbatch token sum stays
    under ``max_tokens_per_gpu``); order is preserved and ``forward_only``
    unpermutes by these indices afterwards. A single sample over budget gets
    its own microbatch; the caller guards the model's sequence capacity.
    """
    schedule: list[list[int]] = []
    current: list[int] = []
    used = 0
    for index, length in enumerate(lengths):
        if current and used + length > budget:
            schedule.append(current)
            current, used = [], 0
        current.append(index)
        used += length
    if current:
        schedule.append(current)
    return schedule


def _response_rows(logits: torch.Tensor, args: Any, total_lengths: list[int], response_lengths: list[int]) -> Any:
    """Per sample, the tempered logit rows predicting its response tokens."""
    from megatron.core import mpu

    if mpu.get_context_parallel_world_size() > 1:
        raise NotImplementedError("the distill teacher pass supports context parallel = 1 only")
    if logits.size(0) != 1:
        raise ValueError(f"teacher logits must have batch size 1, got {logits.shape}")
    tempered = logits.squeeze(0).float()
    temperature = float(args.rollout_temperature)
    if temperature != 1.0:
        tempered = tempered / temperature
    offset = 0
    for total_length, response_length in zip(total_lengths, response_lengths, strict=True):
        yield tempered[offset + total_length - response_length - 1 : offset + total_length - 1]
        offset += total_length


def gather_teacher_log_probs(
    logits: torch.Tensor,
    *,
    args: Any,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
) -> tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:
    """``forward_only`` callback of the exact representation: per sample, the teacher's rows of this vocab shard.

    Rows are normalized over the full vocabulary in chunks of
    ``log_probs_chunk_size``. Returns Megatron's legacy 2-tuple ``(loss,
    reduced)`` the way the runtime's own log-prob callback does: an empty
    loss tensor, and the collected rows as the reduced dict.
    """
    from megatron.core import mpu

    tp_group = mpu.get_tensor_model_parallel_group()
    tp_world = dist.get_world_size(group=tp_group) if dist.is_initialized() else 1
    chunk_size = int(args.log_probs_chunk_size)
    rows_per_sample: list[torch.Tensor] = []
    with torch.no_grad():
        for rows in _response_rows(logits, args, total_lengths, response_lengths):
            stored: list[torch.Tensor] = []
            step = rows.size(0) if chunk_size <= 0 else chunk_size
            for start in range(0, rows.size(0), step):
                chunk = rows[start : start + step]
                log_probs = chunk - global_log_sum_exp(chunk, tp_group, tp_world)[:, None]
                stored.append(log_probs.clamp_min(TEACHER_LOG_PROB_FLOOR).to(torch.float16).cpu())
            rows_per_sample.append(torch.cat(stored, dim=0))
    return torch.empty((0,), device=logits.device), {"distill_teacher_log_probs": rows_per_sample}


def gather_teacher_topk(
    logits: torch.Tensor,
    *,
    args: Any,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
) -> tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:
    """``forward_only`` callback of the top-K representation: the teacher's top-K and its sampled-token log-prob."""
    from megatron.core import mpu

    tp_group = mpu.get_tensor_model_parallel_group()
    tp_world = dist.get_world_size(group=tp_group) if dist.is_initialized() else 1
    tp_rank = dist.get_rank(group=tp_group) if dist.is_initialized() else 0
    top_k = int(args.distill_top_k)
    topk_ids: list[torch.Tensor] = []
    topk_log_probs: list[torch.Tensor] = []
    sampled_log_probs: list[torch.Tensor] = []
    with torch.no_grad():
        for rows, tokens, response_length in zip(
            _response_rows(logits, args, total_lengths, response_lengths),
            unconcat_tokens,
            response_lengths,
            strict=True,
        ):
            ids = native_topk_ids(rows, top_k, tp_group, tp_world, tp_rank)
            sampled = tokens[-response_length:].to(device=rows.device, dtype=torch.long)
            at_ids = gather_log_probs_at_ids(
                rows, torch.cat([ids, sampled[:, None]], dim=-1), tp_group, tp_world, tp_rank
            )
            topk_ids.append(ids.cpu())
            topk_log_probs.append(at_ids[:, :-1].cpu())
            sampled_log_probs.append(at_ids[:, -1].cpu())
    return torch.empty((0,), device=logits.device), {
        "distill_teacher_topk_ids": topk_ids,
        "distill_teacher_topk_log_probs": topk_log_probs,
        "distill_teacher_sampled_log_probs": sampled_log_probs,
    }


def compute_teacher_rows(actor: Any, rollout_data: dict[str, Any], settings: DistillSettings | None = None) -> None:
    """Score every sample's teacher sequence with the teacher and fill the batch's ``distill_teacher_*`` keys.

    The teacher's weights (:mod:`.weights`) are switched in for the pass and
    the actor's switched back afterwards; a copy that moves toward the
    policy moves before the pass.
    """
    from megatron.core import mpu
    from slime.backends.megatron_utils.data import DataIterator
    from slime.backends.megatron_utils.model import forward_only

    global _TEACHER
    args = actor.args
    if settings is None:
        settings = settings_from_args(args)
    teacher_tokens: list[torch.Tensor] = rollout_data["teacher_tokens"]
    response_lengths = [int(value) for value in rollout_data["response_lengths"]]
    capacity = int(args.seq_length)
    lengths = [int(tokens.numel()) for tokens in teacher_tokens]
    for index, (length, response_length) in enumerate(zip(lengths, response_lengths, strict=True)):
        if length <= response_length:
            raise ValueError(f"distill sample {index} teacher sequence carries no prompt before its response")
        if length > capacity:
            raise ValueError(
                f"distill sample {index} teacher sequence is {length} tokens, over the trainer's --seq-length "
                f"{capacity}; set the recipe's max_teacher_tokens so such reports are skipped"
            )
    budget = int(args.max_tokens_per_gpu or capacity)
    device = torch.cuda.current_device()
    vpp = mpu.get_virtual_pipeline_model_parallel_world_size() or 1

    pass_tokens = [tokens.to(device=device, dtype=torch.long) for tokens in teacher_tokens]
    schedule = pack_forward_schedule(lengths, budget)
    view = {
        "tokens": pass_tokens,
        "loss_masks": rollout_data["loss_masks"],
        "total_lengths": lengths,
        "response_lengths": response_lengths,
        "micro_batch_indices": schedule,
    }
    if _TEACHER is None:
        _TEACHER = teacher_weights(settings)
    callback = gather_teacher_log_probs if settings.exact else gather_teacher_topk
    _TEACHER.switch_in(actor)
    try:
        result = forward_only(
            callback,
            args,
            actor.model,
            [DataIterator(view, schedule) for _ in range(vpp)],
            [len(schedule)],
        )
    finally:
        _TEACHER.switch_out(actor)
    if not result:
        return  # not the last pipeline stage; the loss does not run here
    rollout_data.update(result)


__all__ = [
    "ACTOR_TAG",
    "TEACHER_LOG_PROB_FLOOR",
    "TEACHER_TAG",
    "CurrentWeights",
    "MovingCopy",
    "SeparateCheckpoint",
    "TeacherWeights",
    "compute_teacher_rows",
    "gather_teacher_log_probs",
    "gather_teacher_topk",
    "mix_teacher_weights",
    "pack_forward_schedule",
    "teacher_weights",
]
