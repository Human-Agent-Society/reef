"""OpenClaw-RL's Megatron teacher: exact per-candidate log-probs at S^q.

Upstream's topk-select loss REQUIRES a Megatron PRM teacher — its launcher
forces ``OPENCLAW_COMBINE_OPD_TEACHER_SOURCE=megatron`` with the comment that
the inference-side teacher path "does not produce per-cand top-K". The teacher
is the FROZEN BASE model: the same weights the actor starts from, backed up at
init under the ``openclaw_teacher`` tag and swapped in for forward-only passes.

The processor ships candidate token sequences (``teacher_tokens_cand``:
hint-enhanced prompt ids + the native response ids, upstream's
``teacher_tokens_candidates``), never numbers. This module runs upstream's
K-loop — one forward-only pass per candidate slot, cyclically reusing a
sample's candidates up to ``K_max`` (``hint_opd_select`` semantics) — and
fills the three batch keys the loss kernel consumes:

* ``prm_teacher_topk_log_probs_cand``  [K_i, R, K]  teacher log-probs at S^q
* ``prm_teacher_topk_indices_cand``    [K_i, R, K]  S^q broadcast per cand
* ``prm_teacher_native_topk_indices_cand`` [K_i, R, K] the teacher's own
  top-K ids (the overlap-selection signal for ``sequence_optimal``)

Exactness replaces the retired SGLang-prefill approximation: the gather is a
TP-aware full-vocab log-softmax at the requested ids — no top-N window, no
synthetic floor — computed from logits divided by ``rollout_temperature``,
the same scale ``get_log_probs_and_entropy`` puts ``ell_cur``/``ell_old`` on.
The packing schedule and the sharded gathers are the Slime backend's
distillation base (``reef.train.slime_backend.distill``), shared with the
distilling recipes.

The forward itself is slime's own ``forward_only`` (the compute_log_prob
path, byte-identical model execution). S^q rows reach the callback through a
per-pass registry keyed by token-tensor identity, because ``forward_only``
partially applies a fixed kwarg set; the registry holds strong references for
the pass, so ids cannot be reused underneath it.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.distributed as dist
from megatron.core import mpu
from slime.backends.megatron_utils.data import DataIterator
from slime.backends.megatron_utils.model import forward_only

from reef.train.slime_backend.distill.objective import gather_log_probs_at_ids, native_topk_ids
from reef.train.slime_backend.distill.teacher import pack_forward_schedule

logger = logging.getLogger(__name__)

#: Per-pass S^q registry: id(teacher token tensor) -> [R, K] long tensor.
_SQ_BY_TOKENS: dict[int, torch.Tensor] = {}
#: Strong references keeping the registry's keys alive for the pass.
_TOKENS_ALIVE: list[torch.Tensor] = []


def gather_teacher_rows(
    logits: torch.Tensor,
    *,
    args: Any,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
) -> tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:
    """``forward_only`` callback: per sample, teacher rows at S^q.

    Indexing mirrors ``get_log_probs_and_entropy``'s cp1 branch exactly: the
    packed stream concatenates samples by ``total_length``, and the logits
    row predicting response token ``j`` sits at
    ``offset + total_length - response_length - 1 + j``.

    Returns Megatron's legacy 2-tuple ``(loss, reduced)`` the way the
    runtime's own log-prob callback does: an empty loss tensor, and the
    collected rows as the reduced dict. A bare dict would be tuple-unpacked
    into its KEY STRINGS by ``forward_step_calc_loss``.
    """
    if mpu.get_context_parallel_world_size() > 1:
        raise NotImplementedError("the OpenClaw-RL Megatron teacher supports context parallel = 1 only")
    if logits.size(0) != 1:
        raise ValueError(f"teacher logits must have batch size 1, got {logits.shape}")
    tp_group = mpu.get_tensor_model_parallel_group()
    tp_world = dist.get_world_size(group=tp_group) if dist.is_initialized() else 1
    tp_rank = dist.get_rank(group=tp_group) if dist.is_initialized() else 0

    teacher_log_probs: list[torch.Tensor] = []
    teacher_topk_token_ids: list[torch.Tensor] = []
    with torch.no_grad():
        scaled_local_logits = logits.squeeze(0).float()
        temperature = float(getattr(args, "rollout_temperature", 1.0) or 1.0)
        scaled_local_logits = scaled_local_logits / temperature if temperature != 1.0 else scaled_local_logits
        offset = 0
        for tokens, total_length, response_length in zip(
            unconcat_tokens, total_lengths, response_lengths, strict=True
        ):
            rows = scaled_local_logits[offset + total_length - response_length - 1 : offset + total_length - 1]
            sq = _SQ_BY_TOKENS[id(tokens)].to(device=rows.device, dtype=torch.long)
            teacher_log_probs.append(gather_log_probs_at_ids(rows, sq, tp_group, tp_world, tp_rank).cpu())
            teacher_topk_token_ids.append(native_topk_ids(rows, sq.size(-1), tp_group, tp_world, tp_rank).cpu())
            offset += total_length
    return torch.empty((0,), device=logits.device), {
        "lp_at_sq": teacher_log_probs,
        "native_topk": teacher_topk_token_ids,
    }


def compute_openclaw_teacher_cands(actor: Any, rollout_data: dict[str, Any]) -> None:
    """Fill the three ``prm_teacher_*_cand`` keys from the frozen teacher.

    Runs upstream's K-loop against the ``openclaw_teacher`` weights backed up
    at init, then restores the actor's own weights. The actor backup is
    refreshed first so the restore returns the weights training is about to
    use, not the ones init saw.
    """
    # Consumed here, and REMOVED here: the candidate sequences are ragged
    # per-sample lists, and slime's log_rollout_data averages every non-tensor
    # list elementwise — ragged input dies there with int + list. Nothing
    # downstream reads them once the teacher tensors exist.
    cands_per_sample: list[list[list[int]]] = rollout_data.pop("teacher_tokens_cand")
    native_tokens: list[torch.Tensor] = rollout_data["tokens"]
    response_lengths: list[int] = [int(v) for v in rollout_data["response_lengths"]]
    sq_rows: list[torch.Tensor] = rollout_data["topk_indices"]
    if not cands_per_sample or any(not cands for cands in cands_per_sample):
        raise ValueError("every openclawrl sample must carry at least one teacher candidate")

    args = actor.args
    budget = int(getattr(args, "max_tokens_per_gpu", 0) or args.seq_length)
    capacity = int(args.seq_length)
    device = torch.cuda.current_device()
    vpp = mpu.get_virtual_pipeline_model_parallel_world_size() or 1

    # A candidate the model cannot seat falls back to the un-enhanced native
    # sequence (upstream's degenerate K_i=1 anchor), never truncates.
    sanitized: list[list[list[int]]] = []
    for index, cands in enumerate(cands_per_sample):
        native = native_tokens[index].tolist()
        keep: list[list[int]] = []
        for cand in cands:
            if len(cand) > response_lengths[index] and len(cand) <= capacity:
                keep.append([int(v) for v in cand])
            else:
                logger.warning(
                    "teacher candidate for sample %d does not fit (%d tokens, capacity %d); using the anchor",
                    index,
                    len(cand),
                    capacity,
                )
        sanitized.append(keep or [native])
    k_max = max(len(cands) for cands in sanitized)

    collected_lp: list[list[torch.Tensor]] = [[] for _ in sanitized]
    collected_native: list[list[torch.Tensor]] = [[] for _ in sanitized]

    actor.weights_backuper.backup("actor")
    actor._switch_model("openclaw_teacher")
    try:
        for k in range(k_max):
            pass_tokens = [
                torch.as_tensor(cands[k % len(cands)], dtype=torch.long, device=device) for cands in sanitized
            ]
            pass_lengths = [int(t.size(0)) for t in pass_tokens]
            schedule = pack_forward_schedule(pass_lengths, budget)
            view = {
                "tokens": pass_tokens,
                "loss_masks": rollout_data["loss_masks"],
                "total_lengths": pass_lengths,
                "response_lengths": response_lengths,
                "micro_batch_indices": schedule,
            }
            _SQ_BY_TOKENS.clear()
            _TOKENS_ALIVE.clear()
            for tokens, sq in zip(pass_tokens, sq_rows, strict=True):
                _SQ_BY_TOKENS[id(tokens)] = sq
                _TOKENS_ALIVE.append(tokens)
            try:
                result = forward_only(
                    gather_teacher_rows,
                    args,
                    actor.model,
                    [DataIterator(view, schedule) for _ in range(vpp)],
                    [len(schedule)],
                )
            finally:
                _SQ_BY_TOKENS.clear()
                _TOKENS_ALIVE.clear()
            if not result:  # non-last pipeline stages hold no logits
                continue
            for index, (lp, native) in enumerate(zip(result["lp_at_sq"], result["native_topk"], strict=True)):
                if k < len(sanitized[index]):  # cyclic duplicates never ship
                    collected_lp[index].append(lp)
                    collected_native[index].append(native)
    finally:
        actor._switch_model("actor")

    if not any(collected_lp):
        return  # not the last pipeline stage; the loss does not run here
    rollout_data["prm_teacher_topk_log_probs_cand"] = [torch.stack(rows).to(torch.float32) for rows in collected_lp]
    rollout_data["prm_teacher_native_topk_indices_cand"] = [
        torch.stack(rows).to(torch.long) for rows in collected_native
    ]
    rollout_data["prm_teacher_topk_indices_cand"] = [
        sq.to(torch.long).cpu().unsqueeze(0).expand(len(rows), -1, -1).contiguous()
        for sq, rows in zip(sq_rows, collected_lp, strict=True)
    ]


__all__ = [
    "compute_openclaw_teacher_cands",
    "gather_teacher_rows",
    "pack_forward_schedule",
]
