"""Operations over logit rows whose vocabulary is split across the tensor-parallel ranks.

Megatron hands a loss ``[R, V_local]`` logit rows: each tensor-parallel rank
holds one contiguous shard of the vocabulary. Anything that needs the full
distribution (its log-sum-exp, the log-probs at ids that may live on any
shard, the top-K ids) has to reduce across the shards, and its gradient has to
land on the shard that owns each id. These functions take the
tensor-parallel group explicitly and are the identity at world size one, so
CPU tests run them without Megatron. Worker side, torch; the distillation
losses, score centering and the OpenClaw-RL teacher share them.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist


class SumAcrossVocabShards(torch.autograd.Function):
    """Sum per-row partials over the tensor-parallel vocab shards.

    Every rank computes the same total, so the gradient of a replicated loss
    passes through to each rank's partial unchanged.
    """

    @staticmethod
    def forward(ctx: Any, partial: torch.Tensor, tp_group: Any) -> torch.Tensor:
        total = partial.clone()
        dist.all_reduce(total, op=dist.ReduceOp.SUM, group=tp_group)
        return total

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output, None


def sum_across_vocab_shards(partial: torch.Tensor, tp_group: Any, tp_world: int) -> torch.Tensor:
    """``partial`` summed over the vocab shards, differentiable; the identity at world size one."""
    if tp_world <= 1:
        return partial
    return SumAcrossVocabShards.apply(partial, tp_group)


def global_log_sum_exp(logits: torch.Tensor, tp_group: Any, tp_world: int) -> torch.Tensor:
    """log-sum-exp over the full vocabulary of ``[R, V_local]`` logit rows, differentiable."""
    row_max = logits.detach().max(dim=-1).values
    if tp_world > 1:
        dist.all_reduce(row_max, op=dist.ReduceOp.MAX, group=tp_group)
    shard_sum = (logits - row_max[:, None]).exp().sum(dim=-1)
    return row_max + sum_across_vocab_shards(shard_sum, tp_group, tp_world).log()


class GatherAcrossVocabShards(torch.autograd.Function):
    """Rows' values at global vocab ids that may live on any shard.

    Forward: every rank gathers the ids it holds, zeroes the rest, and the
    all-reduce assembles the full rows on every rank. Backward: the gradient
    of the replicated result lands on the owning shard's positions.
    """

    @staticmethod
    def forward(ctx: Any, rows: torch.Tensor, ids: torch.Tensor, tp_group: Any, tp_world: int, tp_rank: int) -> Any:
        v_local = rows.size(-1)
        shard_start = tp_rank * v_local
        in_shard = (ids >= shard_start) & (ids < shard_start + v_local)
        local_ids = (ids - shard_start).clamp(min=0, max=v_local - 1)
        gathered = torch.gather(rows, dim=-1, index=local_ids)
        gathered = torch.where(in_shard, gathered, torch.zeros_like(gathered))
        if tp_world > 1:
            dist.all_reduce(gathered, op=dist.ReduceOp.SUM, group=tp_group)
        ctx.save_for_backward(in_shard, local_ids)
        ctx.rows_shape = rows.shape
        return gathered

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None, None, None]:
        in_shard, local_ids = ctx.saved_tensors
        grad_rows = torch.zeros(ctx.rows_shape, dtype=grad_output.dtype, device=grad_output.device)
        grad_rows.scatter_add_(-1, local_ids, torch.where(in_shard, grad_output, torch.zeros_like(grad_output)))
        return grad_rows, None, None, None, None


def gather_log_probs_at_ids(
    logits: torch.Tensor, ids: torch.Tensor, tp_group: Any, tp_world: int, tp_rank: int
) -> torch.Tensor:
    """Log-probs over the full vocabulary at global ``ids`` (``[R, K]``) of ``[R, V_local]`` logit rows.

    Differentiable and replicated: every rank returns the same values, so a
    loss built on them must be computed identically on every rank.
    """
    logits = logits.to(torch.promote_types(logits.dtype, torch.float32))
    raw = GatherAcrossVocabShards.apply(logits, ids, tp_group, tp_world, tp_rank)
    return raw - global_log_sum_exp(logits, tp_group, tp_world)[:, None]


def native_topk_ids(rows: torch.Tensor, k: int, tp_group: Any, tp_world: int, tp_rank: int) -> torch.Tensor:
    """The global ids of the ``k`` largest values per row of ``[R, V_local]`` rows."""
    v_local = rows.size(-1)
    local_values, local_index = torch.topk(rows, k=min(k, v_local), dim=-1)
    local_ids = local_index + tp_rank * v_local
    if tp_world > 1:
        values_per_rank = [torch.empty_like(local_values) for _ in range(tp_world)]
        ids_per_rank = [torch.empty_like(local_ids) for _ in range(tp_world)]
        dist.all_gather(values_per_rank, local_values.contiguous(), group=tp_group)
        dist.all_gather(ids_per_rank, local_ids.contiguous(), group=tp_group)
        values = torch.cat(values_per_rank, dim=-1)
        ids = torch.cat(ids_per_rank, dim=-1)
    else:
        values, ids = local_values, local_ids
    _, best = torch.topk(values, k=k, dim=-1)
    return torch.gather(ids, dim=-1, index=best)


__all__ = ["gather_log_probs_at_ids", "global_log_sum_exp", "native_topk_ids", "sum_across_vocab_shards"]
