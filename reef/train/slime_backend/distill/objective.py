"""The worker side of a distillation family: the divergences over the vocabulary shards, the loss and the teacher hook.

A recipe's ``objective.py`` forwards its ``@objective``-registered entry
points to :func:`distill_loss` and :func:`distill_actor_pre_train`. The
loss follows ``distil_trainer.py`` of the SDFT reference
(idanshen/Self-Distillation at ``d77573212fa0``): at every response
position of the student's on-policy sample, the divergence between the
teacher's next-token distribution and the student's; the per-sample mean
over the trained response tokens, weighted by the truncated
importance-sampling ratio against the rollout engine's log-probs (averaged
over the response, or applied per token as the SDPO reference does) and by
the sample's own weight from the wire row.

Two representations of the teacher meet the student here. In the exact
representation the teacher's whole next-token distribution is known at every
response position, as ``[R, V_local]`` rows of this rank's shard normalized
over the full vocabulary, and :func:`token_divergence` puts the student's
logit rows against them. In the top-K representation the teacher kept its
log-probs at K ids per position (its own top-K, or the current student's)
and at the sampled token; the student's log-probs at those ids are gathered
across the shards (:func:`gather_log_probs_at_ids`) and
:func:`restricted_divergence` compares the two distributions on them, either
renormalized over the K ids, the reverse KL then estimated at the sampled
token (:func:`sampled_reverse_kl`), or with one bucket for the rest of the
vocabulary (:func:`append_tail_log_probability`).

A divergence is a sum of per-shard terms that all depend on the student's
global log-sum-exp. Letting autograd differentiate a shard's term alone
drops the other shards' dependence on that log-sum-exp, so the kernels
either assemble the divergence from shard sums whose gradient autograd
gets right (the forward KL, linear in the logits) or write the gradient
out (:class:`_ExplicitGradientDivergence`). Every kernel takes the
tensor-parallel group explicitly and reduces across the vocab shards
itself, so the CPU parity tests run it at world size one against the
pure-Python oracle in ``tests/reef_service/reference_algorithms/distill.py``.
Megatron and Slime are imported where a hook runs.
"""

from __future__ import annotations

import math
from argparse import Namespace
from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from reef.train.slime_backend.distill.algorithm import DIVERGENCES, DistillSettings, settings_from_args


class _SumAcrossVocabShards(torch.autograd.Function):
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
    return _SumAcrossVocabShards.apply(partial, tp_group)


def global_log_sum_exp(logits: torch.Tensor, tp_group: Any, tp_world: int) -> torch.Tensor:
    """log-sum-exp over the full vocabulary of ``[R, V_local]`` logit rows, differentiable."""
    row_max = logits.detach().max(dim=-1).values
    if tp_world > 1:
        dist.all_reduce(row_max, op=dist.ReduceOp.MAX, group=tp_group)
    shard_sum = (logits - row_max[:, None]).exp().sum(dim=-1)
    return row_max + sum_across_vocab_shards(shard_sum, tp_group, tp_world).log()


class _GatherAcrossVocabShards(torch.autograd.Function):
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
    raw = _GatherAcrossVocabShards.apply(logits, ids, tp_group, tp_world, tp_rank)
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


def token_divergence(
    student_logits: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    *,
    divergence: str,
    jsd_beta: float = 0.5,
    tp_group: Any,
    tp_world: int,
) -> torch.Tensor:
    """Per-position divergence over the full vocabulary between the teacher and the student.

    ``student_logits`` and ``teacher_log_probs`` are ``[R, V_local]`` rows of
    one vocab shard at the same response positions; the teacher rows are
    already normalized over the full vocabulary. ``forward`` is
    KL(teacher || student), ``reverse`` is KL(student || teacher) and ``jsd``
    the generalized Jensen-Shannon divergence with ``jsd_beta`` the teacher's
    weight in the mixture. Returns one value per row.

    The forward KL is assembled from three shard sums, ``sum p_T log p_T``,
    ``sum p_T z_S`` and the teacher's total mass, plus the student's global
    log-sum-exp; the mass is one up to the storage precision of the teacher
    rows, so its gradient is the exact ``p_S - p_T``.
    """
    if divergence not in DIVERGENCES:
        raise ValueError(f"distill divergence must be one of {', '.join(DIVERGENCES)}, got {divergence!r}")
    student_logits = student_logits.to(torch.promote_types(student_logits.dtype, torch.float32))
    teacher_log_probs = teacher_log_probs.to(torch.promote_types(teacher_log_probs.dtype, torch.float32))
    log_sum_exp = global_log_sum_exp(student_logits, tp_group, tp_world)
    if divergence == "forward":
        teacher_probs = teacher_log_probs.exp()
        teacher_mass = sum_across_vocab_shards(teacher_probs.sum(dim=-1), tp_group, tp_world)
        cross = sum_across_vocab_shards((teacher_probs * student_logits).sum(dim=-1), tp_group, tp_world)
        negative_entropy = sum_across_vocab_shards((teacher_probs * teacher_log_probs).sum(dim=-1), tp_group, tp_world)
        return negative_entropy - cross + teacher_mass * log_sum_exp
    return _ExplicitGradientDivergence.apply(
        student_logits, teacher_log_probs, log_sum_exp.detach(), divergence, jsd_beta, tp_group, tp_world
    )


class _ExplicitGradientDivergence(torch.autograd.Function):
    """A divergence ``sum_v f_v(p_v)`` of the student's probabilities, with its gradient written out.

    For any such sum the gradient on the student's logit ``u`` is
    ``p_u (g_u - sum_v p_v g_v)`` with ``g_v = f_v'(p_v)``: the local factor
    and the all-reduced ``sum_v p_v g_v`` are all a shard needs. The reverse
    KL has ``g_v = log p_v - log t_v`` (its sum is the KL itself); the
    generalized JSD with mixture ``m = beta t + (1 - beta) p`` has
    ``g_v = (1 - beta) (log p_v - log m_v)``.
    """

    @staticmethod
    def forward(
        ctx: Any,
        student_logits: torch.Tensor,
        teacher_log_probs: torch.Tensor,
        log_sum_exp: torch.Tensor,
        divergence: str,
        jsd_beta: float,
        tp_group: Any,
        tp_world: int,
    ) -> torch.Tensor:
        student_log_probs = student_logits - log_sum_exp[:, None]
        probs = student_log_probs.exp()
        if divergence == "reverse":
            factor = student_log_probs - teacher_log_probs
            terms = probs * factor
        else:
            mixture_log_probs = torch.logaddexp(
                teacher_log_probs + math.log(jsd_beta), student_log_probs + math.log1p(-jsd_beta)
            )
            factor = (1.0 - jsd_beta) * (student_log_probs - mixture_log_probs)
            terms = jsd_beta * teacher_log_probs.exp() * (teacher_log_probs - mixture_log_probs) + probs * (
                student_log_probs - mixture_log_probs
            ) * (1.0 - jsd_beta)
        totals = torch.stack([terms.sum(dim=-1), (probs * factor).sum(dim=-1)])
        if tp_world > 1:
            dist.all_reduce(totals, op=dist.ReduceOp.SUM, group=tp_group)
        ctx.save_for_backward(probs, factor, totals[1])
        return totals[0]

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None, None, None, None, None]:
        probs, factor, weighted_factor = ctx.saved_tensors
        grad = grad_output[:, None] * probs * (factor - weighted_factor[:, None])
        return grad, None, None, None, None, None, None


def chunked_token_divergence(
    student_logits: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    *,
    divergence: str,
    jsd_beta: float = 0.5,
    chunk_size: int,
    tp_group: Any,
    tp_world: int,
) -> torch.Tensor:
    """:func:`token_divergence` over row chunks, each recomputed in backward.

    A chunk's full-vocabulary intermediates (the teacher's probabilities, the
    student's exponentials) are the size of the logits themselves; recomputing
    them in backward keeps only the chunk inputs resident, the way Slime
    chunks its own log-prob and entropy computation. ``chunk_size <= 0``
    computes the rows in one piece.
    """
    rows = student_logits.size(0)
    if chunk_size <= 0 or rows <= chunk_size:
        return token_divergence(
            student_logits,
            teacher_log_probs,
            divergence=divergence,
            jsd_beta=jsd_beta,
            tp_group=tp_group,
            tp_world=tp_world,
        )

    def _chunk(student_rows: torch.Tensor, teacher_rows: torch.Tensor) -> torch.Tensor:
        return token_divergence(
            student_rows, teacher_rows, divergence=divergence, jsd_beta=jsd_beta, tp_group=tp_group, tp_world=tp_world
        )

    pieces = [
        checkpoint(
            _chunk,
            student_logits[start : start + chunk_size],
            teacher_log_probs[start : start + chunk_size],
            use_reentrant=False,
        )
        for start in range(0, rows, chunk_size)
    ]
    return torch.cat(pieces, dim=0)


def restricted_divergence(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    *,
    divergence: str,
    jsd_beta: float = 0.5,
    distribution: str = "renormalized",
) -> torch.Tensor:
    """Per-position divergence at the same selected vocabulary ids.

    ``student_log_probs`` and ``teacher_log_probs`` are ``[R, K]`` log-probs
    over the full vocabulary at the same K ids. ``renormalized`` conditions
    each distribution on those ids; ``tail`` keeps their original mass and
    appends one bucket for the rest of the vocabulary. Both inputs are replicated across the
    tensor-parallel ranks, so autograd's gradient is the right one.
    """
    if divergence not in DIVERGENCES:
        raise ValueError(f"distill divergence must be one of {', '.join(DIVERGENCES)}, got {divergence!r}")
    student = student_log_probs.to(torch.promote_types(student_log_probs.dtype, torch.float32))
    teacher = teacher_log_probs.to(torch.promote_types(teacher_log_probs.dtype, torch.float32))
    if distribution == "tail":
        student, teacher = append_tail_log_probability(student), append_tail_log_probability(teacher)
    elif distribution == "renormalized":
        student, teacher = torch.log_softmax(student, -1), torch.log_softmax(teacher, -1)
    else:
        raise ValueError("top-K distribution must be renormalized or tail")
    if divergence == "forward":
        return (teacher.exp() * (teacher - student)).sum(dim=-1)
    if divergence == "reverse":
        return (student.exp() * (student - teacher)).sum(dim=-1)
    mixture = torch.logaddexp(teacher + math.log(jsd_beta), student + math.log1p(-jsd_beta))
    return jsd_beta * (teacher.exp() * (teacher - mixture)).sum(dim=-1) + (1.0 - jsd_beta) * (
        student.exp() * (student - mixture)
    ).sum(dim=-1)


def append_tail_log_probability(log_probs: torch.Tensor) -> torch.Tensor:
    """Add the probability outside selected vocabulary ids, as in SDPO at 7c457fc1b1f6.

    The clamp prevents log(0) when the selected entries exhaust the vocabulary;
    expm1 retains precision when their mass is close to one.
    """
    log_mass = torch.logsumexp(log_probs, dim=-1, keepdim=True).clamp(max=-1e-7)
    tail = torch.log(-torch.expm1(log_mass))
    return torch.cat((log_probs, tail), dim=-1)


def token_importance_weights(
    student_log_probs: torch.Tensor, rollout_log_probs: torch.Tensor, cap: float
) -> torch.Tensor:
    """Detached, capped per-token correction from the sampling engine to the current policy."""
    log_ratio = (student_log_probs - rollout_log_probs).detach().clamp(min=-20.0, max=20.0)
    return log_ratio.to(torch.promote_types(log_ratio.dtype, torch.float32)).exp().clamp(max=cap)


def sampled_reverse_kl(student_log_prob: torch.Tensor, teacher_log_prob: torch.Tensor) -> torch.Tensor:
    """KL(student || teacher) estimated at the sampled token, with the score-function gradient.

    ``student_log_prob`` and ``teacher_log_prob`` are ``[R]`` log-probs of the
    tokens the student sampled. The estimate is ``log p(y) - log t(y)``; its
    gradient is that gap, held fixed, times the gradient of ``log p(y)``, the
    unbiased estimator of the gradient of the expectation over the student's
    samples (the on-policy distillation loss of tinker-cookbook).
    """
    gap = (student_log_prob - teacher_log_prob).detach()
    return gap + gap * (student_log_prob - student_log_prob.detach())


def sequence_importance_weight(
    student_log_probs: torch.Tensor,
    rollout_log_probs: torch.Tensor,
    loss_mask: torch.Tensor,
    cap: float,
) -> torch.Tensor:
    """The truncated importance-sampling weight of one sample.

    Per token, ``min(pi_theta(y_t) / pi_rollout(y_t), cap)``; the weight is
    its mean over the trained response tokens. It corrects the mismatch
    between the rollout engine and the trainer (and, on Reef, any admitted
    staleness), the way TRL's ``vllm_importance_sampling_correction`` and the
    SDFT reference do.
    """
    log_ratio = student_log_probs - rollout_log_probs
    ratio = log_ratio.to(torch.promote_types(log_ratio.dtype, torch.float32)).exp().clamp(max=cap)
    mask = loss_mask.to(ratio.dtype)
    return (ratio * mask).sum() / mask.sum().clamp(min=1.0)


def _exact_divergences(
    settings: DistillSettings, args: Namespace, batch: dict[str, Any], student_rows_per_sample: Any
) -> list[torch.Tensor]:
    """Per sample, the per-token divergence against the teacher's whole distribution."""
    from megatron.core import mpu

    teacher_rows = batch.get("distill_teacher_log_probs")
    if teacher_rows is None:
        raise RuntimeError("distill_teacher_log_probs is missing: the pre-train hook did not score this batch")
    tp_group = mpu.get_tensor_model_parallel_group()
    tp_world = dist.get_world_size(group=tp_group) if dist.is_initialized() else 1
    chunk_size = int(args.log_probs_chunk_size)
    per_sample: list[torch.Tensor] = []
    for index, ((student_rows, _), teacher) in enumerate(zip(student_rows_per_sample, teacher_rows, strict=True)):
        if teacher.size(0) != student_rows.size(0):
            raise ValueError(
                f"distill sample {index} has {teacher.size(0)} teacher rows for a {student_rows.size(0)}-token response"
            )
        per_sample.append(
            chunked_token_divergence(
                student_rows,
                teacher.to(device=student_rows.device, non_blocking=True),
                divergence=settings.divergence,
                jsd_beta=settings.jsd_beta,
                chunk_size=chunk_size,
                tp_group=tp_group,
                tp_world=tp_world,
            )
        )
    return per_sample


def _topk_divergences(
    settings: DistillSettings, batch: dict[str, Any], student_rows_per_sample: Any
) -> list[torch.Tensor]:
    """Per sample, the per-token divergence on the selected top-K ids, or the reverse KL at the sampled token."""
    from megatron.core import mpu

    for key in ("distill_teacher_topk_ids", "distill_teacher_topk_log_probs", "distill_teacher_sampled_log_probs"):
        if batch.get(key) is None:
            raise RuntimeError(f"{key} is missing: the pre-train hook did not score this batch")
    tp_group = mpu.get_tensor_model_parallel_group()
    tp_world = dist.get_world_size(group=tp_group) if dist.is_initialized() else 1
    tp_rank = dist.get_rank(group=tp_group) if dist.is_initialized() else 0
    per_sample: list[torch.Tensor] = []
    rows_and_teacher = zip(
        student_rows_per_sample,
        batch["distill_teacher_topk_ids"],
        batch["distill_teacher_topk_log_probs"],
        batch["distill_teacher_sampled_log_probs"],
        strict=True,
    )
    for index, ((student_rows, sampled_tokens), ids, teacher_at_ids, teacher_at_sampled) in enumerate(
        rows_and_teacher
    ):
        if ids.size(0) != student_rows.size(0):
            raise ValueError(
                f"distill sample {index} has {ids.size(0)} teacher rows for a {student_rows.size(0)}-token response"
            )
        device = student_rows.device
        ids = ids.to(device=device, dtype=torch.long)
        sampled = sampled_tokens.to(device=device, dtype=torch.long)[:, None]
        student_at = gather_log_probs_at_ids(
            student_rows, torch.cat([ids, sampled], dim=-1), tp_group, tp_world, tp_rank
        )
        if settings.divergence == "reverse" and settings.top_k_distribution == "renormalized":
            per_sample.append(sampled_reverse_kl(student_at[:, -1], teacher_at_sampled.to(device=device).float()))
        else:
            per_sample.append(
                restricted_divergence(
                    student_at[:, :-1],
                    teacher_at_ids.to(device=device),
                    divergence=settings.divergence,
                    jsd_beta=settings.jsd_beta,
                    distribution=settings.top_k_distribution,
                )
            )
    return per_sample


def distill_loss(
    args: Namespace,
    batch: dict[str, Any],
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """``--custom-loss-function-path`` entry point: the per-sample mean token divergence to the teacher.

    ``batch`` carries the ``distill_teacher_*`` keys the pre-train hook
    filled and the wire row's ``distill_sample_weights``; ``logits`` is the
    training forward over the plain request. Slime's outer ``loss_function``
    divides the returned sum of weighted per-sample means by the step's
    global batch size, which yields the batch mean. The metrics stay
    unweighted: they describe every sample the step saw.
    """
    from megatron.core import mpu
    from slime.backends.megatron_utils.cp_utils import get_sum_of_sample_mean
    from slime.backends.megatron_utils.loss import get_log_probs_and_entropy, get_responses

    if mpu.get_context_parallel_world_size() > 1:
        raise NotImplementedError("the distill loss supports context parallel = 1 only")
    settings = settings_from_args(args)
    total_lengths = batch["total_lengths"]
    response_lengths = batch["response_lengths"]
    unconcat_tokens = batch["unconcat_tokens"]
    student_rows_per_sample = list(
        get_responses(
            logits,
            args=args,
            unconcat_tokens=unconcat_tokens,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
        )
    )
    if settings.exact:
        per_sample_divergence = _exact_divergences(settings, args, batch, student_rows_per_sample)
    else:
        per_sample_divergence = _topk_divergences(settings, batch, student_rows_per_sample)

    # The SDFT reference leaves the first response tokens out of the loss and
    # of its per-sample denominator.
    loss_masks = batch["loss_masks"]
    if settings.skip_response_tokens > 0:
        loss_masks = [mask.clone() for mask in loss_masks]
        for mask in loss_masks:
            mask[: settings.skip_response_tokens] = 0
        sum_of_sample_mean = get_sum_of_sample_mean(
            total_lengths, response_lengths, loss_masks, None, args.calculate_per_token_loss
        )

    sample_weights = batch.get("distill_sample_weights")
    if sample_weights is None:
        raise RuntimeError("distill_sample_weights is missing: the driver did not build this batch from the wire row")
    divergence = torch.cat(per_sample_divergence, dim=0)
    scaled = per_sample_divergence
    metrics: dict[str, torch.Tensor] = {}
    if settings.importance_sampling_cap > 0:
        rollout_log_probs = batch.get("rollout_log_probs")
        if rollout_log_probs is None:
            raise ValueError(
                "the distill importance-sampling correction needs rollout_log_probs: serve through a backend that "
                "captures them, or set the family's importance-sampling cap to 0"
            )
        with torch.no_grad():
            _, outputs = get_log_probs_and_entropy(
                logits,
                args=args,
                unconcat_tokens=unconcat_tokens,
                total_lengths=total_lengths,
                response_lengths=response_lengths,
                with_entropy=False,
            )
            if settings.importance_sampling_level == "token":
                weights = [
                    token_importance_weights(student, rollout, settings.importance_sampling_cap)
                    for student, rollout in zip(outputs["log_probs"], rollout_log_probs, strict=True)
                ]
            else:
                weights = [
                    sequence_importance_weight(student, rollout, mask, settings.importance_sampling_cap)
                    for student, rollout, mask in zip(outputs["log_probs"], rollout_log_probs, loss_masks, strict=True)
                ]
            student_log_probs = torch.cat(outputs["log_probs"], dim=0).float()
            engine_log_probs = torch.cat(rollout_log_probs, dim=0).float()
        scaled = [sample * weight for sample, weight in zip(per_sample_divergence, weights, strict=True)]
        # Slime sums a micro-batch's metrics over its samples and divides the
        # step's total by the global batch size, so every value here is a sum
        # of per-sample means, as ``sum_of_sample_mean`` produces.
        if settings.importance_sampling_level == "token":
            metrics["distill_is_weight"] = sum_of_sample_mean(torch.cat(weights))
        else:
            metrics["distill_is_weight"] = torch.stack(weights).sum()
        # How far the trainer's forward sits from the rollout engine on the
        # sampled tokens: a large gap means a mismatch to fix, not to weight.
        metrics["distill_student_log_prob"] = sum_of_sample_mean(student_log_probs)
        metrics["distill_rollout_log_prob"] = sum_of_sample_mean(engine_log_probs)
        metrics["distill_log_prob_abs_diff"] = sum_of_sample_mean((student_log_probs - engine_log_probs).abs())

    weighted = torch.cat(
        [sample * float(sample_weight) for sample, sample_weight in zip(scaled, sample_weights, strict=True)], dim=0
    )
    loss = sum_of_sample_mean(weighted)
    if weighted.numel() == 0:
        loss = loss + 0 * logits.sum()
    metrics["loss"] = loss.detach().clone()
    metrics["distill_divergence"] = sum_of_sample_mean(divergence.detach())
    metrics["distill_sample_weight"] = divergence.new_tensor([float(value) for value in sample_weights]).sum()
    return loss, metrics


def distill_actor_pre_train(actor: Any, rollout_data: dict[str, Any]) -> None:
    """``--reef-actor-pre-train-hook-path`` entry point: score every teacher sequence before the step."""
    if not rollout_data.get("teacher_tokens"):
        raise ValueError("every distill sample must carry teacher_tokens")
    from slime.utils.timer import timer

    from reef.train.slime_backend.distill.teacher import compute_teacher_rows

    with timer("distill_teacher"):
        compute_teacher_rows(actor, rollout_data)


__all__ = [
    "chunked_token_divergence",
    "distill_actor_pre_train",
    "distill_loss",
    "gather_log_probs_at_ids",
    "global_log_sum_exp",
    "native_topk_ids",
    "restricted_divergence",
    "sampled_reverse_kl",
    "sequence_importance_weight",
    "sum_across_vocab_shards",
    "token_divergence",
]
