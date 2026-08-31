# Verbatim port of Gen-Verse/OpenClaw-RL's paper objective (Eq. 1):
# ruff: noqa: RET504, RUF005, SIM108, RUF019 — verbatim upstream kernel, style kept as-is
# openclaw-combine/openclaw_topk_select_loss.py plus the pure-tensor kernel it
# imports from hint_opd_loss.py (_opd_one_sample, vocab-parallel gather/LSE,
# masked lse/softmax) and hint_opd_select_loss.py (_overlap_count_per_token,
# _select_k_star_per_token, _gather_along_K).
#
# Mechanical adaptations, each marked in place: vendored import paths; the
# max_seq_lens kwarg dropped (older get_log_probs_and_entropy signature);
# and the knobs upstream reads from OPENCLAW_TOPK_* env vars read from
# ``args`` instead (reef carries them as --openclawrl-* flags; defaults match
# the paper launcher). Everything else is upstream text.
#
# ell_old on S^q is upstream's: gathered from the old-actor forward, never
# from the serving engine. Reef's index set S^q still comes from the
# generation-time capture — that is only an index set, and it is the one the
# teacher was scored against.

from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist
from megatron.core import mpu
from slime.backends.megatron_utils.loss import get_log_probs_and_entropy, get_responses
from slime.utils.ppo_utils import compute_approx_kl, compute_policy_loss

from reef.train.slime_backend.algorithm import objective

_NEG_INF = float("-inf")
# verl-style numerical guard on the log-ratio before exp(). Prevents
# exp(huge) NaNs on rare tail candidates early in training.
_PPO_KL_CLAMP = 20.0


def _local_lse(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Numerically-stable lse over the True positions of ``mask``.

    Both tensors are ``[R, K]``. Returns ``[R, 1]``. If the entire row's
    mask is all-False we return 0 (the row will be masked out downstream).
    """
    masked_vals = torch.where(mask, values, values.new_full((), _NEG_INF))
    row_max = masked_vals.max(dim=-1, keepdim=True).values
    # If a row is all -inf (no valid k), replace its max with 0 so the
    # log/exp stays finite; the row's loss will be zeroed by row_valid.
    row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
    shifted = masked_vals - row_max
    exped = torch.where(mask, shifted.exp(), torch.zeros_like(shifted))
    sum_exp = exped.sum(dim=-1, keepdim=True)
    sum_exp = torch.clamp(sum_exp, min=1e-30)
    return row_max + sum_exp.log()


def _local_softmax(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Softmax over the True positions of ``mask``.

    Both tensors are ``[R, K]``. Returns ``[R, K]``: off-mask entries are 0
    and rows where the mask is all-False return all-zero (the row is
    masked out downstream by ``row_valid``).

    Used for the IS weight ``w_v = softmax(ell_old over S_t)``.
    """
    lse = _local_lse(values, mask)  # [R, 1]
    out = (values - lse).exp()
    out = torch.where(mask, out, torch.zeros_like(out))
    return out


class _VocabParallelGatherRawLogits(torch.autograd.Function):
    """Single-pass raw-logit gather at K global vocab indices, TP-sharded.

    Forward:
        logits:     ``[R, V_local]`` (vocab dim is TP-sharded).
        idx_global: ``[R, K]`` global vocab ids in ``[0, V)``.

        Returns ``raw ∈ [R, K]`` where ``raw[r,k] = logits_global[r, idx[r,k]]``,
        reconstructed across TP via a single all_reduce(SUM) on the masked
        per-rank gather (off-shard entries contribute 0).

    Backward:
        ``∂L/∂logits[r,v] = sum_k g[r,k] · 1{v == idx[r,k] && in_shard}``
        Implemented via scatter_add on the rank's local logits region.

    Memory: saves only ``[R, K]`` index data + the ``in_shard`` mask; the
    raw-logit slice itself is rederived from the saved ``logits`` ref via
    a tiny gather. For training-time use the activation cost is O(R*K)
    instead of O(R*V_local).
    """

    @staticmethod
    def forward(ctx, logits, idx_global, tp_group, tp_world, tp_rank):
        V_local = logits.size(-1)
        shard_lo = tp_rank * V_local

        in_shard = (idx_global >= shard_lo) & (idx_global < shard_lo + V_local)
        idx_local = (idx_global - shard_lo).clamp(min=0, max=V_local - 1)
        gathered = torch.gather(logits, dim=-1, index=idx_local)  # [R, K]
        gathered_masked = torch.where(in_shard, gathered, torch.zeros_like(gathered))
        if tp_world > 1:
            dist.all_reduce(gathered_masked, op=dist.ReduceOp.SUM, group=tp_group)

        ctx.save_for_backward(idx_local, in_shard)
        ctx.logits_shape = logits.shape
        ctx.logits_dtype = logits.dtype
        ctx.logits_device = logits.device
        ctx.tp_world = tp_world
        return gathered_masked  # [R, K]

    @staticmethod
    def backward(ctx, grad_out):
        idx_local, in_shard = ctx.saved_tensors
        # Mask the incoming grad to only this rank's shard, then scatter into the
        # rank-local logits gradient. Off-shard contributions are zeroed.
        masked_grad = torch.where(in_shard, grad_out, torch.zeros_like(grad_out))
        grad_input = torch.zeros(ctx.logits_shape, dtype=ctx.logits_dtype, device=ctx.logits_device)
        grad_input.scatter_add_(dim=-1, index=idx_local, src=masked_grad.to(ctx.logits_dtype))
        return grad_input, None, None, None, None


def _gather_student_raw_logits_at_indices(
    logits_chunk: torch.Tensor,
    indices: torch.Tensor,
    tp_group,
) -> torch.Tensor:
    """Gather raw global logits at ``indices`` (autograd-connected)."""
    tp_world = dist.get_world_size(group=tp_group) if dist.is_initialized() else 1
    tp_rank = dist.get_rank(group=tp_group) if dist.is_initialized() else 0
    return _VocabParallelGatherRawLogits.apply(logits_chunk, indices, tp_group, tp_world, tp_rank)


# ---------------------------------------------------------------------------
# TP-aware global LSE over the full vocab (autograd).
# ---------------------------------------------------------------------------


class _VocabParallelGlobalLSE(torch.autograd.Function):
    """Numerically-stable global LSE over the FULL vocab dim, TP-aware.

    Forward:
        logits:    ``[R, V_local]``  (vocab dim is TP-sharded).
        Returns ``lse ∈ [R, 1]`` where
            ``lse[r] = log sum_{v=0..V-1} exp(logits_global[r, v])``.

    Backward:
        ``∂L/∂logits[r, v] = grad_out[r, 0] * softmax_global(r, v)``.
        Locally, ``softmax_global(r, v) = exp(logits[r, v] - lse[r])`` for
        v in this rank's shard. Each TP rank scatters its own local-shard
        softmax, which together reconstruct the global softmax gradient.

    Memory: backward materializes ``[R, V_local]`` softmax probabilities.
    This is unavoidable for a faithful global ratio.
    """

    @staticmethod
    def forward(ctx, logits, tp_group, tp_world):
        logits_f = logits.float()
        row_max = logits_f.max(dim=-1, keepdim=True).values  # [R, 1]
        if tp_world > 1:
            dist.all_reduce(row_max, op=dist.ReduceOp.MAX, group=tp_group)
        shifted = logits_f - row_max
        sum_exp = shifted.exp().sum(dim=-1, keepdim=True)  # [R, 1] local
        if tp_world > 1:
            dist.all_reduce(sum_exp, op=dist.ReduceOp.SUM, group=tp_group)
        lse = row_max + sum_exp.clamp_min(1e-30).log()  # [R, 1]

        ctx.save_for_backward(logits, lse)
        return lse

    @staticmethod
    def backward(ctx, grad_out):
        # grad_out: [R, 1].   d(lse)/d(logits[v]) = global softmax(v).
        # Local-shard softmax = exp(logits - global_lse). Multiplying by
        # grad_out broadcasts [R, 1] across the local vocab dim.
        logits, lse = ctx.saved_tensors
        softmax_local = (logits.float() - lse).exp()  # [R, V_local]
        grad_in = grad_out * softmax_local
        return grad_in.to(logits.dtype), None, None


def _vocab_parallel_global_lse(
    logits_chunk: torch.Tensor,
    tp_group,
) -> torch.Tensor:
    """Global LSE over the full vocab, TP-aware (autograd-connected).

    Returns ``[R, 1]`` in float32. Use to convert raw logits to global
    log-probabilities: ``log pi(v) = logits[v] - global_lse``.
    """
    tp_world = dist.get_world_size(group=tp_group) if dist.is_initialized() else 1
    return _VocabParallelGlobalLSE.apply(logits_chunk, tp_group, tp_world)


def _opd_one_sample(
    logits_chunk: torch.Tensor,
    *,
    student_indices: torch.Tensor,  # [R, Kq] long, GLOBAL vocab ids
    student_captured_lp: torch.Tensor,  # [R, Kq] serving-engine log-probs (monitor only)
    teacher_indices: torch.Tensor,  # [R, Kp] long
    teacher_lp: torch.Tensor,  # [R, Kp] log pi_T(v) for v in teacher_indices
    eps_lo: float,
    eps_hi: float,
    diff_clip: float | None,
    tp_group,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """verl-aligned top-K OPD surrogate for one sample.

    Pipeline (notation matches the file docstring):
        1. Build ``S_t = student_indices ∩ teacher_indices``, expressed as
           a [R, Kq] boolean mask aligned with ``student_indices``. For the
           ``student`` / ``teacher`` modes Kq == Kp and the two index lists
           are identical, so the mask is all-True.
        2. Pull GLOBAL teacher log-probs ``ell_T(v)`` at ``student_indices``
           via per-row index match. (The teacher gather pass already gave
           us global log pi_T values; we just reorder them.)
        3. Compute GLOBAL student-current log-probs:
              ``ell_cur(v) = raw_logits(v) - global_lse(raw_logits)``
           via the autograd-connected gather + global LSE helpers. Both
           halves contribute gradient to the student's logits.
        4. Detached IS weight on S_t:
              ``w_v = softmax(ell_old over S_t)``,    sum_v w_v = 1.
        5. Detached advantage (with optional magnitude clamp on the diff):
              ``diff_v = clamp(ell_T(v) - ell_old(v), -t, +t)`` (t = diff_clip,
              skipped when diff_clip is None),
              ``A_v   = diff_v * w_v``.
        6. PPO surrogate using the GLOBAL log-ratio:
              ``ppo_kl = ell_old - ell_cur``  (clamped for numerical safety),
              ``L_v    = max(-A_v * rho_v, -A_v * clip(rho_v, 1-eps_lo, 1+eps_hi))``
           via ``compute_policy_loss(ppo_kl, A)``.
        7. Per-token aggregation: SUM over S_t (NOT mean -- the IS weight
           already normalises within S_t, so summing concentrates gradient
           pressure on the high-w_v candidates without dividing it away).

    Returns:
        per_token_pg     [R]  OPD surrogate per token (sum over S_t)
        per_token_clip   [R]  fraction of S_t entries that hit the PPO clip
        per_token_diff   [R]  w-weighted |ell_T - ell_old| per token (monitor)
        row_valid        [R]  bool, True iff |S_t| >= 1
    """
    R = student_indices.size(0)
    if R == 0:
        z = student_indices.new_zeros((0,), dtype=torch.float32)
        b = student_indices.new_zeros((0,), dtype=torch.bool)
        return z, z, z, z, b

    # 1) Build the S_t mask on the student-indices axis.
    if torch.equal(student_indices, teacher_indices):
        # Fast path: student/teacher modes ship identical index sets.
        sub_mask = torch.ones_like(student_indices, dtype=torch.bool)
        eq = None  # not needed
    else:
        # eq: [R, Kq, Kp] booleans, at most one True per (r, k).
        eq = student_indices.unsqueeze(-1) == teacher_indices.unsqueeze(-2)
        sub_mask = eq.any(dim=-1)  # [R, Kq]
    row_valid = sub_mask.any(dim=-1)  # [R]
    mask_f = sub_mask.float()  # [R, Kq]

    # 2) Align teacher GLOBAL log-probs to student_indices ordering.
    if eq is None:
        teacher_lp_aligned = teacher_lp.float()
    else:
        # Weighted sum picks the unique matched teacher_lp per (r, k);
        # off-mask entries collapse to 0 (zeroed out by mask_f anyway).
        eq_f = eq.float()
        teacher_lp_aligned = (eq_f * teacher_lp.unsqueeze(-2).float()).sum(dim=-1)

    # 3) GLOBAL student-current log-probs at student_indices (autograd).
    #    raw_at_K - global_lse(raw)  =  log pi_theta(v).
    student_new_raw = _gather_student_raw_logits_at_indices(logits_chunk, student_indices, tp_group).float()  # [R, Kq]
    global_lse = _vocab_parallel_global_lse(logits_chunk, tp_group)  # [R, 1]
    ell_cur = student_new_raw - global_lse  # [R, Kq]

    #    ell_old comes from THIS forward, detached — never from the serving
    #    engine's capture. Upstream asserts the same invariant
    #    (openclaw_topk_select_loss.py: "old log-probs come from the Megatron
    #    old_actor forward, NOT from rollout"), and with one optimizer step
    #    per rollout and no --keep-old-actor the old actor IS this actor, so
    #    this gather is that forward. It makes rho_v identically 1, putting
    #    the surrogate at its unclipped linear point where the whole teacher
    #    signal survives. Reading a cross-engine capture here instead leaves
    #    rho = exp(ell_cur_megatron - ell_old_sglang) ~ 1 +- drift, and
    #    compute_policy_loss's pessimistic max then zeroes the gradient
    #    exactly on the sign-matching side — the side carrying the teacher's
    #    instruction. ``validate_backend_args`` refuses the multi-step
    #    configurations where old and current diverge.
    ell_old = ell_cur.detach()  # [R, Kq]
    ell_T = teacher_lp_aligned  # [R, Kq]
    # Monitor only: how far the serving engine's capture sits from this
    # forward. This is the drift that used to ride into the surrogate.
    capture_drift = (student_captured_lp.float() - ell_old).abs()  # [R, Kq]

    # 4) Detached IS weight w_v = softmax(ell_old | S_t).
    #    Built from ell_old alone, so the diff_clip below does NOT change w.
    w = _local_softmax(ell_old, sub_mask).detach()  # [R, Kq]

    # 5) Detached advantage A_v = (ell_T - ell_old) * w_v.
    #    Optionally clamp the per-candidate teacher-vs-old log-ratio to
    #    [-diff_clip, +diff_clip]. Bounds advantage magnitude when the
    #    teacher and old student disagree wildly on a candidate (rare-token
    #    candidates, early training, cross-tokenizer distillation).
    diff = (ell_T - ell_old).detach()  # [R, Kq]
    if diff_clip is not None:
        diff = diff.clamp(min=-diff_clip, max=diff_clip)
    advantage = (diff * w).detach()  # [R, Kq]

    # 6) PPO surrogate with the GLOBAL log-ratio.
    #    compute_policy_loss expects ppo_kl = log(p_old / p_new) and returns
    #    -min(rho*A, clip(rho)*A), exactly our L_v.
    ppo_kl = (ell_old - ell_cur).clamp(min=-_PPO_KL_CLAMP, max=_PPO_KL_CLAMP)
    pg, clip = compute_policy_loss(ppo_kl, advantage, eps_lo, eps_hi)

    # 7) SUM over S_t (mask out non-subset entries).
    per_token_pg = (pg * mask_f).sum(dim=-1)  # [R]
    # clipfrac: fraction of S_t entries that were clipped (not summed).
    n_per_token = mask_f.sum(dim=-1).clamp(min=1.0)
    per_token_clip = (clip * mask_f).sum(dim=-1) / n_per_token  # [R]
    # diff monitor: w-weighted |ell_T - ell_old| -- matches the loss's
    # importance weighting so it's a calibrated "mean teacher gap" signal.
    per_token_diff = (diff.abs() * w).sum(dim=-1)  # [R]
    per_token_capture_drift = (capture_drift * w).sum(dim=-1)  # [R]

    # Zero rows with empty subset so they don't pollute the trajectory mean.
    row_valid_f = row_valid.float()
    return (
        per_token_pg * row_valid_f,
        per_token_clip * row_valid_f,
        per_token_diff * row_valid_f,
        per_token_capture_drift * row_valid_f,
        row_valid,
    )


# ---------------------------------------------------------------------------
# Slime entry point
# ---------------------------------------------------------------------------


def _overlap_count_per_token(
    student_idx: torch.Tensor,  # [R, K_q]
    teacher_idx: torch.Tensor,  # [K, R, K_p]
) -> torch.Tensor:
    """``O[k, t] = | S^q_t ∩ S^p_{t,k} |`` via broadcasted equality.

    Memory: ``[K, R, K_q, K_p]`` boolean intermediate. With K_q=K_p=20 and
    R≤8192, K≤8 this is ~25 MB per sample — small.
    """
    eq = student_idx.unsqueeze(0).unsqueeze(-1) == teacher_idx.unsqueeze(-2)
    return eq.any(dim=-1).sum(dim=-1).to(torch.long)  # [K, R]


def _select_k_star_per_token(
    overlap_kr: torch.Tensor | None,  # [K, R] or None for shortest
    *,
    hint_selection: str,
    step_token_spans: list[list[int]] | None,  # per-sample list of [t0, t1]
    R: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute ``k*(t) ∈ [0, K)`` per response token.

    * ``shortest``         : ``k* = 0`` for every token (the rollout module
      orders candidates shortest-first, so candidate 0 is the shortest hint
      that survived the min-token / dedup filter). ``overlap_kr`` may be
      ``None`` since no overlap signal is required.
    * ``token_optimal``    : argmax over candidates per token.
    * ``sequence_optimal`` : argmax over candidates per PRM step
      (broadcast over tokens in the step). When ``step_token_spans`` is
      missing/empty -- the case for the hint_opt_exp single-CoT pipeline
      where the "sequence" IS the whole response -- this collapses to
      per-sample argmax (one ``k*`` for every token in the response).
    """
    if hint_selection == "shortest":
        return torch.zeros(R, dtype=torch.long, device=device)
    if overlap_kr is None:
        raise ValueError(f"overlap_kr is required for hint_selection={hint_selection!r}")
    K, R_kr = overlap_kr.shape
    if R_kr != R:
        raise ValueError(f"overlap_kr R mismatch: {R_kr} vs {R}")
    if K == 1:
        return torch.zeros(R, dtype=torch.long, device=device)
    if hint_selection == "token_optimal":
        return overlap_kr.argmax(dim=0)  # [R]
    if hint_selection == "sequence_optimal":
        if not step_token_spans:
            k_star_scalar = int(overlap_kr.sum(dim=-1).argmax().item())
            return torch.full((R,), k_star_scalar, dtype=torch.long, device=device)
        out = torch.zeros(R, dtype=torch.long, device=device)
        for span in step_token_spans:
            t0, t1 = int(span[0]), int(span[1])
            t0 = max(0, min(t0, R))
            t1 = max(t0, min(t1, R))
            if t1 == t0:
                continue
            seg_score = overlap_kr[:, t0:t1].sum(dim=-1)  # [K]
            out[t0:t1] = int(seg_score.argmax().item())
        return out
    raise ValueError(
        f"Unknown --openclawrl-hint-selection: {hint_selection!r}. "
        "Expected 'shortest' / 'token_optimal' / 'sequence_optimal'."
    )


def _gather_along_K(
    cand_tensor: torch.Tensor,  # [K, R, *]
    k_star_per_token: torch.Tensor,  # [R]
) -> torch.Tensor:
    """Slice ``cand_tensor[k*(t), t, ...]`` per token. Returns ``[R, *]``."""
    R = cand_tensor.size(1)
    trailing = cand_tensor.shape[2:]
    expand_shape = (1, R, *trailing)
    view_shape = (1, R) + tuple(1 for _ in trailing)
    gather_idx = k_star_per_token.view(view_shape).expand(expand_shape)
    return torch.gather(cand_tensor, dim=0, index=gather_idx).squeeze(0)


# ---------------------------------------------------------------------------
# Slime entry point
# ---------------------------------------------------------------------------


# The objective's knobs ride ``args`` (stamped as openclawrl_* by the
# spec's apply_driver_options), not the process environment. Upstream
# passes them through ray's runtime-env; reef's Megatron actors would have
# received them only by raylet inheritance, and a knob that silently falls
# back to a DIFFERENT value than the one configured is the worst kind of
# default -- ``adv_diff_clip`` used to fall back to 2.0 where every config
# sets 1.0, which upstream's launcher calls out as "important in
# stabilizing the training".


def _w_rl(args: Namespace) -> float:
    return float(getattr(args, "openclawrl_w_rl", 1.0))


def _w_opd(args: Namespace) -> float:
    return float(getattr(args, "openclawrl_w_opd", 1.0))


def _eps_clip_lo(args: Namespace) -> float:
    return float(args.eps_clip)


def _eps_clip_hi(args: Namespace) -> float:
    return float(args.eps_clip_high)


def _adv_diff_clip(args: Namespace) -> float | None:
    val = float(getattr(args, "openclawrl_adv_diff_clip", 1.0))
    return val if val > 0.0 else None


@objective("custom_advantage_function_path")
def openclawrl_advantages(args: Namespace, rollout_data: dict) -> None:
    """Keep Reef's advantages through Slime's old-policy preparation pass."""
    advantages = rollout_data.get("advantages")
    if advantages is None:
        raise ValueError("OpenClaw-RL top-K requires externally supplied advantages")
    # Slime's custom-advantage contract requires both keys. The top-K loss
    # consumes only advantages; matching returns keep its rollout metrics
    # well-defined without changing the policy signal.
    rollout_data.setdefault("returns", advantages)


@objective("custom_loss_function_path")
def openclawrl_loss(
    args: Namespace,
    batch: dict,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """``--custom-loss-function-path`` entry point for openclaw_combine_select.

    Dispatches on (``args.openclawrl_subset_mode``, ``args.openclawrl_hint_selection``)
    to cover all 9 cells:

      * ``subset_mode == "teacher"``: actor-side ``train_actor`` already
        collapsed the cand teacher tensors into single-cand keys
        (``prm_teacher_topk_log_probs``, ``prm_teacher_topk_indices``)
        AFTER selecting k* and re-gathering student-old log-probs at the
        chosen S^p. We consume those single-cand keys directly.
      * ``subset_mode in {student, overlap}``: read the cand keys
        directly and slice at k*(t) in-kernel. Under ``student`` the
        teacher cand log-probs are GATHERED at S^q (constant indices
        across k); the per-(k, t) selection signal travels in
        ``prm_teacher_native_topk_indices_cand``.
    """
    hint_selection = str(getattr(args, "openclawrl_hint_selection", "sequence_optimal"))
    subset_mode = str(getattr(args, "openclawrl_subset_mode", "student"))
    if hint_selection not in ("shortest", "token_optimal", "sequence_optimal"):
        raise ValueError(
            f"Unknown --openclawrl-hint-selection: {hint_selection!r}. Expected one of "
            "'shortest', 'token_optimal', 'sequence_optimal'."
        )
    if subset_mode not in ("student", "teacher", "overlap"):
        raise ValueError(
            f"Unknown --openclawrl-subset-mode: {subset_mode!r}. Expected one of 'student', 'teacher', 'overlap'."
        )

    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]

    w_rl = _w_rl(args)
    w_opd = _w_opd(args)
    eps_lo = _eps_clip_lo(args)
    eps_hi = _eps_clip_hi(args)
    diff_clip = _adv_diff_clip(args)
    entropy_coef = float(getattr(args, "entropy_coef", 0.0) or 0.0)
    need_entropy_for_loss = entropy_coef != 0.0

    _, log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=need_entropy_for_loss,
    )
    new_log_probs = torch.cat(log_probs_and_entropy["log_probs"], dim=0)

    # Reef adaptation: old-policy log-probs come from the Megatron
    # old-actor recompute for the sampled-token GRPO branch (upstream
    # semantics), while the OPD branch's ell_old on S^q ships from the
    # serving engine's generation-time top-K capture — the true rollout
    # policy, replacing upstream's megatron old-actor top-K.

    grpo_pg_loss = torch.zeros((), device=logits.device, dtype=torch.float32)
    grpo_pg_clipfrac = torch.zeros((), device=logits.device, dtype=torch.float32)
    ppo_kl_mean_sampled = torch.zeros((), device=logits.device, dtype=torch.float32)
    if w_rl != 0.0:
        old_log_probs = torch.cat(batch["log_probs"], dim=0)
        # Match the OPD branch's verl-style guard before compute_policy_loss()
        # exponentiates the log-ratio. Without this, a deterministic outlier
        # batch can produce inf * 0 -> NaN for OPD-only / zero-advantage tokens.
        ppo_kl_sampled = (old_log_probs - new_log_probs).clamp(
            min=-_PPO_KL_CLAMP,
            max=_PPO_KL_CLAMP,
        )
        rl_advantages = torch.cat(batch["advantages"], dim=0)
        pg_loss_tokens, pg_clipfrac_tokens = compute_policy_loss(ppo_kl_sampled, rl_advantages, eps_lo, eps_hi)
        grpo_pg_loss = sum_of_sample_mean(pg_loss_tokens)
        grpo_pg_clipfrac = sum_of_sample_mean(pg_clipfrac_tokens)
        ppo_kl_mean_sampled = sum_of_sample_mean(ppo_kl_sampled)

    opd_loss = torch.zeros((), device=logits.device, dtype=torch.float32)
    opd_clipfrac_scalar = torch.zeros((), device=logits.device, dtype=torch.float32)
    teacher_student_logp_diff_mean: torch.Tensor | None = None
    capture_drift_mean: torch.Tensor | None = None
    subset_size_mean: torch.Tensor | None = None
    sel_overlap_mean: torch.Tensor | None = None
    sel_k_star_mean: torch.Tensor | None = None

    student_topk_lp = batch.get("topk_log_probs")
    student_topk_idx = batch.get("topk_indices")

    have_student = (
        student_topk_lp is not None
        and student_topk_idx is not None
        and len(student_topk_lp) > 0
        and len(student_topk_idx) > 0
    )

    # The teacher tensors are EITHER cand-suffixed (multi-cand path,
    # subset in {student, overlap}) OR plain single-cand (actor-side
    # collapsed, subset == teacher). Pick which side to read once.
    use_cand_keys = subset_mode != "teacher"
    if use_cand_keys:
        teacher_topk_lp_any = batch.get("prm_teacher_topk_log_probs_cand")
        teacher_topk_idx_any = batch.get("prm_teacher_topk_indices_cand")
        teacher_native_idx_cand = batch.get("prm_teacher_native_topk_indices_cand")
    else:
        teacher_topk_lp_any = batch.get("prm_teacher_topk_log_probs")
        teacher_topk_idx_any = batch.get("prm_teacher_topk_indices")
        teacher_native_idx_cand = None

    have_teacher = (
        teacher_topk_lp_any is not None
        and teacher_topk_idx_any is not None
        and len(teacher_topk_lp_any) > 0
        and len(teacher_topk_idx_any) > 0
    )

    step_spans_per_sample = batch.get("step_wise_step_token_spans")

    if w_opd != 0.0:
        if not (have_student and have_teacher):
            cand_suffix = "_cand" if use_cand_keys else ""
            raise RuntimeError(
                "openclaw_topk_select_loss requires both student top-K "
                "(topk_log_probs / topk_indices) and the teacher top-K "
                f"(prm_teacher_topk_log_probs{cand_suffix} / "
                f"prm_teacher_topk_indices{cand_suffix}) in the batch. "
                "Confirm the serving backend captures generation top-K and "
                "the grader ships per-candidate teacher top-K tensors."
            )
        # Narrow the Optionals for the type checker; have_student/have_teacher
        # established this above.
        if student_topk_lp is None or student_topk_idx is None:
            raise RuntimeError("student top-K tensors disappeared after validation")
        if teacher_topk_lp_any is None or teacher_topk_idx_any is None:
            raise RuntimeError("teacher top-K tensors disappeared after validation")
        # Selection signal sanity for student/{token,sequence}_optimal.
        if (
            subset_mode == "student"
            and hint_selection != "shortest"
            and (teacher_native_idx_cand is None or len(teacher_native_idx_cand) == 0)
        ):
            raise RuntimeError(
                "subset_mode=student with --openclawrl-hint-selection in "
                "{token_optimal, sequence_optimal} requires "
                "prm_teacher_native_topk_indices_cand (the per-candidate "
                "selection signal) in the batch. Confirm slime's "
                "gather_at_indices multi-cand path is engaged. "
                "(shortest does not need it because k*=0 always.)"
            )

        tp_group = mpu.get_tensor_model_parallel_group()
        all_pg = []
        all_clip = []
        all_diff = []
        all_drift = []
        all_size = []
        all_overlap_sel = []
        all_k_star = []

        for i, (logits_chunk, _tokens_chunk) in enumerate(
            get_responses(
                logits,
                args=args,
                unconcat_tokens=batch["unconcat_tokens"],
                total_lengths=total_lengths,
                response_lengths=response_lengths,
            )
        ):
            s_idx = student_topk_idx[i].to(device=logits_chunk.device, dtype=torch.long)
            s_lp = student_topk_lp[i].to(device=logits_chunk.device, dtype=torch.float32)
            R = logits_chunk.size(0)
            if s_idx.dim() != 2 or s_idx.size(0) != R:
                raise ValueError(f"student topk shape mismatch: s_idx={tuple(s_idx.shape)} vs R={R}")

            t_lp_any_i = teacher_topk_lp_any[i].to(device=logits_chunk.device, dtype=torch.float32)
            t_idx_any_i = teacher_topk_idx_any[i].to(device=logits_chunk.device, dtype=torch.long)

            if use_cand_keys:
                # Multi-candidate path (subset in {student, overlap}):
                # tensors are [K, R, K_p].
                if t_idx_any_i.dim() != 3 or t_idx_any_i.size(1) != R:
                    raise ValueError(
                        "teacher topk_cand shape mismatch: "
                        f"t_idx_cand={tuple(t_idx_any_i.shape)} vs R={R}; expected [K, R, K_p]"
                    )
                if t_lp_any_i.shape != t_idx_any_i.shape:
                    raise ValueError(
                        "teacher logp/idx cand shape mismatch: "
                        f"lp={tuple(t_lp_any_i.shape)} idx={tuple(t_idx_any_i.shape)}"
                    )

                # Selection signal (only computed when needed).
                if hint_selection == "shortest":
                    overlap_kr = None
                else:
                    if subset_mode == "student":
                        # Cand log-probs are at S^q (constant across k);
                        # use each candidate's NATIVE top-K for the
                        # overlap signal instead.
                        if teacher_native_idx_cand is None:
                            raise RuntimeError("student subset selection has no native teacher indices")
                        sel_idx_src = teacher_native_idx_cand[i].to(device=logits_chunk.device, dtype=torch.long)
                        if sel_idx_src.shape[1] != R:
                            raise ValueError(
                                "native_topk shape mismatch: "
                                f"native={tuple(sel_idx_src.shape)} vs R={R}; expected [K, R, K_p]"
                            )
                    else:  # overlap
                        sel_idx_src = t_idx_any_i
                    overlap_kr = _overlap_count_per_token(s_idx, sel_idx_src)

                spans_i = (
                    step_spans_per_sample[i]
                    if step_spans_per_sample is not None and i < len(step_spans_per_sample)
                    else None
                )
                k_star_per_token = _select_k_star_per_token(
                    overlap_kr,
                    hint_selection=hint_selection,
                    step_token_spans=spans_i,
                    R=R,
                    device=logits_chunk.device,
                )

                # Slice cand tensors at k*(t).
                t_lp_sel = _gather_along_K(t_lp_any_i, k_star_per_token)
                if subset_mode == "student":
                    # Force kernel's S^p to S^q (cand indices are already
                    # constant across k under student subset; this keeps
                    # the contract explicit).
                    t_idx_sel = s_idx
                else:  # overlap
                    t_idx_sel = _gather_along_K(t_idx_any_i, k_star_per_token)

                if overlap_kr is None:
                    # ``shortest``: report selected-overlap as 0 so the
                    # wandb panel is well-defined and visibly distinct
                    # from the optimal modes (where it ≈ K_q for student).
                    overlap_sel_per_token = torch.zeros(R, device=logits_chunk.device, dtype=torch.float32)
                else:
                    row_idx = torch.arange(R, device=overlap_kr.device)
                    overlap_sel_per_token = overlap_kr[k_star_per_token, row_idx].float()
            else:
                # Legacy single-cand path (subset == teacher): the actor
                # has already done k*-selection AND re-gathered
                # student-old log-probs at the chosen S^p. We consume
                # the collapsed [R, K_p] tensors directly.
                if t_idx_any_i.dim() != 2 or t_idx_any_i.size(0) != R:
                    raise ValueError(
                        f"teacher topk shape mismatch: t_idx={tuple(t_idx_any_i.shape)} vs R={R}; expected [R, K_p]"
                    )
                if t_lp_any_i.shape != t_idx_any_i.shape:
                    raise ValueError("teacher log-prob and index shapes must match")
                t_idx_sel = t_idx_any_i
                t_lp_sel = t_lp_any_i
                overlap_sel_per_token = torch.zeros(R, device=logits_chunk.device, dtype=torch.float32)
                k_star_per_token = torch.zeros(R, device=logits_chunk.device, dtype=torch.long)

            pg_t, clip_t, diff_t, drift_t, valid_t = _opd_one_sample(
                logits_chunk,
                student_indices=s_idx,
                student_captured_lp=s_lp,
                teacher_indices=t_idx_sel,
                teacher_lp=t_lp_sel,
                eps_lo=eps_lo,
                eps_hi=eps_hi,
                diff_clip=diff_clip,
                tp_group=tp_group,
            )
            all_pg.append(pg_t)
            all_clip.append(clip_t)
            all_diff.append(diff_t)
            all_drift.append(drift_t)
            if torch.equal(s_idx, t_idx_sel):
                size_t = s_idx.new_full((s_idx.size(0),), s_idx.size(-1), dtype=torch.float32)
            else:
                eq = s_idx.unsqueeze(-1) == t_idx_sel.unsqueeze(-2)
                size_t = eq.any(dim=-1).float().sum(dim=-1)
            all_size.append(size_t * valid_t.float())
            all_overlap_sel.append(overlap_sel_per_token * valid_t.float())
            all_k_star.append(k_star_per_token.float() * valid_t.float())

        opd_pg_tokens = torch.cat(all_pg, dim=0)
        opd_clip_tokens = torch.cat(all_clip, dim=0)
        opd_diff_tokens = torch.cat(all_diff, dim=0)
        opd_drift_tokens = torch.cat(all_drift, dim=0)
        opd_size_tokens = torch.cat(all_size, dim=0)
        opd_overlap_sel_tokens = torch.cat(all_overlap_sel, dim=0)
        opd_k_star_tokens = torch.cat(all_k_star, dim=0)
        opd_loss = sum_of_sample_mean(opd_pg_tokens)
        opd_clipfrac_scalar = sum_of_sample_mean(opd_clip_tokens)
        teacher_student_logp_diff_mean = sum_of_sample_mean(opd_diff_tokens)
        capture_drift_mean = sum_of_sample_mean(opd_drift_tokens)
        subset_size_mean = sum_of_sample_mean(opd_size_tokens)
        sel_overlap_mean = sum_of_sample_mean(opd_overlap_sel_tokens)
        sel_k_star_mean = sum_of_sample_mean(opd_k_star_tokens)

    if need_entropy_for_loss:
        entropy = torch.cat(log_probs_and_entropy["entropy"], dim=0)
        entropy_loss = sum_of_sample_mean(entropy)
    else:
        entropy_loss = torch.zeros((), device=logits.device, dtype=torch.float32)

    loss = w_rl * grpo_pg_loss + w_opd * opd_loss
    if entropy_coef != 0.0:
        loss = loss - entropy_coef * entropy_loss

    kl_loss = torch.tensor(0.0, device=logits.device)
    kl_loss_coef = float(getattr(args, "kl_loss_coef", 0.0) or 0.0)
    if args.use_kl_loss and batch.get("ref_log_probs") is not None and kl_loss_coef != 0.0:
        ref_log_probs = torch.cat(batch["ref_log_probs"], dim=0)
        kl = compute_approx_kl(
            new_log_probs,
            ref_log_probs,
            kl_loss_type=args.kl_loss_type,
        )
        kl_loss = sum_of_sample_mean(kl)
        loss = loss + kl_loss_coef * kl_loss

    if new_log_probs.numel() == 0:
        loss = loss + 0 * logits.sum()

    train_rollout_logprob_abs_diff = None
    if "rollout_log_probs" in batch and batch["rollout_log_probs"]:
        rollout_lp = torch.cat(batch["rollout_log_probs"], dim=0)
        train_rollout_logprob_abs_diff = sum_of_sample_mean((new_log_probs.detach() - rollout_lp).abs())

    reported: dict[str, torch.Tensor] = {
        "loss": loss.clone().detach(),
        "grpo_pg_loss": grpo_pg_loss.clone().detach(),
        "opd_loss": opd_loss.clone().detach(),
        "entropy_loss": entropy_loss.clone().detach(),
        "grpo_pg_clipfrac": grpo_pg_clipfrac.clone().detach(),
        "opd_pg_clipfrac": opd_clipfrac_scalar.clone().detach(),
        "ppo_kl_sampled": ppo_kl_mean_sampled.clone().detach(),
        "w_rl": torch.tensor(w_rl, device=loss.device),
        "w_opd": torch.tensor(w_opd, device=loss.device),
    }
    if teacher_student_logp_diff_mean is not None:
        reported["opd_teacher_student_logp_topk_abs_mean"] = teacher_student_logp_diff_mean.clone().detach()
    if capture_drift_mean is not None:
        # How far the serving engine's top-K capture sits from this forward.
        # It no longer enters the surrogate; watching it is how we know what
        # reading it as ell_old used to cost.
        reported["opd_capture_vs_actor_logp_abs_mean"] = capture_drift_mean.clone().detach()
    if subset_size_mean is not None:
        reported["opd_subset_size"] = subset_size_mean.clone().detach()
    if sel_overlap_mean is not None:
        reported["sel_overlap_at_k_star"] = sel_overlap_mean.clone().detach()
    if sel_k_star_mean is not None:
        reported["sel_k_star_mean"] = sel_k_star_mean.clone().detach()
    if train_rollout_logprob_abs_diff is not None:
        reported["train_rollout_logprob_abs_diff"] = train_rollout_logprob_abs_diff.clone().detach()
    if args.use_kl_loss:
        reported["kl_loss"] = kl_loss.clone().detach()

    # Embed selection-mode / subset-mode tags as constant ints for wandb
    # grouping without needing the run config.
    mode_id = {"shortest": 0, "token_optimal": 1, "sequence_optimal": 2}[hint_selection]
    subset_id = {"student": 0, "overlap": 1, "teacher": 2}[subset_mode]
    reported["hint_selection_mode_id"] = torch.tensor(mode_id, device=loss.device)
    reported["distill_subset_mode_id"] = torch.tensor(subset_id, device=loss.device)

    return loss, reported


@objective("reef_actor_init_hook_path")
def openclawrl_actor_init(actor: Any) -> None:
    """Back up the freshly loaded actor weights as the frozen Megatron teacher.

    The topk-select objective requires the frozen-base Megatron teacher
    (upstream forces OPENCLAW_COMBINE_OPD_TEACHER_SOURCE=megatron). Fresh init
    just bridge-loaded exactly those weights, so the backup IS the frozen PRM.
    A resumed Megatron checkpoint holds trained weights instead — backing
    those up would silently swap the teacher for the student, so it is
    refused outright. Fresh-vs-resumed is the loader's own dispatch (the
    bridge path reports iteration 0, so the returned rollout id cannot tell).
    """
    from slime.backends.megatron_utils.checkpoint import _is_megatron_checkpoint

    if actor.args.load and _is_megatron_checkpoint(actor.args.load):
        raise RuntimeError(
            "openclawrl resumed from a Megatron checkpoint, so the current weights are "
            "not the frozen base and cannot serve as its teacher. Start from a fresh run "
            "directory (a base-weight reload on resume is not implemented)."
        )
    actor.weights_backuper.backup("openclaw_teacher")


@objective("reef_actor_pre_train_hook_path")
def openclawrl_actor_pre_train(actor: Any, rollout_data: dict) -> None:
    """Run the frozen-teacher K-loop over the batch's hint candidates."""
    if not rollout_data.get("teacher_tokens_cand"):
        return
    from slime.utils.timer import timer

    from recipes.openclawrl.slime.teacher import compute_openclaw_teacher_cands

    with timer("openclaw_teacher"):
        compute_openclaw_teacher_cands(actor, rollout_data)
