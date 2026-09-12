"""A chunkwise gated-delta recurrence: parallel within a chunk, so it trains fast.

mlx-lm's GatedDeltaNet runs its recurrence one token at a time: as a Metal
kernel in eval mode and, because that kernel has no vjp, as a loop of plain
ops in training mode. Sequential is fine for decoding and hopeless for
training: the loop's autodiff keeps three to four state matrices per token per
crossed layer (10.5 MB a token on Qwen3.8-27B), and both directions are a
chain of hundreds of tiny kernels whose launch latency, not arithmetic, is
what the clock measures.

This is the chunkwise form from Gated Delta Networks (Yang, Kautz and
Hatamizadeh, arXiv 2412.06464), ported from flash-linear-attention's
reference implementation (Songlin Yang, Yu Zhang, Zhiyuan Li; MIT). Within a
chunk of ``chunk`` tokens the recurrence is written, via the WY
representation, as a few small matrix products; only between chunks does a
state pass sequentially. So a 700-token row is eleven steps of batched matrix
multiplies rather than seven hundred steps of vector updates, and because
every operation is an ordinary differentiable op, MLX's own autodiff gives
the backward, storing one state per chunk.

The one piece the reference does row by row — inverting the unit lower
triangular matrix of the UT transform — is done here by block forward
substitution, halving the matrix at each of ``log2(chunk)`` levels, so it is
a handful of batched products rather than a loop over rows.

Conventions are mlx-lm's: ``q, k`` of ``[B, T, Hk, Dk]``, ``v`` of ``[B, T,
Hv, Dv]``, ``g`` the per-head decay multiplier (already exponentiated) of
``[B, T, Hv]``, ``beta`` of ``[B, T, Hv]`` (already a sigmoid), the state
``[B, Hv, Dv, Dk]``, and a ``[B, T]`` mask under which a position leaves the
state alone and emits zero. Per-column gating (``g`` of four dimensions) is
not chunked here and falls back to mlx-lm's loop.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import mlx.core as mx

_Recurrence = Callable[..., tuple[mx.array, mx.array]]

#: ``g`` is a decay in (0, 1]; the chunkwise form needs its logarithm, and a
#: decay that has underflowed to zero must stay finite there.
_SMALLEST_DECAY = 1e-30


def _unit_lower_inverse(strict_lower: mx.array) -> mx.array:
    """``(I - M)^-1`` for a strictly lower triangular ``M`` of ``[..., C, C]``.

    By block forward substitution, halving the matrix at each level: with
    ``L = [[A, 0], [C, B]]`` unit lower triangular, ``L^-1 = [[A^-1, 0],
    [-B^-1 C A^-1, B^-1]]``, and the two diagonal blocks are inverted in one
    batched recursive call. ``log2(C)`` levels of a few products each.

    Not the Neumann series ``(I + M)(I + M²)(I + M⁴)…`` although ``M`` is
    nilpotent and that is exact in exact arithmetic: on a model's keys,
    which are close to each other from one token to the next, the powers of
    ``M`` grow combinatorially and the series cancels catastrophically in
    float32. It held on random keys and produced states of 1e35 on real ones.
    """
    size = strict_lower.shape[-1]
    if size == 1:
        return mx.ones_like(strict_lower)
    half = size // 2
    if half * 2 != size:
        raise ValueError(f"the chunk must be a power of two, not {size}")
    lower = mx.eye(size, dtype=strict_lower.dtype) - strict_lower
    diagonal = mx.stack([lower[..., :half, :half], lower[..., half:, half:]], axis=-3)
    # The recursion takes M = I - L, so hand it the blocks' strictly lower part.
    inverses = _unit_lower_inverse(mx.eye(half, dtype=lower.dtype) - diagonal)
    top_inverse, bottom_inverse = inverses[..., 0, :, :], inverses[..., 1, :, :]
    corner = -(bottom_inverse @ lower[..., half:, :half] @ top_inverse)
    zeros = mx.zeros_like(corner)
    return mx.concatenate(
        [
            mx.concatenate([top_inverse, zeros], axis=-1),
            mx.concatenate([corner, bottom_inverse], axis=-1),
        ],
        axis=-2,
    )


def chunkwise_gated_delta(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: mx.array | None,
    chunk: int,
) -> tuple[mx.array, mx.array]:
    """The recurrence over a whole sequence, chunk by chunk; see the module."""
    batch, length, heads, _ = q.shape
    dtype = q.dtype

    # Work in float32 and in [B, H, T, ...], the layout every product below
    # wants. A masked position is an identity step: nothing written (beta 0)
    # and nothing decayed (g 1); its output is zeroed at the end.
    if mask is not None:
        keep = mask[:, :, None]
        beta = mx.where(keep, beta, mx.zeros_like(beta))
        g = mx.where(keep, g, mx.ones_like(g))
    q, k, v = (t.astype(mx.float32).transpose(0, 2, 1, 3) for t in (q, k, v))
    beta = beta.astype(mx.float32).transpose(0, 2, 1)
    log_g = mx.log(mx.maximum(g.astype(mx.float32), _SMALLEST_DECAY)).transpose(0, 2, 1)

    # Pad to whole chunks with identity steps.
    padding = (chunk - length % chunk) % chunk
    if padding:
        q, k, v = (mx.pad(t, [(0, 0), (0, 0), (0, padding), (0, 0)]) for t in (q, k, v))
        beta = mx.pad(beta, [(0, 0), (0, 0), (0, padding)])
        log_g = mx.pad(log_g, [(0, 0), (0, 0), (0, padding)])
    count = (length + padding) // chunk

    v_beta = v * beta[..., None]
    k_beta = k * beta[..., None]
    q, k, v_beta, k_beta = (t.reshape(batch, heads, count, chunk, -1) for t in (q, k, v_beta, k_beta))
    log_g = log_g.reshape(batch, heads, count, chunk)

    # Decay accumulated within each chunk; exp(cum_i - cum_j) for i >= j is
    # the decay between two positions of the same chunk.
    cum = mx.cumsum(log_g, axis=-1)
    decay_within = mx.exp(cum)[..., None]
    lower = mx.tril(mx.ones((chunk, chunk), dtype=mx.bool_))
    strict_lower = mx.tril(mx.ones((chunk, chunk), dtype=mx.bool_), k=-1)
    pairwise = mx.where(lower, mx.exp(mx.where(lower, cum[..., :, None] - cum[..., None, :], 0.0)), 0.0)

    # The UT transform: T = (I + tril(diag(β) K Kᵀ ⊙ decay, -1))^-1, and with
    # it the chunk's keys and values as the recurrence would have written
    # them, corrected for what earlier positions in the chunk erased.
    strict = mx.where(strict_lower, -(k_beta @ k.transpose(0, 1, 2, 4, 3)) * pairwise, 0.0)
    transform = _unit_lower_inverse(strict)
    u = transform @ v_beta
    w = transform @ (k_beta * decay_within)

    # mlx-lm keeps the state as [Dv, Dk]; the products below want [Dk, Dv].
    current = state.astype(mx.float32).transpose(0, 1, 3, 2)
    outputs = []
    for index in range(count):
        q_c, k_c = q[:, :, index], k[:, :, index]
        cum_c = cum[:, :, index]
        attention = mx.where(lower, (q_c @ k_c.transpose(0, 1, 3, 2)) * pairwise[:, :, index], 0.0)
        v_new = u[:, :, index] - w[:, :, index] @ current
        outputs.append((q_c * decay_within[:, :, index]) @ current + attention @ v_new)
        end_decay = mx.exp(cum_c[..., -1])
        to_end = mx.exp(cum_c[..., -1, None] - cum_c)[..., None]
        current = current * end_decay[..., None, None] + (k_c * to_end).transpose(0, 1, 3, 2) @ v_new

    y = mx.concatenate(outputs, axis=2)[:, :, :length].transpose(0, 2, 1, 3)
    if mask is not None:
        y = mx.where(mask[:, :, None, None], y, mx.zeros_like(y))
    return y.astype(dtype), current.transpose(0, 1, 3, 2)


def chunked_recurrence(chunk: int) -> _Recurrence:
    """``mlx_lm.models.gated_delta.gated_delta_ops``, done chunkwise.

    Same signature and semantics: ``(q, k, v, g, beta, state=None, mask=None)
    -> (y, state)``.
    """
    if chunk <= 0 or chunk & (chunk - 1):
        raise ValueError("the recurrence chunk must be a positive power of two")

    def gated_delta_ops(
        q: mx.array,
        k: mx.array,
        v: mx.array,
        g: mx.array,
        beta: mx.array,
        state: mx.array | None = None,
        mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        if g.ndim == 4:
            from mlx_lm.models.gated_delta import gated_delta_ops as sequential

            return sequential(q, k, v, g, beta, state, mask)
        batch, _, key_heads, key_dim = q.shape
        value_heads, value_dim = v.shape[-2:]
        if state is None:
            state = mx.zeros((batch, value_heads, value_dim, key_dim), dtype=mx.float32)
        if (repeat := value_heads // key_heads) > 1:
            # Grouped queries and keys are shared across value heads; repeat
            # them so the maths is one head to one head, and let mx.repeat's
            # own vjp sum the gradient back over the group.
            q = mx.repeat(q, repeat, -2)
            k = mx.repeat(k, repeat, -2)
        return chunkwise_gated_delta(q, k, v, g, beta, state, mask, chunk)

    return gated_delta_ops


@contextmanager
def chunked_gated_delta(chunk: int) -> Iterator[None]:
    """Route mlx-lm's training-mode recurrence through :func:`chunked_recurrence`.

    ``gated_delta_update`` looks ``gated_delta_ops`` up in its module at call
    time, so rebinding that name is enough, and undoing it on exit leaves
    mlx-lm exactly as found. A chunk of 0 changes nothing. An mlx-lm without
    this module has no such layer to route, so there is nothing to do.
    """
    if chunk <= 0:
        yield
        return
    try:
        from mlx_lm.models import gated_delta as module
    except ImportError:
        yield
        return
    original = module.gated_delta_ops
    module.gated_delta_ops = chunked_recurrence(chunk)
    try:
        yield
    finally:
        module.gated_delta_ops = original
