"""A chunkwise gated-delta recurrence for training GatedDeltaNet layers.

mlx-lm runs the GatedDeltaNet recurrence one token at a time. In training
mode that loop is differentiable, but its backward pass keeps several state
matrices per token. This module computes the same recurrence chunk by chunk:
matrix products within a chunk, and one state passed between chunks. MLX's
autodiff then keeps one state per chunk.

The method is the chunkwise form of Gated Delta Networks (Yang, Kautz and
Hatamizadeh, arXiv 2412.06464), ported from flash-linear-attention's
reference implementation (Songlin Yang, Yu Zhang, Zhiyuan Li; MIT). The unit
lower triangular inverse of the UT transform uses block forward substitution.

Conventions are mlx-lm's: ``q, k`` are ``[B, T, Hk, Dk]``, ``v`` is
``[B, T, Hv, Dv]``, ``g`` is the per-head decay multiplier ``[B, T, Hv]``
(already exponentiated), ``beta`` is ``[B, T, Hv]`` (already a sigmoid), the
state is ``[B, Hv, Dv, Dk]``, and under the ``[B, T]`` mask a position leaves
the state unchanged and emits zero. Per-column gating (a four-dimensional
``g``) uses mlx-lm's loop.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import mlx.core as mx
from mlx_lm.models import gated_delta

#: A function with the signature of ``gated_delta_ops``.
GatedDeltaOps = Callable[..., tuple[mx.array, mx.array]]

#: mlx-lm's sequential recurrence, captured before any rebinding.
SEQUENTIAL_GATED_DELTA_OPS = gated_delta.gated_delta_ops

#: ``g`` is a decay in (0, 1]. The chunkwise form takes its logarithm, so a
#: decay that underflowed to zero is raised to this value.
SMALLEST_DECAY = 1e-30


def unit_lower_inverse(strict_lower: mx.array) -> mx.array:
    """``(I - M)^-1`` for a strictly lower triangular ``M`` of ``[..., C, C]``, with ``C`` a power of two.

    With ``L = [[A, 0], [C, B]]`` unit lower triangular,
    ``L^-1 = [[A^-1, 0], [-B^-1 C A^-1, B^-1]]``. One batched recursive call
    inverts both diagonal blocks, so the depth is ``log2(C)``. A Neumann
    series is not used: on keys that change little between tokens, it cancels
    catastrophically in float32.
    """
    size = strict_lower.shape[-1]
    if size == 1:
        return mx.ones_like(strict_lower)
    half = size // 2
    lower = mx.eye(size, dtype=strict_lower.dtype) - strict_lower
    diagonal = mx.stack([lower[..., :half, :half], lower[..., half:, half:]], axis=-3)
    # The recursion takes M = I - L, so it receives the blocks' strictly lower part.
    inverses = unit_lower_inverse(mx.eye(half, dtype=lower.dtype) - diagonal)
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
    chunk_size: int,
) -> tuple[mx.array, mx.array]:
    """The recurrence over a whole sequence, ``chunk_size`` tokens at a time; returns the output and the final state."""
    batch, length, heads, _ = q.shape
    dtype = q.dtype

    # Work in float32 and in [B, H, T, ...]. A masked position writes nothing
    # (beta 0) and decays nothing (g 1); its output is zeroed at the end.
    if mask is not None:
        keep = mask[:, :, None]
        beta = mx.where(keep, beta, mx.zeros_like(beta))
        g = mx.where(keep, g, mx.ones_like(g))
    q, k, v = (t.astype(mx.float32).transpose(0, 2, 1, 3) for t in (q, k, v))
    beta = beta.astype(mx.float32).transpose(0, 2, 1)
    log_decay = mx.log(mx.maximum(g.astype(mx.float32), SMALLEST_DECAY)).transpose(0, 2, 1)

    # Pad to whole chunks with identity steps.
    padding = (chunk_size - length % chunk_size) % chunk_size
    if padding:
        q, k, v = (mx.pad(t, [(0, 0), (0, 0), (0, padding), (0, 0)]) for t in (q, k, v))
        beta = mx.pad(beta, [(0, 0), (0, 0), (0, padding)])
        log_decay = mx.pad(log_decay, [(0, 0), (0, 0), (0, padding)])
    chunk_count = (length + padding) // chunk_size

    v_beta = v * beta[..., None]
    k_beta = k * beta[..., None]
    q, k, v_beta, k_beta = (t.reshape(batch, heads, chunk_count, chunk_size, -1) for t in (q, k, v_beta, k_beta))
    log_decay = log_decay.reshape(batch, heads, chunk_count, chunk_size)

    # exp(cumulative_i - cumulative_j) for i >= j is the decay between two
    # positions of the same chunk.
    cumulative_log_decay = mx.cumsum(log_decay, axis=-1)
    decay_within = mx.exp(cumulative_log_decay)[..., None]
    lower = mx.tril(mx.ones((chunk_size, chunk_size), dtype=mx.bool_))
    strict_lower = mx.tril(mx.ones((chunk_size, chunk_size), dtype=mx.bool_), k=-1)
    pairwise_difference = cumulative_log_decay[..., :, None] - cumulative_log_decay[..., None, :]
    pairwise = mx.where(lower, mx.exp(mx.where(lower, pairwise_difference, 0.0)), 0.0)

    # The UT transform T = (I + tril(diag(beta) K K^T * decay, -1))^-1 gives
    # the chunk's keys and values as the recurrence writes them.
    strict = mx.where(strict_lower, -(k_beta @ k.transpose(0, 1, 2, 4, 3)) * pairwise, 0.0)
    transform = unit_lower_inverse(strict)
    u = transform @ v_beta
    w = transform @ (k_beta * decay_within)

    # mlx-lm keeps the state as [Dv, Dk]; the products below use [Dk, Dv].
    current = state.astype(mx.float32).transpose(0, 1, 3, 2)
    outputs = []
    for index in range(chunk_count):
        q_chunk, k_chunk = q[:, :, index], k[:, :, index]
        chunk_decay = cumulative_log_decay[:, :, index]
        attention = mx.where(lower, (q_chunk @ k_chunk.transpose(0, 1, 3, 2)) * pairwise[:, :, index], 0.0)
        v_new = u[:, :, index] - w[:, :, index] @ current
        outputs.append((q_chunk * decay_within[:, :, index]) @ current + attention @ v_new)
        end_decay = mx.exp(chunk_decay[..., -1])
        to_end = mx.exp(chunk_decay[..., -1, None] - chunk_decay)[..., None]
        current = current * end_decay[..., None, None] + (k_chunk * to_end).transpose(0, 1, 3, 2) @ v_new

    y = mx.concatenate(outputs, axis=2)[:, :, :length].transpose(0, 2, 1, 3)
    if mask is not None:
        y = mx.where(mask[:, :, None, None], y, mx.zeros_like(y))
    return y.astype(dtype), current.transpose(0, 1, 3, 2)


def chunkwise_gated_delta_ops(chunk_size: int) -> GatedDeltaOps:
    """``mlx_lm.models.gated_delta.gated_delta_ops``, computed ``chunk_size`` tokens at a time.

    The result has the signature and the result of the mlx-lm function:
    ``(q, k, v, g, beta, state=None, mask=None) -> (y, state)``.
    """
    if chunk_size <= 0 or chunk_size & (chunk_size - 1):
        raise ValueError("the recurrence chunk size must be a positive power of two")

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
            return SEQUENTIAL_GATED_DELTA_OPS(q, k, v, g, beta, state, mask)
        batch, _, key_heads, key_dim = q.shape
        value_heads, value_dim = v.shape[-2:]
        if state is None:
            state = mx.zeros((batch, value_heads, value_dim, key_dim), dtype=mx.float32)
        if (repeat := value_heads // key_heads) > 1:
            # Grouped queries and keys serve several value heads; mx.repeat's
            # vjp sums their gradient back over the group.
            q = mx.repeat(q, repeat, -2)
            k = mx.repeat(k, repeat, -2)
        return chunkwise_gated_delta(q, k, v, g, beta, state, mask, chunk_size)

    return gated_delta_ops


@contextmanager
def use_chunkwise_gated_delta(chunk_size: int) -> Iterator[None]:
    """Route mlx-lm's training-mode recurrence through :func:`chunkwise_gated_delta_ops` for the block.

    ``gated_delta_update`` reads ``gated_delta_ops`` from its module at call
    time, so the block rebinds that name and restores it on exit. A chunk size
    of 0 changes nothing.
    """
    if chunk_size == 0:
        yield
        return
    original = gated_delta.gated_delta_ops
    gated_delta.gated_delta_ops = chunkwise_gated_delta_ops(chunk_size)
    try:
        yield
    finally:
        gated_delta.gated_delta_ops = original


__all__ = ["chunkwise_gated_delta", "chunkwise_gated_delta_ops", "use_chunkwise_gated_delta"]
