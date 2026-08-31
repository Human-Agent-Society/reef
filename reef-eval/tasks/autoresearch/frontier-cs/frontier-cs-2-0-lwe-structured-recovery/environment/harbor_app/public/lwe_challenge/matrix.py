from __future__ import annotations

from collections.abc import Iterator, Sequence

from .schema import InstanceSpec
from .shake import ShakeStream


_DENSE_COEFFICIENT_TAG = b"\x01"
_SPARSE_SUPPORT_TAG = b"\x02"
_SPARSE_COEFFICIENT_TAG = b"\x03"


def _stream_domain(instance: InstanceSpec, row_index: int, tag: bytes) -> bytes:
    expansion_domain = instance.matrix.expansion_domain.encode("utf-8")
    instance_id = instance.instance_id.encode("ascii")
    return (
        len(expansion_domain).to_bytes(4, "little")
        + expansion_domain
        + len(instance_id).to_bytes(4, "little")
        + instance_id
        + row_index.to_bytes(8, "little")
        + tag
    )


def _row(instance: InstanceSpec, row_index: int) -> tuple[int, ...]:
    seed = bytes.fromhex(instance.matrix.seed_hex)
    if instance.matrix.kind in {"uniform", "small_alphabet"}:
        stream = ShakeStream(
            domain=_stream_domain(instance, row_index, _DENSE_COEFFICIENT_TAG),
            seed=seed,
        )
        if instance.matrix.kind == "uniform":
            return tuple(stream.randbelow(instance.q) for _ in range(instance.n))
        return tuple(
            instance.matrix.alphabet[
                stream.randbelow(len(instance.matrix.alphabet))
            ]
            for _ in range(instance.n)
        )

    if instance.matrix.kind not in {"sparse_uniform", "sparse_small_alphabet"}:
        raise ValueError(f"unsupported matrix kind: {instance.matrix.kind}")
    row_weight = instance.matrix.row_weight
    if row_weight is None:
        raise ValueError("sparse matrix must define row_weight")

    support_stream = ShakeStream(
        domain=_stream_domain(instance, row_index, _SPARSE_SUPPORT_TAG),
        seed=seed,
    )
    support: set[int] = set()
    while len(support) < row_weight:
        support.add(support_stream.randbelow(instance.n))

    coefficient_stream = ShakeStream(
        domain=_stream_domain(instance, row_index, _SPARSE_COEFFICIENT_TAG),
        seed=seed,
    )
    row = [0] * instance.n
    for column in sorted(support):
        if instance.matrix.kind == "sparse_uniform":
            row[column] = coefficient_stream.randbelow(instance.q - 1) + 1
        else:
            row[column] = instance.matrix.alphabet[
                coefficient_stream.randbelow(len(instance.matrix.alphabet))
            ]
    return tuple(row)


def iter_rows(instance: InstanceSpec) -> Iterator[tuple[int, ...]]:
    for row_index in range(instance.m):
        yield _row(instance, row_index)


def materialize_row_block(
    instance: InstanceSpec, start: int, stop: int
) -> tuple[tuple[int, ...], ...]:
    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(stop, int)
        or isinstance(stop, bool)
    ):
        raise TypeError("row block bounds must be integers")
    if not 0 <= start <= stop <= instance.m:
        raise ValueError("row block bounds must satisfy 0 <= start <= stop <= m")
    return tuple(_row(instance, row_index) for row_index in range(start, stop))


def materialize_rows(instance: InstanceSpec) -> tuple[tuple[int, ...], ...]:
    return materialize_row_block(instance, 0, instance.m)


def matvec_mod(
    instance: InstanceSpec, secret: Sequence[int]
) -> tuple[int, ...]:
    if len(secret) != instance.n:
        raise ValueError("secret length must equal n")
    if any(
        not isinstance(value, int) or isinstance(value, bool) for value in secret
    ):
        raise TypeError("secret entries must be integers")
    return tuple(
        sum(coefficient * value for coefficient, value in zip(row, secret))
        % instance.q
        for row in iter_rows(instance)
    )
