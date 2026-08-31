from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .matrix import matvec_mod
from .schema import InstanceSpec


@dataclass(frozen=True, slots=True)
class WitnessVerdict:
    ok: bool
    code: str
    max_abs_error: int | None
    l1_error: int | None
    l2_squared_error: int | None


def centered_mod(value: int, q: int) -> int:
    """Return the canonical centered representative modulo ``q``."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("value must be an integer")
    if not isinstance(q, int) or isinstance(q, bool):
        raise TypeError("q must be an integer")
    if q < 1:
        raise ValueError("q must be at least 1")

    residue = value % q
    if residue >= (q + 1) // 2:
        return residue - q
    return residue


def validate_secret_shape(
    instance: InstanceSpec, secret: Sequence[int]
) -> str | None:
    try:
        secret_length = len(secret)
    except TypeError:
        return "secret_type"
    if secret_length != instance.n:
        return "wrong_length"
    if any(
        not isinstance(value, int) or isinstance(value, bool) for value in secret
    ):
        return "secret_type"
    if instance.secret.kind == "alphabet" and any(
        value not in instance.secret.alphabet for value in secret
    ):
        return "secret_alphabet"
    if instance.secret.kind == "mod_q" and any(
        value < 0 or value >= instance.q for value in secret
    ):
        return "secret_mod_q"
    nonzero_count = sum(value != 0 for value in secret)
    if not (
        instance.secret.min_nonzero
        <= nonzero_count
        <= instance.secret.max_nonzero
    ):
        return "secret_weight"
    return None


def residual(
    instance: InstanceSpec, secret: Sequence[int]
) -> tuple[int, ...]:
    shape_code = validate_secret_shape(instance, secret)
    if shape_code is not None:
        raise ValueError(f"invalid secret: {shape_code}")
    product = matvec_mod(instance, secret)
    return tuple(
        centered_mod(public_value - product_value, instance.q)
        for public_value, product_value in zip(instance.b, product)
    )


def validate_secret(
    instance: InstanceSpec, secret: Sequence[int]
) -> WitnessVerdict:
    shape_code = validate_secret_shape(instance, secret)
    if shape_code is not None:
        return WitnessVerdict(False, shape_code, None, None, None)
    error_values = residual(instance, secret)
    max_abs_error = max(abs(value) for value in error_values)
    l1_error = sum(abs(value) for value in error_values)
    l2_squared_error = sum(value * value for value in error_values)
    if max_abs_error > instance.error.max_abs:
        return WitnessVerdict(
            False,
            "error_linf",
            max_abs_error,
            l1_error,
            l2_squared_error,
        )
    if instance.error.max_l1 is not None and l1_error > instance.error.max_l1:
        return WitnessVerdict(
            False,
            "error_l1",
            max_abs_error,
            l1_error,
            l2_squared_error,
        )
    if (
        instance.error.max_l2_squared is not None
        and l2_squared_error > instance.error.max_l2_squared
    ):
        return WitnessVerdict(
            False,
            "error_l2",
            max_abs_error,
            l1_error,
            l2_squared_error,
        )
    error_nonzero_count = sum(value != 0 for value in error_values)
    if (
        instance.error.max_nonzero is not None
        and error_nonzero_count > instance.error.max_nonzero
    ):
        return WitnessVerdict(
            False,
            "error_weight",
            max_abs_error,
            l1_error,
            l2_squared_error,
        )
    return WitnessVerdict(
        True,
        "ok",
        max_abs_error,
        l1_error,
        l2_squared_error,
    )
