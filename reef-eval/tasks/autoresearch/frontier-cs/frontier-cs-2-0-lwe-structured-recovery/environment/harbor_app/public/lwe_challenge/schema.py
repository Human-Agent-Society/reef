from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Mapping

from .strict_json import loads_object


MAX_CATALOG_BYTES = 64 * 1024 * 1024
MAX_CATALOG_RECORDS = 200
MATRIX_EXPANSION_DOMAIN = "FCS-STRUCTURED-LWE-MATRIX-v1"

# Provisional operational/native envelope. Phase 4 owns release-specific limits;
# these bounds keep Phase 2 inputs within the planned uint32/uint64 C interface,
# and cap aggregate reconstruction work within one 10,800-second evaluation.
MAX_N = 4096
MAX_M = 65536
MAX_Q = 2**32 - 1
MAX_MATRIX_ENTRIES = 2**26
MAX_CATALOG_MATRIX_ENTRIES = 2**28

_HEX_64 = re.compile(r"[0-9a-f]{64}\Z")
_INSTANCE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_MATRIX_KINDS = frozenset(
    {"uniform", "small_alphabet", "sparse_uniform", "sparse_small_alphabet"}
)
_SECRET_DISTRIBUTION_KINDS = frozenset(
    {
        "uniform_mod_q",
        "iid_alphabet",
        "exact_weight_alphabet",
        "balanced_exact_weight_signed",
        "centered_binomial",
    }
)
_SECRET_PREDICATE_KINDS = frozenset({"alphabet", "mod_q"})
_ERROR_DISTRIBUTION_KINDS = frozenset(
    {
        "truncated_discrete_gaussian",
        "centered_binomial",
        "bounded_uniform",
        "sparse_bounded",
    }
)
_CATALOG_KEYS = frozenset({"schema_version", "instances"})
_INSTANCE_KEYS = frozenset(
    {
        "schema_version",
        "instance_id",
        "n",
        "m",
        "q",
        "matrix",
        "b",
        "secret_distribution",
        "secret",
        "error_distribution",
        "error",
        "instance_digest",
    }
)
_MATRIX_KEYS = frozenset(
    {"kind", "seed_hex", "expansion_domain", "alphabet", "row_weight"}
)
_SECRET_DISTRIBUTION_KEYS = frozenset({"kind", "alphabet", "weight", "eta"})
_SECRET_PREDICATE_KEYS = frozenset(
    {"kind", "alphabet", "min_nonzero", "max_nonzero"}
)
_ERROR_DISTRIBUTION_KEYS = frozenset({"kind", "sigma", "eta", "bound", "weight"})
_ERROR_PREDICATE_KEYS = frozenset(
    {"max_abs", "max_l1", "max_l2_squared", "max_nonzero"}
)
_DIGEST_EXCLUDED_FIELDS = ("instance_digest",)


@dataclass(frozen=True, slots=True)
class MatrixSpec:
    kind: Literal[
        "uniform", "small_alphabet", "sparse_uniform", "sparse_small_alphabet"
    ]
    seed_hex: str
    expansion_domain: str
    alphabet: tuple[int, ...] = ()
    row_weight: int | None = None


@dataclass(frozen=True, slots=True)
class SecretDistributionSpec:
    kind: Literal[
        "uniform_mod_q",
        "iid_alphabet",
        "exact_weight_alphabet",
        "balanced_exact_weight_signed",
        "centered_binomial",
    ]
    alphabet: tuple[int, ...]
    weight: int | None
    eta: int | None


@dataclass(frozen=True, slots=True)
class SecretPredicateSpec:
    kind: Literal["alphabet", "mod_q"]
    alphabet: tuple[int, ...]
    min_nonzero: int
    max_nonzero: int


@dataclass(frozen=True, slots=True)
class ErrorPredicateSpec:
    max_abs: int
    max_l1: int | None
    max_l2_squared: int | None
    max_nonzero: int | None


@dataclass(frozen=True, slots=True)
class ErrorDistributionSpec:
    kind: Literal[
        "truncated_discrete_gaussian",
        "centered_binomial",
        "bounded_uniform",
        "sparse_bounded",
    ]
    sigma: float | None
    eta: int | None
    bound: int
    weight: int | None


@dataclass(frozen=True, slots=True)
class InstanceSpec:
    schema_version: int
    instance_id: str
    n: int
    m: int
    q: int
    matrix: MatrixSpec
    b: tuple[int, ...]
    secret_distribution: SecretDistributionSpec
    secret: SecretPredicateSpec
    error_distribution: ErrorDistributionSpec
    error: ErrorPredicateSpec
    instance_digest: str


@dataclass(frozen=True, slots=True)
class Catalog:
    schema_version: int
    catalog_id: str
    instances: tuple[InstanceSpec, ...]
    _instances_by_id: Mapping[str, InstanceSpec] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        by_id = {instance.instance_id: instance for instance in self.instances}
        if len(by_id) != len(self.instances):
            raise ValueError("duplicate instance_id in catalog")
        object.__setattr__(self, "_instances_by_id", MappingProxyType(by_id))

    @classmethod
    def load(cls, path: str | Path) -> Catalog:
        catalog_path = Path(path)
        raw_bytes = _read_regular_catalog(catalog_path)
        if catalog_path.suffix == ".json":
            catalog_format = "json"
        elif catalog_path.suffix == ".jsonl":
            catalog_format = "jsonl"
        else:
            raise ValueError("catalog path must end in .json or .jsonl")
        return cls._load_bytes(raw_bytes, catalog_format)

    @classmethod
    def load_fd(cls, descriptor: int, catalog_format: Literal["json", "jsonl"]) -> Catalog:
        """Load the exact already-open regular file using an explicit wire format."""

        if type(descriptor) is not int or descriptor < 0:
            raise ValueError("catalog descriptor must be a non-negative integer")
        if catalog_format not in {"json", "jsonl"}:
            raise ValueError("catalog format must be exactly json or jsonl")
        return cls._load_bytes(_read_regular_catalog_fd(descriptor), catalog_format)

    @classmethod
    def _load_bytes(
        cls, raw_bytes: bytes, catalog_format: Literal["json", "jsonl"]
    ) -> Catalog:
        if len(raw_bytes) > MAX_CATALOG_BYTES:
            raise ValueError("catalog exceeds the 64 MiB byte limit")
        if catalog_format == "json":
            schema_version, records = _load_json_records(raw_bytes)
            require_increasing_ids = False
        else:
            schema_version, records = _load_jsonl_records(raw_bytes)
            require_increasing_ids = True

        raw_ids = tuple(_record_instance_id(record) for record in records)
        if len(raw_ids) != len(set(raw_ids)):
            raise ValueError("duplicate instance_id in catalog")
        if require_increasing_ids and any(
            previous >= current for previous, current in zip(raw_ids, raw_ids[1:])
        ):
            raise ValueError("JSONL instance IDs must be strictly increasing")

        remaining_matrix_entries = MAX_CATALOG_MATRIX_ENTRIES
        parsed_instances: list[InstanceSpec] = []
        for record in records:
            instance = _parse_instance(record)
            if instance.m > remaining_matrix_entries // instance.n:
                raise ValueError(
                    "catalog matrix dimensions exceed MAX_CATALOG_MATRIX_ENTRIES"
                )
            remaining_matrix_entries -= instance.n * instance.m
            parsed_instances.append(instance)
        instances = tuple(parsed_instances)
        return cls(
            schema_version=schema_version,
            catalog_id=hashlib.sha256(raw_bytes).hexdigest(),
            instances=instances,
        )

    def get(self, instance_id: str) -> InstanceSpec:
        return self._instances_by_id[instance_id]


def canonical_record_bytes(record: Mapping[str, object]) -> bytes:
    serialized = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    try:
        return serialized.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("canonical JSON contains a Unicode surrogate") from exc


def _read_regular_catalog(path: Path) -> bytes:
    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(path, flags)
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("catalog path must be a regular file")
        data = bytearray()
        limit = MAX_CATALOG_BYTES + 1
        while len(data) < limit:
            chunk = os.read(fd, min(1024 * 1024, limit - len(data)))
            if not chunk:
                break
            data.extend(chunk)
        return bytes(data)
    finally:
        os.close(fd)


def _read_regular_catalog_fd(descriptor: int) -> bytes:
    try:
        before = os.fstat(descriptor)
    except OSError as exc:
        raise ValueError("catalog descriptor is not open") from exc
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("catalog descriptor must refer to a regular file")
    if before.st_size > MAX_CATALOG_BYTES:
        raise ValueError("catalog exceeds the 64 MiB byte limit")
    data = bytearray()
    offset = 0
    while offset < before.st_size:
        try:
            chunk = os.pread(descriptor, min(1024 * 1024, before.st_size - offset), offset)
        except OSError as exc:
            raise ValueError("catalog descriptor cannot be read") from exc
        if not chunk:
            raise ValueError("catalog descriptor was truncated")
        data.extend(chunk)
        offset += len(chunk)
    try:
        after = os.fstat(descriptor)
    except OSError as exc:
        raise ValueError("catalog descriptor changed while reading") from exc
    if (
        (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    ):
        raise ValueError("catalog descriptor changed while reading")
    return bytes(data)


def compute_instance_digest(record: Mapping[str, object]) -> str:
    digest_record = dict(record)
    for key in _DIGEST_EXCLUDED_FIELDS:
        digest_record.pop(key, None)
    return hashlib.sha256(canonical_record_bytes(digest_record)).hexdigest()


def _load_json_records(raw_bytes: bytes) -> tuple[int, tuple[Mapping[str, object], ...]]:
    document = loads_object(
        raw_bytes,
        max_bytes=MAX_CATALOG_BYTES,
        max_depth=16,
        max_nodes=1_000_000,
    )
    _require_exact_keys(document, _CATALOG_KEYS, "catalog")
    schema_version = _schema_version(document["schema_version"], "catalog")
    raw_instances = _require_list(document["instances"], "catalog instances")
    if not raw_instances:
        raise ValueError("catalog must contain at least 1 record")
    if len(raw_instances) > MAX_CATALOG_RECORDS:
        raise ValueError("catalog may contain at most 200 records")
    return schema_version, tuple(
        _require_mapping(value, f"instances[{index}]")
        for index, value in enumerate(raw_instances)
    )


def _load_jsonl_records(
    raw_bytes: bytes,
) -> tuple[int, tuple[Mapping[str, object], ...]]:
    if not raw_bytes:
        raise ValueError("catalog must contain at least 1 record")
    if b"\r" in raw_bytes:
        raise ValueError("JSONL catalogs must use LF line endings")
    if not raw_bytes.endswith(b"\n"):
        raise ValueError("JSONL catalog must end with a final newline")
    lines = raw_bytes[:-1].split(b"\n")
    if any(not line for line in lines):
        raise ValueError("JSONL catalog must not contain blank lines")
    if len(lines) > MAX_CATALOG_RECORDS:
        raise ValueError("catalog may contain at most 200 records")

    records: list[Mapping[str, object]] = []
    for index, line in enumerate(lines):
        record = loads_object(
            line,
            max_bytes=MAX_CATALOG_BYTES,
            max_depth=16,
            max_nodes=100_000,
        )
        if line != canonical_record_bytes(record):
            raise ValueError(f"JSONL record {index} is not canonical")
        records.append(record)
    return 1, tuple(records)


def _record_instance_id(record: Mapping[str, object]) -> str:
    _require_instance_keys(record)
    return _parse_instance_id(record["instance_id"])


def _parse_instance(raw: Mapping[str, object]) -> InstanceSpec:
    _require_instance_keys(raw)
    schema_version = _schema_version(raw["schema_version"], "instance")
    instance_id = _parse_instance_id(raw["instance_id"])
    n = _require_integer(raw["n"], "n")
    m = _require_integer(raw["m"], "m")
    q = _require_integer(raw["q"], "q")
    if not 1 <= n <= MAX_N:
        raise ValueError("n must satisfy 1 <= n <= MAX_N")
    if not 1 <= m <= MAX_M:
        raise ValueError("m must satisfy 1 <= m <= MAX_M")
    if not 3 <= q <= MAX_Q:
        raise ValueError("q must satisfy 3 <= q <= MAX_Q")
    if m > MAX_MATRIX_ENTRIES // n:
        raise ValueError("matrix dimensions exceed MAX_MATRIX_ENTRIES")

    matrix = _parse_matrix(raw["matrix"], n=n, q=q)
    b = _parse_b(raw["b"], m=m, q=q)
    secret_distribution, secret = _parse_secret_specs(
        raw["secret_distribution"], raw["secret"], n=n, q=q
    )
    error_distribution, error = _parse_error_specs(
        raw["error_distribution"], raw["error"], m=m
    )

    instance_digest = _require_string(raw["instance_digest"], "instance_digest")
    if _HEX_64.fullmatch(instance_digest) is None:
        raise ValueError("instance_digest must be lowercase 64-character hexadecimal")
    if instance_digest != compute_instance_digest(raw):
        raise ValueError("instance_digest mismatch")

    return InstanceSpec(
        schema_version=schema_version,
        instance_id=instance_id,
        n=n,
        m=m,
        q=q,
        matrix=matrix,
        b=b,
        secret_distribution=secret_distribution,
        secret=secret,
        error_distribution=error_distribution,
        error=error,
        instance_digest=instance_digest,
    )


def _parse_matrix(raw_value: object, *, n: int, q: int) -> MatrixSpec:
    raw = _require_mapping(raw_value, "matrix")
    _require_exact_keys(raw, _MATRIX_KEYS, "matrix")
    kind = _require_choice(raw["kind"], _MATRIX_KINDS, "matrix kind")
    seed_hex = _require_string(raw["seed_hex"], "matrix seed_hex")
    if _HEX_64.fullmatch(seed_hex) is None:
        raise ValueError("matrix seed_hex must encode exactly 32 lowercase-hex bytes")
    expansion_domain = _require_string(
        raw["expansion_domain"], "matrix expansion_domain"
    )
    if expansion_domain != MATRIX_EXPANSION_DOMAIN:
        raise ValueError(f"matrix expansion_domain must be {MATRIX_EXPANSION_DOMAIN}")
    alphabet = _parse_alphabet(raw["alphabet"], q=q, context="matrix alphabet")
    row_weight = _require_optional_integer(raw["row_weight"], "matrix row_weight")

    if kind in {"uniform", "sparse_uniform"} and alphabet:
        raise ValueError(f"{kind} matrix alphabet must be empty")
    if kind in {"small_alphabet", "sparse_small_alphabet"} and not alphabet:
        raise ValueError(f"{kind} matrix alphabet must be nonempty")
    if kind in {"uniform", "small_alphabet"}:
        if row_weight is not None:
            raise ValueError("dense matrix kinds must not set row_weight")
    else:
        if row_weight is None or not 1 <= row_weight <= n:
            raise ValueError("matrix row_weight must be between 1 and n")
    if kind == "sparse_small_alphabet" and 0 in alphabet:
        raise ValueError("sparse_small_alphabet matrix alphabet must exclude zero")

    return MatrixSpec(
        kind=kind,
        seed_hex=seed_hex,
        expansion_domain=expansion_domain,
        alphabet=alphabet,
        row_weight=row_weight,
    )


def _parse_b(raw_value: object, *, m: int, q: int) -> tuple[int, ...]:
    raw = _require_list(raw_value, "b")
    if len(raw) != m:
        raise ValueError("len(b) must equal m")
    b = tuple(_require_integer(value, f"b[{index}]") for index, value in enumerate(raw))
    if any(value < 0 or value >= q for value in b):
        raise ValueError("each b entry must satisfy 0 <= b_i < q")
    return b


def _parse_secret_specs(
    raw_distribution_value: object,
    raw_predicate_value: object,
    *,
    n: int,
    q: int,
) -> tuple[SecretDistributionSpec, SecretPredicateSpec]:
    raw_distribution = _require_mapping(
        raw_distribution_value, "secret_distribution"
    )
    raw_predicate = _require_mapping(raw_predicate_value, "secret")
    _require_exact_keys(
        raw_distribution, _SECRET_DISTRIBUTION_KEYS, "secret_distribution"
    )
    _require_exact_keys(raw_predicate, _SECRET_PREDICATE_KEYS, "secret")

    distribution_kind = _require_choice(
        raw_distribution["kind"],
        _SECRET_DISTRIBUTION_KINDS,
        "secret distribution kind",
    )
    distribution_alphabet = _parse_alphabet(
        raw_distribution["alphabet"], q=q, context="secret distribution alphabet"
    )
    weight = _require_optional_integer(
        raw_distribution["weight"], "secret distribution weight"
    )
    eta = _require_optional_integer(
        raw_distribution["eta"], "secret distribution eta"
    )

    predicate_kind = _require_choice(
        raw_predicate["kind"], _SECRET_PREDICATE_KINDS, "secret predicate kind"
    )
    predicate_alphabet = _parse_alphabet(
        raw_predicate["alphabet"], q=q, context="secret predicate alphabet"
    )
    min_nonzero = _require_integer(
        raw_predicate["min_nonzero"], "secret min_nonzero"
    )
    max_nonzero = _require_integer(
        raw_predicate["max_nonzero"], "secret max_nonzero"
    )
    if not 0 <= min_nonzero <= max_nonzero <= n:
        raise ValueError("secret nonzero bounds must satisfy 0 <= min <= max <= n")

    if distribution_kind == "uniform_mod_q":
        if distribution_alphabet or predicate_alphabet:
            raise ValueError("uniform_mod_q secret alphabets must be empty")
        if predicate_kind != "mod_q":
            raise ValueError("uniform_mod_q requires the mod_q secret predicate")
        if weight is not None or eta is not None:
            raise ValueError("uniform_mod_q must not set secret weight or eta")
    elif distribution_kind == "iid_alphabet":
        if not distribution_alphabet:
            raise ValueError("iid_alphabet requires a nonempty secret alphabet")
        if predicate_kind != "alphabet":
            raise ValueError("iid_alphabet requires the alphabet secret predicate")
        if distribution_alphabet != predicate_alphabet:
            raise ValueError("secret distribution and predicate alphabets must match")
        if weight is not None or eta is not None:
            raise ValueError("iid_alphabet must not set secret weight or eta")
    elif distribution_kind in {
        "exact_weight_alphabet",
        "balanced_exact_weight_signed",
    }:
        if distribution_kind == "balanced_exact_weight_signed":
            if distribution_alphabet != (-1, 1):
                raise ValueError(
                    "balanced_exact_weight_signed alphabet must be exactly [-1, 1]"
                )
        elif not distribution_alphabet or 0 in distribution_alphabet:
            raise ValueError(
                "exact_weight_alphabet requires a nonempty alphabet excluding zero"
            )
        expected_predicate_alphabet = tuple(sorted((*distribution_alphabet, 0)))
        if predicate_kind != "alphabet" or predicate_alphabet != expected_predicate_alphabet:
            raise ValueError(
                "exact-weight secret predicate alphabet must add zero to the distribution alphabet"
            )
        if weight is None or weight != min_nonzero or weight != max_nonzero:
            raise ValueError("exact secret weight must equal both predicate bounds")
        if distribution_kind == "balanced_exact_weight_signed" and weight % 2:
            raise ValueError(
                "balanced_exact_weight_signed requires an even weight"
            )
        if eta is not None:
            raise ValueError(f"{distribution_kind} must not set secret eta")
    else:
        if eta is None or not 1 <= eta <= (q - 1) // 2:
            raise ValueError(
                "centered_binomial secret eta must satisfy 1 <= eta <= (q - 1) // 2"
            )
        if (
            predicate_kind != "alphabet"
            or not _is_centered_binomial_alphabet(distribution_alphabet, eta)
            or not _is_centered_binomial_alphabet(predicate_alphabet, eta)
        ):
            raise ValueError(
                "centered_binomial secret alphabets must exactly match range(-eta, eta + 1)"
            )
        if weight is not None:
            raise ValueError("centered_binomial must not set secret weight")

    return (
        SecretDistributionSpec(
            kind=distribution_kind,
            alphabet=distribution_alphabet,
            weight=weight,
            eta=eta,
        ),
        SecretPredicateSpec(
            kind=predicate_kind,
            alphabet=predicate_alphabet,
            min_nonzero=min_nonzero,
            max_nonzero=max_nonzero,
        ),
    )


def _is_centered_binomial_alphabet(values: tuple[int, ...], eta: int) -> bool:
    return len(values) == 2 * eta + 1 and all(
        value == index - eta for index, value in enumerate(values)
    )


def _parse_error_specs(
    raw_distribution_value: object,
    raw_predicate_value: object,
    *,
    m: int,
) -> tuple[ErrorDistributionSpec, ErrorPredicateSpec]:
    raw_distribution = _require_mapping(
        raw_distribution_value, "error_distribution"
    )
    raw_predicate = _require_mapping(raw_predicate_value, "error")
    _require_exact_keys(
        raw_distribution, _ERROR_DISTRIBUTION_KEYS, "error_distribution"
    )
    _require_exact_keys(raw_predicate, _ERROR_PREDICATE_KEYS, "error")

    distribution_kind = _require_choice(
        raw_distribution["kind"],
        _ERROR_DISTRIBUTION_KINDS,
        "error distribution kind",
    )
    sigma = _require_optional_number(
        raw_distribution["sigma"], "error distribution sigma"
    )
    eta = _require_optional_integer(
        raw_distribution["eta"], "error distribution eta"
    )
    bound = _require_integer(raw_distribution["bound"], "error distribution bound")
    weight = _require_optional_integer(
        raw_distribution["weight"], "error distribution weight"
    )
    if bound < 0:
        raise ValueError("error distribution bound must be nonnegative")

    max_abs = _require_integer(raw_predicate["max_abs"], "error max_abs")
    max_l1 = _require_optional_nonnegative_integer(
        raw_predicate["max_l1"], "error max_l1"
    )
    max_l2_squared = _require_optional_nonnegative_integer(
        raw_predicate["max_l2_squared"], "error max_l2_squared"
    )
    max_nonzero = _require_optional_nonnegative_integer(
        raw_predicate["max_nonzero"], "error max_nonzero"
    )
    if max_abs < 0:
        raise ValueError("error max_abs must be nonnegative")

    if distribution_kind == "truncated_discrete_gaussian":
        if sigma is None or sigma <= 0:
            raise ValueError("Gaussian error sigma must be finite and positive")
        if eta is not None or weight is not None:
            raise ValueError("Gaussian errors must not set eta or weight")
    elif distribution_kind == "centered_binomial":
        if eta is None or eta < 1 or bound != eta:
            raise ValueError("centered-binomial error requires eta >= 1 and bound == eta")
        if sigma is not None or weight is not None:
            raise ValueError("centered-binomial errors must not set sigma or weight")
    elif distribution_kind == "bounded_uniform":
        if sigma is not None or eta is not None or weight is not None:
            raise ValueError("bounded-uniform errors must not set sigma, eta, or weight")
    else:
        if bound < 1:
            raise ValueError("sparse-bounded error bound must be at least 1")
        if weight is None or not 1 <= weight <= m:
            raise ValueError("sparse-bounded error weight must be between 1 and m")
        if sigma is not None or eta is not None:
            raise ValueError("sparse-bounded errors must not set sigma or eta")
        if max_nonzero is None or max_nonzero < weight:
            raise ValueError(
                "error max_nonzero must be at least the sparse distribution weight"
            )

    if max_abs < bound:
        raise ValueError("error max_abs must be at least the distribution bound")

    return (
        ErrorDistributionSpec(
            kind=distribution_kind,
            sigma=sigma,
            eta=eta,
            bound=bound,
            weight=weight,
        ),
        ErrorPredicateSpec(
            max_abs=max_abs,
            max_l1=max_l1,
            max_l2_squared=max_l2_squared,
            max_nonzero=max_nonzero,
        ),
    )


def _parse_alphabet(raw_value: object, *, q: int, context: str) -> tuple[int, ...]:
    raw = _require_list(raw_value, context)
    values = tuple(
        _require_integer(value, f"{context}[{index}]")
        for index, value in enumerate(raw)
    )
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{context} must be sorted and unique")
    if len({value % q for value in values}) != len(values):
        raise ValueError(f"{context} values must be incongruent modulo q")
    lower = -(q // 2)
    upper = (q - 1) // 2
    if any(value < lower or value > upper for value in values):
        raise ValueError(f"{context} values must be centered representatives modulo q")
    return values


def _parse_instance_id(raw_value: object) -> str:
    value = _require_string(raw_value, "instance_id")
    if _INSTANCE_ID.fullmatch(value) is None:
        raise ValueError(
            "instance_id must contain 1 to 64 ASCII letters, digits, dots, underscores, "
            "or hyphens and must start with a letter or digit"
        )
    return value


def _schema_version(raw_value: object, context: str) -> int:
    value = _require_integer(raw_value, f"{context} schema_version")
    if value != 1:
        raise ValueError(f"{context} schema_version must equal 1")
    return value


def _require_exact_keys(
    raw: Mapping[str, object], expected: frozenset[str], context: str
) -> None:
    keys = set(raw)
    unknown = sorted(keys - expected)
    missing = sorted(expected - keys)
    if unknown:
        raise ValueError(f"unknown {context} fields: {', '.join(unknown)}")
    if missing:
        raise ValueError(f"missing {context} fields: {', '.join(missing)}")


def _require_instance_keys(raw: Mapping[str, object]) -> None:
    keys = set(raw)
    unknown = sorted(keys - _INSTANCE_KEYS)
    missing = sorted(_INSTANCE_KEYS - keys)
    if unknown:
        raise ValueError(f"unknown instance fields: {', '.join(unknown)}")
    if missing:
        raise ValueError(f"missing instance fields: {', '.join(missing)}")


def _require_mapping(raw_value: object, context: str) -> Mapping[str, object]:
    if not isinstance(raw_value, dict):
        raise ValueError(f"{context} must be a JSON object")
    return raw_value


def _require_list(raw_value: object, context: str) -> list[object]:
    if not isinstance(raw_value, list):
        raise ValueError(f"{context} must be a JSON array")
    return raw_value


def _require_string(raw_value: object, context: str) -> str:
    if not isinstance(raw_value, str):
        raise ValueError(f"{context} must be a string")
    return raw_value


def _require_choice(
    raw_value: object, choices: frozenset[str] | set[str], context: str
) -> str:
    value = _require_string(raw_value, context)
    if value not in choices:
        raise ValueError(f"unsupported {context}: {value}")
    return value


def _require_integer(raw_value: object, context: str) -> int:
    if type(raw_value) is not int:
        raise ValueError(f"{context} must be an integer")
    return raw_value


def _require_optional_integer(raw_value: object, context: str) -> int | None:
    if raw_value is None:
        return None
    return _require_integer(raw_value, context)


def _require_optional_nonnegative_integer(
    raw_value: object, context: str
) -> int | None:
    value = _require_optional_integer(raw_value, context)
    if value is not None and value < 0:
        raise ValueError(f"{context} must be nonnegative")
    return value


def _require_number(raw_value: object, context: str) -> float:
    if type(raw_value) not in {int, float}:
        raise ValueError(f"{context} must be a finite number")
    try:
        value = float(raw_value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{context} must be a finite number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{context} must be a finite number")
    return value


def _require_optional_number(raw_value: object, context: str) -> float | None:
    if raw_value is None:
        return None
    return _require_number(raw_value, context)
