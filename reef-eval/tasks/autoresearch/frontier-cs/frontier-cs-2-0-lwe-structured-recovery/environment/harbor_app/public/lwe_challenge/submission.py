"""Bounded parsing for untrusted cumulative solution ledgers.

Lexical, encoding, and resource-limit failures are normalized to
``SubmissionContractError(code="invalid_json")`` without reflecting parser details;
decoded top-level contract failures use their own stable codes. Individual solution
failures are returned as one rejection per invalid occurrence, in input order.
Repeated safe IDs take precedence over other per-record codes and invalidate every
occurrence. Consequently, ``invalid_records == len(rejections)``.

Stable rejection codes are ``record_not_object``, ``invalid_instance_id``,
``invalid_record_fields``, ``invalid_secret``, ``unknown_instance_id``, and
``duplicate_instance_id``. Rejections include a submitted ID only after it passes
the safe token grammar and never include a submitted secret value. For unique
occurrences, the code priority follows that list (excluding the duplicate code).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .schema import Catalog, MAX_N
from .strict_json import JsonContractError, loads_object


MAX_SUBMISSION_BYTES = 2_000_000
MAX_SUBMISSION_RECORDS = 200
MAX_SUBMISSION_SECRET_COMPONENTS = MAX_N
MAX_SUBMISSION_NODES = 5 + MAX_SUBMISSION_RECORDS * (
    MAX_SUBMISSION_SECRET_COMPONENTS + 5
)
MAX_SECRET_ABS = 2**63 - 1
_INSTANCE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")


class SubmissionContractError(ValueError):
    """The input cannot be parsed or violates the submission-ledger contract."""

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.code = message if code is None else code


@dataclass(frozen=True, slots=True)
class SubmissionRejection:
    instance_id: str | None
    code: str


@dataclass(frozen=True, slots=True)
class ParsedSubmission:
    records: Mapping[str, tuple[int, ...]]
    duplicate_ids: tuple[str, ...]
    conflicted_ids: tuple[str, ...]
    unknown_ids: tuple[str, ...]
    rejections: tuple[SubmissionRejection, ...]
    invalid_records: int


@dataclass(frozen=True, slots=True)
class _Occurrence:
    instance_id: str | None
    base_code: str | None


def parse_submission(
    data: bytes,
    *,
    catalog: Catalog,
    max_bytes: int = MAX_SUBMISSION_BYTES,
    max_records: int = MAX_SUBMISSION_RECORDS,
) -> ParsedSubmission:
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    if type(max_records) is not int or max_records < 1:
        raise ValueError("max_records must be a positive integer")
    byte_limit = min(max_bytes, MAX_SUBMISSION_BYTES)
    record_limit = min(max_records, MAX_SUBMISSION_RECORDS)
    document = None
    try:
        document = loads_object(
            data,
            max_bytes=byte_limit,
            max_depth=4,
            max_nodes=(
                5
                + record_limit * (MAX_SUBMISSION_SECRET_COMPONENTS + 5)
            ),
        )
    except JsonContractError:
        pass
    if document is None:
        raise SubmissionContractError("invalid_json") from None
    if set(document) != {"schema_version", "solutions"}:
        raise SubmissionContractError(
            "submission fields must be exactly schema_version and solutions",
            code="invalid_submission_fields",
        )
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != 1
    ):
        raise SubmissionContractError(
            "submission schema_version must be integer 1",
            code="unsupported_schema_version",
        )
    solutions = document["solutions"]
    if not isinstance(solutions, list):
        raise SubmissionContractError(
            "submission solutions must be a JSON array",
            code="invalid_solutions",
        )
    if len(solutions) > record_limit:
        raise SubmissionContractError(
            f"submission solutions may contain at most {record_limit} records",
            code="too_many_records",
        )

    candidate_records: dict[str, tuple[int, ...]] = {}
    seen_vectors: dict[str, set[tuple[int, ...]]] = {}
    occurrence_counts: dict[str, int] = {}
    occurrences: list[_Occurrence] = []
    unknown_ids: set[str] = set()
    known_ids = {instance.instance_id for instance in catalog.instances}
    for raw_record in solutions:
        if not isinstance(raw_record, dict):
            occurrences.append(_Occurrence(None, "record_not_object"))
            continue
        raw_instance_id = raw_record.get("instance_id")
        if (
            not isinstance(raw_instance_id, str)
            or _INSTANCE_ID.fullmatch(raw_instance_id) is None
        ):
            occurrences.append(_Occurrence(None, "invalid_instance_id"))
            continue
        instance_id = raw_instance_id
        occurrence_counts[instance_id] = occurrence_counts.get(instance_id, 0) + 1
        if instance_id not in known_ids:
            unknown_ids.add(instance_id)
        fields_are_exact = set(raw_record) == {"instance_id", "secret"}
        raw_secret = raw_record.get("secret")
        secret_is_valid = (
            isinstance(raw_secret, list)
            and len(raw_secret) <= MAX_SUBMISSION_SECRET_COMPONENTS
            and not any(
                type(value) is not int or abs(value) > MAX_SECRET_ABS
                for value in raw_secret
            )
        )
        if secret_is_valid:
            secret = tuple(raw_secret)
            seen_vectors.setdefault(instance_id, set()).add(secret)
        if not fields_are_exact:
            occurrences.append(_Occurrence(instance_id, "invalid_record_fields"))
            continue
        if not secret_is_valid:
            occurrences.append(_Occurrence(instance_id, "invalid_secret"))
            continue
        if instance_id in known_ids:
            candidate_records[instance_id] = secret
            occurrences.append(_Occurrence(instance_id, None))
        else:
            occurrences.append(_Occurrence(instance_id, "unknown_instance_id"))

    duplicate_ids = {
        instance_id
        for instance_id, count in occurrence_counts.items()
        if count > 1
    }
    records = {
        instance_id: candidate_records[instance_id]
        for instance_id in sorted(candidate_records)
        if instance_id not in duplicate_ids
    }

    conflicted_ids = {
        instance_id
        for instance_id, vectors in seen_vectors.items()
        if len(vectors) > 1
    }
    rejections = tuple(
        SubmissionRejection(occurrence.instance_id, code)
        for occurrence in occurrences
        if (
            code := (
                "duplicate_instance_id"
                if occurrence.instance_id in duplicate_ids
                else occurrence.base_code
            )
        )
        is not None
    )
    return ParsedSubmission(
        records=MappingProxyType(records),
        duplicate_ids=tuple(sorted(duplicate_ids)),
        conflicted_ids=tuple(sorted(conflicted_ids)),
        unknown_ids=tuple(sorted(unknown_ids)),
        rejections=rejections,
        invalid_records=len(rejections),
    )
