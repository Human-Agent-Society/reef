"""Secretless scoring for cumulative structured-LWE witness submissions.

The result exposes only stable codes, counts, public IDs, and public catalog
metadata.  Candidate secrets, residuals, norms, paths, and exception text are
never copied into the public result.
"""

from __future__ import annotations

import errno
import os
import stat
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .schema import Catalog
from .strict_json import JsonContractError
from .submission import (
    MAX_SUBMISSION_BYTES,
    SubmissionContractError,
    parse_submission,
)
from .verification import validate_secret

_MAX_INVALID_EXAMPLES = 20


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Immutable public score and aggregate diagnostics."""

    score: float
    score_unbounded: float
    message: str
    metrics: Mapping[str, object]
    solved_ids: tuple[str, ...]


def evaluate_bytes(data: bytes, *, catalog: Catalog) -> EvaluationResult:
    """Parse and score witnesses against public instance predicates."""
    instance_count = len(catalog.instances)
    if instance_count == 0:
        raise ValueError("catalog must contain at least one instance")

    try:
        parsed = parse_submission(data, catalog=catalog)
    except (SubmissionContractError, JsonContractError) as exc:
        code = (
            exc.code
            if isinstance(exc, SubmissionContractError)
            else "invalid_json"
        )
        return _submission_invalid(catalog, code)
    public_instance_ids = frozenset(
        instance.instance_id for instance in catalog.instances
    )
    rejection_counts = Counter(rejection.code for rejection in parsed.rejections)
    invalid_examples = [
        (
            rejection.instance_id
            if rejection.instance_id in public_instance_ids
            else None,
            rejection.code,
        )
        for rejection in parsed.rejections[:_MAX_INVALID_EXAMPLES]
    ]
    solved: list[str] = []
    candidate_invalid_count = 0
    for instance_id, secret in parsed.records.items():
        verdict = validate_secret(catalog.get(instance_id), secret)
        if verdict.ok:
            solved.append(instance_id)
        else:
            candidate_invalid_count += 1
            rejection_counts[verdict.code] += 1
            if len(invalid_examples) < _MAX_INVALID_EXAMPLES:
                invalid_examples.append((instance_id, verdict.code))
    solved_ids = tuple(sorted(solved))
    solved_count = len(solved_ids)
    metrics = _build_metrics(
        catalog=catalog,
        solved_ids=solved_ids,
        submitted_count=len(parsed.records) + parsed.invalid_records,
        duplicate_count=len(parsed.duplicate_ids),
        conflict_count=len(parsed.conflicted_ids),
        unknown_count=len(parsed.unknown_ids),
        invalid_count=parsed.invalid_records + candidate_invalid_count,
        rejection_counts=rejection_counts,
        invalid_examples=tuple(invalid_examples),
    )
    return EvaluationResult(
        score=100.0 * solved_count / instance_count,
        score_unbounded=float(solved_count),
        message=(
            f"scored solved={solved_count} "
            f"submitted={metrics['submitted_count']} "
            f"invalid={metrics['invalid_count']} "
            f"duplicates={metrics['duplicate_count']} "
            f"conflicts={metrics['conflict_count']} "
            f"unknown={metrics['unknown_count']}"
        ),
        metrics=metrics,
        solved_ids=solved_ids,
    )


def evaluate_path(path: str | Path, *, catalog: Catalog) -> EvaluationResult:
    """Boundedly read a regular-file ledger and score its public witnesses."""
    if len(catalog.instances) == 0:
        raise ValueError("catalog must contain at least one instance")

    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        code = _expected_path_error_code(exc)
        if code is None:
            raise
        return _submission_invalid(catalog, code)
    try:
        try:
            metadata = os.fstat(fd)
            if not stat.S_ISREG(metadata.st_mode):
                return _submission_invalid(catalog, "submission_not_regular")
            if metadata.st_size > MAX_SUBMISSION_BYTES:
                return _submission_invalid(catalog, "submission_too_large")
            data = _read_bounded(fd)
        except OSError as exc:
            code = _expected_path_error_code(exc)
            if code is None:
                raise
            return _submission_invalid(catalog, code)
    finally:
        os.close(fd)
    if len(data) > MAX_SUBMISSION_BYTES:
        return _submission_invalid(catalog, "submission_too_large")
    return evaluate_bytes(data, catalog=catalog)


def _read_bounded(fd: int) -> bytes:
    limit = MAX_SUBMISSION_BYTES + 1
    data = bytearray()
    while len(data) < limit:
        chunk = os.read(fd, limit - len(data))
        if not chunk:
            break
        data.extend(chunk)
    return bytes(data)


def _expected_path_error_code(exc: OSError) -> str | None:
    if exc.errno == errno.ENOENT:
        return "submission_missing"
    if exc.errno in {errno.EACCES, errno.ELOOP, errno.ENOTDIR, errno.EPERM}:
        return "submission_unreadable"
    if exc.errno == errno.EISDIR:
        return "submission_not_regular"
    return None


def _submission_invalid(catalog: Catalog, code: str) -> EvaluationResult:
    return EvaluationResult(
        score=0.0,
        score_unbounded=0.0,
        message=f"submission_invalid code={code}",
        metrics=_build_metrics(
            catalog=catalog,
            solved_ids=(),
            submitted_count=0,
            duplicate_count=0,
            conflict_count=0,
            unknown_count=0,
            invalid_count=0,
            rejection_counts={},
            invalid_examples=(),
        ),
        solved_ids=(),
    )


def _build_metrics(
    *,
    catalog: Catalog,
    solved_ids: tuple[str, ...],
    submitted_count: int,
    duplicate_count: int,
    conflict_count: int,
    unknown_count: int,
    invalid_count: int,
    rejection_counts: Mapping[str, int],
    invalid_examples: tuple[tuple[str | None, str], ...],
) -> Mapping[str, object]:
    solved_set = frozenset(solved_ids)
    return MappingProxyType(
        {
            "instance_count": len(catalog.instances),
            "solved_count": len(solved_ids),
            "submitted_count": submitted_count,
            "duplicate_count": duplicate_count,
            "conflict_count": conflict_count,
            "unknown_count": unknown_count,
            "invalid_count": invalid_count,
            "rejection_code_counts": MappingProxyType(
                dict(sorted(rejection_counts.items()))
            ),
            "invalid_examples": invalid_examples,
            "solved_ids": solved_ids,
        }
    )
