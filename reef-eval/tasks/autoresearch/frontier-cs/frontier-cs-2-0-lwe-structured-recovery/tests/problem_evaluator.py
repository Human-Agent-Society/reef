"""Frontier-CS entry point for the public structured-LWE evaluator."""

from __future__ import annotations

import hashlib
import math
import os
import stat
import sys
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import TypeAlias


JsonScalar: TypeAlias = None | bool | int | float | str
PlainData: TypeAlias = JsonScalar | list["PlainData"] | dict[str, "PlainData"]

try:
    _SOURCE_PUBLIC_DIR = (
        Path(__file__).resolve().parent / "harbor" / "app" / "public"
    )
    _INITIAL_CATALOG_OVERRIDE = os.environ.get("FCS_STRUCTURED_LWE_CATALOG")
    _IMPORT_PUBLIC_DIR = Path(
        _SOURCE_PUBLIC_DIR
        if _INITIAL_CATALOG_OVERRIDE is not None
        else os.environ.get("FRONTIER_PUBLIC_DIR", _SOURCE_PUBLIC_DIR)
    ).resolve()
    _IMPORT_PUBLIC_DIR_TEXT = str(_IMPORT_PUBLIC_DIR)
    while _IMPORT_PUBLIC_DIR_TEXT in sys.path:
        sys.path.remove(_IMPORT_PUBLIC_DIR_TEXT)
    sys.path.insert(0, _IMPORT_PUBLIC_DIR_TEXT)

    from lwe_challenge.evaluator_core import evaluate_path  # noqa: E402
    from lwe_challenge.schema import MAX_CATALOG_BYTES, Catalog  # noqa: E402
except Exception:
    if __name__ == "__main__":
        print("infrastructure_error", file=sys.stderr)
        raise SystemExit(1) from None
    raise


_SIDECAR_BYTES = len(f"{'0' * 64}  catalog.jsonl\n".encode("ascii"))


def _catalog_path() -> Path:
    override = os.environ.get("FCS_STRUCTURED_LWE_CATALOG")
    if override is not None:
        return Path(override).resolve()
    public_dir = Path(
        os.environ.get("FRONTIER_PUBLIC_DIR", _SOURCE_PUBLIC_DIR)
    ).resolve()
    return (public_dir / "catalog.jsonl").resolve()


def _catalog() -> Catalog:
    path = _catalog_path()
    production = os.environ.get("FCS_STRUCTURED_LWE_CATALOG") is None
    catalog_bytes = _read_regular_bytes(path, max_bytes=MAX_CATALOG_BYTES)
    digest = hashlib.sha256(catalog_bytes).hexdigest()
    if production:
        expected_sidecar = f"{digest}  catalog.jsonl\n".encode("ascii")
        try:
            sidecar = _read_regular_bytes(
                path.with_name("catalog.sha256"), max_bytes=_SIDECAR_BYTES
            )
        except ValueError:
            raise ValueError("catalog.sha256 is not a valid sidecar") from None
        if sidecar != expected_sidecar:
            raise ValueError("catalog.sha256 does not match catalog.jsonl")
    return _load_catalog(str(path), digest, production)


def _read_regular_bytes(path: Path, *, max_bytes: int) -> bytes:
    flags = (
        os.O_RDONLY
        | os.O_NONBLOCK
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    fd = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("catalog assets must be regular files")
        data = bytearray()
        limit = max_bytes + 1
        while len(data) < limit:
            chunk = os.read(fd, min(1024 * 1024, limit - len(data)))
            if not chunk:
                break
            data.extend(chunk)
    finally:
        os.close(fd)
    if len(data) > max_bytes:
        raise ValueError("catalog asset exceeds its byte limit")
    return bytes(data)


@lru_cache(maxsize=8)
def _load_catalog(path: str, digest: str, production: bool) -> Catalog:
    catalog = Catalog.load(path)
    if catalog.catalog_id != digest:
        raise RuntimeError("catalog changed while it was being verified")
    if production and len(catalog.instances) != 200:
        raise ValueError("production catalog must contain exactly 200 instances")
    return catalog


def _plain_data(value: object) -> PlainData:
    if isinstance(value, float) and not math.isfinite(value):
        raise TypeError("public metrics require finite float values")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        copied: dict[str, PlainData] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("public metrics mappings require string keys")
            copied[key] = _plain_data(item)
        return copied
    if isinstance(value, tuple):
        return [_plain_data(item) for item in value]
    raise TypeError("public metrics contain a non-JSON value")


def prepare() -> dict[str, object]:
    catalog = _catalog()
    return {
        "instance_count": len(catalog.instances),
        "catalog_id": catalog.catalog_id,
    }


def evaluate(solution_path: str) -> tuple[float, float, str, dict[str, object]]:
    catalog = _catalog()
    result = evaluate_path(solution_path, catalog=catalog)
    metrics = _plain_data(result.metrics)
    if not isinstance(metrics, dict):
        raise TypeError("evaluator metrics must be a mapping")
    return result.score, result.score_unbounded, result.message, metrics


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: evaluator.py SOLUTION_JSON", file=sys.stderr)
        return 2
    try:
        score, score_unbounded, message, _metrics = evaluate(argv[1])
    except Exception:
        print("infrastructure_error", file=sys.stderr)
        return 1
    print(message, file=sys.stderr)
    print(f"{score:.12f} {score_unbounded:.12f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
