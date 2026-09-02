"""JSON file helpers shared by the runner and the harness modules."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object in {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    """Write through a sibling temporary file so an interrupted write never leaves a torn file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_once(path: Path, value: dict[str, Any], label: str) -> None:
    """Write ``value`` unless the file already holds it; refuse a different one."""
    expected = json.loads(json.dumps(value, sort_keys=True, default=_json_default))
    if path.exists():
        if read_json(path) != expected:
            raise RuntimeError(f"existing {label} does not match this run: {path}")
        return
    write_json(path, value)


def fingerprint(value: Any) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(value)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return str(value)
