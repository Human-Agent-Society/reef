"""Read-only proposer access to the exact installed Harbor harness source."""

import hashlib
import importlib.util
from pathlib import Path


def source_bundle():
    spec = importlib.util.find_spec("harbor")
    if spec is None or spec.origin is None:
        raise ValueError("the pinned Harbor runtime must be installed")
    base = Path(spec.origin).parent
    selected = {base / "agents/base.py", base / "environments/base.py"}
    for directory in ("agents/terminus_2", "llms", "models/agent", "models/trajectories"):
        selected.update((base / directory).rglob("*.py"))
    return {
        "harbor/" + path.relative_to(base).as_posix(): path.read_text()
        for path in sorted(selected)
        if path.is_file() and not path.is_symlink()
    }


def source_manifest(bundle):
    return {path: hashlib.sha256(text.encode()).hexdigest() for path, text in bundle.items()}


def verified_source(expected):
    bundle = source_bundle()
    if source_manifest(bundle) != expected:
        raise ValueError("installed proposer source differs from the committed source manifest")
    return bundle


def read_source(bundle, path, start=0, limit=120):
    if path not in bundle:
        raise ValueError("unknown frozen source path")
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise ValueError("start must be a non-negative line offset")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 240:
        raise ValueError("limit must be between 1 and 240 lines")
    lines = bundle[path].splitlines()
    return {"path": path, "start": start, "total_lines": len(lines), "lines": lines[start : start + limit]}


def search_source(bundle, query, limit=40):
    if not isinstance(query, str) or not query or len(query) > 200:
        raise ValueError("query must be a nonempty literal string of at most 200 characters")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100 matches")
    matches = []
    for path, text in bundle.items():
        for offset, line in enumerate(text.splitlines()):
            if query in line:
                matches.append({"path": path, "line_offset": offset, "text": line[:2000]})
                if len(matches) == limit:
                    return {"matches": matches, "limit_reached": True}
    return {"matches": matches, "limit_reached": False}
