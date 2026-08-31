"""Aggregate Codex JSONL usage across all CORAL workers and restarts."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def usage_records(root: Path, model: str) -> Iterator[dict[str, Any]]:
    """Yield one normalized record per completed Codex turn."""
    for log_path in sorted(root.rglob("*.log")):
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            usage = (
                event.get("usage") if event.get("type") == "turn.completed" else None
            )
            if not isinstance(usage, dict):
                continue
            yield {
                "model": model,
                "input_tokens": int(usage.get("input_tokens", 0) or 0),
                "cached_input_tokens": int(usage.get("cached_input_tokens", 0) or 0),
                "output_tokens": int(usage.get("output_tokens", 0) or 0),
            }


def main() -> int:
    root = Path(sys.argv[1])
    model = sys.argv[2]
    for record in usage_records(root, model):
        print(json.dumps(record, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
