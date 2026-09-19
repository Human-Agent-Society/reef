"""The child side of the bwrap enforcer: read ``{path, arguments, workdir}`` on stdin, run the tool once, and reply ``{ok, text|error}`` on the saved stdout; it imports nothing from reef."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any


def _run(request: dict[str, Any]) -> str:
    path = Path(str(request["path"]))
    spec = importlib.util.spec_from_file_location(f"reef_native_tool_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"{path.name} is not an importable module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.run(request["arguments"], str(request["workdir"]))
    return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)


def main() -> int:
    # The reply owns the original stdout; what the tool or its children write to fd 1 goes to stderr instead.
    reply_fd = os.dup(1)
    os.dup2(2, 1)
    request = json.loads(sys.stdin.read())
    try:
        reply: dict[str, Any] = {"ok": True, "text": _run(request)}
    except Exception as exc:
        reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    with os.fdopen(reply_fd, "w", encoding="utf-8") as out:
        json.dump(reply, out, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
