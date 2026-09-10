"""The native harness's starting tools and hooks as node entries: a seed the evolution loop can mutate."""

from __future__ import annotations

from typing import Any

_READ_FILE = """\
from pathlib import Path


def run(args, workdir):
    root = Path(workdir).resolve()
    path = (root / str(args.get("path", ""))).resolve()
    if root not in path.parents and path != root:
        return "refused: path escapes the workspace"
    if not path.is_file():
        return f"no such file: {args.get('path', '')}"
    return path.read_text(encoding="utf-8", errors="replace")
"""

_WRITE_FILE = """\
from pathlib import Path


def run(args, workdir):
    root = Path(workdir).resolve()
    path = (root / str(args.get("path", ""))).resolve()
    if root not in path.parents:
        return "refused: path escapes the workspace"
    content = str(args.get("content", ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} characters to {args.get('path', '')}"
"""

_RUN_BASH = """\
import subprocess


def run(args, workdir):
    command = str(args.get("command", ""))
    if not command.strip():
        return "refused: empty command"
    try:
        done = subprocess.run(
            ["bash", "-lc", command], cwd=workdir, capture_output=True, text=True, timeout=60, check=False
        )
    except subprocess.TimeoutExpired:
        return "timed out after 60s"
    out = (done.stdout or "") + (("\\n" + done.stderr) if done.stderr else "")
    return f"exit {done.returncode}\\n{out}"
"""


_EXECUTE = """\
import subprocess
import sys
from pathlib import Path


def run(args, workdir):
    code = str(args.get("code", ""))
    if not code.strip():
        return "refused: empty code"
    # The other tools are plain modules beside this one: the block imports one by name and calls run(args, WORKDIR).
    tools = str(Path(__file__).resolve().parent)
    prelude = f"import sys\\nsys.path.insert(0, {tools!r})\\nWORKDIR = {str(workdir)!r}\\n"
    try:
        done = subprocess.run(
            [sys.executable, "-c", prelude + code], cwd=workdir, capture_output=True, text=True, timeout=60, check=False
        )
    except subprocess.TimeoutExpired:
        return "timed out after 60s"
    out = (done.stdout or "") + (("\\n" + done.stderr) if done.stderr else "")
    return f"exit {done.returncode}\\n{out}"
"""

_LOOP_GUARD = """\
import json

THRESHOLDS = (3, 5, 8)
_chain = {"key": None, "count": 0}


def listen(payload, next):
    decision = next()
    key = json.dumps([payload["name"], payload["arguments"]], sort_keys=True, default=str)
    _chain["count"] = _chain["count"] + 1 if key == _chain["key"] else 1
    _chain["key"] = key
    if _chain["count"] in THRESHOLDS:
        reminder = (
            f"You have called {payload['name']} with the same arguments {_chain['count']} times in a row. "
            "Change approach or answer."
        )
        return {**decision, "contexts": [*decision.get("contexts", []), reminder]}
    return decision
"""


def _tool(
    name: str, description: str, parameters: dict[str, Any], code: str, capabilities: list[str]
) -> dict[str, Any]:
    # The entry id is the node name, so a proposer that sees only (kind, config) pairs can address it.
    return {
        "id": name,
        "name": "native_tool",
        "config": {
            "name": name,
            "description": description,
            "parameters": parameters,
            "capabilities": capabilities,
            "code": code,
        },
    }


def _hook(name: str, event: str, code: str) -> dict[str, Any]:
    return {"id": name, "name": "native_hook", "config": {"name": name, "event": event, "code": code}}


SEED_TOOLS: tuple[dict[str, Any], ...] = (
    _tool(
        "read_file",
        "Read a text file in the workspace.",
        {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        _READ_FILE,
        ["read"],
    ),
    _tool(
        "write_file",
        "Write a text file in the workspace, creating parent directories.",
        {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
        _WRITE_FILE,
        ["write"],
    ),
    _tool(
        "run_bash",
        "Run a bash command in the workspace and return its exit code and output.",
        {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
        _RUN_BASH,
        # A shell can do anything a process can; declared in full so a hook that denies one of them denies bash.
        ["exec", "write", "network"],
    ),
    _tool(
        "execute",
        "Run a Python code block in the workspace; the other tools are importable by name and WORKDIR is set.",
        {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
        _EXECUTE,
        ["exec", "write", "network"],
    ),
)

#: The loop guard is a hook, not loop code: a tree may retune or drop it.
SEED_HOOKS: tuple[dict[str, Any], ...] = (_hook("loop_guard", "post_execute", _LOOP_GUARD),)

#: Today's loop as a graph: think, act while the model calls tools, end when it answers. The loop runs this
#: when a tree carries no graph, so an old tree behaves as before; as a node a proposer may rewrite it.
SEED_GRAPH: dict[str, Any] = {
    "name": "main",
    "start": "think",
    "max_steps": 12,
    "stages": {"think": {"kind": "model"}, "act": {"kind": "tools"}, "done": {"kind": "end", "reason": "completed"}},
    "edges": [
        {"from": "think", "when": "tool_calls", "to": "act"},
        {"from": "think", "when": "text", "to": "done"},
        {"from": "act", "when": "done", "to": "think"},
    ],
}
SEED_GRAPHS: tuple[dict[str, Any], ...] = ({"id": "main", "name": "native_graph", "config": SEED_GRAPH},)

SEED_NODES: tuple[dict[str, Any], ...] = (*SEED_TOOLS, *SEED_HOOKS, *SEED_GRAPHS)

__all__ = ["SEED_GRAPH", "SEED_GRAPHS", "SEED_HOOKS", "SEED_NODES", "SEED_TOOLS"]
