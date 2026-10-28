"""The self referential tools of the serve form: inspect the harness and try a change on this process.

Two built-in tools, built in code and registered only by ``reef-native
serve --self-tools``; the episode form never sees them, so a candidate cannot
pass the checks by calling them, and a tree file cannot take their names. They
run in process whatever ``REEF_NATIVE_ENFORCE`` says, since they are reef's
code and not the tree's. The order a model should use is inspect, then try;
nothing a trial does is served to another session or published.
"""

from __future__ import annotations

import json
import secrets
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from reef.harness.adapters import get_adapter
from reef.harness.adapters.parts import Format
from reef.harness.runners.native import ToolModule, ToolRunner
from reef.harness.runners.native.enforce import ToolFailed
from reef.harness.runners.native.graph import SEED_GRAPH
from reef.harness.runners.native.release_client import ReleaseClient
from reef.harness.tree import HarnessTree, TreePatch

#: How many catalog rows ``harness_inspect("results")`` returns, newest first.
RESULT_ROWS = 20

PATCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": (
        "The change as one patch: each key is a path in the harness tree, each value the file's new text, or null "
        "to remove the file. Paths: native/tools/<name>.py (code defining run(args, workdir) -> str, then the "
        "constants NAME, DESCRIPTION, PARAMETERS as a JSON schema and CAPABILITIES from read, write, exec, "
        "network), native/hooks/<name>.py (code defining listen(payload, next), then NAME and EVENT), "
        "native/graphs/<name>.json (name, start, max_steps, stages, edges), native/agents/<name>.json (name, "
        "prompt, graph, tools, skills, max_steps, max_tool_calls, then), skills/<name>/SKILL.md, rules.md, and "
        "config/models.json (context_window)."
    ),
    "additionalProperties": {"type": ["string", "null"]},
}

INSPECT_DESCRIPTION = (
    "Read the harness you are running in. what=tree returns every file of the live tree by path and the mounted "
    "release id; what=graph shows the main loop graph and every named graph as JSON; what=results lists the "
    "newest releases with the evaluation metrics that admitted each one and, when Reef exposes it, the candidates "
    "the evaluation rejected; what=status shows the mounted release, the follow mode and any pending mount. Read "
    "only. Call it before you change anything."
)
TRY_DESCRIPTION = (
    "Mount a change on this process for the rest of the current turn only. New tools, hooks, rules, skills, "
    "graphs and agents apply from your next step; at the end of the turn the served harness is mounted back. "
    "Nothing is published. A native loop is reviewed code and is refused here."
)


class ServeState(ABC):
    """What the self tools read and drive: the serve process."""

    @property
    @abstractmethod
    def client(self) -> ReleaseClient: ...

    @property
    @abstractmethod
    def release_id(self) -> str | None: ...

    @abstractmethod
    def live_tree(self) -> HarnessTree: ...

    @abstractmethod
    def status(self) -> dict[str, Any]: ...

    @abstractmethod
    def try_patch(self, patch: TreePatch, try_id: str) -> dict[str, Any]: ...


class HostTool(ToolModule):
    """A built-in tool: run in process whatever the enforcer, with a reserved name."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["builtin_tool"] = True
        super().__init__(*args, **kwargs)


def patch_argument(value: Any) -> TreePatch:
    """The ``patch`` argument as a non-empty ``TreePatch``; ToolFailed naming the problem otherwise."""
    try:
        patch = TreePatch.from_json(value)
    except (TypeError, ValueError) as exc:
        raise ToolFailed(f"patch: {exc}") from exc
    if not patch.changes:
        raise ToolFailed("patch must change at least one path")
    return patch


class SelfTools(ToolRunner):
    """The two tools' ``run`` bodies over one serve process."""

    def __init__(self, state: ServeState, action: str = "inspect") -> None:
        self._state = state
        self._action = action

    def __call__(self, args: dict[str, Any], workdir: str, /) -> Any:
        if self._action == "inspect":
            return self.inspect(args, workdir)
        if self._action == "try":
            return self.try_(args, workdir)
        raise ToolFailed(f"unknown self tool action {self._action!r}")

    def inspect(self, args: dict[str, Any], workdir: str) -> Any:
        what = str(args.get("what", ""))
        tree = self._state.live_tree()
        if what == "tree":
            return {"release_id": self._state.release_id, "files": dict(tree.files)}
        if what == "graph":
            graphs = {}
            for path, text in tree.files.items():
                location = get_adapter("native").layout.locate(path)
                if location is not None and location.part.format is Format.NATIVE_GRAPH:
                    graphs[location.name] = json.loads(text)
            return {"main": graphs.get("main", SEED_GRAPH), "graphs": graphs}
        if what == "results":
            return self.results()
        if what == "status":
            return self._state.status()
        raise ToolFailed(f"what must be one of tree, graph, results, status; got {what!r}")

    def results(self) -> dict[str, Any]:
        rows = list(reversed(self._state.client.releases()))[:RESULT_ROWS]
        releases = [
            {
                "release_id": row.get("release_id"),
                "parent_release_id": row.get("parent_release_id"),
                "operation": row.get("operation"),
                "pending": row.get("pending", False),
                "recorded_at": row.get("recorded_at"),
                "evaluation": row.get("metrics"),
            }
            for row in rows
        ]
        head = self._state.release_id or (str(rows[0]["release_id"]) if rows and rows[0].get("release_id") else None)
        rejected = None
        if head is not None:
            manifest = self._state.client.fetch(head)
            evaluation_metrics = manifest.get("evaluation")
            rejected = evaluation_metrics.get("rejected") if isinstance(evaluation_metrics, Mapping) else None
        return {"head": head, "releases": releases, "rejected": rejected}

    def try_(self, args: dict[str, Any], workdir: str) -> Any:
        patch = patch_argument(args.get("patch"))
        for path in patch.changes:
            location = get_adapter("native").layout.locate(path)
            if location is not None and location.part.review:
                raise ToolFailed(f"{path} is reviewed code: a person promotes it")
        return self._state.try_patch(patch, secrets.token_hex(4))


def self_tools(state: ServeState) -> list[HostTool]:
    """The two built-in tools over ``state``, in the order the model should read them."""
    return [
        HostTool(
            "harness_inspect",
            INSPECT_DESCRIPTION,
            {
                "type": "object",
                "properties": {
                    "what": {
                        "type": "string",
                        "enum": ["tree", "graph", "results", "status"],
                        "description": "Which view: tree, graph, results or status.",
                    }
                },
                "required": ["what"],
            },
            SelfTools(state, "inspect"),
            capabilities=("network",),
        ),
        HostTool(
            "harness_try",
            TRY_DESCRIPTION,
            {"type": "object", "properties": {"patch": PATCH_SCHEMA}, "required": ["patch"]},
            SelfTools(state, "try"),
            capabilities=("exec",),
        ),
    ]


__all__ = ["PATCH_SCHEMA", "RESULT_ROWS", "HostTool", "SelfTools", "ServeState", "patch_argument", "self_tools"]
