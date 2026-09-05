"""The self referential tools of the serve form: inspect the harness, try a change on this process, propose it to Reef.

Three host plane tools, built in code and registered only by ``reef-native
serve --self-tools``; the episode form never sees them, so a candidate cannot
win the gate by calling them, and a tree entry cannot take their names. They
run in process whatever ``REEF_NATIVE_ENFORCE`` says, since they are reef's
code and not the tree's. The order a model should use is inspect, then try,
then propose: a trial that helped is one ``harness_propose`` away from the
gate, and nothing a trial does is served to another session or published.
"""

from __future__ import annotations

import json
import secrets
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from reef.harness.native import ToolModule
from reef.harness.native.enforce import ToolFailed
from reef.harness.native.release_client import ReleaseClient
from reef.harness.native.seed import SEED_GRAPH
from reef.train.cordis_backend.strategies import Mutation, MutationError

#: The names the host plane owns; a tree entry that takes one fails to mount.
RESERVED_NAMES = ("harness_inspect", "harness_propose", "harness_try")
#: How many catalog rows ``harness_inspect("verdicts")`` returns, newest first.
VERDICT_ROWS = 20

_MUTATIONS_SCHEMA: dict[str, Any] = {
    "type": "array",
    "description": (
        "The changes, applied in order as one proposal. Each is an object {op, id, options}: op is create, "
        "update or remove; id is the entry id (for a tool, hook, skill, graph or agent the entry id is its name); "
        "options is {name: <kind>, config: {...}} for create and update and is left out for remove. Kinds: "
        "native_tool (config: name, description, parameters as a JSON schema, capabilities from read, write, "
        "exec, network, and code defining run(args, workdir) -> str), native_hook (name, event, code defining "
        "listen(payload, next)), native_graph (name, start, max_steps, stages, edges), native_agent (name, "
        "prompt, graph, tools, skills, max_steps, max_tool_calls, then), skill (name, text), rules (text), "
        "config (target models, data with context_window)."
    ),
    "items": {
        "type": "object",
        "properties": {
            "op": {"type": "string", "enum": ["create", "update", "remove"]},
            "id": {"type": "string"},
            "options": {"type": "object"},
        },
        "required": ["op", "id"],
    },
}

INSPECT_DESCRIPTION = (
    "Read the harness you are running in. what=tree lists the live entries (id, kind, config) and the mounted "
    "release id; what=graph shows the main loop graph and every named graph as JSON; what=verdicts lists the "
    "newest releases with the gate metrics that admitted each one and, when Reef exposes it, the proposals the "
    "gate rejected; what=status shows the mounted release, the follow mode and any pending mount. Read only. "
    "Call it before you change anything."
)
PROPOSE_DESCRIPTION = (
    "Send a change to your harness to Reef for the gate. Reef admits or refuses the mutations at once and "
    "answers with a proposal id; an admitted proposal is measured on real tasks against the current harness "
    "before it is served, and nothing changes in this session. Give the reason in one or two sentences. Try the "
    "change with harness_try first."
)
TRY_DESCRIPTION = (
    "Mount a change on this process for the rest of the current turn only, with the same mutations shape as "
    "harness_propose. New tools, hooks, rules, skills, graphs and agents apply from your next step; at the end "
    "of the turn the served harness is mounted back. Nothing is published. Use it to check a change works "
    "before harness_propose."
)


class ServeState(Protocol):
    """What the self tools read and drive: the serve process."""

    client: ReleaseClient

    @property
    def release_id(self) -> str | None: ...

    @property
    def session_id(self) -> str | None: ...

    def live_entries(self) -> list[dict[str, Any]]: ...

    def status(self) -> dict[str, Any]: ...

    def try_mount(self, mutations: Sequence[Mutation], try_id: str) -> dict[str, Any]: ...


class HostTool(ToolModule):
    """A tool of the host plane: built in code, run in process whatever the enforcer, its name reserved."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["host_plane"] = True
        super().__init__(*args, **kwargs)


def _mutations(raw: Any) -> list[Mutation]:
    if not isinstance(raw, list) or not raw:
        raise ToolFailed("mutations must be a non-empty list of {op, id, options} objects")
    mutations = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ToolFailed(f"mutations[{index}] must be an object with op, id and options")
        options = item.get("options")
        if options is not None and not isinstance(options, Mapping):
            raise ToolFailed(f"mutations[{index}].options must be an object")
        try:
            mutations.append(Mutation(str(item.get("op", "")), str(item.get("id", "")), options))
        except MutationError as exc:
            raise ToolFailed(f"mutations[{index}]: {exc}") from exc
    return mutations


def _records(mutations: Sequence[Mutation]) -> list[dict[str, Any]]:
    return [{"op": m.op, "id": m.id, "options": None if m.options is None else dict(m.options)} for m in mutations]


class SelfTools:
    """The three tools' ``run`` bodies over one serve process."""

    def __init__(self, state: ServeState) -> None:
        self._state = state

    def inspect(self, args: dict[str, Any], workdir: str) -> Any:
        what = str(args.get("what", ""))
        entries = self._state.live_entries()
        if what == "tree":
            return {
                "release_id": self._state.release_id,
                "entries": [{"id": e.get("id"), "kind": e.get("name"), "config": e.get("config")} for e in entries],
            }
        if what == "graph":
            graphs = {
                str(e["config"]["name"]): e["config"]
                for e in entries
                if e.get("name") == "native_graph" and isinstance(e.get("config"), Mapping)
            }
            return {"main": graphs.get("main", SEED_GRAPH), "graphs": graphs}
        if what == "verdicts":
            return self._verdicts()
        if what == "status":
            return self._state.status()
        raise ToolFailed(f"what must be one of tree, graph, verdicts, status; got {what!r}")

    def _verdicts(self) -> dict[str, Any]:
        rows = list(reversed(self._state.client.releases()))[:VERDICT_ROWS]
        releases = [
            {
                "release_id": row.get("release_id"),
                "parent_release_id": row.get("parent_release_id"),
                "operation": row.get("operation"),
                "pending": row.get("pending", False),
                "recorded_at": row.get("recorded_at"),
                "gate": row.get("metrics", row.get("gate")),
            }
            for row in rows
        ]
        head = self._state.release_id or (str(rows[0]["release_id"]) if rows and rows[0].get("release_id") else None)
        rejected = None
        if head is not None:
            gate = self._state.client.fetch(head).get("gate")
            rejected = gate.get("rejected") if isinstance(gate, Mapping) else None
        return {"head": head, "releases": releases, "rejected": rejected}

    def propose(self, args: dict[str, Any], workdir: str) -> Any:
        mutations = _mutations(args.get("mutations"))
        body = {
            "mutations": _records(mutations),
            "reason": str(args.get("reason") or ""),
            "session": self._state.session_id,
            "release_id": self._state.release_id,
        }
        status, payload = self._state.client.propose(body)
        if status == 404:
            raise ToolFailed("this reef has no proposals route")
        return json.dumps(payload, ensure_ascii=False)

    def try_(self, args: dict[str, Any], workdir: str) -> Any:
        return self._state.try_mount(_mutations(args.get("mutations")), secrets.token_hex(4))


def self_tools(state: ServeState) -> list[HostTool]:
    """The three host plane tools over ``state``, in the order the model should read them."""
    tools = SelfTools(state)
    return [
        HostTool(
            "harness_inspect",
            INSPECT_DESCRIPTION,
            {
                "type": "object",
                "properties": {
                    "what": {
                        "type": "string",
                        "enum": ["tree", "graph", "verdicts", "status"],
                        "description": "Which view: tree, graph, verdicts or status.",
                    }
                },
                "required": ["what"],
            },
            tools.inspect,
            capabilities=("network",),
        ),
        HostTool(
            "harness_try",
            TRY_DESCRIPTION,
            {"type": "object", "properties": {"mutations": _MUTATIONS_SCHEMA}, "required": ["mutations"]},
            tools.try_,
            capabilities=("exec",),
        ),
        HostTool(
            "harness_propose",
            PROPOSE_DESCRIPTION,
            {
                "type": "object",
                "properties": {
                    "mutations": _MUTATIONS_SCHEMA,
                    "reason": {"type": "string", "description": "Why this change, in one or two sentences."},
                },
                "required": ["mutations", "reason"],
            },
            tools.propose,
            capabilities=("network",),
        ),
    ]


__all__ = ["RESERVED_NAMES", "VERDICT_ROWS", "HostTool", "SelfTools", "ServeState", "self_tools"]
