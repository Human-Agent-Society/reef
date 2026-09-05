"""Harness composition node kinds: the plugin vocabulary of the Entry tree.

A harness composition is a flat compose Entry tree whose entries name one of
the node kinds below. Each kind's plugin body is its admission gate: it
validates the entry config at load, so a proposal carrying an invalid node
lands as a FAILED fiber and never reaches the ledger. Five kinds cover what
both surveyed harnesses compose from files alone - one JSON config tree,
markdown resource directories, and code-valued extension files - and the
native harness adds kinds only it renders:

- ``config``: a JSON object merged into one of the adapter's declared config
  targets (``settings.json``/``models.json`` for pi, ``opencode.json``).
- ``rules``: context text concatenated into the adapter's rules file
  (``AGENTS.md`` on both harnesses).
- ``agent_command``: a named prompt template (pi ``prompts/``, opencode
  ``command/``).
- ``skill``: a named Agent Skill, rendered as ``skills/<name>/SKILL.md``.
- ``code_extension``: a named code file the harness loads in-process (pi
  ``extensions/``, opencode ``plugin/``).
- ``native_tool``: a named tool of the native harness, a JSON schema plus
  code defining ``run(args, workdir)`` (``native/tools/``).
- ``native_hook``: a named listener at one event of the native loop, code
  defining ``listen(payload, next)`` (``native/hooks/``).
- ``native_graph``: the native loop's control flow as data, stages from a
  closed vocabulary joined by edges keyed by outcome (``native/graphs/``).
- ``native_agent``: one agent of the native loop as data, its own prompt,
  graph, tools, skills, budget, and the agents its text is handed to
  (``native/agents/``).

The plugins hold no services and register no effects: the Entry tree itself
is the state, and ``reef.harness.render`` reads it back out per adapter.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
#: The native loop's events, the only places a native_hook node may listen.
NATIVE_EVENTS = ("pre_step", "pre_execute", "request_error", "post_execute")
#: What a native_tool may declare it does; the loop reports them and a pre_execute hook reads them.
NATIVE_CAPABILITIES = ("read", "write", "exec", "network")
#: The native graph's stage vocabulary: the keys a stage may carry and the outcomes its edges may name.
NATIVE_STAGES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "model": ((), ("tool_calls", "text")),
    "tools": (("allow",), ("done",)),
    "verify": (("check", "pattern", "message"), ("pass", "fail")),
    "message": (("text",), ("done",)),
    "branch": (("cases",), ("else",)),
    "compact": (("fire_ratio", "keep_ratio"), ("done",)),
    "subagent": (("agent",), ("completed", "gave_up", "budget", "ask")),
    "end": (("reason",), ()),
}
#: What one native_agent node may carry beside its name.
NATIVE_AGENT_KEYS = ("prompt", "graph", "tools", "skills", "max_steps", "max_tool_calls", "then")
NATIVE_AGENT_MAX_TOOL_CALLS = 256
NATIVE_AGENT_MAX_THEN = 8
NATIVE_VERIFY_CHECKS = ("last_line_integer", "last_line_matches", "nonempty")
#: What a branch case may test: the run's own counters, or the last assistant text against a pattern.
NATIVE_BRANCH_PREDICATES = ("steps_used_at_least", "tool_errors_at_least", "last_text_matches")
NATIVE_END_REASONS = ("completed", "gave_up")
#: Size caps on one graph, so admission and the interpreter's guard stay cheap.
NATIVE_GRAPH_MAX_STEPS = 32
NATIVE_GRAPH_MAX_STAGES = 16
NATIVE_GRAPH_MAX_EDGES = 64
NATIVE_GRAPH_MAX_CASES = 8
_SECRET_NAME = re.compile(r"(?i)(api[_-]?keys?([_-]?env)?|tokens?|secrets?|passwords?)$")
#: Distinctive credential shapes in free text. A tripwire like _SECRET_NAME:
#: prefixes and key blocks that are never legitimate tree content, chosen so
#: prose about keys (or the tutorial's sk-local placeholder) cannot trip it.
_SECRET_TEXT = re.compile(
    r"(?:sk-[A-Za-z0-9_-]{16,}"
    r"|ghp_[A-Za-z0-9]{20,}"
    r"|github_pat_[A-Za-z0-9_]{20,}"
    r"|gho_[A-Za-z0-9]{20,}"
    r"|xox[baprs]-[A-Za-z0-9-]{10,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----)"
)
#: Instruction-override shapes in recorded traffic: the phrasings that tell a
#: reader to drop its instructions, a forged system message, and chat-template
#: control tokens. A tripwire like _SECRET_TEXT: it matches the directive with
#: its object, never the topic, so a task about prompts or rules passes.
_DIRECTIVE_TEXT = re.compile(
    r"(?i)(?:\b(?:ignore|disregard|forget)\s+(?:(?:all|any|the|your|of|every)\s+)*"
    r"(?:previous|prior|above|earlier|preceding)\s+(?:instructions?|messages?|rules?|prompts?|guidelines?|directions?)\b"
    r"|\b(?:new|updated|real|actual|true|hidden|secret)\s+(?:system|developer)\s+(?:prompt|message|instructions?)\s*:"
    r"|<\|(?:im_start|im_end|system|start_header_id|end_header_id)\|>"
    r"|\[INST\]|<<SYS>>)"
)


def _holds_literal(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple)):
        return any(_holds_literal(item) for item in value)
    return False


def _require_mapping(config: Any) -> Mapping[str, Any]:
    if not isinstance(config, Mapping):
        raise ValueError(f"node config must be an object, got {type(config).__name__}")
    return config


def _require_text(config: Mapping[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"node config requires a non-empty string {key!r}")
    # A lone surrogate survives JSON but cannot be written as UTF-8 at render.
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"node config {key!r} must be UTF-8 encodable text: {exc}") from exc
    return value


def _require_python(options: Mapping[str, Any], where: str) -> str:
    """A ``code`` body the native loop will import: refused here when it cannot even compile."""
    code = _require_text(options, "code")
    try:
        compile(code, f"{options.get('name')}.py", "exec")
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"{where} 'code' does not compile: {exc}") from exc
    _reject_secret_shaped_text(code, f"{where} 'code'")
    return code


def _require_name(config: Mapping[str, Any]) -> str:
    name = _require_text(config, "name")
    # Names become path segments in the rendered tree; the pattern keeps a
    # proposal from escaping its layer directory.
    if not _NAME.fullmatch(name):
        raise ValueError(f"node name {name!r} must match {_NAME.pattern}")
    return name


def _reject_inline_secret(data: Any, path: str) -> None:
    """Refuse a key-named field holding a literal value anywhere in ``data``.

    Tree state persists verbatim: every commit record, the snapshot
    metadata, and the published artifact carry it. No render path needs a
    credential in the tree - the served model binds through
    ``reef.upstream_api_key`` and is injected at episode render, method
    models declare ``evolution.models.<name>.api_key_env`` - so a secret
    field here has no consumer and only a leak to persist. The name match
    (singular, plural, string or string list value) is a tripwire, not the
    boundary; the boundary is that no sanctioned channel puts a credential
    in the tree. The message names the field path, never its value.
    """
    if isinstance(data, Mapping):
        for key, value in data.items():
            where = f"{path}.{key}"
            if isinstance(key, str) and _SECRET_NAME.search(key) and _holds_literal(value):
                raise ValueError(
                    f"config node field {where!r} carries an inline credential; the composition tree "
                    "never holds secrets: set reef.upstream_api_key for the served model or "
                    "evolution.models.<name>.api_key_env for method models"
                )
            _reject_inline_secret(value, where)
    elif isinstance(data, (list, tuple)):
        for index, value in enumerate(data):
            _reject_inline_secret(value, f"{path}[{index}]")


def config_node(ctx: Any, config: Any) -> None:
    """A JSON object merged into one adapter config target (default ``primary``)."""
    options = _require_mapping(config)
    target = options.get("target", "primary")
    if not isinstance(target, str) or not target:
        raise ValueError("config node 'target' must be a non-empty string")
    data = options.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("config node requires an object 'data'")
    try:
        json.dumps(data)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"config node 'data' must be JSON-serializable: {exc}") from exc
    _reject_inline_secret(data, "data")


def secret_shaped(text: str) -> bool:
    """Whether free text carries a credential-shaped literal; shared by the tree boundary and the task ledger."""
    return _SECRET_TEXT.search(text) is not None


def directive_shaped(text: str) -> bool:
    """Whether free text carries an instruction-override phrasing or a chat-template control token; the task ledger's second tripwire."""
    return _DIRECTIVE_TEXT.search(text) is not None


def _reject_secret_shaped_text(text: str, where: str) -> None:
    """Refuse a node body carrying a credential-shaped literal.

    The same boundary as :func:`_reject_inline_secret`, for the free-text
    kinds: tree state persists verbatim, so a pasted key in a rule, skill,
    command, or extension would outlive rotation in every commit record and
    published artifact. The message names the field, never the value.
    """
    if secret_shaped(text):
        raise ValueError(
            f"{where} carries an inline credential; the composition tree never holds secrets: "
            "set reef.upstream_api_key for the served model or "
            "evolution.models.<name>.api_key_env for method models"
        )


def rules_node(ctx: Any, config: Any) -> None:
    """Context text for the adapter's rules file (AGENTS.md on both harnesses)."""
    _reject_secret_shaped_text(_require_text(_require_mapping(config), "text"), "rules node 'text'")


def agent_command_node(ctx: Any, config: Any) -> None:
    """A named prompt template, rendered as one markdown command file."""
    options = _require_mapping(config)
    _require_name(options)
    _reject_secret_shaped_text(_require_text(options, "text"), "agent_command node 'text'")


def skill_node(ctx: Any, config: Any) -> None:
    """A named Agent Skill, rendered as ``skills/<name>/SKILL.md``."""
    options = _require_mapping(config)
    _require_name(options)
    _reject_secret_shaped_text(_require_text(options, "text"), "skill node 'text'")


def code_extension_node(ctx: Any, config: Any) -> None:
    """A named code file the harness loads in-process (extension or plugin)."""
    options = _require_mapping(config)
    _require_name(options)
    _reject_secret_shaped_text(_require_text(options, "code"), "code_extension node 'code'")


def native_tool_node(ctx: Any, config: Any) -> None:
    """A named tool the native harness loads: a description, a JSON schema, and code defining ``run(args, workdir)``."""
    options = _require_mapping(config)
    _require_name(options)
    _require_text(options, "description")
    if not isinstance(options.get("parameters", {}), Mapping):
        raise ValueError("native_tool node 'parameters' must be an object")
    capabilities = options.get("capabilities", [])
    if (
        not isinstance(capabilities, Sequence)
        or isinstance(capabilities, str)
        or len(set(capabilities)) < len(capabilities)
    ):
        raise ValueError(
            f"native_tool node 'capabilities' must be a list of distinct names from {', '.join(NATIVE_CAPABILITIES)}"
        )
    if any(item not in NATIVE_CAPABILITIES for item in capabilities):
        raise ValueError(
            f"native_tool node 'capabilities' must be a list of distinct names from {', '.join(NATIVE_CAPABILITIES)}"
        )
    _require_python(options, "native_tool node")


def native_hook_node(ctx: Any, config: Any) -> None:
    """A named listener the native loop calls at one event: code defining ``listen(payload, next)``."""
    options = _require_mapping(config)
    _require_name(options)
    if options.get("event") not in NATIVE_EVENTS:
        raise ValueError(f"native_hook node 'event' must be one of {', '.join(NATIVE_EVENTS)}")
    _require_python(options, "native_hook node")


def _graph_stage(name: str, stage: Any) -> str:
    """Validate one stage's kind and keys; the kind, so the caller can read its outcomes."""
    if not _NAME.fullmatch(name):
        raise ValueError(f"native_graph stage name {name!r} must match {_NAME.pattern}")
    if not isinstance(stage, Mapping):
        raise ValueError(f"native_graph stage {name!r} must be an object")
    kind = stage.get("kind")
    if kind not in NATIVE_STAGES:
        raise ValueError(f"native_graph stage {name!r} kind must be one of {', '.join(NATIVE_STAGES)}")
    keys, _ = NATIVE_STAGES[kind]
    extra = sorted(set(stage) - {"kind", *keys})
    if extra:
        raise ValueError(f"native_graph stage {name!r} ({kind}) does not take {', '.join(extra)}")
    if kind == "tools":
        allow = stage.get("allow", [])
        if not isinstance(allow, Sequence) or isinstance(allow, str) or len(set(allow)) < len(allow):
            raise ValueError(f"native_graph stage {name!r} 'allow' must be a list of distinct tool names")
        if any(not isinstance(item, str) or not item for item in allow):
            raise ValueError(f"native_graph stage {name!r} 'allow' must be a list of distinct tool names")
    elif kind == "verify":
        check = stage.get("check")
        if check not in NATIVE_VERIFY_CHECKS:
            raise ValueError(f"native_graph stage {name!r} 'check' must be one of {', '.join(NATIVE_VERIFY_CHECKS)}")
        if (check == "last_line_matches") != ("pattern" in stage):
            raise ValueError(
                f"native_graph stage {name!r} takes 'pattern' exactly when its check is last_line_matches"
            )
        if "pattern" in stage:
            pattern = stage["pattern"]
            if not isinstance(pattern, str) or not pattern:
                raise ValueError(f"native_graph stage {name!r} 'pattern' must be a regular expression")
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"native_graph stage {name!r} 'pattern' must be a regular expression: {exc}") from exc
        if "message" in stage:
            _reject_secret_shaped_text(_require_text(stage, "message"), f"native_graph stage {name!r} 'message'")
    elif kind == "message":
        _reject_secret_shaped_text(_require_text(stage, "text"), f"native_graph stage {name!r} 'text'")
    elif kind == "branch":
        _branch_cases(name, stage.get("cases"))
    elif kind == "compact":
        _compact_ratios(name, stage)
    elif kind == "subagent":
        agent = stage.get("agent")
        if not isinstance(agent, str) or not _NAME.fullmatch(agent):
            raise ValueError(f"native_graph stage {name!r} 'agent' must name an agent")
    elif kind == "end" and stage.get("reason", "completed") not in NATIVE_END_REASONS:
        raise ValueError(f"native_graph stage {name!r} 'reason' must be one of {', '.join(NATIVE_END_REASONS)}")
    return kind


def _branch_cases(name: str, cases: Any) -> None:
    """A branch's cases: a closed predicate each, a value of that predicate's type, and a distinct outcome."""
    if not isinstance(cases, Sequence) or isinstance(cases, str) or not 1 <= len(cases) <= NATIVE_GRAPH_MAX_CASES:
        raise ValueError(f"native_graph stage {name!r} 'cases' must be a list of 1 to {NATIVE_GRAPH_MAX_CASES} cases")
    outcomes: set[str] = set()
    for case in cases:
        if not isinstance(case, Mapping) or set(case) != {"when", "value", "outcome"}:
            raise ValueError(f"native_graph stage {name!r} cases must be objects with when, value and outcome")
        when, value, outcome = case["when"], case["value"], case["outcome"]
        if when not in NATIVE_BRANCH_PREDICATES:
            raise ValueError(
                f"native_graph stage {name!r} case 'when' must be one of {', '.join(NATIVE_BRANCH_PREDICATES)}"
            )
        if when == "last_text_matches":
            if not isinstance(value, str) or not value:
                raise ValueError(f"native_graph stage {name!r} case 'value' must be a regular expression")
            try:
                re.compile(value)
            except re.error as exc:
                raise ValueError(
                    f"native_graph stage {name!r} case 'value' must be a regular expression: {exc}"
                ) from exc
        elif isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= NATIVE_GRAPH_MAX_STEPS:
            raise ValueError(
                f"native_graph stage {name!r} case 'value' must be an integer from 0 to {NATIVE_GRAPH_MAX_STEPS}"
            )
        if not isinstance(outcome, str) or not _NAME.fullmatch(outcome) or outcome == "else":
            raise ValueError(f"native_graph stage {name!r} case 'outcome' must be a name other than else")
        if outcome in outcomes:
            raise ValueError(f"native_graph stage {name!r} names outcome {outcome!r} twice")
        outcomes.add(outcome)


def _compact_ratios(name: str, stage: Mapping[str, Any]) -> None:
    """A compact's two ratios of the context window: it fires above one and keeps the other as the tail."""
    ratios: dict[str, float] = {}
    for key in ("fire_ratio", "keep_ratio"):
        value = stage.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 1:
            raise ValueError(f"native_graph stage {name!r} '{key}' must be a number above 0 and at most 1")
        ratios[key] = float(value)
    if ratios["keep_ratio"] >= ratios["fire_ratio"]:
        raise ValueError(f"native_graph stage {name!r} 'keep_ratio' must be below 'fire_ratio'")


def _stage_outcomes(kind: str, stage: Mapping[str, Any]) -> tuple[str, ...]:
    """The outcomes a stage's edges must cover: fixed per kind, plus one per case for a branch."""
    if kind == "branch":
        return (*(str(case["outcome"]) for case in stage["cases"]), "else")
    return NATIVE_STAGES[kind][1]


def _reachable(start: str, edges: Mapping[str, set[str]]) -> set[str]:
    seen, queue = {start}, [start]
    while queue:
        for target in edges.get(queue.pop(), set()):
            if target not in seen:
                seen.add(target)
                queue.append(target)
    return seen


def _has_cycle(nodes: set[str], edges: Mapping[str, set[str]]) -> bool:
    """Whether the edges among ``nodes`` close a cycle (depth first, three colors)."""
    state: dict[str, int] = {}

    def visit(name: str) -> bool:
        state[name] = 1
        for target in edges.get(name, ()):
            if target not in nodes:
                continue
            if state.get(target) == 1 or (state.get(target) is None and visit(target)):
                return True
        state[name] = 2
        return False

    return any(state.get(name) is None and visit(name) for name in nodes)


def validate_native_graph(config: Any) -> Mapping[str, Any]:
    """The admission rules of a native_graph, shared by the tree boundary and the loop's loader.

    A graph passes when every stage is a known kind with only its own keys,
    every outcome of every stage (a branch's cases and its ``else``) has
    exactly one edge, every stage is reachable from ``start``, some end stage
    is reachable from every stage, and every cycle passes through a model
    stage, so the finite step budget ends every run."""
    options = _require_mapping(config)
    _require_name(options)
    extra = sorted(set(options) - {"name", "start", "max_steps", "stages", "edges"})
    if extra:
        raise ValueError(f"native_graph node does not take {', '.join(extra)}")
    max_steps = options.get("max_steps", NATIVE_GRAPH_MAX_STEPS)
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or not 1 <= max_steps <= NATIVE_GRAPH_MAX_STEPS:
        raise ValueError(f"native_graph node 'max_steps' must be an integer from 1 to {NATIVE_GRAPH_MAX_STEPS}")
    stages = options.get("stages")
    if not isinstance(stages, Mapping) or not 1 <= len(stages) <= NATIVE_GRAPH_MAX_STAGES:
        raise ValueError(f"native_graph node 'stages' must be an object of 1 to {NATIVE_GRAPH_MAX_STAGES} stages")
    kinds = {str(name): _graph_stage(str(name), stage) for name, stage in stages.items()}
    outcomes = {name: _stage_outcomes(kind, stages[name]) for name, kind in kinds.items()}
    start = options.get("start")
    if start not in kinds:
        raise ValueError("native_graph node 'start' must name a stage")
    edges = options.get("edges")
    if not isinstance(edges, Sequence) or isinstance(edges, str) or len(edges) > NATIVE_GRAPH_MAX_EDGES:
        raise ValueError(f"native_graph node 'edges' must be a list of at most {NATIVE_GRAPH_MAX_EDGES} edges")
    seen: set[tuple[str, str]] = set()
    targets: dict[str, set[str]] = {name: set() for name in kinds}
    for edge in edges:
        if not isinstance(edge, Mapping) or set(edge) != {"from", "when", "to"}:
            raise ValueError("native_graph edges must be objects with from, when and to")
        source, when, target = edge["from"], edge["when"], edge["to"]
        if source not in kinds or target not in kinds:
            raise ValueError(f"native_graph edge {source!r} -> {target!r} must name stages")
        if when not in outcomes[source]:
            raise ValueError(
                f"native_graph edge from {source!r} names outcome {when!r}, not one of its {kinds[source]} stage"
            )
        if (source, when) in seen:
            raise ValueError(f"native_graph stage {source!r} has two edges for outcome {when!r}")
        seen.add((source, when))
        targets[source].add(target)
    for name in kinds:
        for outcome in outcomes[name]:
            if (name, outcome) not in seen:
                raise ValueError(f"native_graph stage {name!r} has no edge for outcome {outcome!r}")
    unreachable = sorted(set(kinds) - _reachable(start, targets))
    if unreachable:
        raise ValueError(f"native_graph stages {', '.join(unreachable)} are not reachable from {start!r}")
    ends = {name for name, kind in kinds.items() if kind == "end"}
    if not ends:
        raise ValueError("native_graph node needs an end stage")
    reverse: dict[str, set[str]] = {name: set() for name in kinds}
    for source, names in targets.items():
        for target in names:
            reverse[target].add(source)
    reaches_end: set[str] = set()
    for end in ends:
        reaches_end |= _reachable(end, reverse)
    dead = sorted(set(kinds) - reaches_end)
    if dead:
        raise ValueError(f"native_graph stages {', '.join(dead)} cannot reach an end stage")
    unbudgeted = {name for name, kind in kinds.items() if kind not in ("model", "end")}
    if _has_cycle(unbudgeted, targets):
        raise ValueError("native_graph has a cycle without a model stage, so nothing would end it")
    return options


def native_graph_node(ctx: Any, config: Any) -> None:
    """The native loop's control flow as data: stages from a closed vocabulary and edges keyed by outcome."""
    validate_native_graph(config)


def _distinct_names(options: Mapping[str, Any], key: str, where: str, limit: int | None = None) -> None:
    value = options.get(key, [])
    if not isinstance(value, Sequence) or isinstance(value, str) or len(set(value)) < len(value):
        raise ValueError(f"{where} '{key}' must be a list of distinct names")
    if any(not isinstance(item, str) or not _NAME.fullmatch(item) for item in value):
        raise ValueError(f"{where} '{key}' must be a list of distinct names")
    if limit is not None and len(value) > limit:
        raise ValueError(f"{where} '{key}' takes at most {limit} names")


def validate_native_agent(config: Any) -> Mapping[str, Any]:
    """The admission rules of a native_agent, shared by the tree boundary and the loop's loader.

    An agent is a root entry: its own prompt, the graph it runs, the tools and
    skills it alone sees (all of the tree's when unset), its step and tool
    call budget, and ``then``, the agents its final text is handed to in
    order. Whether the names it uses exist is checked at render, where the
    whole tree is in view."""
    options = _require_mapping(config)
    name = _require_name(options)
    extra = sorted(set(options) - {"name", *NATIVE_AGENT_KEYS})
    if extra:
        raise ValueError(f"native_agent node does not take {', '.join(extra)}")
    _reject_secret_shaped_text(_require_text(options, "prompt"), "native_agent node 'prompt'")
    # The default is the built in seed loop, never the root's main graph: a main graph that calls this agent
    # would otherwise call it again from inside its turn.
    graph = options.get("graph", "seed")
    if not isinstance(graph, str) or not _NAME.fullmatch(graph):
        raise ValueError("native_agent node 'graph' must name a graph")
    _distinct_names(options, "tools", "native_agent node")
    _distinct_names(options, "skills", "native_agent node")
    _distinct_names(options, "then", "native_agent node", NATIVE_AGENT_MAX_THEN)
    if name in options.get("then", []):
        raise ValueError(f"native_agent node {name!r} cannot hand its text to itself")
    for key, cap in (("max_steps", NATIVE_GRAPH_MAX_STEPS), ("max_tool_calls", NATIVE_AGENT_MAX_TOOL_CALLS)):
        value = options.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= cap):
            raise ValueError(f"native_agent node '{key}' must be an integer from 1 to {cap}")
    return options


def native_agent_node(ctx: Any, config: Any) -> None:
    """One agent of the native loop as data: its prompt, its graph, what it sees, its budget, and ``then``."""
    validate_native_agent(config)


NODE_KINDS: dict[str, Callable[[Any, Any], None]] = {
    "config": config_node,
    "rules": rules_node,
    "agent_command": agent_command_node,
    "skill": skill_node,
    "code_extension": code_extension_node,
    "native_tool": native_tool_node,
    "native_hook": native_hook_node,
    "native_graph": native_graph_node,
    "native_agent": native_agent_node,
}
"""Entry ``name`` to node plugin; the resolver of the composition loader."""
