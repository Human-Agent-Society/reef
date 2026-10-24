"""Content checks for the files of a harness tree, shared by tree validation and the native loop's loader.

Each check reads one file's content and raises ValueError naming what is wrong.
``validate_tree`` runs them for every file of a tree; the native loop's loader
runs the native ones again on the files it boots from, so a hand-edited file
cannot run unchecked. The free-text screens (``secret_shaped``,
``directive_shaped``) also guard the step records and saved tasks.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from reef.harness.tree.layout import NAME_PATTERN

NAME = re.compile(rf"^{NAME_PATTERN}$")
#: A loop's model step budget when its file names none; the seed graph's.
NATIVE_LOOP_DEFAULT_MAX_STEPS = 12
#: The native loop's events, the only places a native_hook may listen.
NATIVE_EVENTS = ("pre_step", "pre_execute", "request_error", "post_execute")
#: What a native_tool may declare it does; the loop reports them and a pre_execute hook reads them.
NATIVE_CAPABILITIES = ("read", "write", "exec", "network")
#: Names reserved for built-in tools (``reef.harness.runners.native.selftools``); no tree file may take one.
NATIVE_RESERVED_TOOL_NAMES = ("harness_inspect", "harness_propose", "harness_try")
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
#: What one native_agent may carry beside its name.
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
#: Admission bounds pattern length; the interpreter separately bounds search text and execution time.
NATIVE_PATTERN_MAX_LENGTH = 200
SECRET_NAME = re.compile(r"(?i)(api[_-]?keys?([_-]?env)?|tokens?|secrets?|passwords?)$")
#: Distinctive credential shapes in free text, checked like SECRET_NAME:
#: prefixes and key blocks that are never legitimate tree content, chosen so
#: prose about keys (or the tutorial's sk-local placeholder) does not match.
SECRET_TEXT = re.compile(
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
#: control tokens. Like SECRET_TEXT, this check matches the directive with
#: its object, never the topic, so a task about prompts or rules passes.
DIRECTIVE_TEXT = re.compile(
    r"(?i)(?:\b(?:ignore|disregard|forget)\s+(?:(?:all|any|the|your|of|every)\s+)*"
    r"(?:previous|prior|above|earlier|preceding)\s+(?:instructions?|messages?|rules?|prompts?|guidelines?|directions?)\b"
    r"|\b(?:new|updated|real|actual|true|hidden|secret)\s+(?:system|developer)\s+(?:prompt|message|instructions?)\s*:"
    r"|<\|(?:im_start|im_end|system|start_header_id|end_header_id)\|>"
    r"|\[INST\]|<<SYS>>)"
)


def holds_literal(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple)):
        return any(holds_literal(item) for item in value)
    return False


def require_mapping(config: Any) -> Mapping[str, Any]:
    if not isinstance(config, Mapping):
        raise ValueError(f"the value must be a JSON object, got {type(config).__name__}")
    return config


def require_string_field(config: Mapping[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"requires a non-empty string {key!r}")
    # A lone surrogate survives JSON but cannot be written as UTF-8 at render.
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{key!r} must be UTF-8 encodable text: {exc}") from exc
    return value


def require_python_field(options: Mapping[str, Any], where: str) -> str:
    """A ``code`` body the native loop will import: refused here when it cannot even compile."""
    code = require_string_field(options, "code")
    try:
        # The bytes the render writes, as the loop compiles them: a coding cookie reads the same here as at boot.
        compile(code.encode("utf-8"), f"{options.get('name')}.py", "exec")
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"{where} 'code' does not compile: {exc}") from exc
    reject_secret_shaped_text(code, f"{where} 'code'")
    return code


def require_name_field(config: Mapping[str, Any]) -> str:
    name = require_string_field(config, "name")
    # Names become path segments in the rendered tree; the pattern keeps a
    # proposal from escaping its layer directory.
    if not NAME.fullmatch(name):
        raise ValueError(f"name {name!r} must match {NAME.pattern}")
    return name


def reject_inline_secret(data: Any, path: str) -> None:
    """Refuse a key-named field holding a literal value anywhere in ``data``.

    A tree persists verbatim in every git release that holds it. No render path needs a
    credential in the tree - the served model binds through
    ``reef.upstream_api_key`` and is injected at episode render, method
    models declare ``evolution.models.<name>.api_key_env`` - so a secret
    field here has no consumer and only a leak to persist. The name match
    (singular, plural, string or string list value) detects common inline
    credentials; all supported channels keep credentials out of the tree.
    The message names the field path, never its value.
    """
    if isinstance(data, Mapping):
        for key, value in data.items():
            where = f"{path}.{key}"
            if isinstance(key, str) and SECRET_NAME.search(key) and holds_literal(value):
                raise ValueError(
                    f"config field {where!r} carries an inline credential; the harness tree "
                    "never holds secrets: set reef.upstream_api_key for the served model or "
                    "evolution.models.<name>.api_key_env for method models"
                )
            reject_inline_secret(value, where)
    elif isinstance(data, (list, tuple)):
        for index, value in enumerate(data):
            reject_inline_secret(value, f"{path}[{index}]")


def secret_shaped(text: str) -> bool:
    """Whether free text carries a credential-shaped literal; shared by tree validation and saved task checks."""
    return SECRET_TEXT.search(text) is not None


def redact_secret_shaped(text: str) -> str:
    """``text`` with every credential-shaped literal replaced; for records that persist model traffic the tree boundary never saw."""
    return SECRET_TEXT.sub("[redacted credential]", text)


#: Characters kept per text in a record; a longer text ends in a clip marker.
RECORD_TEXT_CAP = 20_000


def clip_record_text(text: str) -> str:
    """``text`` with credential-shaped literals redacted, cut at the record cap with a marker naming how much was dropped.

    The record holds model traffic and proposals before the tree boundary
    saw them, so the boundary's credential tripwire runs here too."""
    text = redact_secret_shaped(text)
    if len(text) <= RECORD_TEXT_CAP:
        return text
    return f"{text[:RECORD_TEXT_CAP]}... [clipped {len(text) - RECORD_TEXT_CAP} chars]"


def directive_shaped(text: str) -> bool:
    """Whether a saved task contains instruction-override phrasing or a chat-template control token."""
    return DIRECTIVE_TEXT.search(text) is not None


def reject_secret_shaped_text(text: str, where: str) -> None:
    """Refuse a file text carrying a credential-shaped literal.

    The same boundary as :func:`reject_inline_secret`, for the free-text
    kinds: a tree persists verbatim, so a pasted key in a rule, skill,
    command, or extension would outlive rotation in every git release that
    holds it. The message names the field, never the value.
    """
    if secret_shaped(text):
        raise ValueError(
            f"{where} carries an inline credential; the harness tree never holds secrets: "
            "set reef.upstream_api_key for the served model or "
            "evolution.models.<name>.api_key_env for method models"
        )


#: A write to the harness process's own stdout or stderr from inside an extension.
CONSOLE_WRITE = re.compile(
    r"\bconsole\.(?:log|error|warn|info|debug|trace|dir|table|group|groupEnd|time|timeEnd|count|assert)\s*\("
    r"|\bprocess\.(?:stdout|stderr)\.write\s*\("
)


def reject_unguarded_console_write(code: str, where: str) -> None:
    """Refuse an extension that writes to the harness's stdout or stderr without a ``ctx.hasUI`` guard.

    An extension runs inside the harness process, and in a session with a UI
    that process owns the screen: a raw write lands in the middle of a drawn
    frame and leaves the terminal without its input box until the next full
    redraw. Console output is the fallback for a session without a UI
    (``-p``, ``--mode json``), so the guard is what separates the two, and it
    is read textually: the write's own line or the line above it must name
    ``hasUI``. In a session with a UI the text belongs in
    ``ctx.ui.notify``, ``ctx.ui.setStatus`` or ``ctx.ui.setWidget``.
    """
    lines = code.splitlines()
    previous = ""
    for number, line in enumerate(lines, start=1):
        if CONSOLE_WRITE.search(line) and "hasUI" not in line and "hasUI" not in previous:
            raise ValueError(
                f"{where} writes to the harness's stdout or stderr at line {number} without a ctx.hasUI guard, "
                "which overwrites the drawn terminal in a session with a UI; show the text with ctx.ui.notify, "
                "ctx.ui.setStatus or ctx.ui.setWidget, and keep console output for the no-UI path "
                "(if (!ctx.hasUI) console.error(...))"
            )
        if line.strip():
            previous = line


def graph_stage(name: str, stage: Any) -> str:
    """Validate one stage's kind and keys; the kind, so the caller can read its outcomes."""
    if not NAME.fullmatch(name):
        raise ValueError(f"native_graph stage name {name!r} must match {NAME.pattern}")
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
            admit_pattern(stage["pattern"], f"native_graph stage {name!r} 'pattern'")
        if "message" in stage:
            reject_secret_shaped_text(require_string_field(stage, "message"), f"native_graph stage {name!r} 'message'")
    elif kind == "message":
        reject_secret_shaped_text(require_string_field(stage, "text"), f"native_graph stage {name!r} 'text'")
    elif kind == "branch":
        branch_cases(name, stage.get("cases"))
    elif kind == "compact":
        compact_ratios(name, stage)
    elif kind == "subagent":
        agent = stage.get("agent")
        if not isinstance(agent, str) or not NAME.fullmatch(agent):
            raise ValueError(f"native_graph stage {name!r} 'agent' must name an agent")
    elif kind == "end" and stage.get("reason", "completed") not in NATIVE_END_REASONS:
        raise ValueError(f"native_graph stage {name!r} 'reason' must be one of {', '.join(NATIVE_END_REASONS)}")
    return kind


def admit_pattern(value: Any, where: str) -> None:
    """A proposed regular expression the loop may run: it compiles and it is short; its running time is bounded
    where it runs, not here."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where} must be a regular expression")
    if len(value) > NATIVE_PATTERN_MAX_LENGTH:
        raise ValueError(f"{where} must be at most {NATIVE_PATTERN_MAX_LENGTH} characters")
    try:
        re.compile(value)
    except re.error as exc:
        raise ValueError(f"{where} must be a regular expression: {exc}") from exc


def branch_cases(name: str, cases: Any) -> None:
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
            admit_pattern(value, f"native_graph stage {name!r} case 'value'")
        elif isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= NATIVE_GRAPH_MAX_STEPS:
            raise ValueError(
                f"native_graph stage {name!r} case 'value' must be an integer from 0 to {NATIVE_GRAPH_MAX_STEPS}"
            )
        if not isinstance(outcome, str) or not NAME.fullmatch(outcome) or outcome == "else":
            raise ValueError(f"native_graph stage {name!r} case 'outcome' must be a name other than else")
        if outcome in outcomes:
            raise ValueError(f"native_graph stage {name!r} names outcome {outcome!r} twice")
        outcomes.add(outcome)


def compact_ratios(name: str, stage: Mapping[str, Any]) -> None:
    """A compact's two ratios of the context window: it fires above one and keeps the other as the tail."""
    ratios: dict[str, float] = {}
    for key in ("fire_ratio", "keep_ratio"):
        value = stage.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 1:
            raise ValueError(f"native_graph stage {name!r} '{key}' must be a number above 0 and at most 1")
        ratios[key] = float(value)
    if ratios["keep_ratio"] >= ratios["fire_ratio"]:
        raise ValueError(f"native_graph stage {name!r} 'keep_ratio' must be below 'fire_ratio'")


def stage_outcomes(kind: str, stage: Mapping[str, Any]) -> tuple[str, ...]:
    """The outcomes a stage's edges must cover: fixed per kind, plus one per case for a branch."""
    if kind == "branch":
        return (*(str(case["outcome"]) for case in stage["cases"]), "else")
    return NATIVE_STAGES[kind][1]


def reachable(start: str, edges: Mapping[str, set[str]]) -> set[str]:
    seen, queue = {start}, [start]
    while queue:
        for target in edges.get(queue.pop(), set()):
            if target not in seen:
                seen.add(target)
                queue.append(target)
    return seen


def has_cycle(nodes: set[str], edges: Mapping[str, set[str]]) -> bool:
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
    """The rules a native_graph file's object must pass, shared by tree validation and the loop's loader.

    A graph passes when every stage is a known kind with only its own keys,
    every outcome of every stage (a branch's cases and its ``else``) has
    exactly one edge, every stage is reachable from ``start``, some end stage
    is reachable from every stage, and every cycle passes through a model
    stage, so the finite step budget ends every run."""
    options = require_mapping(config)
    require_name_field(options)
    extra = sorted(set(options) - {"name", "start", "max_steps", "stages", "edges"})
    if extra:
        raise ValueError(f"native_graph does not take {', '.join(extra)}")
    max_steps = options.get("max_steps", NATIVE_GRAPH_MAX_STEPS)
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or not 1 <= max_steps <= NATIVE_GRAPH_MAX_STEPS:
        raise ValueError(f"native_graph 'max_steps' must be an integer from 1 to {NATIVE_GRAPH_MAX_STEPS}")
    stages = options.get("stages")
    if not isinstance(stages, Mapping) or not 1 <= len(stages) <= NATIVE_GRAPH_MAX_STAGES:
        raise ValueError(f"native_graph 'stages' must be an object of 1 to {NATIVE_GRAPH_MAX_STAGES} stages")
    kinds = {str(name): graph_stage(str(name), stage) for name, stage in stages.items()}
    outcomes = {name: stage_outcomes(kind, stages[name]) for name, kind in kinds.items()}
    start = options.get("start")
    if start not in kinds:
        raise ValueError("native_graph 'start' must name a stage")
    edges = options.get("edges")
    if not isinstance(edges, Sequence) or isinstance(edges, str) or len(edges) > NATIVE_GRAPH_MAX_EDGES:
        raise ValueError(f"native_graph 'edges' must be a list of at most {NATIVE_GRAPH_MAX_EDGES} edges")
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
    unreachable = sorted(set(kinds) - reachable(start, targets))
    if unreachable:
        raise ValueError(f"native_graph stages {', '.join(unreachable)} are not reachable from {start!r}")
    ends = {name for name, kind in kinds.items() if kind == "end"}
    if not ends:
        raise ValueError("native_graph needs an end stage")
    reverse: dict[str, set[str]] = {name: set() for name in kinds}
    for source, names in targets.items():
        for target in names:
            reverse[target].add(source)
    reaches_end: set[str] = set()
    for end in ends:
        reaches_end |= reachable(end, reverse)
    dead = sorted(set(kinds) - reaches_end)
    if dead:
        raise ValueError(f"native_graph stages {', '.join(dead)} cannot reach an end stage")
    unbudgeted = {name for name, kind in kinds.items() if kind not in ("model", "end")}
    if has_cycle(unbudgeted, targets):
        raise ValueError("native_graph has a cycle without a model stage, so nothing would end it")
    return options


def distinct_names(options: Mapping[str, Any], key: str, where: str, limit: int | None = None) -> None:
    value = options.get(key, [])
    if not isinstance(value, Sequence) or isinstance(value, str) or len(set(value)) < len(value):
        raise ValueError(f"{where} '{key}' must be a list of distinct names")
    if any(not isinstance(item, str) or not NAME.fullmatch(item) for item in value):
        raise ValueError(f"{where} '{key}' must be a list of distinct names")
    if limit is not None and len(value) > limit:
        raise ValueError(f"{where} '{key}' takes at most {limit} names")


def validate_native_agent(config: Any) -> Mapping[str, Any]:
    """The rules a native_agent file's object must pass, shared by tree validation and the loop's loader.

    An agent is one file: its own prompt, the graph it runs, the tools and
    skills it alone sees (all of the tree's when unset), its step and tool
    call budget, and ``then``, the agents its final text is handed to in
    order. Whether the names it uses exist is checked by ``validate_tree``,
    where the whole tree is in view."""
    options = require_mapping(config)
    name = require_name_field(options)
    extra = sorted(set(options) - {"name", *NATIVE_AGENT_KEYS})
    if extra:
        raise ValueError(f"native_agent does not take {', '.join(extra)}")
    reject_secret_shaped_text(require_string_field(options, "prompt"), "native_agent 'prompt'")
    # The default is the built in seed loop, never the root's main graph: a main graph that calls this agent
    # would otherwise call it again from inside its turn.
    graph = options.get("graph", "seed")
    if not isinstance(graph, str) or not NAME.fullmatch(graph):
        raise ValueError("native_agent 'graph' must name a graph")
    distinct_names(options, "tools", "native_agent")
    distinct_names(options, "skills", "native_agent")
    distinct_names(options, "then", "native_agent", NATIVE_AGENT_MAX_THEN)
    if name in options.get("then", []):
        raise ValueError(f"native_agent {name!r} cannot hand its text to itself")
    for key, cap in (("max_steps", NATIVE_GRAPH_MAX_STEPS), ("max_tool_calls", NATIVE_AGENT_MAX_TOOL_CALLS)):
        value = options.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= cap):
            raise ValueError(f"native_agent '{key}' must be an integer from 1 to {cap}")
    return options


def unread_field(node: ast.AST, field: str) -> bool:
    """Whether ``field`` of ``node`` binds nothing in the scope that holds ``node``.

    A def, class or lambda body is a scope of its own (its decorators, defaults and bases run in the holding
    scope and are read); a comprehension's target is the comprehension's; a bare annotation binds nothing."""
    if field == "body":
        return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda))
    if field == "target":
        return isinstance(node, ast.comprehension) or (isinstance(node, ast.AnnAssign) and node.value is None)
    return False


def scope_bindings(node: ast.AST) -> Iterator[tuple[str, ast.AST]]:
    """Every name ``node`` binds in the scope that holds it, in source order, each with the node that binds it.

    A binding is a def or class name, an assignment, augmented assignment, annotated assignment, ``del``,
    ``for`` or ``with`` target (through tuple and star unpacking), an import alias, a walrus target (inside a
    comprehension too), an ``except`` handler name, a match capture or a type alias. The bodies of ``if``,
    ``for``, ``while``, ``with``, ``try`` and ``match`` are followed; a function, class or lambda body is not."""
    for field, value in ast.iter_fields(node):
        if unread_field(node, field):
            continue
        for child in value if isinstance(value, list) else [value]:
            if isinstance(child, ast.AST):
                yield from scope_bindings(child)
    if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
        yield node.id, node
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        yield node.name, node
    elif isinstance(node, ast.alias):
        yield node.asname or node.name.split(".")[0], node
    elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and node.name is not None:
        yield node.name, node
    elif isinstance(node, ast.MatchMapping) and node.rest is not None:
        yield node.rest, node


def defines_run_turn(code: str) -> bool:
    """Whether ``run_turn``, as the module body leaves it, is a plain ``def`` taking the context, read off the syntax tree."""
    # The bytes require_python_field compiled: a coding cookie or a BOM parses here as it compiled there.
    tree = ast.parse(code.encode("utf-8"))
    last: ast.AST | None = None
    for name, node in scope_bindings(tree):
        if name == "run_turn":
            last = node
    # A plain def in the module body only: an async run_turn would hand the loop a coroutine and never run, and a
    # def inside an if, for, with, try or match is a binding a static read cannot follow.
    if not isinstance(last, ast.FunctionDef) or not any(node is last for node in tree.body):
        return False
    return bool(last.args.posonlyargs or last.args.args or last.args.vararg)


def validate_native_loop(config: Any) -> Mapping[str, Any]:
    """The rules a native_loop module must pass, shared by tree validation and the loop's loader.

    The loop is code, so admission checks what a static read can: the module
    compiles, carries no credential, and defines a top level ``run_turn``
    taking the context. Whether it runs is the evaluation's to find out, and a
    person promotes it: the kind is in :data:`ALWAYS_REVIEWED_KINDS`."""
    options = require_mapping(config)
    require_name_field(options)
    extra = sorted(set(options) - {"name", "code", "max_steps"})
    if extra:
        raise ValueError(f"native_loop does not take {', '.join(extra)}")
    max_steps = options.get("max_steps", NATIVE_LOOP_DEFAULT_MAX_STEPS)
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or not 1 <= max_steps <= NATIVE_GRAPH_MAX_STEPS:
        raise ValueError(f"native_loop 'max_steps' must be an integer from 1 to {NATIVE_GRAPH_MAX_STEPS}")
    if not defines_run_turn(require_python_field(options, "native_loop")):
        raise ValueError("native_loop 'code' must define a top level run_turn(ctx)")
    return options
