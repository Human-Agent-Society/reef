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
- ``native_hook``: a named listener at one seam of the native loop, code
  defining ``listen(payload, next)`` (``native/hooks/``).

The plugins hold no services and register no effects: the Entry tree itself
is the state, and ``reef.harness.render`` reads it back out per adapter.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from typing import Any

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
#: The native loop's seams, the only places a native_hook node may listen.
NATIVE_SEAMS = ("pre_step", "request_error", "post_execute")
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
    _require_python(options, "native_tool node")


def native_hook_node(ctx: Any, config: Any) -> None:
    """A named listener the native loop calls at one seam: code defining ``listen(payload, next)``."""
    options = _require_mapping(config)
    _require_name(options)
    if options.get("seam") not in NATIVE_SEAMS:
        raise ValueError(f"native_hook node 'seam' must be one of {', '.join(NATIVE_SEAMS)}")
    _require_python(options, "native_hook node")


NODE_KINDS: dict[str, Callable[[Any, Any], None]] = {
    "config": config_node,
    "rules": rules_node,
    "agent_command": agent_command_node,
    "skill": skill_node,
    "code_extension": code_extension_node,
    "native_tool": native_tool_node,
    "native_hook": native_hook_node,
}
"""Entry ``name`` to node plugin; the resolver of the composition loader."""
