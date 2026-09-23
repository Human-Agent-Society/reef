"""dsh adapter quirks: the patch layer, the credential file, skill frontmatter, and the boot scaffold.

dsh composes its plugin tree from bundle layers plus one user patch layer,
``profiles/headless/cordis.patch.yml``: a YAML list of entries addressed by
plugin id. Config nodes write that layer as a JSON object keyed by id, so
two nodes touching one plugin deep merge, and ``finalize_render`` emits the
list: one entry per id (a string starting with ``!!js `` becomes a js
expression, the form dsh's own bundles use), then one ``insert`` entry per
rendered code extension so the loader boots it from its relative path. The
``env`` config target becomes ``.env``, the lowest trust layer of dsh's
launch environment, which is how the model binding's key reaches its
``apiKeyEnv`` route. dsh ignores a SKILL.md without YAML frontmatter, so a
skill node whose text has none gets ``name`` and ``description``
synthesized, and an agent_command renders under the second skill root as a
user invocable skill (``/name``), the only command surface dsh has.

The traps a mutated patch could reopen: the session log must stay plain
JSONL (the reader cannot parse zstd), and the session telemetry and the LLM
title call stay disabled. A composition that flips any of them is rejected
at render, the same gate that rejects an invalid node.

Every model call stays on Reef's model binding. The binding writes the
``llm-pi-ai`` route ``reef`` and selects it in ``agent-default-model``, and
puts the route's key in ``.env``; it renders after the tree and wins every
key it writes. The key is a credential, which a tree cannot hold because
admission refuses an inline credential, so the two entries pass only beside
it, and only with the keys the binding writes. Another route, DeepSeek's own
adapter (``llm-deepseek``), the web search model and its endpoint, and a
provider or a model set for the title call, a subagent, a declared agent or
the compaction summary are the tree choosing where calls go, and are
refused, as is a patch entry that names another package or holds a js
expression in those plugins, where the check cannot read the value.
"""

from __future__ import annotations

import json
from typing import Any

import yaml

from reef.harness.tree.render import RenderError

_PATCH = "dsh/profiles/headless/cordis.patch.yml"
_ENV = "dsh/.env"
_EXTENSIONS = "dsh/profiles/headless/extensions/"
_SKILLS = "dsh/skills/"
_COMMANDS = "dsh-agents/skills/"
_JS = "!!js "

#: The key Reef's binding puts in .env for its route, the entries it writes, and the keys of each.
BINDING_KEY_ENV = "REEF_API_KEY"
BINDING_ROUTE = "reef"
BINDING_ROUTE_KEYS = frozenset({"api", "apiKeyEnv", "baseURL", "displayName", "models"})
BINDING_PLUGINS = ("agent-default-model", "llm-pi-ai")
#: Plugins whose config chooses a model call's provider, endpoint, credential or model, each with where that choice
#: sits: a key of the config (no parent), or a key of the object, or of each object in the list, under the parent.
MODEL_ROUTE_KEYS: dict[str, tuple[tuple[str | None, str], ...]] = {
    "agent-loop": (("agents", "model"), ("agents", "provider")),
    "compaction-basic": (("modelPolicies", "summarizationModel"), ("modelPolicies", "summarizationProvider")),
    "session-title-llm": ((None, "model"), (None, "provider")),
    "tool-subagent": (("agentOptions", "model"), ("agentOptions", "provider")),
    "tool-subagent-fork": (("agentOptions", "model"), ("agentOptions", "provider")),
    "web-search-deepseek": ((None, "apiKeyEnv"), (None, "baseURL"), (None, "model")),
}
#: DeepSeek's own adapter, whose config is its endpoint: Reef never binds it.
UNBOUND_ADAPTERS = ("llm-deepseek",)

# dsh's boot scaffolds the profile beside the rendered patch: a package
# manifest, the empty root entry list, the pnpm workspace file, node_modules
# symlinks into the installation, and the module fallback links. Episode
# state, not residue.
cleanup_whitelist = (
    "dsh/profiles/headless/package.json",
    "dsh/profiles/headless/cordis.yml",
    "dsh/profiles/headless/pnpm-workspace.yaml",
    "dsh/profiles/headless/node_modules/**",
    "dsh/profiles/headless/.dsh-module-fallback/**",
    "dsh/profiles/node_modules/**",
)


class _Js(str):
    """A js expression scalar, dumped with the ``!!js`` tag."""


class _Dumper(yaml.SafeDumper):
    pass


_Dumper.add_representer(_Js, lambda dumper, value: dumper.represent_scalar("tag:yaml.org,2002:js", str(value)))


def _tagged(value: Any) -> Any:
    if isinstance(value, str):
        return _Js(value[len(_JS) :]) if value.startswith(_JS) else value
    if isinstance(value, dict):
        return {key: _tagged(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_tagged(item) for item in value]
    return value


def _patch(entries: dict[str, Any], extensions: list[str]) -> str:
    rows: list[dict[str, Any]] = []
    for plugin, entry in sorted(entries.items()):
        if not isinstance(entry, dict):
            raise RenderError(f"dsh patch entry {plugin!r} must be an object holding config, disabled, or inject")
        rows.append({"id": plugin, **_tagged(entry)})
    if extensions:
        rows.append(
            {"insert": [{"id": f"extension-{name}", "name": f"./extensions/{name}.mjs"} for name in extensions]}
        )
    return yaml.dump(rows, Dumper=_Dumper, sort_keys=True, default_flow_style=False, allow_unicode=True)


def _with_frontmatter(path: str, text: str, user_only: bool) -> str:
    if text.startswith("---\n"):
        return text
    first = next((line.strip().lstrip("#").strip() for line in text.splitlines() if line.strip()), "")
    header: dict[str, Any] = {"name": path.split("/")[-2], "description": first[:200] or path.split("/")[-2]}
    if user_only:
        header["disable-model-invocation"] = True
    return "---\n" + yaml.dump(header, sort_keys=False, default_flow_style=False, allow_unicode=True) + "---\n" + text


def holds_js(value: Any) -> bool:
    if isinstance(value, str):
        return value.startswith(_JS)
    if isinstance(value, dict):
        return any(holds_js(item) for item in value.values())
    if isinstance(value, list):
        return any(holds_js(item) for item in value)
    return False


def check_binding_entry(plugin: str, entry: dict[str, Any], bound: bool) -> None:
    """``llm-pi-ai`` holds only the binding's route and ``agent-default-model`` selects it, beside the binding's key."""
    refusal = f"dsh composition must not set {plugin}: Reef's model binding writes it"
    config = entry.get("config")
    if not bound or set(entry) != {"config"} or not isinstance(config, dict):
        raise RenderError(refusal)
    if plugin == "agent-default-model":
        if set(config) != {"model", "provider"} or config["provider"] != BINDING_ROUTE:
            raise RenderError(refusal)
        return
    providers = config.get("providers")
    if set(config) != {"providers"} or not isinstance(providers, dict) or set(providers) != {BINDING_ROUTE}:
        raise RenderError(f"{refusal}; its only route is {BINDING_ROUTE!r}")
    route = providers[BINDING_ROUTE]
    if not isinstance(route, dict) or set(route) != BINDING_ROUTE_KEYS or route["apiKeyEnv"] != BINDING_KEY_ENV:
        raise RenderError(refusal)


def check_model_route(entries: dict[str, Any], env: dict[str, Any]) -> None:
    """Refuse a model route, an endpoint, a credential or a model the binding did not write."""
    credential = env.get(BINDING_KEY_ENV)
    bound = isinstance(credential, str) and bool(credential.strip())
    refusal = "Reef's model binding chooses the provider, the endpoint, the credential and the model"
    for plugin, entry in entries.items():
        if not isinstance(entry, dict):
            continue
        if "name" in entry:
            raise RenderError(f"dsh composition must not set {plugin}.name: a patch entry keeps its own package")
        if plugin in BINDING_PLUGINS:
            check_binding_entry(plugin, entry, bound)
        elif plugin in UNBOUND_ADAPTERS and set(entry) - {"disabled"}:
            raise RenderError(f"dsh composition must not configure {plugin}: {refusal}")
        elif plugin in MODEL_ROUTE_KEYS:
            if holds_js(entry):
                raise RenderError(f"dsh composition must not write {plugin} as a js expression: {refusal}")
            config = entry.get("config")
            if not isinstance(config, dict):
                continue
            for parent, key in MODEL_ROUTE_KEYS[plugin]:
                value = config if parent is None else config.get(parent)
                for item in value if isinstance(value, list) else [value]:
                    if isinstance(item, dict) and key in item:
                        where = key if parent is None else f"{parent}.{key}"
                        raise RenderError(f"dsh composition must not set {plugin} {where}: {refusal}")


def finalize_render(files: dict[str, str]) -> dict[str, str]:
    entries = json.loads(files[_PATCH])
    log = entries.get("session-persistence-jsonl", {}).get("config", {})
    if log.get("compression") != "none":
        raise RenderError(
            "dsh composition must keep the session log uncompressed (compression: none) so Reef can read it"
        )
    for plugin in ("session-telemetry-otel", "session-title-llm"):
        if entries.get(plugin, {}).get("disabled") is not True:
            raise RenderError(f"dsh composition must keep {plugin} disabled for benchmark episodes")
    check_model_route(entries, json.loads(files[_ENV]))
    extensions = sorted(
        path[len(_EXTENSIONS) : -len(".mjs")]
        for path in files
        if path.startswith(_EXTENSIONS) and path.endswith(".mjs") and "/" not in path[len(_EXTENSIONS) :]
    )
    files[_PATCH] = _patch(entries, extensions)
    files[_ENV] = "".join(f"{key}={value}\n" for key, value in sorted(json.loads(files[_ENV]).items()))
    for path, text in list(files.items()):
        for root, user_only in ((_SKILLS, False), (_COMMANDS, True)):
            if path.startswith(root) and path.endswith("/SKILL.md"):
                files[path] = _with_frontmatter(path, text, user_only)
    return files
