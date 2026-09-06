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
