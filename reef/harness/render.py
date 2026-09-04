"""Shared render engine: a composition's nodes to one harness's native files.

Rendering is a pure function of the node sequence and the adapter
descriptor, so a candidate tree can be rendered before and after a mutation
and compared without touching disk. Config nodes deep-merge into their
target in tree order over the descriptor's enforced defaults; rules nodes
concatenate in tree order into the rules file; named kinds render one file
per node through the descriptor's path templates. The descriptor's
``finalize_render`` quirk gets the last word, so adapter traps (opencode's
``autoupdate: false``) are enforced on every rendered tree, not just the
default one.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from reef.core.errors import ReefError
from reef.harness.descriptor import AdapterDescriptor


class RenderError(ReefError):
    """The composition cannot be rendered for this adapter."""


def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        below = merged.get(key)
        merged[key] = _deep_merge(below, value) if isinstance(below, dict) and isinstance(value, Mapping) else value
    return merged


def render_composition(nodes: Sequence[tuple[str, Any]], descriptor: AdapterDescriptor) -> dict[str, str]:
    """Render ``(kind, config)`` nodes to root-relative file texts."""
    configs = {name: _deep_merge({}, target.defaults) for name, target in descriptor.config_targets.items()}
    rules: list[str] = []
    files: dict[str, str] = {}

    def emit(path: str, text: str) -> None:
        if path in files or any(path == target.path for target in descriptor.config_targets.values()):
            raise RenderError(f"two nodes render to the same path {path!r}")
        files[path] = text

    for kind, config in nodes:
        options: Mapping[str, Any] = config if isinstance(config, Mapping) else {}
        if kind == "config":
            target_name = str(options.get("target", "primary"))
            if target_name not in configs:
                raise RenderError(f"adapter {descriptor.name!r} declares no config target {target_name!r}")
            configs[target_name] = _deep_merge(configs[target_name], options.get("data", {}))
        elif kind == "rules":
            rules.append(str(options.get("text", "")).strip())
        elif kind in ("agent_command", "skill"):
            emit(descriptor.node_paths[kind].format(name=options.get("name")), str(options.get("text", "")))
        elif kind == "code_extension":
            emit(descriptor.node_paths[kind].format(name=options.get("name")), str(options.get("code", "")))
        elif kind in ("native_tool", "native_hook"):
            template = descriptor.node_paths.get(kind)
            if template is None:
                raise RenderError(f"adapter {descriptor.name!r} does not render {kind} nodes")
            # One importable module: the code that defines run or listen, then
            # the declaration as constants, so the node config binds.
            fields: tuple[tuple[str, Any], ...]
            if kind == "native_hook":
                fields = (("NAME", options.get("name")), ("SEAM", options.get("seam")))
            else:
                fields = (
                    ("NAME", options.get("name")),
                    ("DESCRIPTION", options.get("description", "")),
                    ("PARAMETERS", dict(options.get("parameters", {}) or {})),
                    ("CAPABILITIES", list(options.get("capabilities", []) or [])),
                )
            header = "\n".join(f"{key} = {value!r}" for key, value in fields)
            emit(template.format(name=options.get("name")), f"{str(options.get('code', '')).rstrip()}\n\n{header}\n")
        else:
            raise RenderError(f"unknown node kind {kind!r}")

    for name, target in descriptor.config_targets.items():
        files[target.path] = json.dumps(configs[name], indent=2, sort_keys=True) + "\n"
    if rules:
        files[descriptor.node_paths["rules"]] = "\n\n".join(rules) + "\n"
    for path, text in files.items():
        if not text.endswith("\n"):
            files[path] = text + "\n"
    if descriptor.finalize_render is not None:
        files = descriptor.finalize_render(files)
        if not isinstance(files, dict):
            raise RenderError(f"adapter {descriptor.name!r} finalize_render must return the files mapping")
    return files
