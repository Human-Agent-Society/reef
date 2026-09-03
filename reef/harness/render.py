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


def render_composition(
    nodes: Sequence[tuple[str, Any]],
    descriptor: AdapterDescriptor,
    *,
    model_binding_nodes: Sequence[tuple[str, Any]] = (),
) -> dict[str, str]:
    """Render ``(kind, config)`` nodes to root-relative file texts.

    ``model_binding_nodes`` is the trusted, transient provider overlay Reef
    derives from a :class:`~reef.harness.model_binding.ModelBinding`. It is
    deliberately separate from the persisted composition: proxied adapters
    reserve the fields that overlay owns, so an evolved config cannot publish
    a provider redirect that evaluation silently replaces.
    """
    configs = {name: _deep_merge({}, target.defaults) for name, target in descriptor.config_targets.items()}
    rules: list[str] = []
    files: dict[str, str] = {}

    reserved: dict[str, set[str]] = {}
    if descriptor.model_binding_proxy:
        for templates in descriptor.model_binding.values():
            for template in templates:
                binding_target = str(template.get("target", "primary"))
                data = template.get("data", {})
                if isinstance(data, Mapping):
                    reserved.setdefault(binding_target, set()).update(str(key) for key in data)

    def emit(path: str, text: str) -> None:
        if path in files or any(path == target.path for target in descriptor.config_targets.values()):
            raise RenderError(f"two nodes render to the same path {path!r}")
        files[path] = text

    def apply(kind: str, config: Any, *, trusted_binding: bool = False) -> None:
        options: Mapping[str, Any] = config if isinstance(config, Mapping) else {}
        if kind == "config":
            target_name = str(options.get("target", "primary"))
            if target_name not in configs:
                raise RenderError(f"adapter {descriptor.name!r} declares no config target {target_name!r}")
            data = options.get("data", {})
            if not isinstance(data, Mapping):
                raise RenderError("config node data must be an object")
            overlap = sorted(reserved.get(target_name, set()) & set(data)) if not trusted_binding else []
            if overlap:
                raise RenderError(
                    f"{descriptor.name} config fields are reserved for Reef's transient model binding: "
                    f"{', '.join(overlap)}"
                )
            configs[target_name] = _deep_merge(configs[target_name], data)
        elif kind == "rules":
            rules.append(str(options.get("text", "")).strip())
        elif kind in ("agent_command", "skill"):
            emit(descriptor.node_paths[kind].format(name=options.get("name")), str(options.get("text", "")))
        elif kind == "code_extension":
            emit(descriptor.node_paths[kind].format(name=options.get("name")), str(options.get("code", "")))
        elif kind == "native_tool":
            template = descriptor.node_paths.get(kind)
            if template is None:
                raise RenderError(f"adapter {descriptor.name!r} does not render native_tool nodes")
            # One importable module: the declaration as constants, then the code that defines run(args, workdir).
            header = "\n".join(
                f"{key} = {value!r}"
                for key, value in (
                    ("NAME", options.get("name")),
                    ("DESCRIPTION", options.get("description", "")),
                    ("PARAMETERS", dict(options.get("parameters", {}) or {})),
                )
            )
            emit(template.format(name=options.get("name")), f"{header}\n\n{options.get('code', '')}")
        else:
            raise RenderError(f"unknown node kind {kind!r}")

    for kind, config in nodes:
        apply(kind, config)
    for kind, config in model_binding_nodes:
        if kind != "config":
            raise RenderError("model binding may only contribute config nodes")
        apply(kind, config, trusted_binding=True)

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
