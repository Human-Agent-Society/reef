"""Render a harness tree into the files one adapter's binary reads.

``render_tree`` is a pure function of a tree, an adapter descriptor and
optional config overlays, so a candidate and the current tree render the
same way for an episode, for an install, and for the files a client pulls.
Each config target merges, in order, the descriptor's enforced defaults, the
tree's ``config/<target>.json`` and the overlays, so an overlay wins every
key it writes. ``ModelBinding.config_overlays`` makes the overlays that point
the harness at a model. Every other file lands at the render path of its part
(the adapter's ``parts``). The shared checks then refuse a config that
fails a check its target declares, and the descriptor's ``renderer``
processes the config files and the files of the ``skill`` and ``command``
formats, checks the model route, writes each config file in its harness's
format, and gets the last word in ``finalize_render``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

import tomli_w
import yaml

from reef.core.errors import ReefError
from reef.harness.adapters.descriptor import AdapterDescriptor
from reef.harness.adapters.matchers import Each, select
from reef.harness.adapters.parts import Format
from reef.harness.tree.tree import HarnessTree


class RenderError(ReefError):
    """The tree cannot be rendered for this adapter."""


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """``base`` with ``overlay`` merged in: nested objects merge key by key, every other value of ``overlay`` wins."""
    merged = dict(base)
    for key, value in overlay.items():
        below = merged.get(key)
        merged[key] = deep_merge(below, value) if isinstance(below, Mapping) and isinstance(value, Mapping) else value
    return merged


def render_tree(
    tree: HarnessTree,
    descriptor: AdapterDescriptor,
    overlays: Sequence[tuple[str, Mapping[str, Any]]] = (),
    *,
    layer: bool = False,
) -> dict[str, str]:
    """The files ``descriptor``'s binary reads for ``tree``, by path relative to the episode or install root.

    ``overlays`` are ``(config target, data)`` pairs, merged in order after
    the tree's config. A model binding's overlays point the harness at a
    model; without them, the files carry no endpoint, as a client pulls them.
    Every rendered file ends in a newline. Raises RenderError for a path that
    no part of the adapter takes, a config file that is not a JSON object, a
    config target the adapter does not declare, two files that land on one
    path, and whatever the adapter's renderer refuses.

    With ``layer``, the files are a layer over a person's own configuration
    (``AdapterDescriptor.layer``): no config target starts from its episode
    defaults, the targets' checks do not run, and the renderer's
    ``layer_files`` adds what the layer's arguments name. The defaults and
    checks keep episodes deterministic, and the tree passed the checks when it
    was published. RenderError when the adapter has no layer.
    """
    if layer and descriptor.layer is None:
        raise RenderError(f"adapter {descriptor.name!r} does not run on a person's machine: it declares no layer")
    configs = {
        name: {} if layer else deep_merge({}, target.defaults) for name, target in descriptor.config_targets.items()
    }
    owners = {target.path: f"config target {name!r}" for name, target in descriptor.config_targets.items()}
    files: dict[str, str] = {}
    formats: dict[str, Format] = {}
    for path, text in tree.files.items():
        location = descriptor.layout.locate(path)
        if location is None:
            raise RenderError(f"{path} is not a path of the {descriptor.name} harness tree")
        if location.part.format is Format.CONFIG:
            if location.name not in configs:
                raise RenderError(f"adapter {descriptor.name!r} declares no config target {location.name!r}")
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                raise RenderError(f"{path} is not JSON: {exc}") from exc
            if not isinstance(data, Mapping):
                raise RenderError(f"{path} must hold a JSON object")
            configs[location.name] = deep_merge(configs[location.name], data)
            continue
        rendered_path = location.part.render_path(location.name)
        if rendered_path in owners:
            raise RenderError(f"{path} and {owners[rendered_path]} both render to {rendered_path!r}; rename one")
        owners[rendered_path] = path
        files[rendered_path] = text if text.endswith("\n") else text + "\n"
        formats[rendered_path] = location.part.format
    for target_name, data in overlays:
        if target_name not in configs:
            raise RenderError(f"adapter {descriptor.name!r} declares no config target {target_name!r}")
        configs[target_name] = deep_merge(configs[target_name], data)
    for name, target in descriptor.config_targets.items():
        files[target.path] = json.dumps(configs[name], indent=2, sort_keys=True) + "\n"
    rendered = apply_adapter_renderer(descriptor, files, formats, checks=not layer)
    return descriptor.renderer.layer_files(rendered) if layer else rendered


def apply_adapter_renderer(
    descriptor: AdapterDescriptor, files: dict[str, str], formats: Mapping[str, Format], *, checks: bool = True
) -> dict[str, str]:
    """Run the adapter's renderer steps over the laid-out files, whose config files are JSON objects.

    ``formats`` maps each rendered path that is not a config file to its part's format. ``checks`` runs the config
    targets' checks; a layer's render skips them.
    """
    renderer = descriptor.renderer
    merged = {target.path: json.loads(files[target.path]) for target in descriptor.config_targets.values()}
    check_model_route(descriptor, merged)
    if checks:
        check_config(descriptor, merged)
    configs = {path: renderer.process_config(path, config) for path, config in merged.items()}
    skills = {
        path: renderer.process_skill(path, files[path]) for path, format in formats.items() if format is Format.SKILL
    }
    commands = {
        path: renderer.process_command(path, files[path])
        for path, format in formats.items()
        if format is Format.COMMAND
    }
    files.update(skills)
    files.update(commands)
    renderer.check_model_route(configs, skills, commands)
    for path, config in configs.items():
        files[path] = write_config(descriptor.name, path, config)
    return renderer.finalize_render(files)


def check_model_route(descriptor: AdapterDescriptor, configs: Mapping[str, Mapping[str, Any]]) -> None:
    """Refuse a tree that sets a key Reef's model binding writes, unless the binding's credential is there too.

    Every model call goes to Reef, through the endpoint, credential and model the binding writes. The binding renders
    after the tree and wins every key it writes, so the tree can set one of those keys only where no binding is
    applied, as when validation renders a tree alone. The binding's credential tells the two apart: validation
    refuses an inline credential, so only the binding can hold one. Each adapter refuses the other routes its harness
    reads, such as a fallback model, in its renderer's ``check_model_route``.
    """
    keys = [
        (descriptor.config_targets[template["target"]].path, segments, isinstance(value, str) and "{api_key}" in value)
        for templates in descriptor.model_binding.values()
        for template in templates
        for segments, value in binding_leaves(template["data"], ())
    ]
    bound = any(
        isinstance(value, str) and bool(value.strip())
        for path, segments, is_credential in keys
        if is_credential
        for value in values_at(configs[path], segments)
    )
    if bound:
        return
    for path, segments, _ in keys:
        if values_at(configs[path], segments):
            raise RenderError(
                f"{descriptor.name} harness tree must not set {'.'.join(segments)} in {path}: "
                "Reef's model binding writes it"
            )


def check_config(descriptor: AdapterDescriptor, configs: Mapping[str, Mapping[str, Any]]) -> None:
    """Refuse a tree whose merged config fails one of the checks its target declares.

    The checks run once for every adapter, after the binding check and before the renderer's steps, so a declared
    check holds on every rendered tree and no adapter can skip it. A refusal names the key, the entry or item of
    an ``Each`` that fails, and the check's reason, or its match expression when it gives none. It never names the
    value, which can be a credential the binding wrote.
    """
    for target in descriptor.config_targets.values():
        for check in target.checks:
            for path, value in select(configs[target.path], check.segments):
                if check.matcher.matches(value):
                    continue
                failure = check.matcher.first_failure(value) if isinstance(check.matcher, Each) else None
                where = " ".join(part for part in (".".join(path), failure) if part) or "the whole file"
                raise RenderError(
                    f"{descriptor.name} harness tree fails the check on {where} in {target.path}: "
                    f"{check.reason or f'it must match {check.source}'}"
                )


def binding_leaves(data: Mapping[str, Any], segments: tuple[str, ...]) -> list[tuple[tuple[str, ...], Any]]:
    """Each key path a binding template writes, with the template's value there."""
    leaves: list[tuple[tuple[str, ...], Any]] = []
    for key, value in data.items():
        if isinstance(value, Mapping) and value:
            leaves.extend(binding_leaves(value, (*segments, str(key))))
        else:
            leaves.append(((*segments, str(key)), value))
    return leaves


def values_at(data: Any, segments: tuple[str, ...]) -> list[Any]:
    """The values at a binding key path in ``data``; a segment holding a placeholder such as ``{model}`` matches any
    key."""
    if not segments:
        return [data]
    if not isinstance(data, Mapping):
        return []
    first, rest = segments[0], segments[1:]
    if "{" in first:
        return [value for item in data.values() for value in values_at(item, rest)]
    return values_at(data[first], rest) if first in data else []


def write_config(adapter: str, path: str, config: Mapping[str, Any]) -> str:
    """The text of one config file in the format its suffix names: TOML, YAML, ``KEY=value`` lines for ``.env``, and
    JSON for ``.json`` or any other suffix, as every config file was written before formats followed suffixes."""
    pure = PurePosixPath(path)
    # A dotfile such as .env has no suffix, only a name.
    suffix = pure.suffix or pure.name
    if suffix == ".toml":
        try:
            return tomli_w.dumps(config)
        except (TypeError, ValueError) as exc:
            raise RenderError(f"{adapter} config cannot be represented as TOML: {exc}") from exc
    if suffix in (".yaml", ".yml"):
        return yaml.dump(dict(config), sort_keys=True, default_flow_style=False, allow_unicode=True)
    if suffix == ".env":
        return "".join(f"{key}={value}\n" for key, value in sorted(config.items()))
    return json.dumps(config, indent=2, sort_keys=True) + "\n"
