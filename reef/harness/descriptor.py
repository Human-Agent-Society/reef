"""Declarative adapter descriptors: how one harness renders, runs, and cleans.

An ``AdapterDescriptor`` is loaded from a ``descriptor.yaml`` and states
everything the shared engines need to drive one harness binary:

- ``config_targets``: named JSON config files with their enforced defaults
  (a ``primary`` target is required; ``config`` nodes merge into targets).
- ``node_paths``: root-relative render paths for the other node kinds;
  named kinds carry a ``{name}`` placeholder.
- ``argv``/``env``: the headless invocation (``{prompt}`` substituted per
  episode) and the relocation environment (``{root}`` substituted per
  episode) that points the binary's whole composition at the episode root.
- ``trajectory``: which session-log reader applies and where it reads.
- ``cleanup_whitelist``: root-relative patterns for files the harness
  legitimately creates (session storage, boot mutations); anything else
  left behind is reported as residue.
- ``writable_paths``: root-relative state directories that a hosted sandbox
  makes writable while keeping the rendered files inside them read-only.
- ``install`` (optional): the vendor's install channel for the binary at a
  pinned version, consumed by the served install script; reef never hosts
  or proxies binary bytes.
- ``inherit_env`` (optional): names of host environment variables the episode
  may inherit. Episodes are otherwise hermetic - ``_inherited_env`` keeps only
  PATH, SYSTEMROOT and TMPDIR - which is right for a harness that needs
  nothing from the host. An adapter that reaches a hosted sandbox needs that
  provider's credential, and naming it here keeps the exception explicit
  instead of widening the allowlist for every adapter.
- ``self_isolating`` (optional): the adapter runs episodes inside its own
  container, so it cannot be nested in Reef's jail. A deployment that
  configures the sandbox executor for one of these refuses to boot rather
  than running it unisolated under a config that promised otherwise.

A descriptor may name a ``quirks`` module: its ``cleanup_whitelist`` extends
the declared one and its ``finalize_render`` callable gets the last word on
the rendered tree (the seam that enforces adapter traps such as opencode's
``autoupdate: false``). External adapters register through the
``reef.harness_adapters`` entry point group, each entry resolving to an
``AdapterDescriptor`` or a zero-argument callable returning one.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from reef.core.errors import ReefError

ENTRY_POINT_GROUP = "reef.harness_adapters"

#: Node kinds rendered to one path per named node; templates need ``{name}``.
NAMED_NODE_KINDS = ("agent_command", "skill", "code_extension")
#: Named kinds an adapter may leave out; a mutation of that kind then fails to render under it.
OPTIONAL_NODE_KINDS = ("native_tool", "native_hook")


class DescriptorError(ReefError):
    """The adapter descriptor is missing, malformed, or self-inconsistent."""


@dataclass(frozen=True)
class ConfigTarget:
    """One JSON config file: its render path and the defaults merged first."""

    path: str
    defaults: Mapping[str, Any] = field(default_factory=dict)


#: Vendor install kinds the install-script generator can render: an npm
#: package at a version, or a git checkout at a ref installed editable into
#: a venv (the channel of a Python agent that publishes no wheel).
INSTALL_KINDS = ("npm", "git")

#: Install fields land inside generated shell text, so their charsets are
#: pinned to what package registries actually use: names may add ``@`` and
#: ``/`` for scopes, versions and refs stay to dotted identifiers, and a
#: repository is an https URL.
_INSTALL_PACKAGE_PATTERN = re.compile(r"^[@A-Za-z0-9._/-]+$")
_INSTALL_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
_INSTALL_REPOSITORY_PATTERN = re.compile(r"^https://[A-Za-z0-9._/-]+$")


@dataclass(frozen=True)
class InstallSpec:
    """How a consumer gets the harness binary: the vendor's channel and pin.

    Reef never hosts or proxies binaries; this section only names the
    vendor's own install path (``kind``), the package it installs, the
    pinned ``version`` (what ``--version`` must report), and where the
    installed binary lands relative to the install prefix (``binary_path``).
    The ``git`` kind adds the ``repository`` to clone and the ``ref`` to
    check out.
    """

    kind: str
    package: str
    version: str
    binary_path: str
    repository: str = ""
    ref: str = ""


@dataclass(frozen=True)
class AdapterDescriptor:
    """Everything the shared render and episode engines know about one harness."""

    name: str
    binary: str
    argv: tuple[str, ...]
    env: Mapping[str, str]
    config_targets: Mapping[str, ConfigTarget]
    node_paths: Mapping[str, str]
    trajectory_format: str
    trajectory_path: str
    cleanup_whitelist: tuple[str, ...] = ()
    writable_paths: tuple[str, ...] = ()
    finalize_render: Callable[[dict[str, str]], dict[str, str]] | None = None
    install: InstallSpec | None = None
    #: True when the adapter isolates episodes itself (a task container) and
    #: cannot be nested inside :class:`~reef.harness.executor.SandboxExecutor`.
    self_isolating: bool = False
    #: Host environment variable names this adapter's episodes may inherit.
    inherit_env: tuple[str, ...] = ()
    #: ``config`` node templates that point this harness at a model endpoint,
    #: keyed by API dialect (``openai``, ``responses``, ``anthropic``): ``{base_url}``,
    #: ``{api_key}`` and ``{model}`` substitute into string values. Reef appends
    #: the matching set when it runs evaluation episodes, so the served tree
    #: never carries a provider binding.
    model_binding: Mapping[str, tuple[Mapping[str, Any], ...]] = field(default_factory=dict)


def _require_str(data: Mapping[str, Any], key: str, where: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise DescriptorError(f"{where} requires a non-empty string {key!r}")
    return value


def _str_list(value: Any, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise DescriptorError(f"{where} must be a list of non-empty strings")
    return tuple(value)


def _relative_paths(value: Any, where: str) -> tuple[str, ...]:
    paths = _str_list(value, where)
    for path in paths:
        pure = PurePosixPath(path)
        if pure.is_absolute() or ".." in pure.parts or not pure.parts or pure.parts[0] == "workspace":
            raise DescriptorError(f"{where} entries must stay below the episode root and outside workspace")
    return paths


def load_descriptor(path: Path) -> AdapterDescriptor:
    """Load and validate one ``descriptor.yaml``, resolving its quirks module."""
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise DescriptorError(f"cannot load adapter descriptor {path}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise DescriptorError(f"adapter descriptor {path} must be a YAML object")
    name = _require_str(data, "name", "descriptor")
    where = f"descriptor {name!r}"
    files = data.get("files")
    if not isinstance(files, Mapping):
        raise DescriptorError(f"{where} requires a 'files' object")
    config_targets = _parse_config_targets(files.get("config"), where)
    node_paths = _parse_node_paths(files, where)
    trajectory = data.get("trajectory")
    if not isinstance(trajectory, Mapping):
        raise DescriptorError(f"{where} requires a 'trajectory' object")
    env = data.get("env", {})
    if not isinstance(env, Mapping) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in env.items()
    ):
        raise DescriptorError(f"{where} 'env' must map strings to strings")
    whitelist = _str_list(data.get("cleanup_whitelist", []), f"{where} 'cleanup_whitelist'")
    writable_paths = _relative_paths(data.get("writable_paths", []), f"{where} 'writable_paths'")
    inherit_env = _str_list(data.get("inherit_env", []), f"{where} 'inherit_env'")
    self_isolating = data.get("self_isolating", False)
    if not isinstance(self_isolating, bool):
        raise DescriptorError(f"{where} 'self_isolating' must be a boolean")
    finalize, quirk_whitelist = _load_quirks(data.get("quirks"), where)
    return AdapterDescriptor(
        name=name,
        binary=_require_str(data, "binary", where),
        argv=_str_list(data.get("argv"), f"{where} 'argv'"),
        env=dict(env),
        config_targets=config_targets,
        node_paths=node_paths,
        trajectory_format=_require_str(trajectory, "format", f"{where} trajectory"),
        trajectory_path=_require_str(trajectory, "path", f"{where} trajectory"),
        cleanup_whitelist=whitelist + quirk_whitelist,
        writable_paths=writable_paths,
        finalize_render=finalize,
        install=_parse_install(data.get("install"), where),
        inherit_env=inherit_env,
        self_isolating=self_isolating,
        model_binding=_parse_model_binding(data.get("model_binding"), config_targets, where),
    )


def _parse_config_targets(raw: Any, where: str) -> dict[str, ConfigTarget]:
    if not isinstance(raw, Mapping) or "primary" not in raw:
        raise DescriptorError(f"{where} requires files.config with a 'primary' target")
    targets: dict[str, ConfigTarget] = {}
    for target_name, target in raw.items():
        if not isinstance(target, Mapping):
            raise DescriptorError(f"{where} config target {target_name!r} must be an object")
        defaults = target.get("defaults", {})
        if not isinstance(defaults, Mapping):
            raise DescriptorError(f"{where} config target {target_name!r} defaults must be an object")
        targets[str(target_name)] = ConfigTarget(
            path=_require_str(target, "path", f"{where} config target {target_name!r}"),
            defaults=dict(defaults),
        )
    return targets


def _parse_model_binding(
    raw: Any,
    config_targets: Mapping[str, ConfigTarget],
    where: str,
) -> dict[str, tuple[Mapping[str, Any], ...]]:
    """``model_binding`` is a mapping of API dialect to config node templates;
    a bare list is the ``openai`` set."""
    if raw is None:
        return {}
    if isinstance(raw, list):
        raw = {"openai": raw}
    if not isinstance(raw, Mapping) or not raw:
        raise DescriptorError(
            f"{where} 'model_binding' must map an api (openai, responses, anthropic) to config node templates"
        )
    parsed: dict[str, tuple[Mapping[str, Any], ...]] = {}
    for api, templates in raw.items():
        label = f"{where} model_binding.{api}"
        if not isinstance(templates, list) or not templates:
            raise DescriptorError(f"{label} must be a non-empty list of config node templates")
        nodes: list[Mapping[str, Any]] = []
        for index, node in enumerate(templates):
            if not isinstance(node, Mapping):
                raise DescriptorError(f"{label}[{index}] must be an object")
            target = str(node.get("target", "primary"))
            if target not in config_targets:
                raise DescriptorError(f"{label}[{index}] names unknown config target {target!r}")
            data = node.get("data")
            if not isinstance(data, Mapping):
                raise DescriptorError(f"{label}[{index}] requires a 'data' object")
            nodes.append({"target": target, "data": dict(data)})
        parsed[str(api)] = tuple(nodes)
    return parsed


def _parse_install(raw: Any, where: str) -> InstallSpec | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise DescriptorError(f"{where} 'install' must be an object")
    kind = _require_str(raw, "kind", f"{where} install")
    if kind not in INSTALL_KINDS:
        known = ", ".join(INSTALL_KINDS)
        raise DescriptorError(f"{where} install kind {kind!r} is not supported; known kinds: {known}")
    package = _require_str(raw, "package", f"{where} install")
    if not _INSTALL_PACKAGE_PATTERN.fullmatch(package):
        raise DescriptorError(f"{where} install 'package' {package!r} must match {_INSTALL_PACKAGE_PATTERN.pattern}")
    version = _require_str(raw, "version", f"{where} install")
    if not _INSTALL_VERSION_PATTERN.fullmatch(version):
        raise DescriptorError(f"{where} install 'version' {version!r} must match {_INSTALL_VERSION_PATTERN.pattern}")
    repository = ref = ""
    if kind == "git":
        repository = _require_str(raw, "repository", f"{where} install")
        if not _INSTALL_REPOSITORY_PATTERN.fullmatch(repository):
            raise DescriptorError(
                f"{where} install 'repository' {repository!r} must match {_INSTALL_REPOSITORY_PATTERN.pattern}"
            )
        ref = _require_str(raw, "ref", f"{where} install")
        if not _INSTALL_VERSION_PATTERN.fullmatch(ref):
            raise DescriptorError(f"{where} install 'ref' {ref!r} must match {_INSTALL_VERSION_PATTERN.pattern}")
    return InstallSpec(
        kind=kind,
        package=package,
        version=version,
        binary_path=_require_str(raw, "binary_path", f"{where} install"),
        repository=repository,
        ref=ref,
    )


def _parse_node_paths(files: Mapping[str, Any], where: str) -> dict[str, str]:
    node_paths = {"rules": _require_str(files, "rules", f"{where} files")}
    for kind in NAMED_NODE_KINDS:
        template = _require_str(files, kind, f"{where} files")
        if "{name}" not in template:
            raise DescriptorError(f"{where} files.{kind} template must contain {{name}}")
        node_paths[kind] = template
    for kind in OPTIONAL_NODE_KINDS:
        if kind not in files:
            continue
        template = _require_str(files, kind, f"{where} files")
        if "{name}" not in template:
            raise DescriptorError(f"{where} files.{kind} template must contain {{name}}")
        node_paths[kind] = template
    return node_paths


def _load_quirks(
    module_name: Any, where: str
) -> tuple[Callable[[dict[str, str]], dict[str, str]] | None, tuple[str, ...]]:
    if module_name is None:
        return None, ()
    if not isinstance(module_name, str):
        raise DescriptorError(f"{where} 'quirks' must be a dotted module name")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise DescriptorError(f"{where} cannot import quirks module {module_name!r}: {exc}") from exc
    finalize = getattr(module, "finalize_render", None)
    if finalize is not None and not callable(finalize):
        raise DescriptorError(f"{where} quirks finalize_render must be callable")
    whitelist = tuple(getattr(module, "cleanup_whitelist", ()))
    if not all(isinstance(item, str) and item for item in whitelist):
        raise DescriptorError(f"{where} quirks cleanup_whitelist must contain non-empty strings")
    return finalize, whitelist


def external_descriptors() -> dict[str, AdapterDescriptor]:
    """Adapters other distributions register on the entry point group."""
    from importlib.metadata import entry_points

    descriptors: dict[str, AdapterDescriptor] = {}
    for entry_point in entry_points(group=ENTRY_POINT_GROUP):
        loaded = entry_point.load()
        descriptor = loaded() if not isinstance(loaded, AdapterDescriptor) else loaded
        if not isinstance(descriptor, AdapterDescriptor):
            raise DescriptorError(f"entry point {entry_point.name!r} did not produce an AdapterDescriptor")
        descriptors[descriptor.name] = descriptor
    return descriptors
