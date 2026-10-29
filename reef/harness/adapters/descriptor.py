"""Declarative adapter descriptors: how one harness renders, runs, and cleans.

An ``AdapterDescriptor`` is loaded from a ``descriptor.yaml`` and the
``tree.yaml`` beside it, and states everything the shared engines need to
drive one harness binary:

- ``layout``: the parts of the harness tree the adapter takes, from
  ``tree.yaml`` (``reef.harness.adapters.parts``): each part's tree path, format
  and render path. A tree file that no part takes is refused by validation
  and at render, never dropped.
- ``config_targets``: named JSON config files (``config:``) with their
  enforced defaults (a ``primary`` target is required; the tree's config part
  merges into a target) and their checks (a key of the merged config and the
  matcher every value there must match; see
  :mod:`reef.harness.adapters.matchers`).
- ``argv``/``env``: the headless invocation (``{prompt}`` substituted per
  rollout) and the relocation environment (``{root}`` substituted per
  rollout) that points the binary's whole configuration at the rollout root.
- ``trajectory``: which session-log reader applies and where it reads.
- ``cleanup_whitelist``: root-relative patterns for files the harness
  legitimately creates (session storage, boot mutations); anything else
  left behind is reported as residue.
- ``writable_paths``: root-relative state directories that a hosted sandbox
  makes writable while keeping the rendered files inside them read-only.
- ``install`` (optional): the vendor's install channel for the binary at a
  pinned version, which Reef installs on the host that runs the rollouts;
  Reef never hosts or proxies binary bytes.
- ``self_isolating`` (optional): the adapter runs rollouts inside its own
  container, so nesting in Reef's jail is refused unless its execution quirk
  validates a compatible configuration (such as a remote task environment).
- ``host_env`` (optional): service environment variables a local rollout
  keeps, for a host tool the relocated ``HOME`` would otherwise hide.
- ``is_root_bind_mounted`` (optional): the binary bind-mounts paths below the
  rollout root into a container, so on macOS a local rollout makes its root
  under the home directory, which Docker's VM shares with the host.

A descriptor may name a ``quirks`` module: its ``cleanup_whitelist`` extends
the declared one and the :class:`AdapterRenderer` subclass it defines adds the
adapter's own render steps (the seam that enforces what the declaration cannot
state, such as opencode's agent and command shapes; what a config must keep or
must not set is declared under ``config.<target>.checks`` and checked by
the shared render). External adapters register through the
``reef.harness_adapters`` entry point group, each entry resolving to an
``AdapterDescriptor`` or a zero-argument callable returning one.
"""

from __future__ import annotations

import importlib
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from reef.core.errors import ReefError
from reef.execution.sandbox.process import ProgramExecutor
from reef.harness.adapters.matchers import KeyDefault, Matcher, parse_matcher, select
from reef.harness.adapters.parts import TREE_FILE, TreeLayout, TreeLayoutError, load_tree_layout


class ExecutionValidator(ABC):
    """Adapter-specific constraints checked by the shared rollout lifecycle."""

    @abstractmethod
    def __call__(self, files: Mapping[str, str], executor: ProgramExecutor) -> None: ...


class AdapterRenderer:
    """An adapter's own render steps, each given only the files of its concern.

    The shared render runs them in a fixed order: ``process_config`` for each config file, ``process_skill`` and
    ``process_command`` for each file of a part whose format is ``skill`` or ``command``, ``check_model_route``,
    then ``finalize_render`` after the render has written each config file as text in the format its suffix names. Before any of them, the render
    refuses a tree that sets a key Reef's model binding writes without the binding's credential, and a tree whose
    config fails a check its target declares. A rule on config values belongs in those checks; a step is for
    changing files and for what a check cannot state, such as a model in a skill's frontmatter or a value that
    must name an agent the config defines.

    Although ``finalize_render`` is an omnipower API that gives the users the maximum freedom for customization,
    it is a last resort since that method has a large impact surface and make it hard to maintain. So instead users
    should try using ``process_config``, ``process_skill`` and ``process_command`` whenever possible.

    Each default leaves its input unchanged, so an adapter overrides only the steps it needs. A step refuses a tree
    by raising ``RenderError``.

    The render uses the class itself and never makes an instance, and every step is a static method, which
    ``__init_subclass__`` checks: a step's output depends only on its arguments, so no step can pass state to a
    later one or to another render. Keep module level state out of the steps too; the check cannot see it.
    """

    #: The steps a subclass may override, each as a static method.
    STEPS = (
        "process_config",
        "process_skill",
        "process_command",
        "check_model_route",
        "finalize_render",
    )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for step in AdapterRenderer.STEPS:
            if step in vars(cls) and not isinstance(vars(cls)[step], staticmethod):
                raise TypeError(f"{cls.__name__}.{step} must be a static method, so render steps keep no state")

    @staticmethod
    def process_config(path: str, config: dict[str, Any]) -> dict[str, Any]:
        """Check and adjust one merged config file, the tree's config entries over the descriptor's defaults.

        ``path`` is the file's root-relative path, such as ``codex/config.toml``; the returned config is written
        there in the format its suffix names.

        A config is the file's content as a JSON object: its values are only ``dict``, ``list``, ``str``, ``int``,
        ``float``, ``bool`` or ``None``, nested to any depth. It is typed ``Any`` because a precise recursive JSON type
        would make every nested read (``config.get("env", {}).get(name)``) need an ``isinstance`` check first; a step
        checks the type of a value before it relies on it instead.
        """
        return config

    @staticmethod
    def process_skill(path: str, text: str) -> str:
        """Check and adjust one rendered file of a ``skill`` format part."""
        return text

    @staticmethod
    def process_command(path: str, text: str) -> str:
        """Check and adjust one rendered file of a ``command`` format part."""
        return text

    @staticmethod
    def check_model_route(
        configs: Mapping[str, Mapping[str, Any]], skills: Mapping[str, str], commands: Mapping[str, str]
    ) -> None:
        """Refuse a model route the harness reads beyond the keys Reef's model binding writes, such as a fallback
        model or a second provider.

        ``configs`` maps each config file's path to its content after ``process_config``. ``skills`` and
        ``commands`` map the path of each file of a ``skill`` and of a ``command`` format part to its text after
        ``process_skill`` and ``process_command``, for a harness that reads a model from their frontmatter. The render has already
        refused a tree that sets one of the binding's own keys without the binding's credential.

        A config is the file's content as a JSON object: its values are only ``dict``, ``list``, ``str``, ``int``,
        ``float``, ``bool`` or ``None``, nested to any depth. It is typed ``Any`` because a precise recursive JSON type
        would make every nested read (``config.get("env", {}).get(name)``) need an ``isinstance`` check first; a step
        checks the type of a value before it relies on it instead.
        """

    @staticmethod
    def finalize_render(files: dict[str, str]) -> dict[str, str]:
        """Work that spans several files, and any config file the format of its suffix does not fit.

        ``files`` holds every rendered file as text, each config file already written in the format its suffix
        names; a change to a config file reads that text again. The returned mapping is the rendered tree.
        """
        return files


ENTRY_POINT_GROUP = "reef.harness_adapters"


class DescriptorError(ReefError):
    """The adapter descriptor is missing, malformed, or self-inconsistent."""


#: The ``key`` of a check on the whole config file.
WHOLE_FILE_KEY = "."


@dataclass(frozen=True)
class ConfigCheck:
    """One condition on a merged config file: every value at ``key`` matches ``matcher``.

    ``key`` is dot separated key names, or ``.`` for the whole file; ``source`` is the match expression as the
    descriptor writes it, and ``reason`` says why the check exists. A refusal names both.
    """

    key: str
    matcher: Matcher
    source: str
    reason: str = ""

    @property
    def segments(self) -> tuple[str, ...]:
        return () if self.key == WHOLE_FILE_KEY else tuple(self.key.split("."))


@dataclass(frozen=True)
class ConfigTarget:
    """One JSON config file: its render path, the defaults merged first, and the checks the merged file must pass."""

    path: str
    defaults: Mapping[str, Any] = field(default_factory=dict)
    #: The shared render runs them on the merged config, the binding included, before any adapter step runs, so no
    #: adapter can skip them.
    checks: tuple[ConfigCheck, ...] = ()


#: Vendor install kinds Reef can install for rollouts: an npm
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
    """Everything the shared render and rollout engines know about one harness."""

    name: str
    binary: str
    argv: tuple[str, ...]
    env: Mapping[str, str]
    config_targets: Mapping[str, ConfigTarget]
    layout: TreeLayout
    trajectory_format: str
    trajectory_path: str
    cleanup_whitelist: tuple[str, ...] = ()
    writable_paths: tuple[str, ...] = ()
    #: The adapter's own render steps; the base class changes nothing.
    renderer: type[AdapterRenderer] = AdapterRenderer
    install: InstallSpec | None = None
    #: True when the adapter isolates rollouts itself; nesting is refused
    #: unless validate_execution checks a compatible configuration.
    self_isolating: bool = False
    #: True when the ``{prompt}`` argv slot names a task directory (a Harbor task) rather than text the
    #: model reads, so a gate can hand the adapter task directories instead of prompts.
    is_prompt_task_directory: bool = False
    #: Config templates (``target`` and ``data``) that point this harness at a model endpoint,
    #: keyed by API dialect (``openai``, ``responses``, ``anthropic``): ``{base_url}``,
    #: ``{api_key}`` and ``{model}`` substitute into string values. Reef appends
    #: the matching set when it runs evaluation rollouts, so the served tree
    #: never carries a provider binding.
    model_binding: Mapping[str, tuple[Mapping[str, Any], ...]] = field(default_factory=dict)
    #: Service environment variables a rollout under the local executor keeps: the service's own value, else
    #: the default, where ``{home}`` is the service's home directory; an empty default leaves the variable
    #: unset. For a host tool the relocated ``HOME`` would otherwise hide, such as terminus's docker CLI.
    host_env: Mapping[str, str] = field(default_factory=dict)
    #: True when the binary bind-mounts paths below the rollout root into a container. On macOS Docker runs in a VM
    #: that shares the home directory, and colima does not share ``$TMPDIR``, so a rollout under the local executor
    #: then makes its root under ``~/.reef/rollouts``; on other platforms the root stays in the temp directory.
    is_root_bind_mounted: bool = False
    #: Optional quirk that validates the executor against the rendered tree,
    #: replacing the blanket self_isolating restriction before launch.
    validate_execution: ExecutionValidator | None = None


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
            raise DescriptorError(f"{where} entries must stay below the rollout root and outside workspace")
    return paths


def load_descriptor(path: Path) -> AdapterDescriptor:
    """Load and validate one ``descriptor.yaml`` and the ``tree.yaml`` beside it, resolving its quirks module."""
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise DescriptorError(f"cannot load adapter descriptor {path}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise DescriptorError(f"adapter descriptor {path} must be a YAML object")
    name = _require_str(data, "name", "descriptor")
    where = f"descriptor {name!r}"
    config_targets = _parse_config_targets(data.get("config"), where)
    try:
        layout = load_tree_layout(Path(path).with_name(TREE_FILE))
    except TreeLayoutError as exc:
        raise DescriptorError(f"{where}: {exc}") from exc
    trajectory = data.get("trajectory")
    if not isinstance(trajectory, Mapping):
        raise DescriptorError(f"{where} requires a 'trajectory' object")
    env = data.get("env", {})
    if not isinstance(env, Mapping) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in env.items()
    ):
        raise DescriptorError(f"{where} 'env' must map strings to strings")
    host_env = data.get("host_env", {})
    if not isinstance(host_env, Mapping) or not all(
        isinstance(key, str) and key.isidentifier() and isinstance(value, str) for key, value in host_env.items()
    ):
        raise DescriptorError(f"{where} 'host_env' must map variable names to default strings")
    overlap = sorted(set(host_env) & set(env))
    if overlap:
        raise DescriptorError(f"{where} 'host_env' names variables 'env' already sets: {', '.join(overlap)}")
    whitelist = _str_list(data.get("cleanup_whitelist", []), f"{where} 'cleanup_whitelist'")
    writable_paths = _relative_paths(data.get("writable_paths", []), f"{where} 'writable_paths'")
    self_isolating = data.get("self_isolating", False)
    if not isinstance(self_isolating, bool):
        raise DescriptorError(f"{where} 'self_isolating' must be a boolean")
    is_prompt_task_directory = data.get("is_prompt_task_directory", False)
    if not isinstance(is_prompt_task_directory, bool):
        raise DescriptorError(f"{where} 'is_prompt_task_directory' must be a boolean")
    is_root_bind_mounted = data.get("is_root_bind_mounted", False)
    if not isinstance(is_root_bind_mounted, bool):
        raise DescriptorError(f"{where} 'is_root_bind_mounted' must be a boolean")
    renderer, quirk_whitelist, validate_execution = _load_quirks(data.get("quirks"), where)
    model_binding = _parse_model_binding(data.get("model_binding"), config_targets, where)
    return AdapterDescriptor(
        name=name,
        binary=_require_str(data, "binary", where),
        argv=_str_list(data.get("argv"), f"{where} 'argv'"),
        env=dict(env),
        config_targets=config_targets,
        layout=layout,
        trajectory_format=_require_str(trajectory, "format", f"{where} trajectory"),
        trajectory_path=_require_str(trajectory, "path", f"{where} trajectory"),
        cleanup_whitelist=whitelist + quirk_whitelist,
        writable_paths=writable_paths,
        renderer=renderer,
        install=_parse_install(data.get("install"), where),
        self_isolating=self_isolating,
        is_prompt_task_directory=is_prompt_task_directory,
        host_env=dict(host_env),
        is_root_bind_mounted=is_root_bind_mounted,
        model_binding=model_binding,
        validate_execution=validate_execution,
    )


def _parse_config_targets(raw: Any, where: str) -> dict[str, ConfigTarget]:
    if not isinstance(raw, Mapping) or "primary" not in raw:
        raise DescriptorError(f"{where} requires config with a 'primary' target")
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
            checks=_parse_checks(target.get("checks", []), defaults, f"{where} config target {target_name!r} checks"),
        )
    return targets


def _parse_checks(raw: Any, defaults: Mapping[str, Any], where: str) -> tuple[ConfigCheck, ...]:
    """A target's ``checks``: each with ``key`` (one key or a list), ``match`` and an optional ``reason``.

    A check with a list of keys is one check per key, in order. A check without a key is refused, so a forgotten key
    never checks the whole file; ``.`` names the whole file.
    """
    if not isinstance(raw, list):
        raise DescriptorError(f"{where} must be a list")
    checks: list[ConfigCheck] = []
    for index, entry in enumerate(raw):
        place = f"{where}[{index}]"
        if not isinstance(entry, Mapping) or set(entry) - {"key", "match", "reason"}:
            raise DescriptorError(f"{place} must be an object with key and match, and optionally reason")
        source = entry.get("match")
        reason = entry.get("reason", "")
        if not isinstance(source, str) or not source.strip() or not isinstance(reason, str):
            raise DescriptorError(f"{place} requires a match expression, and its reason must be a string")
        if "key" not in entry:
            raise DescriptorError(
                f"{place} requires a key: dot separated key names, or {WHOLE_FILE_KEY!r} for the whole file"
            )
        keys = entry["key"]
        for key in _str_list(keys, f"{place} key") if isinstance(keys, list) else [keys]:
            if not isinstance(key, str) or (key != WHOLE_FILE_KEY and "" in key.split(".")):
                raise DescriptorError(
                    f"{place} key must be dot separated key names, or {WHOLE_FILE_KEY!r} for the whole file"
                )
            segments = () if key == WHOLE_FILE_KEY else tuple(key.split("."))
            default = None if {"*", "**"} & set(segments) else KeyDefault(select(dict(defaults), segments)[0][1])
            try:
                matcher = parse_matcher(source, default)
            except ValueError as error:
                raise DescriptorError(f"{place} match: {error}") from error
            checks.append(ConfigCheck(key=key, matcher=matcher, source=" ".join(source.split()), reason=reason))
    return tuple(checks)


def _parse_model_binding(
    raw: Any,
    config_targets: Mapping[str, ConfigTarget],
    where: str,
) -> dict[str, tuple[Mapping[str, Any], ...]]:
    """``model_binding`` is a mapping of API dialect to config templates;
    a bare list is the ``openai`` set."""
    if raw is None:
        return {}
    if isinstance(raw, list):
        raw = {"openai": raw}
    if not isinstance(raw, Mapping) or not raw:
        raise DescriptorError(
            f"{where} 'model_binding' must map an api (openai, responses, anthropic) to config templates"
        )
    parsed: dict[str, tuple[Mapping[str, Any], ...]] = {}
    for api, templates in raw.items():
        label = f"{where} model_binding.{api}"
        if not isinstance(templates, list) or not templates:
            raise DescriptorError(f"{label} must be a non-empty list of config templates")
        parsed_templates: list[Mapping[str, Any]] = []
        for index, template in enumerate(templates):
            if not isinstance(template, Mapping):
                raise DescriptorError(f"{label}[{index}] must be an object")
            target = str(template.get("target", "primary"))
            if target not in config_targets:
                raise DescriptorError(f"{label}[{index}] names unknown config target {target!r}")
            data = template.get("data")
            if not isinstance(data, Mapping):
                raise DescriptorError(f"{label}[{index}] requires a 'data' object")
            parsed_templates.append({"target": target, "data": dict(data)})
        parsed[str(api)] = tuple(parsed_templates)
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


def _load_quirks(
    module_name: Any, where: str
) -> tuple[type[AdapterRenderer], tuple[str, ...], ExecutionValidator | None]:
    if module_name is None:
        return AdapterRenderer, (), None
    if not isinstance(module_name, str):
        raise DescriptorError(f"{where} 'quirks' must be a dotted module name")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise DescriptorError(f"{where} cannot import quirks module {module_name!r}: {exc}") from exc
    # The renderer is the one AdapterRenderer subclass the module defines.
    renderers = [
        value
        for value in vars(module).values()
        if isinstance(value, type) and issubclass(value, AdapterRenderer) and value.__module__ == module.__name__
    ]
    if len(renderers) > 1:
        names = ", ".join(sorted(renderer.__name__ for renderer in renderers))
        raise DescriptorError(f"{where} quirks module defines more than one AdapterRenderer subclass: {names}")
    renderer = renderers[0] if renderers else AdapterRenderer
    whitelist = tuple(getattr(module, "cleanup_whitelist", ()))
    if not all(isinstance(item, str) and item for item in whitelist):
        raise DescriptorError(f"{where} quirks cleanup_whitelist must contain non-empty strings")
    validate_execution = getattr(module, "validate_execution", None)
    if validate_execution is not None and not isinstance(validate_execution, ExecutionValidator):
        raise DescriptorError(f"{where} quirks validate_execution must inherit ExecutionValidator")
    return renderer, whitelist, validate_execution


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
