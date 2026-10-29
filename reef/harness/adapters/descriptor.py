"""Declarative adapter descriptors: how one harness renders, runs, and cleans.

An ``AdapterDescriptor`` is loaded from an adapter's ``adapter.yaml``, which
declares four things and, optionally, the checks of the agent's config files:

- ``evolve``: what Reef may evolve. Each kind of file
  (``reef.harness.adapters.parts.KINDS``) maps to the path where the agent
  reads it; the kind gives the tree path, the format and the merge. With the
  ``config`` kind it makes ``layout``. A tree file that no part takes is
  refused by validation and at render, never dropped.
- ``dirs``: the agent's directories under the root, each with the variable
  that moves it there, or none. They make the relocation ``env``, the
  ``writable_paths`` that a hosted sandbox opens (rendered files inside stay
  read-only), and the ``cleanup_whitelist``: what the agent writes there is
  its state, and any other file it leaves behind is residue. ``sessions``
  names the session log's path and reader (``trajectory_format``).
- ``model``: per API dialect, the config templates that point the agent at a
  model endpoint (``model_binding``).
- ``run``: ``episode``, the headless invocation (``{prompt}`` substituted
  per episode) and the variables beside the relocation ones, and ``layer``,
  how a person's machine runs the agent (``AdapterLayer``).
- ``config`` (optional checks): named JSON config files with their enforced
  defaults (a ``primary`` target is required) and their checks (a key of the
  merged config and the matcher every value there must match; see
  :mod:`reef.harness.adapters.matchers`). A default may name a path under the
  root as ``{root}/...``; the program that writes the rendered files puts the
  root there (``AdapterDescriptor.place_root``).
- ``install`` (optional): the vendor's install channel for the binary at a
  pinned version, which Reef installs on the host that runs the episodes,
  and a client installs on a person's machine that lacks the agent; Reef
  never hosts or proxies binary bytes.
- ``self_isolating`` (optional): the adapter runs episodes inside its own
  container, so nesting in Reef's jail is refused unless its
  ``execution_validator`` accepts the configuration (such as a remote task
  environment).
- ``host_env`` (optional): service environment variables a local episode
  keeps, for a host tool the relocated ``HOME`` would otherwise hide.
- ``is_root_bind_mounted`` (optional): the binary bind-mounts paths below the
  episode root into a container, so on macOS a local episode makes its root
  under the home directory, which Docker's VM shares with the host.

A descriptor may name a ``renderer``, an :class:`AdapterRenderer` subclass
that adds the adapter's own render steps (the seam that enforces what the
declaration cannot state, such as opencode's agent and command shapes; what a
config must keep or must not set is declared under ``config.<target>.checks``
and checked by the shared render), and an ``execution_validator``, an
:class:`ExecutionValidator` subclass. Each is a ``package.module:ClassName``
reference, the form a deployment names a method in. External adapters register through the
``reef.harness_adapters`` entry point group, each entry resolving to an
``AdapterDescriptor`` or a zero-argument callable returning one.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from reef.core.errors import ReefError
from reef.core.plugins import Plugin, PluginConfigError, PluginReference
from reef.harness.adapters.matchers import KeyDefault, Matcher, parse_matcher, select
from reef.harness.adapters.parts import TreeLayout, TreeLayoutError, layout_from_evolve
from reef.sandbox.process import EpisodeExecutor


class ExecutionValidator(Plugin, ABC):
    """Adapter-specific constraints checked by the shared episode lifecycle.

    A descriptor names the class in ``execution_validator``; the loader builds it with no arguments.
    """

    @abstractmethod
    def __call__(self, files: Mapping[str, str], executor: EpisodeExecutor) -> None: ...


class AdapterRenderer(Plugin):
    """An adapter's own render steps, each given only the files of its concern.

    The shared render runs them in a fixed order: ``process_config`` for each config file, ``process_skill`` and
    ``process_command`` for each file of a part whose format is ``skill`` or ``command``, ``check_model_route``,
    then ``finalize_render`` after the render has written each config file as text in the format its suffix names.
    For a layer, ``reef.harness.sync.layer_install`` then runs ``layer_files``. Before any of them, the render refuses a tree that sets a key Reef's model binding writes without the binding's credential,
    and a tree whose config fails a check its target declares. A rule on config values belongs in those checks; a step is for
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
        "layer_files",
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

    @staticmethod
    def layer_files(files: dict[str, str]) -> dict[str, str]:
        """The files of a layer over a person's own configuration (``AdapterLayer``): ``files``, the tree rendered
        for the layer, with what the layer's ``register`` commands name and the tree lacks, such as the marketplace
        a plugin installs from, and without the files the install keeps no use for."""
        return files


ENTRY_POINT_GROUP = "reef.harness_adapters"
#: The file a bundled adapter is declared in, in its directory.
ADAPTER_FILE = "adapter.yaml"
#: The top-level keys of an ``adapter.yaml``.
ADAPTER_KEYS = frozenset(
    {
        "name",
        "binary",
        "install",
        "renderer",
        "execution_validator",
        "evolve",
        "dirs",
        "sessions",
        "model",
        "run",
        "config",
        "host_env",
        "self_isolating",
        "is_prompt_task_directory",
        "is_root_bind_mounted",
    }
)


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


#: Vendor install kinds Reef can install for episodes: an npm
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
class AdapterLayer:
    """How a person's machine runs the agent with a release over the person's own configuration.

    The client writes the release's layer files (``reef.harness.sync.layer_install``) under a directory of its
    own, the install, and runs the binary with ``args`` added and ``env`` set. The agent still loads the person's
    own configuration, which the client never writes. An agent that loads a part from no path a flag can name
    loads it once the agent's own commands in ``register`` record the install, as ``codex plugin add`` does.

    Values: ``args`` are non-empty groups of arguments, such as a flag and its value; ``env`` maps variables to
    values. ``register`` are the agent's commands, each its arguments, that the client runs after it installs a
    release's files; each must succeed when it runs again. ``unregister`` are the commands that undo them, which
    ``disconnect`` runs. Templates hold ``{root}`` (the install), ``{model}`` and ``{base_url}`` (the binding's),
    and ``{token}`` (only in ``env``: a command line is visible to other processes).
    Purpose: ``load_descriptor`` parses ``run.layer``; ``reef.harness.sync.layer_install`` fills it in.
    """

    args: tuple[tuple[str, ...], ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    register: tuple[tuple[str, ...], ...] = ()
    unregister: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        for name, groups in (("args", self.args), ("register", self.register), ("unregister", self.unregister)):
            if any(not group or not all(isinstance(arg, str) for arg in group) for group in groups):
                raise DescriptorError(f"layer {name} must be non-empty groups of strings")
            if any("{token}" in arg for group in groups for arg in group):
                raise DescriptorError(
                    f"layer {name} must not hold {{token}}: a command line is visible to other processes"
                )


@dataclass(frozen=True)
class AdapterDescriptor:
    """Everything the shared render and episode engines know about one harness."""

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
    #: True when the adapter isolates episodes itself; nesting is refused
    #: unless validate_execution checks a compatible configuration.
    self_isolating: bool = False
    #: True when the ``{prompt}`` argv slot names a task directory (a Harbor task) rather than text the
    #: model reads, so a gate can hand the adapter task directories instead of prompts.
    is_prompt_task_directory: bool = False
    #: Config templates (``target`` and ``data``) that point this harness at a model endpoint,
    #: keyed by API dialect (``openai``, ``responses``, ``anthropic``): ``{base_url}``,
    #: ``{api_key}`` and ``{model}`` substitute into string values. Reef appends
    #: the matching set when it runs evaluation episodes, so the served tree
    #: never carries a provider binding.
    model_binding: Mapping[str, tuple[Mapping[str, Any], ...]] = field(default_factory=dict)
    #: Service environment variables an episode under the local executor keeps: the service's own value, else
    #: the default, where ``{home}`` is the service's home directory; an empty default leaves the variable
    #: unset. For a host tool the relocated ``HOME`` would otherwise hide, such as a docker CLI.
    host_env: Mapping[str, str] = field(default_factory=dict)
    #: True when the binary bind-mounts paths below the episode root into a container. On macOS Docker runs in a VM
    #: that shares the home directory, and colima does not share ``$TMPDIR``, so an episode under the local executor
    #: then makes its root under ``~/.reef/episodes``; on other platforms the root stays in the temp directory.
    is_root_bind_mounted: bool = False
    #: Checks the executor against the rendered tree before launch, in place of the blanket self_isolating
    #: refusal; ``None`` keeps that refusal.
    execution_validator: ExecutionValidator | None = None
    #: How a person's machine runs the agent with a release over its own configuration; ``None`` when the adapter
    #: does not run on a person's machine.
    layer: AdapterLayer | None = None

    def place_root(self, files: Mapping[str, str], root: str) -> dict[str, str]:
        """``files`` with ``{root}`` in each config file replaced by ``root``, the directory that holds the render
        roots: the episode root, or a client's install. The root is escaped as a JSON string's content."""
        config_paths = {target.path for target in self.config_targets.values()}
        placed = json.dumps(root)[1:-1]
        return {path: text.replace("{root}", placed) if path in config_paths else text for path, text in files.items()}


def require_str(data: Mapping[str, Any], key: str, where: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise DescriptorError(f"{where} requires a non-empty string {key!r}")
    return value


def str_list(value: Any, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise DescriptorError(f"{where} must be a list of non-empty strings")
    return tuple(value)


def relative_paths(value: Any, where: str) -> tuple[str, ...]:
    paths = str_list(value, where)
    for path in paths:
        pure = PurePosixPath(path)
        if pure.is_absolute() or ".." in pure.parts or not pure.parts or pure.parts[0] == "workspace":
            raise DescriptorError(f"{where} entries must stay below the episode root and outside workspace")
    return paths


def load_descriptor(path: Path) -> AdapterDescriptor:
    """Load and validate one ``adapter.yaml``, importing the classes it names."""
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise DescriptorError(f"cannot load adapter descriptor {path}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise DescriptorError(f"adapter descriptor {path} must be a YAML object")
    name = require_str(data, "name", "descriptor")
    where = f"descriptor {name!r}"
    unknown = sorted(str(key) for key in data if key not in ADAPTER_KEYS)
    if unknown:
        raise DescriptorError(f"{where} has unknown keys: {', '.join(unknown)}")
    config_targets = parse_config_targets(data.get("config"), where)
    try:
        layout = layout_from_evolve(
            data.get("evolve"), {target: config.path for target, config in config_targets.items()}, name
        )
    except TreeLayoutError as exc:
        raise DescriptorError(f"{where}: {exc}") from exc
    sessions = data.get("sessions")
    if not isinstance(sessions, Mapping):
        raise DescriptorError(f"{where} requires 'sessions': the session log's path and format")
    dirs = data.get("dirs", {})
    if not isinstance(dirs, Mapping) or not all(
        variable is None or (isinstance(variable, str) and variable.isidentifier()) for variable in dirs.values()
    ):
        raise DescriptorError(f"{where} 'dirs' must map directories to the variable that moves each, or null")
    state = relative_paths([str(directory) for directory in dirs], f"{where} 'dirs'")
    run = data.get("run")
    if not isinstance(run, Mapping) or not set(run) <= {"episode", "layer"}:
        raise DescriptorError(f"{where} requires 'run': its episode, and optionally its layer")
    episode = run.get("episode")
    if not isinstance(episode, Mapping) or not set(episode) <= {"argv", "env"}:
        raise DescriptorError(f"{where} run 'episode' must be an object of argv and env")
    flags = episode.get("env", {})
    if not isinstance(flags, Mapping) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in flags.items()
    ):
        raise DescriptorError(f"{where} run episode 'env' must map strings to strings")
    env = {str(variable): f"{{root}}/{directory}" for directory, variable in dirs.items() if variable}
    moved = sorted(set(env) & set(flags))
    if moved:
        raise DescriptorError(f"{where} run episode 'env' sets variables that 'dirs' sets: {', '.join(moved)}")
    env.update(flags)
    host_env = data.get("host_env", {})
    if not isinstance(host_env, Mapping) or not all(
        isinstance(key, str) and key.isidentifier() and isinstance(value, str) for key, value in host_env.items()
    ):
        raise DescriptorError(f"{where} 'host_env' must map variable names to default strings")
    overlap = sorted(set(host_env) & set(env))
    if overlap:
        raise DescriptorError(
            f"{where} 'host_env' names variables the episode 'env' already sets: {', '.join(overlap)}"
        )
    self_isolating = data.get("self_isolating", False)
    if not isinstance(self_isolating, bool):
        raise DescriptorError(f"{where} 'self_isolating' must be a boolean")
    is_prompt_task_directory = data.get("is_prompt_task_directory", False)
    if not isinstance(is_prompt_task_directory, bool):
        raise DescriptorError(f"{where} 'is_prompt_task_directory' must be a boolean")
    is_root_bind_mounted = data.get("is_root_bind_mounted", False)
    if not isinstance(is_root_bind_mounted, bool):
        raise DescriptorError(f"{where} 'is_root_bind_mounted' must be a boolean")
    renderer = AdapterRenderer
    if data.get("renderer") is not None:
        renderer = AdapterRenderer.referenced_class(
            parse_plugin_reference(data["renderer"], AdapterRenderer, f"{where} 'renderer'")
        )
    execution_validator = None
    if data.get("execution_validator") is not None:
        setting = f"{where} 'execution_validator'"
        validator_class = ExecutionValidator.referenced_class(
            parse_plugin_reference(data["execution_validator"], ExecutionValidator, setting)
        )
        execution_validator = validator_class()
    model_binding = parse_model_binding(data.get("model"), config_targets, where)
    layer = parse_layer(run.get("layer"), where)
    return AdapterDescriptor(
        name=name,
        binary=require_str(data, "binary", where),
        argv=str_list(episode.get("argv"), f"{where} run episode 'argv'"),
        env=env,
        config_targets=config_targets,
        layout=layout,
        trajectory_format=require_str(sessions, "format", f"{where} sessions"),
        trajectory_path=require_str(sessions, "path", f"{where} sessions"),
        cleanup_whitelist=tuple(f"{directory}/**" for directory in state),
        writable_paths=state,
        renderer=renderer,
        install=parse_install(data.get("install"), where),
        self_isolating=self_isolating,
        is_prompt_task_directory=is_prompt_task_directory,
        host_env=dict(host_env),
        is_root_bind_mounted=is_root_bind_mounted,
        model_binding=model_binding,
        execution_validator=execution_validator,
        layer=layer,
    )


def parse_layer(raw: Any, where: str) -> AdapterLayer | None:
    """``run.layer``, or ``None`` without one; DescriptorError names the first malformed field."""
    if raw is None:
        return None
    if not isinstance(raw, Mapping) or not set(raw) <= {"args", "env", "register", "unregister"}:
        raise DescriptorError(f"{where} run 'layer' must be an object of args, env, register and unregister")
    groups: dict[str, tuple[tuple[str, ...], ...]] = {}
    for name in ("args", "register", "unregister"):
        value = raw.get(name, [])
        if not isinstance(value, list) or not all(isinstance(group, list) for group in value):
            raise DescriptorError(f"{where} run layer {name!r} must be a list of argument groups")
        groups[name] = tuple(tuple(group) for group in value)
    env = raw.get("env", {})
    if not isinstance(env, Mapping) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
        raise DescriptorError(f"{where} run layer 'env' must map strings to strings")
    try:
        return AdapterLayer(
            args=groups["args"], env=dict(env), register=groups["register"], unregister=groups["unregister"]
        )
    except DescriptorError as exc:
        raise DescriptorError(f"{where}: {exc}") from exc


def parse_config_targets(raw: Any, where: str) -> dict[str, ConfigTarget]:
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
            path=require_str(target, "path", f"{where} config target {target_name!r}"),
            defaults=dict(defaults),
            checks=parse_checks(target.get("checks", []), defaults, f"{where} config target {target_name!r} checks"),
        )
    return targets


def parse_checks(raw: Any, defaults: Mapping[str, Any], where: str) -> tuple[ConfigCheck, ...]:
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
        for key in str_list(keys, f"{place} key") if isinstance(keys, list) else [keys]:
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


def parse_model_binding(
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
        raise DescriptorError(f"{where} 'model' must map an api (openai, responses, anthropic) to config templates")
    parsed: dict[str, tuple[Mapping[str, Any], ...]] = {}
    for api, templates in raw.items():
        label = f"{where} model.{api}"
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


def parse_install(raw: Any, where: str) -> InstallSpec | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise DescriptorError(f"{where} 'install' must be an object")
    kind = require_str(raw, "kind", f"{where} install")
    if kind not in INSTALL_KINDS:
        known = ", ".join(INSTALL_KINDS)
        raise DescriptorError(f"{where} install kind {kind!r} is not supported; known kinds: {known}")
    package = require_str(raw, "package", f"{where} install")
    if not _INSTALL_PACKAGE_PATTERN.fullmatch(package):
        raise DescriptorError(f"{where} install 'package' {package!r} must match {_INSTALL_PACKAGE_PATTERN.pattern}")
    version = require_str(raw, "version", f"{where} install")
    if not _INSTALL_VERSION_PATTERN.fullmatch(version):
        raise DescriptorError(f"{where} install 'version' {version!r} must match {_INSTALL_VERSION_PATTERN.pattern}")
    repository = ref = ""
    if kind == "git":
        repository = require_str(raw, "repository", f"{where} install")
        if not _INSTALL_REPOSITORY_PATTERN.fullmatch(repository):
            raise DescriptorError(
                f"{where} install 'repository' {repository!r} must match {_INSTALL_REPOSITORY_PATTERN.pattern}"
            )
        ref = require_str(raw, "ref", f"{where} install")
        if not _INSTALL_VERSION_PATTERN.fullmatch(ref):
            raise DescriptorError(f"{where} install 'ref' {ref!r} must match {_INSTALL_VERSION_PATTERN.pattern}")
    return InstallSpec(
        kind=kind,
        package=package,
        version=version,
        binary_path=require_str(raw, "binary_path", f"{where} install"),
        repository=repository,
        ref=ref,
    )


def parse_plugin_reference(value: Any, base: type[Plugin], where: str) -> PluginReference:
    """The plugin reference that ``value``, a ``package.module:ClassName`` string, makes; DescriptorError unless
    its class inherits ``base``."""
    if not isinstance(value, str):
        raise DescriptorError(f"{where} must be 'package.module:ClassName'")
    try:
        return base.reference(value, setting=where)
    except PluginConfigError as exc:
        raise DescriptorError(str(exc)) from exc


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
