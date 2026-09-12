"""Build CLI help and merge explicit dotted overrides into deployment configuration.

Type conversion is shared with YAML through ``reef.core.config``. This module
owns command-line syntax and precedence, not component defaults or process startup.
"""

from __future__ import annotations

import argparse
import copy
import re
from collections.abc import Mapping
from typing import Any

import yaml

from reef.core.config import ConfigArgument
from reef.core.errors import DeployConfigError
from reef.runtime.executor.arguments import normalize_native_options
from reef.service.deploy.component_config import component_config_arguments, deployment_config_arguments
from reef.service.deploy.service_config import service_config_arguments, service_override

_DESCRIPTION = """reef serve — connect an external provider or start a configured stack.

With no selected config, --inference.upstream-url and --inference.upstream-model start Reef's
record-only recipe on 127.0.0.1:8900. No YAML is required.
Alternatively, --inference.model-path starts managed SGLang inference and Reef;
--inference.tensor-parallel-size selects the visible GPU count (default: 1).
Config files are selected explicitly with -c; REEF_CONFIG and ./reef.yaml
are not discovered by the launcher.

``reef serve -c <stack>.yaml`` assembles the configured components and starts
their processes in dependency order. Only unversioned legacy files declare
a ``services`` list. Each service's ``ready`` probe must pass before the next
starts. After all services are up, Reef blocks until SIGTERM/SIGINT; a
watchdog thread detects unexpected exits and tears the stack down.

The Reef HTTP child receives the effective configuration from the launcher.

Config overrides:
  Public settings below share type conversion with YAML. Explicit CLI
  values override YAML; omitted settings use the dataclass defaults.
  Use the full public namespace; legacy aliases remain accepted.
  Lists and objects take one quoted JSON/YAML value, including [] or {}.
  Selected recipe/runtime fields share these rules; use -c <file> --help
  to inspect their definitions. Versioned files reject unknown public
  fields; legacy custom-stack keys retain their compatibility parsing.

  Examples:
    reef serve --inference.model-path Qwen/Qwen2.5-1.5B-Instruct
    reef serve --inference.upstream-url http://localhost:8000 --inference.upstream-model my-model
    reef serve -c stack.yaml --inference.model-path Qwen/Qwen2.5-1.5B-Instruct
    reef serve -c path/to/local-sglang.yaml --reef.port 9000
    reef serve -c stack.yaml --training.options.save /tmp/ckpt
"""


def build_parser(*, service_arguments: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reef serve",
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="Optional config file path, relative to the working directory; no file is loaded unless selected.",
    )
    if service_arguments:
        for argument in service_config_arguments():
            argument.add_to(parser)
    return parser


_OPTION_PATHS = {
    "inference.options.": ("reef", "inference_options"),
    "training.options.": ("reef", "training_backend_options"),
}


def native_override(key: str) -> tuple[tuple[str, ...], str] | None:
    """Resolve a single native flag under its public options namespace."""
    for prefix, path in _OPTION_PATHS.items():
        if key.startswith(prefix):
            name = key[len(prefix) :].replace("_", "-")
            if not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9-]*", name):
                raise DeployConfigError(f"invalid backend option name: {key}")
            return path, name
    return None


def object_override_path(key: str, arguments: tuple[ConfigArgument, ...]) -> tuple[str, ...] | None:
    """Locate a leaf in a declared opaque object; its component owns value validation."""
    prefixes: list[tuple[str, tuple[str, ...]]] = []
    for argument in arguments:
        if argument.kind == "object":
            public = ".".join(argument.public_path or argument.path)
            prefixes.extend((name + ".", argument.path) for name in (public, public.replace("_", "-")))
    for prefix, path in sorted(prefixes, key=lambda item: len(item[0]), reverse=True):
        if key.startswith(prefix):
            suffix = tuple(key[len(prefix) :].split("."))
            if not all(suffix):
                raise DeployConfigError("object override paths must have non-empty fields")
            return (*path, *suffix)
    return None


class InvalidOverrideError(ValueError):
    """A leftover ``reef serve`` argument is not a valid ``--key`` override."""


class _ImplicitOverride(str):
    """A valueless flag, retained until the selected component schema is known."""


def _parse_overrides(extras: list[str]) -> dict[str, str]:
    """Parse leftover ``--key value`` / ``--key=value`` pairs from ``parse_known_args``.

    Anything that is not a well-formed ``--key`` override — an orphan
    positional, an unknown short option, or a ``--`` with no name — is
    rejected instead of silently discarded, so a typo fails fast at the CLI
    rather than surfacing as an unrelated missing-config error downstream.
    """
    overrides: dict[str, str] = {}
    i = 0
    while i < len(extras):
        token = extras[i]
        if not token.startswith("--"):
            raise InvalidOverrideError(f"unrecognized argument: {token!r}")
        key = token[2:]
        has_value = True
        if "=" in key:
            key, value = key.split("=", 1)
            i += 1
        elif i + 1 < len(extras) and not extras[i + 1].startswith("--"):
            value = extras[i + 1]
            i += 2
        else:
            value = _ImplicitOverride("true")
            has_value = False
            i += 1
        if not key:
            raise InvalidOverrideError(f"override is missing a name: {token!r}")
        declared = service_override(key, value)
        if declared is not None:
            argument, _ = declared
            if not has_value and argument.kind != "bool":
                raise InvalidOverrideError(f"argument --{key} requires a value")
            if has_value and f"--{key}" in argument.negative_flags:
                raise InvalidOverrideError(f"argument --{key} does not take a value")
        # Preserve the last occurrence's position as well as its value, so
        # a repeated spelling can still override an intervening alias.
        overrides.pop(key, None)
        overrides[key] = value
    return overrides


def _coerce_value(raw: str) -> Any:
    """Parse a CLI string into a YAML-compatible Python value (int, bool, str, ...)."""
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def _apply_overrides(
    config: dict[str, Any], overrides: dict[str, str], *, arguments: tuple[ConfigArgument, ...] = ()
) -> dict[str, Any]:
    """Merge CLI overrides into a copy of the config dict.

    Bare keys (no dot) target the ``reef`` section; dotted keys traverse
    nested sections (e.g. ``training.checkpoint_dir``).
    """
    config = copy.deepcopy(config)
    for key, raw_value in overrides.items():
        native = native_override(key)
        if native is not None:
            path, option = native
            options_node = config
            for part in path[:-1]:
                options_node = options_node.setdefault(part, {})
            current = options_node.get(path[-1], {})
            options = normalize_native_options(_coerce_value(current) if isinstance(current, str) else current)
            options[option] = (
                _coerce_value(raw_value)
                if raw_value in {"true", "false", "null"} or raw_value.startswith(("[", "{"))
                else str(raw_value)
            )
            options_node[path[-1]] = options
            continue
        declared = service_override(key, raw_value)
        if declared is None:
            for candidate in arguments:
                if f"--{key}" in (*candidate.flags, *candidate.negative_flags):
                    if isinstance(raw_value, _ImplicitOverride) and candidate.kind != "bool":
                        raise InvalidOverrideError(f"argument --{key} requires a value")
                    if f"--{key}" in candidate.negative_flags:
                        if not isinstance(raw_value, _ImplicitOverride):
                            raise InvalidOverrideError(f"argument --{key} does not take a value")
                        raw_value = "false"
                    declared = candidate, raw_value
                    break
        if declared is not None:
            argument, raw_value = declared
            key = ".".join(argument.path)
        if declared is None:
            object_path = object_override_path(key, (*service_config_arguments(), *arguments))
            if object_path is not None:
                key = ".".join(object_path)
        if declared is None:
            for public, internal in (("recipe.runtime.", "reef.runtime."), ("recipe.config.", "reef.data.")):
                if key.startswith(public):
                    key = internal + key[len(public) :].replace("-", "_")
                    break
        if "." not in key and declared is None:
            key = f"reef.{key}"
        parts = key.split(".")
        node: dict[str, Any] = config
        for index, part in enumerate(parts[:-1]):
            if part not in node:
                node[part] = {}
            existing = node[part]
            if parts[0] == "execution" and index == 1 and isinstance(existing, str):
                profiles = config.get("executors", {})
                if not isinstance(profiles, Mapping):
                    raise InvalidOverrideError("executors must be an object")
                existing = copy.deepcopy(profiles.get(existing, {"backend": existing}))
                node[part] = existing
            if not isinstance(existing, dict):
                prefix = ".".join(parts[: index + 1])
                raise InvalidOverrideError(f"override path {prefix!r} is not a section")
            node = existing
        # Public fields are converted by argparse, just like YAML values.
        # Keep generic recipe/custom-stack overrides on their legacy path.
        node[parts[-1]] = str(raw_value) if declared is not None else _coerce_value(raw_value)
    return config


def build_serve_parser(
    *, service_arguments: bool = True, config: Mapping[str, Any] | None = None
) -> argparse.ArgumentParser:
    """``reef serve``'s own arguments: the service child's parser plus the profile and model flags.

    Only the launcher takes them; ``python -m reef.service`` still refuses ``--recipe``."""
    parser = build_parser(service_arguments=service_arguments)
    if config is not None:
        for argument in component_config_arguments(config):
            argument.add_to(parser)
    elif service_arguments:
        for argument in deployment_config_arguments():
            argument.add_to(parser)
    parser.add_argument(
        "--recipe",
        default=None,
        metavar="NAME",
        help="Start a built in recipe's profile instead of a config file (harness-evolve).",
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="[PROVIDER/]MODEL",
        help="The upstream model; a known provider prefix (ollama, openai) fills the URL and the key.",
    )
    return parser
