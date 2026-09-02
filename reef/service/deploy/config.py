"""Config loading and interpolation shared by app assembly and orchestrator.

``reef serve`` configs are YAML with two interpolation passes: ``${VAR}``
against the process environment at load time (with ``REEF_PYTHON`` defaulting
to the current interpreter), and ``${dotted.path}`` against the config itself
when a service command or setting is materialized.
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from reef.core.errors import ReefError


class DeployConfigError(ReefError):
    """A ``reef serve`` deployment config cannot be loaded or is invalid."""


try:
    import yaml
except ImportError as exc:  # pragma: no cover - environment-dependent
    raise DeployConfigError("PyYAML is required: pip install pyyaml") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[3]
_ENV_VAR_RE = re.compile(r"\$\{(\w+)\}")
_CFG_VAR_RE = re.compile(r"\$\{([\w.]+)\}")


def _interp_env(value: str, environ: Mapping[str, str]) -> str:
    return _ENV_VAR_RE.sub(lambda m: environ.get(m.group(1), ""), value)


def _deep_interp_env(obj: Any, environ: Mapping[str, str]) -> Any:
    if isinstance(obj, dict):
        return {key: _deep_interp_env(item, environ) for key, item in obj.items()}
    if isinstance(obj, list):
        return [_deep_interp_env(item, environ) for item in obj]
    if isinstance(obj, str):
        return _interp_env(obj, environ)
    return obj


def config_value(
    config: Mapping[str, Any],
    *path: str,
    default: Any = None,
    expand: bool = True,
) -> Any:
    """Read a dotted-path config value as a stripped string (bools/None pass through)."""
    node: Any = config
    for key in path:
        if not isinstance(node, dict):
            node = None
            break
        node = node.get(key)
    if node is None or (isinstance(node, str) and not node.strip()):
        node = default
    if node is None or isinstance(node, bool):
        return node
    value = str(node).strip()
    return os.path.expanduser(value) if expand else value


def interpolate_config(config: Mapping[str, Any], value: str) -> str:
    """Substitute ``${dotted.path}`` references against the config itself."""

    def repl(match: re.Match[str]) -> str:
        resolved = config_value(config, *match.group(1).split("."), default=None)
        return str(resolved) if resolved is not None else match.group(0)

    return _CFG_VAR_RE.sub(repl, value)


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise DeployConfigError(f"config not found: {path}\n  pass a deployment stack with: reef serve -c <path>")
    try:
        with open(path) as handle:
            config = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise DeployConfigError(f"config {path} is not valid YAML: {exc}") from exc
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise DeployConfigError(f"config {path} must be a YAML object at the root, not {type(config).__name__}")
    environ = dict(os.environ)
    environ.setdefault("REEF_PYTHON", sys.executable)
    return _deep_interp_env(config, environ)


def validate_services(config: Mapping[str, Any], config_path: str | Path) -> list[dict[str, Any]]:
    """Services with valid commands and unique names, checked before any process starts."""
    services = config.get("services")
    if not isinstance(services, list) or not services:
        raise DeployConfigError(f"config {config_path} must declare a non-empty 'services' list")
    names: list[str] = []
    for index, service in enumerate(services):
        if not isinstance(service, dict):
            raise DeployConfigError(
                f"config {config_path}: services[{index}] must be an object, not {type(service).__name__}"
            )
        name = service.get("name")
        if not isinstance(name, str) or not name.strip():
            raise DeployConfigError(f"config {config_path}: services[{index}] must have a non-empty 'name'")
        command = service.get("command")
        valid_string = isinstance(command, str) and bool(command.strip())
        valid_list = (
            isinstance(command, list)
            and bool(command)
            and all(isinstance(argument, str) for argument in command)
            and bool(command[0].strip())
        )
        if not valid_string and not valid_list:
            raise DeployConfigError(
                f"config {config_path}: services[{index}] must have a non-empty 'command' string or list of strings"
            )
        names.append(name)
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise DeployConfigError(
            f"config {config_path}: service names must be unique; duplicated: {', '.join(duplicates)}"
        )
    return services


def is_local_path(value: str) -> bool:
    """True if the value is an existing local filesystem path, False if it's an HF repo ID."""
    return Path(os.path.expanduser(value)).exists()


def resolve_hf_snapshot(repo_id: str) -> str:
    """Download an HF repo ID to a local snapshot and return the local path.

    If the value is already a local path, return it unchanged.
    """
    value = os.path.expanduser(repo_id)
    if Path(value).exists():
        return value
    if "/" not in value or value.startswith(("/", "~")):
        raise DeployConfigError(
            f"model path is not a local directory and not a valid HF repo ID: {repo_id}\n"
            f"  set reef.model_path to a local path or an HF repo ID like 'Qwen/Qwen2.5-1.5B-Instruct'"
        )
    try:
        from reef.artifact.sources import HuggingFaceSource, download_huggingface_snapshot, parse_artifact_source

        source = parse_artifact_source(value)
        if not isinstance(source, HuggingFaceSource):
            raise DeployConfigError(f"model path is not a valid HF repo ID: {repo_id}")
        snapshot = download_huggingface_snapshot(source)
        return str(snapshot.local_path)
    except Exception as exc:
        if isinstance(exc, DeployConfigError):
            raise
        raise DeployConfigError(f"failed to download HF model {value}: {exc}") from exc


def resolve_model_paths(config: dict[str, Any]) -> bool:
    """Download HF repo IDs for model paths and replace them with local snapshot paths.

    Handles ``reef.model_path`` and ``training.megatron_checkpoint_path``.
    Returns True if any path was changed (i.e. the config dict no longer
    matches what's on disk).
    """
    changed = False
    reef_section = config.get("reef")
    if isinstance(reef_section, dict) and isinstance(reef_section.get("model_path"), str):
        resolved = resolve_hf_snapshot(reef_section["model_path"])
        if resolved != reef_section["model_path"]:
            _log(f"downloaded HF model {reef_section['model_path']} -> {resolved}")
            reef_section["model_path"] = resolved
            changed = True
    training_section = config.get("training")
    if isinstance(training_section, dict) and isinstance(training_section.get("megatron_checkpoint_path"), str):
        resolved = resolve_hf_snapshot(training_section["megatron_checkpoint_path"])
        if resolved != training_section["megatron_checkpoint_path"]:
            _log(f"downloaded HF model {training_section['megatron_checkpoint_path']} -> {resolved}")
            training_section["megatron_checkpoint_path"] = resolved
            changed = True
    return changed


def _log(msg: str) -> None:
    import sys

    print(f"[reef] {msg}", file=sys.stderr)


__all__ = [
    "PROJECT_ROOT",
    "DeployConfigError",
    "config_value",
    "interpolate_config",
    "is_local_path",
    "load_config",
    "resolve_hf_snapshot",
    "resolve_model_paths",
    "validate_services",
]
