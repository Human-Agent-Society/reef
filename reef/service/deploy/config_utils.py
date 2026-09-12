"""Config loading and interpolation shared by app assembly and orchestrator.

``reef serve`` configs are YAML with two interpolation passes: ``${VAR}``
against the process environment at load time (with ``REEF_PYTHON`` defaulting
to the current interpreter and ``${VAR:?}`` requiring a non-empty value), and
``${dotted.path}`` against the config itself when a service command or setting
is materialized.
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from reef.core.config import config_value, interpolate_config
from reef.core.errors import DeployConfigError

try:
    import yaml
except ImportError as exc:  # pragma: no cover - environment-dependent
    raise DeployConfigError("PyYAML is required: pip install pyyaml") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[3]
_ENV_VAR_RE = re.compile(r"\$\{(\w+)(:\?)?\}")


def _deep_interp_env(obj: Any, environ: Mapping[str, str], missing: dict[str, list[str]], location: str = "") -> Any:
    if isinstance(obj, dict):
        return {
            key: _deep_interp_env(item, environ, missing, f"{location}.{key}" if location else str(key))
            for key, item in obj.items()
        }
    if isinstance(obj, list):
        return [_deep_interp_env(item, environ, missing, f"{location}[{index}]") for index, item in enumerate(obj)]
    if isinstance(obj, str):

        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            value = environ.get(name, "")
            if match.group(2) and not value.strip():
                locations = missing.setdefault(name, [])
                if location not in locations:
                    locations.append(location)
            return value

        return _ENV_VAR_RE.sub(replace, obj)
    return obj


def interpolate_environment(config: Mapping[str, Any], config_path: str | Path) -> dict[str, Any]:
    """Expand environment references, reporting every missing required variable."""
    environ = dict(os.environ)
    environ.setdefault("REEF_PYTHON", sys.executable)
    missing: dict[str, list[str]] = {}
    resolved = _deep_interp_env(dict(config), environ, missing)
    if missing:
        details = "\n".join(f"  {name} ({', '.join(locations)})" for name, locations in missing.items())
        raise DeployConfigError(
            f"config {config_path}: missing or empty required environment variables:\n{details}\n"
            "Set these variables before running reef serve, or override the corresponding config fields."
        )
    return resolved


def interpolate_config_values(config: Mapping[str, Any], value: Any) -> Any:
    """Expand references in structured values without stringifying containers."""
    if isinstance(value, str):
        return interpolate_config(config, value)
    if isinstance(value, Mapping):
        return {key: interpolate_config_values(config, item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [interpolate_config_values(config, item) for item in value]
    return value


def load_config(config_path: str | Path, *, interpolate_env: bool = True) -> dict[str, Any]:
    """Read a config relative to the working directory; optionally defer interpolation."""
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise DeployConfigError(f"config not found: {path}\n  pass a deployment stack with: reef serve -c <path>")
    try:
        with open(path) as handle:
            config = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise DeployConfigError(f"config {path} is not valid YAML: {exc}") from exc
    except OSError as exc:
        raise DeployConfigError(f"cannot read config {path}: {exc.strerror}") from exc
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise DeployConfigError(f"config {path} must be a YAML object at the root, not {type(config).__name__}")
    return interpolate_environment(config, path) if interpolate_env else config


def recipe_source_root(config: Mapping[str, Any], config_path: str | Path) -> Path | None:
    """The directory a dotted ``reef.recipe`` class imports from, found beside the config.

    A dotted reference such as ``recipes.sao.recipe:SAORecipe`` names a
    package that is not pip-installed: the ``recipes/`` cookbook ships with
    the source checkout, next to the ``serve.yaml`` that selects it. Walk up
    from the config file to the nearest directory holding that top-level
    package and return it, so ``reef serve`` can put it on the import path
    of every service instead of each launcher exporting ``PYTHONPATH`` by
    hand. ``None`` when the recipe is a bare name (core or preset) or when no
    ancestor of the config holds the package; the import then fails in the
    service exactly as it does today.
    """
    reference = config_value(config, "reef", "recipe")
    if not isinstance(reference, str) or ":" not in reference:
        return None
    top_level = reference.partition(":")[0].partition(".")[0]
    if not top_level:
        return None
    config_dir = Path(config_path).resolve().parent
    for ancestor in (config_dir, *config_dir.parents):
        if (ancestor / top_level / "__init__.py").is_file():
            return ancestor
    return None


__all__ = [
    "PROJECT_ROOT",
    "DeployConfigError",
    "config_value",
    "interpolate_config",
    "load_config",
]
