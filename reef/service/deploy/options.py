"""Transport native backend options without duplicating backend argument schemas."""

from __future__ import annotations

import re

from reef.core.config import ConfigArgument
from reef.runtime.executor.arguments import native_arguments, normalize_native_options
from reef.service.deploy.config import DeployConfigError

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


__all__ = ["native_arguments", "native_override", "normalize_native_options", "object_override_path"]
