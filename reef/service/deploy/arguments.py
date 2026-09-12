"""Typed argparse definitions shared by deployment YAML and command-line input.

Only declared settings pass through this adapter. Custom service stacks and
recipe-owned mappings keep their existing parsers.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import types
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NoReturn, Union, get_args, get_origin, get_type_hints

import yaml


def config_metadata(help: str, *, path: tuple[str, ...] = ()) -> dict[str, Any]:
    """Describe a public setting; an omitted path means ``reef.<field>``."""
    return {"config_help": help, "config_path": path}


def config_option(default: Any = dataclasses.MISSING, *, help: str) -> Any:
    """Declare a setting's default and CLI help beside its type."""
    return dataclasses.field(default=default, metadata=config_metadata(help))


class ConfigArgumentParser(argparse.ArgumentParser):
    """Library parsing reports errors without printing input values or exiting."""

    def error(self, message: str) -> NoReturn:
        raise ValueError(message)


@dataclass(frozen=True)
class ConfigArgument:
    """One setting's YAML path, CLI aliases, default and value parser."""

    name: str
    path: tuple[str, ...]
    kind: str
    nullable: bool
    default: Any
    help: str

    @property
    def destination(self) -> str:
        """Keep the config's recipe separate from the launcher's profile."""
        return "config_recipe" if self.name == "recipe" else self.name

    @property
    def flags(self) -> tuple[str, ...]:
        dotted = ".".join(self.path)
        names = [dotted.replace("_", "-"), dotted]
        # The launcher owns --recipe; the setting remains --reef.recipe.
        if self.path[0] == "reef" and self.name != "recipe":
            names = [self.name.replace("_", "-"), self.name, *names]
        return tuple(dict.fromkeys(f"--{name}" for name in names))

    @property
    def negative_flags(self) -> tuple[str, ...]:
        return tuple(f"--no-{flag[2:]}" for flag in self.flags) if self.kind == "bool" else ()

    def parse(self, raw: str) -> Any:
        """Convert either source with the same rules; never echo credentials."""
        try:
            if self.kind == "str":
                return raw
            if self.nullable and raw == "null":
                return None
            if self.kind == "int":
                return int(raw)
            if self.kind == "float":
                value = float(raw)
                if not math.isfinite(value):
                    raise ValueError("non-finite number")
                return value
            if self.kind == "bool":
                if raw.lower() in {"true", "yes", "on", "1"}:
                    return True
                if raw.lower() in {"false", "no", "off", "0"}:
                    return False
                raise ValueError("invalid boolean")
            value = yaml.safe_load(raw)
            if self.kind == "object" and isinstance(value, dict):
                return value
            if self.kind == "strings" and isinstance(value, list) and all(isinstance(item, str) for item in value):
                return tuple(value)
        except (ValueError, TypeError, yaml.YAMLError):
            pass
        expected = {"object": "an object", "strings": "a list of strings"}.get(self.kind, f"a valid {self.kind}")
        raise argparse.ArgumentTypeError(f"{'.'.join(self.path)} must be {expected}")

    def add_to(self, parser: argparse.ArgumentParser) -> None:
        options: dict[str, Any] = {
            "dest": self.destination,
            "type": self.parse,
            "default": copy.deepcopy(self.default),
            "help": self.help,
        }
        if self.kind == "bool":
            options.update(nargs="?", const="true")
        parser.add_argument(*self.flags, **options)
        if self.negative_flags:
            parser.add_argument(
                *self.negative_flags,
                dest=self.destination,
                action="store_const",
                const=False,
                default=argparse.SUPPRESS,
                help=f"Disable {'.'.join(self.path)}.",
            )

    def encode(self, value: Any) -> str:
        """Convert a YAML value to an argv value without losing empty containers."""
        if isinstance(value, str):
            return value
        if self.kind in {"strings", "object"}:
            try:
                return json.dumps(value)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{'.'.join(self.path)} must contain JSON-compatible values") from exc
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)


def config_arguments(settings_type: type) -> tuple[ConfigArgument, ...]:
    """Read the declared fields of a settings dataclass, not runtime objects."""
    annotations = get_type_hints(settings_type)
    arguments = []
    for field in dataclasses.fields(settings_type):
        if "config_help" not in field.metadata:
            continue
        annotation = annotations[field.name]
        nullable = False
        if get_origin(annotation) in (Union, types.UnionType):
            members = get_args(annotation)
            nullable = type(None) in members
            annotation = next(member for member in members if member is not type(None))
        kinds = {str: "str", int: "int", float: "float", bool: "bool", tuple: "strings", Mapping: "object"}
        kind = kinds.get(get_origin(annotation) or annotation)
        if kind is None:
            raise TypeError(f"unsupported config field type: {field.name}")
        default = field.default
        if default is dataclasses.MISSING:
            default = field.default_factory() if field.default_factory is not dataclasses.MISSING else None
        arguments.append(
            ConfigArgument(
                field.name,
                field.metadata["config_path"] or ("reef", field.name),
                kind,
                nullable,
                default,
                field.metadata["config_help"],
            )
        )
    return tuple(arguments)
