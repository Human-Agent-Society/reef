"""Shared field declarations and parsing for service, recipe and runtime settings.

Components own their dataclasses and validation. This module reads those
definitions without importing components or constructing their runtimes.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import sys
import types
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NoReturn, Union, get_args, get_origin, get_type_hints

import yaml


def config_metadata(
    help: str = "", *, path: tuple[str, ...] = (), env: str | None = None, allow_nonfinite: bool = False
) -> dict[str, Any]:
    """Describe a public setting; an omitted path means ``reef.<field>``."""
    return {"config_help": help, "config_path": path, "config_env": env, "config_allow_nonfinite": allow_nonfinite}


def config_option(
    default: Any = dataclasses.MISSING,
    *,
    default_factory: Any = dataclasses.MISSING,
    help: str = "",
    env: str | None = None,
    allow_nonfinite: bool = False,
) -> Any:
    """Declare a setting's default and CLI help beside its type."""
    return dataclasses.field(
        default=default,
        default_factory=default_factory,
        metadata=config_metadata(help, env=env, allow_nonfinite=allow_nonfinite),
    )


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
    env: str | None = None
    allow_nonfinite: bool = False
    required: bool = False

    @property
    def destination(self) -> str:
        """Keep the config's recipe separate from the launcher's profile."""
        if len(self.path) > 2:
            return ".".join(self.path)
        return "config_recipe" if self.name == "recipe" else self.name

    @property
    def flags(self) -> tuple[str, ...]:
        dotted = ".".join(self.path)
        names = [dotted.replace("_", "-"), dotted]
        # The launcher owns --recipe; the setting remains --reef.recipe.
        if len(self.path) == 2 and self.path[0] == "reef" and self.name != "recipe":
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
                if not self.allow_nonfinite and not math.isfinite(value):
                    raise ValueError("non-finite number")
                return value
            if self.kind == "bool":
                if raw.strip().lower() in {"true", "yes", "on", "1"}:
                    return True
                if raw.strip().lower() in {"false", "no", "off", "0"}:
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
        if value is None:
            if self.nullable:
                return "null"
            raise ValueError(f"{'.'.join(self.path)} does not accept null")
        if isinstance(value, str):
            return value
        if self.kind == "str":
            raise ValueError(f"{'.'.join(self.path)} must be a string")
        if self.kind in {"strings", "object"}:
            try:
                return json.dumps(value)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{'.'.join(self.path)} must contain JSON-compatible values") from exc
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)


def config_arguments(settings_type: type, *, prefix: tuple[str, ...] = ("reef",)) -> tuple[ConfigArgument, ...]:
    """Read the declared fields of a settings dataclass, not runtime objects."""
    arguments = []
    for field in dataclasses.fields(settings_type):
        if "config_help" not in field.metadata:
            continue
        annotation = field.type
        if isinstance(annotation, str):
            # Resolve only this declaration: unrelated runtime annotations and
            # Python 3.10's KW_ONLY sentinel are not part of the config schema.
            owner = next(
                base for base in settings_type.__mro__ if field.name in base.__dict__.get("__annotations__", {})
            )
            declaration = types.SimpleNamespace(__annotations__={field.name: annotation})
            annotation = get_type_hints(declaration, globalns=vars(sys.modules[owner.__module__]))[field.name]
        nullable = False
        if get_origin(annotation) in (Union, types.UnionType):
            members = get_args(annotation)
            nullable = type(None) in members
            concrete = [member for member in members if member is not type(None)]
            if not nullable or len(concrete) != 1:
                raise TypeError(f"unsupported config union: {field.name}")
            annotation = concrete[0]
        kinds = {
            str: "str",
            int: "int",
            float: "float",
            bool: "bool",
            tuple: "strings",
            Mapping: "object",
            dict: "object",
        }
        kind = kinds.get(get_origin(annotation) or annotation)
        if kind == "strings" and get_args(annotation) != (str, Ellipsis):
            kind = None
        if kind == "object" and get_args(annotation) not in ((), (str, Any)):
            kind = None
        if kind is None:
            raise TypeError(f"unsupported config field type: {field.name}")
        default = field.default
        if default is dataclasses.MISSING:
            default = field.default_factory() if field.default_factory is not dataclasses.MISSING else None
        arguments.append(
            ConfigArgument(
                field.name,
                field.metadata["config_path"] or (*prefix, field.name),
                kind,
                nullable,
                default,
                field.metadata["config_help"],
                field.metadata.get("config_env"),
                field.metadata.get("config_allow_nonfinite", False),
                field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING,
            )
        )
    return tuple(arguments)


def parse_config_values(
    arguments: tuple[ConfigArgument, ...],
    values: Mapping[str, Any],
    *,
    environ: Mapping[str, str] | None = None,
    include_defaults: bool = True,
) -> dict[str, Any]:
    """Parse declared values with config > declared environment > defaults.

    Callers select the component and extract its mapping. Unknown fields are
    rejected here; opaque plugin payloads must be declared as object fields.
    """
    if not isinstance(values, Mapping):
        raise ValueError("component config must be an object")
    known = {argument.name for argument in arguments}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown config fields: {', '.join(sorted(map(str, unknown)))}")
    parser = ConfigArgumentParser(add_help=False, allow_abbrev=False)
    argv = []
    supplied = set()
    for argument in arguments:
        argument.add_to(parser)
        value = values.get(argument.name, dataclasses.MISSING)
        if value is dataclasses.MISSING and environ is not None and argument.env is not None:
            value = environ.get(argument.env, dataclasses.MISSING)
        if value is dataclasses.MISSING:
            if include_defaults and argument.required:
                raise ValueError(f"{'.'.join(argument.path)} is required")
            continue
        supplied.add(argument.name)
        if value is None and argument.nullable:
            parser.set_defaults(**{argument.destination: None})
        elif argument.kind == "object" and isinstance(value, Mapping):
            # Already-typed opaque options may include backend-owned Python
            # objects. Their contents are validated by that adapter, not JSON.
            parser.set_defaults(**{argument.destination: dict(value)})
        else:
            argv.append(f"{argument.flags[0]}={argument.encode(value)}")
    parsed = vars(parser.parse_args(argv))
    return {
        argument.name: parsed[argument.destination]
        for argument in arguments
        if include_defaults or argument.name in supplied
    }
