"""The base contract and wire codec for typed reports.

A report type declares its fields. Annotate them and the base class derives
everything else — parsing, per-field validation with messages that name the
broken field, and the wire body. Cross-field rules go in an optional
:meth:`ReportBase.validate` hook, which runs on both parse and construction.

The declaration is shared by every party that touches the wire:

- Producers build the exact report body with :meth:`ReportBase.to_dict`.
- The service parses a recipe's declared type before durable storage.
- Processors receive the parsed instance through their report context and
  rehydrate the same type during durable replay.

The report type is a floor, not a ceiling: parsing validates declared fields
and ignores everything else, so producers may attach extra ``feedback`` and
``metadata`` keys freely. Bundled concrete contracts live in this package;
out-of-tree methods declare theirs in their own package.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import types
import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from reef.core.errors import ReefError

__all__ = [
    "ReportBase",
    "ReportValidationError",
]

_MISSING = object()


class ReportValidationError(ReefError):
    """A report does not satisfy its recipe's declared schema.

    The message names the offending field and rule, because it travels to
    two audiences far apart in time: the HTTP 400 body at ingress, and the
    named terminal-judgment reason counted in processor metrics.
    """


@dataclass(frozen=True)
class ReportBase:
    """One typed report: wire dict in, typed fields out.

    Subclasses declare fields and, when the contract has cross-field rules,
    override :meth:`validate`. ``score`` occupies the payload's top-level
    score channel; every other field lives under ``metadata``. Field types are
    enforced from the annotations: ``float`` (finite), ``int`` (integral),
    ``str``, ``bool``, ``Mapping[str, str]``, each optionally ``| None``.
    Parsing is pure, cheap, and payload-only; deployment-relative checks (does
    the announced value match this scenario's configuration?) belong to the
    processor.
    """

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Cross-field rules; raise :class:`ReportValidationError`. Runs on parse and construction."""

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ReportBase:
        """Parse a normalized report payload; raise :class:`ReportValidationError` naming any broken field."""
        kwargs: dict[str, Any] = {}
        for spec in cls._wire_specs():
            value = _read(payload, spec.path)
            if value is _MISSING:
                if spec.required:
                    raise ReportValidationError(f"{spec.path} is required")
                continue
            kwargs[spec.name] = _checked(value, spec.annotation, spec.nullable, spec.path)
        return cls(**kwargs)

    def report_payload(self) -> dict[str, Any]:
        """The report-type-owned wire fields; defaults are omitted."""
        body: dict[str, Any] = {}
        for spec in type(self)._wire_specs():
            value = getattr(self, spec.name)
            if not spec.required and value == spec.default:
                continue
            _write(body, spec.path, value)
        return body

    def to_dict(
        self,
        *,
        references: Sequence[str] = (),
        agent_record_id: str | None = None,
    ) -> dict[str, Any]:
        """The complete ``POST /reef/report`` body for this report.

        Extra ``feedback``/``metadata`` keys are the producer's to add on
        the returned dict — the report type only owns its declared fields.
        """
        body = self.report_payload()
        if references:
            body["references"] = list(references)
        if agent_record_id is not None:
            body["agent_record_id"] = agent_record_id
        return body

    @classmethod
    @functools.cache
    def _wire_specs(cls) -> tuple[_WireSpec, ...]:
        hints = typing.get_type_hints(cls)
        specs = []
        for field in dataclasses.fields(cls):
            annotation, nullable = _report_annotation(hints[field.name], f"{cls.__name__}.{field.name}")
            specs.append(
                _WireSpec(
                    name=field.name,
                    path=_default_path(field.name),
                    annotation=annotation,
                    nullable=nullable,
                    required=field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING,
                    default=None if field.default is dataclasses.MISSING else field.default,
                )
            )
        return tuple(specs)


@dataclass(frozen=True)
class _WireSpec:
    name: str
    path: str
    annotation: type
    nullable: bool
    required: bool
    default: Any


def _default_path(name: str) -> str:
    """``score`` is top-level; every method field lives under ``metadata``."""
    return "score" if name == "score" else f"metadata.{name}"


def _read(payload: Mapping[str, Any], path: str) -> Any:
    node: Any = payload
    for key in path.split("."):
        if not isinstance(node, Mapping) or key not in node:
            return _MISSING
        node = node[key]
    return node


def _write(body: dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    node = body
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def _report_annotation(annotation: Any, label: str) -> tuple[type, bool]:
    """Normalize one supported report annotation or reject the declaration."""
    nullable = False
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        arms = typing.get_args(annotation)
        concrete = [arm for arm in arms if arm is not type(None)]
        nullable = len(concrete) < len(arms)
        if not nullable or len(concrete) != 1:
            raise TypeError(
                f"{label} has unsupported report annotation {annotation!r}; "
                "use one supported type, optionally followed by | None"
            )
        annotation = concrete[0]
    if annotation in (float, int, str, bool):
        return annotation, nullable
    if typing.get_origin(annotation) in (Mapping, dict) and typing.get_args(annotation) == (str, str):
        return dict, nullable
    raise TypeError(
        f"{label} has unsupported report annotation {annotation!r}; "
        "supported types are float, int, str, bool, and Mapping[str, str], optionally | None"
    )


def _checked(value: Any, annotation: type, nullable: bool, path: str) -> Any:
    """Validate ``value`` against a field annotation; raise naming ``path``."""
    if value is None and nullable:
        return None
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ReportValidationError(f"{path} must be a number")
        if not math.isfinite(value):
            raise ReportValidationError(f"{path} must be finite")
        return float(value)
    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ReportValidationError(f"{path} must be an integer")
        if not float(value).is_integer():
            raise ReportValidationError(f"{path} must be an integer")
        return int(value)
    if annotation is str:
        if not isinstance(value, str):
            raise ReportValidationError(f"{path} must be a string")
        return value
    if annotation is bool:
        if not isinstance(value, bool):
            raise ReportValidationError(f"{path} must be a boolean")
        return value
    if annotation is dict:
        if not isinstance(value, Mapping) or not all(
            isinstance(key, str) and isinstance(item, str) for key, item in value.items()
        ):
            raise ReportValidationError(f"{path} must map strings to strings")
        return dict(value)
    raise AssertionError(f"unhandled normalized report annotation: {annotation!r}")
