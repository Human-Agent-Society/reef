"""A scenario's serving config: the serving fields that it sets over the deployment's values (RFC #1097).

A component declares the fields that a scenario may set and checks their values. For the component that an
inference runtime serves, that is ``InferenceRuntime.serving_config``. Control stores the scenario's
``ServingConfig`` with its settings. The serving request carries the override of the served component to
inference, which applies it to the engine (``InferenceRuntime.apply_serving_config``) and to each request
(``InferenceHandler.request_payload``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from reef.core.values import JsonValue, require_text


def require_json(value: object, name: str) -> JsonValue:
    """``value`` when it is a ``JsonValue`` whose numbers are finite; else ``ValueError`` naming ``name``."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
        return value
    if isinstance(value, list):
        return [require_json(item, f"{name}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, dict):
        return {
            require_text(key, f"a key of {name}"): require_json(item, f"{name}.{key}") for key, item in value.items()
        }
    raise ValueError(f"{name} must be a JSON value, not {type(value).__name__}")


@dataclass(frozen=True)
class ServingConfig:
    """The serving fields that one scenario sets over its deployment's values, by component.

    Values: ``overrides`` maps a component name, a non-empty string, to the component's override: a non-empty
    mapping from a field name, a non-empty string, to a ``JsonValue`` other than null whose numbers are finite. A
    component that sets no field has no key, so ``ServingConfig()`` is the config of a scenario that sets nothing.
    Construction checks these conditions and copies the mappings.

    Purpose: control stores it in ``ScenarioSettings.serving_config`` and changes it with ``updated``. The serving
    request carries the override of the served component (``component``) to inference.
    """

    overrides: Mapping[str, Mapping[str, JsonValue]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.overrides, Mapping):
            raise ValueError("a serving config maps component names to overrides")
        overrides: dict[str, dict[str, JsonValue]] = {}
        for component, override in self.overrides.items():
            require_text(component, "a serving config component")
            if not isinstance(override, Mapping) or not override:
                raise ValueError(f"the serving config of component {component!r} must be a non-empty object")
            checked: dict[str, JsonValue] = {}
            for name, value in override.items():
                require_text(name, f"a serving field of component {component!r}")
                if value is None:
                    raise ValueError(f"serving field {component}.{name} must not be null")
                checked[name] = require_json(value, f"serving field {component}.{name}")
            overrides[component] = checked
        object.__setattr__(self, "overrides", overrides)

    def component(self, name: str) -> dict[str, JsonValue]:
        """A copy of the override of component ``name``: empty when the scenario sets no field of it."""
        return dict(self.overrides.get(name, {}))

    def updated(self, update: object) -> ServingConfig:
        """The config after ``update``, a mapping from component name to a mapping from field name to a value.

        For each component ``c`` and field ``f`` of ``update``, a value other than null sets ``f`` of ``c``, and
        null removes it. Every other field keeps its value, and a component left with no field has no override.
        ``ValueError`` when ``update`` is not such a mapping, or when a value is no ``JsonValue``.
        """
        if not isinstance(update, Mapping):
            raise ValueError("serving_config must map component names to objects of fields")
        overrides = {component: dict(override) for component, override in self.overrides.items()}
        for component, fields in update.items():
            require_text(component, "a serving config component")
            if not isinstance(fields, Mapping):
                raise ValueError(f"the serving config of component {component!r} must be an object")
            override = overrides.setdefault(component, {})
            for name, value in fields.items():
                require_text(name, f"a serving field of component {component!r}")
                if value is None:
                    override.pop(name, None)
                else:
                    override[name] = require_json(value, f"serving field {component}.{name}")
            if not override:
                overrides.pop(component)
        return ServingConfig(overrides)

    def to_json(self) -> dict[str, dict[str, JsonValue]]:
        """The config as a JSON object: ``overrides`` itself."""
        return {component: dict(override) for component, override in self.overrides.items()}

    @classmethod
    def from_json(cls, value: object) -> ServingConfig:
        """The config that ``to_json`` wrote; ``ValueError`` for any other value."""
        if not isinstance(value, Mapping):
            raise ValueError("a serving config is a JSON object")
        return cls(value)


__all__ = ["ServingConfig", "require_json"]
