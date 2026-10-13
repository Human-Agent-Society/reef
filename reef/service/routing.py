"""The routing manifest: per-scenario model routing declared once, validated twice.

A manifest names the logical inference targets a deployment may route to and,
for each scenario, the policy Semantic Router applies and the targets that
scenario allows. Automatic selection (keyword decisions and the catch-all
default) and explicit client pins are both confined to the scenario's allowed
targets; Reef independently re-validates the scenario and target against this
same manifest before running a request. Requests without routing keep their
existing path: a deployment enables routing by pointing ``reef.routing_manifest``
at a manifest file.

The manifest carries no credentials: target entries address public endpoints
and provider model names only. Its canonical SHA-256 digest identifies the
routing configuration a deployment runs under (not a model version).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import yaml

#: Service names ``reef serve`` owns when routing is enabled.
SEMANTIC_ROUTER_SERVICE = "semantic-router"
ENVOY_SERVICE = "envoy"


class RoutingManifestError(ValueError):
    """A routing manifest cannot be loaded or contradicts itself."""


def routing_digest(manifest: Mapping[str, object]) -> str:
    """The SHA-256 of the manifest's canonical JSON form: stable across key order."""
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True)
class RoutingTarget:
    """One logical target: an OpenAI-compatible endpoint and the model to request."""

    target_id: str
    endpoint: str
    model: str


@dataclass(frozen=True)
class KeywordDecision:
    """One keyword decision: its first matching entry selects the target."""

    name: str
    keywords: tuple[str, ...]
    target: str


@dataclass(frozen=True)
class ScenarioRouting:
    """One scenario's policy and its allowed targets.

    ``static`` pins every request of the scenario to one target. Otherwise the
    ordered ``keywords`` decisions select a target and ``default`` catches the
    rest; explicit client pins select from ``allowed_targets`` either way. At
    least one channel (``static``, a keyword decision, ``default`` or a pin)
    must be usable, so a scenario declares ``allowed_targets`` and any policy.
    """

    scenario: str
    allowed_targets: tuple[str, ...]
    static: str | None = None
    keywords: tuple[KeywordDecision, ...] = ()
    default: str | None = None

    def validate(self, targets: Mapping[str, RoutingTarget]) -> None:
        defined = set(targets)
        allowed = set(self.allowed_targets)
        unknown_allowed = allowed - defined
        if unknown_allowed:
            raise RoutingManifestError(
                f"scenario {self.scenario!r} allows undefined targets: {', '.join(sorted(unknown_allowed))}"
            )
        referenced = {decision.target for decision in self.keywords}
        for reference in (self.static, self.default):
            if reference is not None:
                referenced.add(reference)
        unknown = referenced - defined
        if unknown:
            raise RoutingManifestError(
                f"scenario {self.scenario!r} routes to undefined targets: {', '.join(sorted(unknown))}"
            )
        outside = referenced - allowed
        if outside:
            raise RoutingManifestError(
                f"scenario {self.scenario!r} routes to targets outside its allowed set: {', '.join(sorted(outside))}"
            )
        if not referenced and not self.allowed_targets:
            raise RoutingManifestError(f"scenario {self.scenario!r} allows no targets")


@dataclass(frozen=True)
class RoutingManifest:
    """The parsed, self-consistent routing manifest."""

    targets: Mapping[str, RoutingTarget]
    scenarios: Mapping[str, ScenarioRouting]
    default_target: str | None
    digest: str

    def validate_scenario(self, scenario: str, target: str | None) -> None:
        """Raise unless the scenario exists and the target is one of its allowed targets.

        The independent check Reef runs before executing a routed request;
        ``target`` is the effective target after automatic selection or a pin.
        """
        routing = self.scenarios.get(scenario)
        if routing is None:
            raise RoutingManifestError(f"scenario {scenario!r} is not in the routing manifest")
        if target is not None and target not in routing.allowed_targets:
            raise RoutingManifestError(
                f"target {target!r} is not allowed for scenario {scenario!r}; "
                f"allowed: {', '.join(routing.allowed_targets)}"
            )

    def resolve_scenario_target(self, scenario: str, pinned: str | None = None) -> str:
        """The scenario's target for a request: an allowed pin, or the policy's selection."""
        routing = self.scenarios.get(scenario)
        if routing is None:
            raise RoutingManifestError(f"scenario {scenario!r} is not in the routing manifest")
        if pinned is not None:
            self.validate_scenario(scenario, pinned)
            return pinned
        if routing.static is not None:
            return routing.static
        if routing.default is not None:
            return routing.default
        if self.default_target is not None and self.default_target in routing.allowed_targets:
            return self.default_target
        raise RoutingManifestError(
            f"scenario {scenario!r} has no automatic selection; clients must pin one of: "
            f"{', '.join(routing.allowed_targets)}"
        )


def _require_mapping(value: object, what: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RoutingManifestError(f"{what} must be an object")
    return value


def _parse_target(target_id: str, entry: object) -> RoutingTarget:
    entry = _require_mapping(entry, f"target {target_id!r}")
    endpoint = entry.get("endpoint")
    model = entry.get("model")
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise RoutingManifestError(f"target {target_id!r} must declare a non-empty endpoint")
    try:
        url = urlsplit(endpoint)
        valid = url.scheme in ("http", "https") and bool(url.hostname)
        if url.port is not None and not 1 <= url.port <= 65535:
            valid = False
    except ValueError:
        valid = False
    if not valid:
        raise RoutingManifestError(f"target {target_id!r} endpoint must be an HTTP(S) URL with a valid host")
    if not isinstance(model, str) or not model.strip():
        raise RoutingManifestError(f"target {target_id!r} must declare a non-empty provider model name")
    return RoutingTarget(target_id=target_id, endpoint=endpoint.strip(), model=model.strip())


def _parse_keywords(scenario: str, keywords: object) -> tuple[KeywordDecision, ...]:
    if not isinstance(keywords, Sequence) or isinstance(keywords, (str, bytes)):
        raise RoutingManifestError(f"scenario {scenario!r} keywords must be a list")
    decisions: list[KeywordDecision] = []
    for index, entry in enumerate(keywords):
        entry = _require_mapping(entry, f"scenario {scenario!r} keywords[{index}]")
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            raise RoutingManifestError(f"scenario {scenario!r} keywords[{index}] must have a non-empty name")
        matched = entry.get("keywords")
        if (
            not isinstance(matched, Sequence)
            or isinstance(matched, (str, bytes))
            or not matched
            or any(not isinstance(keyword, str) or not keyword for keyword in matched)
        ):
            raise RoutingManifestError(
                f"scenario {scenario!r} keyword decision {name!r} needs a non-empty list of keyword strings"
            )
        target = entry.get("target")
        if not isinstance(target, str) or not target.strip():
            raise RoutingManifestError(f"scenario {scenario!r} keyword decision {name!r} must name a target")
        decisions.append(KeywordDecision(name=name.strip(), keywords=tuple(matched), target=target.strip()))
    return tuple(decisions)


def _parse_scenario(scenario: str, entry: object) -> ScenarioRouting:
    entry = _require_mapping(entry, f"scenario {scenario!r}")
    allowed = entry.get("targets")
    if (
        not isinstance(allowed, Sequence)
        or isinstance(allowed, (str, bytes))
        or not allowed
        or any(not isinstance(target, str) or not target.strip() for target in allowed)
    ):
        raise RoutingManifestError(f"scenario {scenario!r} must declare a non-empty list of allowed targets")
    allowed_targets = tuple(dict.fromkeys(target.strip() for target in allowed))
    if len(allowed_targets) != len(allowed):
        raise RoutingManifestError(f"scenario {scenario!r} repeats a target in its allowed list")
    static = entry.get("static")
    if static is not None and (not isinstance(static, str) or not static.strip()):
        raise RoutingManifestError(f"scenario {scenario!r} static must name one target")
    default = entry.get("default")
    if default is not None and (not isinstance(default, str) or not default.strip()):
        raise RoutingManifestError(f"scenario {scenario!r} default must name one target")
    return ScenarioRouting(
        scenario=scenario,
        allowed_targets=allowed_targets,
        static=static.strip() if isinstance(static, str) else None,
        keywords=_parse_keywords(scenario, entry.get("keywords", ())),
        default=default.strip() if isinstance(default, str) else None,
    )


def parse_routing_manifest(manifest: Mapping[str, object]) -> RoutingManifest:
    """Validate an already-loaded manifest mapping into a :class:`RoutingManifest`."""
    manifest = _require_mapping(manifest, "routing manifest")
    raw_targets = manifest.get("targets")
    if not isinstance(raw_targets, Mapping) or not raw_targets:
        raise RoutingManifestError("routing manifest must declare a non-empty 'targets' object")
    targets = {target_id: _parse_target(target_id, entry) for target_id, entry in raw_targets.items()}
    raw_scenarios = manifest.get("scenarios")
    if not isinstance(raw_scenarios, Mapping) or not raw_scenarios:
        raise RoutingManifestError("routing manifest must declare a non-empty 'scenarios' object")
    scenarios: dict[str, ScenarioRouting] = {}
    for scenario, entry in raw_scenarios.items():
        if not isinstance(scenario, str) or not scenario.strip():
            raise RoutingManifestError("scenario names must be non-empty strings")
        scenarios[scenario] = _parse_scenario(scenario, entry)
    default_target = manifest.get("default_target")
    if default_target is not None:
        if not isinstance(default_target, str) or not default_target.strip():
            raise RoutingManifestError("default_target must name one target")
        default_target = default_target.strip()
        if default_target not in targets:
            raise RoutingManifestError(f"default_target {default_target!r} is not a declared target")
    parsed = RoutingManifest(
        targets=targets,
        scenarios=scenarios,
        default_target=default_target,
        digest=routing_digest(manifest),
    )
    for scenario_routing in scenarios.values():
        scenario_routing.validate(targets)
    return parsed


def load_routing_manifest(source: str | Path) -> RoutingManifest:
    """Load and validate the manifest file at ``source``."""
    path = Path(source)
    try:
        loaded = yaml.safe_load(path.read_text())
    except FileNotFoundError as exc:
        raise RoutingManifestError(f"routing manifest not found: {path}") from exc
    except (yaml.YAMLError, UnicodeDecodeError) as exc:
        raise RoutingManifestError(f"routing manifest {path} is not valid YAML: {exc}") from exc
    if not isinstance(loaded, Mapping):
        raise RoutingManifestError(f"routing manifest {path} must be an object")
    return parse_routing_manifest(loaded)
