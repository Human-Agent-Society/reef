"""Validate scenario-scoped external-model routing before Scenario resolution.

A client names its Scenario in ``x-reef-scenario`` and asks for the automatic
model or one of that Scenario's targets. The Semantic Router's routing preview
selects an automatic request through the Scenario's private entrypoint
(``reef.service.routing_preview``); an explicit target is checked here without
the router. Both checks use one immutable deployment manifest and the router
runtime it was published with. External revisions are deployment
declarations, not Reef artifacts or engine-verified runtime load IDs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Literal
from urllib.parse import urlsplit

import yaml

from reef.core.errors import ReefError
from reef.inference.http import HttpInferenceHandler, provider_request_headers
from reef.service.wire import RELEASE_ID_HEADER, SCENARIO_HEADER

#: The catch-all rule that sends automatic requests to a scenario's default target.
DEFAULT_RULE = "default"
#: Names the Semantic Router reserves for its own recipes.
RESERVED_SCENARIOS = frozenset({"default", "global"})
#: Model names under this prefix are Semantic Router entrypoints that only Reef may send.
ENTRYPOINT_PREFIX = "reef-entrypoint/"
#: Selection headers a routing gateway used to set. Reef now selects itself, so a client sending one is refused.
RESERVED_HEADERS = frozenset({"x-selected-model", "x-reef-sr-decision"})
RESERVED_HEADER_PREFIX = "x-reef-routing-"
#: Headers a routed request may carry once; ``x-reef-*`` headers are single-valued as well.
SINGLE_VALUE_HEADERS = frozenset({"authorization", "x-api-key", "content-type"})
#: A Semantic Router runtime hash: the SHA-256 of the configuration a router runtime serves.
ROUTER_CONFIG_HASH = re.compile(r"[0-9a-f]{64}")
#: A routing snapshot version or a deployment configuration digest.
SHA256_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
#: A keyword rule's name; the Semantic Router reports it as the decision name.
RULE_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}")
#: An environment variable name that may hold a target's credential.
ENVIRONMENT_VARIABLE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
#: Every status a ``RoutingRequestError`` reports; ``reef.service.errors`` translates each one.
ROUTING_ERROR_STATUSES = frozenset({400, 403, 409, 413, 429, 502, 503, 504})
#: Each provider model-list check during readiness waits at most this long, or the inference timeout if shorter.
#: The managed owner budgets an ``open`` with the same limit.
MAX_READINESS_TIMEOUT_SECONDS = 10.0


def is_router_config_hash(value: object) -> bool:
    """Whether ``value`` is a Semantic Router runtime hash: 64 lowercase hexadecimal characters."""
    return isinstance(value, str) and ROUTER_CONFIG_HASH.fullmatch(value) is not None


def scenario_entrypoint(scenario: str) -> str:
    """The private Semantic Router model name that selects one Scenario's recipe."""
    return ENTRYPOINT_PREFIX + scenario


def reserved_model_name(name: str) -> bool:
    """Whether a model name falls in the private entrypoint namespace, ignoring case."""
    return name.casefold().startswith(ENTRYPOINT_PREFIX)


class RoutingRequestError(ReefError):
    """A routed request that Reef refuses or cannot serve, with the HTTP status the caller receives.

    Values: ``code`` is a non-empty machine-readable reason, such as ``unknown_target`` or
    ``routing_timeout``; codes form an open set. ``status`` is an int in ``ROUTING_ERROR_STATUSES``. 400, 403
    and 413 refuse the request itself; 409 means the request's routing decision belongs to another snapshot;
    429 means the router is at its preview capacity; 502 is a router reply Reef cannot use, 503 a router or
    target that is unavailable, and 504 a router that did not answer in time.
    Purpose: request checks, the routing preview and readiness raise it before any provider call;
    ``reef.service.errors.translate_error`` turns it into the response ``"<code>: <message>"``.
    """

    def __init__(self, message: str, *, code: str = "invalid_route", status: int = 400) -> None:
        if not isinstance(code, str) or not code:
            raise ValueError("routing error code must be a non-empty string")
        if type(status) is not int or status not in ROUTING_ERROR_STATUSES:
            raise ValueError(f"routing error status must be one of {sorted(ROUTING_ERROR_STATUSES)}")
        super().__init__(message)
        self.code = code
        self.status = status


def manifest_mapping(value: object, name: str, allowed: set[str]) -> dict[str, object]:
    """Read one manifest object and reject misspelled or non-string keys."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str) or key not in allowed:
            raise ValueError(f"{name} contains an unknown field: {key!r}")
        result[key] = item
    return result


def manifest_string(value: object, name: str) -> str:
    """Read a manifest string without silently changing identity.

    A manifest string is a non-empty ``str`` that ``str.strip`` leaves unchanged and that has no code point
    below 32. Other control characters, such as DEL, are not refused.
    """
    if not isinstance(value, str) or not value or value != value.strip() or any(ord(char) < 32 for char in value):
        raise ValueError(f"{name} must be a non-empty string without surrounding whitespace or control characters")
    return value


def distinct_strings(value: object, name: str) -> tuple[str, ...]:
    """A non-empty list or tuple of distinct manifest strings, as a tuple in the same order."""
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    items = tuple(manifest_string(item, name) for item in value)
    if len(set(items)) != len(items):
        raise ValueError(f"{name} must not repeat a value")
    return items


def manifest_strings(value: object, name: str) -> tuple[str, ...]:
    """Read a manifest field that must be a non-empty list of distinct manifest strings."""
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a non-empty list")
    return distinct_strings(value, name)


@dataclass(frozen=True)
class RoutingTarget:
    """One external model deployment that a routing manifest names by its logical target ID.

    Values: ``target_id``, ``provider_model``, ``revision`` and ``url`` are manifest strings (see
    ``manifest_string``). ``target_id`` is outside the private entrypoint
    namespace (``reserved_model_name``). ``url`` is an http or https URL with a host and a valid port if one
    is given, and no credentials, query or fragment. ``role`` is None or a manifest string; it is a label,
    not a closed set. ``api_key_env`` is None or an environment variable name. ``enabled`` is a bool.
    Purpose: ``parse_target`` makes it from one ``targets`` entry. A request is served only by an enabled
    target; a disabled one stays in the catalog, so a policy update cannot change the catalog. Several
    Scenarios may list the same target; listing it does not reserve it.
    """

    target_id: str
    provider_model: str
    revision: str
    url: str
    role: str | None = None
    api_key_env: str | None = None
    enabled: bool = True

    def __post_init__(self) -> None:
        manifest_string(self.target_id, "target ID")
        name = f"targets.{self.target_id}"
        if reserved_model_name(self.target_id):
            raise ValueError(
                f"target ID {self.target_id!r} must not start with the reserved prefix {ENTRYPOINT_PREFIX!r}"
            )
        manifest_string(self.provider_model, f"{name}.provider_model")
        manifest_string(self.revision, f"{name}.revision")
        parsed_url = urlsplit(manifest_string(self.url, f"{name}.url"))
        if (
            parsed_url.scheme not in ("http", "https")
            or not parsed_url.hostname
            or parsed_url.username is not None
            or parsed_url.password is not None
            or parsed_url.query
            or parsed_url.fragment
        ):
            raise ValueError(f"{name}.url must be an HTTP URL without credentials, query, or fragment")
        if parsed_url.port is not None and parsed_url.port < 1:
            raise ValueError(f"{name}.url must use a valid port")
        if self.role is not None:
            manifest_string(self.role, f"{name}.role")
        if self.api_key_env is not None and not ENVIRONMENT_VARIABLE_NAME.fullmatch(
            manifest_string(self.api_key_env, f"{name}.api_key_env")
        ):
            raise ValueError(f"{name}.api_key_env must name an environment variable")
        if type(self.enabled) is not bool:
            raise ValueError(f"{name}.enabled must be a boolean")


@dataclass(frozen=True)
class RoutingRule:
    """A keyword rule that sends a matching automatic request to one target.

    Values: ``name`` matches ``RULE_NAME`` and is not ``DEFAULT_RULE``. ``keywords`` is a non-empty tuple of
    distinct manifest strings in manifest order; distinctness compares the exact strings, and a list given
    to the constructor is stored as a tuple. ``target_id`` is a manifest string.
    Purpose: ``parse_scenario`` makes it. The renderer turns it into one Semantic Router keyword signal: the
    rule matches a request whose text contains any keyword, ignoring case. Its Scenario and the snapshot
    check that ``target_id`` is one of the Scenario's enabled targets.
    """

    name: str
    keywords: tuple[str, ...]
    target_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not RULE_NAME.fullmatch(self.name) or self.name == DEFAULT_RULE:
            raise ValueError(f"routing rule names must be short identifiers other than {DEFAULT_RULE!r}")
        object.__setattr__(self, "keywords", distinct_strings(self.keywords, f"rules.{self.name}.keywords"))
        manifest_string(self.target_id, f"rules.{self.name}.target")


@dataclass(frozen=True)
class RoutingScenario:
    """The targets one Scenario may use and the rules that choose among them.

    Values: ``name`` is a manifest string outside ``RESERVED_SCENARIOS``. ``target_ids`` is a non-empty tuple of
    distinct manifest strings; ``default_target`` ∈ ``target_ids``. ``rules`` is a tuple, possibly empty, of
    ``RoutingRule`` values with distinct names, and ``rule.target_id`` ∈ ``target_ids`` for each rule. Earlier
    rules take priority. The constructor stores a list of target IDs or rules as a tuple.
    Which of these targets exist and are enabled is checked by the ``RoutingSnapshot`` that holds the
    Scenario.
    Purpose: ``parse_scenario`` makes it. ``target_for_rule`` maps the Semantic Router's decision name to a
    target: ``DEFAULT_RULE`` to ``default_target``, a rule name to its target, and any other name to None.
    """

    name: str
    default_target: str
    target_ids: tuple[str, ...]
    rules: tuple[RoutingRule, ...] = ()

    def __post_init__(self) -> None:
        if manifest_string(self.name, "scenario name") in RESERVED_SCENARIOS:
            raise ValueError(f"scenario name {self.name!r} is reserved by the Semantic Router")
        name = f"scenarios.{self.name}"
        object.__setattr__(self, "target_ids", distinct_strings(self.target_ids, f"{name}.targets"))
        if manifest_string(self.default_target, f"{name}.default_target") not in self.target_ids:
            raise ValueError(f"{name}.default_target must identify one of its targets")
        if not isinstance(self.rules, (list, tuple)) or not all(isinstance(rule, RoutingRule) for rule in self.rules):
            raise ValueError(f"{name}.rules must be a list or tuple of routing rules")
        rules = tuple(self.rules)
        names = [rule.name for rule in rules]
        if len(set(names)) != len(names):
            raise ValueError(f"{name}.rules repeats a rule name")
        for rule in rules:
            if rule.target_id not in self.target_ids:
                raise ValueError(f"{name}.rules.{rule.name}.target must identify one of its targets")
        object.__setattr__(self, "rules", rules)

    def target_for_rule(self, rule_name: str) -> str | None:
        """Return the target an automatic selection rule chooses, if the rule exists."""
        if rule_name == DEFAULT_RULE:
            return self.default_target
        return next((rule.target_id for rule in self.rules if rule.name == rule_name), None)


def parse_target(target_id: str, value: object) -> RoutingTarget:
    """Read one ``targets`` entry, refusing unknown and explicitly null fields; ``RoutingTarget`` checks domains."""
    target = manifest_mapping(
        value,
        f"targets.{target_id}",
        {"provider_model", "revision", "url", "role", "api_key_env", "enabled"},
    )
    role = manifest_string(target["role"], f"targets.{target_id}.role") if "role" in target else None
    api_key_env = (
        manifest_string(target["api_key_env"], f"targets.{target_id}.api_key_env") if "api_key_env" in target else None
    )
    enabled = target.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError(f"targets.{target_id}.enabled must be a boolean")
    return RoutingTarget(
        target_id,
        manifest_string(target.get("provider_model"), f"targets.{target_id}.provider_model"),
        manifest_string(target.get("revision"), f"targets.{target_id}.revision"),
        manifest_string(target.get("url"), f"targets.{target_id}.url"),
        role,
        api_key_env,
        enabled,
    )


def parse_scenario(name: str, value: object) -> RoutingScenario:
    """Read one Scenario entry and its rules; ``RoutingScenario`` and ``RoutingRule`` check their domains."""
    scenario = manifest_mapping(value, f"scenarios.{name}", {"default_target", "targets", "rules"})
    raw_rules = scenario.get("rules", [])
    if not isinstance(raw_rules, list):
        raise ValueError(f"scenarios.{name}.rules must be a list")
    rules: list[RoutingRule] = []
    for raw_rule in raw_rules:
        rule = manifest_mapping(raw_rule, f"scenarios.{name}.rules[]", {"name", "keywords", "target"})
        rule_name = manifest_string(rule.get("name"), f"scenarios.{name}.rules[].name")
        rules.append(
            RoutingRule(
                rule_name,
                manifest_strings(rule.get("keywords"), f"scenarios.{name}.rules.{rule_name}.keywords"),
                manifest_string(rule.get("target"), f"scenarios.{name}.rules.{rule_name}.target"),
            )
        )
    return RoutingScenario(
        name,
        manifest_string(scenario.get("default_target"), f"scenarios.{name}.default_target"),
        manifest_strings(scenario.get("targets"), f"scenarios.{name}.targets"),
        tuple(rules),
    )


@dataclass(frozen=True)
class RoutingSnapshot:
    """One validated routing manifest: its target catalog, its Scenarios and the digest that names it.

    Values: ``automatic_model`` is a manifest string outside the private entrypoint namespace and is not a
    target ID. ``targets`` is a non-empty read-only mapping of ``RoutingTarget`` values in which each key
    equals its target's ``target_id``. ``scenarios`` is a non-empty read-only mapping of ``RoutingScenario``
    values in which each key equals its Scenario's ``name``. For each Scenario ``s``: ``target_ids(s) ⊆ keys(targets)``, and its default target and every
    rule target are enabled. Targets may be shared between Scenarios. ``version`` is a ``sha256:<hex>``
    digest; the constructor checks only its form.
    Purpose: ``from_mapping`` makes it and sets ``version`` to the SHA-256 of the manifest's canonical JSON,
    including fields the snapshot does not keep. ``RoutingAdapter`` and the renderer read it; nothing
    changes it after construction.
    """

    automatic_model: str
    targets: Mapping[str, RoutingTarget]
    scenarios: Mapping[str, RoutingScenario]
    version: str

    def __post_init__(self) -> None:
        targets = MappingProxyType(dict(self.targets))
        scenarios = MappingProxyType(dict(self.scenarios))
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "scenarios", scenarios)
        if reserved_model_name(manifest_string(self.automatic_model, "automatic_model")):
            raise ValueError(f"automatic_model must not start with the reserved prefix {ENTRYPOINT_PREFIX!r}")
        if not targets:
            raise ValueError("targets must be a non-empty object")
        if not scenarios:
            raise ValueError("scenarios must be a non-empty object")
        for target_id, target in targets.items():
            if not isinstance(target, RoutingTarget):
                raise ValueError(f"targets.{target_id} must be a routing target")
            if target_id != target.target_id:
                raise ValueError(f"targets key {target_id!r} must equal its target ID")
        if self.automatic_model in targets:
            raise ValueError("automatic_model must not also identify a target")
        for scenario_name, scenario in scenarios.items():
            if not isinstance(scenario, RoutingScenario):
                raise ValueError(f"scenarios.{scenario_name} must be a routing Scenario")
            if scenario_name != scenario.name:
                raise ValueError(f"scenarios key {scenario_name!r} must equal its Scenario name")
            for target_id in scenario.target_ids:
                if target_id not in targets:
                    raise ValueError(f"scenarios.{scenario_name}.targets references unknown target {target_id!r}")
            if not targets[scenario.default_target].enabled:
                raise ValueError(f"scenarios.{scenario_name}.default_target must identify one of its enabled targets")
            for rule in scenario.rules:
                if not targets[rule.target_id].enabled:
                    raise ValueError(
                        f"scenarios.{scenario_name}.rules.{rule.name}.target must identify one of its enabled targets"
                    )
        if not isinstance(self.version, str) or not SHA256_DIGEST.fullmatch(self.version):
            raise ValueError("routing snapshot version must be a sha256 digest")

    @classmethod
    def load(cls, path: str | Path) -> RoutingSnapshot:
        """Load YAML or JSON once; no request reads a mutable config file."""
        try:
            value: object = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError("invalid routing manifest YAML") from exc
        return cls.from_mapping(value)

    @classmethod
    def from_mapping(cls, value: object) -> RoutingSnapshot:
        """Validate schema version 2 and hash its canonical JSON representation.

        The parsers read the manifest's shape and refuse unknown or explicitly null fields; the value types'
        constructors check every field's domain and the snapshot checks references between them.
        """
        manifest = manifest_mapping(
            value,
            "routing manifest",
            {"schema_version", "automatic_model", "targets", "scenarios", "deployment"},
        )
        if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 2:
            raise ValueError("routing manifest schema_version must be 2")
        automatic_model = manifest_string(manifest.get("automatic_model"), "automatic_model")
        deployment = manifest_mapping(
            manifest.get("deployment"), "deployment", {"semantic_router_image", "configuration_digest"}
        )
        image = manifest_string(deployment.get("semantic_router_image"), "deployment.semantic_router_image")
        if not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", image):
            raise ValueError("deployment.semantic_router_image must pin an image by sha256 digest")
        if "configuration_digest" in deployment:
            configuration_digest = manifest_string(
                deployment["configuration_digest"], "deployment.configuration_digest"
            )
            if not SHA256_DIGEST.fullmatch(configuration_digest):
                raise ValueError("deployment.configuration_digest must be a sha256 digest")
        raw_targets = manifest.get("targets")
        if not isinstance(raw_targets, Mapping):
            raise ValueError("targets must be a non-empty object")
        targets: dict[str, RoutingTarget] = {}
        for raw_id, raw_target in raw_targets.items():
            target_id = manifest_string(raw_id, "target ID")
            targets[target_id] = parse_target(target_id, raw_target)
        raw_scenarios = manifest.get("scenarios")
        if not isinstance(raw_scenarios, Mapping):
            raise ValueError("scenarios must be a non-empty object")
        scenarios: dict[str, RoutingScenario] = {}
        for raw_name, raw_scenario in raw_scenarios.items():
            name = manifest_string(raw_name, "scenario name")
            scenarios[name] = parse_scenario(name, raw_scenario)
        canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        version = "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return cls(automatic_model, targets, scenarios, version)


def check_policy_change(current: RoutingSnapshot, candidate: RoutingSnapshot) -> None:
    """Allow a policy update to change only rules, defaults and Scenario membership within the same catalog.

    The managed owner checks a candidate before it changes the router, and the
    REEF child checks it again before it publishes the candidate.

    Raises:
        ValueError: The candidate changes something that needs a restart.
    """
    if candidate.targets != current.targets:
        raise ValueError("target catalog, provider, revision or endpoint changed; restart the deployment")
    if candidate.automatic_model != current.automatic_model:
        raise ValueError("automatic selector changed; restart the deployment")
    if set(candidate.scenarios) != set(current.scenarios):
        raise ValueError("Scenario names or router entrypoints changed; restart the deployment")


@dataclass(frozen=True)
class RoutingDecision:
    """The target one routed request uses, and why, under the snapshot it was admitted with.

    Values: ``target`` is an enabled ``RoutingTarget``. ``snapshot_version`` is a ``sha256:<hex>`` digest, and
    ``scenario`` and ``original_selector`` are manifest strings. The ``reason`` cases are exhaustive:

    - ``explicit_pin``: the caller named the target, so ``original_selector == target.target_id``, and
      ``rule_name`` and ``router_config_hash`` are None.
    - ``default``: the router chose the Scenario's default, so ``rule_name == DEFAULT_RULE``.
    - ``rule``: the router matched a keyword rule, so ``rule_name`` matches ``RULE_NAME`` and is not
      ``DEFAULT_RULE``.

    For ``default`` and ``rule``, ``router_config_hash`` is a router runtime hash: that of the router that
    selected.
    Purpose: ``RoutingAdapter.explicit_decision`` makes pins and ``checked_selection`` makes the other two,
    with the snapshot's automatic model as ``original_selector``. Each also checks that the target belongs
    to the Scenario in that snapshot, which a decision alone cannot. The
    request keeps one decision for its provider call and every retry; ``metadata`` is its stored record form.
    """

    target: RoutingTarget
    snapshot_version: str
    scenario: str
    original_selector: str
    reason: Literal["default", "rule", "explicit_pin"]
    rule_name: str | None = None
    #: The Semantic Router runtime that selected an automatic request; None for an explicit pin.
    router_config_hash: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.target, RoutingTarget) or not self.target.enabled:
            raise ValueError("a routing decision must use an enabled routing target")
        if not isinstance(self.snapshot_version, str) or not SHA256_DIGEST.fullmatch(self.snapshot_version):
            raise ValueError("a routing decision's snapshot_version must be a sha256 digest")
        manifest_string(self.scenario, "routing decision scenario")
        manifest_string(self.original_selector, "routing decision selector")
        if self.reason == "explicit_pin":
            if (
                self.original_selector != self.target.target_id
                or self.rule_name is not None
                or self.router_config_hash is not None
            ):
                raise ValueError("an explicit pin names its own target and has no rule or router runtime")
            return
        if self.reason == "default":
            selected_by_rule = self.rule_name == DEFAULT_RULE
        elif self.reason == "rule":
            selected_by_rule = (
                isinstance(self.rule_name, str)
                and RULE_NAME.fullmatch(self.rule_name) is not None
                and self.rule_name != DEFAULT_RULE
            )
        else:
            raise ValueError("a routing decision's reason must be default, rule or explicit_pin")
        if not selected_by_rule:
            raise ValueError(f"a {self.reason} decision has the wrong rule name")
        if not is_router_config_hash(self.router_config_hash):
            raise ValueError("an automatic routing decision needs the selecting router's runtime hash")

    def metadata(self) -> dict[str, object]:
        """Return server-owned selection fields for the stored record only."""
        return {
            "target_id": self.target.target_id,
            "role": self.target.role,
            "provider_model": self.target.provider_model,
            "model_revision": self.target.revision,
            "identity_source": "deployment_manifest",
            "policy_id": "static",
            "policy_version": self.snapshot_version,
            "snapshot_version": self.snapshot_version,
            "router_config_hash": self.router_config_hash,
            "original_selector": self.original_selector,
            "reason": self.reason,
            "rule_name": self.rule_name,
        }


@dataclass(frozen=True)
class RoutingAdapter:
    """One published routing policy: the manifest, its router runtime, and a separate HTTP handler per target.

    Values: ``snapshot`` is a ``RoutingSnapshot``. ``router_config_hash`` is None or a router runtime hash
    (``is_router_config_hash``). ``handlers`` (to ``HttpInferenceHandler`` values) and ``api_keys`` are
    read-only mappings whose keys are target IDs of ``snapshot``; either may be partial or empty. Each
    credential is a non-empty string. ``readiness_timeout_s`` is finite with
    ``0 < readiness_timeout_s ≤ MAX_READINESS_TIMEOUT_SECONDS``.
    Purpose: ``from_manifest`` makes the first adapter, with one handler per enabled target and a credential
    only for an enabled target that names ``api_key_env``; the others call their provider without one.
    ``RoutingPublication.publish`` replaces it with a new snapshot and runtime that keep those handlers.
    A request captures one adapter when it is admitted and keeps it for selection, its provider call and
    every retry. With ``router_config_hash`` None no automatic request can select: a managed child starts
    this way, closed. ``handler_for`` reports a target without a handler as unavailable (503).
    """

    snapshot: RoutingSnapshot
    #: The verified Semantic Router runtime whose previews select for this policy. None until the owner
    #: verifies one: a managed child starts this way, closed, and no automatic request can select.
    router_config_hash: str | None
    handlers: Mapping[str, HttpInferenceHandler]
    api_keys: Mapping[str, str] = field(default_factory=dict, repr=False, compare=False)
    readiness_timeout_s: float = MAX_READINESS_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, RoutingSnapshot):
            raise ValueError("a routing adapter needs a routing snapshot")
        if self.router_config_hash is not None and not is_router_config_hash(self.router_config_hash):
            raise ValueError("router_config_hash must be 64 lowercase hexadecimal characters")
        object.__setattr__(self, "handlers", MappingProxyType(dict(self.handlers)))
        object.__setattr__(self, "api_keys", MappingProxyType(dict(self.api_keys)))
        if not (set(self.handlers) | set(self.api_keys)).issubset(self.snapshot.targets):
            raise ValueError("routing handlers and credentials must belong to the snapshot's targets")
        if not all(isinstance(handler, HttpInferenceHandler) for handler in self.handlers.values()):
            raise ValueError("routing handlers must be HTTP inference handlers")
        if any(not isinstance(key, str) or not key for key in self.api_keys.values()):
            raise ValueError("routing credentials must be non-empty strings")
        timeout = self.readiness_timeout_s
        if (
            not isinstance(timeout, (int, float))
            or isinstance(timeout, bool)
            or not math.isfinite(timeout)
            or not 0 < timeout <= MAX_READINESS_TIMEOUT_SECONDS
        ):
            raise ValueError(f"routing readiness timeout must be in (0, {MAX_READINESS_TIMEOUT_SECONDS:g}] seconds")

    @classmethod
    def from_manifest(
        cls,
        path: str | Path,
        *,
        router_config_hash: str | None,
        environ: Mapping[str, str] | None = None,
        timeout_s: float = 300.0,
    ) -> RoutingAdapter:
        """Resolve credential references at startup without storing them in metadata."""
        snapshot = RoutingSnapshot.load(path)
        environment = os.environ if environ is None else environ
        handlers: dict[str, HttpInferenceHandler] = {}
        api_keys: dict[str, str] = {}
        for target in snapshot.targets.values():
            if not target.enabled:
                continue
            key = environment.get(target.api_key_env) if target.api_key_env is not None else None
            if target.api_key_env is not None and not key:
                raise ValueError(
                    f"routing target {target.target_id!r} requires environment variable {target.api_key_env}"
                )
            if key is not None:
                api_keys[target.target_id] = key
            handlers[target.target_id] = HttpInferenceHandler(
                target.url,
                request_headers=provider_request_headers(key) if key is not None else None,
                timeout_s=timeout_s,
            )
        return cls(snapshot, router_config_hash, handlers, api_keys, min(timeout_s, MAX_READINESS_TIMEOUT_SECONDS))

    async def check_ready(self) -> None:
        """Check configured model availability; this does not verify model weights."""
        from aiohttp import ClientError, ClientSession, ClientTimeout

        probes: dict[tuple[str, str | None], set[str]] = {}
        for target in self.snapshot.targets.values():
            if target.enabled:
                endpoint = (target.url.rstrip("/"), self.api_keys.get(target.target_id))
                probes.setdefault(endpoint, set()).add(target.provider_model)
        async with ClientSession(timeout=ClientTimeout(total=self.readiness_timeout_s)) as session:
            for (url, key), expected_models in probes.items():
                headers = {"Authorization": f"Bearer {key}"} if key is not None else {}
                try:
                    async with session.get(f"{url}/v1/models", headers=headers) as response:
                        if response.status != 200:
                            raise RoutingRequestError(
                                f"routing readiness check returned HTTP {response.status}",
                                code="target_unavailable",
                                status=503,
                            )
                        payload: object = await response.json()
                except (ClientError, TimeoutError, json.JSONDecodeError) as exc:
                    raise RoutingRequestError(
                        "routing readiness check failed", code="target_unavailable", status=503
                    ) from exc
                model_entries = payload.get("data") if isinstance(payload, Mapping) else None
                if not isinstance(model_entries, list):
                    raise RoutingRequestError(
                        "routing readiness response must contain a model list", code="target_unavailable", status=503
                    )
                actual_models = {
                    item["id"]
                    for item in model_entries
                    if isinstance(item, Mapping) and isinstance(item.get("id"), str)
                }
                if not expected_models.issubset(actual_models):
                    raise RoutingRequestError(
                        "routing provider does not expose all configured models", code="target_unavailable", status=503
                    )

    def check_request(self, headers: Iterable[tuple[str, str]], path: str) -> RoutingScenario:
        """The Scenario a routed request names, checked before its body is read.

        ``headers`` are the received headers with every repeat, so a header
        sent twice is refused instead of one copy winning.
        """
        if path != "/v1/chat/completions":
            raise RoutingRequestError("routing supports only /v1/chat/completions", code="unsupported_route")
        values: dict[str, list[str]] = {}
        for name, value in headers:
            values.setdefault(name.lower(), []).append(value)
        for name, items in values.items():
            if len(items) > 1 and (name in SINGLE_VALUE_HEADERS or name.startswith("x-reef-")):
                raise RoutingRequestError(f"request repeats the {name} header", code="duplicate_header")
            if name in RESERVED_HEADERS or name.startswith(RESERVED_HEADER_PREFIX):
                raise RoutingRequestError(f"{name} is not accepted from clients", code="reserved_header")
        if values.get(RELEASE_ID_HEADER, [""])[0].strip():
            raise RoutingRequestError("routed external targets do not accept x-reef-release-id", code="release_pin")
        name = values.get(SCENARIO_HEADER, [""])[0]
        if not name:
            raise RoutingRequestError("routing requires x-reef-scenario", code="missing_scenario")
        scenario = self.snapshot.scenarios.get(name)
        if scenario is None:
            raise RoutingRequestError("routing Scenario is not configured", code="unknown_scenario", status=403)
        return scenario

    def explicit_decision(self, scenario: RoutingScenario, payload: Mapping[str, object]) -> RoutingDecision | None:
        """The decision for a request naming one of ``scenario``'s targets, made without the router.

        Returns None for a request of the automatic model, which the router selects.
        """
        selector = payload.get("model")
        if not isinstance(selector, str) or not selector:
            raise RoutingRequestError("request body must name a model", code="missing_model")
        if selector == self.snapshot.automatic_model:
            return None
        if reserved_model_name(selector):
            raise RoutingRequestError("private router entrypoints cannot be requested", code="unknown_target")
        target = self.snapshot.targets.get(selector)
        if target is None:
            raise RoutingRequestError("routing selector is missing or unknown", code="unknown_target")
        if selector not in scenario.target_ids:
            raise RoutingRequestError(
                "routing target is not available to the Scenario", code="target_not_allowed", status=403
            )
        if not target.enabled:
            raise RoutingRequestError("routing target is disabled", code="target_unavailable", status=503)
        return RoutingDecision(target, self.snapshot.version, scenario.name, selector, "explicit_pin")

    def handler_for(self, decision: RoutingDecision) -> HttpInferenceHandler:
        """Return the handler captured for this snapshot, never rebinding a shared one."""
        if (
            decision.snapshot_version != self.snapshot.version
            or self.snapshot.targets.get(decision.target.target_id) != decision.target
        ):
            raise RoutingRequestError(
                "routing decision belongs to another snapshot", code="snapshot_mismatch", status=409
            )
        handler = self.handlers.get(decision.target.target_id)
        if handler is None:
            raise RoutingRequestError("routing target is unavailable", code="target_unavailable", status=503)
        return handler
