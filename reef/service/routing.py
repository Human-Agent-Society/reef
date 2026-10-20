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
    """A routed request violates the deployment's selection contract."""

    def __init__(self, message: str, *, code: str = "invalid_route", status: int = 400) -> None:
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
    """Read a non-empty manifest string without silently changing identity."""
    if not isinstance(value, str) or not value or value != value.strip() or any(ord(char) < 32 for char in value):
        raise ValueError(f"{name} must be a non-empty string without surrounding whitespace or control characters")
    return value


def manifest_strings(value: object, name: str) -> tuple[str, ...]:
    """Read a non-empty list of distinct manifest strings."""
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    items = tuple(manifest_string(item, name) for item in value)
    if len(set(items)) != len(items):
        raise ValueError(f"{name} must not repeat a value")
    return items


@dataclass(frozen=True)
class RoutingTarget:
    """An immutable external deployment selected by its logical ID."""

    target_id: str
    provider_model: str
    revision: str
    url: str
    role: str | None = None
    api_key_env: str | None = None
    enabled: bool = True


@dataclass(frozen=True)
class RoutingRule:
    """A keyword rule that sends matching automatic requests to one target."""

    name: str
    keywords: tuple[str, ...]
    target_id: str


@dataclass(frozen=True)
class RoutingScenario:
    """The targets one Scenario may use and the rules that choose among them."""

    name: str
    default_target: str
    target_ids: tuple[str, ...]
    rules: tuple[RoutingRule, ...] = ()

    def target_for_rule(self, rule_name: str) -> str | None:
        """Return the target an automatic selection rule chooses, if the rule exists."""
        if rule_name == DEFAULT_RULE:
            return self.default_target
        return next((rule.target_id for rule in self.rules if rule.name == rule_name), None)


def parse_target(target_id: str, value: object) -> RoutingTarget:
    """Validate one external deployment declaration."""
    target = manifest_mapping(
        value,
        f"targets.{target_id}",
        {"provider_model", "revision", "url", "role", "api_key_env", "enabled"},
    )
    provider_model = manifest_string(target.get("provider_model"), f"targets.{target_id}.provider_model")
    revision = manifest_string(target.get("revision"), f"targets.{target_id}.revision")
    url = manifest_string(target.get("url"), f"targets.{target_id}.url")
    parsed_url = urlsplit(url)
    if (
        parsed_url.scheme not in ("http", "https")
        or not parsed_url.hostname
        or parsed_url.username is not None
        or parsed_url.password is not None
        or parsed_url.query
        or parsed_url.fragment
    ):
        raise ValueError(f"targets.{target_id}.url must be an HTTP URL without credentials, query, or fragment")
    if parsed_url.port is not None and parsed_url.port < 1:
        raise ValueError(f"targets.{target_id}.url must use a valid port")
    role = manifest_string(target["role"], f"targets.{target_id}.role") if "role" in target else None
    api_key_env = (
        manifest_string(target["api_key_env"], f"targets.{target_id}.api_key_env") if "api_key_env" in target else None
    )
    if api_key_env is not None and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", api_key_env):
        raise ValueError(f"targets.{target_id}.api_key_env must name an environment variable")
    enabled = target.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError(f"targets.{target_id}.enabled must be a boolean")
    return RoutingTarget(target_id, provider_model, revision, url, role, api_key_env, enabled)


def parse_scenario(name: str, value: object, targets: Mapping[str, RoutingTarget]) -> RoutingScenario:
    """Validate one Scenario's target list and its automatic selection rules."""
    if name in RESERVED_SCENARIOS:
        raise ValueError(f"scenario name {name!r} is reserved by the Semantic Router")
    scenario = manifest_mapping(value, f"scenarios.{name}", {"default_target", "targets", "rules"})
    target_ids = manifest_strings(scenario.get("targets"), f"scenarios.{name}.targets")
    for target_id in target_ids:
        if target_id not in targets:
            raise ValueError(f"scenarios.{name}.targets references unknown target {target_id!r}")
    default_target = manifest_string(scenario.get("default_target"), f"scenarios.{name}.default_target")
    if default_target not in target_ids or not targets[default_target].enabled:
        raise ValueError(f"scenarios.{name}.default_target must identify one of its enabled targets")
    raw_rules = scenario.get("rules", [])
    if not isinstance(raw_rules, list):
        raise ValueError(f"scenarios.{name}.rules must be a list")
    rules: list[RoutingRule] = []
    for raw_rule in raw_rules:
        rule = manifest_mapping(raw_rule, f"scenarios.{name}.rules[]", {"name", "keywords", "target"})
        rule_name = manifest_string(rule.get("name"), f"scenarios.{name}.rules[].name")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", rule_name) or rule_name == DEFAULT_RULE:
            raise ValueError(f"scenarios.{name}.rules[].name must be a short identifier other than {DEFAULT_RULE!r}")
        if any(existing.name == rule_name for existing in rules):
            raise ValueError(f"scenarios.{name}.rules repeats rule {rule_name!r}")
        keywords = manifest_strings(rule.get("keywords"), f"scenarios.{name}.rules.{rule_name}.keywords")
        target_id = manifest_string(rule.get("target"), f"scenarios.{name}.rules.{rule_name}.target")
        if target_id not in target_ids or not targets[target_id].enabled:
            raise ValueError(f"scenarios.{name}.rules.{rule_name}.target must identify one of its enabled targets")
        rules.append(RoutingRule(rule_name, keywords, target_id))
    return RoutingScenario(name, default_target, target_ids, tuple(rules))


@dataclass(frozen=True)
class RoutingSnapshot:
    """A deployment manifest whose digest excludes resolved credentials."""

    automatic_model: str
    targets: Mapping[str, RoutingTarget]
    scenarios: Mapping[str, RoutingScenario]
    version: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "targets", MappingProxyType(dict(self.targets)))
        object.__setattr__(self, "scenarios", MappingProxyType(dict(self.scenarios)))

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
        """Validate schema version 2 and hash its canonical JSON representation."""
        manifest = manifest_mapping(
            value,
            "routing manifest",
            {"schema_version", "automatic_model", "targets", "scenarios", "deployment"},
        )
        if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 2:
            raise ValueError("routing manifest schema_version must be 2")
        automatic_model = manifest_string(manifest.get("automatic_model"), "automatic_model")
        if reserved_model_name(automatic_model):
            raise ValueError(f"automatic_model must not start with the reserved prefix {ENTRYPOINT_PREFIX!r}")
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
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", configuration_digest):
                raise ValueError("deployment.configuration_digest must be a sha256 digest")
        raw_targets = manifest.get("targets")
        if not isinstance(raw_targets, Mapping) or not raw_targets:
            raise ValueError("targets must be a non-empty object")
        targets: dict[str, RoutingTarget] = {}
        for raw_id, raw_target in raw_targets.items():
            target_id = manifest_string(raw_id, "target ID")
            if target_id == automatic_model:
                raise ValueError("automatic_model must not also identify a target")
            if reserved_model_name(target_id):
                raise ValueError(
                    f"target ID {target_id!r} must not start with the reserved prefix {ENTRYPOINT_PREFIX!r}"
                )
            targets[target_id] = parse_target(target_id, raw_target)
        raw_scenarios = manifest.get("scenarios")
        if not isinstance(raw_scenarios, Mapping) or not raw_scenarios:
            raise ValueError("scenarios must be a non-empty object")
        scenarios: dict[str, RoutingScenario] = {}
        for raw_name, raw_scenario in raw_scenarios.items():
            name = manifest_string(raw_name, "scenario name")
            scenarios[name] = parse_scenario(name, raw_scenario, targets)
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
    """The validated external target and policy snapshot for one request."""

    target: RoutingTarget
    snapshot_version: str
    scenario: str
    original_selector: str
    reason: Literal["default", "rule", "explicit_pin"]
    rule_name: str | None = None
    #: The Semantic Router runtime that selected an automatic request; None for an explicit pin.
    router_config_hash: str | None = None

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

    A request captures one adapter when it is admitted and keeps it for selection, its provider call and
    every retry; publishing a policy replaces the whole adapter.
    """

    snapshot: RoutingSnapshot
    #: The verified Semantic Router runtime whose previews select for this policy. None until the owner
    #: verifies one: a managed child starts this way, closed, and no automatic request can select.
    router_config_hash: str | None
    handlers: Mapping[str, HttpInferenceHandler]
    api_keys: Mapping[str, str] = field(default_factory=dict, repr=False, compare=False)
    readiness_timeout_s: float = MAX_READINESS_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if self.router_config_hash is not None and not is_router_config_hash(self.router_config_hash):
            raise ValueError("router_config_hash must be 64 lowercase hexadecimal characters")
        object.__setattr__(self, "handlers", MappingProxyType(dict(self.handlers)))
        object.__setattr__(self, "api_keys", MappingProxyType(dict(self.api_keys)))

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
