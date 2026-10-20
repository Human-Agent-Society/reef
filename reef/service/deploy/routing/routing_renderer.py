"""Render the private Semantic Router configuration and container for one managed routing generation.

The router runs unmodified in one container. Only its authenticated HTTP API
is published, on host loopback: REEF selects through ``/api/v1/routing/preview``
and the owner manages configuration. Its built-in gRPC listener and metrics
stay inside the container. Every Scenario becomes a private entrypoint and a
native recipe whose keyword rules and catch-all default each name one target.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from reef.service.deploy.routing.routing_config import RoutingDeploymentSettings
from reef.service.routing import DEFAULT_RULE, RoutingScenario, RoutingSnapshot, RoutingTarget, scenario_entrypoint

#: Container variables holding the router's bearer tokens; Compose passes the values from the owner's environment.
MANAGEMENT_TOKEN_ENV = "REEF_SR_MANAGEMENT_TOKEN"
SELECTION_TOKEN_ENV = "REEF_SR_SELECTION_TOKEN"
#: The owner's role: configuration and readiness only.
MANAGEMENT_ROLE = "reef-policy"
#: REEF's role: selection previews only.
SELECTION_ROLE = "reef-selection"
#: Writable private directory mounted as the router's configuration directory.
ROUTER_CONFIG_DIRECTORY = "router-config"
#: The router's HTTP API inside its container (start-router.sh enables it there).
ROUTER_HTTP_PORT = 8080
#: SR requires a backend address for every model, although previews only select a model.
#: This unused loopback address satisfies the configuration schema; tests check for backend calls.
UNUSED_BACKEND = "127.0.0.1:9"


@dataclass(frozen=True)
class ManagedRoutingRender:
    """Identity supplied by the deployment controller for one generation's resources.

    Values: ``deployment_id`` equals ``settings.deployment_id``; ``owner_id`` and ``generation`` are non-empty
    strings, used only as container and network labels. They need not be registry identities: candidate
    validation renders with the placeholder labels ``"validation"``.
    Purpose: the controller makes it for ``generate``, which writes the generation's files with these
    labels. It owns no container and holds no credential.
    """

    settings: RoutingDeploymentSettings
    deployment_id: str
    owner_id: str
    generation: str

    def __post_init__(self) -> None:
        if self.settings.deployment_id != self.deployment_id:
            raise ValueError("managed renderer identity must match its deployment settings")
        if any(not isinstance(label, str) or not label for label in (self.owner_id, self.generation)):
            raise ValueError("managed renderer owner and generation labels must be non-empty strings")


def write_private_document(path: Path, document: object) -> None:
    """Create private YAML without following a preexisting symlink."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        yaml.safe_dump(document, stream, sort_keys=False)


def scenario_recipe(scenario: RoutingScenario, enabled: Mapping[str, RoutingTarget]) -> dict[str, object]:
    """Build one native recipe: keyword rules escalate, and the catch-all serves the default target.

    Earlier rules keep higher priorities, all above zero; the router ranks the
    condition-free catch-all after every matched rule. Each decision has one
    static target, and REEF reads the matched rule from the preview reply.
    """

    def decision(name: str, priority: int, conditions: list[dict[str, str]], target_id: str) -> dict[str, object]:
        if target_id not in scenario.target_ids or target_id not in enabled:
            raise ValueError(f"scenario {scenario.name!r} decision {name!r} must reference its enabled targets")
        return {
            "name": name,
            "priority": priority,
            "rules": {"operator": "AND", "conditions": conditions},
            "modelRefs": [{"model": target_id, "use_reasoning": False}],
            "algorithm": {"type": "static"},
        }

    decisions = [
        decision(rule.name, len(scenario.rules) - index, [{"type": "keyword", "name": rule.name}], rule.target_id)
        for index, rule in enumerate(scenario.rules)
    ]
    decisions.append(decision(DEFAULT_RULE, 0, [], scenario.default_target))
    routing: dict[str, object] = {"decisions": decisions}
    if scenario.rules:
        routing["signals"] = {
            "keywords": [
                {"name": rule.name, "operator": "OR", "keywords": list(rule.keywords), "case_sensitive": False}
                for rule in scenario.rules
            ]
        }
    return {"name": scenario.name, "routing": routing}


def router_configuration(snapshot: RoutingSnapshot, settings: RoutingDeploymentSettings) -> dict[str, object]:
    """The router's native configuration; it names token variables, never token values."""
    enabled = {name: target for name, target in snapshot.targets.items() if target.enabled}
    providers = [
        {
            "name": name,
            "provider_model_id": target.provider_model,
            "api_format": "openai",
            "backend_refs": [
                {"name": "unused-" + str(index), "endpoint": UNUSED_BACKEND, "protocol": "http", "provider": "vllm"}
            ],
        }
        for index, (name, target) in enumerate(enabled.items())
    ]
    return {
        "version": "v0.3",
        # The router's own inference listener stays on container loopback and is never published.
        "listeners": [{"name": "reef-unused", "address": "127.0.0.1", "port": 18899}],
        # No deployment-wide default model: a request outside every entrypoint fails instead of using
        # another Scenario's model.
        "providers": {"models": providers},
        "entrypoints": [{"model_names": [scenario_entrypoint(name)], "recipe": name} for name in snapshot.scenarios],
        "recipes": [scenario_recipe(scenario, enabled) for scenario in snapshot.scenarios.values()],
        "global": {
            "router": {
                # An explicit list replaces the router's built-in automatic names.
                "auto_model_names": [snapshot.automatic_model],
                "auto_model_name": snapshot.automatic_model,
                "clear_route_cache": True,
                "skip_processing": {"enabled": False},
                "learning": {"enabled": False},
                "fallback": {"enabled": False},
                "model_selection": {"enabled": True, "method": "static"},
            },
            "stores": {"response_cache": {"enabled": False}},
            "services": {
                "api": {
                    "routing_preview": {
                        "request_timeout_seconds": settings.selection_timeout_seconds,
                        "max_concurrency": settings.preview_max_concurrency,
                    }
                },
                # A container-wide bind is published only on host loopback; the router refuses it unless
                # bearer authentication names its tokens by variable.
                "management_api": {
                    "bind_address": "0.0.0.0",
                    "port": ROUTER_HTTP_PORT,
                    "remote_exposure": True,
                    "auth": {
                        "mode": "bearer",
                        "tokens": [
                            {"env": MANAGEMENT_TOKEN_ENV, "role": MANAGEMENT_ROLE},
                            {"env": SELECTION_TOKEN_ENV, "role": SELECTION_ROLE},
                        ],
                        # Explicit roles replace the router's built-in ones.
                        "roles": {
                            MANAGEMENT_ROLE: ["config.read", "config.write", "ready.read"],
                            SELECTION_ROLE: ["classify.invoke"],
                        },
                    },
                },
                "observability": {"tracing": {"enabled": False}},
            },
            "model_catalog": {
                "embeddings": {
                    "semantic": {
                        "mmbert_model_path": "",
                        "qwen3_model_path": "",
                        "gemma_model_path": "",
                        "bert_model_path": "",
                        "multimodal_model_path": "",
                    }
                },
                "modules": {
                    "prompt_guard": {"enabled": False, "model_ref": "", "model_id": ""},
                    "classifier": {
                        "domain": {"model_ref": "", "model_id": "", "category_mapping_path": ""},
                        "pii": {"model_ref": "", "model_id": "", "pii_mapping_path": ""},
                    },
                    "feedback_detector": {"enabled": False, "model_ref": "", "model_id": ""},
                },
            },
        },
    }


def generate(manifest_path: Path, output: Path, managed: ManagedRoutingRender) -> RoutingSnapshot:
    """Render one frozen generation into a new or empty private directory; containers stay with the caller.

    Token values never enter this renderer, its configuration digest or any generated file.
    """
    settings = managed.settings
    if output.is_symlink():
        raise ValueError("routing output directory must not be a symlink")
    if output.exists() and any(output.iterdir()):
        raise ValueError("managed rendering requires a new or empty generation directory")
    try:
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        # A configuration error like any invalid manifest: a policy update rejects it and keeps serving.
        raise ValueError(f"invalid routing manifest YAML in {manifest_path}") from exc
    RoutingSnapshot.from_mapping(manifest)
    configuration = {
        "renderer": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sr_management_port": settings.sr_management_port,
        "selection_timeout_seconds": settings.selection_timeout_seconds,
        "preview_max_concurrency": settings.preview_max_concurrency,
        "sr_cpus": settings.sr_cpus,
        "sr_memory_bytes": settings.sr_memory_bytes,
        "ready_timeout_seconds": settings.ready_timeout_seconds,
    }
    encoded_configuration = json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()
    manifest["deployment"]["configuration_digest"] = "sha256:" + hashlib.sha256(encoded_configuration).hexdigest()
    snapshot = RoutingSnapshot.from_mapping(manifest)
    router_config = router_configuration(snapshot, settings)
    labels = {
        "io.reef.routing.deployment": managed.deployment_id,
        "io.reef.routing.owner": managed.owner_id,
        "io.reef.routing.generation": managed.generation,
    }
    router_service: dict[str, object] = {
        "image": manifest["deployment"]["semantic_router_image"],
        "labels": labels,
        # Only the HTTP API, only on host loopback. The router's gRPC and metrics ports stay unpublished.
        "ports": [f"127.0.0.1:{settings.sr_management_port}:{ROUTER_HTTP_PORT}"],
        # The management API writes its configuration by atomic rename in this private directory.
        "volumes": [f"./{ROUTER_CONFIG_DIRECTORY}:/app/config"],
        "command": ["/app/config/config.yaml"],
        # Compose passes the owner's values; no file or argument contains them.
        "environment": [MANAGEMENT_TOKEN_ENV, SELECTION_TOKEN_ENV],
        "restart": "no",
        "cpus": settings.sr_cpus,
        "mem_limit": settings.sr_memory_bytes,
        "healthcheck": {
            # /ready requires ready.read. printf is a shell builtin and curl reads the header from stdin,
            # so the token is not in the healthcheck definition or any argument list.
            "test": [
                "CMD-SHELL",
                (
                    f"printf 'Authorization: Bearer %s\\n' \"$${MANAGEMENT_TOKEN_ENV}\" | "
                    f"curl --fail --silent --max-time 1 -H @- http://127.0.0.1:{ROUTER_HTTP_PORT}/ready"
                ),
            ],
            "interval": "1s",
            "timeout": "1s",
            "retries": 3,
            "start_period": f"{settings.ready_timeout_seconds}s",
        },
    }
    compose = {"services": {"semantic-router": router_service}, "networks": {"default": {"labels": labels}}}
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    output.chmod(0o700)
    write_private_document(output / "manifest.yaml", manifest)
    write_private_document(output / "router.yaml", router_config)
    write_private_document(output / "compose.yaml", compose)
    router_directory = output / ROUTER_CONFIG_DIRECTORY
    router_directory.mkdir(mode=0o700)
    write_private_document(router_directory / "config.yaml", router_config)
    return snapshot
