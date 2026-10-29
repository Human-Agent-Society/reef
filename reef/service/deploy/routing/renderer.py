"""Render the private Semantic Router configuration and container for one managed routing generation.

The router runs unmodified in one container. Only its authenticated HTTP API
is published, on host loopback: REEF selects through ``/api/v1/routing/preview``
and the owner manages configuration. Its built-in gRPC listener and metrics
stay inside the container. Every Scenario becomes a private entrypoint and a
native recipe whose keyword or domain rules and catch-all default each name one
target. A manifest with a domain classifier also gets that classifier's four
files, checked against their digests on the host and mounted read-only.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from reef.service.deploy.routing.settings import RoutingDeploymentSettings
from reef.service.routing import (
    DEFAULT_RULE,
    DomainClassifier,
    RoutingScenario,
    RoutingSnapshot,
    RoutingTarget,
    scenario_entrypoint,
)

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
#: The router container's directory of the domain classifier; each file is bind-mounted into it by itself.
DOMAIN_CLASSIFIER_DIRECTORY = "/models/reef-domain"
#: The native classifier fallback category; preview domain matching uses the threshold separately.
FALLBACK_DOMAIN = "other"
#: The classifier's label files (``config.json`` and ``category_mapping.json``) are read whole up to this size.
LABEL_FILE_LIMIT_BYTES = 1024 * 1024
#: The router finds the classifier at its local path; these keep any Hugging Face client in it offline.
OFFLINE_ENVIRONMENT = ("HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1", "HF_DATASETS_OFFLINE=1")


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
    """Build one native recipe: keyword and domain rules escalate, and the catch-all serves the default target.

    Each rule becomes one signal of its kind, named after the rule, and one decision with that signal as
    its only condition. Earlier rules keep higher priorities, all above zero, whatever their kind; the
    router ranks the condition-free catch-all after every matched rule. Each decision has one static
    target, and REEF reads the matched rule from the preview reply.
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
        decision(
            rule.name,
            len(scenario.rules) - index,
            [{"type": "domain" if rule.domains else "keyword", "name": rule.name}],
            rule.target_id,
        )
        for index, rule in enumerate(scenario.rules)
    ]
    decisions.append(decision(DEFAULT_RULE, 0, [], scenario.default_target))
    routing: dict[str, object] = {"decisions": decisions}
    signals: dict[str, object] = {}
    keyword_rules = [rule for rule in scenario.rules if rule.keywords]
    if keyword_rules:
        signals["keywords"] = [
            {
                "name": rule.name,
                "operator": "AND" if rule.match == "all" else "OR",
                "keywords": list(rule.keywords),
                "case_sensitive": rule.case_sensitive,
            }
            for rule in keyword_rules
        ]
    domain_rules = [rule for rule in scenario.rules if rule.domains]
    if domain_rules:
        signals["domains"] = [{"name": rule.name, "mmlu_categories": list(rule.domains)} for rule in domain_rules]
    if signals:
        routing["signals"] = signals
    return {"name": scenario.name, "routing": routing}


def router_configuration(snapshot: RoutingSnapshot, settings: RoutingDeploymentSettings) -> dict[str, object]:
    """The router's native configuration; it names token variables, never token values.

    Every learned module is disabled, except the domain classifier when the snapshot has one: it then runs on
    CPU from ``DOMAIN_CLASSIFIER_DIRECTORY`` with the snapshot's threshold. Below that threshold, preview
    matches no domain rule; other matching rules or the Scenario default can still select a target.
    """
    enabled = {name: target for name, target in snapshot.targets.items() if target.enabled}
    domain_module: dict[str, object] = {"model_ref": "", "model_id": "", "category_mapping_path": ""}
    model_catalog: dict[str, object] = {}
    if snapshot.domain_classifier is not None:
        # The native configuration shape that loaded this classifier from a local path in the pinned router.
        model_catalog["system"] = {"domain_classifier": DOMAIN_CLASSIFIER_DIRECTORY}
        domain_module = {
            "enabled": True,
            "model_ref": "domain_classifier",
            "model_id": DOMAIN_CLASSIFIER_DIRECTORY,
            "category_mapping_path": DOMAIN_CLASSIFIER_DIRECTORY + "/category_mapping.json",
            "use_cpu": True,
            "threshold": snapshot.domain_classifier.threshold,
            "fallback_category": FALLBACK_DOMAIN,
        }
    model_catalog["embeddings"] = {
        "semantic": {
            "mmbert_model_path": "",
            "qwen3_model_path": "",
            "gemma_model_path": "",
            "bert_model_path": "",
            "multimodal_model_path": "",
        }
    }
    model_catalog["modules"] = {
        "prompt_guard": {"enabled": False, "model_ref": "", "model_id": ""},
        "classifier": {"domain": domain_module, "pii": {"model_ref": "", "model_id": "", "pii_mapping_path": ""}},
        "feedback_detector": {"enabled": False, "model_ref": "", "model_id": ""},
    }
    # Each model is metadata only: previews select a model, and Reef calls every provider itself.
    providers = [
        {"name": name, "provider_model_id": target.provider_model, "api_format": "openai"}
        for name, target in enabled.items()
    ]
    return {
        "version": "v0.3",
        # Without an inference listener the router needs no backend per model, and its hot reload compares no
        # backend topology, so models can be added, removed or changed in place.
        "listeners": [],
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
            "model_catalog": model_catalog,
        },
    }


def read_classifier_file(directory: str, name: str, digest: str) -> bytes:
    """Hash one classifier file; return its bytes for ``config.json`` and ``category_mapping.json``, else ``b""``.

    The file must be a regular file, opened without following a symlink in its own name; ``O_NONBLOCK`` keeps a
    FIFO from blocking the open and does not change reading a regular file. The returned bytes are the hashed
    bytes, so the labels read from them belong to the digest.

    Raises:
        ValueError: The file cannot be opened or read, is not a regular file, is a label file larger than
            ``LABEL_FILE_LIMIT_BYTES``, or does not have ``digest``.
    """
    field_name = f"deployment.domain_classifier.files.{name}"
    label_file = name in ("config.json", "category_mapping.json")
    sha256 = hashlib.sha256()
    content = bytearray()
    try:
        descriptor = os.open(os.path.join(directory, name), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        # Checked before ``fdopen``, which refuses a directory with a less exact error.
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise ValueError(f"{field_name} must be a regular file")
        with os.fdopen(descriptor, "rb") as stream:
            while block := stream.read(1024 * 1024):
                sha256.update(block)
                if label_file:
                    content += block
                    if len(content) > LABEL_FILE_LIMIT_BYTES:
                        raise ValueError(f"{field_name} is larger than {LABEL_FILE_LIMIT_BYTES} bytes")
    except OSError as exc:
        raise ValueError(f"{field_name} cannot be read as a regular file without a symlink: {exc.strerror}") from exc
    if "sha256:" + sha256.hexdigest() != digest:
        raise ValueError(f"{field_name} does not match its sha256 digest")
    return bytes(content)


def classifier_labels(config: bytes, mapping: bytes) -> tuple[str, ...]:
    """The classifier's labels in index order, once its ``config.json`` and ``category_mapping.json`` agree.

    ``config.json`` maps the indexes ``"0"`` to ``"n-1"`` to distinct labels in ``id2label``.
    ``category_mapping.json`` lists the same labels in the same order in ``idx_to_category``, and
    ``category_to_idx`` is its inverse, as the router requires. The labels include ``FALLBACK_DOMAIN``.

    Raises:
        ValueError: Either file is not that form, or they disagree.
    """
    name = "deployment.domain_classifier"

    def indexed(value: object, field_name: str) -> tuple[str, ...]:
        if (
            not isinstance(value, dict)
            or not value
            or set(value) != {str(index) for index in range(len(value))}
            or not all(isinstance(label, str) and label for label in value.values())
        ):
            raise ValueError(f"{name} {field_name} must map the indexes 0 to n-1 to label strings")
        return tuple(value[str(index)] for index in range(len(value)))

    try:
        config_value, mapping_value = json.loads(config), json.loads(mapping)
    except (ValueError, RecursionError) as exc:
        raise ValueError(f"{name} config.json and category_mapping.json must be valid, bounded-depth JSON") from exc
    if not isinstance(config_value, dict) or not isinstance(mapping_value, dict):
        raise ValueError(f"{name} config.json and category_mapping.json must be JSON objects")
    labels = indexed(config_value.get("id2label"), "config.json id2label")
    if len(set(labels)) != len(labels):
        raise ValueError(f"{name} config.json id2label repeats a label")
    if indexed(mapping_value.get("idx_to_category"), "category_mapping.json idx_to_category") != labels:
        raise ValueError(f"{name} category_mapping.json must list the labels of config.json in the same order")
    inverse = mapping_value.get("category_to_idx")
    if (
        not isinstance(inverse, dict)
        or any(type(index) is not int for index in inverse.values())
        or inverse != {label: index for index, label in enumerate(labels)}
    ):
        raise ValueError(f"{name} category_mapping.json category_to_idx must invert idx_to_category")
    if FALLBACK_DOMAIN not in labels:
        raise ValueError(f"{name} must have the configured fallback label {FALLBACK_DOMAIN!r}")
    return labels


def check_domain_classifier(snapshot: RoutingSnapshot) -> None:
    """Check on this host the snapshot's classifier files and the labels its domain rules name.

    The model directory must exist and contain no symlink. Each file must have its digest, the label files must
    agree (``classifier_labels``), and every label a domain rule names must be one of the classifier's labels.
    A domain rule's name must not equal a label, ignoring case: the router reports a predicted label that no
    rule names under that label itself, which would then match the rule with that name. A snapshot without a
    classifier passes. The digests are checked when a generation is rendered, not afterwards: a running
    router reads files the host can still change.

    Raises:
        ValueError: A file, the directory or a rule fails these checks.
    """
    classifier: DomainClassifier | None = snapshot.domain_classifier
    if classifier is None:
        return
    directory = classifier.model_directory
    if os.path.realpath(directory) != directory or not os.path.isdir(directory):
        raise ValueError("deployment.domain_classifier.model_directory must be an existing directory without symlinks")
    contents = {
        name: read_classifier_file(directory, name, digest) for name, digest in sorted(classifier.files.items())
    }
    labels = classifier_labels(contents["config.json"], contents["category_mapping.json"])
    folded = {label.casefold() for label in labels}
    for scenario in snapshot.scenarios.values():
        for rule in scenario.rules:
            name = f"scenarios.{scenario.name}.rules.{rule.name}"
            unknown = [label for label in rule.domains if label not in labels]
            if unknown:
                raise ValueError(f"{name}.domains names labels the classifier does not have: {unknown}")
            if rule.domains and rule.name.casefold() in folded:
                raise ValueError(f"{name} is named like a classifier label; give the domain rule another name")


def generate(manifest_path: Path, output: Path, managed: ManagedRoutingRender) -> RoutingSnapshot:
    """Render one frozen generation into a new or empty private directory; containers stay with the caller.

    Token values never enter this renderer, its configuration digest or any generated file. A domain
    classifier is checked (``check_domain_classifier``) before any file is written.
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
    check_domain_classifier(snapshot)
    router_config = router_configuration(snapshot, settings)
    labels = {
        "io.reef.routing.deployment": managed.deployment_id,
        "io.reef.routing.owner": managed.owner_id,
        "io.reef.routing.generation": managed.generation,
    }
    # The management API writes its configuration by atomic rename in this private directory.
    volumes: list[object] = [f"./{ROUTER_CONFIG_DIRECTORY}:/app/config"]
    # Compose passes the owner's values; no file or argument contains them.
    environment = [MANAGEMENT_TOKEN_ENV, SELECTION_TOKEN_ENV]
    if snapshot.domain_classifier is not None:
        # Mount only checked files, read-only. Escape literal dollars against Compose interpolation.
        volumes += [
            {
                "type": "bind",
                "source": os.path.join(snapshot.domain_classifier.model_directory, name).replace("$", "$$"),
                "target": f"{DOMAIN_CLASSIFIER_DIRECTORY}/{name}",
                "read_only": True,
                "bind": {"create_host_path": False},
            }
            for name in sorted(snapshot.domain_classifier.files)
        ]
        environment += OFFLINE_ENVIRONMENT
    router_service: dict[str, object] = {
        "image": manifest["deployment"]["semantic_router_image"],
        "labels": labels,
        # Only the HTTP API, only on host loopback. The router's gRPC and metrics ports stay unpublished.
        "ports": [f"127.0.0.1:{settings.sr_management_port}:{ROUTER_HTTP_PORT}"],
        "volumes": volumes,
        "command": ["/app/config/config.yaml"],
        "environment": environment,
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
