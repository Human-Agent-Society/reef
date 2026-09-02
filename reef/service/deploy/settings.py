"""Translate a ``reef serve`` config into service settings and run them.

``build_parser`` is the ``reef serve`` CLI surface; ``service_settings_from_config``
converts a loaded config's ``reef`` section into a :class:`ServiceSettings`;
``run_service`` is the internal HTTP child's entrypoint (reads ``REEF_CONFIG``,
builds the app via :mod:`reef.service.assembly`, and serves it). The process
orchestrator that starts the surrounding stack lives in
:mod:`reef.service.deploy.orchestrator`.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reef.service.deploy.config import config_value, interpolate_config, load_config

_DESCRIPTION = """reef serve — start a stack from a config.

``reef serve -c <stack>.yaml`` reads the config's ``services``
list and starts every declared process (SGLang, Slime driver, Reef, and so on)
in dependency order. Each service's ``ready`` probe must pass before the next
starts. After all services are up, Reef blocks until SIGTERM/SIGINT; a
watchdog thread detects unexpected exits and tears the stack down.

The Reef HTTP child is an internal service process configured from the same
YAML file. Public startup is always config-driven.

Config overrides:
  Any ``--key value`` pair not recognized as a flag is applied as a config
  override. Bare keys target the ``reef`` section; use dotted notation for
  other sections. Values are YAML-coerced (ints, bools, etc.).

  Examples:
    reef serve --model_path Qwen2.5-1.5B-Instruct
    reef serve -c path/to/local-sglang.yaml --port 9000
    reef serve --training.checkpoint_dir /tmp/ckpt
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reef serve",
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="Config file path (default: reef.yaml or $REEF_CONFIG).",
    )
    return parser


@dataclass(frozen=True)
class ServiceSettings:
    """The HTTP service's settings, translated from a deployment config.

    Recipe-specific config fields (batch sizes, group counts, checkpoint cadence)
    are not fields here: they stay in ``recipe_settings`` — the raw ``reef``
    config section — and each recipe extracts its own via
    ``WeightTrainingRecipe.service_config``, so their defaults live with the recipe.
    """

    recipe: str
    host: str = "0.0.0.0"
    port: int = 8900
    tokens: tuple[str, ...] = ()
    ray_address: str | None = None
    ray_namespace: str = "reef"
    ray_actor_name: str = "reef-train-bridge"
    inference_url: str | None = None
    model_path: str | None = None
    #: The OpenAI-compatible provider no-update recipes proxy to (no ``/v1``
    #: suffix), its credential, and the model name to request from it. The
    #: only place the upstream is named: the HTTP service forwards to it, and
    #: the training side derives the model binding it hands to methods and
    #: evaluation episodes from it. ``upstream_model`` is a provider model
    #: name, unlike ``model_path``, which is local weights for training.
    upstream_url: str | None = None
    upstream_api_key: str | None = None
    upstream_model: str | None = None
    #: The provider's API dialect, ``openai`` (default) or ``anthropic``.
    upstream_api: str = "openai"
    inference_timeout_s: float = 300.0
    train_timeout_s: float | None = None
    inference_backend_factory: str | None = None
    inference_backend_config: Mapping[str, Any] = field(default_factory=dict)
    inference_retry_initial_s: float = 0.05
    inference_retry_max_s: float = 1.0
    inference_retry_timeout_s: float = 300.0
    artifact_repository: str = ".reef/artifacts.git"
    artifact_work_dir: str = ".reef/artifact-work"
    artifact_cache_dir: str = ".reef/artifact-cache"
    agent_record_dir: str = ".reef/agent-record"
    allow_implicit_scenario_creation: bool = True
    #: Deployment-level experiment provider settings, sourced from
    #: ``observability.wandb``.
    wandb_config: Mapping[str, Any] = field(default_factory=dict)
    training_settings: Mapping[str, Any] = field(default_factory=dict)
    #: Optional pre-publication checkpoint evaluator and selection plugin.
    evaluation_settings: Mapping[str, Any] | None = None
    #: The flat ``reef`` config section, interpolated; recipes read their own
    #: config fields from it (see the class docstring).
    recipe_settings: Mapping[str, Any] = field(default_factory=dict)


def _config_service_value(config: Mapping[str, Any], *path: str, default: Any = None, expand: bool = True) -> Any:
    value = config_value(config, *path, default=default, expand=expand)
    if isinstance(value, str):
        return interpolate_config(config, value)
    return value


def _config_service_mapping(config: Mapping[str, Any], *path: str) -> Mapping[str, Any]:
    value: Any = config
    for key in path:
        if not isinstance(value, Mapping):
            return {}
        value = value.get(key)
    return {} if value is None else value


def _config_optional_mapping(config: Mapping[str, Any], *path: str) -> Mapping[str, Any] | None:
    """Return an optional object while rejecting a present non-object value."""
    value: Any = config
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{'.'.join(path)} must be an object")
    return value


def _reef_section(config: Mapping[str, Any]) -> dict[str, Any]:
    """The flat ``reef`` section with per-value interpolation applied."""
    section = config.get("reef")
    if not isinstance(section, Mapping):
        return {}
    return {key: _config_service_value(config, "reef", key) for key in section}


#: ``reef.*`` keys the service consumes under a different field name. The
#: service's vocabulary is ``ServiceSettings``' fields plus these, so the
#: recipe-owned remainder of the section never includes them.
SERVICE_CONFIG_ALIASES: Mapping[str, str] = {"token": "tokens"}


def service_owned_keys() -> frozenset[str]:
    """Every ``reef.*`` key the service layer consumes."""
    non_reef_fields = {"evaluation_settings", "training_settings", "wandb_config"}
    return frozenset(
        settings_field.name
        for settings_field in dataclasses.fields(ServiceSettings)
        if settings_field.name not in non_reef_fields
    ) | frozenset((*SERVICE_CONFIG_ALIASES, "process"))


def _service_tokens(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Accepted Bearer tokens from ``reef.token`` (one) and ``reef.tokens`` (a list).

    Every accepted token is equivalent; listing several lets a caller rotate
    its credential without downtime. Empty entries are dropped so an unset
    ``${REEF_TOKEN}`` does not become a credential.
    """
    tokens: list[str] = []
    single = _config_service_value(config, "reef", "token")
    if isinstance(single, str) and single:
        tokens.append(single)
    listed = _config_service_mapping(config, "reef").get("tokens")
    if listed is not None:
        if isinstance(listed, str) or not isinstance(listed, Sequence):
            raise ValueError("reef.tokens must be a list of strings")
        for item in listed:
            if not isinstance(item, str):
                raise ValueError("reef.tokens must be a list of strings")
            value = interpolate_config(config, item).strip()
            if value:
                tokens.append(value)
    return tuple(dict.fromkeys(tokens))


def service_settings_from_config(config: Mapping[str, Any]) -> ServiceSettings:
    """Translate the config's ``reef`` section into HTTP service settings."""
    selected_recipe = _config_service_value(config, "reef", "recipe")
    if not isinstance(selected_recipe, str) or not selected_recipe:
        raise ValueError("config must declare a non-empty reef.recipe")
    inference_timeout_s = float(_config_service_value(config, "reef", "inference_timeout_s", default="300"))
    train_timeout_raw = _config_service_value(config, "reef", "train_timeout_s")
    train_timeout_s = float(train_timeout_raw) if train_timeout_raw is not None else None
    return ServiceSettings(
        host=_config_service_value(config, "reef", "host", default="0.0.0.0"),
        port=int(_config_service_value(config, "reef", "port", default="8900")),
        tokens=_service_tokens(config),
        recipe=selected_recipe,
        ray_address=_config_service_value(config, "reef", "ray_address"),
        ray_namespace=_config_service_value(config, "reef", "ray_namespace", default="reef"),
        ray_actor_name=_config_service_value(
            config,
            "reef",
            "ray_actor_name",
            default="reef-train-bridge",
        ),
        inference_url=_config_service_value(config, "reef", "inference_url"),
        model_path=_config_service_value(config, "reef", "model_path"),
        upstream_url=_config_service_value(config, "reef", "upstream_url"),
        upstream_api_key=_config_service_value(config, "reef", "upstream_api_key"),
        upstream_model=_config_service_value(config, "reef", "upstream_model"),
        upstream_api=_config_service_value(config, "reef", "upstream_api", default="openai"),
        inference_timeout_s=inference_timeout_s,
        train_timeout_s=train_timeout_s,
        inference_backend_factory=_config_service_value(
            config,
            "reef",
            "inference_backend_factory",
        ),
        inference_backend_config=_config_service_mapping(
            config,
            "reef",
            "inference_backend_config",
        ),
        inference_retry_initial_s=float(
            _config_service_value(config, "reef", "inference_retry_initial_s", default="0.05")
        ),
        inference_retry_max_s=float(_config_service_value(config, "reef", "inference_retry_max_s", default="1")),
        inference_retry_timeout_s=float(
            _config_service_value(config, "reef", "inference_retry_timeout_s", default=inference_timeout_s)
        ),
        artifact_repository=_config_service_value(
            config,
            "reef",
            "artifact_repository",
            default=".reef/artifacts.git",
        ),
        artifact_work_dir=_config_service_value(
            config,
            "reef",
            "artifact_work_dir",
            default=".reef/artifact-work",
        ),
        artifact_cache_dir=_config_service_value(
            config,
            "reef",
            "artifact_cache_dir",
            default=".reef/artifact-cache",
        ),
        agent_record_dir=_config_service_value(
            config,
            "reef",
            "agent_record_dir",
            default=".reef/agent-record",
        ),
        allow_implicit_scenario_creation=bool(
            _config_service_value(config, "reef", "allow_implicit_scenario_creation", default=True)
        ),
        wandb_config=_config_service_mapping(config, "observability", "wandb"),
        training_settings=_config_service_mapping(config, "training"),
        evaluation_settings=_config_optional_mapping(config, "evaluation"),
        recipe_settings=_reef_section(config),
    )


def run_service(config_path: str | Path | None = None) -> int:
    """Run the internal Reef HTTP child from the orchestrator's config."""
    selected_config = config_path or os.environ.get("REEF_CONFIG")
    if selected_config is None:
        raise SystemExit("[reef] ERROR: internal service requires REEF_CONFIG")
    settings = service_settings_from_config(load_config(selected_config))
    from reef.service.assembly import build_app

    app = build_app(settings)
    from aiohttp import web

    web.run_app(app, host=settings.host, port=settings.port)
    return 0
