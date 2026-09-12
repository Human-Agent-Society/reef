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
import copy
import dataclasses
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from reef.service.cors import console_origins
from reef.service.deploy.arguments import (
    ConfigArgument,
    ConfigArgumentParser,
    config_arguments,
    config_metadata,
    config_option,
)
from reef.service.deploy.config import config_value, interpolate_config, load_config
from reef.storage.postgres import postgres_url, validate_postgres_schema
from reef.storage.records import RecordRetention

_DESCRIPTION = """reef serve — start a stack from a config.

``reef serve -c <stack>.yaml`` reads the config's ``services``
list and starts every declared process (SGLang, Slime driver, Reef, and so on)
in dependency order. Each service's ``ready`` probe must pass before the next
starts. After all services are up, Reef blocks until SIGTERM/SIGINT; a
watchdog thread detects unexpected exits and tears the stack down.

The Reef HTTP child is an internal service process configured from the same
YAML file. Public startup is always config-driven.

Config overrides:
  Public settings below share type conversion with YAML. Explicit CLI
  values override YAML; omitted settings use the dataclass defaults.
  Both --upstream-model and legacy --upstream_model spellings work.
  Lists and objects take one quoted JSON/YAML value, including [] or {}.
  Recipe and custom-stack overrides retain their existing YAML coercion:
  bare keys target ``reef``; dotted keys target other sections.

  Examples:
    reef serve -c stack.yaml --model-path Qwen/Qwen2.5-1.5B-Instruct
    reef serve -c path/to/local-sglang.yaml --port 9000
    reef serve --training.checkpoint_dir /tmp/ckpt
"""


def build_parser(*, service_arguments: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reef serve",
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="Config file path, relative to the working directory (default: $REEF_CONFIG or ./reef.yaml).",
    )
    if service_arguments:
        for argument in service_config_arguments():
            argument.add_to(parser)
    return parser


@dataclass(frozen=True)
class ServiceSettings:
    """The HTTP service's settings, translated from a deployment config.

    Recipe-specific config fields (batch sizes, group counts, checkpoint cadence)
    are not fields here: they stay in ``recipe_settings`` — the raw ``reef``
    config section — and each recipe extracts its own via
    ``WeightTrainingRecipe.service_config``, so their defaults live with the recipe.
    """

    recipe: str = config_option(help="Recipe implementation or named deployment preset (the launcher owns --recipe).")
    host: str = config_option("0.0.0.0", help="HTTP bind address.")
    port: int = config_option(8900, help="HTTP bind port.")
    tokens: tuple[str, ...] = config_option((), help="Accepted bearer tokens as a JSON/YAML list.")
    console_origins: tuple[str, ...] = config_option((), help="Allowed console origins as a JSON/YAML list.")
    ray_address: str | None = config_option(None, help="Ray cluster address.")
    ray_namespace: str = config_option("reef", help="Ray namespace for the training bridge.")
    ray_actor_name: str = config_option("reef-train-bridge", help="Training bridge actor name.")
    inference_url: str | None = config_option(None, help="Local inference endpoint.")
    model_path: str | None = config_option(None, help="Local model directory or Hugging Face repository ID.")
    #: The OpenAI-compatible provider no-update recipes proxy to (no ``/v1``
    #: suffix), its credential, and the model name to request from it. The
    #: only place the upstream is named: the HTTP service forwards to it, and
    #: the training side derives the model binding it hands to methods and
    #: evaluation episodes from it. ``upstream_model`` is a provider model
    #: name, unlike ``model_path``, which is local weights for training.
    upstream_url: str | None = config_option(None, help="Upstream provider base URL.")
    upstream_api_key: str | None = config_option(None, help="Upstream provider credential.")
    upstream_model: str | None = config_option(None, help="Model name requested from the upstream provider.")
    #: The provider's API dialect: ``openai`` (default), ``responses``, or ``anthropic``.
    upstream_api: str = config_option("openai", help="Provider API dialect.")
    inference_timeout_s: float = config_option(300.0, help="Inference request timeout in seconds.")
    train_timeout_s: float | None = config_option(None, help="Training request timeout in seconds.")
    inference_backend_factory: str | None = config_option(None, help="Dotted inference backend factory.")
    inference_backend_config: Mapping[str, Any] = field(
        default_factory=dict, metadata=config_metadata("Inference backend options as a JSON/YAML object.")
    )
    inference_retry_initial_s: float = config_option(0.05, help="Initial inference retry delay in seconds.")
    inference_retry_max_s: float = config_option(1.0, help="Maximum inference retry delay in seconds.")
    inference_retry_timeout_s: float = config_option(
        300.0, help="Retry deadline in seconds; defaults to the inference timeout."
    )
    artifact_repository: str = config_option(".reef/artifacts.git", help="Artifact repository location.")
    artifact_work_dir: str = config_option(".reef/artifact-work", help="Artifact working directory.")
    artifact_cache_dir: str = config_option(".reef/artifact-cache", help="Artifact cache directory.")
    agent_record_dir: str = config_option(".reef/agent-record", help="Agent record directory.")
    record_backend: str = config_option("sqlite", help="Record storage backend.")
    record_database_url: str | None = field(
        default=None, repr=False, metadata=config_metadata("PostgreSQL connection URL.")
    )
    record_database_schema: str = config_option("reef_records", help="PostgreSQL schema.")
    agent_record_retention_days: float = config_option(7.0, help="Record retention in days.")
    agent_record_retention_max_bytes: int = config_option(20 * 1024**3, help="Maximum retained record bytes.")
    allow_implicit_scenario_creation: bool = config_option(True, help="Allow requests to create scenarios implicitly.")
    #: Deployment-level experiment provider settings, sourced from
    #: ``observability.wandb``.
    wandb_config: Mapping[str, Any] = field(
        default_factory=dict,
        metadata=config_metadata("W&B settings as a JSON/YAML object.", path=("observability", "wandb")),
    )
    training_settings: Mapping[str, Any] = field(
        default_factory=dict, metadata=config_metadata("Training settings as a JSON/YAML object.", path=("training",))
    )
    #: Optional pre-publication checkpoint evaluator and selection plugin.
    evaluation_settings: Mapping[str, Any] | None = field(
        default=None,
        metadata=config_metadata("Candidate evaluation settings as a JSON/YAML object.", path=("evaluation",)),
    )
    #: The flat ``reef`` config section, interpolated; recipes read their own
    #: config fields from it (see the class docstring).
    recipe_settings: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        RecordRetention(self.agent_record_retention_days, self.agent_record_retention_max_bytes)
        if self.record_backend not in {"sqlite", "postgres"}:
            raise ValueError("reef.record_backend must be sqlite or postgres")
        if self.record_backend == "postgres":
            if not isinstance(self.record_database_url, str) or not self.record_database_url:
                raise ValueError("reef.record_database_url is required for the postgres backend")
            postgres_url(self.record_database_url)
            validate_postgres_schema(self.record_database_schema)
        elif self.record_database_url is not None or self.record_database_schema != "reef_records":
            raise ValueError("record_database_url and record_database_schema require reef.record_backend: postgres")


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
    ) | frozenset(SERVICE_CONFIG_ALIASES)


@lru_cache(maxsize=1)
def service_config_arguments() -> tuple[ConfigArgument, ...]:
    """Public settings derive their types and defaults from ServiceSettings."""
    return (
        *config_arguments(ServiceSettings),
        ConfigArgument("token", ("reef", "token"), "str", True, None, "One accepted bearer token."),
    )


def service_override(key: str, value: str) -> tuple[ConfigArgument, str] | None:
    """Identify a declared option, retaining legacy names and dotted paths."""
    flag = f"--{key}"
    for argument in service_config_arguments():
        if flag in argument.flags:
            return argument, value
        if flag in argument.negative_flags:
            return argument, "false"
    return None


def _argument_value(config: Mapping[str, Any], argument: ConfigArgument) -> Any:
    node: Any = config
    for key in argument.path:
        if not isinstance(node, Mapping):
            raise ValueError(f"{'.'.join(argument.path[:-1])} must be an object")
        node = node.get(key)
        if node is None:
            return None
    if isinstance(node, str):
        return os.path.expanduser(interpolate_config(config, node.strip())) if node.strip() else None
    if argument.kind == "strings" and isinstance(node, (list, tuple)):
        return [interpolate_config(config, item) if isinstance(item, str) else item for item in node]
    return node


def parse_service_arguments(
    config: Mapping[str, Any], *, cli_paths: frozenset[tuple[str, ...]] = frozenset()
) -> dict[str, Any]:
    """Parse YAML arguments followed by explicit CLI values with one parser.

    The caller has already replaced overridden environment references and
    expanded the effective config. CLI paths identify the values to append
    last; containers are encoded as objects, never flattened into shell text.
    """
    parser = ConfigArgumentParser(prog="reef serve", add_help=False, allow_abbrev=False)
    yaml_args: list[str] = []
    cli_args: list[str] = []
    for argument in service_config_arguments():
        argument.add_to(parser)
        value = _argument_value(config, argument)
        if value is not None:
            target = cli_args if argument.path in cli_paths else yaml_args
            target.append(f"{argument.flags[0]}={argument.encode(value)}")
    values = vars(parser.parse_args([*yaml_args, *cli_args]))
    return {argument.name: values[argument.destination] for argument in service_config_arguments()}


def normalize_service_config(
    config: Mapping[str, Any], *, cli_paths: frozenset[tuple[str, ...]] = frozenset()
) -> dict[str, Any]:
    """Validate explicit public settings and pass the same values to children.

    Do not add defaults to the deployment mapping: recipes must still know
    which of their fields the operator supplied, and custom stacks may have
    no Reef HTTP child at all.
    """
    values = parse_service_arguments(config, cli_paths=cli_paths)
    normalized = copy.deepcopy(dict(config))
    for argument in service_config_arguments():
        if _argument_value(config, argument) is None:
            continue
        node = normalized
        for key in argument.path[:-1]:
            node = node[key]
        node[argument.path[-1]] = values[argument.name]
    return normalized


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
    values = parse_service_arguments(config)
    if not isinstance(values["recipe"], str) or not values["recipe"]:
        raise ValueError("config must declare a non-empty reef.recipe")
    values["tokens"] = _service_tokens({"reef": {"token": values.pop("token"), "tokens": values["tokens"]}})
    values["console_origins"] = console_origins(values["console_origins"])
    # The legacy retry deadline follows the request timeout unless supplied.
    if _config_service_value(config, "reef", "inference_retry_timeout_s") is None:
        values["inference_retry_timeout_s"] = values["inference_timeout_s"]
    return ServiceSettings(**values, recipe_settings=_reef_section(config))


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
