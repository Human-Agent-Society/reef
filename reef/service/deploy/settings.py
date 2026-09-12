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

from reef.core.config import ConfigArgument, config_arguments, config_metadata, config_option, parse_config_values
from reef.service.cors import console_origins
from reef.service.deploy.config import config_value, interpolate_config, interpolate_config_values, load_config
from reef.storage.postgres import postgres_url, validate_postgres_schema
from reef.storage.records import RecordRetention

_DESCRIPTION = """reef serve — connect an external provider or start a configured stack.

With no selected config, --inference.upstream-url and --inference.upstream-model start Reef's
record-only recipe on 127.0.0.1:8900. YAML and a services list are optional.
Alternatively, --inference.model-path starts managed SGLang inference and Reef;
--inference.tensor-parallel-size selects the visible GPU count (default: 1).
Config files are selected explicitly with -c; REEF_CONFIG and ./reef.yaml
are not discovered by the launcher.

``reef serve -c <stack>.yaml`` reads the config's ``services``
list and starts every declared process (SGLang, Slime driver, Reef, and so on)
in dependency order. Each service's ``ready`` probe must pass before the next
starts. After all services are up, Reef blocks until SIGTERM/SIGINT; a
watchdog thread detects unexpected exits and tears the stack down.

The Reef HTTP child receives the effective configuration from the launcher.

Config overrides:
  Public settings below share type conversion with YAML. Explicit CLI
  values override YAML; omitted settings use the dataclass defaults.
  Use the full public namespace; legacy aliases remain accepted.
  Lists and objects take one quoted JSON/YAML value, including [] or {}.
  Selected recipe/runtime fields share these rules; use -c <file> --help
  to inspect their definitions. Versioned files reject unknown public
  fields; legacy custom-stack keys retain their compatibility parsing.

  Examples:
    reef serve --inference.model-path Qwen/Qwen2.5-1.5B-Instruct
    reef serve --inference.upstream-url http://localhost:8000 --inference.upstream-model my-model
    reef serve -c stack.yaml --inference.model-path Qwen/Qwen2.5-1.5B-Instruct
    reef serve -c path/to/local-sglang.yaml --service.port 9000
    reef serve -c stack.yaml --training.config.checkpoint_dir /tmp/ckpt
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
        help="Optional config file path, relative to the working directory; no file is loaded unless selected.",
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

    recipe: str = config_option(
        public_path=("recipe", "implementation"),
        help="Recipe implementation or named deployment preset (the launcher owns --recipe).",
    )
    host: str = config_option("0.0.0.0", public_path=("service", "host"), help="HTTP bind address.")
    port: int = config_option(8900, public_path=("service", "port"), help="HTTP bind port.")
    tokens: tuple[str, ...] = config_option(
        (), public_path=("service", "tokens"), help="Accepted bearer tokens as a JSON/YAML list."
    )
    console_origins: tuple[str, ...] = config_option(
        (), public_path=("service", "console_origins"), help="Allowed console origins as a JSON/YAML list."
    )
    ray_address: str | None = config_option(None, public_path=("training", "ray_address"), help="Ray cluster address.")
    ray_namespace: str = config_option(
        "reef", public_path=("training", "ray_namespace"), help="Ray namespace for the training bridge."
    )
    ray_actor_name: str = config_option(
        "reef-train-bridge", public_path=("training", "ray_actor_name"), help="Training bridge actor name."
    )
    inference_url: str | None = config_option(None, public_path=("inference", "url"), help="Local inference endpoint.")
    model_path: str | None = config_option(
        None, public_path=("inference", "model_path"), help="Local model directory or Hugging Face repository ID."
    )
    inference_backend: str | None = config_option(
        None, public_path=("inference", "backend"), help="Managed local inference backend (default: sglang)."
    )
    tensor_parallel_size: int | None = config_option(
        None, public_path=("inference", "tensor_parallel_size"), help="Managed local inference GPU count (default: 1)."
    )
    inference_options: Mapping[str, Any] = field(
        default_factory=dict,
        metadata=config_metadata("Native inference engine options.", public_path=("inference", "options")),
    )
    training_backend_options: Mapping[str, Any] = field(
        default_factory=dict,
        metadata=config_metadata("Native Slime driver options.", public_path=("training", "options")),
    )
    #: The OpenAI-compatible provider no-update recipes proxy to (no ``/v1``
    #: suffix), its credential, and the model name to request from it. The
    #: only place the upstream is named: the HTTP service forwards to it, and
    #: the training side derives the model binding it hands to methods and
    #: evaluation episodes from it. ``upstream_model`` is a provider model
    #: name, unlike ``model_path``, which is local weights for training.
    upstream_url: str | None = config_option(
        None, public_path=("inference", "upstream_url"), help="Upstream provider base URL."
    )
    upstream_api_key: str | None = config_option(
        None, public_path=("inference", "upstream_api_key"), help="Upstream provider credential."
    )
    upstream_model: str | None = config_option(
        None, public_path=("inference", "upstream_model"), help="Model name requested from the upstream provider."
    )
    #: The provider's API dialect: ``openai`` (default), ``responses``, or ``anthropic``.
    upstream_api: str = config_option(
        "openai", public_path=("inference", "upstream_api"), help="Provider API dialect."
    )
    inference_timeout_s: float = config_option(
        300.0, public_path=("inference", "timeout_s"), help="Inference request timeout in seconds."
    )
    train_timeout_s: float | None = config_option(
        None, public_path=("training", "timeout_s"), help="Training request timeout in seconds."
    )
    inference_backend_factory: str | None = config_option(
        None, public_path=("inference", "backend_factory"), help="Dotted inference backend factory."
    )
    inference_backend_config: Mapping[str, Any] = field(
        default_factory=dict,
        metadata=config_metadata(
            "Inference backend options as a JSON/YAML object.", public_path=("inference", "backend_config")
        ),
    )
    inference_retry_initial_s: float = config_option(
        0.05, public_path=("inference", "retry_initial_s"), help="Initial inference retry delay in seconds."
    )
    inference_retry_max_s: float = config_option(
        1.0, public_path=("inference", "retry_max_s"), help="Maximum inference retry delay in seconds."
    )
    inference_retry_timeout_s: float = config_option(
        300.0,
        public_path=("inference", "retry_timeout_s"),
        help="Retry deadline in seconds; defaults to the inference timeout.",
    )
    artifact_repository: str = config_option(
        ".reef/artifacts.git", public_path=("storage", "artifact_repository"), help="Artifact repository location."
    )
    artifact_work_dir: str = config_option(
        ".reef/artifact-work", public_path=("storage", "artifact_work_dir"), help="Artifact working directory."
    )
    artifact_cache_dir: str = config_option(
        ".reef/artifact-cache", public_path=("storage", "artifact_cache_dir"), help="Artifact cache directory."
    )
    agent_record_dir: str = config_option(
        ".reef/agent-record", public_path=("storage", "agent_record_dir"), help="Agent record directory."
    )
    record_backend: str = config_option(
        "sqlite", public_path=("storage", "record_backend"), help="Record storage backend."
    )
    record_database_url: str | None = field(
        default=None,
        repr=False,
        metadata=config_metadata("PostgreSQL connection URL.", public_path=("storage", "record_database_url")),
    )
    record_database_schema: str = config_option(
        "reef_records", public_path=("storage", "record_database_schema"), help="PostgreSQL schema."
    )
    agent_record_retention_days: float = config_option(
        7.0, public_path=("storage", "agent_record_retention_days"), help="Record retention in days."
    )
    agent_record_retention_max_bytes: int = config_option(
        20 * 1024**3,
        public_path=("storage", "agent_record_retention_max_bytes"),
        help="Maximum retained record bytes.",
    )
    allow_implicit_scenario_creation: bool = config_option(
        True,
        public_path=("service", "allow_implicit_scenario_creation"),
        help="Allow requests to create scenarios implicitly.",
    )
    #: Deployment-level experiment provider settings, sourced from
    #: ``observability.wandb``.
    wandb_config: Mapping[str, Any] = field(
        default_factory=dict,
        metadata=config_metadata("W&B settings as a JSON/YAML object.", path=("observability", "wandb")),
    )
    training_settings: Mapping[str, Any] = field(
        default_factory=dict,
        metadata=config_metadata(
            "Training settings as a JSON/YAML object.", path=("training",), public_path=("training", "config")
        ),
    )
    #: Optional pre-publication checkpoint evaluator and selection plugin.
    evaluation_settings: Mapping[str, Any] | None = field(
        default=None,
        metadata=config_metadata("Candidate evaluation settings as a JSON/YAML object.", path=("evaluation",)),
    )
    #: The flat ``reef`` config section, interpolated; recipes read their own
    #: config fields from it (see the class docstring).
    recipe_settings: Mapping[str, Any] = field(default_factory=dict, repr=False)
    preset_config: Mapping[str, Any] | None = field(default=None, repr=False)

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

    return {key: interpolate_config_values(config, value) for key, value in section.items()}


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
        ConfigArgument(
            "token",
            ("reef", "token"),
            "str",
            True,
            None,
            "One accepted bearer token.",
            public_path=("service", "token"),
        ),
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
    return interpolate_config_values(config, node)


def parse_service_arguments(
    config: Mapping[str, Any], *, cli_paths: frozenset[tuple[str, ...]] = frozenset()
) -> dict[str, Any]:
    """Parse the merged public values with the shared component parser.

    The caller has already replaced overridden environment references and
    expanded the effective config. ``cli_paths`` remains accepted for callers
    of the earlier adapter; precedence is already present in the mapping.
    """
    if "schema-version" in config:
        from reef.service.deploy.layout import translate_layout

        config = translate_layout(config)
    arguments = service_config_arguments()
    supplied = {}
    for argument in arguments:
        value = _argument_value(config, argument)
        if value is not None:
            supplied[argument.name] = value
    # Precedence is already represented by the merged mapping. Keep cli_paths
    # in this compatibility entrypoint; conversion is shared with components.
    return parse_config_values(
        tuple(dataclasses.replace(argument, required=False) for argument in arguments), supplied, environ=os.environ
    )


def normalize_service_config(
    config: Mapping[str, Any], *, cli_paths: frozenset[tuple[str, ...]] = frozenset()
) -> dict[str, Any]:
    """Validate explicit public settings and pass the same values to children.

    Do not add defaults to the deployment mapping: recipes must still know
    which of their fields the operator supplied, and custom stacks may have
    no Reef HTTP child at all.
    """
    if "schema-version" in config:
        from reef.service.deploy.layout import translate_layout

        config = translate_layout(config)
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
    if "schema-version" in config:
        from reef.service.deploy.components import component_config_arguments
        from reef.service.deploy.layout import normalize_component_layout, translate_layout

        config = translate_layout(config)
        config = normalize_component_layout(config, component_config_arguments(config))
    values = parse_service_arguments(config)
    if not isinstance(values["recipe"], str) or not values["recipe"]:
        raise ValueError("config must declare a non-empty reef.recipe")
    values["tokens"] = _service_tokens({"reef": {"token": values.pop("token"), "tokens": values["tokens"]}})
    values["console_origins"] = console_origins(values["console_origins"])
    # The legacy retry deadline follows the request timeout unless supplied.
    if _config_service_value(config, "reef", "inference_retry_timeout_s") is None:
        values["inference_retry_timeout_s"] = values["inference_timeout_s"]
    # Preserve shared execution settings for presets and directly selected recipes.
    preset = dict(config) if "implementation" in config or ":" in values["recipe"] else None
    return ServiceSettings(**values, recipe_settings=_reef_section(config), preset_config=preset)


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
