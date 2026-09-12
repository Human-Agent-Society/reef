"""Assemble the standard external-provider deployment from service settings."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

from reef.runtime.adapters.inference_proxy import PROVIDER_APIS
from reef.service.deploy.config import DeployConfigError
from reef.service.deploy.settings import service_override, service_settings_from_config
from reef.service.profiles import profile_names

# Only these environment fallbacks belong to configuration-free startup.
# Existing files continue to opt in through their own environment references.
_ENVIRONMENT_FIELDS = {
    "upstream_url": "REEF_UPSTREAM_URL",
    "upstream_model": "REEF_UPSTREAM_MODEL",
    "upstream_api_key": "REEF_UPSTREAM_API_KEY",
    "token": "REEF_TOKEN",
}
_CONFIGURED_FIELDS = {
    "model_path",
    "inference_url",
    "inference_backend_factory",
    "inference_backend_config",
    "ray_address",
    "ray_namespace",
    "ray_actor_name",
    "train_timeout_s",
    "training_settings",
    "evaluation_settings",
}


def provider_config(overrides: Mapping[str, str], environ: Mapping[str, str]) -> dict[str, Any]:
    """Supply standard defaults; the orchestrator applies and parses overrides."""
    for key, value in overrides.items():
        declared = service_override(key, value)
        if declared is None:
            raise DeployConfigError(f"unknown option --{key} for provider startup; use -c for a custom stack")
        argument, _ = declared
        if argument.name == "host" and not value.strip():
            raise DeployConfigError("--host must be non-empty")
        if argument.name in _CONFIGURED_FIELDS:
            raise DeployConfigError(f"--{key} requires a configured stack (-c); provider startup uses an upstream")
        if argument.name == "recipe" and value != "recipe":
            raise DeployConfigError(
                "provider startup uses the core recipe; select other recipes with a config/profile"
            )
    reef: dict[str, Any] = {"recipe": "recipe", "host": "127.0.0.1"}
    for field, variable in _ENVIRONMENT_FIELDS.items():
        if environ.get(variable, "").strip():
            reef[field] = environ[variable]
    return {"reef": reef, "run_dir": ".reef/run"}


def assemble_provider_services(config: dict[str, Any]) -> None:
    """Validate typed inputs and add the owned HTTP process and readiness probe."""
    settings = service_settings_from_config(config)
    if not settings.upstream_url or not settings.upstream_model:
        raise DeployConfigError(
            "provider startup requires --upstream-url and --upstream-model (or their REEF_UPSTREAM_* variables).\n"
            "  Example: reef serve --upstream-url http://localhost:8000 --upstream-model my-model\n"
            f"  Alternatively pass -c <file> or --recipe <name>; recipes with a profile: {', '.join(profile_names())}"
        )
    try:
        upstream = urlsplit(settings.upstream_url)
        valid_url = upstream.scheme in {"http", "https"} and bool(upstream.hostname)
        if upstream.port is not None and not 1 <= upstream.port <= 65535:
            valid_url = False
    except ValueError:
        valid_url = False
    if not valid_url:
        raise DeployConfigError("--upstream-url must be an HTTP(S) URL with a valid host and port")
    if not settings.host or not 1 <= settings.port <= 65535:
        raise DeployConfigError("provider startup requires a non-empty --host and --port between 1 and 65535")
    if settings.upstream_api not in PROVIDER_APIS:
        raise DeployConfigError("--upstream-api must be openai, responses, or anthropic")
    if settings.inference_timeout_s <= 0:
        raise DeployConfigError("--inference-timeout-s must be positive")
    host = settings.host
    if host == "0.0.0.0":
        host = "127.0.0.1"
    elif host == "::":
        host = "::1"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    endpoint = f"http://{host}:{settings.port}"
    python = os.environ.get("REEF_PYTHON", sys.executable)
    config["services"] = [
        {
            "name": "reef",
            "executor": "uni",
            "command": [python, "-m", "reef.service"],
            "endpoint": endpoint,
            "ready": [
                python,
                "-c",
                "import sys, urllib.request; "
                "urllib.request.build_opener(urllib.request.ProxyHandler({})).open(sys.argv[1], timeout=5).close()",
                f"{endpoint}/healthz",
            ],
            "ready_timeout": 30,
        }
    ]
