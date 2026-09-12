"""Assemble external-provider or managed inference and their Reef HTTP process."""

from __future__ import annotations

import os
import socket
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from reef.runtime.adapters.inference_proxy import PROVIDER_APIS
from reef.runtime.executor.arguments import native_arguments
from reef.service.deploy.config_utils import DeployConfigError
from reef.service.deploy.service_config import ServiceConfig, service_config_from_mapping
from reef.service.profiles import profile_names


@dataclass(frozen=True)
class InferenceBackend:
    """Native command bindings and readiness for an OpenAI-compatible engine."""

    command: tuple[str, ...]
    health_path: str
    reserved_options: tuple[str, ...] = ()


INFERENCE_BACKENDS = {
    "sglang": InferenceBackend(
        command=(
            "{python}",
            "-m",
            "sglang.launch_server",
            "--model-path",
            "{model_path}",
            "--served-model-name",
            "{served_model_name}",
            "--host",
            "{host}",
            "--port",
            "{port}",
            "--tp",
            "{tensor_parallel_size}",
        ),
        health_path="/health",
        reserved_options=(
            "model",
            "tp-size",
            "tensor-parallel-size",
            "config",
            "config-file",
            "yaml-config",
            "api-key",
            "dp",
            "dp-size",
            "data-parallel-size",
            "nnodes",
            "node-rank",
            "dist-init-addr",
        ),
    ),
}


def http_readiness_command(python: str, endpoint: str) -> list[str]:
    """Check HTTP readiness without a shell or an optional HTTP client."""
    return [
        python,
        "-c",
        "import sys, urllib.request; "
        "urllib.request.build_opener(urllib.request.ProxyHandler({})).open(sys.argv[1], timeout=5).close()",
        endpoint,
    ]


def prepare_inference(config: dict[str, Any], settings: ServiceConfig) -> dict[str, Any]:
    """Resolve launch choices once, before model downloads or service creation."""
    backend = settings.inference_backend or "sglang"
    definition = INFERENCE_BACKENDS.get(backend)
    if definition is None:
        raise DeployConfigError(f"managed local inference supports: {', '.join(sorted(INFERENCE_BACKENDS))}")
    parallel_size = settings.tensor_parallel_size if settings.tensor_parallel_size is not None else 1
    if parallel_size < 1:
        raise DeployConfigError("--inference.tensor-parallel-size must be positive")
    num_gpus = settings.inference_num_gpus if settings.inference_num_gpus is not None else parallel_size
    if num_gpus != parallel_size:
        raise DeployConfigError(
            "standalone inference launches one engine; inference.num-gpus must equal inference.tensor-parallel-size"
        )
    if settings.upstream_url or settings.upstream_model or settings.upstream_api != "openai":
        raise DeployConfigError("--inference.model-path cannot be combined with upstream provider selection")
    reserved = {token[2:] for token in definition.command if token.startswith("--")}
    extra_args = native_arguments(settings.inference_options, reserved=reserved | set(definition.reserved_options))
    python = os.environ.get("REEF_PYTHON", sys.executable)
    # The HTTP child and inference engine share one resolved loopback endpoint. The
    # actual server bind remains authoritative if another process races it.
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        if port == settings.port:
            with socket.socket() as alternative:
                alternative.bind(("127.0.0.1", 0))
                port = alternative.getsockname()[1]
    endpoint = f"http://127.0.0.1:{port}"
    config["reef"].update(
        inference_backend=backend,
        tensor_parallel_size=parallel_size,
        inference_num_gpus=num_gpus,
        upstream_url=endpoint,
        upstream_model=settings.model_path,
        upstream_api_key=None,
    )
    bindings = {
        "python": python,
        "model_path": "${reef.model_path}",
        "served_model_name": settings.model_path,
        "host": "127.0.0.1",
        "port": str(port),
        "tensor_parallel_size": str(parallel_size),
    }
    return {
        "name": backend,
        "executor": "uni",
        "endpoint": endpoint,
        "command": [*(token.format_map(bindings) for token in definition.command), *extra_args],
        "ready": http_readiness_command(python, endpoint + definition.health_path),
        "ready_timeout": config.get("ready_timeout", 3600),
    }


# Only these environment fallbacks belong to configuration-free startup.
# Existing files continue to opt in through their own environment references.
_ENVIRONMENT_FIELDS = {
    "upstream_url": "REEF_UPSTREAM_URL",
    "upstream_model": "REEF_UPSTREAM_MODEL",
    "upstream_api_key": "REEF_UPSTREAM_API_KEY",
    "token": "REEF_TOKEN",
}
_CONFIGURED_FIELDS = {
    "inference_url",
    "inference_backend_factory",
    "inference_backend_config",
    "ray_address",
    "ray_namespace",
    "ray_actor_name",
    "train_timeout_s",
    "training_backend",
    "training_ready_timeout",
    "training_settings",
    "training_backend_options",
    "evaluation_settings",
}


def command_line_config(environ: Mapping[str, str]) -> dict[str, Any]:
    """Seed optional environment fallbacks before selecting component schemas."""
    reef: dict[str, Any] = {"recipe": "recipe", "host": "127.0.0.1"}
    for field, variable in _ENVIRONMENT_FIELDS.items():
        if environ.get(variable, "").strip():
            reef[field] = environ[variable]
    return {"reef": reef, "run_dir": ".reef/run"}


def assemble_provider_services(config: dict[str, Any]) -> None:
    """Validate typed inputs and add the owned HTTP process and readiness probe."""
    unsupported = set(config.get("reef", {})) & _CONFIGURED_FIELDS
    if unsupported or "training" in config or "evaluation" in config:
        raise DeployConfigError(
            "select a weight-training recipe for training; custom process stacks require unversioned legacy YAML"
        )
    settings = service_config_from_mapping(config)
    if config.get("reef", {}).get("runtime"):
        if not settings.host.strip() or not 1 <= settings.port <= 65535:
            raise DeployConfigError("invalid reef.host or reef.port")
        config["services"] = [http_service(config, settings)]
        return
    local_service = None
    if settings.model_path:
        # Validate the public bind and timeout before checking GPU dependencies.
        if not 1 <= settings.port <= 65535 or settings.inference_timeout_s <= 0:
            raise DeployConfigError("local inference requires a valid --reef.port and positive --inference.timeout-s")
        local_service = prepare_inference(config, settings)
        settings = service_config_from_mapping(config)
    elif (
        settings.inference_backend is not None
        or settings.tensor_parallel_size is not None
        or settings.inference_num_gpus is not None
        or settings.inference_options
    ):
        raise DeployConfigError(
            "--inference.backend and --inference.tensor-parallel-size require --inference.model-path"
        )
    if not settings.upstream_url or not settings.upstream_model:
        raise DeployConfigError(
            "provider startup requires --inference.upstream-url and --inference.upstream-model "
            "(or their REEF_UPSTREAM_* variables).\n"
            "  Example: reef serve --inference.upstream-url http://localhost:8000 --inference.upstream-model my-model\n"
            "  Or start local inference: reef serve --inference.model-path Qwen/Qwen2.5-1.5B-Instruct\n"
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
        raise DeployConfigError("--inference.upstream-url must be an HTTP(S) URL with a valid host and port")
    if not settings.host or not 1 <= settings.port <= 65535:
        raise DeployConfigError(
            "provider startup requires a non-empty --reef.host and --reef.port between 1 and 65535"
        )
    if settings.upstream_api not in PROVIDER_APIS:
        raise DeployConfigError("--inference.upstream-api must be openai, responses, or anthropic")
    if settings.inference_timeout_s <= 0:
        raise DeployConfigError("--inference.timeout-s must be positive")
    config["services"] = [http_service(config, settings)]
    if local_service is not None:
        config["services"][0]["depends_on"] = [local_service["name"]]
        config["services"].insert(0, local_service)


def http_service(config: Mapping[str, Any], settings: ServiceConfig) -> dict[str, Any]:
    """Build the standard local HTTP child and its readiness probe."""
    host = settings.host
    if host == "0.0.0.0":
        host = "127.0.0.1"
    elif host == "::":
        host = "::1"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    endpoint = f"http://{host}:{settings.port}"
    python = os.environ.get("REEF_PYTHON", sys.executable)
    return {
        "name": "reef",
        "executor": "uni",
        "command": [python, "-m", "reef.service"],
        "endpoint": endpoint,
        "ready": http_readiness_command(python, f"{endpoint}/healthz"),
        "ready_timeout": config.get("ready_timeout", 30),
    }


def is_local_path(value: str) -> bool:
    """True if the value is an existing local filesystem path, False if it's an HF repo ID."""
    return Path(os.path.expanduser(value)).exists()


def resolve_hf_snapshot(repo_id: str) -> str:
    """Download an HF repo ID to a local snapshot and return the local path.

    If the value is already a local path, return it unchanged.
    """
    value = os.path.expanduser(repo_id)
    if is_local_path(repo_id):
        return value
    if "/" not in value or value.startswith(("/", "~")):
        raise DeployConfigError(
            f"model path is not a local directory and not a valid HF repo ID: {repo_id}\n"
            f"  set reef.model_path to a local path or an HF repo ID like 'Qwen/Qwen2.5-1.5B-Instruct'"
        )
    try:
        from reef.artifact.sources import HuggingFaceSource, download_huggingface_snapshot, parse_artifact_source

        source = parse_artifact_source(value)
        if not isinstance(source, HuggingFaceSource):
            raise DeployConfigError(f"model path is not a valid HF repo ID: {repo_id}")
        snapshot = download_huggingface_snapshot(source)
        return str(snapshot.local_path)
    except Exception as exc:
        if isinstance(exc, DeployConfigError):
            raise
        raise DeployConfigError(f"failed to download HF model {value}: {exc}") from exc


def resolve_model_paths(config: dict[str, Any]) -> bool:
    """Download HF repo IDs for model paths and replace them with local snapshot paths.

    Handles ``reef.model_path`` and ``training.megatron_checkpoint_path``.
    Returns True if any path was changed (i.e. the config dict no longer
    matches what's on disk).
    """
    changed = False
    reef_section = config.get("reef")
    if isinstance(reef_section, dict) and isinstance(reef_section.get("model_path"), str):
        resolved = resolve_hf_snapshot(reef_section["model_path"])
        if resolved != reef_section["model_path"]:
            _log(f"downloaded HF model {reef_section['model_path']} -> {resolved}")
            reef_section["model_path"] = resolved
            changed = True
    training_section = config.get("training")
    if isinstance(training_section, dict) and isinstance(training_section.get("megatron_checkpoint_path"), str):
        resolved = resolve_hf_snapshot(training_section["megatron_checkpoint_path"])
        if resolved != training_section["megatron_checkpoint_path"]:
            _log(f"downloaded HF model {training_section['megatron_checkpoint_path']} -> {resolved}")
            training_section["megatron_checkpoint_path"] = resolved
            changed = True
    return changed


def _log(msg: str) -> None:
    import sys

    print(f"[reef] {msg}", file=sys.stderr)
