"""Build managed inference services from declarative backend launch definitions."""

from __future__ import annotations

import os
import socket
import sys
from dataclasses import dataclass
from typing import Any

from reef.service.deploy.config import DeployConfigError
from reef.service.deploy.options import native_arguments
from reef.service.deploy.settings import ServiceSettings


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


def prepare_inference(config: dict[str, Any], settings: ServiceSettings) -> dict[str, Any]:
    """Resolve launch choices once, before model downloads or service creation."""
    backend = settings.inference_backend or "sglang"
    definition = INFERENCE_BACKENDS.get(backend)
    if definition is None:
        raise DeployConfigError(f"managed local inference supports: {', '.join(sorted(INFERENCE_BACKENDS))}")
    parallel_size = settings.tensor_parallel_size if settings.tensor_parallel_size is not None else 1
    if parallel_size < 1:
        raise DeployConfigError("--tensor-parallel-size must be positive")
    if settings.upstream_url or settings.upstream_model or settings.upstream_api != "openai":
        raise DeployConfigError("--model-path cannot be combined with upstream provider selection")
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
