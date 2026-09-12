"""OpenClawRL owns the judge and user simulator needed by its training method."""

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from reef.core.config import config_arguments, config_option, parse_config_values
from reef.runtime.executor.arguments import native_arguments, normalize_native_options
from reef.service.deploy.inference import INFERENCE_BACKENDS, http_readiness_command


@dataclass(frozen=True)
class ModelServer:
    model_path: str = config_option("")
    port: int = config_option(23001)
    tensor_parallel_size: int = config_option(1)
    served_model_name: str = config_option("")
    ready_timeout: int = config_option(600)
    options: Mapping[str, Any] = config_option(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_path.strip():
            raise ValueError("OpenClawRL model server requires model-path")
        if not 1 <= self.port <= 65535 or self.tensor_parallel_size < 1 or self.ready_timeout <= 0:
            raise ValueError("invalid OpenClawRL model server port, parallel size or readiness timeout")

    def process(self, name: str) -> dict[str, Any]:
        definition = INFERENCE_BACKENDS["sglang"]
        python = os.environ.get("REEF_PYTHON", sys.executable)
        bindings = {
            "python": python,
            "model_path": self.model_path,
            "host": "0.0.0.0",
            "port": str(self.port),
            "served_model_name": self.served_model_name or self.model_path,
            "tensor_parallel_size": str(self.tensor_parallel_size),
        }
        reserved = {word[2:] for word in definition.command if word.startswith("--")}
        options = native_arguments(self.options, reserved=reserved | set(definition.reserved_options))
        return {
            "name": name,
            "executor": "ray",
            "endpoint": f"http://{{host}}:{self.port}",
            "command": [*(word.format_map(bindings) for word in definition.command), *options],
            "ready": http_readiness_command(python, f"http://127.0.0.1:{self.port}/health"),
            "ready_timeout": self.ready_timeout,
            "resources": {"num_gpus": self.tensor_parallel_size},
        }


def prepare_dependencies(config: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    """Bind the PRM client and reserve method model workers before Slime starts."""
    processes = []
    ports = set()
    for field_name, name, port in (("prm", "prm-sglang", 23001), ("user_simulator", "user-llm-sglang", 30001)):
        supplied = config.get(field_name, {})
        if not supplied:
            continue
        # Reuse the shared field parser for both YAML objects and CLI leaf overrides.
        normalized = {key.replace("-", "_"): value for key, value in normalize_native_options(supplied).items()}
        settings = ModelServer(**parse_config_values(config_arguments(ModelServer), {"port": port, **normalized}))
        if settings.port in ports:
            raise ValueError("OpenClawRL model servers require distinct ports")
        ports.add(settings.port)
        if field_name == "prm":
            if config.get("prm_url"):
                raise ValueError("configure either a managed prm model or an external prm-url")
            config["prm_url"] = "${endpoints.prm-sglang}"
            config["prm_tokenizer_path"] = settings.model_path
        processes.append(settings.process(name))
    return tuple(processes)
