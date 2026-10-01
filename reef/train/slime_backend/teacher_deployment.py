"""Managed independent teacher process for Slime distillation deployments."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from typing import Any

from reef.core.errors import DeployConfigError
from reef.runtime.executor.arguments import native_arguments, normalize_native_options
from reef.train.deployment import INFERENCE_RESERVED_OPTIONS

TEACHER_SERVICE = "distill-teacher"


def teacher_service(settings: Mapping[str, Any]) -> dict[str, Any] | None:
    """Reserve a separate Ray allocation and stop the teacher with its deployment."""
    if not settings.get("teacher_model_path"):
        if settings.get("teacher_options"):
            raise DeployConfigError("teacher.options requires teacher.model-path")
        return None
    size = settings["teacher_num_gpus"]
    port = settings["teacher_port"]
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise DeployConfigError("teacher.num-gpus must be a positive integer")
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535 or port == settings["port"]:
        raise DeployConfigError("teacher.port must be valid and distinct from reef.port")
    options = normalize_native_options(settings["teacher_options"])
    reserved = set(INFERENCE_RESERVED_OPTIONS) | {
        "tokenizer-path",
        "tokenizer-mode",
        "revision",
        "tokenizer-revision",
        "enable-lora",
        "lora-paths",
        "load-format",
        "enable-mis",
        "skip-tokenizer-init",
    }
    arguments = native_arguments(options, reserved=reserved)
    python = os.environ.get("REEF_PYTHON", sys.executable)
    return {
        "name": TEACHER_SERVICE,
        "executor": "ray",
        "resources": {"num_gpus": size, "num_cpus": 1},
        "endpoint": f"http://{{host}}:{port}",
        "command": [
            python,
            "-m",
            "sglang.launch_server",
            "--model-path",
            "${reef.teacher_model_path}",
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
            "--tp",
            str(size),
            *arguments,
        ],
        "ready": [
            python,
            "-c",
            "import sys, urllib.request; "
            "urllib.request.build_opener(urllib.request.ProxyHandler({})).open(sys.argv[1], timeout=5).close()",
            "${endpoints.distill-teacher}/health",
        ],
        "ready_timeout": settings["training_ready_timeout"],
    }
