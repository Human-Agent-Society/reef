"""Load the selected inference implementation without importing training code."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from importlib.metadata import entry_points
from typing import Any

from reef.core.errors import DeployConfigError
from reef.runtime.deployment import InferenceService

_BUILTINS = {
    "sglang": "reef.inference.sglang.deployment:create_inference",
    "vllm": "reef.inference.vllm.deployment:create_inference",
}


def inference_service_for(name: str | None, config: Mapping[str, Any]) -> InferenceService:
    """Construct an unallocated receiver from a built-in, entry point or reference."""
    selected = name or "sglang"
    reference = _BUILTINS.get(selected, selected)
    if ":" not in reference:
        matches = tuple(entry_points(group="reef.inference_backends", name=selected))
        if len(matches) != 1:
            reason = "unknown" if not matches else "ambiguous"
            raise DeployConfigError(
                f"{reason} inference backend {selected!r}; install its integration or use module:factory"
            )
        reference = matches[0].value
    module, _, attribute = reference.partition(":")
    try:
        factory = getattr(importlib.import_module(module), attribute)
    except (ImportError, AttributeError) as exc:
        raise DeployConfigError(f"cannot load inference backend {selected!r}: {exc}") from exc
    if not callable(factory):
        raise DeployConfigError("inference backend must name an inference service factory")
    service = factory(config)
    if not isinstance(service, InferenceService):
        raise DeployConfigError("inference backend factory must return an InferenceService")
    return service
