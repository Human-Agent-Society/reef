"""The inference component: its contracts, and the selection of its implementation without training code.

An ``InferenceService`` owns its engines and borrows ``InferenceResources``.
``inference_service_for`` builds the service that the ``inference.backend``
setting names.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Any

from reef.core.errors import DeployConfigError
from reef.execution.executor import Executor
from reef.execution.executor.placement import DeploymentResources
from reef.inference.interfaces import InferenceBackend


class InferenceResources(DeploymentResources):
    """Allocation that supplies a backend-native inference reservation."""

    @property
    @abstractmethod
    def inference_placement(self) -> Any:
        """Return the borrowed placement handle understood by the selected backend."""


@dataclass(frozen=True)
class InferenceConnection:
    """What a started ``InferenceService`` returns: its control protocol and, when it has one, its control executor.

    ``protocol`` is the service's non-empty ``connection_protocol``; it names
    the control vocabulary. ``control`` is the borrowed executor that the
    service's ``InferenceBackend`` calls, or None for a receiver that the
    backend reaches by other means, such as HTTP.
    """

    protocol: str
    control: Executor | None


class InferenceService(ABC):
    """An inference component that owns its engines but borrows reservations."""

    @property
    @abstractmethod
    def connection_protocol(self) -> str:
        """Control protocol provided by the selected engine integration."""

    @abstractmethod
    def start(self, resources: DeploymentResources) -> InferenceConnection:
        """Start engines in supplied resources and return a borrowed connection."""

    @abstractmethod
    def backend(self, connection: InferenceConnection) -> InferenceBackend:
        """Return this receiver's backend for Reef-owned coordination."""

    @abstractmethod
    def check_health(self) -> None:
        """Raise when the component is not ready."""

    @abstractmethod
    def poll(self) -> None:
        """Raise when the engines failed, without blocking behind active work: a busy engine has not failed.

        The inference component's health route calls it.
        """

    @abstractmethod
    def close(self) -> None:
        """Release owned engines, including partial starts, idempotently."""


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
