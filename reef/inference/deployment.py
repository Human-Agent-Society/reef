"""The inference component: its contracts, and the selection of its implementation without training code.

An ``InferenceService`` owns its engines and borrows the GPUs of an ``InferenceResources``.
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
from reef.execution.executor.placement import (
    DeploymentResources,
    GpuBundles,
    ModelGpuLayout,
    ModelGpuReservation,
    reserve_model_gpus,
)
from reef.inference.interfaces import InferenceBackend


class InferenceResources(DeploymentResources):
    """The GPUs that one engine start reserves on this host, in the current Ray session.

    ``num_gpus`` is a positive int. ``start`` reserves that many GPU bundles
    for inference and ``close`` releases them; the Ray session outlives both.
    ``inference_placement`` is the reserved bundles between ``start`` and
    ``close``, and None otherwise.

    Purpose: the remote inference component's server makes one for each
    engine start; ``InferenceService.start`` reads ``inference_placement``.
    """

    def __init__(self, num_gpus: int) -> None:
        if isinstance(num_gpus, bool) or not isinstance(num_gpus, int) or num_gpus <= 0:
            raise ValueError("an engine start needs a positive GPU count")
        self.num_gpus = num_gpus
        self.reservation: ModelGpuReservation | None = None

    @property
    def inference_placement(self) -> GpuBundles | None:
        if self.reservation is None:
            return None
        return self.reservation.inference

    def start(self) -> None:
        self.reservation = reserve_model_gpus(ModelGpuLayout(training_gpus=0, inference_gpus=self.num_gpus))

    def close(self) -> None:
        reservation, self.reservation = self.reservation, None
        if reservation is not None:
            reservation.release()


@dataclass(frozen=True)
class InferenceConnection:
    """What a started ``InferenceService`` returns: its control executor, when it has one.

    ``control`` is the borrowed executor that the service's
    ``InferenceBackend`` calls, or None for a service whose backend reaches
    the engines by other means, such as HTTP.

    Purpose: ``InferenceService.start`` makes it; ``InferenceService.backend``
    of the same service reads it.
    """

    control: Executor | None


class InferenceService(ABC):
    """An inference component that owns its engines but borrows reservations."""

    @abstractmethod
    def start(self, resources: InferenceResources) -> InferenceConnection:
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
