"""The inference component: its contracts, and the selection of its implementation without training code.

An ``InferenceService`` owns its engines and borrows the GPUs of an ``InferenceResources``.
``inference_service_for`` builds the service that the ``inference.backend``
setting names.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self

from reef.core.errors import DeployConfigError
from reef.core.plugins import Plugin, PluginConfigError
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


class InferenceService(Plugin, ABC):
    """An inference component that owns its engines but borrows reservations."""

    @classmethod
    @abstractmethod
    def from_config(cls, config: Mapping[str, Any]) -> Self:
        """Build the unstarted service from its input ``config``, without allocating anything.

        Raise ``ValueError`` for an input that the service cannot run.
        """

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


@dataclass(frozen=True)
class BuiltinEngine:
    """The classes of one bundled engine, as ``package.module:ClassName`` references.

    ``service`` names the ``InferenceService`` that runs the engine on an inference component. ``handler`` names
    the ``InferenceHandler`` that speaks the engine's request dialect. They are references so that reading
    ``BUILTIN_ENGINES`` imports no engine integration.
    """

    service: str
    handler: str


#: The bundled engines, by the name that ``inference.backend`` gives them.
BUILTIN_ENGINES: Mapping[str, BuiltinEngine] = {
    "sglang": BuiltinEngine(
        "reef.inference.sglang.service:SGLangInferenceService", "reef.inference.sglang.chat:SGLangInferenceHandler"
    ),
    "vllm": BuiltinEngine(
        "reef.inference.vllm.service:VLLMInferenceService", "reef.inference.vllm.chat:VLLMInferenceHandler"
    ),
}

#: The entry point group whose entries name other ``InferenceService`` classes.
INFERENCE_BACKEND_GROUP = "reef.inference_backends"


def inference_service_for(name: str | None, config: Mapping[str, Any]) -> InferenceService:
    """The unstarted service of the ``inference.backend`` value ``name`` (default ``sglang``), built from ``config``.

    ``name`` is a bundled engine, an installed ``reef.inference_backends`` entry point, or a
    ``package.module:ClassName`` reference; each names an ``InferenceService`` class, built with its
    ``from_config``. Raise ``DeployConfigError`` for another name.
    """
    try:
        service_class = InferenceService.resolve_class(
            name or "sglang",
            builtins={engine: classes.service for engine, classes in BUILTIN_ENGINES.items()},
            entry_point_group=INFERENCE_BACKEND_GROUP,
            setting="inference backend",
        )
    except PluginConfigError as exc:
        raise DeployConfigError(str(exc)) from exc
    return service_class.from_config(config)
