"""The inference component: its contracts, and the selection of its implementation without training code.

An ``InferenceService`` owns its engines and borrows ``InferenceResources``.
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


class InferenceService(Plugin, ABC):
    """An inference component that owns its engines but borrows reservations."""

    @classmethod
    @abstractmethod
    def from_config(cls, config: Mapping[str, Any]) -> Self:
        """Build the unstarted service from its input ``config``, without allocating anything.

        Raise ``ValueError`` for an input that the service cannot run.
        """

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
