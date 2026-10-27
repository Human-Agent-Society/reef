"""Lightweight training deployment contracts, separate from model execution."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Literal, get_args

from reef.core.errors import DeployConfigError
from reef.runtime.deployment import RuntimeConfigError, training_component_factory_for
from reef.runtime.executor.arguments import normalize_native_options

#: The ``inference.backend`` that runs no engine: no weights leave the trainer and no request is served.
NO_INFERENCE_BACKEND = "none"

#: Who serves a deployment's checkpoints: inference engines, or the training backend itself.
DefaultServing = Literal["engines", "trainer"]


class TrainingDeployment(ABC):
    """Lightweight integration definition; preparation never allocates model resources.

    ``default_serving`` is the serving of a deployment that names none and
    sets no engine setting: ``engines`` or ``trainer``. A subclass that sets
    another value is refused when the class is defined.
    """

    requires_local_model: bool = True
    supports_teacher_engine: bool = False
    default_serving: DefaultServing = "engines"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.default_serving not in get_args(DefaultServing):
            raise TypeError(
                f"{cls.__name__}.default_serving must be one of {get_args(DefaultServing)}, "
                f"got {cls.default_serving!r}"
            )

    @abstractmethod
    def prepare(self, config: dict[str, Any], settings: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
        """Validate integration inputs and describe dependencies of the HTTP process.

        Bind derived component values in config, using existing process definitions
        for any children. Return an empty tuple for an in-process runtime.
        """

    @abstractmethod
    def component_config(self, settings: Mapping[str, Any], *, loss_family: str) -> dict[str, Any]:
        """Describe the training component for ``build_training_component``, without constructing it.

        The config's ``type`` is the dotted reference of the component's ``TrainingComponentFactory``.

        ``loss_family`` is the recipe's startup loss family, which a backend
        that starts its workers with one loss family reads.
        """

    def http_environment(self, settings: Mapping[str, Any]) -> Mapping[str, str]:
        """Variables the HTTP service process gets, where the training component runs.

        Values may name ``${endpoints.<service>}`` of a service that
        :meth:`prepare` returned; ``reef serve`` resolves them before it starts
        the HTTP service.
        """
        return {}

    def trainer_serving_config(self, settings: Mapping[str, Any]) -> dict[str, Any] | None:
        """The runtime config of the backend's own serving (``inference.serving: trainer``), or None without one."""
        return None


class InProcessTrainingDeployment(TrainingDeployment):
    """An integration whose training component factory builds its training component in process.

    Subclasses name their ``component_type``, the dotted reference of a
    ``TrainingComponentFactory``. The factory owns option parsing and imports
    its execution dependencies only when it builds the component.
    """

    component_type: str

    def prepare(self, config: dict[str, Any], settings: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
        if {"ray_address", "ray_namespace"} & config["reef"].keys() or "training" in config.get("execution", {}):
            raise DeployConfigError(
                "in-process training owns its execution; remove Ray and training executor settings"
            )
        try:
            factory = training_component_factory_for(self.component_type)
            # Use the component's schema for both CLI and YAML options, without a model.
            factory.parse_config(self.component_config(settings, loss_family=""), os.environ)
        except RuntimeConfigError as exc:
            raise DeployConfigError(str(exc)) from exc
        return ()

    def component_config(self, settings: Mapping[str, Any], *, loss_family: str) -> dict[str, Any]:
        options = {
            key.replace("-", "_"): value
            for key, value in normalize_native_options(settings["training_backend_options"]).items()
        }
        managed = {"type", "model_path", "inference_timeout_s", "train_timeout_s", "connect"}
        if managed & options.keys():
            raise DeployConfigError("training.options cannot override managed runtime fields")
        return {
            **options,
            "type": self.component_type,
            "inference_timeout_s": settings["inference_timeout_s"],
            "train_timeout_s": settings["train_timeout_s"],
        }
