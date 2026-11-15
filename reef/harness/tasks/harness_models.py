"""The models of a harness evolution deployment: where the model under test answers, and the method's own models.

``HarnessModels`` makes the ``ModelBindings`` of one scenario's runs. The model under test is the deployment's
inference runtime; with the service's address known, a scenario's runs reach it through this Reef's evaluation
route, so they sample the snapshot that the scenario serves.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from urllib.parse import quote

from reef.core.errors import ConfigError
from reef.core.model_metadata import ModelMetadata
from reef.harness.tasks.model_binding import ModelBinding, ModelBindings
from reef.inference.interfaces import InferenceRuntime


@dataclass(frozen=True)
class HarnessModels:
    """The models that the runs of one harness evolution deployment call.

    Values: ``runtime`` is the deployment's inference runtime, the model under test, ``None`` when the deployment
    has none; ``model_name`` names its model when the runtime names none. ``adapter`` is the harness adapter, and
    ``model_metadata`` the configured metadata by model name, which only the ``codex`` adapter binds. ``named`` are
    the method's own models (``evolution.models``), none named ``served``. ``service_url`` is the base URL at which
    this Reef answers inference, ``None`` when a run calls the runtime's own endpoint; ``service_token`` is a bearer
    token that the service accepts, and ``service_component`` names the snapshot component that the harness is when
    the snapshot has several.
    Purpose: the owner of the runtime and of the service's address makes one. ``HarnessTrainingComponent`` reads
    ``bindings`` for each backend, and the harness service reads ``served`` for the model a client's layer calls.
    """

    runtime: InferenceRuntime | None
    adapter: str
    model_name: str | None = None
    model_metadata: Mapping[str, ModelMetadata] = field(default_factory=dict)
    named: Mapping[str, ModelBinding] = field(default_factory=dict)
    service_url: str | None = None
    service_token: str | None = field(default=None, repr=False)
    service_component: str | None = None

    def __post_init__(self) -> None:
        if "served" in self.named:
            raise ValueError("HarnessModels.named may not name a model 'served'; that is the model under test")

    def served(self, scenario: str | None = None) -> ModelBinding:
        """The served model's endpoint.

        With ``service_url`` set and a scenario named, this Reef's own evaluation route for that scenario: a run
        then samples the snapshot the scenario serves, weights, request defaults and all, and its calls are kept
        by nobody. A component of a composed snapshot names itself in the route, so the snapshot's hooks for the
        component a candidate replaces stay out. Otherwise the runtime's own endpoint. Raise ``ConfigError``
        without a runtime, and for a runtime that names no endpoint a harness can call.
        """
        if self.runtime is None:
            raise ConfigError(
                "harness evolution requires an inference runtime: set reef.upstream_url (and reef.upstream_model) "
                "in the deployment config"
            )
        try:
            binding = ModelBinding.from_runtime(self.runtime, model=self.model_name)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        if self.adapter == "codex":
            # Only Codex needs discovery; other adapters' startup stays independent of /models.
            binding = binding.with_metadata(self.model_metadata.get(binding.model))
        if self.service_url is None or scenario is None:
            return binding
        # The name is free form: quoted, so a slash or a space stays one segment.
        route = f"{self.service_url}/reef/scenarios/{quote(scenario, safe='')}"
        if self.service_component is not None:
            route += f"/components/{quote(self.service_component, safe='')}"
        return replace(binding, base_url=f"{route}/evaluation", api_key=self.service_token)

    def bindings(self, scenario: str | None = None) -> ModelBindings:
        """The model bindings of ``scenario``'s runs: ``served(scenario)`` and the named models."""
        return ModelBindings(served=self.served(scenario), named=dict(self.named))


__all__ = ["HarnessModels"]
