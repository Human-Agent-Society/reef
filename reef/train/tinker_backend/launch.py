"""Tinker deployment definition and runtime factory; no SDK import at discovery.

Tinker trains in the HTTP service's process: ``TinkerTrainingComponent`` runs
each job. Serving is the deployment's ``inference`` section
(``reef.service.deploy.serving``). By default Tinker samples too
(``inference.serving: trainer``, ``reef.inference.tinker``). Reef's engines can
serve the checkpoints instead when ``training.options.export-adapter`` saves
each checkpoint's PEFT adapter.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from reef.core.config import config_arguments, parse_config_values
from reef.core.errors import DeployConfigError
from reef.runtime.deployment import TrainingComponentFactory
from reef.runtime.executor.arguments import normalize_native_options
from reef.runtime.jobs import TrainingComponent
from reef.train.deployment import InProcessTrainingDeployment
from reef.train.tinker_backend.config import TinkerConfig


def tinker_settings(options: Mapping[str, Any]) -> TinkerConfig:
    """Parse ``training.options`` into the Tinker settings, with the declared types (CLI and YAML values are text)."""
    values = {key.replace("-", "_"): value for key, value in normalize_native_options(options).items()}
    try:
        return TinkerConfig(
            **parse_config_values(config_arguments(TinkerConfig, prefix=("training", "options")), values)
        )
    except (TypeError, ValueError) as exc:
        raise DeployConfigError(f"training.options: {exc}") from exc


class TinkerDeployment(InProcessTrainingDeployment):
    """Train with Tinker in the HTTP service; serve through Tinker's sampler unless the inference section says otherwise."""

    component_type = "reef.train.tinker_backend.launch:component_factory"
    requires_local_model = False
    default_serving = "trainer"

    def prepare(self, config: dict[str, Any], settings: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
        tinker = tinker_settings(settings["training_backend_options"])
        if not os.environ.get(tinker.api_key_env, "").strip():
            raise DeployConfigError(f"Tinker requires the environment variable {tinker.api_key_env}")
        return super().prepare(config, settings)

    def trainer_serving_config(self, settings: Mapping[str, Any]) -> dict[str, Any]:
        """Tinker's sampler, which serves each release's manifest with the training settings' key and project."""
        tinker = tinker_settings(settings["training_backend_options"])
        return {
            "type": "tinker",
            "api_key_env": tinker.api_key_env,
            "project_id": tinker.project_id,
            "base_url": tinker.base_url,
            "timeout_s": tinker.inference_timeout_s,
        }


class TinkerComponentFactory(TrainingComponentFactory):
    """Build the in-process Tinker training component."""

    kind = TinkerDeployment.component_type

    def config_type(self) -> type:
        return TinkerConfig

    def __call__(self, config: Mapping[str, Any], model_path: str, environ: Mapping[str, str]) -> TrainingComponent:
        from reef.tinker.sampling import TinkerSDKSampler
        from reef.train.tinker_backend.client import TinkerSDKClient
        from reef.train.tinker_backend.component import TinkerTrainingComponent

        settings = TinkerConfig(**{key: value for key, value in config.items() if key != "type"})
        key = environ.get(settings.api_key_env)
        if not key:
            raise ValueError(f"Tinker requires the environment variable {settings.api_key_env}")
        client = TinkerSDKClient(model_path, settings, key)
        sampler = None
        try:
            # The sampler of the component's own rollouts, for task jobs.
            sampler = TinkerSDKSampler(
                model_path,
                api_key=key,
                project_id=settings.project_id,
                base_url=settings.base_url,
                timeout_s=settings.inference_timeout_s,
            )
            return TinkerTrainingComponent(model_path, settings, client, sampler=sampler)
        except BaseException:
            client.close()
            if sampler is not None:
                sampler.close()
            raise


component_factory = TinkerComponentFactory()
