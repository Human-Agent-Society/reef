"""Tinker and Spindle full-parameter deployment definitions and runtime factories; no SDK import at discovery.

Tinker trains in the HTTP service's process: ``TinkerTrainingComponent`` runs
each job. Serving is the deployment's ``inference`` section
(``reef.service.deploy.serving``). By default Tinker samples too
(``inference.serving: trainer``, ``reef.inference.tinker``). Reef's engines can
serve the checkpoints instead: they load each release's adapter, which
``reef.tinker.weights.TinkerSamplerWeights`` downloads from the manifest's
sampler path.

``spindle_full`` trains all the parameters on a Spindle deployment the same way
(``SpindleFullTrainingComponent``, RFC #996). Spindle's sampler serves its
releases; Reef's engines cannot, since Spindle serves no full weights to
download.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from reef.core.config import config_arguments, parse_config_values
from reef.core.errors import DeployConfigError
from reef.runtime.deployment import TrainingComponentFactory
from reef.runtime.executor.arguments import normalize_native_options
from reef.runtime.jobs import ReleaseFormat, TrainingComponent
from reef.train.deployment import EngineWeights, InProcessTrainingDeployment
from reef.train.tinker_backend.config import SpindleFullConfig, TinkerConfig, TinkerServiceConfig


def service_settings[T: TinkerServiceConfig](options: Mapping[str, Any], config_type: type[T]) -> T:
    """Parse ``training.options`` into ``config_type``, with the declared types (CLI and YAML values are text)."""
    values = {key.replace("-", "_"): value for key, value in normalize_native_options(options).items()}
    try:
        return config_type(
            **parse_config_values(config_arguments(config_type, prefix=("training", "options")), values)
        )
    except (TypeError, ValueError) as exc:
        raise DeployConfigError(f"training.options: {exc}") from exc


def tinker_settings(options: Mapping[str, Any]) -> TinkerConfig:
    """Parse ``training.options`` into the Tinker settings."""
    return service_settings(options, TinkerConfig)


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

    def engine_weights(self, settings: Mapping[str, Any]) -> EngineWeights:
        """Each release's adapter at its sampler path, downloaded into ``state_dir/sampler-weights``."""
        tinker = tinker_settings(settings["training_backend_options"])
        return EngineWeights(
            ReleaseFormat("adapter", tinker.lora_rank, ("all",)),
            {
                "module": "reef.tinker.weights:TinkerSamplerWeights",
                "config": {
                    "cache_dir": str(Path(tinker.state_dir).expanduser() / "sampler-weights"),
                    "api_key_env": tinker.api_key_env,
                    "base_url": tinker.base_url,
                    "project_id": tinker.project_id,
                    "timeout_s": tinker.inference_timeout_s,
                },
            },
        )


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
        client = TinkerSDKClient(model_path, settings, key, lora_rank=settings.lora_rank, seed=settings.seed)
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


def spindle_full_settings(options: Mapping[str, Any]) -> SpindleFullConfig:
    """Parse ``training.options`` into the Spindle full-parameter settings."""
    return service_settings(options, SpindleFullConfig)


class SpindleFullDeployment(InProcessTrainingDeployment):
    """Train all the parameters on a Spindle deployment in the HTTP service; serve through Spindle's sampler.

    The deployment's model is the base's repo (``training.options.base``), which the Spindle deployment's
    full-parameter engine names as its ``model``.
    """

    component_type = "reef.train.tinker_backend.launch:spindle_full_component_factory"
    requires_local_model = False
    default_serving = "trainer"

    def prepare(self, config: dict[str, Any], settings: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
        spindle = spindle_full_settings(settings["training_backend_options"])
        if not os.environ.get(spindle.api_key_env, "").strip():
            raise DeployConfigError(f"Spindle requires the environment variable {spindle.api_key_env}")
        return super().prepare(config, settings)

    def trainer_serving_config(self, settings: Mapping[str, Any]) -> dict[str, Any]:
        """Spindle's sampler, which serves each full-parameter release with the training settings' key."""
        spindle = spindle_full_settings(settings["training_backend_options"])
        return {
            "type": "tinker",
            "api_key_env": spindle.api_key_env,
            "project_id": spindle.project_id,
            "base_url": spindle.base_url,
            "timeout_s": spindle.inference_timeout_s,
            "parameterization": "full",
        }


class SpindleFullComponentFactory(TrainingComponentFactory):
    """Build the in-process Spindle full-parameter training component."""

    kind = SpindleFullDeployment.component_type

    def config_type(self) -> type:
        return SpindleFullConfig

    def __call__(self, config: Mapping[str, Any], model_path: str, environ: Mapping[str, str]) -> TrainingComponent:
        from reef.tinker.sampling import TinkerSDKSampler
        from reef.train.tinker_backend.client import TinkerSDKClient
        from reef.train.tinker_backend.full import SpindleFullTrainingComponent

        settings = SpindleFullConfig(**{key: value for key, value in config.items() if key != "type"})
        key = environ.get(settings.api_key_env)
        if not key:
            raise ValueError(f"Spindle requires the environment variable {settings.api_key_env}")
        client = TinkerSDKClient(model_path, settings, key, lora_rank=None)
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
            return SpindleFullTrainingComponent(model_path, settings, client, sampler=sampler)
        except BaseException:
            client.close()
            if sampler is not None:
                sampler.close()
            raise


spindle_full_component_factory = SpindleFullComponentFactory()
