"""Lightweight Tinker deployment and runtime factory; no SDK import at discovery."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reef.runtime.base import InferenceRuntime
from reef.runtime.registry import RuntimeFactory
from reef.train.deployment import InProcessTrainingDeployment
from reef.train.tinker_backend.config import TinkerConfig


class TinkerDeployment(InProcessTrainingDeployment):
    runtime_type = "reef.train.tinker_backend.launch:runtime_factory"
    requires_local_model = False


class TinkerRuntimeFactory(RuntimeFactory):
    kind = TinkerDeployment.runtime_type

    def config_type(self) -> type:
        return TinkerConfig

    def __call__(
        self, config: Mapping[str, Any], model_path: str, recipe_config: Mapping[str, Any], environ: Mapping[str, str]
    ) -> InferenceRuntime:
        from reef.train.tinker_backend.client import TinkerSDKClient
        from reef.train.tinker_backend.runtime import TinkerRuntime

        settings = TinkerConfig(**{key: value for key, value in config.items() if key != "type"})
        key = environ.get(settings.api_key_env)
        if not key:
            raise ValueError(f"Tinker requires the environment variable {settings.api_key_env}")
        client = TinkerSDKClient(model_path, settings, key)
        try:
            return TinkerRuntime(model_path, settings, client)
        except BaseException:
            client.close()
            raise


runtime_factory = TinkerRuntimeFactory()
