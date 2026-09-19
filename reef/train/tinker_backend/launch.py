"""Tinker deployment definition and runtime factory; no SDK import at discovery.

Two topologies share one backend name. Without a local inference engine,
the HTTP service runs Tinker's training and sampling in process. With
``inference.backend: sglang``, Reef's model driver runs the local engines and
its coordinator, and Tinker trains behind that coordinator as a hosted
sender whose "weight transfer" is loading adapter files into the engines.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from reef.core.config import config_value
from reef.core.errors import DeployConfigError
from reef.runtime.deployment import CoordinatorConfig, RuntimeConfigError, RuntimeFactory, RuntimeRegistry
from reef.runtime.executor.arguments import normalize_native_options
from reef.runtime.executor.connection import DEFAULT_ACTOR_NAME, DEFAULT_NAMESPACE
from reef.runtime.executor.placement import ModelGpuLayout
from reef.runtime.interfaces import InferenceRuntime, TrainingRuntime
from reef.train.algos.registry import loss_family_refs
from reef.train.deployment import (
    InProcessTrainingDeployment,
    TrainingDeploymentPlan,
    driver_service,
    inference_handler_factory_for,
    prepare_inference_config,
    require_ray_roles,
)
from reef.train.tinker_backend.config import TinkerConfig

#: Mirrors reef.train.tinker_backend.losses without importing it at discovery time.
TINKER_LOSS_BACKEND = "tinker"
BUILTIN_TINKER_LOSSES = frozenset({"importance_sampling"})


def _local_engine_selected(settings: Mapping[str, Any]) -> bool:
    return any(
        settings.get(name) is not None for name in ("inference_backend", "inference_num_gpus", "tensor_parallel_size")
    )


def tinker_settings(options: Mapping[str, Any]) -> TinkerConfig:
    """Parse ``training.options`` into the Tinker settings, in either topology."""
    values = {key.replace("-", "_"): value for key, value in normalize_native_options(options).items()}
    try:
        return TinkerConfig(**values)
    except (TypeError, ValueError) as exc:
        raise DeployConfigError(f"training.options: {exc}") from exc


class TinkerDeployment(InProcessTrainingDeployment):
    runtime_type = "reef.train.tinker_backend.launch:runtime_factory"
    requires_local_model = False

    def prepare(self, config: dict[str, Any], settings: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
        if not _local_engine_selected(settings):
            return super().prepare(config, settings)
        backend = settings["inference_backend"]
        if backend not in (None, "sglang") and ":" not in str(backend):
            # A dotted factory receives the same SGLang-shaped inference input and interprets it itself.
            raise DeployConfigError("Tinker serves through a local engine only with inference.backend: sglang")
        if settings["inference_url"] or config.get("reef", {}).get("runtime"):
            raise DeployConfigError("a managed local engine discovers its inference connection from the driver")
        if settings["colocate"]:
            raise DeployConfigError("a hosted trainer has no training GPUs; remove training.colocate")
        options = normalize_native_options(settings["training_backend_options"])
        tinker = tinker_settings(options)
        require_ray_roles(config, "rollout", backend="Tinker")
        prepare_inference_config(config, settings, options)
        engine_options = config["reef"]["inference_options"]
        for name in ("enable-lora", "max-lora-rank", "max-loaded-loras", "max-loras-per-batch"):
            if name in engine_options or name.replace("-", "_") in engine_options:
                raise DeployConfigError(f"inference.options.{name} is set from training.options for Tinker adapters")
        config["reef"].update(
            training_backend_options=options,
            ray_namespace=settings["ray_namespace"] or DEFAULT_NAMESPACE,
            ray_actor_name=settings["ray_actor_name"] or DEFAULT_ACTOR_NAME,
            inference_handler_factory=settings["inference_handler_factory"],
        )
        if not os.environ.get(tinker.api_key_env, "").strip():
            raise DeployConfigError(f"Tinker requires the environment variable {tinker.api_key_env}")
        return (driver_service("tinker-driver", settings, {}),)

    def create_training_plan(self, config: Mapping[str, Any], *, loss_family: str) -> TrainingDeploymentPlan:
        from reef.train.tinker_backend.training import (
            TinkerDeploymentResources,
            TinkerTrainingService,
            sglang_inference_config,
        )

        ray_address = os.environ.get("RAY_ADDRESS", "").strip()
        if not ray_address:
            raise RuntimeError("RAY_ADDRESS is required")
        namespace = os.environ.get("REEF_RAY_NAMESPACE", DEFAULT_NAMESPACE)
        actor_name = os.environ.get("REEF_RAY_ACTOR_NAME", DEFAULT_ACTOR_NAME)
        reef = config.get("reef", {})
        model = config_value(config, "reef", "model_path")
        if not isinstance(model, str) or not model:
            raise RuntimeError("reef.model_path is required")
        tinker = tinker_settings(reef.get("training_backend_options") or {})
        key = os.environ.get(tinker.api_key_env, "").strip()
        if not key:
            raise RuntimeError(f"Tinker requires the environment variable {tinker.api_key_env}")
        # The recipe is imported in this process; hand the coordinator's process the same reference.
        reference = loss_family_refs(TINKER_LOSS_BACKEND).get(loss_family)
        if reference is None and loss_family not in BUILTIN_TINKER_LOSSES:
            raise RuntimeError(f"recipe loss family {loss_family!r} has no Tinker implementation registered")
        return TrainingDeploymentPlan(
            resources=TinkerDeploymentResources(
                ModelGpuLayout(0, int(reef["inference_num_gpus"])), ray_address=ray_address, namespace=namespace
            ),
            training=TinkerTrainingService(model, tinker, key, loss_family=loss_family, loss_reference=reference),
            inference_config=sglang_inference_config({**reef, "model_path": model}, tinker),
            coordinator=CoordinatorConfig(
                options={
                    "name": actor_name,
                    "namespace": namespace,
                    "max_concurrency": 64,
                    # The coordinator's worker imports the trainer and the recipe's loss from this driver's path.
                    "runtime_env": {"env_vars": {"PYTHONPATH": os.environ.get("PYTHONPATH", "")}},
                }
            ),
        )

    def runtime_config(
        self, settings: Mapping[str, Any], *, max_staleness: int, connector: Any = None
    ) -> dict[str, Any]:
        if not _local_engine_selected(settings):
            return super().runtime_config(settings, max_staleness=max_staleness, connector=connector)
        ray_address = settings["ray_address"]
        if not isinstance(ray_address, str) or not ray_address.strip():
            raise ValueError("reef.ray_address is required")
        if settings["inference_timeout_s"] <= 0:
            raise ValueError("reef.inference_timeout_s must be positive")
        if settings["train_timeout_s"] is not None and settings["train_timeout_s"] <= 0:
            raise ValueError("reef.train_timeout_s must be positive when set")
        runtime_config: dict[str, Any] = {
            "type": "coordinator_training",
            "inference_runtime": settings["inference_backend"] or "sglang",
            "actor_name": settings["ray_actor_name"],
            "namespace": settings["ray_namespace"],
            "ray_address": ray_address,
            "inference_timeout_s": settings["inference_timeout_s"],
            "train_timeout_s": settings["train_timeout_s"],
        }
        if max_staleness:
            runtime_config["max_staleness"] = max_staleness
        handler_factory = inference_handler_factory_for(settings["inference_handler_factory"])
        if handler_factory is not None:
            runtime_config["inference_handler_factory"] = handler_factory
        if not isinstance(settings["inference_handler_config"], Mapping):
            raise ValueError("reef.inference_handler_config must be an object")
        if settings["inference_handler_config"]:
            runtime_config["inference_handler_config"] = dict(settings["inference_handler_config"])
        if connector is not None:
            runtime_config["connect"] = connector
        return runtime_config


class TinkerRuntimeFactory(RuntimeFactory):
    """Build the in-process trainer and the separately registered Tinker sampling runtime."""

    kind = TinkerDeployment.runtime_type

    def config_type(self) -> type:
        return TinkerConfig

    def __call__(
        self, config: Mapping[str, Any], model_path: str, recipe_config: Mapping[str, Any], environ: Mapping[str, str]
    ) -> tuple[TrainingRuntime, InferenceRuntime]:
        from reef.train.tinker_backend.client import TinkerSDKClient
        from reef.train.tinker_backend.runtime import TinkerTrainingRuntime

        settings = TinkerConfig(**{key: value for key, value in config.items() if key != "type"})
        key = environ.get(settings.api_key_env)
        if not key:
            raise ValueError(f"Tinker requires the environment variable {settings.api_key_env}")
        client = TinkerSDKClient(model_path, settings, key)
        try:
            training = TinkerTrainingRuntime(model_path, settings, client)
        except BaseException:
            client.close()
            raise
        try:
            inference = RuntimeRegistry().build(
                {
                    "type": "tinker",
                    "api_key_env": settings.api_key_env,
                    "project_id": settings.project_id,
                    "timeout_s": settings.inference_timeout_s,
                },
                model_path=model_path,
                recipe_config=recipe_config,
                environ=environ,
            )
            if not isinstance(inference, InferenceRuntime):
                raise RuntimeConfigError("the Tinker sampling factory must return an InferenceRuntime")
        except BaseException:
            training.shutdown()
            raise
        return training, inference


runtime_factory = TinkerRuntimeFactory()
