"""Tinker deployment definition and runtime factory; no SDK import at discovery.

Tinker trains in the HTTP service's process: ``TinkerTrainingComponent`` runs
each job behind ``JobTrainingRuntime``. Without an inference engine, Tinker
samples too (``reef.inference.tinker``). With ``inference.backend: sglang``,
Reef's serving engines run on an inference component: the one that
``inference.remote`` names, or one that ``reef serve`` starts on this host.
The control plane loads each scenario's saved adapter into them.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from reef.core.errors import DeployConfigError
from reef.runtime.deployment import RuntimeConfigError, RuntimeFactory, RuntimeRegistry
from reef.runtime.executor.arguments import normalize_native_options
from reef.runtime.interfaces import InferenceRuntime, TrainingRuntime
from reef.train.deployment import InProcessTrainingDeployment, inference_handler_factory_for, prepare_inference_config
from reef.train.tinker_backend.config import TinkerConfig


def tinker_settings(options: Mapping[str, Any]) -> TinkerConfig:
    """Parse ``training.options`` into the Tinker settings."""
    values = {key.replace("-", "_"): value for key, value in normalize_native_options(options).items()}
    try:
        return TinkerConfig(**values)
    except (TypeError, ValueError) as exc:
        raise DeployConfigError(f"training.options: {exc}") from exc


def sglang_engine_config(
    model_path: str, num_gpus: int, tensor_parallel_size: int, options: Mapping[str, Any], config: TinkerConfig
) -> dict[str, Any]:
    """The SGLang input of engines that serve Tinker's adapters: single-node tensor-parallel engines."""
    engine_options = {key.replace("-", "_"): value for key, value in options.items()}
    engine_options.update(
        model_path=model_path,
        trust_remote_code=True,
        skip_server_warmup=True,
        enable_metrics=True,
        enable_lora=True,
        max_lora_rank=config.lora_rank,
        max_loaded_loras=config.max_loaded_adapters,
        max_loras_per_batch=config.max_loaded_adapters,
    )
    engine_options.setdefault("lora_target_modules", ["all"])
    return {
        "model_path": model_path,
        "num_gpus": num_gpus,
        "gpus_per_engine": tensor_parallel_size,
        "gpus_per_node": max(num_gpus, tensor_parallel_size),
        "options": engine_options,
        "executor": "auto",
    }


class TinkerDeployment(InProcessTrainingDeployment):
    runtime_type = "reef.train.tinker_backend.launch:runtime_factory"
    requires_local_model = False
    supports_remote_inference = True

    def serves_on_inference_component(self, settings: Mapping[str, Any]) -> bool:
        return bool(settings.get("inference_remote")) or any(
            settings.get(name) is not None
            for name in ("inference_backend", "inference_num_gpus", "tensor_parallel_size")
        )

    def prepare(self, config: dict[str, Any], settings: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
        if not self.serves_on_inference_component(settings):
            return super().prepare(config, settings)
        backend = settings["inference_backend"]
        if backend not in (None, "sglang") and ":" not in str(backend):
            # A dotted factory receives the same SGLang-shaped inference input and interprets it itself.
            raise DeployConfigError("Tinker serves through inference engines only with inference.backend: sglang")
        if settings["inference_url"] or config.get("reef", {}).get("runtime"):
            raise DeployConfigError("Tinker reaches its serving engines through their inference component")
        if settings["colocate"]:
            raise DeployConfigError("a hosted trainer has no training GPUs; remove training.colocate")
        if {"training", "rollout"} & config.get("execution", {}).keys():
            raise DeployConfigError("Tinker trains in the HTTP service's process; remove the execution roles")
        options = normalize_native_options(settings["training_backend_options"])
        tinker = tinker_settings(options)
        prepare_inference_config(config, settings, options)
        engine_options = config["reef"]["inference_options"]
        for name in ("enable-lora", "max-lora-rank", "max-loaded-loras", "max-loras-per-batch"):
            if name in engine_options or name.replace("-", "_") in engine_options:
                raise DeployConfigError(f"inference.options.{name} is set from training.options for Tinker adapters")
        if not os.environ.get(tinker.api_key_env, "").strip():
            raise DeployConfigError(f"Tinker requires the environment variable {tinker.api_key_env}")
        # The HTTP service starts the engines before it serves its first request.
        config["ready_timeout"] = max(int(config.get("ready_timeout", 30)), int(settings["training_ready_timeout"]))
        return ()

    def runtime_config(self, settings: Mapping[str, Any], *, connector: Any = None) -> dict[str, Any]:
        runtime_config = super().runtime_config(settings, connector=connector)
        if "serving" in runtime_config:
            raise DeployConfigError("training.options cannot set serving; the deployment selects the engines")
        if not self.serves_on_inference_component(settings):
            return runtime_config
        if not isinstance(settings["inference_handler_config"], Mapping):
            raise ValueError("reef.inference_handler_config must be an object")
        runtime_config["serving"] = {
            "remote": dict(settings["inference_remote"]),
            "engine": settings["inference_backend"] or "sglang",
            "num_gpus": int(settings["inference_num_gpus"]),
            "tensor_parallel_size": int(settings["tensor_parallel_size"]),
            "options": dict(settings["inference_options"] or {}),
            "inference_handler_factory": settings["inference_handler_factory"],
            "inference_handler_config": dict(settings["inference_handler_config"]),
        }
        return runtime_config


class TinkerRuntimeFactory(RuntimeFactory):
    """Build the in-process trainer and its separately registered serving runtime."""

    kind = TinkerDeployment.runtime_type

    def config_type(self) -> type:
        return TinkerConfig

    def __call__(
        self, config: Mapping[str, Any], model_path: str, recipe_config: Mapping[str, Any], environ: Mapping[str, str]
    ) -> tuple[TrainingRuntime, InferenceRuntime]:
        from reef.train.job_runtime import JobTrainingRuntime
        from reef.train.tinker_backend.client import TinkerSDKClient
        from reef.train.tinker_backend.component import TinkerTrainingComponent

        settings = TinkerConfig(**{key: value for key, value in config.items() if key != "type"})
        key = environ.get(settings.api_key_env)
        if not key:
            raise ValueError(f"Tinker requires the environment variable {settings.api_key_env}")
        serving = settings.serving
        client = TinkerSDKClient(model_path, settings, key)
        try:
            component = TinkerTrainingComponent(model_path, settings, client, export_adapter=bool(serving))
        except BaseException:
            client.close()
            raise
        # Engines serve one adapter per scenario, so each scenario trains its own adapter.
        training = JobTrainingRuntime(component, component.state_dir / "runtime", per_scenario=bool(serving))
        try:
            if serving:
                remote = serving["remote"]
                inference_config = {
                    "type": "remote_adapters",
                    "remote": remote,
                    "engine": serving["engine"],
                    "engine_config": sglang_engine_config(
                        remote.get("model_path") or model_path,
                        serving["num_gpus"],
                        serving["tensor_parallel_size"],
                        serving["options"],
                        settings,
                    ),
                    "num_gpus": serving["num_gpus"],
                    "adapter_capacity": settings.max_loaded_adapters,
                    "inference_timeout_s": settings.inference_timeout_s,
                    "inference_handler_config": serving["inference_handler_config"],
                }
                handler_factory = inference_handler_factory_for(serving["inference_handler_factory"])
                if handler_factory is not None:
                    inference_config["inference_handler_factory"] = handler_factory
            else:
                inference_config = {
                    "type": "tinker",
                    "api_key_env": settings.api_key_env,
                    "project_id": settings.project_id,
                    "timeout_s": settings.inference_timeout_s,
                }
            inference = RuntimeRegistry().build(
                inference_config, model_path=model_path, recipe_config=recipe_config, environ=environ
            )
            if not isinstance(inference, InferenceRuntime):
                raise RuntimeConfigError("the Tinker serving factory must return an InferenceRuntime")
        except BaseException:
            training.shutdown()
            raise
        return training, inference


runtime_factory = TinkerRuntimeFactory()
