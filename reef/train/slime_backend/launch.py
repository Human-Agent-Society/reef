"""Slime's deployment definition and its runtime factory, without GPU imports at discovery.

Slime trains in the HTTP service's process: the runtime factory builds a
``SlimeTrainingComponent`` that owns its Ray session and training workers.
Reef's serving engines run on an inference component, which ``reef serve``
starts on the same Ray cluster unless ``inference.remote`` names one; a
selected checkpoint reaches them as files. A distillation teacher runs as its
own service.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from typing import Any

from reef.core.config import config_value, interpolate_config
from reef.core.errors import DeployConfigError
from reef.runtime.deployment import RuntimeConfigError, RuntimeFactory, RuntimeRegistry, http_readiness_command
from reef.runtime.executor.arguments import native_arguments, normalize_native_options
from reef.runtime.interfaces import InferenceRuntime
from reef.runtime.jobs import TrainingComponent
from reef.train.deployment import (
    INFERENCE_RESERVED_OPTIONS,
    NO_INFERENCE_BACKEND,
    TrainingDeployment,
    inference_handler_factory_for,
    prepare_inference_config,
    require_ray_roles,
)


def global_batch_size(options: Mapping[str, Any]) -> int:
    """Read ``--global-batch-size`` from normalized, resolved Slime ``options``: the samples in one training step.

    The component cuts each job's data into optimizer steps of this size, so
    the option must be explicit, even when Slime could derive it from
    ``--num-steps-per-rollout``. The value is a
    positive int or its decimal string.
    """
    value = options.get("global-batch-size")
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DeployConfigError(
            f"training.options.global-batch-size must be a positive integer, the samples in one training step; "
            f"got {value!r}"
        )
    return value


OptionValue = bool | int | float | str | list["OptionValue"] | dict[str, "OptionValue"] | None


def expand_option_references(
    config: Mapping[str, object], options: Mapping[str, OptionValue]
) -> dict[str, OptionValue]:
    """Resolve configuration references in nested native training and inference options."""

    def _expand(value: OptionValue) -> OptionValue:
        if isinstance(value, str):
            return interpolate_config(config, value)
        if isinstance(value, dict):
            return {key: _expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_expand(item) for item in value]
        return value

    return {name: _expand(value) for name, value in options.items()}


#: The variable that carries the teacher engine's URL to the HTTP service, where Slime trains.
TEACHER_URL_ENV = "REEF_DISTILL_TEACHER_URL"


def slime_arguments(settings: Mapping[str, Any], *, loss_family: str, environ: Mapping[str, str]) -> list[str]:
    """Adapt the resolved service settings to the pinned Slime parser.

    Keep generated inference flags out of training.options. The serving
    engines' size reaches Slime as its rollout flags, from which Reef builds the
    engines' input; Slime itself starts no engine. References such as
    ``${reef.model_path}`` in the options resolve against ``settings``, and the
    teacher's URL comes from ``TEACHER_URL_ENV`` in ``environ``.
    """
    config = {"reef": dict(settings)}
    training_options = expand_option_references(config, settings.get("training_backend_options") or {})
    if settings.get("teacher_model_path"):
        if not loss_family:
            raise DeployConfigError("a teacher engine requires a distillation loss family")
        managed = {
            f"{loss_family}-teacher-url": environ[TEACHER_URL_ENV],
            f"{loss_family}-teacher-model-path": settings["teacher_model_path"],
            f"{loss_family}-teacher-timeout": settings.get("teacher_timeout", 300.0),
        }
        if any(name in training_options for name in (*managed, f"{loss_family}-teacher-checkpoint")):
            raise DeployConfigError("teacher section replaces native teacher URL, model path, timeout and checkpoint")
        training_options.update(managed)
    arguments = native_arguments(training_options)
    if settings.get("inference_num_gpus") is None:
        return arguments
    if settings.get("inference_backend") == NO_INFERENCE_BACKEND:
        return [*arguments, *native_arguments({"rollout-num-gpus": 0})]
    options = {
        "rollout-num-gpus": settings["inference_num_gpus"],
        "rollout-num-gpus-per-engine": settings["tensor_parallel_size"],
    }
    if settings.get("inference_backend") == "vllm":
        # vLLM engine options reach the driver through the resolved config;
        # Slime's parser only knows SGLang's names.
        return [*arguments, *native_arguments(options)]
    for name, value in expand_option_references(config, settings.get("inference_options") or {}).items():
        # Slime has dedicated router bind flags and passes other router flags
        # directly to RouterArgs. Engine flags are all prefixed by Slime.
        flag = (
            name
            if name.startswith("router-") and name not in {"router-ip", "router-port", "router-request-timeout-secs"}
            else "sglang-" + name
        )
        options[flag] = value
    return [*arguments, *native_arguments(options)]


#: The teacher engine's service name; its endpoint reaches the driver as ``${endpoints.distill-teacher}``.
TEACHER_SERVICE = "distill-teacher"
#: SGLang options the teacher section sets itself or that would change which model it serves.
TEACHER_RESERVED_OPTIONS = frozenset(INFERENCE_RESERVED_OPTIONS) | {
    "tokenizer-path",
    "tokenizer-mode",
    "revision",
    "tokenizer-revision",
    "enable-lora",
    "lora-paths",
    "load-format",
    "enable-mis",
    "skip-tokenizer-init",
}


def teacher_service(settings: Mapping[str, Any]) -> dict[str, Any] | None:
    """The teacher engine as a Ray service with its own GPUs, stopped with its deployment."""
    if not settings.get("teacher_model_path"):
        if settings.get("teacher_options"):
            raise DeployConfigError("teacher.options requires teacher.model-path")
        return None
    size = settings["teacher_num_gpus"]
    port = settings["teacher_port"]
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise DeployConfigError("teacher.num-gpus must be a positive integer")
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535 or port == settings["port"]:
        raise DeployConfigError("teacher.port must be valid and distinct from reef.port")
    options = normalize_native_options(settings["teacher_options"])
    arguments = native_arguments(options, reserved=set(TEACHER_RESERVED_OPTIONS))
    python = os.environ.get("REEF_PYTHON", sys.executable)
    return {
        "name": TEACHER_SERVICE,
        "executor": "ray",
        "resources": {"num_gpus": size, "num_cpus": 1},
        "endpoint": f"http://{{host}}:{port}",
        "command": [
            python,
            "-m",
            "sglang.launch_server",
            "--model-path",
            "${reef.teacher_model_path}",
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
            "--tp",
            str(size),
            *arguments,
        ],
        "ready": http_readiness_command(python, f"${{endpoints.{TEACHER_SERVICE}}}/health"),
        "ready_timeout": settings["training_ready_timeout"],
    }


class SlimeDeployment(TrainingDeployment):
    """Train with Slime in the HTTP service, and serve its selected checkpoints on an inference component."""

    supports_teacher_engine = True
    supports_remote_inference = True
    runtime_type = "reef.train.slime_backend.launch:runtime_factory"

    def serves_on_inference_component(self, settings: Mapping[str, Any]) -> bool:
        return settings["inference_backend"] != NO_INFERENCE_BACKEND

    def prepare(self, config: dict[str, Any], settings: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
        model = config_value(config, "reef", "model_path")
        if settings["inference_url"] or config.get("reef", {}).get("runtime"):
            raise DeployConfigError("Slime reaches its serving engines through their inference component")
        if settings["inference_backend"] not in (None, "sglang", "vllm", NO_INFERENCE_BACKEND):
            raise DeployConfigError("Slime serves through inference.backend: sglang, vllm or none")
        require_ray_roles(config, "training", backend="Slime")
        options = normalize_native_options(settings["training_backend_options"])
        native_arguments(options)
        # Refuse a missing batch size at deploy time; the HTTP service reads it again, resolved.
        global_batch_size(expand_option_references(config, {"global-batch-size": options.get("global-batch-size")}))
        if settings["inference_backend"] == "vllm" and int(options.get("megatron-lora-rank", 0) or 0) > 0:
            raise DeployConfigError("inference.backend: vllm does not serve LoRA adapters yet; use full weights")
        prepare_inference_config(config, settings, options)
        reef = config["reef"]
        if settings["inference_backend"] == "vllm" and reef["inference_num_gpus"] != reef["tensor_parallel_size"]:
            # vLLM ships no router, so one engine serves.
            raise DeployConfigError(
                "inference.backend: vllm serves one engine per stack; "
                "set inference.num-gpus equal to inference.tensor-parallel-size"
            )
        checkpoint = options.get("hf-checkpoint")
        if checkpoint is not None and (
            not isinstance(checkpoint, str)
            or os.path.expanduser(interpolate_config(config, checkpoint).strip()) != model
        ):
            raise DeployConfigError("training.options.hf-checkpoint must match inference.model-path")
        # Resolve/download the model once; Slime and the serving engines read that same path.
        options["hf-checkpoint"] = "${reef.model_path}"
        reef.update(training_backend_options=options)
        # The HTTP service loads the model into Slime's workers before it serves its first request.
        config["ready_timeout"] = max(int(config.get("ready_timeout", 30)), int(settings["training_ready_timeout"]))
        teacher = teacher_service(settings)
        return () if teacher is None else (teacher,)

    def http_environment(self, settings: Mapping[str, Any]) -> Mapping[str, str]:
        if not settings.get("teacher_model_path"):
            return {}
        return {TEACHER_URL_ENV: f"${{endpoints.{TEACHER_SERVICE}}}"}

    def runtime_config(self, settings: Mapping[str, Any], *, loss_family: str) -> dict[str, Any]:
        if settings["inference_timeout_s"] <= 0:
            raise ValueError("reef.inference_timeout_s must be positive")
        if settings["train_timeout_s"] is not None and settings["train_timeout_s"] <= 0:
            raise ValueError("reef.train_timeout_s must be positive when set")
        if not isinstance(settings["inference_handler_config"], Mapping):
            raise ValueError("reef.inference_handler_config must be an object")
        runtime_config: dict[str, Any] = {
            "type": self.runtime_type,
            "loss_family": loss_family,
            "recipe": settings["recipe"],
            "ray_address": settings["ray_address"] or os.environ.get("RAY_ADDRESS", ""),
            "namespace": settings["ray_namespace"],
            "execution": dict(settings["execution_config"]),
            "train_timeout_s": settings["train_timeout_s"],
            "inference_timeout_s": settings["inference_timeout_s"],
            "settings": {
                key: settings[key]
                for key in (
                    "model_path",
                    "training_backend_options",
                    "teacher_model_path",
                    "teacher_timeout",
                    "inference_backend",
                    "inference_num_gpus",
                    "tensor_parallel_size",
                    "inference_options",
                )
            },
        }
        if self.serves_on_inference_component(settings):
            runtime_config["serving"] = {
                "remote": dict(settings["inference_remote"]),
                "engine": settings["inference_backend"] or "sglang",
                "num_gpus": int(settings["inference_num_gpus"]),
                "inference_handler_factory": settings["inference_handler_factory"],
                "inference_handler_config": dict(settings["inference_handler_config"]),
            }
        return runtime_config


class SlimeRuntimeFactory(RuntimeFactory):
    """Build the Slime training component and the runtime that serves its checkpoints."""

    kind = SlimeDeployment.runtime_type

    def __call__(
        self, config: Mapping[str, Any], model_path: str, recipe_config: Mapping[str, Any], environ: Mapping[str, str]
    ) -> tuple[TrainingComponent, InferenceRuntime]:
        from reef.train.slime_backend.component import SlimeSettings, create_slime_component
        from reef.train.slime_backend.inference import remote_inference_config
        from reef.train.slime_backend.reef_adapters.megatron.lora import lora_engine_slots, megatron_lora_enabled
        from reef.train.slime_backend.reef_adapters.train_groups import DEFAULT_TRAIN_RPC_TIMEOUT_S

        settings = config["settings"]
        ray_address = config["ray_address"]
        if not isinstance(ray_address, str) or not ray_address.strip():
            raise RuntimeConfigError("Slime trains on a Ray cluster: set training.ray-address or RAY_ADDRESS")
        train_timeout_seconds = config.get("train_timeout_s")
        if train_timeout_seconds is None:
            train_rpc_timeout_s: float = DEFAULT_TRAIN_RPC_TIMEOUT_S
        else:
            train_rpc_timeout_s = float(train_timeout_seconds)
        component = create_slime_component(
            SlimeSettings(
                arguments=tuple(slime_arguments(settings, loss_family=config["loss_family"], environ=environ)),
                loss_family=config["loss_family"],
                recipe=config["recipe"],
                execution=config["execution"],
                ray_address=ray_address,
                namespace=config["namespace"],
                train_rpc_timeout_s=train_rpc_timeout_s,
            )
        )
        try:
            serving = config.get("serving")
            if serving is None:
                inference_config: dict[str, Any] = {
                    "type": "none",
                    "inference_timeout_s": config["inference_timeout_s"],
                }
            else:
                args = component.args
                lora = megatron_lora_enabled(args)
                remote = serving["remote"]
                inference_config = {
                    "type": "remote_serving",
                    "layout": "adapter" if lora else "full",
                    "remote": remote,
                    "engine": serving["engine"],
                    "engine_config": remote_inference_config(
                        args,
                        serving["engine"],
                        settings["inference_options"],
                        model_path=remote.get("model_path") or model_path,
                    ),
                    "num_gpus": serving["num_gpus"],
                    "adapter_capacity": lora_engine_slots(args) if lora else 1,
                    "inference_timeout_s": config["inference_timeout_s"],
                    "inference_handler_config": serving["inference_handler_config"],
                }
                handler_factory = inference_handler_factory_for(serving["inference_handler_factory"])
                if handler_factory is not None:
                    inference_config["inference_handler_factory"] = handler_factory
            inference = RuntimeRegistry().build(
                inference_config, model_path=model_path, recipe_config=recipe_config, environ=environ
            )
            if not isinstance(inference, InferenceRuntime):
                raise RuntimeConfigError("the Slime serving factory must return an InferenceRuntime")
        except BaseException:
            component.close()
            raise
        return component, inference


runtime_factory = SlimeRuntimeFactory()
