"""Slime's deployment definition and its runtime factory, without GPU imports at discovery.

Slime trains in the HTTP service's process: the runtime factory builds a
``SlimeTrainingComponent`` that owns its Ray session and training workers.
Serving is not Slime's: the deployment's ``inference`` section configures it
(``reef.service.deploy.serving``), and an activated checkpoint reaches the
serving engines as files. A distillation teacher runs as its own service.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from typing import Any

from reef.core.config import config_value, interpolate_config
from reef.core.errors import DeployConfigError
from reef.runtime.deployment import (
    RuntimeConfigError,
    RuntimeRegistry,
    TrainingComponentFactory,
    http_readiness_command,
)
from reef.runtime.executor.arguments import MANAGED_ENGINE_OPTIONS, native_arguments, normalize_native_options
from reef.runtime.executor.config import role_executor_settings, select_executor
from reef.runtime.interfaces import InferenceRuntime, SharedGpuInferenceRuntime
from reef.runtime.jobs import TrainingComponent
from reef.train.deployment import TrainingDeployment

# These alter engine creation or topology outside Reef's inference request.
# Reject argparse abbreviations as well as full names, even for omitted values.
INFERENCE_LAUNCH_OPTIONS = frozenset(
    {
        "rollout-num-gpus",
        "rollout-num-gpus-per-engine",
        "rollout-external",
        "rollout-external-engine-addrs",
        "prefill-num-servers",
    }
)
# Reef owns GPU sharing; native rollout placement cannot replace its handoff.
PLACEMENT_OPTIONS = frozenset({"colocate", "offload-rollout"})


def refuse_serving_options(training_options: Mapping[str, Any]) -> None:
    """Refuse native training options that would configure serving engines or share GPUs with them.

    Serving engines are configured by ``inference``. Shared placement uses
    ``training.colocate`` and Reef's admission control, not Slime's rollout
    manager. Raise ``DeployConfigError`` for native options that bypass either.
    """
    for name in training_options:
        if (
            name.startswith(("sglang-", "router-"))
            or "sglang".startswith(name)
            or "router".startswith(name)
            or any(flag.startswith(name) for flag in INFERENCE_LAUNCH_OPTIONS)
        ):
            raise DeployConfigError(
                f"training.options.{name} configures inference; use inference.num-gpus, "
                "inference.tensor-parallel-size or inference.options instead"
            )
        if any(flag.startswith(name) for flag in PLACEMENT_OPTIONS):
            raise DeployConfigError(f"training.options.{name} shares GPUs between training and serving; remove it")


def require_ray_roles(config: dict[str, Any], *roles: str, backend: str) -> None:
    """Default the named execution roles to Ray and refuse any other executor for them."""
    execution = config.setdefault("execution", {})
    for role in roles:
        execution.setdefault(role, "ray")
        if select_executor(role_executor_settings(config, role), role=role).settings.backend != "ray":
            raise DeployConfigError(f"automatic {backend} training requires execution.{role}.backend: ray")


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

    Slime starts no rollout engine (``--rollout-num-gpus 0``); the serving
    engines are the inference section's. References such as
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
    return [*native_arguments(training_options), *native_arguments({"rollout-num-gpus": 0})]


#: The teacher engine's service name; the HTTP service, where Slime trains, reads its endpoint as
#: ``${endpoints.distill-teacher}``.
TEACHER_SERVICE = "distill-teacher"
#: SGLang options the teacher section sets itself or that would change which model it serves.
TEACHER_RESERVED_OPTIONS = MANAGED_ENGINE_OPTIONS | {
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
    """Train with Slime in the HTTP service; the inference section serves its activated checkpoints."""

    supports_teacher_engine = True
    component_type = "reef.train.slime_backend.launch:component_factory"

    def prepare(self, config: dict[str, Any], settings: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
        model = config_value(config, "reef", "model_path")
        require_ray_roles(config, "training", backend="Slime")
        options = normalize_native_options(settings["training_backend_options"])
        native_arguments(options)
        # Refuse a missing batch size at deploy time; the HTTP service reads it again, resolved.
        global_batch_size(expand_option_references(config, {"global-batch-size": options.get("global-batch-size")}))
        refuse_serving_options(options)
        if settings["training_colocate"]:
            if settings["inference_backend"] != "sglang" or settings["inference_remote"]:
                raise DeployConfigError("training.colocate requires local SGLang inference")
            if settings["inference_serving"] not in (None, "engines"):
                raise DeployConfigError("training.colocate requires inference.serving: engines")
            if settings["teacher_model_path"] or settings["inference_handler_factory"]:
                raise DeployConfigError("training.colocate does not support teachers or custom inference handlers")
            if settings["inference_prefix_sharing"]:
                raise DeployConfigError("training.colocate requires inference.prefix-sharing: false")
            shared_options = expand_option_references(config, options)
            dimensions: dict[str, int] = {}
            for name, default in (("actor-num-nodes", 1), ("actor-num-gpus-per-node", 1), ("megatron-lora-rank", 0)):
                value = shared_options.get(name, default)
                if isinstance(value, str) and value.strip().isdigit():
                    value = int(value)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise DeployConfigError(f"training.options.{name} must be a non-negative integer")
                dimensions[name] = value
            nodes = dimensions["actor-num-nodes"]
            width = dimensions["actor-num-gpus-per-node"]
            if nodes != 1 or settings["inference_num_gpus"] != width or settings["tensor_parallel_size"] != width:
                raise DeployConfigError("training.colocate requires one node and one engine on all actor GPUs")
            if dimensions["megatron-lora-rank"] or shared_options.get("release-train"):
                raise DeployConfigError("training.colocate supports full-weight training with persistent workers only")
            if shared_options.get("offload-train") is False or shared_options.get("no-offload-train"):
                raise DeployConfigError("training.colocate requires offload-train")
            options["offload-train"] = True
            require_ray_roles(config, "rollout", backend="Slime colocate")
        if options.get("use-rollout-routing-replay"):
            raise DeployConfigError(
                "training.options.use-rollout-routing-replay needs each rollout's routed experts, "
                "which Reef's served samples do not record"
            )
        checkpoint = options.get("hf-checkpoint")
        if checkpoint is not None and (
            not isinstance(checkpoint, str)
            or os.path.expanduser(interpolate_config(config, checkpoint).strip()) != model
        ):
            raise DeployConfigError("training.options.hf-checkpoint must match inference.model-path")
        # Resolve/download the model once; Slime and the serving engines read that same path.
        options["hf-checkpoint"] = "${reef.model_path}"
        config["reef"].update(training_backend_options=options)
        # The HTTP service loads the model into Slime's workers before it serves its first request.
        config["ready_timeout"] = max(int(config.get("ready_timeout", 30)), int(settings["training_ready_timeout"]))
        teacher = teacher_service(settings)
        return () if teacher is None else (teacher,)

    def http_environment(self, settings: Mapping[str, Any]) -> Mapping[str, str]:
        if not settings.get("teacher_model_path"):
            return {}
        return {TEACHER_URL_ENV: f"${{endpoints.{TEACHER_SERVICE}}}"}

    def component_config(self, settings: Mapping[str, Any], *, loss_family: str) -> dict[str, Any]:
        return {
            "type": self.component_type,
            "loss_family": loss_family,
            "recipe": settings["recipe"],
            "ray_address": settings["ray_address"] or os.environ.get("RAY_ADDRESS", ""),
            "namespace": settings["ray_namespace"],
            "execution": dict(settings["execution_config"]),
            "train_timeout_s": settings["train_timeout_s"],
            "colocate": settings["training_colocate"],
            "settings": {
                key: settings[key]
                for key in ("model_path", "training_backend_options", "teacher_model_path", "teacher_timeout")
            },
        }

    def build_serving(
        self, component: TrainingComponent, config: Mapping[str, Any], *, model_path: str
    ) -> InferenceRuntime:
        from reef.train.slime_backend.component import SlimeTrainingComponent

        if isinstance(component, SlimeTrainingComponent) and component.colocate:
            if component.resources is None:
                raise RuntimeConfigError("colocated Slime has no GPU reservation")
            serving = RuntimeRegistry().build(config, model_path=model_path, resources=component.resources)
            if not isinstance(serving, SharedGpuInferenceRuntime):
                serving.shutdown()
                raise RuntimeConfigError("colocated Slime requires a shared GPU serving runtime")
            component.serving = serving
            component.start_jobs()
            return serving
        return super().build_serving(component, config, model_path=model_path)


class SlimeComponentFactory(TrainingComponentFactory):
    """Build the Slime training component."""

    kind = SlimeDeployment.component_type

    def __call__(self, config: Mapping[str, Any], model_path: str, environ: Mapping[str, str]) -> TrainingComponent:
        from reef.train.slime_backend.component import SlimeSettings, create_slime_component
        from reef.train.slime_backend.reef_adapters.train_groups import DEFAULT_TRAIN_TIMEOUT_S

        settings = config["settings"]
        ray_address = config["ray_address"]
        if not isinstance(ray_address, str) or not ray_address.strip():
            raise RuntimeConfigError("Slime trains on a Ray cluster: set training.ray-address or RAY_ADDRESS")
        # SlimeSettings checks that the timeout is positive.
        train_timeout_s = config["train_timeout_s"]
        return create_slime_component(
            SlimeSettings(
                arguments=tuple(slime_arguments(settings, loss_family=config["loss_family"], environ=environ)),
                loss_family=config["loss_family"],
                recipe=config["recipe"],
                execution=config["execution"],
                ray_address=ray_address,
                namespace=config["namespace"],
                colocate=config.get("colocate", False),
                train_timeout_s=DEFAULT_TRAIN_TIMEOUT_S if train_timeout_s is None else float(train_timeout_s),
            )
        )


component_factory = SlimeComponentFactory()
