"""How a weight-training deployment serves its releases, from its ``inference`` settings alone.

The serving runtime is chosen and configured here, not by the training
backend. ``serving_mode`` reads ``inference.serving`` (or its default), and
``serving_runtime_config`` writes the runtime config that the HTTP service
builds once the training component exists, because the engines need the
component's ``ReleaseFormat``. The training backend contributes only its
default mode and, for ``trainer`` serving, the config of its own serving
runtime.
"""

from __future__ import annotations

import importlib
from dataclasses import asdict
from importlib.metadata import entry_points
from typing import Any

from reef.core.errors import DeployConfigError
from reef.runtime.executor.arguments import MANAGED_ENGINE_OPTIONS, native_arguments, normalize_native_options
from reef.runtime.executor.config import role_executor_settings
from reef.runtime.interfaces import InferenceHandler
from reef.runtime.jobs import ReleaseFormat
from reef.service.deploy.service_config import ServiceConfig
from reef.train.deployment import NO_INFERENCE_BACKEND, TrainingDeployment

#: Reef's engines on an inference component serve the releases.
ENGINES = "engines"
#: The training backend's own serving runtime serves them, such as Tinker's sampler.
TRAINER = "trainer"
#: Nothing serves; the deployment trains on uploaded or imported data.
NONE = "none"

#: Native options that Reef sets itself for a serving engine, in SGLang's names: the managed engine's,
#: and the adapter table that the release format and inference.max-loaded-adapters set.
INFERENCE_RESERVED_OPTIONS = MANAGED_ENGINE_OPTIONS | {
    "enable-lora",
    "max-lora-rank",
    "max-loaded-loras",
    "max-loras-per-batch",
    "lora-target-modules",
}

#: The settings that configure Reef's engines, with their public names.
ENGINE_SETTINGS = (
    ("inference_backend", "inference.backend"),
    ("inference_num_gpus", "inference.num-gpus"),
    ("tensor_parallel_size", "inference.tensor-parallel-size"),
    ("inference_options", "inference.options"),
    ("inference_remote", "inference.remote"),
    ("inference_handler_factory", "inference.handler-factory"),
    ("inference_handler_config", "inference.handler-config"),
    ("inference_prefix_sharing", "inference.prefix-sharing"),
)


def inference_handler_factory_for(path: str | None) -> type[InferenceHandler] | None:
    """Load an optional request handler factory selected by deployment config."""
    if path is None:
        return None
    if not isinstance(path, str) or not path.strip():
        raise ValueError("inference.handler-factory must be a non-empty dotted path")
    module_path, separator, attribute = path.strip().rpartition(".")
    if not separator or not module_path or not attribute:
        raise ValueError("inference.handler-factory must be a dotted path")
    try:
        factory = getattr(importlib.import_module(module_path), attribute)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"cannot load inference.handler-factory {path!r}") from exc
    if not isinstance(factory, type) or not issubclass(factory, InferenceHandler):
        raise ValueError(f"inference.handler-factory {path!r} must inherit InferenceHandler")
    return factory


def configured_engine_settings(settings: ServiceConfig) -> list[str]:
    """The public names of the engine settings that ``settings`` sets."""
    values = asdict(settings)
    configured = [name for key, name in ENGINE_SETTINGS if values[key] not in (None, {}, False)]
    if settings.inference_max_loaded_adapters != 1:
        configured.append("inference.max-loaded-adapters")
    return configured


def serving_mode(settings: ServiceConfig, backend: TrainingDeployment) -> str:
    """The serving of a weight-training deployment: ``engines``, ``trainer``, ``none`` or a ``module:Factory``.

    ``inference.serving`` names it. Without it, ``inference.backend: none``
    serves nothing, any engine setting selects ``engines``, and otherwise the
    training backend's ``default_serving`` applies. Raise
    ``DeployConfigError`` for a mode that the settings contradict: engine
    settings without ``engines``, ``trainer`` for a backend without its own
    serving, or another unknown name.
    """
    selected = settings.inference_serving
    configured = configured_engine_settings(settings)
    if settings.inference_backend == NO_INFERENCE_BACKEND:
        if selected not in (None, NONE):
            raise DeployConfigError(f"inference.backend: none serves nothing; remove inference.serving: {selected}")
        return NONE
    if selected is None:
        selected = ENGINES if configured else backend.default_serving
    if selected not in (ENGINES, TRAINER, NONE) and ":" not in selected:
        raise DeployConfigError(
            f"inference.serving must be engines, trainer, none or module:Factory, not {selected!r}"
        )
    if selected != ENGINES and configured:
        raise DeployConfigError(f"inference.serving: {selected} runs no Reef engine; remove {', '.join(configured)}")
    if selected == TRAINER and backend.trainer_serving_config(asdict(settings)) is None:
        raise DeployConfigError("inference.serving: trainer needs a training backend that serves its own releases")
    if ":" not in selected and settings.inference_serving_config:
        raise DeployConfigError("inference.serving-config configures a module:Factory serving runtime only")
    return selected


def prepare_engine_serving(config: dict[str, Any], settings: ServiceConfig) -> None:
    """Check the engine settings and record their resolved values in ``config["reef"]``.

    The engines are tensor-parallel engines of ``inference.tensor-parallel-size``
    GPUs (default 1) on ``inference.num-gpus`` GPUs (default one engine).
    ``inference.options`` are native engine options without Reef's managed
    names. Raise ``DeployConfigError`` for anything else.
    """
    parallel_size = 1 if settings.tensor_parallel_size is None else settings.tensor_parallel_size
    num_gpus = parallel_size if settings.inference_num_gpus is None else settings.inference_num_gpus
    if parallel_size <= 0 or num_gpus <= 0 or num_gpus % parallel_size:
        raise DeployConfigError(
            "inference.num-gpus and inference.tensor-parallel-size must be positive; "
            "num-gpus must be divisible by tensor-parallel-size"
        )
    options = normalize_native_options(settings.inference_options)
    native_arguments(options, reserved=set(INFERENCE_RESERVED_OPTIONS))
    if any(name.startswith("sglang-") for name in options):
        raise DeployConfigError("inference.options uses native SGLang names without the sglang- prefix")
    engine = settings.inference_backend or "sglang"
    if (
        engine not in ("sglang", "vllm")
        and ":" not in engine
        and not entry_points(group="reef.inference_backends", name=engine)
    ):
        raise DeployConfigError(
            f"unknown inference backend {engine!r}: name sglang, vllm, an installed reef.inference_backends "
            "entry point, or module:factory"
        )
    if engine == "vllm" and num_gpus != parallel_size:
        # vLLM ships no router, so one engine serves.
        raise DeployConfigError(
            "inference.backend: vllm serves one engine; set inference.num-gpus equal to inference.tensor-parallel-size"
        )
    config["reef"].update(
        inference_backend=engine,
        inference_num_gpus=num_gpus,
        tensor_parallel_size=parallel_size,
        inference_options=options,
    )


def serving_runtime_config(
    settings: ServiceConfig, backend: TrainingDeployment, release_format: ReleaseFormat | None
) -> dict[str, Any]:
    """The runtime config of the deployment's serving, for releases of ``release_format``.

    ``engines`` gives a ``remote_serving`` config over the resolved engine
    settings, the inference component of ``inference.remote`` and the
    ``rollout`` role's executor. ``trainer`` gives the backend's own config,
    and ``none`` the null runtime. A ``module:Factory`` gets
    ``inference.serving-config`` and the release format as ``release_format``
    (``None`` when the releases hold no weight files). Raise
    ``DeployConfigError`` when Reef's engines would serve releases without
    weight files.
    """
    mode = serving_mode(settings, backend)
    if mode == NONE:
        return {"type": "none", "inference_timeout_s": settings.inference_timeout_s}
    if mode == TRAINER:
        own = backend.trainer_serving_config(asdict(settings))
        if own is None:
            raise DeployConfigError("inference.serving: trainer needs a training backend that serves its own releases")
        return own
    if mode != ENGINES:
        return {
            **dict(settings.inference_serving_config),
            "type": mode,
            "release_format": None if release_format is None else release_format.to_dict(),
        }
    if release_format is None:
        raise DeployConfigError(
            "the training backend's releases hold no weight files that engines load; configure it to export them "
            "(for Tinker, training.options.export-adapter: true) or set inference.serving: trainer"
        )
    if (
        settings.inference_backend is None
        or settings.inference_num_gpus is None
        or settings.tensor_parallel_size is None
    ):
        raise DeployConfigError(
            "engine serving needs the resolved inference.backend, inference.num-gpus and tensor-parallel-size"
        )
    rollout = role_executor_settings(settings.execution_config, "rollout")
    config: dict[str, Any] = {
        "type": "remote_serving",
        "remote": dict(settings.inference_remote),
        "release_format": release_format.to_dict(),
        "engine": settings.inference_backend,
        "num_gpus": settings.inference_num_gpus,
        "tensor_parallel_size": settings.tensor_parallel_size,
        "options": dict(settings.inference_options),
        "adapter_capacity": settings.inference_max_loaded_adapters,
        "prefix_sharing": settings.inference_prefix_sharing,
        "executor": rollout.backend,
        "executor_options": dict(rollout.options),
        "inference_timeout_s": settings.inference_timeout_s,
        "inference_handler_config": dict(settings.inference_handler_config),
    }
    handler_factory = inference_handler_factory_for(settings.inference_handler_factory)
    if handler_factory is not None:
        config["inference_handler_factory"] = handler_factory
    return config
