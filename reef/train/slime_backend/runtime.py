"""Connect a Slime trainer to Reef without choosing its inference implementation."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from reef.core.config import config_option
from reef.runtime.deployment import RayRuntimeConfig, RuntimeConfigError, RuntimeFactory, RuntimeRegistry, runtime_pair
from reef.runtime.executor.connection import DEFAULT_ACTOR_NAME, DEFAULT_NAMESPACE, connect_ray_coordinator
from reef.runtime.interfaces import InferenceRuntime, TrainingRuntime
from reef.train.runtime import ExecutorTrainingRuntime

#: Coordinator connection keys forwarded verbatim to :class:`SlimeTrainingRuntime`.
TRAINING_CONFIG_KEYS = (
    "actor_name",
    "namespace",
    "ray_address",
    "inference_timeout_s",
    "train_timeout_s",
    "max_staleness",
)
#: Keys forwarded verbatim to the independently selected inference factory.
INFERENCE_CONFIG_KEYS = (
    "inference_url",
    "inference_timeout_s",
    "inference_handler_factory",
    "inference_handler_config",
)


class SlimeTrainingRuntime(ExecutorTrainingRuntime):
    """Attach to Slime's separately launched Reef coordinator in its Ray cluster."""

    def __init__(
        self,
        *,
        actor_name: str = DEFAULT_ACTOR_NAME,
        namespace: str = DEFAULT_NAMESPACE,
        ray_address: str | None = None,
        inference_timeout_s: float = 300.0,
        train_timeout_s: float | None = None,
        max_staleness: int = 0,
    ) -> None:
        super().__init__(
            connect_ray_coordinator(
                actor_name=actor_name,
                namespace=namespace,
                ray_address=ray_address,
                inference_timeout_s=inference_timeout_s,
                train_timeout_s=train_timeout_s,
            ),
            max_staleness=max_staleness,
        )


@dataclass(frozen=True)
class SlimeRuntimeConfig(RayRuntimeConfig):
    inference_runtime: str = config_option(
        "", help="Independently selected inference runtime kind or factory reference."
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.inference_runtime:
            raise ValueError("runtime.inference_runtime must name the selected inference implementation")


class SlimeRuntimeFactory(RuntimeFactory):
    """Build a Slime trainer and bind the independently selected inference runtime."""

    kind = "slime_training"

    def config_type(self) -> type:
        return SlimeRuntimeConfig

    def parse_config(self, config: Mapping[str, Any], environ: Mapping[str, str]) -> dict[str, Any]:
        injected = {key: config[key] for key in ("connect", "inference_handler_factory") if key in config}
        parsed = super().parse_config({key: value for key, value in config.items() if key not in injected}, environ)
        return {**{key: value for key, value in parsed.items() if key in config}, **injected}

    def __call__(
        self,
        config: Mapping[str, Any],
        model_path: str,
        recipe_config: Mapping[str, Any],
        environ: Mapping[str, str],
    ) -> tuple[TrainingRuntime, InferenceRuntime]:
        registry = RuntimeRegistry()
        if "connect" in config:
            # Preserve explicit Python connectors used by externally managed deployments.
            injected = {key: value for key, value in config.items() if key != "inference_runtime"}
            built = registry.build({**injected, "type": "ray_training"}, model_path=model_path, environ=environ)
            pair = runtime_pair(built)
            if pair is None:
                raise RuntimeConfigError("the configured training connector must return a runtime pair")
            return pair
        training = SlimeTrainingRuntime(**{key: config[key] for key in TRAINING_CONFIG_KEYS if key in config})
        try:
            inference = registry.build(
                {
                    **{key: config[key] for key in INFERENCE_CONFIG_KEYS if key in config},
                    "type": config["inference_runtime"],
                    "control": training.train_group_handle,
                },
                model_path=model_path,
                recipe_config=recipe_config,
                environ=environ,
            )
            if not isinstance(inference, InferenceRuntime):
                # A misconfigured selection may have built a whole pair; release it.
                for component in inference if isinstance(inference, tuple) else ():
                    with suppress(Exception):
                        component.shutdown()
                raise RuntimeConfigError("the selected inference factory must return an InferenceRuntime")
            return training, inference
        except BaseException:
            with suppress(Exception):
                training.shutdown()
            raise
