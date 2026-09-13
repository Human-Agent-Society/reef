"""Serializable SGLang deployment configuration with no training-runtime types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from reef.runtime.executor import Executor


@dataclass(frozen=True)
class SGLangGroupConfig:
    worker_type: str
    num_gpus: int
    gpus_per_engine: int
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.worker_type not in {"regular", "prefill", "decode", "encoder", "placeholder"}:
            raise ValueError(f"unknown SGLang worker type: {self.worker_type}")
        if (
            self.num_gpus <= 0
            or self.gpus_per_engine <= 0
            or (self.worker_type != "placeholder" and self.num_gpus % self.gpus_per_engine)
        ):
            raise ValueError("SGLang groups require a positive, whole number of engines")


@dataclass(frozen=True)
class SGLangModelConfig:
    name: str
    groups: tuple[SGLangGroupConfig, ...]
    update_weights: bool = True


@dataclass(frozen=True)
class SGLangConfig:
    model_path: str
    num_gpus: int
    gpus_per_engine: int
    gpus_per_node: int
    options: dict[str, Any] = field(default_factory=dict)
    models: tuple[SGLangModelConfig, ...] = ()
    external_engines: tuple[dict[str, Any], ...] = ()
    router_host: str | None = None
    router_port: int | None = None
    router_options: dict[str, Any] = field(default_factory=dict)
    env_vars: dict[str, str] = field(default_factory=dict)
    offload: bool = False
    shared_gpus: int = 0
    check_weights: bool = False
    pause_mode: str = "in_place"
    health_enabled: bool = False
    health_interval: float = 30
    health_timeout: float = 30
    health_first_wait: float = 60
    request_timeout: float = 600
    startup_timeout: float = 14400
    executor: str | type[Executor] = "reef.runtime.sglang.executor:SGLangExecutor"
    executor_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        options = {key.replace("-", "_"): value for key, value in self.options.items()}
        for key in ("disable_radix_cache", "incremental_streaming_output"):
            if options.get(key, True) is not True:
                raise ValueError(f"Reef SGLang inference requires {key}=true")
            options[key] = True
        object.__setattr__(self, "options", options)
        if min(self.num_gpus, self.gpus_per_engine, self.gpus_per_node) <= 0:
            raise ValueError("SGLang GPU capacities must be positive")
        if self.request_timeout <= 0 or self.startup_timeout <= 0:
            raise ValueError("SGLang timeouts must be positive")
        if self.external_engines:
            return
        groups = [group for model in self.resolved_models for group in model.groups]
        if not self.external_engines and sum(group.num_gpus for group in groups) != self.num_gpus:
            raise ValueError("SGLang model groups must match the reserved inference GPUs")
        for group in groups:
            for key in ("disable_radix_cache", "incremental_streaming_output"):
                if group.options.get(key, True) is not True:
                    raise ValueError(f"Reef SGLang inference requires {key}=true in every group")
            if self.offload and group.worker_type in {"prefill", "decode"}:
                raise ValueError("colocated SGLang serving requires regular engines")
            width = group.gpus_per_engine
            if width > self.gpus_per_node and width % self.gpus_per_node:
                raise ValueError("a multi-node SGLang engine must use whole nodes")

    @property
    def resolved_models(self) -> tuple[SGLangModelConfig, ...]:
        return self.models or (
            SGLangModelConfig("default", (SGLangGroupConfig("regular", self.num_gpus, self.gpus_per_engine),)),
        )
