"""Serializable SGLang deployment configuration with no training-runtime types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from reef.inference.sglang.lora_schema import adapter_scoped_prefix_cache_supported
from reef.runtime.executor import Executor

#: Bound on one control RPC or engine launch; a weight update legitimately takes hours.
CONTROL_TIMEOUT_S = 14_400


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
    executor: str | type[Executor] = "reef.inference.sglang.executor:SGLangExecutor"
    executor_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.pause_mode not in {"in_place", "retract"}:
            raise ValueError(f"unknown SGLang pause mode: {self.pause_mode}")
        options = {key.replace("-", "_"): value for key, value in self.options.items()}
        if options.get("incremental_streaming_output", True) is not True:
            raise ValueError("Reef SGLang inference requires incremental_streaming_output=true")
        options["incremental_streaming_output"] = True
        if "disable_radix_cache" not in options:
            options["disable_radix_cache"] = not self._prefix_sharing_is_safe(options)
        self._require_safe_prefix_sharing(options)
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
            if group.options.get("incremental_streaming_output", True) is not True:
                raise ValueError("Reef SGLang inference requires incremental_streaming_output=true in every group")
            self._require_safe_prefix_sharing({**options, **group.options})
            if self.offload and group.worker_type in {"prefill", "decode"}:
                raise ValueError("colocated SGLang serving requires regular engines")
            if self.pause_mode == "retract" and group.worker_type in {"prefill", "decode"}:
                # The pinned SGLang scheduler cannot retract disaggregated requests.
                raise ValueError("retracting publication requires regular SGLang engines, not PD disaggregation")
            width = group.gpus_per_engine
            if width > self.gpus_per_node and width % self.gpus_per_node:
                raise ValueError("a multi-node SGLang engine must use whole nodes")

    def _prefix_sharing_is_safe(self, options: dict[str, Any]) -> bool:
        """Whether no radix-cache entry can outlive the weights that built it.

        An entry carries no runtime-load-ID identity. Only a ``retract`` pause
        releases every in-flight request's KV, which lets the pause clear the
        cache before a publication; an ``in_place`` pause keeps both. Under
        LoRA, SGLang must also key entries by adapter, because one engine holds
        several scenarios' adapters and the same prefix has different KV under each.
        """
        if self.pause_mode != "retract":
            return False
        return not options.get("enable_lora") or adapter_scoped_prefix_cache_supported()

    def _require_safe_prefix_sharing(self, options: dict[str, Any]) -> None:
        disabled = options["disable_radix_cache"]
        if disabled is True:
            return
        if disabled is not False:
            raise ValueError(f"disable_radix_cache must be a boolean, not {disabled!r}")
        if self.pause_mode != "retract":
            raise ValueError("Reef SGLang inference requires disable_radix_cache=true unless publication retracts")
        if not self._prefix_sharing_is_safe(options):
            raise ValueError(
                "Reef LoRA serving requires disable_radix_cache=true: "
                "the loaded SGLang does not key prefix-cache entries by adapter"
            )

    @property
    def resolved_models(self) -> tuple[SGLangModelConfig, ...]:
        return self.models or (
            SGLangModelConfig("default", (SGLangGroupConfig("regular", self.num_gpus, self.gpus_per_engine),)),
        )
