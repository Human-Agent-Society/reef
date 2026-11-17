"""Serializable SGLang deployment configuration with no training-runtime types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from reef.execution.executor import Executor
from reef.inference.engine_control import EngineSettings
from reef.inference.sglang.lora_schema import adapter_scoped_prefix_cache_supported


@dataclass(frozen=True)
class SGLangSettings(EngineSettings):
    """One model served by identical single-node SGLang engines on the reserved inference GPUs.

    Values: the three GPU counts are positive, ``num_gpus`` is a multiple of
    ``gpus_per_engine``, and ``gpus_per_engine`` is at most ``gpus_per_node``.
    ``options`` holds SGLang ``ServerArgs`` fields by their underscore names;
    construction sets ``incremental_streaming_output`` to true and gives
    ``disable_radix_cache`` a value. ``router_port`` is the port the router
    binds, or ``None`` for a probed one. ``pause_mode`` is ``in_place`` or
    ``retract``. Both timeouts are positive seconds.
    Purpose: the serving assembly writes the value (``serving_engine_config``);
    the engine group, the engines and the worker read it.
    """

    model_path: str
    num_gpus: int
    gpus_per_engine: int
    gpus_per_node: int
    options: dict[str, Any] = field(default_factory=dict)
    router_port: int | None = None
    router_options: dict[str, Any] = field(default_factory=dict)
    env_vars: dict[str, str] = field(default_factory=dict)
    pause_mode: str = "in_place"
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
        if self.num_gpus % self.gpus_per_engine:
            raise ValueError("SGLang inference requires a whole number of engines")
        if self.gpus_per_engine > self.gpus_per_node:
            raise ValueError("an SGLang engine must fit on one node")
        if self.request_timeout <= 0 or self.startup_timeout <= 0:
            raise ValueError("SGLang timeouts must be positive")

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
    def engine_count(self) -> int:
        return self.num_gpus // self.gpus_per_engine
