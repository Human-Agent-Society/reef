"""Serializable vLLM deployment configuration with no training-runtime types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from reef.execution.executor import Executor
from reef.inference.engine_control import EngineSettings

#: Engine arguments Reef derives from its own settings; ``options`` may not repeat them.
RESERVED_OPTIONS = frozenset({"model", "host", "port", "tensor_parallel_size", "enable_sleep_mode"})


@dataclass(frozen=True)
class VLLMSettings(EngineSettings):
    """One model served by identical single-node vLLM engines on the reserved inference GPUs."""

    model_path: str
    num_gpus: int
    gpus_per_engine: int
    gpus_per_node: int
    options: dict[str, Any] = field(default_factory=dict)
    #: The endpoint that balances across engines; required when more than one engine serves.
    router_url: str | None = None
    #: Where each engine's port probe starts on its host.
    engine_port_base: int = 15000
    env_vars: dict[str, str] = field(default_factory=dict)
    pause_mode: str = "in_place"
    request_timeout: float = 600
    startup_timeout: float = 14400
    executor: str | type[Executor] = "reef.inference.vllm.control:VLLMExecutor"
    executor_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.pause_mode not in {"in_place", "retract"}:
            raise ValueError(f"unknown vLLM pause mode: {self.pause_mode}")
        if min(self.num_gpus, self.gpus_per_engine, self.gpus_per_node) <= 0:
            raise ValueError("vLLM GPU capacities must be positive")
        if self.num_gpus % self.gpus_per_engine:
            raise ValueError("vLLM inference requires a whole number of engines")
        if self.gpus_per_engine > self.gpus_per_node:
            raise ValueError("a vLLM engine must fit on one node")
        if self.request_timeout <= 0 or self.startup_timeout <= 0:
            raise ValueError("vLLM timeouts must be positive")
        if not 1 <= self.engine_port_base <= 65500:
            raise ValueError("engine_port_base must be a port between 1 and 65500")
        if self.engine_count > 1 and not self.router_url:
            raise ValueError(
                "serving more than one vLLM engine requires router_url, the endpoint balancing across them"
            )
        options = {key.replace("-", "_"): value for key, value in self.options.items()}
        if reserved := RESERVED_OPTIONS.intersection(options):
            raise ValueError(f"Reef derives these vLLM options from its own settings: {sorted(reserved)}")
        options.setdefault("logprobs_mode", "processed_logprobs")
        # The generate route takes the model's generation_config.json as request
        # defaults, which can add top-k, top-p and a temperature the trainer never
        # sees; vLLM's own defaults sample the full distribution at temperature 1.
        options.setdefault("generation_config", "vllm")
        # A prefix-cache entry carries no weight-version identity, so sharing is
        # safe only when every publication retracts in-flight KV and resets the cache.
        sharing = options.setdefault("enable_prefix_caching", self.pause_mode == "retract")
        if not isinstance(sharing, bool):
            raise ValueError(f"enable_prefix_caching must be a boolean, not {sharing!r}")
        if sharing and self.pause_mode != "retract":
            raise ValueError("Reef vLLM inference requires enable_prefix_caching=false unless publication retracts")
        object.__setattr__(self, "options", options)

    @property
    def engine_count(self) -> int:
        return self.num_gpus // self.gpus_per_engine
