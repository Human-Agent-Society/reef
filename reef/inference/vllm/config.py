"""Serializable vLLM deployment configuration with no training-runtime types."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from reef.runtime.executor import Executor

#: Bound on one control RPC or engine launch; a weight update legitimately takes hours.
CONTROL_TIMEOUT_S = 14_400

#: Engine arguments Reef derives from its own settings; ``options`` may not repeat them.
RESERVED_OPTIONS = frozenset({"model", "host", "port", "tensor_parallel_size", "enable_sleep_mode"})

#: How the engine selects Reef's connector; see :mod:`reef.inference.vllm.connector`.
REEF_CONNECTOR_CONFIG: dict[str, Any] = {
    "kv_connector": "ReefConnector",
    "kv_connector_module_path": "reef.inference.vllm.connector",
    "kv_role": "kv_both",
}


def kv_transfer_config(configured: Any) -> dict[str, Any]:
    """Reef's connector alone, or beside the configured connector under vLLM's ``MultiConnector``."""
    if configured is None:
        return dict(REEF_CONNECTOR_CONFIG)
    if isinstance(configured, str):
        configured = json.loads(configured)
    if not isinstance(configured, dict) or not isinstance(configured.get("kv_connector"), str):
        raise ValueError("kv_transfer_config must be a JSON object naming kv_connector")
    if configured["kv_connector"] == "ReefConnector":
        return dict(configured)
    extra = dict(configured.get("kv_connector_extra_config", {}))
    # An existing MultiConnector gains Reef's connector; any other connector becomes its sibling.
    children = list(extra.get("connectors", [])) if configured["kv_connector"] == "MultiConnector" else [configured]
    if not any(child.get("kv_connector") == "ReefConnector" for child in children):
        children.append(dict(REEF_CONNECTOR_CONFIG))
    return {
        "kv_connector": "MultiConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {**extra, "connectors": children},
    }


@dataclass(frozen=True)
class VLLMConfig:
    """One model served by identical single-node vLLM engines on the reserved inference GPUs."""

    model_path: str
    num_gpus: int
    gpus_per_engine: int
    gpus_per_node: int
    options: dict[str, Any] = field(default_factory=dict)
    #: The endpoint that balances across engines; required when more than one engine serves.
    router_url: str | None = None
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
        if self.engine_count > 1 and not self.router_url:
            raise ValueError(
                "serving more than one vLLM engine requires router_url, the endpoint balancing across them"
            )
        if self.check_weights:
            raise ValueError("vLLM has no weights checker; set check_weights to false")
        options = {key.replace("-", "_"): value for key, value in self.options.items()}
        if reserved := RESERVED_OPTIONS.intersection(options):
            raise ValueError(f"Reef derives these vLLM options from its own settings: {sorted(reserved)}")
        if "kv_offloading_size" in options or "kv_offloading_backend" in options:
            raise ValueError(
                "kv_offloading_size replaces the configured KV connector; "
                "list OffloadingConnector in kv_transfer_config instead"
            )
        options.setdefault("logprobs_mode", "processed_logprobs")
        # A prefix-cache entry carries no weight-version identity, so sharing is
        # safe only when every publication retracts in-flight KV and resets the cache.
        sharing = options.setdefault("enable_prefix_caching", self.pause_mode == "retract")
        if not isinstance(sharing, bool):
            raise ValueError(f"enable_prefix_caching must be a boolean, not {sharing!r}")
        if sharing and self.pause_mode != "retract":
            raise ValueError("Reef vLLM inference requires enable_prefix_caching=false unless publication retracts")
        options["kv_transfer_config"] = kv_transfer_config(options.get("kv_transfer_config"))
        object.__setattr__(self, "options", options)

    @property
    def engine_count(self) -> int:
        return self.num_gpus // self.gpus_per_engine
