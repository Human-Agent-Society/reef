"""vLLM engine actor installed inside Reef inference control actors.

Every control operation is one HTTP call to the engine's own server. The
routes under ``VLLM_SERVER_DEV_MODE`` (pause, sleep, weight version, reload,
collective RPC) are vLLM's reinforcement-learning surface; Reef treats them as
a private interface pinned to the supported vLLM release.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any

import requests

from reef.inference.process import node_address_and_port, wait_ready
from reef.inference.vllm.config import VLLMConfig
from reef.inference.vllm.process import EngineProcess, launch_server
from reef.runtime.interfaces import InferenceMemoryOperations
from reef.runtime.scheduler import InferenceMemory

logger = logging.getLogger(__name__)

#: Regions the coordinator may name. vLLM frees CUDA graphs with the KV cache, so that tag adds nothing.
MEMORY_REGIONS = ("weights", "kv_cache", "cuda_graph")


class ReefVLLMEngine:
    """Own one native vLLM server on this node's reserved GPUs."""

    def __init__(self, config: VLLMConfig, rank: int, gpu_ids: Sequence[int]) -> None:
        if len(gpu_ids) != config.gpus_per_engine:
            raise ValueError(f"vLLM engine {rank} needs {config.gpus_per_engine} GPUs, got {list(gpu_ids)}")
        self.config = config
        self.rank = rank
        self.gpu_ids = tuple(int(gpu) for gpu in gpu_ids)
        self.process: EngineProcess | None = None
        self.server_host = ""
        self.server_port = 0
        self._memory = InferenceMemory(_VLLMMemoryOperations(self), MEMORY_REGIONS)

    def node_address_and_port(self, start_port: int = 15000) -> tuple[str, int]:
        """This actor's node address and a free serving port, chosen where the server will bind."""
        return node_address_and_port(start_port=start_port)

    def init(self, host: str, port: int) -> None:
        self.server_host, self.server_port = host, port
        self.process = launch_server(self.server_arguments(host, port), self.server_environment())
        wait_ready(self.get_url(), self.process, self.config.startup_timeout, path="/health")
        logger.info("vLLM engine %d serves %s on GPUs %s", self.rank, self.get_url(), list(self.gpu_ids))

    def server_arguments(self, host: str, port: int) -> list[str]:
        """The server command line: placement, what Reef serving needs, then the configured options."""
        arguments = [
            "--model",
            self.config.model_path,
            "--host",
            host.strip("[]"),
            "--port",
            str(port),
            "--tensor-parallel-size",
            str(self.config.gpus_per_engine),
        ]
        if self.config.gpus_per_engine > 1:
            # The engine runs inside a Ray actor's process tree; its workers must not join that Ray cluster.
            arguments.extend(("--distributed-executor-backend", "mp"))
        if self.config.offload:
            arguments.append("--enable-sleep-mode")
        for key, value in self.config.options.items():
            flag = key.replace("_", "-")
            if value is True:
                arguments.append(f"--{flag}")
            elif value is False:
                arguments.append(f"--no-{flag}")
            elif isinstance(value, (dict, list)):
                arguments.extend((f"--{flag}", json.dumps(value)))
            else:
                arguments.extend((f"--{flag}", str(value)))
        return arguments

    def server_environment(self) -> dict[str, str]:
        environment = {"CUDA_VISIBLE_DEVICES": ",".join(str(gpu) for gpu in self.gpu_ids), **self.config.env_vars}
        if self.config.options.get("enable_lora"):
            environment["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "1"
        return environment

    def get_url(self) -> str:
        return f"http://{self.server_host}:{self.server_port}"

    def health_generate(self, timeout: float = 5) -> bool:
        response = requests.get(f"{self.get_url()}/health", timeout=timeout)
        response.raise_for_status()
        return True

    def shutdown(self) -> None:
        if self.process is None:
            return
        self.process.shutdown()
        if self.process.is_alive():
            raise RuntimeError("vLLM process did not retire")
        self.process = None

    # -- Weight version -----------------------------------------------------------

    def get_runtime_load_id(self) -> str:
        """vLLM spells the runtime load ID ``weight_version`` on its wire; only Reef's side renames it."""
        version = self._get("weight_info").get("weight_version")
        if not isinstance(version, str) or not version:
            raise RuntimeError("vLLM weight_info reports no weight_version (runtime load ID)")
        return version

    def set_runtime_load_id(self, runtime_load_id: str) -> dict[str, Any]:
        return self._post("update_weight_version", body={"new_version": str(runtime_load_id)})

    # -- Generation barrier -------------------------------------------------------

    def pause_generation(self, mode: str = "in_place") -> dict[str, Any]:
        """Stop scheduling with requests kept in place.

        ``retract`` additionally frees every in-flight request's KV and resets
        the prefix cache, so no entry built by the outgoing weights survives
        the publication; the requests re-prefill under the new weights.
        """
        if mode not in {"in_place", "retract"}:
            raise ValueError(f"unknown vLLM pause mode: {mode}")
        result = self._post("pause", params={"mode": "keep", "clear_cache": "false"})
        if mode == "retract":
            self._require_success(self._post("reset_prefix_cache", params={"reset_running_requests": "true"}))
        return result

    def continue_generation(self) -> dict[str, Any]:
        return self._post("resume")

    def flush_cache(self) -> None:
        """Reset the prefix cache; fails while requests still hold KV."""
        self._require_success(self._post("reset_prefix_cache"))

    # -- Weights and adapters -----------------------------------------------------

    def update_weights_from_disk(self, model_path: str, runtime_load_id: str | None = None) -> dict[str, Any]:
        result = self._post(
            "collective_rpc", body={"method": "reload_weights", "kwargs": {"weights_path": model_path}}
        )
        if runtime_load_id is not None:
            self.set_runtime_load_id(runtime_load_id)
        return result

    def load_lora_adapter_from_disk(self, lora_name: str, lora_path: str) -> dict[str, Any]:
        """Load a PEFT adapter directory the engine's host can read, under ``lora_name``."""
        return self._adapter_change("v1/load_lora_adapter", {"lora_name": lora_name, "lora_path": lora_path})

    def unload_lora_adapter(self, lora_name: str) -> dict[str, Any]:
        return self._adapter_change("v1/unload_lora_adapter", {"lora_name": lora_name})

    def _adapter_change(self, endpoint: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        """vLLM answers adapter routes with a message and a status; report both in the ``success`` shape."""
        response = requests.post(
            f"{self.get_url()}/{endpoint}", json=dict(payload), timeout=self.config.request_timeout
        )
        return {"success": response.ok, "message": response.text}

    # -- Memory -------------------------------------------------------------------

    def release_memory_occupation(self, tags: Sequence[str] | None = None) -> None:
        if tags and "weights" not in tags:
            # ``release_kv_cache_memory`` needs an engine with no requests at all, and
            # sleep level 1 moves the weights to host memory; neither keeps a resident base.
            raise ValueError("vLLM releases the KV cache only together with the weights")
        self._memory.release(tags or None)

    def resume_memory_occupation(self, tags: Sequence[str] | None = None) -> None:
        self._memory.resume(tags or None)

    # -- HTTP ---------------------------------------------------------------------

    def _get(self, endpoint: str) -> Any:
        response = requests.get(f"{self.get_url()}/{endpoint}", timeout=min(30.0, self.config.request_timeout))
        return self._body(response)

    def _post(
        self, endpoint: str, *, params: Mapping[str, Any] | None = None, body: Mapping[str, Any] | None = None
    ) -> Any:
        response = requests.post(
            f"{self.get_url()}/{endpoint}",
            params=None if params is None else dict(params),
            json=None if body is None else dict(body),
            timeout=self.config.request_timeout,
        )
        return self._body(response)

    @staticmethod
    def _body(response: requests.Response) -> Any:
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise requests.HTTPError(
                f"{exc}; response body: {response.text}", request=exc.request, response=exc.response
            ) from exc
        if not response.content:
            return {}
        try:
            return response.json()
        except ValueError:
            return {"message": response.text}

    @staticmethod
    def _require_success(result: Any) -> None:
        if isinstance(result, dict) and result.get("success") is False:
            raise RuntimeError(f"vLLM refused the cache reset: {result!r}")


class _VLLMMemoryOperations(InferenceMemoryOperations):
    """Map Reef's regions onto vLLM's sleep levels and wake-up tags."""

    def __init__(self, engine: ReefVLLMEngine) -> None:
        self.engine = engine

    def release(self, regions: Sequence[str]) -> None:
        # Level 2 discards weights and KV; requests stay queued and re-prefill after wake-up.
        self.engine._post("sleep", params={"level": "2", "mode": "keep"})

    def resume(self, regions: Sequence[str]) -> None:
        tags = [region for region in ("weights", "kv_cache") if region in regions]
        if tags:
            self.engine._post("wake_up", params={"tags": tags})
