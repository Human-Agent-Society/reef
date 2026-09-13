"""SGLang engine extension installed inside Reef inference control actors."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from contextlib import suppress
from functools import cached_property
from typing import Any

import requests
from urllib3.exceptions import NewConnectionError

from reef.runtime.inference_memory import InferenceMemory
from reef.runtime.sglang.config import SGLangConfig
from reef.runtime.sglang.lora_schema import require_lora_distributed_request_schema, require_lora_tensor_request_schema
from reef.runtime.sglang.process import launch_engine, local_gpu_id, node_address_and_port, wait_ready

logger = logging.getLogger(__name__)


class ReefSGLangEngine:
    """Own one native SGLang process, or borrow an existing HTTP engine."""

    _get_current_node_ip_and_free_port = staticmethod(node_address_and_port)

    def __init__(
        self,
        config: SGLangConfig,
        rank: int,
        worker_type: str = "regular",
        base_gpu_id: int = 0,
        sglang_overrides: dict[str, Any] | None = None,
        num_gpus_per_engine: int | None = None,
        external_url: str | None = None,
    ) -> None:
        self.config = config
        self.rank = rank
        self.worker_type = worker_type
        self.base_gpu_id = base_gpu_id
        self.sglang_overrides = {key.replace("-", "_"): value for key, value in (sglang_overrides or {}).items()}
        options = config.options
        if options.get("enable_lora"):
            require_lora_tensor_request_schema()
            require_lora_distributed_request_schema()
            protected = (
                "enable_lora",
                "max_lora_rank",
                "max_loaded_loras",
                "max_loras_per_batch",
                "lora_target_modules",
                "enable_weights_cpu_backup",
                "tokenizer_worker_num",
            )
            conflicts = [
                key
                for key in protected
                if key in self.sglang_overrides and self.sglang_overrides[key] != options.get(key)
            ]
            if conflicts:
                raise ValueError(f"SGLang overrides conflict with Reef LoRA serving requirements: {conflicts}")
        self.num_gpus_per_engine = num_gpus_per_engine or config.gpus_per_engine
        self.external_url = external_url
        self.process = None
        self.node_rank = rank % max(1, self.num_gpus_per_engine // config.gpus_per_node)
        self.router_ip: str | None = None
        self.router_port: int | None = None
        self.server_host = ""
        self.server_port = 0

    def init(
        self,
        dist_init_addr: str,
        port: int,
        nccl_port: int | None,
        host: str,
        disaggregation_bootstrap_port: int | None = None,
        router_ip: str | None = None,
        router_port: int | None = None,
    ) -> None:
        from dataclasses import fields

        from sglang.srt.server_args import ServerArgs

        options = dict(self.config.options)
        options.update(self.sglang_overrides)
        pp = int(options.get("pp_size") or 1)
        tp = int(options.get("tp_size") or self.num_gpus_per_engine // pp)
        if tp * pp != self.num_gpus_per_engine:
            raise ValueError("SGLang TP * PP must match GPUs per engine")
        options.update(
            model_path=options.get("model_path", self.config.model_path),
            host=host.strip("[]"),
            port=port,
            nccl_port=nccl_port,
            node_rank=self.node_rank,
            nnodes=max(1, self.num_gpus_per_engine // self.config.gpus_per_node),
            dist_init_addr=dist_init_addr,
            tp_size=tp,
            pp_size=pp,
            base_gpu_id=local_gpu_id(self.base_gpu_id),
            gpu_id_step=1,
        )
        if self.worker_type in {"prefill", "decode"}:
            options["disaggregation_mode"] = self.worker_type
            if self.worker_type == "prefill":
                if disaggregation_bootstrap_port is None:
                    raise ValueError("prefill requires a bootstrap port")
                options.update(
                    disaggregation_bootstrap_port=disaggregation_bootstrap_port,
                    load_balance_method="follow_bootstrap_room",
                )
            else:
                options["prefill_round_robin_balance"] = True
                options.pop("enable_hierarchical_cache", None)
        if self.worker_type == "encoder":
            options["encoder_only"] = True
        names = {item.name for item in fields(ServerArgs)}
        if "cuda_graph_backend_prefill" in names and options.get("enable_memory_saver"):
            options.setdefault("cuda_graph_backend_prefill", "disabled")
            if options["cuda_graph_backend_prefill"] is None:
                options["cuda_graph_backend_prefill"] = "disabled"
        options = {key: value for key, value in options.items() if key in names}
        self.server_host, self.server_port = host, port
        self.router_ip, self.router_port = router_ip, router_port
        if self.external_url:
            actual = requests.get(f"{self.external_url}/get_server_info", timeout=30)
            actual.raise_for_status()
            info = actual.json()
            info = info.get("server_args", info)
            for key in ("enable_memory_saver", "disable_radix_cache", "incremental_streaming_output"):
                if key in options and info.get(key) != options[key]:
                    raise ValueError(f"external SGLang engine has incompatible {key}")
        else:
            self.process = launch_engine(options)
            if self.node_rank == 0:
                wait_ready(self.get_url(), self.process, self.config.startup_timeout)
        self._register_to_router(options)

    def _register_to_router(self, server_args_dict: dict[str, Any]) -> None:
        if self.node_rank != 0 or self.worker_type == "encoder":
            return
        self._sync_scheduler_runtime_load_id(self.get_runtime_load_id())
        if self.router_ip and self.router_port:
            payload = {"url": self.get_url(), "worker_type": self.worker_type}
            if self.worker_type == "prefill":
                payload["bootstrap_port"] = server_args_dict["disaggregation_bootstrap_port"]
            result = requests.post(f"http://{self.router_ip}:{self.router_port}/workers", json=payload, timeout=30)
            result.raise_for_status()

    def get_url(self) -> str:
        return self.external_url or f"http://{self.server_host}:{self.server_port}"

    def health_generate(self, timeout: float = 5) -> bool:
        response = requests.get(f"{self.get_url()}/health_generate", timeout=timeout)
        response.raise_for_status()
        return True

    def shutdown(self) -> None:
        if self.external_url:
            return
        if self.router_ip and self.node_rank == 0 and self.worker_type != "encoder":
            with suppress(requests.RequestException):
                url = f"http://{self.router_ip}:{self.router_port}/workers"
                response = requests.get(url, timeout=5)
                response.raise_for_status()
                for worker in response.json()["workers"]:
                    if worker["url"] == self.get_url():
                        requests.delete(f"{url}/{worker['id']}", timeout=5).raise_for_status()
        if self.process is not None:
            from sglang.srt.utils import kill_process_tree

            if self.process.is_alive():
                kill_process_tree(self.process.pid)
            self.process.join(timeout=10)
            if self.process.is_alive():
                raise RuntimeError("SGLang process did not retire")
            self.process = None

    def check_weights(self, action: str):
        return self._make_request("weights_checker", {"action": action})

    def pull_weights(self, target_version: int, *, source_dir: str, local_checkpoint_dir: str):
        return self._make_request(
            "pull_weights",
            {"target_version": target_version, "source_dir": source_dir, "local_checkpoint_dir": local_checkpoint_dir},
        )

    def update_weights_from_tensor(
        self,
        serialized_named_tensors: list[str],
        load_format: str | None = None,
        flush_cache: bool = False,
        weight_version: str | None = None,
    ):
        return self._make_request(
            "update_weights_from_tensor",
            {
                "serialized_named_tensors": serialized_named_tensors,
                "load_format": load_format,
                "flush_cache": flush_cache,
                "weight_version": weight_version,
            },
        )

    def init_weights_update_group(self, master_address, master_port, rank_offset, world_size, group_name, backend):
        return self._make_request(
            "init_weights_update_group",
            {
                "master_address": master_address,
                "master_port": master_port,
                "rank_offset": rank_offset,
                "world_size": world_size,
                "group_name": group_name,
                "backend": backend,
            },
        )

    def destroy_weights_update_group(self, group_name):
        with suppress(requests.RequestException):
            return self._make_request("destroy_weights_update_group", {"group_name": group_name})
        return None

    def update_weights_from_distributed(
        self,
        names,
        dtypes,
        shapes,
        group_name,
        flush_cache=False,
        weight_version: str | None = None,
        load_format: str | None = None,
    ):
        return self._make_request(
            "update_weights_from_distributed",
            {
                "names": names,
                "dtypes": [str(dtype).replace("torch.", "") for dtype in dtypes],
                "shapes": shapes,
                "group_name": group_name,
                "flush_cache": flush_cache,
                "weight_version": weight_version,
                "load_format": load_format,
            },
        )

    def post_process_weights(self, restore_weights_before_load: bool = False, post_process_quantization: bool = False):
        return self._make_request(
            "post_process_weights",
            {
                "restore_weights_before_load": restore_weights_before_load,
                "post_process_quantization": post_process_quantization,
            },
        )

    def _make_request(
        self,
        endpoint: str,
        payload: dict | None = None,
        *,
        timeout: float | None = None,
    ):
        if self.node_rank != 0:
            return None
        response = requests.post(
            f"http://{self.server_host}:{self.server_port}/{endpoint}",
            json=payload or {},
            timeout=self._weight_update_timeout_s() if timeout is None else timeout,
        )
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            raise requests.exceptions.HTTPError(
                f"{exc}; response body: {response.text}",
                request=exc.request,
                response=exc.response,
            ) from exc
        return response.json()

    def get_runtime_load_id(self):
        """Read the supported model-info endpoint instead of Slime's deprecated route.

        SGLang spells the runtime-load-ID ``weight_version`` on every wire
        surface (``ServerArgs``, ``/model_info``, ``/update_weight_version``,
        the ``UpdateWeights*ReqInput`` structs). The exchange with the server
        must keep SGLang's field name; only Reef's side of the boundary calls
        the identity a runtime-load-ID.
        """
        if self.node_rank != 0:
            return None
        response = requests.get(
            f"http://{self.server_host}:{self.server_port}/model_info",
            timeout=min(30.0, self._weight_update_timeout_s()),
        )
        response.raise_for_status()
        version = response.json().get("weight_version")
        if not isinstance(version, str) or not version:
            raise RuntimeError("SGLang model_info reports no weight_version (runtime-load-ID)")
        return version

    def _sync_scheduler_runtime_load_id(self, runtime_load_id: str | None) -> None:
        if self.node_rank != 0:
            return
        if not isinstance(runtime_load_id, str) or not runtime_load_id:
            raise RuntimeError("SGLang model_info must report a non-empty weight_version (runtime-load-ID)")
        updated = self._make_request(
            "set_internal_state",
            {"server_args": {"weight_version": runtime_load_id}},
        )
        if not isinstance(updated, list) or not updated or any(value is not True for value in updated):
            raise RuntimeError(
                "SGLang scheduler rejected runtime-load-ID synchronization; install and enable Reef's SGLang plugin"
            )

    def flush_cache(self):
        if self.node_rank != 0:
            return
        for _ in range(60):
            try:
                response = requests.get(
                    f"http://{self.server_host}:{self.server_port}/flush_cache",
                    timeout=min(5.0, self._weight_update_timeout_s()),
                )
                if response.status_code == 200:
                    return
                logger.info("Error flushing cache: HTTP %s %r", response.status_code, response.text)
            except NewConnectionError:
                raise
            except Exception as exc:
                logger.info("Error flushing cache: %s", exc)
            time.sleep(1)
        raise TimeoutError("Timeout while flushing cache.")

    def set_runtime_load_id(self, runtime_load_id: str):
        version = str(runtime_load_id)
        result = self._make_request(
            "update_weight_version",
            {"new_version": version, "abort_all_requests": False},
        )
        self._sync_scheduler_runtime_load_id(version)
        return result

    def load_lora_adapter_from_tensors(
        self,
        lora_name: str,
        config_dict: dict,
        serialized_named_tensors: list,
        load_format: str | None = None,
        pinned: bool = False,
        expected_checksums: dict | None = None,
    ):
        payload = {
            "lora_name": lora_name,
            "config_dict": config_dict,
            "serialized_named_tensors": serialized_named_tensors,
            "pinned": pinned,
        }
        if load_format is not None:
            payload["load_format"] = load_format
        if expected_checksums is not None:
            payload["expected_checksums"] = expected_checksums
        return self._make_request("load_lora_adapter_from_tensors", payload)

    def load_lora_adapter_from_distributed(
        self,
        lora_name: str,
        config_dict: dict,
        names: list[str],
        dtypes: list[Any],
        shapes: list[Any],
        group_name: str,
        pinned: bool = False,
        upsert: bool = True,
    ):
        return self._make_request(
            "load_lora_adapter_from_distributed",
            {
                "lora_name": lora_name,
                "config_dict": config_dict,
                "names": names,
                "dtypes": [str(dtype).replace("torch.", "") for dtype in dtypes],
                "shapes": shapes,
                "group_name": group_name,
                "pinned": pinned,
                "upsert": upsert,
            },
        )

    @cached_property
    def _memory(self) -> InferenceMemory:
        return InferenceMemory(_SGLangMemoryOperations(self), ("weights", "kv_cache", "cuda_graph"))

    def release_memory_occupation(self, tags: list[str] | None = None):
        self._memory.release(tags or None)

    def resume_memory_occupation(self, tags: list[str] | None = None):
        self._memory.resume(tags or None)

    def unload_lora_adapter(self, lora_name: str):
        return self._make_request("unload_lora_adapter", {"lora_name": lora_name})

    def update_weights_from_disk(
        self,
        model_path: str,
        load_format: str | None = None,
        runtime_load_id: str | None = None,
        files: list[str] | None = None,
        flush_cache: bool = False,
    ):
        payload: dict[str, Any] = {"model_path": model_path, "flush_cache": flush_cache}
        if load_format is not None:
            payload["load_format"] = load_format
        if runtime_load_id is not None:
            payload["weight_version"] = runtime_load_id
        if files is not None:
            payload["files"] = files
        return self._make_request("update_weights_from_disk", payload)

    def pause_generation(self, mode: str = "retract"):
        return self._make_request("pause_generation", {"mode": mode})

    def continue_generation(self):
        return self._make_request("continue_generation")

    def _weight_update_timeout_s(self) -> float:
        return self.config.request_timeout


class _SGLangMemoryOperations:
    """Keep native tag names and HTTP acknowledgement at the SGLang boundary."""

    def __init__(self, engine: ReefSGLangEngine) -> None:
        self.engine = engine

    def _change(self, endpoint: str, regions: Sequence[str]) -> None:
        result = self.engine._make_request(endpoint, {"tags": list(regions)})
        if isinstance(result, dict) and result.get("success") is False:
            raise RuntimeError(f"SGLang refused {endpoint}: {result!r}")

    def release(self, regions: Sequence[str]) -> None:
        self.engine.flush_cache()
        self._change("release_memory_occupation", regions)

    def resume(self, regions: Sequence[str]) -> None:
        self._change("resume_memory_occupation", regions)
