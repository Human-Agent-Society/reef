"""SGLang engine extension installed inside Reef inference control actors."""

from __future__ import annotations

import logging
import time
from contextlib import suppress
from dataclasses import fields, is_dataclass
from typing import Any

import requests
from urllib3.exceptions import NewConnectionError

from reef.inference.interfaces import InferenceEngine
from reef.inference.process import check_server_owner, ports_in_use
from reef.inference.sglang.config import SGLangSettings
from reef.inference.sglang.process import launch_engine, local_gpu_id, node_address_and_port, wait_ready

logger = logging.getLogger(__name__)


class ReefSGLangEngine(InferenceEngine):
    """Own one native SGLang process on one node."""

    _get_current_node_ip_and_free_port = staticmethod(node_address_and_port)
    ports_in_use = staticmethod(ports_in_use)

    def __init__(self, config: SGLangSettings, rank: int, base_gpu_id: int = 0) -> None:
        self.config = config
        self.rank = rank
        self.base_gpu_id = base_gpu_id
        self.process = None
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
        router_ip: str | None = None,
        router_port: int | None = None,
    ) -> None:
        options = self._server_options(dist_init_addr, port, nccl_port, host)
        self.server_host, self.server_port = host, port
        self.router_ip, self.router_port = router_ip, router_port
        process = launch_engine(options)
        self.process = process
        wait_ready(self.get_url(), process, self.config.startup_timeout)
        check_server_owner(host, port, process.pid)
        self._register_to_router()

    def _server_options(self, dist_init_addr: str, port: int, nccl_port: int | None, host: str) -> dict[str, Any]:
        """The ``ServerArgs`` fields for this engine: the configured options, then placement."""
        from sglang.srt.server_args import ServerArgs

        options = dict(self.config.options)
        pp = int(options.get("pp_size") or 1)
        tp = int(options.get("tp_size") or self.config.gpus_per_engine // pp)
        if tp * pp != self.config.gpus_per_engine:
            raise ValueError("SGLang TP * PP must match GPUs per engine")
        options.update(
            model_path=options.get("model_path", self.config.model_path),
            host=host.strip("[]"),
            port=port,
            nccl_port=nccl_port,
            dist_init_addr=dist_init_addr,
            tp_size=tp,
            pp_size=pp,
            base_gpu_id=local_gpu_id(self.base_gpu_id),
            gpu_id_step=1,
        )
        if is_dataclass(ServerArgs):
            names = {item.name for item in fields(ServerArgs)}
        else:
            # SGLang 0.5.21 represents ServerArgs as a msgspec Struct, which names its fields itself.
            names = set(ServerArgs.__struct_fields__)
        if "cuda_graph_backend_prefill" in names and options.get("enable_memory_saver"):
            options.setdefault("cuda_graph_backend_prefill", "disabled")
            if options["cuda_graph_backend_prefill"] is None:
                options["cuda_graph_backend_prefill"] = "disabled"
        return {key: value for key, value in options.items() if key in names}

    def _register_to_router(self) -> None:
        self._sync_scheduler_runtime_load_id(self.get_runtime_load_id())
        if self.router_ip and self.router_port:
            payload = {"url": self.get_url(), "worker_type": "regular"}
            result = requests.post(f"http://{self.router_ip}:{self.router_port}/workers", json=payload, timeout=30)
            result.raise_for_status()

    def get_url(self) -> str:
        return f"http://{self.server_host}:{self.server_port}"

    def shutdown(self) -> None:
        if self.router_ip:
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

    def _make_request(
        self,
        endpoint: str,
        payload: dict | None = None,
        *,
        timeout: float | None = None,
    ):
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

    def get_runtime_load_id(self) -> str:
        """Read the supported model-info endpoint instead of Slime's deprecated route.

        SGLang spells the runtime-load-ID ``weight_version`` on every wire
        surface (``ServerArgs``, ``/model_info``, ``/update_weight_version``,
        the ``UpdateWeights*ReqInput`` structs). The exchange with the server
        must keep SGLang's field name; only Reef's side of the boundary calls
        the identity a runtime-load-ID.
        """
        response = requests.get(
            f"http://{self.server_host}:{self.server_port}/model_info",
            timeout=min(30.0, self._weight_update_timeout_s()),
        )
        response.raise_for_status()
        version = response.json().get("weight_version")
        if not isinstance(version, str) or not version:
            raise RuntimeError("SGLang model_info reports no weight_version (runtime-load-ID)")
        return version

    def _sync_scheduler_runtime_load_id(self, runtime_load_id: str) -> object:
        """Give the engine ``runtime_load_id`` as its ``weight_version`` and return SGLang's answer.

        ``/update_weight_version`` sets the tokenizer manager's version and, in
        SGLang 0.5.21, forwards it to every scheduler, which records it in its
        serving context.
        """
        if not isinstance(runtime_load_id, str) or not runtime_load_id:
            raise RuntimeError("SGLang model_info must report a non-empty weight_version (runtime-load-ID)")
        answer = self._make_request(
            "update_weight_version",
            {"new_version": runtime_load_id, "abort_all_requests": False},
        )
        if not isinstance(answer, dict) or answer.get("success") is not True:
            raise RuntimeError(f"SGLang did not accept the runtime-load-ID {runtime_load_id!r}: {answer!r}")
        return answer

    def flush_cache(self) -> None:
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

    def set_runtime_load_id(self, runtime_load_id: str) -> object:
        return self._sync_scheduler_runtime_load_id(str(runtime_load_id))

    def load_lora_adapter_from_disk(self, lora_name: str, lora_path: str, pinned: bool = False) -> object:
        """Load a PEFT adapter directory the engine's host can read, under ``lora_name``."""
        payload: dict[str, Any] = {"lora_name": lora_name, "lora_path": lora_path}
        if pinned:
            payload["pinned"] = True
        return self._make_request("load_lora_adapter", payload)

    def unload_lora_adapter(self, lora_name: str) -> object:
        return self._make_request("unload_lora_adapter", {"lora_name": lora_name})

    def update_weights_from_disk(
        self,
        model_path: str,
        load_format: str | None = None,
        runtime_load_id: str | None = None,
        files: list[str] | None = None,
        flush_cache: bool = False,
    ) -> object:
        payload: dict[str, Any] = {"model_path": model_path, "flush_cache": flush_cache}
        if load_format is not None:
            payload["load_format"] = load_format
        if runtime_load_id is not None:
            payload["weight_version"] = runtime_load_id
        if files is not None:
            payload["files"] = files
        return self._make_request("update_weights_from_disk", payload)

    def pause_generation(self, mode: str = "retract") -> object:
        return self._make_request("pause_generation", {"mode": mode})

    def continue_generation(self) -> object:
        return self._make_request("continue_generation")

    def _weight_update_timeout_s(self) -> float:
        return self.config.request_timeout
