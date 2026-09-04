"""SGLang engine extension installed inside Reef rollout-manager actors."""

from __future__ import annotations

import logging
import time
from typing import Any

import requests
from slime.backends.sglang_utils.sglang_engine import SGLangEngine
from urllib3.exceptions import NewConnectionError

from reef.train.slime_backend.reef_adapters.megatron.lora import megatron_lora_enabled, sglang_lora_target_modules
from reef.train.slime_backend.reef_adapters.sglang.lora_schema import (
    require_lora_distributed_request_schema,
    require_lora_tensor_request_schema,
)
from reef.train.slime_backend.reef_adapters.worker_hooks import reef_node_ip_and_free_port, reef_rollout_env_vars

logger = logging.getLogger(__name__)


class ReefSGLangEngine(SGLangEngine):
    """Bound weight-update requests and synchronize scheduler runtime load IDs."""

    _get_current_node_ip_and_free_port = staticmethod(reef_node_ip_and_free_port)

    def __init__(
        self,
        args: Any,
        rank: int,
        worker_type: str = "regular",
        base_gpu_id: int | None = None,
        sglang_overrides: dict[str, Any] | None = None,
        num_gpus_per_engine: int | None = None,
    ) -> None:
        # Slime creates each engine as a separate Ray actor. Configure LoRA on
        # the actor constructor itself so the requirements cross that process
        # boundary instead of depending on a manager-local monkey patch.
        overrides = {key.replace("-", "_"): value for key, value in (sglang_overrides or {}).items()}
        if megatron_lora_enabled(args):
            require_lora_tensor_request_schema()
            require_lora_distributed_request_schema()
            configured = getattr(args, "max_loaded_loras", None)
            max_loaded = 1 if configured is None else int(configured)
            if max_loaded < 1:
                raise ValueError("--max-loaded-loras must be at least 1")
            required = {
                "enable_lora": True,
                # One shared base model can serve several scenario-owned
                # adapters; Reef's residency manager bounds them to this cap.
                "max_loras_per_batch": max_loaded,
                "max_loaded_loras": max_loaded,
                "max_lora_rank": int(args.megatron_lora_rank),
                "lora_target_modules": sglang_lora_target_modules(args),
                "enable_weights_cpu_backup": True,
                # Distributed upsert resolves a stable adapter id in the
                # tokenizer process. Multiple tokenizer workers keep separate
                # registries, so SGLang deliberately rejects that topology.
                "tokenizer_worker_num": 1,
            }
            conflicts = {
                key: (overrides[key], value)
                for key, value in required.items()
                if key in overrides and overrides[key] != value
            }
            if conflicts:
                raise ValueError(f"SGLang overrides conflict with Reef LoRA serving requirements: {conflicts}")
            overrides.update(required)
            logger.info(
                "SGLang LoRA RL enabled: slots=%d rank=%d targets=%s",
                max_loaded,
                int(args.megatron_lora_rank),
                required["lora_target_modules"],
            )
        super().__init__(
            args,
            rank,
            worker_type=worker_type,
            base_gpu_id=base_gpu_id,
            sglang_overrides=overrides,
            num_gpus_per_engine=num_gpus_per_engine,
        )

    def _register_to_router(self, server_args_dict):
        # Fail closed before the router can admit work to a scheduler that did
        # not load Reef's runtime-load-ID plugin.
        if self.worker_type != "encoder":
            self._sync_scheduler_runtime_load_id(self.get_runtime_load_id())
        return super()._register_to_router(server_args_dict)

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
        minutes = getattr(getattr(self, "args", None), "distributed_timeout_minutes", 10)
        if not isinstance(minutes, int | float) or isinstance(minutes, bool) or minutes <= 0:
            raise ValueError("distributed_timeout_minutes must be a positive number")
        return float(minutes) * 60


def install_sglang_extensions() -> None:
    """Install extensions in the dedicated rollout-manager process only."""
    from slime.ray import rollout, utils

    # Slime constructs each SGLang Ray actor with an explicit runtime_env,
    # which otherwise drops the deployment address inherited by this manager.
    # Keep Slime's router and spawned engines on the configured deployment
    # interface (loopback for one host, a routable address for multiple hosts).
    utils.RAY_DEFAULT_ENV_VARS.update(reef_rollout_env_vars())

    rollout.SGLangEngine = ReefSGLangEngine


__all__ = [
    "ReefSGLangEngine",
    "install_sglang_extensions",
]
