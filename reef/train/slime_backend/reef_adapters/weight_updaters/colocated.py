"""Exact-version tensor weight publication for colocated and remote engines."""

from __future__ import annotations

from typing import Any

import ray
import torch
import torch.distributed as dist
from ray import ObjectRef
from slime.backends.megatron_utils.update_weight.update_weight_from_distributed import (
    post_process_weights,
    update_weights_from_distributed,
)
from slime.backends.megatron_utils.update_weight.update_weight_from_tensor import (
    UpdateWeightFromTensor,
    _send_to_colocated_engine,
)
from slime.utils.distributed_utils import get_gloo_group
from tqdm import tqdm

from reef.train.slime_backend.reef_adapters.megatron.hf_export import MegatronToHfWeightIterator
from reef.train.slime_backend.reef_adapters.megatron.lora import (
    adapter_checksum_verification_due,
    build_sglang_lora_config,
    is_lora_weight_name,
    lora_engine_identity,
    megatron_lora_enabled,
    scenario_adapter_name,
    validate_sglang_base_checksum_response,
)
from reef.train.slime_backend.reef_adapters.weight_updaters.base import SynchronizedWeightUpdateMixin
from reef.train.slime_backend.reef_adapters.weight_updaters.lora_transport import (
    send_lora_to_colocated_engine,
    send_lora_to_distributed_engines,
    tensor_checksums,
    validate_lora_update_results,
    verify_replica_adapter_checksums,
)


class ReefUpdateWeightFromTensor(SynchronizedWeightUpdateMixin, UpdateWeightFromTensor):
    """Tensor updater that can publish a complete base or one LoRA adapter."""

    def __init__(
        self,
        *args: Any,
        runtime_load_id_incarnation: str | None = None,
        is_lora: bool | None = None,
        tie_word_embeddings: bool = False,
        **kwargs: Any,
    ) -> None:
        slime_args = args[0] if args else kwargs["args"]
        if getattr(slime_args, "megatron_to_hf_mode", "raw") == "bridge":
            from slime.backends.megatron_utils.update_weight.hf_weight_iterator_base import HfWeightIteratorBase

            original_create = HfWeightIteratorBase.create
            HfWeightIteratorBase.create = staticmethod(MegatronToHfWeightIterator)
            try:
                super().__init__(*args, **kwargs)
            finally:
                HfWeightIteratorBase.create = staticmethod(original_create)
        else:
            super().__init__(*args, **kwargs)
        self._initialize_runtime_load_id(runtime_load_id_incarnation)
        self.is_lora = megatron_lora_enabled(self.args) if is_lora is None else is_lora
        self.tie_word_embeddings = tie_word_embeddings
        self._base_checksums: dict[str, str] | None = None
        if self.is_lora:
            self._lora_config = build_sglang_lora_config(self.args)
            self._lora_loaded_engine_ids: set[str] = set()
            self._lora_rollout_engines: tuple[Any, ...] = ()
            # The scenario whose adapter the Megatron slot holds. Which
            # revisions the engines keep is the bridge's residency manager's
            # business; the updater only loads under the name it is given.
            self.active_scenario: str | None = None

    def lora_name_for_publication(self) -> str:
        """The engine adapter name the next publication loads under."""
        if not self.active_scenario:
            raise RuntimeError("LoRA publication requires an active scenario")
        return scenario_adapter_name(self.active_scenario, str(self.runtime_load_id))

    def publish_lora_adapter(self, lora_name: str) -> None:
        """Load the slot's adapter under ``lora_name`` without a version bump.

        Used at restart to make every scenario's committed adapter resident
        again; the caller owns pausing and the serving runtime load ID.
        """
        if not self.is_lora:
            raise RuntimeError("adapter publication requires a LoRA-enabled actor")
        adapter_tensors = self.export_lora_adapter_tensors(self.weights_getter())
        expected_checksums = None
        # Restart republication runs once per scenario, not once per
        # training step, so it always verifies rather than sampling.
        if getattr(self.args, "check_lora_weight_equal", False):
            expected_checksums = tensor_checksums(adapter_tensors)
            verify_replica_adapter_checksums(expected_checksums)
        refs, long_lived_tensors, engine_id = self._send_lora_params(
            adapter_tensors,
            expected_checksums=expected_checksums,
            lora_name=lora_name,
        )
        validate_lora_update_results(ray.get(refs))
        if engine_id is not None:
            self._lora_loaded_engine_ids.add(engine_id)
        if long_lived_tensors is not None:
            long_lived_tensors.clear()
        adapter_tensors.clear()
        dist.barrier(group=get_gloo_group())

    def connect_rollout_engines(self, *args: Any, **kwargs: Any) -> None:
        self._ipc_engine = None
        rollout_engines = tuple(args[0] if args else kwargs["rollout_engines"])
        super().connect_rollout_engines(*args, **kwargs)
        if self.is_lora:
            self._lora_rollout_engines = rollout_engines

    @torch.no_grad()
    def update_weights(self, *, manage_generation: bool = True, force_full: bool = False) -> None:
        self.weight_update_sequence += 1
        serving_engines = self._lora_rollout_engines if self.is_lora else self.rollout_engines
        if self.rank == 0 and manage_generation:
            pause_mode = getattr(self.args, "weight_update_pause_mode", "retract")
            ray.get([engine.pause_generation.remote(pause_mode) for engine in serving_engines])
            ray.get([engine.flush_cache.remote() for engine in serving_engines])

        if self.rank == 0:
            if self.is_lora and getattr(self.args, "verify_lora_base_weights", False):
                self._capture_or_verify_base_checksums("before adapter publication")
            elif self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=True,
                    post_process_quantization=False,
                    rollout_engines=self.rollout_engines,
                )
        dist.barrier(group=get_gloo_group())

        progress = tqdm(desc="Update weights", total=0) if self.rank == 0 else None
        megatron_local_weights = self.weights_getter()
        if self.is_lora:
            adapter_tensors: list[tuple[str, torch.Tensor]] = []
            preparation_error: BaseException | None = None
            try:
                adapter_tensors = self.export_lora_adapter_tensors(megatron_local_weights)
            except BaseException as exc:
                preparation_error = exc
            self._raise_synchronized_update_error(preparation_error, phase="prepare LoRA publication")

            checksum_error: BaseException | None = None
            expected_checksums = None
            try:
                if adapter_checksum_verification_due(self.args, self.weight_update_sequence):
                    expected_checksums = tensor_checksums(adapter_tensors)
                    verify_replica_adapter_checksums(expected_checksums)
            except BaseException as exc:
                checksum_error = exc
            self._raise_synchronized_update_error(checksum_error, phase="verify LoRA replicas")

            publication_error: BaseException | None = None
            engine_id = None
            lora_name = self.lora_name_for_publication()
            try:
                refs, long_lived_tensors, engine_id = self._send_lora_params(
                    adapter_tensors,
                    expected_checksums=expected_checksums,
                    lora_name=lora_name,
                )
                validate_lora_update_results(ray.get(refs))
            except BaseException as exc:
                publication_error = exc
            self._raise_synchronized_update_error(publication_error, phase="LoRA publication")
            if engine_id is not None:
                self._lora_loaded_engine_ids.add(engine_id)
            if long_lived_tensors is not None:
                long_lived_tensors.clear()
            adapter_tensors.clear()
        else:
            param_info_buckets = (
                self._non_expert_param_info_buckets if self._expert_transfer_plan else self._full_param_info_buckets
            )
            for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(
                megatron_local_weights,
                param_info_buckets=param_info_buckets,
            ):
                refs, long_lived_tensors = self._send_hf_params(hf_named_tensors)
                ray.get(refs)
                refs.clear()
                if long_lived_tensors is not None:
                    long_lived_tensors.clear()
                hf_named_tensors.clear()
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()

        if self._expert_transfer_plan:
            self._update_expert_weights(megatron_local_weights)
        if progress is not None:
            progress.close()
        dist.barrier(group=get_gloo_group())
        if self.rank == 0:
            if self.is_lora:
                if getattr(self.args, "verify_lora_base_weights", False):
                    self._capture_or_verify_base_checksums("after adapter publication")
                ray.get([engine.set_runtime_load_id.remote(str(self.runtime_load_id)) for engine in serving_engines])
            elif self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=False,
                    post_process_quantization=True,
                    rollout_engines=self.rollout_engines,
                )
            if manage_generation:
                ray.get([engine.continue_generation.remote() for engine in serving_engines])
        dist.barrier(group=get_gloo_group())

    def export_lora_adapter_tensors(
        self,
        megatron_local_weights: Any | None = None,
    ) -> list[tuple[str, torch.Tensor]]:
        """Export only adapter tensors through the same Bridge path used online."""

        if not self.is_lora:
            raise RuntimeError("LoRA adapter export requires a LoRA-enabled actor")
        if megatron_local_weights is None:
            megatron_local_weights = self.weights_getter()
        adapter_tensors: list[tuple[str, torch.Tensor]] = []
        for chunk in self._hf_weight_iterator.get_hf_weight_chunks(
            megatron_local_weights,
            weight_type="lora",
        ):
            adapter_tensors.extend(chunk)
        if not adapter_tensors:
            raise RuntimeError("LoRA publication produced zero adapter tensors; refusing to continue")
        unexpected = [name for name, _ in adapter_tensors if not is_lora_weight_name(name)]
        if unexpected:
            raise RuntimeError(f"LoRA publication contains frozen/base tensors: {unexpected[:8]}")
        return adapter_tensors

    def _capture_or_verify_base_checksums(self, phase: str) -> None:
        responses = ray.get([engine.check_weights.remote("checksum") for engine in self._lora_rollout_engines])
        current = {
            str(index): validate_sglang_base_checksum_response(
                response,
                engine_index=index,
                tie_word_embeddings=self.tie_word_embeddings,
            )
            for index, response in enumerate(responses)
        }
        if self._base_checksums is None:
            self._base_checksums = current
            return
        if current != self._base_checksums:
            changed = {
                key: {"expected": self._base_checksums.get(key), "actual": value}
                for key, value in current.items()
                if self._base_checksums.get(key) != value
            }
            raise RuntimeError(f"SGLang frozen base checksum changed {phase}: {changed}")

    def _send_hf_params(self, hf_named_tensors: list[tuple[str, torch.Tensor]]) -> tuple[list[ObjectRef], Any]:
        all_refs = []
        refs_colocated, long_lived_tensors = _send_to_colocated_engine(
            hf_named_tensors,
            ipc_engine=self._ipc_engine,
            ipc_gather_src=self._ipc_gather_src,
            ipc_gather_group=self._ipc_gather_group,
            runtime_load_id=str(self.runtime_load_id),
        )
        all_refs.extend(refs_colocated)
        if self.use_distribute and self._is_distributed_src_rank:
            all_refs.extend(
                update_weights_from_distributed(
                    self._group_name,
                    self._model_update_groups,
                    str(self.runtime_load_id),
                    self.distributed_rollout_engines,
                    hf_named_tensors,
                )
            )
        return all_refs, long_lived_tensors

    def _send_lora_params(
        self,
        hf_named_tensors: list[tuple[str, torch.Tensor]],
        *,
        expected_checksums: dict[str, str] | None,
        lora_name: str,
    ) -> tuple[list[ObjectRef], Any, str | None]:
        engine_id = lora_engine_identity(self._ipc_engine) if self._ipc_engine is not None else None
        refs, long_lived_tensors = send_lora_to_colocated_engine(
            hf_named_tensors,
            ipc_engine=self._ipc_engine,
            ipc_gather_src=self._ipc_gather_src,
            ipc_gather_group=self._ipc_gather_group,
            lora_config=self._lora_config,
            lora_name=lora_name,
            # Every publication has a fresh versioned name; nothing is ever
            # reloaded in place, so no pre-unload of the same name is needed.
            lora_loaded=False,
            expected_checksums=expected_checksums,
        )
        if self.use_distribute and self._is_distributed_src_rank:
            refs.extend(
                send_lora_to_distributed_engines(
                    hf_named_tensors,
                    rollout_engines=self.distributed_rollout_engines,
                    model_update_group=self._model_update_groups,
                    group_name=self._group_name,
                    lora_config=self._lora_config,
                    lora_name=lora_name,
                )
            )
        return refs, long_lived_tensors, engine_id


__all__ = ["ReefUpdateWeightFromTensor"]
