"""Crash-safe distributed weight publication layered onto raw Slime."""

from __future__ import annotations

from typing import Any

import ray
import torch
import torch.distributed as dist
from slime.backends.megatron_utils.update_weight.update_weight_from_distributed import (
    UpdateWeightFromDistributed,
    post_process_weights,
)
from tqdm import tqdm

from reef.train.slime_backend.reef_adapters.megatron.hf_export import MegatronToHfWeightIterator
from reef.train.slime_backend.reef_adapters.megatron.lora import (
    build_sglang_lora_config,
    is_lora_weight_name,
    megatron_lora_enabled,
    scenario_adapter_name,
    validate_sglang_base_checksum_response,
)
from reef.train.slime_backend.reef_adapters.weight_updaters.base import SynchronizedWeightUpdateMixin
from reef.train.slime_backend.reef_adapters.weight_updaters.lora_transport import (
    send_lora_to_distributed_engines,
    tensor_checksums,
    validate_lora_update_results,
    verify_replica_adapter_checksums,
)


class ReefUpdateWeightFromDistributed(SynchronizedWeightUpdateMixin, UpdateWeightFromDistributed):
    """Distributed updater with exact runtime load IDs and synchronized failures."""

    def __init__(
        self,
        *args: Any,
        runtime_load_id_incarnation: str | None = None,
        is_lora: bool | None = None,
        tie_word_embeddings: bool = False,
        **kwargs: Any,
    ) -> None:
        slime_args = args[0] if args else kwargs["args"]
        weights_getter = args[2] if len(args) > 2 else kwargs["weights_getter"]
        super().__init__(*args, **kwargs)
        self._initialize_runtime_load_id(runtime_load_id_incarnation)
        self.is_lora = megatron_lora_enabled(slime_args) if is_lora is None else is_lora
        self.tie_word_embeddings = tie_word_embeddings
        self._base_checksums: dict[str, str] | None = None
        if self.is_lora:
            self._lora_weights_getter = weights_getter
            self._lora_config = build_sglang_lora_config(slime_args)
            self._lora_hf_weight_iterator = MegatronToHfWeightIterator(
                slime_args,
                self.model,
                self.model_name,
                self.quantization_config,
            )
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
        """Load the slot's adapter under ``lora_name`` without a version bump."""
        if not self.is_lora:
            raise RuntimeError("adapter publication requires a LoRA-enabled actor")
        self._reset_source_phases()
        adapter_tensors = self.export_lora_adapter_tensors()
        if getattr(self.args, "check_lora_weight_equal", False):
            verify_replica_adapter_checksums(tensor_checksums(adapter_tensors))
        publication_error: BaseException | None = None
        try:
            if dist.get_rank() == 0:
                refs = send_lora_to_distributed_engines(
                    adapter_tensors,
                    rollout_engines=self.rollout_engines,
                    model_update_group=self._model_update_groups,
                    group_name=self._group_name,
                    lora_config=self._lora_config,
                    lora_name=lora_name,
                )
                validate_lora_update_results(ray.get(refs))
        except BaseException as exc:
            publication_error = exc
        self._raise_synchronized_update_error(publication_error, phase="distributed adapter publication")

    @torch.no_grad()
    def update_weights(self, *, manage_generation: bool = True, force_full: bool = False) -> None:
        self.weight_update_sequence += 1
        self._reset_source_phases()
        if self.is_lora:
            self._update_lora_weights(manage_generation=manage_generation)
            return
        if manage_generation:
            pause_mode = getattr(self.args, "weight_update_pause_mode", "retract")
            self._run_rank_zero_action(
                lambda: ray.get([engine.pause_generation.remote(pause_mode) for engine in self.rollout_engines]),
                phase="pause generation",
            )
            self._run_rank_zero_action(
                lambda: ray.get([engine.flush_cache.remote() for engine in self.rollout_engines]),
                phase="flush rollout cache",
            )
        if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
            self._run_rank_zero_action(
                lambda: post_process_weights(
                    restore_weights_before_load=True,
                    post_process_quantization=False,
                    rollout_engines=self.rollout_engines,
                ),
                phase="preprocess rollout weights",
            )
        pbar = tqdm(desc=f"[{self._group_name}] Update weights", total=0) if self._is_pp_src_rank else None
        self._send_weights(pbar)
        if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
            self._run_rank_zero_action(
                lambda: post_process_weights(
                    restore_weights_before_load=False,
                    post_process_quantization=True,
                    rollout_engines=self.rollout_engines,
                ),
                phase="postprocess rollout weights",
            )
        if manage_generation:
            self._run_rank_zero_action(
                lambda: ray.get([engine.continue_generation.remote() for engine in self.rollout_engines]),
                phase="continue generation",
            )

    def _update_lora_weights(self, *, manage_generation: bool) -> None:
        if manage_generation:
            pause_mode = getattr(self.args, "weight_update_pause_mode", "in_place")
            self._run_rank_zero_action(
                lambda: ray.get([engine.pause_generation.remote(pause_mode) for engine in self.rollout_engines]),
                phase="pause generation",
            )
            self._run_rank_zero_action(
                lambda: ray.get([engine.flush_cache.remote() for engine in self.rollout_engines]),
                phase="flush rollout cache",
            )
        if getattr(self.args, "verify_lora_base_weights", False):
            self._run_rank_zero_action(
                lambda: self._capture_or_verify_base_checksums("before adapter publication"),
                phase="verify frozen base before adapter publication",
            )

        adapter_tensors: list[tuple[str, torch.Tensor]] = []
        preparation_error: BaseException | None = None
        try:
            adapter_tensors = self.export_lora_adapter_tensors()
        except BaseException as exc:
            preparation_error = exc
        self._raise_synchronized_update_error(preparation_error, phase="prepare distributed LoRA publication")

        checksum_error: BaseException | None = None
        try:
            if getattr(self.args, "check_lora_weight_equal", False):
                verify_replica_adapter_checksums(tensor_checksums(adapter_tensors))
        except BaseException as exc:
            checksum_error = exc
        self._raise_synchronized_update_error(checksum_error, phase="verify distributed LoRA replicas")

        publication_error: BaseException | None = None
        lora_name = self.lora_name_for_publication()
        try:
            if dist.get_rank() == 0:
                refs = send_lora_to_distributed_engines(
                    adapter_tensors,
                    rollout_engines=self.rollout_engines,
                    model_update_group=self._model_update_groups,
                    group_name=self._group_name,
                    lora_config=self._lora_config,
                    lora_name=lora_name,
                )
                validate_lora_update_results(ray.get(refs))
        except BaseException as exc:
            publication_error = exc
        self._raise_synchronized_update_error(publication_error, phase="distributed LoRA publication")

        if getattr(self.args, "verify_lora_base_weights", False):
            self._run_rank_zero_action(
                lambda: self._capture_or_verify_base_checksums("after adapter publication"),
                phase="verify frozen base after adapter publication",
            )
        self._run_rank_zero_action(
            lambda: ray.get(
                [engine.set_runtime_load_id.remote(str(self.runtime_load_id)) for engine in self.rollout_engines]
            ),
            phase="commit LoRA runtime load ID",
        )
        if manage_generation:
            self._run_rank_zero_action(
                lambda: ray.get([engine.continue_generation.remote() for engine in self.rollout_engines]),
                phase="continue generation",
            )

    def _capture_or_verify_base_checksums(self, phase: str) -> None:
        responses = ray.get([engine.check_weights.remote("checksum") for engine in self.rollout_engines])
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

    def export_lora_adapter_tensors(
        self,
        megatron_local_weights: Any | None = None,
    ) -> list[tuple[str, torch.Tensor]]:
        """Export the full PEFT adapter for online publication or checkpoints."""
        if not self.is_lora:
            raise RuntimeError("LoRA adapter export requires a LoRA-enabled actor")
        if megatron_local_weights is None:
            megatron_local_weights = self._lora_weights_getter()
        adapter_tensors: list[tuple[str, torch.Tensor]] = []
        for chunk in self._lora_hf_weight_iterator.get_hf_weight_chunks(
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


__all__ = ["ReefUpdateWeightFromDistributed"]
