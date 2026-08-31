"""Export Megatron training weights as Hugging Face tensors."""

from __future__ import annotations

import dataclasses
from contextlib import contextmanager

from reef.train.slime_backend.reef_adapters.megatron.lora import (
    is_lora_weight_name,
    replace_adapter_task_weights_from_backup,
)


def patch_megatron_expert_cache_to_cpu() -> None:
    """Keep GPT-OSS expert merge caches off GPU during HF conversion."""
    try:
        from megatron.bridge.models.gpt_oss.gpt_oss_bridge import GPTOSSBridge
    except ImportError:
        return
    if getattr(GPTOSSBridge, "_reef_cpu_cache", False):
        return

    original = GPTOSSBridge.maybe_modify_converted_hf_weight

    def merge_with_cpu_cache(self, task, converted_weights):
        result = original(self, task, {name: value.cpu() for name, value in converted_weights.items()})
        return {name: value.cuda() for name, value in result.items()} if result else result

    GPTOSSBridge.maybe_modify_converted_hf_weight = merge_with_cpu_cache
    GPTOSSBridge._reef_cpu_cache = True


def patch_hf_config(config):
    configs = []
    seen = set()

    def add(candidate):
        if candidate is not None and id(candidate) not in seen:
            seen.add(id(candidate))
            configs.append(candidate)

    add(config)
    add(getattr(config, "config", None))
    for candidate in configs:
        add(getattr(candidate, "text_config", None))
    for candidate in configs:
        rope = getattr(candidate, "rope_parameters", None) or getattr(candidate, "rope_scaling", None)
        if isinstance(rope, dict) and "rope_theta" in rope and not hasattr(candidate, "rope_theta"):
            candidate.rope_theta = rope["rope_theta"]
    return config


@contextmanager
def _patched_megatron_model(model):
    try:
        from megatron.core.utils import unwrap_model
    except ImportError:
        from megatron.core.pipeline_parallel.utils import unwrap_model

    unwrapped = unwrap_model(model)[0]
    config = unwrapped.config
    added = not hasattr(config, "share_embeddings_and_output_weights")
    if added:
        config.share_embeddings_and_output_weights = unwrapped.share_embeddings_and_output_weights
    try:
        yield
    finally:
        if added:
            delattr(config, "share_embeddings_and_output_weights")


class MegatronToHfWeightIterator:
    """Adapt Megatron Bridge's lazy conversion stream to Slime's updater."""

    def __init__(self, args, model, model_name, quantization_config, **_kwargs):
        from megatron.bridge import AutoBridge

        from reef.train.slime_backend.reef_adapters.megatron.bridges import register_reef_model_bridges

        # Bridge dispatches on the HF architecture; Reef's bridges for the
        # layouts slime builds must be registered before that resolution.
        register_reef_model_bridges()
        self.args = args
        self.model = model
        self.quantization_config = quantization_config
        self._bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)
        patch_hf_config(getattr(self._bridge, "hf_pretrained", None))
        patch_megatron_expert_cache_to_cpu()

    def get_hf_weight_chunks(
        self,
        megatron_local_weights,
        progress_desc="Update weights",
        param_info_buckets=None,
        *,
        weight_type="base",
    ):
        if param_info_buckets is not None:
            raise ValueError("the HF export iterator owns its conversion plan; param_info_buckets is raw-mode only")
        if weight_type not in {"base", "lora"}:
            raise ValueError(f"unsupported Megatron-to-HF weight type: {weight_type!r}")

        from slime.backends.megatron_utils.megatron_to_hf import postprocess_hf_param
        from slime.backends.megatron_utils.megatron_to_hf.processors import quantize_params
        from slime.backends.megatron_utils.misc_utils import strip_param_name_prefix
        from slime.utils.misc import chunk_named_params_by_size

        renamed = {strip_param_name_prefix(name): value for name, value in megatron_local_weights.items()}
        with _patched_megatron_model(self.model):
            model_bridge = self._bridge._model_bridge
            original_materialize = model_bridge.materialize_adapter_weights

            def materialize_from_backup(tasks):
                return original_materialize(replace_adapter_task_weights_from_backup(tasks, renamed))

            model_bridge.materialize_adapter_weights = materialize_from_backup
            try:
                if weight_type == "lora":
                    named_weights = self._bridge.export_adapter_weights(self.model, cpu=False, show_progress=False)
                else:
                    tasks = _replace_conversion_task_weights(self._bridge.get_conversion_tasks(self.model), renamed)
                    named_weights = self._bridge.export_hf_weights(
                        self.model,
                        cpu=False,
                        conversion_tasks=tasks,
                        merge_adapter_weights=False,
                    )

                megatron_names = _MegatronNameResolver(model_bridge.mapping_registry())

                def converted():
                    for named_weight in named_weights:
                        # Bridge yields (hf_name, weight); slime's earlier Bridge fork
                        # appended the Megatron parameter name, which slime's
                        # post-processing (vocab-padding removal, quantization) keys on.
                        hf_name, weight = named_weight[0], named_weight[1]
                        megatron_name = named_weight[2] if len(named_weight) > 2 else megatron_names.resolve(hf_name)
                        hf_name = hf_name.replace(".base_layer.", ".")
                        if is_lora_weight_name(hf_name) != (weight_type == "lora"):
                            continue
                        value = postprocess_hf_param(
                            args=self.args,
                            megatron_param_name=megatron_name,
                            hf_param_name=hf_name,
                            param=weight,
                        )
                        if weight_type == "lora":
                            yield hf_name, value
                        else:
                            yield from quantize_params(
                                args=self.args,
                                megatron_name=megatron_name,
                                converted_named_params=[(hf_name, value)],
                                quantization_config=self.quantization_config,
                            )

                yield from chunk_named_params_by_size(converted(), chunk_size=self.args.update_weight_buffer_size)
            finally:
                delattr(model_bridge, "materialize_adapter_weights")


class _MegatronNameResolver:
    """Recover the Megatron parameter name behind an exported HF tensor name.

    Adapter tensors resolve through their base projection (``...q_proj.lora_A``
    -> ``...q_proj``); a tied ``lm_head`` resolves to ``output_layer.weight`` so
    slime strips its vocab padding. Names the registry does not know fall back
    to the HF name itself, which slime's post-processing leaves untouched.
    """

    def __init__(self, mapping_registry):
        self._registry = mapping_registry
        self._cache: dict[str, str] = {}

    def resolve(self, hf_name: str) -> str:
        if hf_name in self._cache:
            return self._cache[hf_name]
        base_name = hf_name.replace(".base_layer.", ".")
        for lora_suffix in (".lora_A.weight", ".lora_B.weight"):
            if base_name.endswith(lora_suffix):
                base_name = base_name[: -len(lora_suffix)] + ".weight"
                break
        megatron_name = hf_name
        try:
            mapping = self._registry.hf_to_megatron_lookup(base_name)
        except Exception:  # a lookup failure only costs the name, not the export
            mapping = None
        if mapping is not None and isinstance(getattr(mapping, "megatron_param", None), str):
            megatron_name = mapping.megatron_param
        self._cache[hf_name] = megatron_name
        return megatron_name


def _replace_conversion_task_weights(tasks, weights):
    def replace(task):
        if task is None or task.param_weight is None:
            return task
        key = f"vp_stages.{task.vp_stage}.{task.param_name}"
        if key not in weights:
            raise KeyError(f"HF export weight {key!r} is missing from the actor backup")
        return dataclasses.replace(task, param_weight=weights[key].cuda())

    return _MapWithLength(replace, tasks)


class _MapWithLength:
    def __init__(self, function, values):
        self.function = function
        self.values = values

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        return (self.function(value) for value in self.values)


__all__ = ["MegatronToHfWeightIterator", "patch_hf_config", "patch_megatron_expert_cache_to_cpu"]
