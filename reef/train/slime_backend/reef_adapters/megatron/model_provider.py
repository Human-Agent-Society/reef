"""Megatron model-provider extension used for Reef actor LoRA."""

from __future__ import annotations

import inspect

from reef.train.slime_backend.reef_adapters.megatron.lora import apply_megatron_lora


def provide_actor_model(pre_process: bool = True, post_process: bool = True, vp_stage: int | None = None):
    """Build through Slime (or a chained provider), then apply Reef LoRA."""
    from megatron.training.global_vars import get_args
    from slime.backends.megatron_utils.model_provider import _get_model_provider_func
    from slime.utils.misc import load_function

    args = get_args()
    chained_path = getattr(args, "reef_chained_model_provider_path", None)
    if chained_path:
        chained = load_function(chained_path)
        kwargs: dict[str, object] = {"pre_process": pre_process, "post_process": post_process}
        if "vp_stage" in inspect.signature(chained).parameters:
            kwargs["vp_stage"] = vp_stage
        model = chained(**kwargs)
    else:
        current_path = args.custom_model_provider_path
        args.custom_model_provider_path = None
        try:
            provider = _get_model_provider_func(args, role="actor")
            model = provider(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
        finally:
            args.custom_model_provider_path = current_path
    return apply_megatron_lora(model, args)


__all__ = ["provide_actor_model"]
