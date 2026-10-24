"""Prepare a local Qwen3 YaRN view and use the same RoPE in both training roles."""

import argparse
import json
from pathlib import Path

CONTEXT_LENGTH = 131072
ORIGINAL_CONTEXT = 32768
SCALING_FACTOR = 4.0


def prepare_model(source: Path, destination: Path) -> None:
    """Link immutable model files and write a separate 128k configuration."""
    source = source.resolve(strict=True)
    config = json.loads((source / "config.json").read_text())
    if config.get("model_type") != "qwen3":
        raise ValueError("The Retail long-context recipe requires a dense Qwen3 checkpoint")
    if destination.exists():
        raise ValueError("Use a fresh destination for the long-context model view")
    rope = config.get("rope_parameters") or config.get("rope_scaling") or {}
    if rope.get("rope_type", rope.get("type", "default")) != "default":
        raise ValueError("Start from an unscaled Qwen3 checkpoint")
    theta = rope.get("rope_theta", config.get("rope_theta", 1000000.0))
    scaling = {
        "rope_type": "yarn",
        "factor": SCALING_FACTOR,
        "original_max_position_embeddings": ORIGINAL_CONTEXT,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
    }
    config.pop("rope_parameters", None)
    config["rope_theta"] = theta
    config["rope_scaling"] = scaling
    config["max_position_embeddings"] = CONTEXT_LENGTH
    destination.mkdir(parents=True)
    for path in source.iterdir():
        if path.name != "config.json":
            (destination / path.name).symlink_to(path)
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def provide_model(pre_process: bool = True, post_process: bool = True, vp_stage: int | None = None):
    """Build Slime's standard Qwen3, then install its HF configuration's YaRN.

    Slime wraps this provider with the value head for the critic. Both roles
    use the same frequencies and attention magnitude. No weights are changed.
    """
    from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import YarnRotaryEmbedding
    from megatron.training.global_vars import get_args
    from slime.backends.megatron_utils.model_provider import _get_model_provider_func
    from transformers import AutoConfig

    args = get_args()
    hf = AutoConfig.from_pretrained(args.hf_checkpoint, local_files_only=True)
    rope = getattr(hf, "rope_parameters", None) or getattr(hf, "rope_scaling", None) or {}
    if (
        hf.model_type != "qwen3"
        or rope.get("rope_type") != "yarn"
        or rope.get("factor") != SCALING_FACTOR
        or rope.get("original_max_position_embeddings") != ORIGINAL_CONTEXT
        or getattr(hf, "max_position_embeddings", None) != CONTEXT_LENGTH
    ):
        raise ValueError("Prepare the 128k Qwen3 model with long_context.py before starting Reef")
    current_provider = args.custom_model_provider_path
    args.custom_model_provider_path = None
    try:
        model = _get_model_provider_func(args, role="actor")(
            pre_process=pre_process, post_process=post_process, vp_stage=vp_stage
        )
    finally:
        args.custom_model_provider_path = current_provider
    config = model.config
    config.yarn_rotary_scaling_factor = SCALING_FACTOR
    config.yarn_original_max_position_embeddings = ORIGINAL_CONTEXT
    config.yarn_beta_fast = 32.0
    config.yarn_beta_slow = 1.0
    config.yarn_mscale = 1.0
    config.yarn_mscale_all_dim = 0.0
    config.yarn_correction_range_round_to_int = True
    # Fused RoPE does not support YaRN's non-unit cosine/sine magnitude.
    config.apply_rope_fusion = False
    model.position_embedding_type = "yarn"
    model.rotary_pos_emb = YarnRotaryEmbedding(
        kv_channels=config.kv_channels,
        rotary_percent=args.rotary_percent,
        rotary_interleaved=config.rotary_interleaved,
        rotary_base=args.rotary_base,
        scaling_factor=SCALING_FACTOR,
        original_max_position_embeddings=ORIGINAL_CONTEXT,
        beta_fast=32.0,
        beta_slow=1.0,
        mscale=1.0,
        mscale_all_dim=0.0,
        correction_range_round_to_int=True,
        use_cpu_initialization=config.use_cpu_initialization,
        cp_group=model.pg_collection.cp,
    )
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    prepare_model(arguments.source, arguments.destination)
