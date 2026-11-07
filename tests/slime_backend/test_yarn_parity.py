"""The YaRN embedding Reef installs in Megatron equals the one transformers computes for the same config."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("megatron.core")
transformers = pytest.importorskip("transformers")

from reef.train.slime_backend.reef_adapters.megatron.model_provider import (
    install_yarn_rotary_embedding,
    read_yarn_rope_scaling,
)

HEAD_DIM = 128
QWEN3_128K = {
    "model_type": "qwen3",
    "hidden_size": 1024,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": HEAD_DIM,
    "max_position_embeddings": 131072,
    "rope_theta": 1000000.0,
    "rope_scaling": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768},
}


@pytest.mark.integration
def test_installed_yarn_matches_transformers_at_every_position(tmp_path) -> None:
    # Megatron's YarnRotaryEmbedding moves its frequencies to the current CUDA device.
    if not torch.cuda.is_available():
        pytest.skip("Megatron's YaRN embedding computes on CUDA")
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    (tmp_path / "config.json").write_text(json.dumps(QWEN3_128K), encoding="utf-8")
    device = torch.device("cuda")
    config = transformers.AutoConfig.from_pretrained(str(tmp_path), local_files_only=True)
    inv_freq, attention_factor = ROPE_INIT_FUNCTIONS["yarn"](config, device)
    positions = QWEN3_128K["max_position_embeddings"]
    angles = torch.outer(torch.arange(positions, device=device, dtype=torch.float32), inv_freq.float())

    scaling = read_yarn_rope_scaling(str(tmp_path))
    model = SimpleNamespace(
        config=SimpleNamespace(
            kv_channels=HEAD_DIM, rotary_interleaved=False, use_cpu_initialization=False, apply_rope_fusion=True
        ),
        pg_collection=SimpleNamespace(cp=None),
        position_embedding_type="rope",
        rotary_pos_emb=None,
    )
    install_yarn_rotary_embedding(model, SimpleNamespace(rotary_base=1000000.0, rotary_percent=1.0), scaling)
    emb, mscale = model.rotary_pos_emb.get_emb(positions)
    installed_angles = emb[:, 0, 0, : HEAD_DIM // 2].float()

    assert mscale == pytest.approx(attention_factor) == pytest.approx(0.1 * math.log(4.0) + 1.0)
    assert torch.equal(installed_angles, angles)
    assert torch.equal(torch.cos(installed_angles) * mscale, torch.cos(angles) * attention_factor)
