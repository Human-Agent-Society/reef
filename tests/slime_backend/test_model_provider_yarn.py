"""Reef's Megatron provider installs the checkpoint's YaRN scaling, as the HF model serves it, and nothing else."""

from __future__ import annotations

import json
import math
import sys
import types
from types import SimpleNamespace

import pytest

from reef.train.slime_backend.reef_adapters.megatron.model_provider import (
    YarnRopeScaling,
    install_yarn_rotary_embedding,
    read_yarn_rope_scaling,
    yarn_mscale,
)

QWEN3_YARN = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}


def write_config(tmp_path, **config) -> str:
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return str(tmp_path)


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [
        {"model_type": "qwen3", "rope_theta": 1000000.0},
        {"model_type": "qwen3", "rope_theta": 1000000.0, "rope_scaling": {"rope_type": "default"}},
        {"model_type": "qwen3", "rope_scaling": None},
    ],
)
def test_a_default_rotary_embedding_reads_as_none(tmp_path, config) -> None:
    assert read_yarn_rope_scaling(write_config(tmp_path, **config)) is None


@pytest.mark.unit
def test_reads_qwen3_yarn_with_the_hugging_face_attention_factor(tmp_path) -> None:
    scaling = read_yarn_rope_scaling(write_config(tmp_path, rope_theta=1000000.0, rope_scaling=QWEN3_YARN))

    assert scaling == YarnRopeScaling(
        factor=4.0,
        original_max_position_embeddings=32768,
        beta_fast=32.0,
        beta_slow=1.0,
        mscale=1.0,
        mscale_all_dim=0.0,
        truncate=True,
        rope_theta=1000000.0,
    )
    # transformers: attention_factor = get_mscale(factor) = 0.1 * ln(factor) + 1 by default.
    assert scaling.attention_factor == pytest.approx(0.1 * math.log(4.0) + 1.0)
    assert yarn_mscale(1.0, 1.0) == 1.0


@pytest.mark.unit
def test_reads_the_newer_rope_parameters_layout(tmp_path) -> None:
    # transformers >= 4.57 keeps the theta and the scaling together under rope_parameters, typed by "type".
    scaling = read_yarn_rope_scaling(
        write_config(
            tmp_path,
            rope_parameters={
                "type": "yarn",
                "factor": 2.0,
                "original_max_position_embeddings": 4096,
                "rope_theta": 10000.0,
                "beta_fast": 16.0,
                "beta_slow": 2.0,
                "mscale": 1.0,
                "mscale_all_dim": 1.0,
                "truncate": False,
                "attention_factor": 1.0,
            },
        )
    )

    assert (scaling.factor, scaling.original_max_position_embeddings, scaling.rope_theta) == (2.0, 4096, 10000.0)
    assert (scaling.beta_fast, scaling.beta_slow, scaling.truncate) == (16.0, 2.0, False)
    # Equal mscale terms cancel: Hugging Face would compute attention_factor 1.
    assert scaling.attention_factor == pytest.approx(1.0)


@pytest.mark.unit
@pytest.mark.parametrize(
    "rope, message",
    [
        ({"rope_type": "linear", "factor": 2.0}, "default and yarn"),
        ({"rope_type": "yarn", "factor": 4.0}, "original_max_position_embeddings"),
        ({**QWEN3_YARN, "attention_factor": 2.0}, "attention_factor"),
        ({**QWEN3_YARN, "factor": 1.0}, "factor must be"),
    ],
)
def test_refuses_scalings_megatron_cannot_reproduce(tmp_path, rope, message) -> None:
    with pytest.raises(ValueError, match=message):
        read_yarn_rope_scaling(write_config(tmp_path, rope_theta=1000000.0, rope_scaling=rope))


def stub_yarn_embedding(monkeypatch) -> list[dict]:
    """Install a fake ``megatron.core...YarnRotaryEmbedding`` that records its keyword arguments."""
    calls: list[dict] = []

    class FakeYarnRotaryEmbedding:
        def __init__(self, **kwargs) -> None:
            calls.append(kwargs)

    parts = ["megatron", "core", "models", "common", "embeddings", "yarn_rotary_pos_embedding"]
    for depth in range(1, len(parts) + 1):
        name = ".".join(parts[:depth])
        module = types.ModuleType(name)
        if depth < len(parts):
            module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules[".".join(parts)].YarnRotaryEmbedding = FakeYarnRotaryEmbedding
    return calls


def fake_model() -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            kv_channels=128, rotary_interleaved=False, use_cpu_initialization=False, apply_rope_fusion=True
        ),
        pg_collection=SimpleNamespace(cp="cp-group"),
        position_embedding_type="rope",
        rotary_pos_emb="rope-embedding",
    )


@pytest.mark.unit
def test_install_rebuilds_the_rotary_embedding_from_the_scaling(monkeypatch) -> None:
    calls = stub_yarn_embedding(monkeypatch)
    model = fake_model()
    scaling = YarnRopeScaling(4.0, 32768, 32.0, 1.0, 1.0, 0.0, True, 1000000.0)

    install_yarn_rotary_embedding(model, SimpleNamespace(rotary_base=1000000.0, rotary_percent=1.0), scaling)

    assert calls == [
        {
            "kv_channels": 128,
            "rotary_percent": 1.0,
            "rotary_interleaved": False,
            "rotary_base": 1000000.0,
            "scaling_factor": 4.0,
            "original_max_position_embeddings": 32768,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 0.0,
            "correction_range_round_to_int": True,
            "use_cpu_initialization": False,
            "cp_group": "cp-group",
        }
    ]
    assert model.position_embedding_type == "yarn"
    assert model.rotary_pos_emb != "rope-embedding"
    config = model.config
    assert config.apply_rope_fusion is False
    assert (config.yarn_rotary_scaling_factor, config.yarn_original_max_position_embeddings) == (4.0, 32768)
    assert (config.yarn_beta_fast, config.yarn_beta_slow) == (32.0, 1.0)
    assert (config.yarn_mscale, config.yarn_mscale_all_dim) == (1.0, 0.0)
    assert config.yarn_correction_range_round_to_int is True


@pytest.mark.unit
def test_install_refuses_a_rotary_base_that_differs_from_rope_theta(monkeypatch) -> None:
    calls = stub_yarn_embedding(monkeypatch)
    model = fake_model()
    scaling = YarnRopeScaling(4.0, 32768, 32.0, 1.0, 1.0, 0.0, True, 1000000.0)

    with pytest.raises(ValueError, match="rotary-base"):
        install_yarn_rotary_embedding(model, SimpleNamespace(rotary_base=10000.0, rotary_percent=1.0), scaling)

    assert calls == []
    assert model.position_embedding_type == "rope"
