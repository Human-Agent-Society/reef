"""The slime-layout Qwen3.5 bridge maps the parameter names slime's model has."""

from __future__ import annotations

import fnmatch
import importlib
import sys
import types
from dataclasses import dataclass, field

import pytest

from reef.train.slime_backend.reef_adapters.megatron.bridges import qwen3_5 as bridge_module

# Parameter names of the Megatron actor slime_plugins.models.qwen3_5 builds for
# Qwen3.5-0.8B (24 layers, full attention every 4th), as the actor reports
# them after slime strips its module prefix. Layer 3 is full attention, layer
# 0 is GatedDeltaNet; the MLP is dense in both.
SLIME_QWEN35_PARAMS = [
    "embedding.word_embeddings.weight",
    "decoder.final_layernorm.weight",
    "decoder.layers.0.self_attention.input_layernorm.weight",
    "decoder.layers.0.self_attention.linear_attn.in_proj_qkv.weight",
    "decoder.layers.0.self_attention.linear_attn.in_proj_z.weight",
    "decoder.layers.0.self_attention.linear_attn.in_proj_b.weight",
    "decoder.layers.0.self_attention.linear_attn.in_proj_a.weight",
    "decoder.layers.0.self_attention.linear_attn.conv1d.weight",
    "decoder.layers.0.self_attention.linear_attn.dt_bias",
    "decoder.layers.0.self_attention.linear_attn.A_log",
    "decoder.layers.0.self_attention.linear_attn.norm.weight",
    "decoder.layers.0.self_attention.linear_attn.out_proj.weight",
    "decoder.layers.0.mlp.linear_fc1.layer_norm_weight",
    "decoder.layers.0.mlp.linear_fc1.weight",
    "decoder.layers.0.mlp.linear_fc2.weight",
    "decoder.layers.3.self_attention.linear_qkv.layer_norm_weight",
    "decoder.layers.3.self_attention.linear_qkv.weight",
    "decoder.layers.3.self_attention.q_layernorm.weight",
    "decoder.layers.3.self_attention.k_layernorm.weight",
    "decoder.layers.3.self_attention.linear_proj.weight",
    "decoder.layers.3.mlp.linear_fc1.layer_norm_weight",
    "decoder.layers.3.mlp.linear_fc1.weight",
    "decoder.layers.3.mlp.linear_fc2.weight",
]

# The adapter bases Bridge LoRA wraps for the default Reef target modules.
LORA_BASES = [
    "decoder.layers.3.self_attention.linear_qkv.weight",
    "decoder.layers.3.self_attention.linear_proj.weight",
    "decoder.layers.0.mlp.linear_fc1.weight",
    "decoder.layers.0.mlp.linear_fc2.weight",
]


def _all_patterns() -> dict[str, object]:
    patterns: dict[str, object] = {}
    patterns.update(bridge_module.DIRECT_MAPPINGS)
    patterns.update(bridge_module.REPLICATED_MAPPINGS)
    patterns[bridge_module.QKV_MAPPING["megatron_param"]] = {
        key: value for key, value in bridge_module.QKV_MAPPING.items() if key != "megatron_param"
    }
    patterns[bridge_module.GATED_MLP_MAPPING["megatron_param"]] = {
        key: value for key, value in bridge_module.GATED_MLP_MAPPING.items() if key != "megatron_param"
    }
    return patterns


def _lookup(name: str):
    matches = [(pattern, hf) for pattern, hf in _all_patterns().items() if fnmatch.fnmatchcase(name, pattern)]
    assert len(matches) == 1, f"{name!r} matched {[m[0] for m in matches]}"
    return matches[0]


@pytest.mark.parametrize("name", SLIME_QWEN35_PARAMS)
def test_every_slime_parameter_has_exactly_one_mapping(name):
    _lookup(name)


def test_lora_bases_map_onto_the_hf_projections_sglang_serves():
    assert _lookup(LORA_BASES[0])[1] == {
        "q": "model.language_model.layers.*.self_attn.q_proj.weight",
        "k": "model.language_model.layers.*.self_attn.k_proj.weight",
        "v": "model.language_model.layers.*.self_attn.v_proj.weight",
    }
    assert _lookup(LORA_BASES[1])[1] == "model.language_model.layers.*.self_attn.o_proj.weight"
    assert _lookup(LORA_BASES[2])[1] == {
        "gate": "model.language_model.layers.*.mlp.gate_proj.weight",
        "up": "model.language_model.layers.*.mlp.up_proj.weight",
    }
    assert _lookup(LORA_BASES[3])[1] == "model.language_model.layers.*.mlp.down_proj.weight"


def test_gdn_tensors_are_replicated_and_named_as_hf_names_them():
    for megatron_param, hf_param in bridge_module.REPLICATED_MAPPINGS.items():
        assert megatron_param.startswith("decoder.layers.*.self_attention.")
        assert hf_param.startswith("model.language_model.layers.*.")
        assert megatron_param.rsplit(".", 1)[-1] == hf_param.rsplit(".", 1)[-1]


def test_no_mapping_uses_bridges_vl_prefixes():
    for pattern in _all_patterns():
        assert not pattern.startswith("language_model."), pattern
        assert not pattern.startswith("vision_model."), pattern


# --- registration against a stand-in Megatron-Bridge ----------------------------


@dataclass
class _Mapping:
    kind: str
    kwargs: dict = field(default_factory=dict)


class _FakeRegistry:
    def __init__(self, *mappings):
        self.mappings = list(mappings)


class _FakeModelBridge:
    registrations: list[dict] = []

    @classmethod
    def register_bridge(cls, *, source, target, provider=None, model_type=None):
        def decorate(bridge_cls):
            cls.registrations.append(
                {"source": source, "target": target, "provider": provider, "model_type": model_type, "cls": bridge_cls}
            )
            return bridge_cls

        return decorate


def _mapping_cls(kind):
    def make(*args, **kwargs):
        if args:
            kwargs = {"megatron_param": args[0], "hf_param": args[1], **kwargs}
        return _Mapping(kind, kwargs)

    return make


@pytest.fixture
def fake_bridge(monkeypatch):
    class GPTModel:
        pass

    modules = {
        "megatron": types.ModuleType("megatron"),
        "megatron.bridge": types.ModuleType("megatron.bridge"),
        "megatron.bridge.models": types.ModuleType("megatron.bridge.models"),
        "megatron.bridge.models.conversion": types.ModuleType("megatron.bridge.models.conversion"),
        "megatron.bridge.models.conversion.mapping_registry": types.ModuleType("mapping_registry"),
        "megatron.bridge.models.conversion.model_bridge": types.ModuleType("model_bridge"),
        "megatron.bridge.models.conversion.param_mapping": types.ModuleType("param_mapping"),
        "megatron.core": types.ModuleType("megatron.core"),
        "megatron.core.models": types.ModuleType("megatron.core.models"),
        "megatron.core.models.gpt": types.ModuleType("megatron.core.models.gpt"),
        "megatron.core.models.gpt.gpt_model": types.ModuleType("gpt_model"),
    }
    modules["megatron.bridge.models.conversion.mapping_registry"].MegatronMappingRegistry = _FakeRegistry
    modules["megatron.bridge.models.conversion.model_bridge"].MegatronModelBridge = _FakeModelBridge
    param_mapping = modules["megatron.bridge.models.conversion.param_mapping"]
    for kind in ("AutoMapping", "GatedMLPMapping", "QKVMapping", "ReplicatedMapping"):
        setattr(param_mapping, kind, _mapping_cls(kind))
    modules["megatron.core.models.gpt.gpt_model"].GPTModel = GPTModel
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    _FakeModelBridge.registrations = []
    module = importlib.reload(bridge_module)
    yield module, GPTModel
    importlib.reload(bridge_module)


def test_registration_targets_gpt_model_and_is_idempotent(fake_bridge):
    module, gpt_model = fake_bridge
    module.register_slime_qwen35_bridge()
    module.register_slime_qwen35_bridge()
    assert len(_FakeModelBridge.registrations) == 1
    registration = _FakeModelBridge.registrations[0]
    assert registration["source"] == "Qwen3_5ForConditionalGeneration"
    assert registration["target"] is gpt_model
    assert registration["model_type"] == "qwen3_5"

    bridge = registration["cls"]()
    registry = bridge.mapping_registry()
    kinds = {mapping.kind for mapping in registry.mappings}
    assert kinds == {"AutoMapping", "ReplicatedMapping", "QKVMapping", "GatedMLPMapping"}
    assert len(registry.mappings) == (len(module.DIRECT_MAPPINGS) + len(module.REPLICATED_MAPPINGS) + 2)
    with pytest.raises(NotImplementedError):
        bridge.provider_bridge(None)


# --- exported-name recovery for Bridge's two-field weight tuples ------------------


class _Mapped:
    def __init__(self, megatron_param):
        self.megatron_param = megatron_param


class _Registry:
    def __init__(self, table):
        self.table = table
        self.calls = 0

    def hf_to_megatron_lookup(self, hf_name):
        self.calls += 1
        if hf_name == "boom":
            raise KeyError(hf_name)
        return _Mapped(self.table[hf_name]) if hf_name in self.table else None


def test_megatron_name_resolver_maps_adapters_tied_heads_and_unknowns():
    pytest.importorskip("torch")  # hf_export imports the LoRA helpers, which need torch
    from reef.train.slime_backend.reef_adapters.megatron.hf_export import _MegatronNameResolver

    registry = _Registry(
        {
            "model.language_model.layers.3.self_attn.q_proj.weight": "decoder.layers.3.self_attention.linear_qkv.weight",
            "lm_head.weight": "output_layer.weight",
        }
    )
    resolver = _MegatronNameResolver(registry)
    lora_a = "model.language_model.layers.3.self_attn.q_proj.lora_A.weight"
    assert resolver.resolve(lora_a) == "decoder.layers.3.self_attention.linear_qkv.weight"
    assert (
        resolver.resolve("model.language_model.layers.3.self_attn.q_proj.base_layer.lora_B.weight")
        == "decoder.layers.3.self_attention.linear_qkv.weight"
    )
    assert resolver.resolve("lm_head.weight") == "output_layer.weight"
    assert resolver.resolve("model.visual.blocks.0.attn.qkv.weight") == "model.visual.blocks.0.attn.qkv.weight"
    assert resolver.resolve("boom") == "boom"
    calls = registry.calls
    assert resolver.resolve(lora_a) == "decoder.layers.3.self_attention.linear_qkv.weight"
    assert registry.calls == calls
