"""Regression coverage for the bridge-export backup naming translation.

Slime's actor backup uses global parameter names, while Bridge conversion tasks
(and Reef's lookups) key on ``vp_stages.{stage}.{local_name}``.
``rekey_backup_to_vp_stages`` pairs Reef's local enumeration of the model
chunks with pinned Slime's two-argument global enumeration. These tests run it
on real torch modules and stub the two Slime modules that the function imports.
"""

from __future__ import annotations

import re
import sys
import types
from collections.abc import Iterator

import pytest

torch = pytest.importorskip("torch")

#: The offsets that pinned Slime adds: chunk 1 starts at global layer 2, and
#: this rank's first expert has global index 4.
LAYER_OFFSETS = (0, 2)
EXPERT_OFFSET = 4

#: Each local key that the function must return, and the backup name it reads.
EXPECTED_BACKUP_NAMES = {
    "vp_stages.0.decoder.layers.0.self_attention.linear_qkv.to_wrap.weight": (
        "decoder.layers.0.self_attention.linear_qkv.to_wrap.weight"
    ),
    "vp_stages.0.decoder.layers.0.self_attention.linear_qkv.adapter.linear_in.weight": (
        "decoder.layers.0.self_attention.linear_qkv.adapter.linear_in.weight"
    ),
    "vp_stages.0.decoder.layers.0.self_attention.linear_qkv.adapter.linear_out.weight": (
        "decoder.layers.0.self_attention.linear_qkv.adapter.linear_out.weight"
    ),
    "vp_stages.0.decoder.layers.1.mlp.router.weight": "decoder.layers.1.mlp.router.weight",
    "vp_stages.0.decoder.layers.1.mlp.experts.linear_fc1.weight0": "decoder.layers.1.mlp.experts.linear_fc1.weight4",
    "vp_stages.0.decoder.layers.1.mlp.experts.linear_fc1.weight1": "decoder.layers.1.mlp.experts.linear_fc1.weight5",
    "vp_stages.0.decoder.layers.1.mlp.router.expert_bias": "decoder.layers.1.mlp.router.expert_bias",
    "vp_stages.1.decoder.layers.0.mlp.router.weight": "decoder.layers.2.mlp.router.weight",
    "vp_stages.1.decoder.layers.0.mlp.experts.linear_fc1.weight0": "decoder.layers.2.mlp.experts.linear_fc1.weight4",
    "vp_stages.1.decoder.final_layernorm.weight": "decoder.final_layernorm.weight",
    "vp_stages.1.decoder.layers.0.mlp.router.expert_bias": "decoder.layers.2.mlp.router.expert_bias",
}


def module_with(**members: torch.nn.Module | torch.Tensor) -> torch.nn.Module:
    """A module that holds ``members`` in order: submodules, parameters, and other tensors as buffers."""
    module = torch.nn.Module()
    for name, member in members.items():
        if isinstance(member, torch.nn.Module):
            module.add_module(name, member)
        elif isinstance(member, torch.nn.Parameter):
            module.register_parameter(name, member)
        else:
            module.register_buffer(name, member)
    return module


def weight() -> torch.nn.Parameter:
    # Equal values everywhere, so that only object identity tells two weights apart.
    return torch.nn.Parameter(torch.zeros(2, 2))


def router() -> torch.nn.Module:
    # local_tokens_per_expert is an ordinary buffer, which neither enumeration includes.
    return module_with(weight=weight(), expert_bias=torch.zeros(2), local_tokens_per_expert=torch.zeros(2))


def model_chunks() -> list[torch.nn.Module]:
    """Two virtual-pipeline chunks, each wrapped twice, as DDP and Float16Module wrap a Megatron model.

    Chunk 0 holds an attention layer with LoRA adapter weights and an MoE layer. Chunk 1 holds an MoE layer and
    the final layer norm.
    """
    qkv = module_with(
        to_wrap=module_with(weight=weight()),
        adapter=module_with(linear_in=module_with(weight=weight()), linear_out=module_with(weight=weight())),
    )
    attention_layer = module_with(self_attention=module_with(linear_qkv=qkv))
    first_experts = module_with(linear_fc1=module_with(weight0=weight(), weight1=weight()))
    first_moe_layer = module_with(mlp=module_with(router=router(), experts=first_experts))
    last_experts = module_with(linear_fc1=module_with(weight0=weight()))
    last_moe_layer = module_with(mlp=module_with(router=router(), experts=last_experts))
    final_layernorm = module_with(weight=weight())
    first_decoder = module_with(layers=torch.nn.ModuleList([attention_layer, first_moe_layer]))
    last_decoder = module_with(layers=torch.nn.ModuleList([last_moe_layer]), final_layernorm=final_layernorm)
    chunks = [module_with(decoder=first_decoder), module_with(decoder=last_decoder)]
    return [module_with(module=module_with(module=chunk)) for chunk in chunks]


def strip_param_name_prefix(name: str) -> str:
    while name.startswith("module."):
        name = name.removeprefix("module.")
    return name


def named_params_and_buffers(
    args: types.SimpleNamespace, model: list[torch.nn.Module]
) -> Iterator[tuple[str, torch.Tensor]]:
    """Pinned Slime's enumeration: its order, its global names, and the model's own tensor objects.

    Like Slime 8a53b680, it takes exactly ``(args, model)``; a ``convert_to_global_name`` argument is a TypeError.
    """
    for vp_stage, chunk in enumerate(model):
        chunk_weights = list(chunk.named_parameters())
        chunk_weights += [(name, buffer) for name, buffer in chunk.named_buffers() if "expert_bias" in name]
        for name, tensor in chunk_weights:
            layer = re.fullmatch(r"module\.module\.decoder\.layers\.(\d+)\.(.+)", name)
            if layer is None:
                yield name, tensor
                continue
            rest = layer[2]
            expert = re.fullmatch(r"(mlp\.experts\.\w+\.weight)(\d+)", rest)
            if expert is not None:
                rest = f"{expert[1]}{int(expert[2]) + args.expert_offset}"
            yield f"module.module.decoder.layers.{int(layer[1]) + args.layer_offsets[vp_stage]}.{rest}", tensor


def slime_args() -> types.SimpleNamespace:
    return types.SimpleNamespace(layer_offsets=LAYER_OFFSETS, expert_offset=EXPERT_OFFSET)


def actor_backup(model: list[torch.nn.Module]) -> dict[str, torch.Tensor]:
    """The actor backup as ``get_hf_weight_chunks`` passes it: copies under module-stripped global names."""
    return {
        strip_param_name_prefix(name): tensor.detach().clone()
        for name, tensor in named_params_and_buffers(slime_args(), model)
    }


@pytest.fixture()
def slime_modules(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Install the two Slime modules that the function imports; return the one with the enumeration."""
    misc = types.ModuleType("slime.backends.megatron_utils.misc_utils")
    misc.strip_param_name_prefix = strip_param_name_prefix
    common = types.ModuleType("slime.backends.megatron_utils.update_weight.common")
    common.named_params_and_buffers = named_params_and_buffers
    for name, module in {
        "slime": types.ModuleType("slime"),
        "slime.backends": types.ModuleType("slime.backends"),
        "slime.backends.megatron_utils": types.ModuleType("slime.backends.megatron_utils"),
        "slime.backends.megatron_utils.misc_utils": misc,
        "slime.backends.megatron_utils.update_weight": types.ModuleType("slime.backends.megatron_utils.update_weight"),
        "slime.backends.megatron_utils.update_weight.common": common,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return common


@pytest.mark.unit
def test_global_named_backup_is_rekeyed_to_its_own_backup_tensors(slime_modules: types.ModuleType) -> None:
    from reef.train.slime_backend.reef_adapters.megatron.hf_export import rekey_backup_to_vp_stages

    model = model_chunks()
    backup = actor_backup(model)
    backup_before = dict(backup)
    live = {strip_param_name_prefix(name): tensor for name, tensor in named_params_and_buffers(slime_args(), model)}

    rekeyed = rekey_backup_to_vp_stages(slime_args(), model, backup)

    assert set(rekeyed) == set(EXPECTED_BACKUP_NAMES)
    for local_name, backup_name in EXPECTED_BACKUP_NAMES.items():
        assert rekeyed[local_name] is backup[backup_name]
        assert rekeyed[local_name] is not live[backup_name]
    # The input is unchanged: the same names hold the same tensor objects.
    assert backup.keys() == backup_before.keys()
    assert all(backup[name] is tensor for name, tensor in backup_before.items())


@pytest.mark.unit
def test_vp_stages_keys_pass_through_and_win(slime_modules: types.ModuleType) -> None:
    from reef.train.slime_backend.reef_adapters.megatron.hf_export import rekey_backup_to_vp_stages

    already_local, kept = torch.ones(2, 2), torch.ones(1)
    backup = {
        "decoder.final_layernorm.weight": torch.zeros(2, 2),
        "vp_stages.1.decoder.final_layernorm.weight": already_local,
        "vp_stages.0.some.other.entry": kept,
    }

    rekeyed = rekey_backup_to_vp_stages(slime_args(), model_chunks(), backup)

    assert set(rekeyed) == {"vp_stages.1.decoder.final_layernorm.weight", "vp_stages.0.some.other.entry"}
    assert rekeyed["vp_stages.1.decoder.final_layernorm.weight"] is already_local
    assert rekeyed["vp_stages.0.some.other.entry"] is kept


@pytest.mark.unit
def test_missing_globals_are_skipped_not_fabricated(slime_modules: types.ModuleType) -> None:
    from reef.train.slime_backend.reef_adapters.megatron.hf_export import rekey_backup_to_vp_stages

    model = model_chunks()
    global_name = "decoder.layers.2.mlp.router.expert_bias"
    backup = {global_name: actor_backup(model)[global_name]}

    rekeyed = rekey_backup_to_vp_stages(slime_args(), model, backup)

    assert set(rekeyed) == {"vp_stages.1.decoder.layers.0.mlp.router.expert_bias"}
    assert rekeyed["vp_stages.1.decoder.layers.0.mlp.router.expert_bias"] is backup[global_name]


@pytest.mark.unit
@pytest.mark.parametrize("change", ["drops_the_last_weight", "swaps_the_first_two_weights"])
def test_a_slime_enumeration_that_no_longer_matches_the_model_fails(
    slime_modules: types.ModuleType, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    from reef.train.slime_backend.reef_adapters.megatron.hf_export import rekey_backup_to_vp_stages

    model = model_chunks()
    backup = actor_backup(model)
    slime_order = list(named_params_and_buffers(slime_args(), model))
    if change == "drops_the_last_weight":
        changed_order = slime_order[:-1]
        expected = "hold 11 parameters and expert_bias buffers, but Slime's global enumeration names 10 weights"
    else:
        # The same count and equal values: only the tensor objects show that the order changed.
        assert torch.equal(slime_order[0][1], slime_order[1][1])
        changed_order = [slime_order[1], slime_order[0], *slime_order[2:]]
        expected = (
            "local weight 'vp_stages.0.decoder.layers.0.self_attention.linear_qkv.to_wrap.weight' and Slime's global "
            "weight 'module.module.decoder.layers.0.self_attention.linear_qkv.adapter.linear_in.weight' are different"
        )

    def changed_enumeration(
        args: types.SimpleNamespace, chunks: list[torch.nn.Module]
    ) -> Iterator[tuple[str, torch.Tensor]]:
        return iter(changed_order)

    monkeypatch.setattr(slime_modules, "named_params_and_buffers", changed_enumeration)

    with pytest.raises(RuntimeError) as error:
        rekey_backup_to_vp_stages(slime_args(), model, backup)
    assert expected in str(error.value)
