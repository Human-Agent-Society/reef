"""Regression coverage for the bridge-export backup naming translation.

Slime f655e13d (#2251) switched the actor backup to global parameter names
unconditionally, while Bridge conversion tasks (and Reef's lookups) key on
``vp_stages.{vp}.{local_name}``. ``rekey_backup_to_vp_stages`` translates the
former into the latter by zipping slime's two enumerators. These tests stub
the two slime modules that the function imports.
"""

from __future__ import annotations

import sys
import types

import pytest

pytest.importorskip("torch")


@pytest.fixture()
def slime_naming_stubs(monkeypatch):
    """Install fake slime modules exposing the two naming enumerators.

    The fake model has two vp stages; stage 1 owns final_layernorm, mirroring
    a PP layout. Global names carry module prefixes and global layer indices;
    local names are per-chunk with local indices.
    """

    per_mode = {
        # convert_to_global_name=False -> vp_stages local names
        False: [
            ("vp_stages.0.decoder.layers.0.self_attention.linear_qkv.weight", None),
            ("vp_stages.0.decoder.layers.1.mlp.linear_fc1.weight", None),
            ("vp_stages.1.decoder.layers.0.mlp.linear_fc2.weight", None),
            ("vp_stages.1.decoder.final_layernorm.weight", None),
        ],
        # convert_to_global_name=True -> module-prefixed global names
        True: [
            ("module.module.decoder.layers.0.self_attention.linear_qkv.weight", None),
            ("module.module.decoder.layers.1.mlp.linear_fc1.weight", None),
            ("module.module.decoder.layers.2.mlp.linear_fc2.weight", None),
            ("module.module.decoder.final_layernorm.weight", None),
        ],
    }

    def named_params_and_buffers(args, model, convert_to_global_name=True, **_kw):
        return iter(per_mode[convert_to_global_name])

    def strip_param_name_prefix(name):
        prefix = "module."
        while name.startswith(prefix):
            name = name.removeprefix(prefix)
        return name

    misc = types.ModuleType("slime.backends.megatron_utils.misc_utils")
    misc.strip_param_name_prefix = strip_param_name_prefix
    common = types.ModuleType("slime.backends.megatron_utils.update_weight.common")
    common.named_params_and_buffers = named_params_and_buffers
    common.all_gather_param = lambda name, param: param

    for name, module in {
        "slime": types.ModuleType("slime"),
        "slime.backends": types.ModuleType("slime.backends"),
        "slime.backends.megatron_utils": types.ModuleType("slime.backends.megatron_utils"),
        "slime.backends.megatron_utils.misc_utils": misc,
        "slime.backends.megatron_utils.update_weight": types.ModuleType("slime.backends.megatron_utils.update_weight"),
        "slime.backends.megatron_utils.update_weight.common": common,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return per_mode


@pytest.mark.unit
def test_global_named_backup_is_rekeyed_to_vp_stages(slime_naming_stubs) -> None:
    from reef.train.slime_backend.reef_adapters.megatron.hf_export import rekey_backup_to_vp_stages

    # The backup as slime >= #2251 hands it over, already module-stripped.
    backup = {
        "decoder.layers.0.self_attention.linear_qkv.weight": "qkv",
        "decoder.layers.1.mlp.linear_fc1.weight": "fc1",
        "decoder.layers.2.mlp.linear_fc2.weight": "fc2",
        "decoder.final_layernorm.weight": "ln",
    }

    rekeyed = rekey_backup_to_vp_stages(args=None, model=None, backup_weights=backup)

    assert rekeyed == {
        "vp_stages.0.decoder.layers.0.self_attention.linear_qkv.weight": "qkv",
        "vp_stages.0.decoder.layers.1.mlp.linear_fc1.weight": "fc1",
        "vp_stages.1.decoder.layers.0.mlp.linear_fc2.weight": "fc2",
        # The key the 30B run died on: present after translation.
        "vp_stages.1.decoder.final_layernorm.weight": "ln",
    }


@pytest.mark.unit
def test_vp_stages_keys_pass_through_and_win(slime_naming_stubs) -> None:
    from reef.train.slime_backend.reef_adapters.megatron.hf_export import rekey_backup_to_vp_stages

    backup = {
        "decoder.final_layernorm.weight": "from-translation",
        "vp_stages.1.decoder.final_layernorm.weight": "already-local",
        "vp_stages.0.some.other.entry": "kept",
    }

    rekeyed = rekey_backup_to_vp_stages(args=None, model=None, backup_weights=backup)

    assert rekeyed["vp_stages.1.decoder.final_layernorm.weight"] == "already-local"
    assert rekeyed["vp_stages.0.some.other.entry"] == "kept"


@pytest.mark.unit
def test_missing_globals_are_skipped_not_fabricated(slime_naming_stubs) -> None:
    from reef.train.slime_backend.reef_adapters.megatron.hf_export import rekey_backup_to_vp_stages

    backup = {"decoder.layers.0.self_attention.linear_qkv.weight": "qkv"}

    rekeyed = rekey_backup_to_vp_stages(args=None, model=None, backup_weights=backup)

    assert rekeyed == {"vp_stages.0.decoder.layers.0.self_attention.linear_qkv.weight": "qkv"}
