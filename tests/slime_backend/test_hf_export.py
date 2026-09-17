"""The colocated HF export reads the actor backup under either of Slime's naming schemes."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")  # hf_export imports the LoRA helpers, which need torch

from reef.train.slime_backend.reef_adapters.megatron.hf_export import stage_named_backup


def test_global_named_backup_is_filed_under_the_single_pipeline_stage() -> None:
    args = SimpleNamespace(pipeline_model_parallel_size=1, virtual_pipeline_model_parallel_size=None)
    backup = {"decoder.final_layernorm.weight": "norm", "decoder.layers.3.mlp.linear_fc1.weight": "fc1"}

    assert stage_named_backup(backup, args) == {
        "vp_stages.0.decoder.final_layernorm.weight": "norm",
        "vp_stages.0.decoder.layers.3.mlp.linear_fc1.weight": "fc1",
    }


def test_stage_named_backup_is_returned_as_is() -> None:
    args = SimpleNamespace(pipeline_model_parallel_size=2, virtual_pipeline_model_parallel_size=2)
    backup = {"vp_stages.1.decoder.layers.0.mlp.linear_fc1.weight": "fc1"}

    assert stage_named_backup(backup, args) is backup


def test_global_named_backup_refuses_several_pipeline_stages() -> None:
    args = SimpleNamespace(pipeline_model_parallel_size=2, virtual_pipeline_model_parallel_size=None)

    with pytest.raises(RuntimeError, match="only with one pipeline stage; this deployment has 2"):
        stage_named_backup({"decoder.layers.0.mlp.linear_fc1.weight": "fc1"}, args)
