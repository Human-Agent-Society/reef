"""The long-context view must preserve weights and refuse ambiguous scaling."""

import json

import pytest

from recipes.sao.examples.servelearnbench.long_context import prepare_model


def test_model_view_preserves_source_and_links_weights(tmp_path):
    source, view = tmp_path / "source", tmp_path / "view"
    source.mkdir()
    config = {"model_type": "qwen3", "max_position_embeddings": 40960, "rope_theta": 1000000}
    original = json.dumps(config)
    (source / "config.json").write_text(original)
    (source / "model.safetensors").write_bytes(b"immutable weights")
    (source / "tokenizer.json").write_text("{}")
    prepare_model(source, view)
    result = json.loads((view / "config.json").read_text())
    assert (source / "config.json").read_text() == original
    assert result["max_position_embeddings"] == 131072
    assert result["rope_scaling"]["original_max_position_embeddings"] == 32768
    assert result["rope_scaling"]["factor"] == 4
    assert result["rope_scaling"]["rope_type"] == "yarn"
    assert (view / "model.safetensors").resolve() == source / "model.safetensors"
    assert (view / "tokenizer.json").is_symlink()
    with pytest.raises(ValueError, match="fresh destination"):
        prepare_model(source, view)


@pytest.mark.parametrize(
    "config",
    [
        {"model_type": "qwen3_moe"},
        {"model_type": "qwen3", "rope_scaling": {"rope_type": "yarn", "factor": 2}},
    ],
)
def test_model_view_refuses_wrong_architecture_or_existing_scaling(tmp_path, config):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError):
        prepare_model(source, tmp_path / "view")
    assert not (tmp_path / "view").exists()
