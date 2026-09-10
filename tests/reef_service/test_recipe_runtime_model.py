from __future__ import annotations

import pytest

from reef.recipe.config import load_recipe_config
from reef.recipe.errors import RecipeConfigError
from reef.recipe.registry import build_named_recipe
from reef.runtime.adapters.inference_proxy import InferenceProxyRuntime


def test_recipe_loader_preserves_environment_references(tmp_path, monkeypatch) -> None:
    path = tmp_path / "demo.yaml"
    path.write_text(
        "implementation: recipe\nmodel:\n  path: ${MODEL_ID}\n"
        "evolution:\n  tasks: ['Print ${MODEL_ID} and ${UNSET_VAR}']\n"
    )
    monkeypatch.setenv("MODEL_ID", "process-model")

    config = load_recipe_config(path)

    assert config["model"]["path"] == "${MODEL_ID}"
    assert config["evolution"]["tasks"] == ["Print ${MODEL_ID} and ${UNSET_VAR}"]


def test_named_recipe_reuses_default_runtime_model(tmp_path) -> None:
    (tmp_path / "demo.yaml").write_text("implementation: recipe\n")
    runtime = InferenceProxyRuntime(base_url="http://provider", model_path="selected-model")

    recipe = build_named_recipe("demo", {}, config_directory=tmp_path, default_runtime=runtime)

    assert recipe.runtime is runtime
    assert recipe.runtime.model_path == "selected-model"


@pytest.mark.parametrize("model_id", [None, "", "   "])
def test_missing_model_fails_before_recipe_build(tmp_path, model_id) -> None:
    (tmp_path / "demo.yaml").write_text("implementation: recipe\n")
    runtime = None if model_id is None else InferenceProxyRuntime(base_url="http://provider", model_path=model_id)

    with pytest.raises(RecipeConfigError, match=r"requires a non-empty model\.path"):
        build_named_recipe("demo", {}, config_directory=tmp_path, default_runtime=runtime)


@pytest.mark.parametrize("model_yaml", ["", "model:\n  path: selected-model\n"])
def test_recipe_runtime_config_does_not_inherit_default_runtime_model(tmp_path, model_yaml) -> None:
    (tmp_path / "demo.yaml").write_text(
        "implementation: recipe\nruntime:\n  type: inference_proxy\n  base_url: http://recipe-provider\n" + model_yaml
    )
    runtime = InferenceProxyRuntime(base_url="http://default-provider", model_path="default-model")

    if not model_yaml:
        with pytest.raises(RecipeConfigError, match=r"requires a non-empty model\.path"):
            build_named_recipe("demo", {}, config_directory=tmp_path, default_runtime=runtime)
    else:
        recipe = build_named_recipe("demo", {}, config_directory=tmp_path, default_runtime=runtime)
        assert recipe.runtime is not runtime
        assert recipe.runtime.model_path == "selected-model"


@pytest.mark.parametrize("model_yaml", ["null", "''", "'   '", "42"])
def test_explicit_invalid_model_is_not_replaced_by_runtime_model(tmp_path, model_yaml) -> None:
    (tmp_path / "demo.yaml").write_text(f"implementation: recipe\nmodel:\n  path: {model_yaml}\n")
    runtime = InferenceProxyRuntime(base_url="http://provider", model_path="default-model")

    with pytest.raises(RecipeConfigError, match=r"requires a non-empty model\.path"):
        build_named_recipe("demo", {}, config_directory=tmp_path, default_runtime=runtime)
