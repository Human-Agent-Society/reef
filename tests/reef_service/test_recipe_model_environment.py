from __future__ import annotations

import pytest

from reef.recipe.config import load_recipe_config
from reef.recipe.errors import RecipeConfigError
from reef.recipe.registry import build_named_recipe


@pytest.mark.parametrize("environ", [{}, {"MODEL_ID": ""}, {"MODEL_ID": "   "}])
def test_missing_model_environment_fails_before_recipe_build(tmp_path, environ) -> None:
    path = tmp_path / "demo.yaml"
    path.write_text("implementation: recipe\nmodel:\n  path: ${MODEL_ID}\n")

    with pytest.raises(RecipeConfigError, match=r"model\.path requires environment variable MODEL_ID"):
        build_named_recipe("demo", environ, config_directory=tmp_path)


def test_model_environment_is_expanded_as_data_without_changing_prompt_text(tmp_path, monkeypatch) -> None:
    path = tmp_path / "demo.yaml"
    path.write_text(
        "implementation: recipe\nmodel:\n  path: ${MODEL_ID}\n"
        "evolution:\n  tasks: ['Print ${MODEL_ID} and ${UNSET_VAR}']\n"
    )
    monkeypatch.setenv("MODEL_ID", "process-model")
    selected = 'provider/model:latest#tag"'

    config = load_recipe_config(path, environ={"MODEL_ID": selected})

    assert config["model"]["path"] == selected
    assert config["evolution"]["tasks"] == ["Print ${MODEL_ID} and ${UNSET_VAR}"]
    assert load_recipe_config(path)["model"]["path"] == "process-model"


def test_named_recipe_passes_explicit_model_environment_to_runtime(tmp_path, monkeypatch) -> None:
    (tmp_path / "demo.yaml").write_text(
        "implementation: recipe\nmodel:\n  path: ${MODEL_ID}\n"
        "runtime:\n  type: inference_proxy\n  base_url: http://provider\n"
    )
    monkeypatch.setenv("MODEL_ID", "process-model")

    recipe = build_named_recipe("demo", {"MODEL_ID": "selected-model"}, config_directory=tmp_path)

    assert recipe.runtime.model_path == "selected-model"


def test_literal_model_id_still_works_without_environment(tmp_path) -> None:
    path = tmp_path / "demo.yaml"
    path.write_text("implementation: recipe\nmodel:\n  path: provider/literal-model\n")

    assert load_recipe_config(path, environ={})["model"]["path"] == "provider/literal-model"
