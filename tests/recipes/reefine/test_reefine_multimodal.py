"""Reefine's multimodal gateway: its settings, and the agent that reaches it.

Reef's inference service relays no multimodal call; only the agent's trials reach the gateway.
"""

from __future__ import annotations


import pytest

from recipes.beta.reefine.multimodal import MultimodalSettings
from reef.core.plugins import PluginConfigError
from reef.recipe.loading import load_recipe


@pytest.mark.unit
def test_multimodal_settings_read_the_preset_address_and_key() -> None:
    assert MultimodalSettings.from_config(None, {}) is None
    settings = MultimodalSettings.from_config({"api_key": " sk-or "}, {})
    assert (settings.preset.name, settings.base_url, settings.api_key) == (
        "openrouter",
        "https://openrouter.ai/api",
        "sk-or",
    )
    assert "sk-or" not in repr(settings)
    # A variable instead, as evolution.models take one; a literal wins.
    assert MultimodalSettings.from_config({"api_key_env": "MM_KEY"}, {"MM_KEY": "sk-env"}).api_key == "sk-env"
    assert (
        MultimodalSettings.from_config({"api_key": "sk-lit", "api_key_env": "MM_KEY"}, {"MM_KEY": "x"}).api_key
        == "sk-lit"
    )
    # No key of its own (the profile's ${REEF_MULTIMODAL_API_KEY} unset is empty): the upstream's, only when the
    # upstream is the same gateway.
    keyless = MultimodalSettings.from_config({"api_key": ""}, {})
    assert keyless.provider("https://openrouter.ai/api/", "upstream").api_key == "upstream"
    assert keyless.provider("http://127.0.0.1:11434", "ollama") is None
    compatible = MultimodalSettings.from_config({"preset": "openai-compatible", "url": "https://gw.example/"}, {})
    assert compatible.base_url == "https://gw.example"
    with pytest.raises(ValueError, match="needs url"):
        MultimodalSettings.from_config({"preset": "openai-compatible"}, {})
    with pytest.raises(ValueError, match="unknown multimodal preset"):
        MultimodalSettings.from_config({"preset": "nope"}, {})


@pytest.mark.unit
def test_the_method_settles_its_gateway_key_against_the_served_model() -> None:
    """The deployment writes ``multimodal`` in the method's config; the method settles the gateway's key against
    the served model of its job, so the agent's trials reach the gateway."""
    from recipes.beta.reefine import ReefineMethod, ReefineRecipe

    settings = {
        "evolution": {
            "evaluation": {"tasks": ["[health] reply reef-ok"]},
            "method": {"config": {"multimodal": {"preset": "openrouter"}}},
            "agent": None,
        }
    }
    recipe = load_recipe(ReefineRecipe, settings, {})
    assert isinstance(recipe, ReefineRecipe)
    method = ReefineMethod.from_config(recipe.evolution.method["config"])
    assert method.multimodal is not None
    assert method.multimodal.provider("https://openrouter.ai/api", "sk-upstream").api_key == "sk-upstream"
    assert method.multimodal.provider("http://127.0.0.1:11434", "ollama") is None
    with pytest.raises(PluginConfigError, match="multimodal must be a mapping"):
        ReefineMethod.from_config({"multimodal": "openrouter"})
