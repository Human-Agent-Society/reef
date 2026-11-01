"""Reefine's multimodal gateway: its settings, and the agent that reaches it.

Reef's inference service relays no multimodal call; only the agent's trials reach the gateway.
"""

from __future__ import annotations

import pytest

from recipes.beta.reefine.multimodal import MultimodalSettings
from reef.inference.http import InferenceProxyRuntime
from reef.recipe.errors import RecipeConfigError


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
def test_reefine_hands_its_gateway_to_the_agent() -> None:
    """The agent's trials (``ReefineRecipe.build_backend`` binds the agent to this provider) reach the gateway."""
    from recipes.beta.reefine import ReefineRecipe

    settings = {
        "evolution": {
            "evaluation": {"tasks": ["[health] reply reef-ok"]},
            "multimodal": {"preset": "openrouter"},
            "agent": None,
        }
    }
    kwargs = ReefineRecipe._recipe_kwargs(settings, {})
    upstream = InferenceProxyRuntime(base_url="https://openrouter.ai/api", api_key="sk-upstream", model_path="m")
    recipe = ReefineRecipe(**kwargs, runtime=upstream)
    assert recipe.multimodal_provider is not None and recipe.multimodal_provider.api_key == "sk-upstream"
    elsewhere = InferenceProxyRuntime(base_url="http://127.0.0.1:11434", api_key="ollama", model_path="m")
    assert ReefineRecipe(**kwargs, runtime=elsewhere).multimodal_provider is None
    with pytest.raises(RecipeConfigError, match=r"evolution\.multimodal"):
        ReefineRecipe._recipe_kwargs({"evolution": {**settings["evolution"], "multimodal": "openrouter"}}, {})
