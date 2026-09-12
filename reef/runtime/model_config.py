"""The current model override shared by a scenario and its recipe."""

from __future__ import annotations

from dataclasses import dataclass, field

from reef.runtime.adapters.inference_proxy import InferenceProxyRuntime


@dataclass(eq=False)
class ModelConfig:
    """One model selection; replacing runtime leaves earlier snapshots intact."""

    runtime: InferenceProxyRuntime | None = field(default=None, repr=False)

    @classmethod
    def from_value(cls, value: object) -> ModelConfig:
        """Validate a model override; null selects the recipe's default."""
        return cls(runtime=InferenceProxyRuntime.from_model_config(value))

    def view(self) -> dict[str, object] | None:
        """Describe the selection for API acknowledgements without its credential."""
        runtime = self.runtime
        if runtime is None:
            return None
        return {
            "url": runtime.base_url,
            "model": runtime.model_path,
            "api": runtime.api,
            "has_api_key": bool(runtime.api_key),
        }
