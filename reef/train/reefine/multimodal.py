"""Provider values used by Reefine proposal execution."""

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProviderPreset:
    """A kind of multimodal gateway: its default address, its path for each Reef route it serves, and where its
    models are listed (``{base_url}`` and ``{modality}`` are filled in)."""

    name: str
    base_url: str | None
    paths: Mapping[str, str]
    models_url: str


@dataclass(frozen=True)
class MultimodalProvider:
    """One configured multimodal gateway: a preset, its address (no ``/v1`` suffix) and its key."""

    preset: ProviderPreset
    base_url: str
    api_key: str = field(repr=False)

    def upstream_path(self, route: str) -> str | None:
        """The provider's path for one of Reef's multimodal routes, or ``None`` when it serves none."""
        return self.preset.paths.get(route)

    def models_url(self, modality: str) -> str:
        """Where the provider lists its models of one output modality (``speech``, ``image``, ``embeddings``,
        ``decisions``)."""
        return self.preset.models_url.format(base_url=self.base_url, modality=modality)
