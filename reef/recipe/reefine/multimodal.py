"""Reefine's multimodal gateway and the relay that sends a scenario's multimodal calls to it.

One gateway that serves images, embeddings, speech and decisions behind a
single key: OpenRouter by default, or an OpenAI-compatible one (OrcaRouter,
LiteLLM, ...). Reefine reads it from ``evolution.multimodal``, hands Reef a
:class:`ProviderRelay` for its scenarios, and gives the same gateway to its
agent proposer. The relay keeps the key the way the upstream runtime keeps the
chat key: request bodies pass through unchanged, in the gateway's own format,
and the preset only decides where each of Reef's routes lands and how an agent
lists the models.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from reef.inference.http import HOP_BY_HOP_HEADERS
from reef.runtime.interfaces import InferenceStream, MultimodalRelay, UpstreamStatusError


@dataclass(frozen=True)
class ProviderPreset:
    """A kind of multimodal gateway: its default address, its path for each Reef route it serves, and where its
    models are listed (``{base_url}`` and ``{modality}`` are filled in)."""

    name: str
    base_url: str | None
    paths: Mapping[str, str]
    models_url: str


PRESETS = {
    "openrouter": ProviderPreset(
        name="openrouter",
        base_url="https://openrouter.ai/api",
        paths={
            "/v1/images": "/v1/images",
            "/v1/embeddings": "/v1/embeddings",
            "/v1/audio/speech": "/v1/audio/speech",
            # OpenRouter's structured decision API lives outside /v1.
            "/v1/decisions": "/alpha/decisions",
        },
        models_url="{base_url}/v1/models?output_modalities={modality}",
    ),
    # OpenAI's routes, as the OpenAI-compatible gateways (OrcaRouter, LiteLLM, ...) serve them; no decisions.
    "openai-compatible": ProviderPreset(
        name="openai-compatible",
        base_url=None,
        paths={
            "/v1/images": "/v1/images/generations",
            "/v1/embeddings": "/v1/embeddings",
            "/v1/audio/speech": "/v1/audio/speech",
        },
        models_url="{base_url}/v1/models",
    ),
}
DEFAULT_PRESET = "openrouter"


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


@dataclass(frozen=True)
class MultimodalSettings:
    """A recipe's ``multimodal`` section, read: the preset, the gateway's address and the key its environment held
    (empty when none); :meth:`provider` settles the key against the chat upstream."""

    preset: ProviderPreset
    base_url: str
    api_key: str = field(default="", repr=False)

    @classmethod
    def from_config(cls, section: Any, environ: Mapping[str, str]) -> MultimodalSettings | None:
        """``{preset, url, api_key}``: the preset (``openrouter`` by default), the gateway's address when not the
        preset's, and its key, as ``inference.upstream_api_key`` takes the chat key (``${REEF_MULTIMODAL_API_KEY}``
        in the profile, or ``--recipe.config.evolution.multimodal.api_key`` on the command line); empty is no key.
        ``api_key_env`` names a variable instead, as for ``evolution.models``. No section is no settings."""
        if section is None:
            return None
        if not isinstance(section, Mapping):
            raise ValueError("multimodal must be a mapping of preset, url and api_key")
        name = str(section.get("preset") or DEFAULT_PRESET).strip()
        preset = PRESETS.get(name)
        if preset is None:
            raise ValueError(f"unknown multimodal preset {name!r}; known: {', '.join(sorted(PRESETS))}")
        address = str(section.get("url") or preset.base_url or "").strip().rstrip("/")
        if not address:
            raise ValueError(f"the {name} multimodal preset needs url: the gateway's address, no /v1")
        key = section.get("api_key")
        if key is not None and not isinstance(key, str):
            raise ValueError("multimodal.api_key must be a string")
        key_env = section.get("api_key_env")
        if key_env is not None and (not isinstance(key_env, str) or not key_env.isidentifier()):
            raise ValueError("multimodal.api_key_env must name an environment variable")
        if key is None and key_env:
            key = environ.get(key_env)
        return cls(preset=preset, base_url=address, api_key=(key or "").strip())

    def provider(self, upstream_url: str | None, upstream_api_key: str | None) -> MultimodalProvider | None:
        """The provider, its key the configured one, else the chat upstream's when the upstream is the same
        address; ``None`` without a key, and multimodal calls answer 501."""
        key = self.api_key
        if not key and upstream_url and upstream_url.strip().rstrip("/") == self.base_url:
            key = upstream_api_key or ""
        return MultimodalProvider(preset=self.preset, base_url=self.base_url, api_key=key) if key else None


class ProviderRelay(MultimodalRelay):
    """Relay multimodal calls to one provider at its own path for each route, with its key and nothing of Reef's."""

    def __init__(self, provider: MultimodalProvider, *, timeout_s: float = 300.0) -> None:
        self.provider = provider
        self._timeout_s = timeout_s

    async def relay(self, path: str, payload: dict[str, Any]) -> InferenceStream:
        from aiohttp import ClientSession, ClientTimeout

        upstream_path = self.provider.upstream_path(path)
        if upstream_path is None:
            raise NotImplementedError(f"the {self.provider.preset.name} provider serves no {path}")
        headers = {"Authorization": f"Bearer {self.provider.api_key}", "Accept-Encoding": "identity"}
        # The session outlives this call: the stream owns it until closed.
        session = ClientSession(timeout=ClientTimeout(total=self._timeout_s), auto_decompress=False)
        try:
            response = await session.post(f"{self.provider.base_url}{upstream_path}", json=payload, headers=headers)
        except Exception:
            await session.close()
            raise

        async def close() -> None:
            response.close()
            await session.close()

        if response.status >= 400:
            try:
                body = (await response.read()).decode(errors="replace")
            finally:
                await close()
            raise UpstreamStatusError(
                f"the {self.provider.preset.name} provider returned {response.status}: {body[:400]}",
                status=response.status,
            )
        return InferenceStream(
            status=response.status,
            headers={
                name: value for name, value in response.headers.items() if name.lower() not in HOP_BY_HOP_HEADERS
            },
            chunks=response.content.iter_any(),
            close=close,
        )


__all__ = [
    "DEFAULT_PRESET",
    "PRESETS",
    "MultimodalProvider",
    "MultimodalSettings",
    "ProviderPreset",
    "ProviderRelay",
]
