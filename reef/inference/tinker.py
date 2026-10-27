"""Tinker's hosted sampler as a Reef inference runtime.

The trainer in ``reef.train.tinker_backend`` publishes checkpoints as
artifacts holding a ``tinker-checkpoint.json`` manifest that names the
immutable remote sampler; this module serves whatever sampler a frozen
artifact names, through Tinker's SDK, as an OpenAI text chat endpoint with
the exact training capture Reef records. The sampling itself is
``reef.tinker.sampling``, which the trainer's rollouts use too;
the manifest in the artifact is the whole contract about which sampler
serves.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from threading import RLock
from typing import Any

from reef_client.sse import synthesize_sse_events

from reef.artifact.artifact import Artifact
from reef.core.config import config_option
from reef.runtime.deployment import RuntimeFactory, config_secret
from reef.runtime.interfaces import (
    InferenceHandler,
    InferenceRuntime,
    InferenceStream,
    RuntimeLoadId,
    UpstreamStatusError,
    new_runtime_load_id_incarnation,
)
from reef.runtime.serving import holds_weights
from reef.tinker.checkpoint import MANIFEST, TinkerCheckpoint
from reef.tinker.sampling import TinkerSampler, TinkerSDKSampler, sample_chat

#: The hosted Tinker service, which ``service=None`` names.
TINKER_URL = "https://tinker.thinkingmachines.ai"


def checkpoint_sampler_path(artifact: Artifact, base_model: str, service: str | None) -> str | None:
    """The sampler a materialized release names, or None for Reef's empty base tree.

    ``ValueError``: the release is not materialized; its manifest names another model or another service than
    ``service``; or it holds files but no manifest.
    """
    path = artifact.materialize().local_path
    if path is None:
        raise ValueError("Tinker requires a materialized checkpoint manifest")
    if (path / MANIFEST).exists():
        checkpoint = TinkerCheckpoint.read(path)
        checkpoint.check_compatible(base_model, service=service)
        return checkpoint.sampler_path
    # Reef's empty base tree is the base model; an arbitrary uploaded weight directory must not become it.
    if holds_weights(path):
        raise ValueError(f"artifact is missing {MANIFEST}")
    return None


class TinkerInferenceRuntime(InferenceRuntime):
    """Serve the sampler each release names, on the Tinker-compatible service ``service`` (None: hosted Tinker).

    Every sampler served in this process gets a runtime load ID of this
    incarnation; activating a release whose sampler is not the one served
    mints a new one. A release of another service is refused.
    """

    def __init__(
        self,
        sampler: TinkerSampler,
        *,
        base_model: str,
        service: str | None = None,
        inference_timeout_s: float = 300.0,
    ) -> None:
        super().__init__(base_url=service or TINKER_URL, inference_timeout_s=inference_timeout_s)
        self._sampler = sampler
        self._model = base_model
        self._service = service
        self._handler = TinkerInferenceHandler(self, sampler, base_model)
        self._lock = RLock()
        self._incarnation = new_runtime_load_id_incarnation()
        self._load_count = 0
        self._versions: dict[str | None, str] = {}
        self._releases: dict[str, tuple[str | None, str]] = {}
        self._active, self._version = self._remember(None)
        self._closed = False

    @property
    def inference_handler(self) -> InferenceHandler:
        return self._handler

    @property
    def model_path(self) -> str:
        return self._model

    def current_runtime_load_id(self) -> str:
        """The version of the active sampler: the base model's until a release is activated."""
        with self._lock:
            return self._version

    # -- Snapshots

    def _remember(self, sampler_path: str | None) -> tuple[str | None, str]:
        """The load ID a sampler already serves under in this incarnation, else a new one."""
        version = self._versions.get(sampler_path)
        if version is None:
            _, version = self._new_load(sampler_path)
            self._versions[sampler_path] = version
        return sampler_path, version

    def _new_load(self, sampler_path: str | None) -> tuple[str | None, str]:
        version = str(RuntimeLoadId(self._incarnation, self._load_count))
        self._load_count += 1
        return sampler_path, version

    def snapshot(self, artifact: Artifact) -> tuple[str | None, str]:
        """Resolve exactly the frozen artifact; old in-flight calls keep their sampler."""
        with self._lock:
            if artifact.ref.release_id in self._releases:
                return self._releases[artifact.ref.release_id]
            sampler_path = checkpoint_sampler_path(artifact, self._model, self._service)
            # A release that carries the served sampler forward (a step of
            # another component, a rollback to the same weights) was never
            # activated; it serves under the load the engine reports now,
            # not under the first load of that sampler.
            carried = sampler_path == self._active
            selected = (self._active, self._version) if carried else self._remember(sampler_path)
            self._releases[artifact.ref.release_id] = selected
            return selected

    # -- Serving releases

    def activate_checkpoint(self, artifact: Artifact, *, scenario: str | None = None) -> str:
        """Serve the release's sampler; a sampler other than the served one is a new serving update."""
        with self._lock:
            sampler_path = checkpoint_sampler_path(artifact, self._model, self._service)
            if sampler_path != self._active:
                # Even a sampler served earlier in this incarnation, as after a rollback, serves under a new load.
                self._active, self._version = self._new_load(sampler_path)
            self._releases[artifact.ref.release_id] = (self._active, self._version)
            return self._version

    def shutdown(self) -> None:
        self.close_admission()
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._sampler.close()


class TinkerInferenceHandler(InferenceHandler):
    """Serve text chat from the sampler the frozen artifact resolves to."""

    def __init__(self, runtime: TinkerInferenceRuntime, sampler: TinkerSampler, model: str) -> None:
        self._runtime = runtime
        self._sampler = sampler
        self._model = model

    async def inference(self, artifact: Artifact, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if path != "/v1/chat/completions":
            raise UpstreamStatusError("Tinker supports /v1/chat/completions", status=400)
        return await asyncio.to_thread(self._sample, artifact, payload)

    def _sample(self, artifact: Artifact, payload: dict[str, Any]) -> dict[str, Any]:
        sampler_path, version = self._runtime.snapshot(artifact)
        return sample_chat(self._sampler, sampler_path, payload, model=self._model, runtime_load_id=version)

    async def inference_stream(self, artifact: Artifact, path: str, payload: dict[str, Any]) -> InferenceStream:
        return chat_stream(await self.inference(artifact, path, payload), payload)


# -- Request and response shaping ---------------------------------------------


def chat_stream(response: dict[str, Any], payload: dict[str, Any]) -> InferenceStream:
    """Emit a completed response as buffered OpenAI SSE, keeping it as the record."""
    include_usage = bool((payload.get("stream_options") or {}).get("include_usage"))

    async def chunks() -> AsyncIterator[bytes]:
        for event in synthesize_sse_events(response, include_usage=include_usage):
            yield event.encode()

    return InferenceStream(
        status=200, headers={"Content-Type": "text/event-stream"}, chunks=chunks(), record_response=response
    )


# -- Factory ------------------------------------------------------------------


@dataclass(frozen=True)
class TinkerInferenceConfig:
    """What the sampler needs; the trainer passes these from its own ``training.options``."""

    api_key: str | None = config_option(None, help="Tinker API key; prefer api_key_env.")
    api_key_env: str | None = config_option("TINKER_API_KEY", help="Environment variable holding the Tinker API key.")
    project_id: str | None = config_option(None, help="Optional Tinker project ID.")
    base_url: str | None = config_option(
        None, help="Base URL of the Tinker-compatible service; unset is Thinking Machines' hosted Tinker."
    )
    timeout_s: float = config_option(300.0, help="Sampling request timeout in seconds.")

    def __post_init__(self) -> None:
        if self.timeout_s <= 0:
            raise ValueError("runtime.timeout_s must be positive")
        if self.base_url is not None and not self.base_url.startswith(("https://", "http://")):
            raise ValueError("runtime.base_url must be an http(s) URL")


class TinkerInferenceRuntimeFactory(RuntimeFactory):
    """Build the Tinker sampling runtime for a base model; the SDK loads only here."""

    kind = "tinker"

    def config_type(self) -> type:
        return TinkerInferenceConfig

    def __call__(
        self,
        config: Mapping[str, Any],
        model_path: str,
        recipe_config: Mapping[str, Any],
        environ: Mapping[str, str],
    ) -> InferenceRuntime:
        key = config_secret(config, environ, "api_key", "api_key_env")
        if not key:
            raise ValueError(f"Tinker requires the environment variable {config.get('api_key_env')}")
        sampler = TinkerSDKSampler(
            model_path,
            api_key=key,
            project_id=config.get("project_id"),
            base_url=config.get("base_url"),
            timeout_s=config["timeout_s"],
        )
        try:
            return TinkerInferenceRuntime(
                sampler,
                base_model=model_path,
                service=config.get("base_url"),
                inference_timeout_s=config["timeout_s"],
            )
        except BaseException:
            sampler.close()
            raise
