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
import json
import uuid
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from threading import RLock
from typing import Any

from reef_client.sse import synthesize_sse_events

from reef.artifact.artifact import REPOSITORY_FILES, Artifact
from reef.core.config import config_option
from reef.runtime.deployment import RuntimeFactory, config_secret
from reef.runtime.interfaces import InferenceHandler, InferenceRuntime, InferenceStream, UpstreamStatusError
from reef.tinker.sampling import TinkerSampler, TinkerSDKSampler, sample_chat

#: The manifest a Tinker checkpoint artifact carries; the trainer writes it, this side reads it.
MANIFEST = "tinker-checkpoint.json"
TINKER_URL = "https://tinker.thinkingmachines.ai"


def checkpoint_sampler_path(artifact: Artifact, base_model: str) -> str | None:
    """The remote sampler a materialized artifact names; None for Reef's empty base tree."""
    path = artifact.materialize().local_path
    if path is None:
        raise ValueError("Tinker requires a materialized checkpoint manifest")
    manifest = path / MANIFEST
    if manifest.exists():
        value = json.loads(manifest.read_text())
        if not isinstance(value, Mapping) or value.get("base_model") != base_model:
            raise ValueError("Tinker checkpoint manifest does not name the served base model")
        sampler = value.get("sampler_path")
        if not isinstance(sampler, str) or not sampler.startswith("tinker://"):
            raise ValueError("Tinker checkpoint manifest must name a tinker:// sampler")
        return sampler
    # Reef's empty base tree represents the initial seeded adapter.
    # An arbitrary uploaded weight directory must not silently become base.
    contents = {entry.name for entry in path.iterdir()} - REPOSITORY_FILES
    if contents:
        raise ValueError(f"artifact is missing {MANIFEST}")
    return None


class TinkerInferenceRuntime(InferenceRuntime):
    """Serve the sampler each frozen artifact names, and follow Reef's publication of new ones.

    Every snapshot served in this process gets a runtime load ID of this
    incarnation; activating a release whose sampler is not the one served
    mints a new one.
    """

    def __init__(
        self,
        sampler: TinkerSampler,
        *,
        base_model: str,
        base_url: str = TINKER_URL,
        inference_timeout_s: float = 300.0,
    ) -> None:
        super().__init__(base_url=base_url, inference_timeout_s=inference_timeout_s)
        self._sampler = sampler
        self._model = base_model
        self._handler = TinkerInferenceHandler(self, sampler, base_model)
        self._lock = RLock()
        self._incarnation = uuid.uuid4().hex
        self._load_count = 0
        self._versions: dict[str | None, str] = {}
        self._releases: dict[str, tuple[str | None, str]] = {}
        self._active, self._version = self._remember(None)
        self._closed = False
        # The base model the runtime opens with is the published head until an artifact is bound.
        self.mark_published()

    @property
    def inference_handler(self) -> InferenceHandler:
        return self._handler

    @property
    def model_path(self) -> str:
        return self._model

    def serving_runtime_load_id(self) -> str:
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
        version = f"{self._incarnation}:{self._load_count}"
        self._load_count += 1
        return sampler_path, version

    def snapshot(self, artifact: Artifact) -> tuple[str | None, str]:
        """Resolve exactly the frozen artifact; old in-flight calls keep their sampler."""
        with self._lock:
            if artifact.ref.release_id in self._releases:
                return self._releases[artifact.ref.release_id]
            sampler_path = checkpoint_sampler_path(artifact, self._model)
            # A release that carries the served sampler forward (a step of
            # another component, a rollback to the same weights) was never
            # activated; it serves under the load the engine reports now,
            # not under the first load of that sampler.
            carried = sampler_path == self._active
            selected = (self._active, self._version) if carried else self._remember(sampler_path)
            self._releases[artifact.ref.release_id] = selected
            return selected

    # -- Weight updates

    def restore_checkpoint(self, artifact: Artifact, *, scenario: str | None = None) -> str:
        # Rollback validates the target first; activate_checkpoint binds the
        # republished artifact. No mutable remote serving slot needs restoring.
        return self.snapshot(artifact)[1]

    def activate_checkpoint(self, artifact: Artifact, *, scenario: str | None = None) -> str:
        """Serve the release's sampler; a sampler other than the served one is a new serving update."""
        with self._lock:
            sampler_path = checkpoint_sampler_path(artifact, self._model)
            if sampler_path != self._active:
                # Even a sampler served earlier in this incarnation, as after a rollback, serves under a new load.
                self._active, self._version = self._new_load(sampler_path)
            self._releases[artifact.ref.release_id] = (self._active, self._version)
            self.mark_published()
            return self._version

    def shutdown(self) -> None:
        self.pause_admission()
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
    timeout_s: float = config_option(300.0, help="Sampling request timeout in seconds.")

    def __post_init__(self) -> None:
        if self.timeout_s <= 0:
            raise ValueError("runtime.timeout_s must be positive")


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
            model_path, api_key=key, project_id=config.get("project_id"), timeout_s=config["timeout_s"]
        )
        try:
            return TinkerInferenceRuntime(sampler, base_model=model_path, inference_timeout_s=config["timeout_s"])
        except BaseException:
            sampler.close()
            raise
