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
from reef.artifact.sources import read_reference
from reef.core.config import config_option
from reef.core.evaluation import ArtifactKind, PolicyModel
from reef.runtime.deployment import RuntimeFactory, config_secret
from reef.runtime.interfaces import (
    InferenceHandler,
    InferenceRuntime,
    InferenceStream,
    RuntimeLoadId,
    UpstreamStatusError,
    new_runtime_load_id_incarnation,
)
from reef.tinker.checkpoint import MANIFEST, TinkerCheckpoint
from reef.tinker.sampling import TinkerSampler, TinkerSDKSampler, sample_chat

#: The hosted Tinker service, which ``service=None`` names.
TINKER_URL = "https://tinker.thinkingmachines.ai"


def checkpoint_sampler_path(
    artifact: Artifact, base_model: str, service: str | None, *, full: bool = False
) -> str | None:
    """The sampler that a release's checkpoint manifest names; ``None`` for the base model's own weights.

    A LoRA's base artifact holds the untrained adapter's manifest, so its sampler serves the base model's outputs.
    Under full-parameter training (``full``), a release without a manifest has the base model's weights: the
    component's base, a reference to the base revision, or an import of the base's files (RFC #996).
    ``ValueError``: the release is not materialized; it has no manifest and is not full-parameter, or refers to
    another repo; or its manifest names another model or another service than ``service``.
    """
    path = artifact.materialize().local_path
    if path is None:
        raise ValueError("Tinker requires a materialized checkpoint manifest")
    if not (path / MANIFEST).exists():
        if not full:
            raise ValueError(f"artifact is missing {MANIFEST}")
        revision = read_reference(path)
        if revision is not None and revision.repo != base_model:
            raise ValueError(f"the artifact refers to {revision.uri}, not to the base model {base_model}")
        return None
    checkpoint = TinkerCheckpoint.read(path)
    checkpoint.check_compatible(base_model, service=service)
    return checkpoint.sampler_path


class TinkerInferenceRuntime(InferenceRuntime):
    """Serve the sampler each release names, on the Tinker-compatible service ``service`` (None: hosted Tinker).

    ``served_weight_kind`` is ``model_adapter`` for a LoRA's releases, and ``model`` for the releases of
    Spindle's full-parameter training, whose base serves the base model (``checkpoint_sampler_path``).
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
        weight_kind: ArtifactKind = "model_adapter",
    ) -> None:
        super().__init__(base_url=service or TINKER_URL, inference_timeout_s=inference_timeout_s)
        if weight_kind not in ("model", "model_adapter"):
            raise ValueError(f"a Tinker sampler serves a model or a model_adapter, not {weight_kind!r}")
        self._sampler = sampler
        self._model = base_model
        self._service = service
        self._kind = weight_kind
        self._handler = TinkerInferenceHandler(self, sampler, base_model)
        self._lock = RLock()
        self._incarnation = new_runtime_load_id_incarnation()
        self._load_count = 0
        #: The runtime load ID that each sampler served under last, by sampler path (None: the base model).
        self._versions: dict[str | None, str] = {}
        self._active, self._version = self._remember(None)
        self._closed = False

    @property
    def served_weight_kind(self) -> ArtifactKind:
        return self._kind

    def sampler_path(self, artifact: Artifact) -> str | None:
        """The sampler that serves ``artifact``; ``None`` for the base model (``checkpoint_sampler_path``)."""
        return checkpoint_sampler_path(artifact, self._model, self._service, full=self._kind == "model")

    def served_model_name(self, scenario: str) -> str:
        """The name of the sampler that serves now: its sampler path, or the base model before any release.
        A request under this name reaches that sampler on the service at ``base_url``."""
        with self._lock:
            return self._active or self._model

    def sampler_load(self, model_name: str) -> tuple[str | None, str]:
        """The sampler that ``model_name`` names (None for the base model) and its runtime load ID."""
        sampler_path = None if model_name == self._model else model_name
        with self._lock:
            if sampler_path == self._active:
                return self._active, self._version
            return self._remember(sampler_path)

    @property
    def serves_artifacts_on_request(self) -> bool:
        return True

    def artifact_model(self, artifact: Artifact) -> PolicyModel:
        """The model that serves ``artifact``: each call samples its own artifact.

        A LoRA's release is the base model with the adapter that its manifest names. A full-parameter release is a
        model of its own, which Spindle serves under its sampler path; its base is the base model.
        """
        sampler_path = self.sampler_path(artifact)
        if self._kind == "model":
            return PolicyModel(sampler_path or self._model)
        return PolicyModel(self._model, sampler_path)

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

    # -- Serving releases

    def activate_checkpoint(self, artifact: Artifact, *, scenario: str | None = None) -> str:
        """Serve the release's sampler; a sampler other than the served one is a new serving update."""
        with self._lock:
            sampler_path = self.sampler_path(artifact)
            if sampler_path != self._active:
                # Even a sampler served earlier in this incarnation, as after an activation, serves under a new load.
                self._active, self._version = self._new_load(sampler_path)
                self._versions[sampler_path] = self._version
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
        """Sample from the sampler that the request's ``model`` names (``address_model``)."""
        if path != "/v1/chat/completions":
            raise UpstreamStatusError("Tinker supports /v1/chat/completions", status=400)
        return await asyncio.to_thread(self.sample, payload)

    def sample(self, payload: dict[str, Any]) -> dict[str, Any]:
        sampler_path, version = self._runtime.sampler_load(str(payload.get("model") or self._model))
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
    parameterization: str = config_option(
        "lora", help="lora: each release is a LoRA's manifest; full: Spindle's full-parameter releases."
    )

    def __post_init__(self) -> None:
        if self.timeout_s <= 0:
            raise ValueError("runtime.timeout_s must be positive")
        if self.parameterization not in ("lora", "full"):
            raise ValueError("runtime.parameterization must be lora or full")
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
                weight_kind="model" if config.get("parameterization") == "full" else "model_adapter",
            )
        except BaseException:
            sampler.close()
            raise
