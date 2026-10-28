"""The weight files of a Tinker release for Reef's own serving engines.

A Tinker release holds only its manifest (``reef.tinker.checkpoint``); the weights stay on the Tinker-compatible
service. ``TinkerSamplerWeights`` downloads the LoRA adapter that the manifest's ``sampler_path`` names and converts
it into a PEFT adapter in the base model's parameter names, which SGLang and vLLM load. This is tinker-cookbook's
own flow: ``weights.download`` of a ``sampler_weights`` path, then ``weights.build_lora_adapter``.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import tempfile
import urllib.request
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from reef.core.config import config_option
from reef.core.errors import ReefError
from reef.core.plugins import PluginConfigError, PluginSettings
from reef.core.weight_layout import WeightLayout
from reef.runtime.interfaces import WeightFiles
from reef.runtime.serving import ReleaseWeights
from reef.tinker.checkpoint import TinkerCheckpoint
from reef.tinker.service import close_service, tinker_sdk


@dataclass(frozen=True)
class TinkerSamplerWeightsSettings(PluginSettings):
    """The config of ``TinkerSamplerWeights``."""

    cache_dir: str = config_option(
        ".reef/tinker-sampler-weights",
        help="Directory of the downloaded and converted adapters, one per sampler path.",
    )
    api_key_env: str = config_option("TINKER_API_KEY", help="Environment variable containing the Tinker API key.")
    base_url: str | None = config_option(
        None, help="Base URL of the Tinker-compatible service that holds the weights; unset is hosted Tinker."
    )
    project_id: str | None = config_option(None, help="Optional Tinker project ID.")
    timeout_s: float = config_option(600.0, help="Seconds to wait for the service's download URL.")


class TinkerSamplerWeights(ReleaseWeights):
    """The adapter at a Tinker release's ``sampler_path``, downloaded from ``base_url`` and converted to PEFT.

    Values: ``cache_dir`` holds ``<sha256 of the sampler path>/adapter/`` for each adapter downloaded so far;
    ``api_key`` is a non-empty key of the service at ``base_url`` (``None`` is hosted Tinker). ``weight_files``
    gives the cached adapter, and downloads and converts it on first use. A sampler path names immutable weights,
    so a cached adapter never goes stale. The service must serve checkpoint archives: hosted Tinker does; Spindle
    does not, and its weights are read from its own Volume instead.

    Purpose: ``TinkerDeployment.engine_weights`` names it, and the engine serving runtimes call it at each load.
    """

    def __init__(
        self,
        cache_dir: Path,
        *,
        api_key: str,
        base_url: str | None = None,
        project_id: str | None = None,
        timeout_s: float = 600.0,
    ) -> None:
        if not api_key:
            raise ValueError("TinkerSamplerWeights needs a non-empty API key")
        self.cache_dir = cache_dir
        self.api_key = api_key
        self.base_url = base_url
        self.project_id = project_id
        self.timeout_s = timeout_s

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Self:
        settings = TinkerSamplerWeightsSettings.parse(config, plugin_class=cls)
        api_key = os.environ.get(settings.api_key_env, "").strip()
        if not api_key:
            raise PluginConfigError(f"TinkerSamplerWeights requires the environment variable {settings.api_key_env}")
        return cls(
            Path(settings.cache_dir).expanduser().resolve(),
            api_key=api_key,
            base_url=settings.base_url,
            project_id=settings.project_id,
            timeout_s=settings.timeout_s,
        )

    def weight_files(self, release: Path, layout: WeightLayout) -> WeightFiles:
        if layout != "adapter":
            raise ReefError(f"a Tinker release serves as an adapter, not as {layout} weights")
        try:
            checkpoint = TinkerCheckpoint.read(release)
        except (OSError, TypeError, ValueError) as exc:
            raise ReefError(f"the release in {release} holds no Tinker checkpoint manifest: {exc}") from exc
        if checkpoint.service != self.base_url:
            raise ReefError(
                f"the release's weights are held by {checkpoint.service or 'hosted Tinker'}, "
                f"not by {self.base_url or 'hosted Tinker'}"
            )
        try:
            from tinker_cookbook import weights
        except ImportError as exc:
            raise RuntimeError(
                "serving Tinker checkpoints on Reef's engines needs tinker-cookbook: uv pip install 'reef-infra[tinker]'"
            ) from exc
        entry = self.cache_dir / hashlib.sha256(checkpoint.sampler_path.encode()).hexdigest()
        adapter = entry / "adapter"
        if not (adapter / "adapter_config.json").is_file():
            entry.mkdir(parents=True, exist_ok=True)
            staging = entry / f".staging-{uuid.uuid4().hex}"
            try:
                raw = staging / "tinker"
                self.download(checkpoint.sampler_path, raw)
                converted = staging / "adapter"
                weights.build_lora_adapter(
                    base_model=checkpoint.base_model, adapter_path=str(raw), output_path=str(converted)
                )
                # Another load may have finished the same adapter meanwhile; both conversions are equal.
                if not (adapter / "adapter_config.json").is_file():
                    shutil.rmtree(adapter, ignore_errors=True)
                    os.replace(converted, adapter)
            finally:
                shutil.rmtree(staging, ignore_errors=True)
        return WeightFiles(adapter, "adapter")

    def download(self, path: str, directory: Path) -> None:
        """Extract the checkpoint archive of the ``tinker://`` path ``path`` into the new directory ``directory``."""
        sdk = tinker_sdk()
        service = sdk.ServiceClient(api_key=self.api_key, project_id=self.project_id, base_url=self.base_url)
        status = "errored"
        try:
            try:
                response = (
                    service.create_rest_client()
                    .get_checkpoint_archive_url_from_tinker_path(path)
                    .result(timeout=self.timeout_s)
                )
            except sdk.APIStatusError as exc:
                raise ReefError(f"the service gives no download of {path}: {exc}") from exc
            with tempfile.TemporaryDirectory(prefix="reef-tinker-archive-") as work:
                archive = Path(work) / "checkpoint.tar"
                urllib.request.urlretrieve(response.url, archive)
                with tarfile.open(archive) as tar:
                    # The data filter refuses links, devices and paths outside ``directory``.
                    tar.extractall(directory, filter="data")
            status = "success"
        finally:
            close_service(sdk, service, status, timeout_s=self.timeout_s)


__all__ = ["TinkerSamplerWeights", "TinkerSamplerWeightsSettings"]
