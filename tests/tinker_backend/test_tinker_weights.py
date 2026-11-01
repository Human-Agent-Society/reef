"""The weight files of a Tinker release for Reef's engines: the adapter at its sampler path, downloaded once."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from reef.core.errors import ReefError
from reef.core.plugins import PluginConfigError
from reef.runtime.serving import ReleaseFiles
from reef.tinker.checkpoint import TinkerCheckpoint
from reef.tinker.weights import TinkerSamplerWeights

MODEL = "Qwen/Qwen3-8B"
CHECKPOINT = TinkerCheckpoint(MODEL, 32, "tinker://run/weights/4", "tinker://run/sampler_weights/4")


class APIStatusError(Exception):
    pass


class Service:
    """The SDK's ``ServiceClient`` over ``archives``, sampler paths to the files of their checkpoint archive."""

    def __init__(self, sdk: SDK) -> None:
        self.sdk = sdk

    def create_rest_client(self) -> Service:
        return self

    def get_checkpoint_archive_url_from_tinker_path(self, path: str) -> SimpleNamespace:
        self.sdk.requested.append(path)
        if path not in self.sdk.archives:
            raise APIStatusError("405 archive downloads are not served")
        archive = self.sdk.root / f"{len(self.sdk.requested)}.tar"
        with tarfile.open(archive, "w") as tar:
            for name, body in self.sdk.archives[path].items():
                info = tarfile.TarInfo(name)
                info.size = len(body)
                tar.addfile(info, io.BytesIO(body))
        return SimpleNamespace(result=lambda timeout: SimpleNamespace(url=archive.as_uri()))

    def close(self, status: str) -> SimpleNamespace:
        self.sdk.closed.append(status)
        return SimpleNamespace(result=lambda timeout: None)


class SDK:
    APIStatusError = APIStatusError
    NotFoundError = LookupError

    def __init__(self, root: Path) -> None:
        self.root = root
        self.archives: dict[str, dict[str, bytes]] = {}
        self.requested: list[str] = []
        self.closed: list[str] = []
        self.sessions: list[dict[str, object]] = []
        self.conversions: list[tuple[str, bytes]] = []

    def ServiceClient(self, **options: object) -> Service:  # noqa: N802
        self.sessions.append(options)
        return Service(self)


@pytest.fixture
def sdk(tmp_path, monkeypatch) -> SDK:
    value = SDK(tmp_path)
    value.archives[CHECKPOINT.sampler_path] = {"adapter_config.json": b"{}", "adapter_model.safetensors": b"tinker"}
    monkeypatch.setattr("reef.tinker.weights.tinker_sdk", lambda: value)

    def build_lora_adapter(*, base_model, adapter_path, output_path):
        value.conversions.append((base_model, (Path(adapter_path) / "adapter_model.safetensors").read_bytes()))
        Path(output_path).mkdir()
        (Path(output_path) / "adapter_config.json").write_text(json.dumps({"r": 32}))

    cookbook = ModuleType("tinker_cookbook")
    cookbook.weights = SimpleNamespace(build_lora_adapter=build_lora_adapter)
    monkeypatch.setitem(sys.modules, "tinker_cookbook", cookbook)
    return value


def release(tmp_path: Path, checkpoint: TinkerCheckpoint = CHECKPOINT) -> Path:
    directory = tmp_path / "release"
    checkpoint.write(directory)
    return directory


@pytest.mark.unit
def test_the_adapter_at_the_sampler_path_is_downloaded_converted_and_then_cached(tmp_path, sdk):
    weights = TinkerSamplerWeights(tmp_path / "cache", api_key="key", project_id="p")
    files = weights.weight_files(release(tmp_path), "adapter")
    digest = hashlib.sha256(CHECKPOINT.sampler_path.encode()).hexdigest()
    assert (files.path, files.layout) == (tmp_path / "cache" / digest / "adapter", "adapter")
    assert sdk.conversions == [(MODEL, b"tinker")]
    assert sdk.sessions == [{"api_key": "key", "project_id": "p", "base_url": None}] and sdk.closed == ["success"]
    # Only the converted adapter stays: the download and the staging directory are gone.
    assert [path.name for path in (tmp_path / "cache" / digest).iterdir()] == ["adapter"]
    # A sampler path names immutable weights, so a second load reads the cache.
    assert weights.weight_files(release(tmp_path), "adapter") == files
    assert sdk.requested == [CHECKPOINT.sampler_path]


@pytest.mark.unit
def test_a_release_the_weights_cannot_give_is_refused(tmp_path, sdk, monkeypatch):
    weights = TinkerSamplerWeights(tmp_path / "cache", api_key="key")
    with pytest.raises(ReefError, match="serves as an adapter, not as full weights"):
        weights.weight_files(release(tmp_path), "full")
    with pytest.raises(ReefError, match=r"held by https://spindle\.example, not by hosted Tinker"):
        weights.weight_files(release(tmp_path, replace(CHECKPOINT, service="https://spindle.example")), "adapter")
    missing = TinkerCheckpoint(MODEL, 32, "tinker://run/weights/5", "tinker://run/sampler_weights/5")
    with pytest.raises(ReefError, match="gives no download of tinker://run/sampler_weights/5"):
        weights.weight_files(release(tmp_path, missing), "adapter")
    assert sdk.closed == ["errored"]
    with pytest.raises(ReefError, match="holds no Tinker checkpoint manifest"):
        weights.weight_files(tmp_path / "empty", "adapter")
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    with pytest.raises(PluginConfigError, match="TINKER_API_KEY"):
        TinkerSamplerWeights.from_config({"cache_dir": str(tmp_path)})


@pytest.mark.unit
def test_a_release_without_an_adapter_gives_no_adapter_files(tmp_path):
    with pytest.raises(ReefError, match="holds no PEFT adapter in adapter/"):
        ReleaseFiles().weight_files(tmp_path, "adapter")
    assert ReleaseFiles().weight_files(tmp_path, "full").path == tmp_path
