from __future__ import annotations

import io
import json

import pytest

from reef_client import SkillNotServedError
from reef_client import skill as skill_client


class _FakeResponse(io.BytesIO):
    def __init__(self, text: str, version: str) -> None:
        super().__init__(text.encode("utf-8"))
        self.headers = {"x-reef-artifact-version": version}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _serve(monkeypatch, text: str, version: str) -> dict:
    seen: dict = {}
    manifest = json.dumps({"artifact_version": version, "files": {"skills/SKILL.md": text}})

    def fake_urlopen(request, timeout=0):
        seen["url"] = request.full_url
        seen["headers"] = dict(request.headers)
        return _FakeResponse(manifest, version)

    monkeypatch.setattr(skill_client.urllib.request, "urlopen", fake_urlopen)
    return seen


def test_fetch_skill_sends_scenario_and_token(monkeypatch) -> None:
    seen = _serve(monkeypatch, "Always check units.", "version-1")
    text, version = skill_client.fetch_skill("http://reef:8900/", "my-agent", token="secret")
    assert (text, version) == ("Always check units.", "version-1")
    assert seen["url"] == "http://reef:8900/reef/harness"
    assert seen["headers"]["X-reef-scenario"] == "my-agent"
    assert seen["headers"]["Authorization"] == "Bearer secret"


def test_fetch_skill_raises_when_the_scenario_serves_no_skill(monkeypatch) -> None:
    manifest = json.dumps({"artifact_version": "version-1", "files": {}})

    def fake_urlopen(request, timeout=0):
        return _FakeResponse(manifest, "version-1")

    monkeypatch.setattr(skill_client.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(SkillNotServedError, match=r"serves no skills/SKILL\.md"):
        skill_client.fetch_skill("http://reef:8900", "s")


def test_sync_skill_writes_only_when_changed(monkeypatch, tmp_path) -> None:
    _serve(monkeypatch, "v1 text", "version-1")
    out = tmp_path / "skills" / "SKILL.md"
    version, changed = skill_client.sync_skill("http://reef:8900", "s", out)
    assert changed and out.read_text(encoding="utf-8") == "v1 text" and version == "version-1"
    stamp = out.stat().st_mtime_ns
    version, changed = skill_client.sync_skill("http://reef:8900", "s", out)
    assert not changed and out.stat().st_mtime_ns == stamp
    _serve(monkeypatch, "v2 text", "version-2")
    version, changed = skill_client.sync_skill("http://reef:8900", "s", out)
    assert changed and out.read_text(encoding="utf-8") == "v2 text" and version == "version-2"
