"""Guarantees of reef.harness.render and the bundled adapter descriptors."""

from __future__ import annotations

from pathlib import Path

import pytest

from reef.harness.adapters import available_adapters, get_adapter
from reef.harness.render import RenderError, render_composition

GOLDENS = Path(__file__).parent / "data" / "harness_goldens"

# One node of every kind; the models config target exists only on pi.
NODES = [
    ("config", {"data": {"defaultModel": "qwen/qwen3-8b", "defaultProvider": "qwen"}}),
    (
        "config",
        {
            "target": "models",
            "data": {
                "providers": {
                    "qwen": {
                        "api": "openai-completions",
                        "apiKey": "dummy",
                        "baseUrl": "http://localhost:8000/v1",
                        "models": [{"id": "qwen3-8b"}],
                    }
                }
            },
        },
    ),
    ("rules", {"text": "Answer briefly."}),
    ("rules", {"text": "Prefer the standard library."}),
    ("skill", {"name": "notes", "text": "# Notes skill\n\nKeep short notes."}),
    ("agent_command", {"name": "summarize", "text": "Summarize $1."}),
    ("code_extension", {"name": "tracer", "code": "export default function tracer() {}"}),
]


def golden_tree(adapter: str) -> dict[str, str]:
    root = GOLDENS / adapter
    return {path.relative_to(root).as_posix(): path.read_text() for path in sorted(root.rglob("*")) if path.is_file()}


def test_pi_render_matches_the_golden_tree() -> None:
    assert render_composition(NODES, get_adapter("pi")) == golden_tree("pi")


def test_opencode_render_matches_the_golden_tree() -> None:
    nodes = [node for node in NODES if node[1].get("target") != "models"]
    assert render_composition(nodes, get_adapter("opencode")) == golden_tree("opencode")


def test_config_nodes_deep_merge_in_tree_order() -> None:
    files = render_composition(
        [
            ("config", {"data": {"compaction": {"enabled": True, "keep": 4}, "defaultProvider": "a"}}),
            ("config", {"data": {"compaction": {"keep": 8}}}),  # later node wins per key
        ],
        get_adapter("pi"),
    )
    assert '"defaultProvider": "a"' in files["pi-agent/settings.json"]  # sibling keys survive the merge
    assert '"enabled": true' in files["pi-agent/settings.json"]
    assert '"keep": 8' in files["pi-agent/settings.json"]


def test_unknown_config_target_is_rejected() -> None:
    with pytest.raises(RenderError, match="no config target 'models'"):
        render_composition([("config", {"target": "models", "data": {}})], get_adapter("opencode"))


def test_two_nodes_cannot_render_to_the_same_path() -> None:
    skill = ("skill", {"name": "notes", "text": "# notes"})
    with pytest.raises(RenderError, match="same path"):
        render_composition([skill, skill], get_adapter("pi"))


def test_opencode_quirk_rejects_reopened_autoupdate() -> None:
    with pytest.raises(RenderError, match="autoupdate false"):
        render_composition([("config", {"data": {"autoupdate": True}})], get_adapter("opencode"))


def test_bundled_adapters_are_discoverable() -> None:
    assert set(available_adapters()) >= {"opencode", "pi"}
