"""Admission of artifacts that carry a Hugging Face PEFT adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from reef.artifact import AdapterArtifactError, Artifact, PEFTValidator, read_peft_config

BASE_MODEL = "Qwen/Qwen3.6-27B"


def adapter(tmp_path: Path, *, weights: bool = True, layout: str = "safetensors", **overrides: Any) -> Artifact:
    """A minimal Hugging Face PEFT adapter directory, published as an artifact."""
    config: dict[str, Any] = {
        "peft_type": "LORA",
        "base_model_name_or_path": BASE_MODEL,
        "r": 32,
        "lora_alpha": 16,
        "target_modules": ["q_proj", "v_proj"],
    }
    config.update(overrides)
    root = tmp_path / "adapter"
    root.mkdir(parents=True, exist_ok=True)
    (root / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    if weights:
        (root / f"adapter_model.{layout}").write_bytes(b"\x00")
    return Artifact.local(root)


def test_validate_accepts_an_hf_peft_layout(tmp_path: Path) -> None:
    PEFTValidator(BASE_MODEL).validate(adapter(tmp_path))


def test_validate_accepts_the_bin_weight_layout(tmp_path: Path) -> None:
    PEFTValidator(BASE_MODEL).validate(adapter(tmp_path, layout="bin"))


def test_validate_accepts_any_peft_type(tmp_path: Path) -> None:
    PEFTValidator().validate(adapter(tmp_path, peft_type="DORA"))


def test_validate_rejects_a_tree_without_an_adapter_config(tmp_path: Path) -> None:
    root = tmp_path / "weights-only"
    root.mkdir()
    (root / "adapter_model.safetensors").write_bytes(b"\x00")
    with pytest.raises(AdapterArtifactError, match=r"adapter_config\.json"):
        PEFTValidator().validate(Artifact.local(root))


def test_validate_rejects_unreadable_json(tmp_path: Path) -> None:
    artifact = adapter(tmp_path)
    (tmp_path / "adapter" / "adapter_config.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(AdapterArtifactError, match="not readable JSON"):
        PEFTValidator().validate(artifact)
    (tmp_path / "adapter" / "adapter_config.json").write_text("[]", encoding="utf-8")
    with pytest.raises(AdapterArtifactError, match="JSON object"):
        read_peft_config(tmp_path / "adapter")


def test_validate_rejects_a_config_without_a_peft_type(tmp_path: Path) -> None:
    with pytest.raises(AdapterArtifactError, match="peft_type"):
        PEFTValidator().validate(adapter(tmp_path, peft_type=""))


def test_validate_rejects_an_adapter_without_weights(tmp_path: Path) -> None:
    with pytest.raises(AdapterArtifactError, match="no adapter weights"):
        PEFTValidator().validate(adapter(tmp_path, weights=False))


def test_validate_rejects_a_non_positive_rank(tmp_path: Path) -> None:
    with pytest.raises(AdapterArtifactError, match="positive integer"):
        PEFTValidator().validate(adapter(tmp_path, r=0))


def test_validate_rejects_an_adapter_fit_to_another_base_model(tmp_path: Path) -> None:
    with pytest.raises(AdapterArtifactError, match="never saw"):
        PEFTValidator(BASE_MODEL).validate(adapter(tmp_path, base_model_name_or_path="other/base"))


def test_validate_skips_the_base_model_check_when_unconfigured(tmp_path: Path) -> None:
    PEFTValidator().validate(adapter(tmp_path, base_model_name_or_path="other/base"))
