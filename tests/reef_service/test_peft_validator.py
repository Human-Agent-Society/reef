"""Admission of artifacts that carry a Hugging Face PEFT adapter."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from reef.artifact import AdapterArtifactError, Artifact, PEFTValidator, read_peft_config
from reef.artifact.peft import ADAPTER_PROVENANCE, PROVENANCE_SCHEMA, read_provenance


@contextmanager
def fail_on(message: str):
    """pytest.raises with the message treated as text, not a regex."""
    with pytest.raises(AdapterArtifactError) as raised:
        yield raised
    assert message in str(raised.value)


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


def provenance(root: Path, **overrides: Any) -> dict[str, Any]:
    """Reef's sidecar for the adapter `adapter()` wrote, written into it."""
    config_path = root / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    document: dict[str, Any] = {
        "schema": PROVENANCE_SCHEMA,
        "base_model": {"name_or_path": config["base_model_name_or_path"], "tokenizer": {}},
        "peft": {key: config[key] for key in ("peft_type", "r", "lora_alpha", "target_modules")},
        "dtype": "bfloat16",
        "source": {"scenario": "math", "scenario_step": 7},
        "files": {name: digest(root / name) for name in ("adapter_config.json", "adapter_model.safetensors")},
    }
    document.update(overrides)
    (root / ADAPTER_PROVENANCE).write_text(json.dumps(document), encoding="utf-8")
    return document


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_an_artifact_without_provenance_stays_admissible(tmp_path: Path) -> None:
    """An offline SFT run or a Hub adapter is legitimate; it just cannot be audited."""
    artifact = adapter(tmp_path)
    assert read_provenance(Path(artifact.local_path)) is None
    PEFTValidator(BASE_MODEL).validate(artifact)


def test_requiring_provenance_refuses_an_unauditable_adapter(tmp_path: Path) -> None:
    with pytest.raises(AdapterArtifactError, match="Reef exported"):
        PEFTValidator(BASE_MODEL, require_provenance=True).validate(adapter(tmp_path))


def test_a_reef_exported_adapter_validates_against_its_provenance(tmp_path: Path) -> None:
    artifact = adapter(tmp_path)
    document = provenance(Path(artifact.local_path))
    PEFTValidator(BASE_MODEL, require_provenance=True).validate(artifact)
    assert document["source"] == {"scenario": "math", "scenario_step": 7}


def test_provenance_from_an_unknown_schema_is_refused(tmp_path: Path) -> None:
    """A reader that does not know the rules must not validate against the wrong ones."""
    artifact = adapter(tmp_path)
    provenance(Path(artifact.local_path), schema=PROVENANCE_SCHEMA + 1)
    with pytest.raises(AdapterArtifactError, match="understands"):
        PEFTValidator(BASE_MODEL).validate(artifact)


def test_modified_adapter_weights_fail_their_recorded_checksum(tmp_path: Path) -> None:
    artifact = adapter(tmp_path)
    root = Path(artifact.local_path)
    provenance(root)
    (root / "adapter_model.safetensors").write_bytes(b"\x01")
    with pytest.raises(AdapterArtifactError, match="corrupted or modified after export"):
        PEFTValidator(BASE_MODEL).validate(artifact)


def test_a_file_the_provenance_covers_must_exist(tmp_path: Path) -> None:
    """A shard listed at export and missing on disk is a truncated artifact."""
    artifact = adapter(tmp_path)
    root = Path(artifact.local_path)
    document = provenance(root)
    document["files"]["adapter_model-00002-of-00002.safetensors"] = "0" * 64
    (root / ADAPTER_PROVENANCE).write_text(json.dumps(document), encoding="utf-8")
    with fail_on("the adapter is incomplete"):
        PEFTValidator(BASE_MODEL).validate(artifact)


def test_provenance_disagreeing_with_the_config_is_refused(tmp_path: Path) -> None:
    """The exported tensors match one rank; the artifact must not claim both."""
    artifact = adapter(tmp_path)
    root = Path(artifact.local_path)
    document = provenance(root)
    document["peft"]["r"] = 8
    (root / ADAPTER_PROVENANCE).write_text(json.dumps(document), encoding="utf-8")
    with fail_on("match only one of them"):
        PEFTValidator(BASE_MODEL).validate(artifact)


def test_provenance_naming_another_base_model_is_refused(tmp_path: Path) -> None:
    artifact = adapter(tmp_path)
    root = Path(artifact.local_path)
    document = provenance(root, base_model={"name_or_path": "meta-llama/Llama-3-8B"})
    assert document["base_model"]["name_or_path"] != BASE_MODEL
    with fail_on("disagree about what it was fit to"):
        PEFTValidator(BASE_MODEL).validate(artifact)


def test_target_module_order_is_not_a_disagreement(tmp_path: Path) -> None:
    """PEFT stores target_modules as a set, so its written order is arbitrary."""
    artifact = adapter(tmp_path)
    root = Path(artifact.local_path)
    document = provenance(root)
    document["peft"]["target_modules"] = ["v_proj", "q_proj"]
    document["peft"]["lora_alpha"] = 16.0
    (root / ADAPTER_PROVENANCE).write_text(json.dumps(document), encoding="utf-8")
    PEFTValidator(BASE_MODEL).validate(artifact)


def test_provenance_without_checksums_audits_nothing(tmp_path: Path) -> None:
    artifact = adapter(tmp_path)
    provenance(Path(artifact.local_path), files={})
    with fail_on("audits nothing"):
        PEFTValidator(BASE_MODEL).validate(artifact)
