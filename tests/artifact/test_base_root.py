"""The base root: an artifact that holds ``base.json`` alone and stands for the deployment's base model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reef.artifact import (
    BASE_FILE,
    Artifact,
    BaseRoot,
    DirectoryFiles,
    GitLFSRepositoryBackend,
    InMemoryRepositoryBackend,
    base_root_of,
    read_base_root,
)
from reef.artifact.base_root import BASE_FORMAT

LORA_INPUTS = {"model": "/models/qwen", "lora_rank": "8", "lora_alpha": "16", "lora_target_modules": "q_proj,v_proj"}


def base_root(directory: Path, inputs: dict[str, str]) -> Path:
    """A new directory that holds the base file of ``inputs`` alone."""
    directory.mkdir(parents=True)
    (directory / BASE_FILE).write_text(BaseRoot(inputs).to_text(), encoding="utf-8")
    return directory


@pytest.mark.unit
@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        ({"lora_rank": "8"}, "names its base model"),
        ({"model": ""}, "non-empty string"),
        ({"model": "m", "lora_rank": 8}, "non-empty string"),
        (["model"], "a mapping"),
    ],
)
def test_a_base_root_names_its_model_and_holds_only_text_inputs(inputs, message):
    with pytest.raises(ValueError, match=message):
        BaseRoot(inputs)


@pytest.mark.unit
def test_the_base_file_reads_back_to_an_equal_value_and_another_file_is_refused(tmp_path):
    base = BaseRoot(LORA_INPUTS)
    assert base.model == "/models/qwen"
    assert base.to_dict() == {"format": BASE_FORMAT, "inputs": LORA_INPUTS}
    directory = base_root(tmp_path / "root", LORA_INPUTS)
    assert read_base_root(directory) == base
    assert base_root_of(DirectoryFiles(directory)) == base
    # An artifact without the file is no base root, whatever else it holds.
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "config.json").write_text("{}")
    assert read_base_root(tmp_path / "weights") is None
    assert base_root_of(DirectoryFiles(tmp_path / "weights")) is None
    for text, message in (
        ("not json", "is no JSON object"),
        (json.dumps({"inputs": LORA_INPUTS}), "the object {format, inputs}"),
        (json.dumps({"format": "reef-base-root-2", "inputs": LORA_INPUTS}), "this Reef reads"),
        (json.dumps({"format": BASE_FORMAT, "inputs": {"lora_rank": "8"}}), "names its base model"),
    ):
        (directory / BASE_FILE).write_text(text)
        with pytest.raises(ValueError, match=message):
            read_base_root(directory)
        with pytest.raises(ValueError, match=message):
            base_root_of(DirectoryFiles(directory))


@pytest.mark.unit
def test_a_base_root_holds_no_other_file_but_the_repositorys_own(tmp_path):
    directory = base_root(tmp_path / "root", LORA_INPUTS)
    (directory / "reef-artifact.json").write_text("{}")
    assert base_root_of(DirectoryFiles(directory)) == BaseRoot(LORA_INPUTS)
    (directory / "adapter").mkdir()
    (directory / "adapter" / "adapter_model.safetensors").write_bytes(b"a")
    with pytest.raises(ValueError, match=r"holds base\.json alone.*adapter/adapter_model\.safetensors"):
        base_root_of(DirectoryFiles(directory))


@pytest.mark.unit
def test_a_base_root_matches_only_the_default_start_of_equal_inputs():
    base = BaseRoot(LORA_INPUTS)
    base.check_inputs(dict(LORA_INPUTS))
    with pytest.raises(ValueError, match="lora_rank: '8', not '16'"):
        base.check_inputs({**LORA_INPUTS, "lora_rank": "16"})
    with pytest.raises(ValueError, match="model: '/models/qwen', not '/models/other'"):
        base.check_inputs({**LORA_INPUTS, "model": "/models/other"})
    # An input that only one side has differs too.
    with pytest.raises(ValueError, match="service: None, not 'http://spindle'"):
        base.check_inputs({**LORA_INPUTS, "service": "http://spindle"})


@pytest.mark.integration
@pytest.mark.parametrize("backend_kind", ["memory", "git_lfs"])
def test_a_repository_registers_a_base_root_and_reads_it_back(tmp_path, fake_git_lfs, backend_kind):
    if backend_kind == "memory":
        factory = InMemoryRepositoryBackend.factory(directory=tmp_path / "repository")
    else:
        factory = GitLFSRepositoryBackend.factory(
            tmp_path / "artifacts.git", work_dir=tmp_path / "work", cache_dir=tmp_path / "cache"
        )
    backend = factory("math")
    root = backend.register(Artifact.local(base_root(tmp_path / "root", LORA_INPUTS)), metadata={})
    assert root.parent_revision is None and backend.current() == root
    directory = backend.materialize(root).local_path
    assert directory is not None
    assert read_base_root(directory) == BaseRoot(LORA_INPUTS)
    # The files that the repository adds beside the base file leave it a base root.
    assert base_root_of(DirectoryFiles(directory)) == BaseRoot(LORA_INPUTS)
    assert (directory / BASE_FILE).read_text(encoding="utf-8") == BaseRoot(LORA_INPUTS).to_text()
