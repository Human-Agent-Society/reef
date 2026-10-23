"""Slime's THD context-parallel index map with context parallel size 1.

``docker/Dockerfile.reef`` applies ``docker/patch/slime_thd_cp1_indices.patch`` to Slime's
``slime_plugins/models/qwen3_5_vl_utils.py`` (THUDM/slime#2376). These tests apply the patch to a copy of the
installed file, or read the installed file when the image build already applied it. They skip when Slime is
missing.

Remove these tests with the patch when the Slime pin has a fix.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.unit

PATCH = Path(__file__).resolve().parents[2] / "docker" / "patch" / "slime_thd_cp1_indices.patch"


@pytest.fixture
def get_packed_cp_local_indices(tmp_path):
    if importlib.util.find_spec("slime_plugins") is None:
        pytest.skip("Slime is not installed")
    installed = Path(importlib.util.find_spec("slime_plugins.models.qwen3_5_vl_utils").origin)
    patched = tmp_path / "qwen3_5_vl_utils.py"
    shutil.copy(installed, patched)
    if "if cp_size == 1:" not in patched.read_text():
        subprocess.run(["git", "apply", "-p3", str(PATCH)], cwd=tmp_path, check=True)
    spec = importlib.util.spec_from_file_location("patched_qwen3_5_vl_utils", patched)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_packed_cp_local_indices


def test_thd_cp_indices_are_identity_without_cp(get_packed_cp_local_indices):
    indices = get_packed_cp_local_indices([0, 3, 4, 7], cp_size=1, cp_rank=0, device=torch.device("cpu"))
    assert indices.tolist() == list(range(7))


def test_thd_cp_indices_with_cp_are_unchanged(get_packed_cp_local_indices):
    indices = get_packed_cp_local_indices([0, 8, 12], cp_size=2, cp_rank=1, device=torch.device("cpu"))
    assert indices.tolist() == [2, 3, 4, 5, 9, 10]
    with pytest.raises(ValueError, match="must be divisible by 2 \\* CP size 2"):
        get_packed_cp_local_indices([0, 6], cp_size=2, cp_rank=0, device=torch.device("cpu"))
