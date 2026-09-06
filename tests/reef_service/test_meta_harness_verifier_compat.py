import asyncio
import hashlib
import json
from types import SimpleNamespace

import pytest

from recipes.meta_harness.examples.terminal_bench import verifier_compat as compat


@pytest.fixture
def source(monkeypatch):
    original = ("def cleanup():\n" + compat.BEFORE + "\nassert original_correctness\n").encode()
    patched = original.replace(compat.BEFORE.encode(), compat.AFTER.encode())
    monkeypatch.setattr(compat, "ORIGINAL_SHA256", hashlib.sha256(original).hexdigest())
    monkeypatch.setattr(compat, "PATCHED_SHA256", hashlib.sha256(patched).hexdigest())
    return original, patched


def test_patch_refuses_unreviewed_or_already_patched_verifiers(source):
    original, patched = source
    assert compat.patch_tests(original) == patched
    for unsafe in (original + b"# changed\n", patched):
        with pytest.raises(ValueError, match="exact reviewed"):
            compat.patch_tests(unsafe)


@pytest.mark.parametrize("fail", [False, True])
def test_staging_restores_verifier_and_preserves_task_even_on_failure(tmp_path, monkeypatch, source, fail):
    from harbor.verifier.verifier import Verifier

    original, patched = source
    task_dir = tmp_path / compat.TASK
    tests = task_dir / "tests"
    tests.mkdir(parents=True)
    (tests / "test_outputs.py").write_bytes(original)
    (tests / "test.sh").write_text("original test command")
    visited = []

    async def original_verify(self):
        dirs, folder, script = self._resolve_tests()
        visited.append(folder)
        assert dirs == [folder] and folder != tests
        assert (folder / "test_outputs.py").read_bytes() == patched
        assert script.read_text() == "original test command"
        if fail:
            raise TimeoutError("original timeout")
        return "original verifier result"

    monkeypatch.setattr(Verifier, "verify", original_verify)
    compat.install(compat.PROTOCOL)
    installed = Verifier.verify
    compat.install(compat.PROTOCOL)
    assert Verifier.verify is installed

    def resolve():
        return [tests], tests, tests / "test.sh"

    verifier = SimpleNamespace(
        task=SimpleNamespace(paths=SimpleNamespace(task_dir=task_dir)),
        trial_paths=SimpleNamespace(trial_dir=tmp_path / "trial"),
        _skip_tests_upload=False,
        step_name=None,
        _resolve_tests=resolve,
    )
    if fail:
        with pytest.raises(TimeoutError):
            asyncio.run(Verifier.verify(verifier))
    else:
        assert asyncio.run(Verifier.verify(verifier)) == "original verifier result"
    assert verifier._resolve_tests is resolve
    assert (tests / "test_outputs.py").read_bytes() == original
    assert not visited[0].exists()
    receipt = json.loads((tmp_path / "trial/verifier-compatibility.json").read_text())
    assert receipt["protocol"] == compat.PROTOCOL and receipt["assertions_unchanged"]


def test_other_tasks_use_unmodified_verifier(monkeypatch, tmp_path):
    from harbor.verifier.verifier import Verifier

    async def original_verify(self):
        return "unmodified"

    monkeypatch.setattr(Verifier, "verify", original_verify)
    compat.install(compat.PROTOCOL)
    verifier = SimpleNamespace(task=SimpleNamespace(paths=SimpleNamespace(task_dir=tmp_path / "another-task")))
    assert asyncio.run(Verifier.verify(verifier)) == "unmodified"
    with pytest.raises(ValueError, match="unknown"):
        compat.install("unreviewed")
