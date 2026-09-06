"""Explicit, pinned TB2 verifier compatibility for the matched E2B protocol.

This is an adapted benchmark protocol. It changes only Gloo destruction after
the distributed assertions; it does not change task instructions or rewards.
The original downloaded task stays untouched. No oracle is packaged here.
"""

import hashlib
import json
import shutil
from functools import wraps
from pathlib import Path
from tempfile import TemporaryDirectory

PROTOCOL = "tb2-torch-gloo-cleanup-v1"
TASK = "torch-tensor-parallelism"
ORIGINAL_SHA256 = "4ae4285e15fba1fcf13d7b7c0638ac8c2e2b0797c937abab1fb6b136fd508290"
PATCHED_SHA256 = "bd86cde250dfde363744416f1c26f7977692c7fbfd259286e9994164710739b1"
BEFORE = "    dist.barrier()\n    dist.destroy_process_group()\n"
AFTER = (
    "    dist.barrier()\n"
    "    # Keep the Gloo backend alive until its Python no-GIL holder is released.\n"
    '    backend = dist.distributed_c10d._get_default_group()._get_backend(torch.device("cpu"))\n'
    "    dist.destroy_process_group()\n"
    "    del backend\n"
)


def patch_tests(source: bytes) -> bytes:
    if hashlib.sha256(source).hexdigest() != ORIGINAL_SHA256:
        raise ValueError("Gloo compatibility requires the exact reviewed pinned verifier")
    text = source.decode()
    if text.count(BEFORE) != 1:
        raise ValueError("Gloo cleanup is not unambiguous")
    patched = text.replace(BEFORE, AFTER).encode()
    if hashlib.sha256(patched).hexdigest() != PATCHED_SHA256:
        raise ValueError("Gloo compatibility patch differs from its reviewed digest")
    return patched


def install(protocol):
    """Install once inside a disposable runner, before any candidate executes."""
    if protocol != PROTOCOL:
        raise ValueError("unknown verifier compatibility protocol")
    from harbor.verifier.verifier import Verifier

    if getattr(Verifier.verify, "_reef_verifier_protocol", None) == PROTOCOL:
        return
    original_verify = Verifier.verify

    @wraps(original_verify)
    async def verify(self):
        if self.task.paths.task_dir.name != TASK:
            return await original_verify(self)
        source_dirs, source_dir, script = self._resolve_tests()
        if self._skip_tests_upload or self.step_name is not None or source_dirs != [source_dir]:
            raise ValueError("Gloo compatibility requires the original single-step verifier")
        paths = list(source_dir.rglob("*"))
        if (
            any(path.is_symlink() for path in paths)
            or sum(path.stat().st_size for path in paths if path.is_file()) > 1024 * 1024
        ):
            raise ValueError("unexpected verifier source tree")
        patched = patch_tests((source_dir / "test_outputs.py").read_bytes())
        with TemporaryDirectory(prefix="reef-verifier-") as temporary:
            staged = Path(temporary) / "tests"
            shutil.copytree(source_dir, staged)
            (staged / "test_outputs.py").write_bytes(patched)
            receipt = {
                "protocol": PROTOCOL,
                "task": TASK,
                "original_tests_sha256": ORIGINAL_SHA256,
                "patched_tests_sha256": PATCHED_SHA256,
                "assertions_unchanged": True,
            }
            self.trial_paths.trial_dir.mkdir(parents=True, exist_ok=True)
            (self.trial_paths.trial_dir / "verifier-compatibility.json").write_text(
                json.dumps(receipt, indent=2) + "\n"
            )
            # Only source resolution changes. Harbor still uploads, executes,
            # downloads, parses rewards and applies the original time budget.
            had_override = "_resolve_tests" in self.__dict__
            prior = self.__dict__.get("_resolve_tests")
            self._resolve_tests = lambda: ([staged], staged, staged / script.relative_to(source_dir))
            try:
                return await original_verify(self)
            finally:
                if had_override:
                    self._resolve_tests = prior
                else:
                    del self._resolve_tests

    verify._reef_verifier_protocol = PROTOCOL
    Verifier.verify = verify
