"""Pin, prepare, and identify the runtime used by both comparison arms."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import os
import platform
import tempfile
from pathlib import Path

PINS = {"harbor": "0.20.0", "litellm": "1.99.0", "openai": "2.54.0", "e2b": "2.46.4"}
_ORIGINAL_TMUX = "38642b794c335880"
_PATCHED_TMUX = "59b2774b5f3a77d571cf3fa223f4e3eb8377c3ef9b354c8a271f4c3bc1e40661"


def check_capacity(concurrency: int, *, benchmark_limit: int = 32) -> None:
    from e2b import Sandbox

    from .reap_sandboxes import _environment, _sandboxes
    from .tasks import read_tasks

    if type(benchmark_limit) is not int or benchmark_limit not in (32, 64):
        raise ValueError("benchmark sandbox allocation must be 32 or 64")
    alive = _sandboxes(Sandbox)
    names = {task.split("/")[-1] for task in read_tasks(Path(__file__).with_name("tasks-89.txt"))}
    ours = sum(
        _environment(sandbox) in names
        or (getattr(sandbox, "metadata", None) or {}).get("reef_role")
        in ("meta-harness-runner", "meta-harness-runtime-build")
        for sandbox in alive
    )
    if ours + concurrency > benchmark_limit or len(alive) + concurrency > 100:
        raise RuntimeError(
            f"insufficient sandbox capacity: {ours}/{benchmark_limit} benchmark, {len(alive)}/100 account; "
            f"the next wave needs {concurrency}. Existing sandboxes were left untouched."
        )


def tmux_source() -> Path:
    spec = importlib.util.find_spec("harbor")
    if spec is None or spec.origin is None:
        raise RuntimeError("install the pinned Terminal-Bench runtime first")
    return Path(spec.origin).parent / "agents/terminus_2/tmux_session.py"


def prepare_harbor(path: Path | None = None) -> None:
    """Apply the bounded compatibility fix to this environment, once.

    Replace the inode: uv can hard-link package files into multiple venvs;
    modifying a file in place would also change unrelated running experiments.
    """
    path = path or tmux_source()
    text = path.read_text()
    start = text.index("    async def _attempt_tmux_installation(")
    end = text.index("    async def _install_recording_tools(", start)
    method = text[start:end]
    if "except (asyncio.TimeoutError, E2BTimeoutException):" in method:
        if hashlib.sha256(text.encode()).hexdigest() != _PATCHED_TMUX:
            raise RuntimeError("prepared Harbor source changed; review it before running")
        return
    if not hashlib.sha256(text.encode()).hexdigest().startswith(_ORIGINAL_TMUX):
        raise RuntimeError("unrecognized Harbor tool-installation source; review the compatibility patch")
    method = method.replace(
        "        try:\n",
        "        from e2b.exceptions import TimeoutException as E2BTimeoutException\n\n        try:\n",
        1,
    )
    method = method.replace("except asyncio.TimeoutError:", "except (asyncio.TimeoutError, E2BTimeoutException):", 1)
    updated = text[:start] + method + text[end:]
    fd, temporary = tempfile.mkstemp(prefix=".reef-tmux-", suffix=".py", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(updated)
        os.chmod(temporary, path.stat().st_mode)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def runtime_fingerprint():
    versions = {name: importlib.metadata.version(name) for name in PINS}
    if versions != PINS:
        raise RuntimeError(f"runtime versions differ from the comparison pins: {versions}; expected {PINS}")
    source = tmux_source()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != _PATCHED_TMUX:
        raise RuntimeError("prepare this runtime with python -m recipes.meta_harness.examples.terminal_bench.runtime")
    return {"versions": versions, "python": platform.python_version(), "harbor_tmux_sha256": digest}


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    prepare_harbor()
    print(runtime_fingerprint())


if __name__ == "__main__":
    main()
