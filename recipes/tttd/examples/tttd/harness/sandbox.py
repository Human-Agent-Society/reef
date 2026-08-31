"""Isolated subprocess execution for generated discovery programs.

Generic sandbox machinery: write a Python source to a temp directory, import
it in a child process, invoke an entrypoint with pickled kwargs, and return
the result. Includes CPU affinity leasing and memory limits.

Used by :class:`~harness.scorer.ProgramScorer` for direct host-side scoring,
and by the SAO benchmark's scripted executor for fast unit tests.
"""

from __future__ import annotations

import contextlib
import os
import pickle
import signal
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProgramExecution:
    result: Any
    stdout: str
    stderr: str


class ProgramExecutionError(RuntimeError):
    """A generated program failed, timed out, or returned no result."""


def execute_program(
    source: str,
    *,
    entrypoint: str = "run",
    timeout_s: float = 1_100,
    max_cpus: int = 1,
    max_memory_bytes: int = 16 * 2**30,
    work_dir: str | Path | None = None,
    entrypoint_kwargs: dict[str, Any] | None = None,
) -> ProgramExecution:
    """Import ``source`` in a child and invoke its entrypoint (forwarding entrypoint_kwargs when supplied)."""
    if not source.strip():
        raise ValueError("program source must be non-empty")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    if max_cpus < 1:
        raise ValueError("max_cpus must be positive")
    if max_memory_bytes < 1:
        raise ValueError("max_memory_bytes must be positive")
    if not entrypoint.isidentifier():
        raise ValueError("entrypoint must be a Python identifier")

    parent = None if work_dir is None else Path(work_dir)
    if parent is not None:
        parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="reef-tttd-program-", dir=parent) as temporary:
        root = Path(temporary)
        program_path = root / "program.py"
        result_path = root / "result.pkl"
        runner_path = root / "runner.py"
        program_path.write_text(source, encoding="utf-8")

        threads = str(max_cpus)
        environment = {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "HOME": str(root),
            "TMPDIR": str(root),
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": threads,
            "MKL_NUM_THREADS": threads,
            "OPENBLAS_NUM_THREADS": threads,
            "NUMEXPR_NUM_THREADS": threads,
            "VECLIB_MAXIMUM_THREADS": threads,
            "BLIS_NUM_THREADS": threads,
        }
        if "LD_LIBRARY_PATH" in os.environ:
            environment["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]
        with _lease_cpus(max_cpus) as cpu_affinity:
            runner_path.write_text(
                _runner_source(
                    program_path,
                    result_path,
                    entrypoint,
                    cpu_affinity,
                    entrypoint_kwargs or {},
                    max_memory_bytes,
                ),
                encoding="utf-8",
            )
            process = subprocess.Popen(
                [sys.executable, str(runner_path)],
                cwd=root,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            try:
                stdout_bytes, stderr_bytes = process.communicate(timeout=timeout_s)
            except subprocess.TimeoutExpired as exc:
                _terminate_process_group(process)
                raise ProgramExecutionError(f"program timed out after {timeout_s:g} seconds") from exc

        stdout = stdout_bytes.decode(errors="replace")
        stderr = stderr_bytes.decode(errors="replace")
        if process.returncode != 0:
            detail = stderr[-2_000:].strip()
            raise ProgramExecutionError(
                f"program exited with code {process.returncode}" + (f": {detail}" if detail else "")
            )
        if not result_path.is_file():
            raise ProgramExecutionError("program returned no result")
        try:
            envelope = pickle.loads(result_path.read_bytes())
        except Exception as exc:
            raise ProgramExecutionError(f"cannot read program result: {exc}") from exc
        if not isinstance(envelope, dict) or set(envelope) != {"ok", "value"}:
            raise ProgramExecutionError("program result envelope is invalid")
        if not envelope["ok"]:
            raise ProgramExecutionError(f"program execution failed: {envelope['value']}")
        return ProgramExecution(envelope["value"], stdout, stderr)


def _runner_source(
    program_path: Path,
    result_path: Path,
    entrypoint: str,
    cpu_affinity: tuple[int, ...],
    entrypoint_kwargs: dict[str, Any],
    max_memory_bytes: int,
) -> str:
    try:
        serialized_kwargs = pickle.dumps(entrypoint_kwargs)
    except Exception as exc:
        raise ValueError(f"entrypoint_kwargs must be pickle-serializable: {exc}") from exc
    return f"""\
import importlib.util
import os
import pickle
import traceback

try:
    if {cpu_affinity!r} and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {set(cpu_affinity)!r})
except Exception:
    pass

try:
    import resource
    resource.setrlimit(resource.RLIMIT_AS, ({max_memory_bytes!r}, {max_memory_bytes!r}))
except Exception:
    pass

try:
    spec = importlib.util.spec_from_file_location("reef_tttd_generated_program", {str(program_path)!r})
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kwargs = pickle.loads({serialized_kwargs!r})
    value = getattr(module, {entrypoint!r})(**kwargs)
    envelope = {{"ok": True, "value": value}}
except BaseException:
    envelope = {{"ok": False, "value": traceback.format_exc()}}

with open({str(result_path)!r}, "wb") as result_file:
    pickle.dump(envelope, result_file)
"""


_CPU_CONDITION = threading.Condition()
_AVAILABLE_CPUS: set[int] | None = None


@contextlib.contextmanager
def _lease_cpus(max_cpus: int):
    global _AVAILABLE_CPUS
    if not hasattr(os, "sched_getaffinity"):
        yield ()
        return

    allowed = set(os.sched_getaffinity(0))
    requested = min(max_cpus, len(allowed))
    with _CPU_CONDITION:
        if _AVAILABLE_CPUS is None:
            _AVAILABLE_CPUS = set(allowed)
        while len(_AVAILABLE_CPUS) < requested:
            _CPU_CONDITION.wait()
        leased = tuple(sorted(_AVAILABLE_CPUS)[:requested])
        _AVAILABLE_CPUS.difference_update(leased)
    try:
        yield leased
    finally:
        with _CPU_CONDITION:
            assert _AVAILABLE_CPUS is not None
            _AVAILABLE_CPUS.update(leased)
            _CPU_CONDITION.notify_all()


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=1)
        return
    except subprocess.TimeoutExpired:
        pass
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait()
