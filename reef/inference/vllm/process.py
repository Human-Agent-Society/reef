"""The native vLLM server process owned by one engine actor."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from collections.abc import Mapping, Sequence


class EngineProcess:
    """One ``vllm serve`` process group: the API server and the engine cores it spawns."""

    def __init__(self, process: subprocess.Popen[bytes]) -> None:
        self._process = process

    @property
    def pid(self) -> int:
        return self._process.pid

    def is_alive(self) -> bool:
        return self._process.poll() is None

    def shutdown(self, timeout: float = 10) -> None:
        """Terminate the whole process group, escalating to SIGKILL after ``timeout`` seconds."""
        if not self.is_alive():
            return
        os.killpg(self._process.pid, signal.SIGTERM)
        try:
            self._process.wait(timeout)
        except subprocess.TimeoutExpired:
            os.killpg(self._process.pid, signal.SIGKILL)
            self._process.wait(timeout)


def launch_server(arguments: Sequence[str], env: Mapping[str, str]) -> EngineProcess:
    """Start vLLM's OpenAI-compatible server with the control routes Reef drives enabled."""
    environment = {**os.environ, **env, "VLLM_SERVER_DEV_MODE": "1"}
    # The inference allocator must not inherit training's expandable segments.
    environment.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    environment.pop("PYTORCH_ALLOC_CONF", None)
    process = subprocess.Popen(
        [sys.executable, "-m", "vllm.entrypoints.openai.api_server", *arguments],
        env=environment,
        start_new_session=True,
    )
    return EngineProcess(process)
