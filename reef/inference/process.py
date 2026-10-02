"""Process, port and Ray actor helpers shared by the native inference integrations."""

from __future__ import annotations

import os
import socket
import time
from collections.abc import Sequence
from contextlib import ExitStack, suppress
from typing import Any

import requests


def node_address_and_port(start_port: int = 15000, consecutive: int = 1) -> tuple[str, int]:
    """This node's serving address and the first port of a free range of ``consecutive`` ports."""
    import ray

    address = os.environ.get("REEF_INFERENCE_HOST") or ray.util.get_node_ip_address()
    address = address.strip("[]")
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    for port in range(start_port, 65536 - consecutive):
        with suppress(OSError):
            with ExitStack() as stack:
                for offset in range(consecutive):
                    sock = stack.enter_context(socket.socket(family, socket.SOCK_STREAM))
                    sock.bind((address, port + offset))
            return (f"[{address}]" if family == socket.AF_INET6 else address), port
    raise RuntimeError("no free inference port range")


def local_gpu_id(physical_gpu: int) -> int:
    """The index of ``physical_gpu`` within this process's visible devices."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return physical_gpu
    devices = [int(value.strip()) for value in visible.split(",") if value.strip()]
    if physical_gpu in devices:
        return devices.index(physical_gpu)
    if 0 <= physical_gpu < len(devices):
        return physical_gpu
    raise ValueError(f"GPU {physical_gpu} is outside CUDA_VISIBLE_DEVICES")


def wait_ready(url: str, process: Any, timeout: float, *, path: str = "/health_generate") -> None:
    """Poll ``url + path`` until it answers 200, the process dies, or ``timeout`` seconds pass."""
    deadline = time.monotonic() + timeout
    while process.is_alive() and time.monotonic() < deadline:
        try:
            response = requests.get(url + path, timeout=5)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise RuntimeError(f"inference process did not become ready at {url}")


def retire_engines(engines: Sequence[Any], *, timeout: float = 30) -> None:
    """Ask each engine actor to shut down, then kill it; failures never block retirement."""
    import ray

    pending = []
    for engine in engines:
        with suppress(Exception):
            pending.append(engine.shutdown.remote())
    if pending:
        with suppress(Exception):
            ray.get(pending, timeout=timeout)
    for engine in engines:
        with suppress(Exception):
            ray.kill(engine, no_restart=True)


class RayHealthProbe:
    """Keep one outstanding RPC; a busy actor is not presumed dead."""

    def __init__(self) -> None:
        self.pending: Any = None

    def poll(self, actor: Any, method: str) -> None:
        import ray

        if self.pending is None:
            self.pending = getattr(actor, method).remote()
        ready, _ = ray.wait([self.pending], timeout=0)
        if not ready:
            return
        pending, self.pending = self.pending, None
        result = ray.get(pending)
        if isinstance(result, dict) and result.get("ok") is False and result.get("recoverable") is not True:
            raise RuntimeError(f"model component failed its health check: {result!r}")


__all__ = ["RayHealthProbe", "local_gpu_id", "node_address_and_port", "retire_engines", "wait_ready"]
