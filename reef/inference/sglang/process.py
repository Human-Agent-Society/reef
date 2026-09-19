"""Native SGLang process entrypoints and node-local address allocation."""

from __future__ import annotations

import multiprocessing
import os
import socket
import time
from contextlib import ExitStack, suppress
from typing import Any

import requests


def node_address_and_port(start_port: int = 15000, consecutive: int = 1) -> tuple[str, int]:
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
    raise RuntimeError("no free SGLang port range")


def local_gpu_id(physical_gpu: int) -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return physical_gpu
    devices = [int(value.strip()) for value in visible.split(",") if value.strip()]
    if physical_gpu in devices:
        return devices.index(physical_gpu)
    if 0 <= physical_gpu < len(devices):
        return physical_gpu
    raise ValueError(f"GPU {physical_gpu} is outside CUDA_VISIBLE_DEVICES")


def _run_engine(options: dict[str, Any]) -> None:
    # The inference allocator must not inherit training's expandable segments.
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    os.environ.pop("PYTORCH_ALLOC_CONF", None)
    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.server_args import ServerArgs

    launch_server(ServerArgs(**options))


def launch_engine(options: dict[str, Any]) -> Any:
    if options.get("encoder_only"):
        from sglang.srt.disaggregation.encode_server import launch_server_process
        from sglang.srt.server_args import ServerArgs

        return launch_server_process(ServerArgs(**options), start_method="spawn", wait_for_server=True)
    process = multiprocessing.get_context("spawn").Process(target=_run_engine, args=(options,))
    process.start()
    return process


def _run_router(options: dict[str, Any]) -> None:
    from sglang_router.launch_router import RouterArgs, launch_router

    launch_router(RouterArgs(**options))


def launch_router(options: dict[str, Any]) -> Any:
    process = multiprocessing.get_context("spawn").Process(target=_run_router, args=(options,))
    process.start()
    return process


def wait_ready(url: str, process: Any, timeout: float, *, path: str = "/health_generate") -> None:
    deadline = time.monotonic() + timeout
    while process.is_alive() and time.monotonic() < deadline:
        try:
            response = requests.get(url + path, timeout=5)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise RuntimeError(f"SGLang process did not become ready at {url}")
