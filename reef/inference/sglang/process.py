"""Native SGLang process entrypoints; address and readiness helpers come from :mod:`reef.inference.process`."""

from __future__ import annotations

import multiprocessing
import os
from typing import Any

from reef.inference.process import local_gpu_id, node_address_and_port, wait_ready


def _run_engine(options: dict[str, Any]) -> None:
    # The inference allocator must not inherit training's expandable segments.
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    os.environ.pop("PYTORCH_ALLOC_CONF", None)
    from sglang.srt.server_args import ServerArgs

    if options.get("encoder_only"):
        # SGLang's own launcher starts an encoder-only server through this module.
        from sglang.srt.disaggregation.encode_server import launch_server
    else:
        from sglang.srt.entrypoints.http_server import launch_server
    launch_server(ServerArgs(**options))


def launch_engine(options: dict[str, Any]) -> Any:
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


__all__ = ["launch_engine", "launch_router", "local_gpu_id", "node_address_and_port", "wait_ready"]
