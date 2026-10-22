"""Run the remote inference component on a GPU host: ``python -m reef.inference.remote``."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from aiohttp import web

from reef.inference.remote.config import DEFAULT_CONTROL_TOKEN_ENV, DEFAULT_INFERENCE_TOKEN_ENV
from reef.inference.remote.server import RemoteInferenceServer


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m reef.inference.remote",
        description=(
            "Serve this host's inference engines to a Reef control plane. The control plane starts the engines "
            "named in its inference.remote section and publishes trained weights to them."
        ),
    )
    parser.add_argument(
        "--engine",
        default="sglang",
        help="the only engine backend the control plane may start; its inference.backend must match (default: sglang)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="bind port (default: 8000)")
    parser.add_argument(
        "--work-dir", type=Path, default=Path("reef-remote"), help="directory for uploaded weight files"
    )
    parser.add_argument(
        "--control-token-env",
        default=DEFAULT_CONTROL_TOKEN_ENV,
        help=f"environment variable with the control token (default: {DEFAULT_CONTROL_TOKEN_ENV})",
    )
    parser.add_argument(
        "--inference-token-env",
        default=DEFAULT_INFERENCE_TOKEN_ENV,
        help=f"environment variable with the inference token (default: {DEFAULT_INFERENCE_TOKEN_ENV})",
    )
    parser.add_argument("--ray-address", default=None, help="join this Ray cluster instead of starting one")
    parser.add_argument(
        "--ray-num-gpus", type=int, default=None, help="GPUs the started Ray session declares (default: detected)"
    )
    arguments = parser.parse_args(argv)
    tokens = []
    for variable in (arguments.control_token_env, arguments.inference_token_env):
        token = os.environ.get(variable, "").strip()
        if not token:
            parser.error(f"the environment variable {variable} must hold a token")
        tokens.append(token)
    if tokens[0] == tokens[1]:
        parser.error("the control and inference tokens must differ")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    import ray

    if arguments.ray_address:
        ray.init(address=arguments.ray_address, namespace="reef-remote-inference")
    else:
        ray.init(
            address="local",
            num_gpus=arguments.ray_num_gpus,
            include_dashboard=False,
            namespace="reef-remote-inference",
        )
    try:
        server = RemoteInferenceServer(
            engine=arguments.engine,
            control_token=tokens[0],
            inference_token=tokens[1],
            work_dir=arguments.work_dir.resolve(),
        )
        web.run_app(server.application(), host=arguments.host, port=arguments.port, print=None)
    finally:
        ray.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
