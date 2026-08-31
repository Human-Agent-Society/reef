"""Provider-neutral HTTP service for isolated Guidance-TTT task verifiers."""

from __future__ import annotations

import argparse
import json
import os
import platform
import threading
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .edgebench_adapter import evaluate_vliw_solution
from .trimul_official_runner import run_official_trimul_evaluation


@dataclass(frozen=True)
class VerifierServiceConfig:
    task_id: str
    token: str
    concurrency: int = 1
    max_body_bytes: int = 4 * 1024 * 1024
    max_runner_timeout_s: int = 1_200
    evaluator_dir: Path | None = None
    evaluation_gpu_label: str = ""
    vliw_config: dict[str, Any] = field(default_factory=dict)


def evaluate_request(payload: dict[str, Any], config: VerifierServiceConfig) -> dict[str, Any]:
    solution = payload.get("solution")
    if not isinstance(solution, str) or not solution.strip():
        raise ValueError("request must contain a non-empty string field 'solution'")
    requested_timeout = int(payload.get("runner_timeout_s", config.max_runner_timeout_s))
    timeout_s = max(1, min(requested_timeout, config.max_runner_timeout_s))

    if config.task_id == "vliw_kernel_optimization":
        verifier_config = {"provider": "docker", **config.vliw_config}
        result = evaluate_vliw_solution(solution, timeout_s=timeout_s, config=verifier_config)
        return {
            "provider": "reef_http_verifier",
            "report": {
                "all_correct": result.valid,
                "score_cycles": result.cycles,
                "error": "" if result.valid else result.message,
                "results": result.artifacts.get("results", []),
                "best_cycles": result.artifacts.get("best_cycles"),
                "passed_thresholds": result.artifacts.get("passed_thresholds", []),
            },
            "judge_image": result.artifacts.get("judge_image"),
            "runner_returncode": result.artifacts.get("runner_returncode"),
            "stdout": result.artifacts.get("stdout", ""),
            "stderr": result.artifacts.get("stderr", ""),
            "elapsed_s": result.artifacts.get("elapsed_s"),
        }
    if config.task_id == "trimul":
        if config.evaluator_dir is None:
            raise ValueError("TriMul service requires evaluator_dir")
        result = run_official_trimul_evaluation(
            solution,
            evaluator_dir=config.evaluator_dir,
            timeout_s=timeout_s,
            subprocess_env=_candidate_subprocess_env(),
        )
        result["provider"] = "reef_http_verifier"
        result["evaluation_gpu"] = config.evaluation_gpu_label
        result["runtime_versions"] = {
            "python": platform.python_version(),
            "platform": platform.platform(),
        }
        return result
    raise ValueError(f"unsupported verifier service task: {config.task_id!r}")


def _candidate_subprocess_env() -> dict[str, str]:
    """Pass only compiler/GPU runtime state, never service credentials, to candidate code."""
    allowed = {
        "CC",
        "CFLAGS",
        "PATH",
        "CXX",
        "CXXFLAGS",
        "CUDAHOSTCXX",
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
        "CPATH",
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES",
        "NVIDIA_REQUIRE_CUDA",
        "CUDA_CACHE_MAXSIZE",
        "CUDA_CACHE_PATH",
        "TORCH_CUDA_ARCH_LIST",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "PYTHONPATH",
        "OMP_NUM_THREADS",
        "TMPDIR",
        "TMP",
        "TEMP",
        "LANG",
        "LC_ALL",
        "XDG_CACHE_HOME",
    }
    environment = {key: value for key, value in os.environ.items() if key in allowed}
    environment.update({"PYTHONUNBUFFERED": "1", "PYTHONNOUSERSITE": "1"})
    return environment


class _VerifierServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], config: VerifierServiceConfig):
        super().__init__(address, _VerifierHandler)
        self.config = config
        self.slots = threading.BoundedSemaphore(config.concurrency)


class _VerifierHandler(BaseHTTPRequestHandler):
    server: _VerifierServer

    def do_POST(self) -> None:
        if self.path != "/evaluate":
            self._json_response(HTTPStatus.NOT_FOUND, {"error": "use POST /evaluate"})
            return
        if self.server.config.token:
            expected = f"Bearer {self.server.config.token}"
            if self.headers.get("Authorization") != expected:
                self._json_response(HTTPStatus.UNAUTHORIZED, {"error": "invalid bearer token"})
                return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._json_response(HTTPStatus.BAD_REQUEST, {"error": "invalid Content-Length"})
            return
        if content_length < 1 or content_length > self.server.config.max_body_bytes:
            self._json_response(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "request body size is invalid"})
            return
        try:
            payload = json.loads(self.rfile.read(content_length))
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            with self.server.slots:
                result = evaluate_request(payload, self.server.config)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            self._json_response(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        except Exception as exc:
            self._json_response(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": f"{type(exc).__name__}: {exc}"},
            )
            return
        self._json_response(HTTPStatus.OK, result)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[guidance-verifier] {self.address_string()} {format % args}", flush=True)

    def _json_response(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, sort_keys=True).encode()
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("vliw_kernel_optimization", "trimul"), required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--token-env", default="GUIDANCE_TTT_VERIFIER_TOKEN")
    parser.add_argument("--allow-unauthenticated", action="store_true")
    parser.add_argument("--max-runner-timeout-s", type=int, default=1_200)
    parser.add_argument("--evaluator-dir", type=Path, default=None)
    parser.add_argument("--evaluation-gpu-label", default="")
    parser.add_argument("--vliw-judge-image", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.concurrency < 1 or args.max_runner_timeout_s < 1:
        raise ValueError("concurrency and max runner timeout must be positive")
    token = os.environ.get(args.token_env, "")
    if not token and not args.allow_unauthenticated:
        raise RuntimeError(
            f"verifier bearer token is missing from {args.token_env}; "
            "set it or pass --allow-unauthenticated on a trusted private network"
        )
    vliw_config = {}
    if args.vliw_judge_image:
        vliw_config["judge_image"] = args.vliw_judge_image
    config = VerifierServiceConfig(
        task_id=args.task,
        token=token,
        concurrency=args.concurrency,
        max_runner_timeout_s=args.max_runner_timeout_s,
        evaluator_dir=args.evaluator_dir,
        evaluation_gpu_label=args.evaluation_gpu_label,
        vliw_config=vliw_config,
    )
    server = _VerifierServer((args.host, args.port), config)
    print(f"[guidance-verifier] serving {args.task} on http://{args.host}:{args.port}/evaluate", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
