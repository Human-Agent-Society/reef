#!/usr/bin/env python3
"""Evaluator for the Frontier-CS 2.0 Vector DB ANN task."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
import math
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib import request

import numpy as np


def _read_evaluation_config() -> dict[str, int]:
    config_path = Path(__file__).with_name("config.yaml")
    if not config_path.exists():
        return {}

    values: dict[str, int] = {}
    in_evaluation = False
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        if not raw_line.startswith((" ", "\t")):
            in_evaluation = line == "evaluation:"
            continue
        if not in_evaluation:
            continue
        stripped = line.strip()
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key in {"query_concurrency", "queries_per_worker"} and value:
            values[key] = int(value)
    return values


def _config_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


_EVALUATION_CONFIG = _read_evaluation_config()
CONFIG_CONCURRENCY = int(_EVALUATION_CONFIG.get("query_concurrency", 8))
CONFIG_QUERIES_PER_WORKER = int(_EVALUATION_CONFIG.get("queries_per_worker", 64))


DIM = 128
N_BASE = _config_int("FRONTIER_VECTOR_DB_N", 1000000)
CONCURRENCY = _config_int("FRONTIER_VECTOR_DB_CONCURRENCY", CONFIG_CONCURRENCY)
QUERIES_PER_WORKER = _config_int(
    "FRONTIER_VECTOR_DB_QUERIES_PER_WORKER", CONFIG_QUERIES_PER_WORKER
)
N_QUERIES = _config_int(
    "FRONTIER_VECTOR_DB_Q", CONCURRENCY * QUERIES_PER_WORKER
)
TOP_K = _config_int("FRONTIER_VECTOR_DB_TOP_K", 10)
SEED = _config_int("FRONTIER_VECTOR_DB_SEED", 20260528)
TARGET_RECALL = float(os.environ.get("FRONTIER_VECTOR_DB_TARGET_RECALL", "0.95"))
QUERY_NOISE = float(os.environ.get("FRONTIER_VECTOR_DB_QUERY_NOISE", "0.02"))
BUILD_TIMEOUT_SECONDS = _config_int("FRONTIER_VECTOR_DB_BUILD_TIMEOUT", 600)
LOAD_TIMEOUT_SECONDS = _config_int("FRONTIER_VECTOR_DB_LOAD_TIMEOUT", 900)
LOAD_PENALTY_WEIGHT = float(os.environ.get("FRONTIER_VECTOR_DB_LOAD_PENALTY", "0.01"))
BATCH_SIZE = _config_int("FRONTIER_VECTOR_DB_BATCH_SIZE", 1000)
WARMUP = _config_int("FRONTIER_VECTOR_DB_WARMUP", 32)
CACHE_DIR = Path(os.environ.get("FRONTIER_VECTOR_DB_CACHE", "/tmp/frontier_vector_db_ann"))

_BENCHMARK: "Benchmark | None" = None


@dataclass
class Benchmark:
    base_path: Path
    queries_path: Path
    truth: np.ndarray
    baseline_qps: float
    baseline_effective_qps: float
    baseline_seconds: float
    baseline_load_seconds: float


class QueryTooSlow(RuntimeError):
    def __init__(
        self,
        elapsed_seconds: float,
        optimistic_qps: float,
        optimistic_effective_qps: float,
        completed_queries: int,
    ):
        super().__init__(
            "query phase stopped early because even an optimistic final "
            "effective_qps cannot beat the baseline"
        )
        self.elapsed_seconds = elapsed_seconds
        self.optimistic_qps = optimistic_qps
        self.optimistic_effective_qps = optimistic_effective_qps
        self.completed_queries = completed_queries


class LoadTooSlow(RuntimeError):
    def __init__(
        self,
        load_seconds: float,
        optimistic_effective_qps: float,
        inserted: int,
    ):
        super().__init__(
            "load/index phase stopped early because even zero query time "
            "cannot beat the baseline effective_qps"
        )
        self.load_seconds = load_seconds
        self.optimistic_effective_qps = optimistic_effective_qps
        self.inserted = inserted


def prepare() -> dict:
    print(
        f"[vector-db-ann] preparing benchmark n_base={N_BASE} "
        f"n_queries={N_QUERIES} top_k={TOP_K} cache={CACHE_DIR}",
        flush=True,
    )
    benchmark = _ensure_benchmark()
    print(
        f"[vector-db-ann] benchmark ready baseline_qps={benchmark.baseline_qps:.6f} "
        f"baseline_effective_qps={benchmark.baseline_effective_qps:.6f} "
        f"baseline_seconds={benchmark.baseline_seconds:.6f} "
        f"baseline_load_seconds={benchmark.baseline_load_seconds:.6f}",
        flush=True,
    )
    return {
        "n_base": N_BASE,
        "n_queries": N_QUERIES,
        "top_k": TOP_K,
        "baseline_qps": benchmark.baseline_qps,
        "baseline_effective_qps": benchmark.baseline_effective_qps,
        "baseline_seconds": benchmark.baseline_seconds,
        "baseline_load_seconds": benchmark.baseline_load_seconds,
    }


def _write_vectors(path: Path, values: np.ndarray) -> None:
    values.astype(np.float32, copy=False).tofile(path)


def _load_vectors(path: Path, rows: int) -> np.memmap:
    return np.memmap(path, dtype=np.float32, mode="r", shape=(rows, DIM))


def _generate_data(base_path: Path, queries_path: Path) -> None:
    rng = np.random.default_rng(SEED)
    chunk = 50_000
    base = np.memmap(base_path, dtype=np.float32, mode="w+", shape=(N_BASE, DIM))
    for start in range(0, N_BASE, chunk):
        end = min(start + chunk, N_BASE)
        base[start:end] = rng.standard_normal((end - start, DIM), dtype=np.float32)
    base.flush()

    ids = rng.integers(0, N_BASE, size=N_QUERIES)
    selected = np.asarray(base[ids], dtype=np.float32)
    noise = rng.standard_normal((N_QUERIES, DIM), dtype=np.float32) * QUERY_NOISE
    _write_vectors(queries_path, selected + noise)


def _run_reference_server(port: int, base_path: str) -> None:
    import faiss

    _ = base_path
    index = faiss.IndexIDMap(faiss.IndexFlatL2(DIM))

    class ReferenceHandler(BaseHTTPRequestHandler):
        server_version = "FrontierVectorFaissReference/1.0"

        def _write_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._write_json(200, {"status": "ok"})
                return
            self._write_json(404, {"status": "error", "error": "not found"})

        def do_POST(self) -> None:
            if self.path == "/insert":
                self._write_json(200, {"status": "ok"})
                return
            if self.path == "/bulk_insert":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    payload = json.loads(self.rfile.read(length).decode("utf-8"))
                    items = payload.get("vectors", [])
                    inserted = len(items)
                    if inserted:
                        vectors = np.asarray(
                            [item["vector"] for item in items], dtype=np.float32
                        )
                        ids = np.asarray(
                            [int(item["id"]) for item in items], dtype=np.int64
                        )
                        if vectors.shape != (inserted, DIM):
                            raise ValueError("bulk_insert vector shape mismatch")
                        index.add_with_ids(vectors, ids)
                    self._write_json(200, {"status": "ok", "inserted": inserted})
                except Exception as exc:
                    self._write_json(400, {"status": "error", "error": str(exc)})
                return
            if self.path != "/search":
                self._write_json(404, {"status": "error", "error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                vector = np.asarray(payload["vector"], dtype=np.float32)
                top_k = int(payload.get("top_k", TOP_K))
                if vector.shape != (DIM,):
                    raise ValueError("query vector has wrong dimension")
                if top_k != TOP_K:
                    raise ValueError("unexpected top_k")
                distances, ids = index.search(vector.reshape(1, DIM), top_k)
                results = [
                    {"id": int(id_), "distance": float(distance)}
                    for id_, distance in zip(ids[0], distances[0])
                ]
                self._write_json(200, {"results": results})
            except Exception as exc:
                self._write_json(400, {"status": "error", "error": str(exc)})

        def log_message(self, fmt: str, *args: object) -> None:
            return

    ThreadingHTTPServer(("127.0.0.1", port), ReferenceHandler).serve_forever()


def _measure_reference_baseline(base_path: Path, queries: np.ndarray):
    port = _free_port()
    process = subprocess.Popen(
        [
            "python3",
            str(Path(__file__).resolve()),
            "--reference-server",
            str(port),
            str(base_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        _wait_for_server(port, time.time() + 120)
        base = _load_vectors(base_path, N_BASE)
        inserted, load_seconds = _bulk_insert(f"http://127.0.0.1:{port}", base)
        if inserted != N_BASE:
            raise RuntimeError(f"reference inserted {inserted}; expected {N_BASE}")
        results, latencies, baseline_seconds = _run_queries(
            f"http://127.0.0.1:{port}", queries
        )
        return results, latencies, baseline_seconds, load_seconds
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def _ensure_benchmark() -> Benchmark:
    global _BENCHMARK
    if _BENCHMARK is not None:
        return _BENCHMARK

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    base_path = CACHE_DIR / f"base_n{N_BASE}_d{DIM}_seed{SEED}.f32"
    queries_path = CACHE_DIR / f"queries_q{N_QUERIES}_d{DIM}_seed{SEED}.f32"
    truth_path = CACHE_DIR / (
        f"truth_n{N_BASE}_q{N_QUERIES}_k{TOP_K}_seed{SEED}.u32"
    )
    meta_path = CACHE_DIR / (
        f"baseline_faiss_http_bulk_load_v3_n{N_BASE}_q{N_QUERIES}_k{TOP_K}_seed{SEED}.json"
    )

    expected_base_bytes = N_BASE * DIM * 4
    expected_query_bytes = N_QUERIES * DIM * 4
    if (
        not base_path.exists()
        or base_path.stat().st_size != expected_base_bytes
        or not queries_path.exists()
        or queries_path.stat().st_size != expected_query_bytes
    ):
        print("[vector-db-ann] generating hidden vectors", flush=True)
        _generate_data(base_path, queries_path)
        truth_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)

    base = _load_vectors(base_path, N_BASE)
    queries = _load_vectors(queries_path, N_QUERIES)

    if truth_path.exists() and meta_path.exists():
        truth = np.fromfile(truth_path, dtype=np.uint32).reshape(N_QUERIES, TOP_K)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        baseline_seconds = float(meta["baseline_seconds"])
        baseline_qps = float(meta["baseline_qps"])
        baseline_load_seconds = float(meta["baseline_load_seconds"])
        baseline_effective_qps = float(meta["baseline_effective_qps"])
    else:
        print("[vector-db-ann] running Faiss HTTP exact baseline", flush=True)
        truth, _, baseline_seconds, baseline_load_seconds = _measure_reference_baseline(
            base_path, queries
        )
        truth.astype(np.uint32, copy=False).tofile(truth_path)
        baseline_qps = N_QUERIES / baseline_seconds
        baseline_effective_seconds = baseline_seconds + (
            LOAD_PENALTY_WEIGHT * baseline_load_seconds
        )
        baseline_effective_qps = N_QUERIES / max(baseline_effective_seconds, 1e-9)
        meta_path.write_text(
            json.dumps(
                {
                    "baseline_seconds": baseline_seconds,
                    "baseline_qps": baseline_qps,
                    "baseline_load_seconds": baseline_load_seconds,
                    "baseline_effective_qps": baseline_effective_qps,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    _BENCHMARK = Benchmark(
        base_path=base_path,
        queries_path=queries_path,
        truth=truth,
        baseline_qps=baseline_qps,
        baseline_effective_qps=baseline_effective_qps,
        baseline_seconds=baseline_seconds,
        baseline_load_seconds=baseline_load_seconds,
    )
    return _BENCHMARK


def _invalid(message: str, metrics: dict | None = None):
    return 0.0, 0.0, message, metrics or {}


def _copy_project(src: Path, dst: Path) -> None:
    ignore = shutil.ignore_patterns("target", ".git", ".frontier-cs")
    shutil.copytree(src, dst, ignore=ignore)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _post_json(url: str, payload: dict, timeout: float = 60.0) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _wait_for_server(port: int, deadline: float) -> None:
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                pass
            return
        except Exception as exc:
            last_error = exc
            time.sleep(0.25)
    raise RuntimeError(f"server did not become ready: {last_error}")


def _bulk_insert(
    base_url: str,
    base: np.ndarray,
    *,
    min_effective_qps: float | None = None,
) -> tuple[int, float]:
    inserted = 0
    start_time = time.perf_counter()
    deadline = start_time + LOAD_TIMEOUT_SECONDS
    early_stop_deadline: float | None = None
    if min_effective_qps is not None and LOAD_PENALTY_WEIGHT > 0.0:
        early_stop_deadline = start_time + (
            N_QUERIES / (min_effective_qps * LOAD_PENALTY_WEIGHT)
        )

    def check_load_budget() -> None:
        if early_stop_deadline is None:
            return
        load_seconds = max(time.perf_counter() - start_time, 1e-9)
        optimistic_effective_seconds = LOAD_PENALTY_WEIGHT * load_seconds
        optimistic_effective_qps = N_QUERIES / max(
            optimistic_effective_seconds, 1e-9
        )
        if optimistic_effective_qps <= min_effective_qps:
            raise LoadTooSlow(
                load_seconds,
                optimistic_effective_qps,
                inserted,
            )

    for start in range(0, N_BASE, BATCH_SIZE):
        check_load_budget()
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError(f"bulk_insert timed out after {LOAD_TIMEOUT_SECONDS}s")
        if early_stop_deadline is not None:
            remaining = min(remaining, early_stop_deadline - time.perf_counter())
            if remaining <= 0:
                check_load_budget()
                raise TimeoutError("bulk_insert early-stop budget expired")
        end = min(start + BATCH_SIZE, N_BASE)
        vectors = [
            {"id": int(i), "vector": base[i].astype(float).tolist()}
            for i in range(start, end)
        ]
        payload = {"vectors": vectors}
        try:
            response = _post_json(
                f"{base_url}/bulk_insert",
                payload,
                timeout=max(0.05, min(300.0, remaining)),
            )
        except Exception:
            check_load_budget()
            raise
        inserted += int(response.get("inserted", 0))
        check_load_budget()
    load_seconds = max(time.perf_counter() - start_time, 1e-9)
    return inserted, load_seconds


def _search_one(base_url: str, query_index: int, vector: np.ndarray) -> tuple[int, list[int], float]:
    start = time.perf_counter()
    response = _post_json(
        f"{base_url}/search",
        {"vector": vector.astype(float).tolist(), "top_k": TOP_K},
        timeout=120.0,
    )
    latency_ms = (time.perf_counter() - start) * 1000.0
    ids = [int(item.get("id", -1)) for item in response.get("results", [])[:TOP_K]]
    return query_index, ids, latency_ms


def _run_queries(
    base_url: str,
    queries: np.ndarray,
    *,
    load_seconds: float = 0.0,
    min_effective_qps: float | None = None,
) -> tuple[np.ndarray, list[float], float]:
    for i in range(min(WARMUP, N_QUERIES)):
        try:
            _search_one(base_url, i, queries[i])
        except Exception:
            pass

    results = np.zeros((N_QUERIES, TOP_K), dtype=np.uint32)
    latencies: list[float] = []
    start = time.perf_counter()
    pool = ThreadPoolExecutor(max_workers=CONCURRENCY)
    futures = [
        pool.submit(_search_one, base_url, i, queries[i])
        for i in range(N_QUERIES)
    ]
    should_wait = True
    try:
        for future in as_completed(futures):
            query_index, ids, latency_ms = future.result()
            if len(ids) != TOP_K:
                raise ValueError("search response did not contain top_k results")
            if len(set(ids)) != len(ids):
                raise ValueError("search response contains duplicate vector ids")
            if any(id_ < 0 or id_ >= N_BASE for id_ in ids):
                raise ValueError("search response contains an out-of-range vector id")
            results[query_index] = np.asarray(ids, dtype=np.uint32)
            latencies.append(latency_ms)

            if min_effective_qps is not None:
                elapsed = max(time.perf_counter() - start, 1e-9)
                optimistic_qps = N_QUERIES / elapsed
                optimistic_effective_seconds = elapsed + (
                    LOAD_PENALTY_WEIGHT * load_seconds
                )
                optimistic_effective_qps = N_QUERIES / max(
                    optimistic_effective_seconds, 1e-9
                )
                if optimistic_effective_qps <= min_effective_qps:
                    should_wait = False
                    raise QueryTooSlow(
                        elapsed,
                        optimistic_qps,
                        optimistic_effective_qps,
                        len(latencies),
                    )
    finally:
        pool.shutdown(wait=should_wait, cancel_futures=not should_wait)
    duration = max(time.perf_counter() - start, 1e-9)
    return results, latencies, duration


def _recall_at_k(results: np.ndarray, truth: np.ndarray) -> float:
    hits = 0
    for got, expected in zip(results, truth):
        hits += len(set(int(x) for x in got) & set(int(x) for x in expected))
    return hits / float(N_QUERIES * TOP_K)


def evaluate(solution_path: str):
    root = Path(solution_path)
    if not root.is_dir():
        return _invalid("submission path must be a Rust project directory")
    if not (root / "Cargo.toml").exists():
        return _invalid("Cargo.toml not found in submission directory")

    benchmark = _ensure_benchmark()

    with tempfile.TemporaryDirectory(prefix="frontier_vector_db_ann_") as tmp:
        workdir = Path(tmp) / "project"
        _copy_project(root, workdir)
        try:
            subprocess.run(
                ["cargo", "build", "--release", "--quiet"],
                cwd=workdir,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=BUILD_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return _invalid("cargo build timed out")
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", errors="replace")
            return _invalid(f"cargo build failed: {stderr[-800:]}")

        base = _load_vectors(benchmark.base_path, N_BASE)
        queries = _load_vectors(benchmark.queries_path, N_QUERIES)
        port = _free_port()
        base_url = f"http://127.0.0.1:{port}"
        process = subprocess.Popen(
            ["cargo", "run", "--release", "--quiet"],
            cwd=workdir,
            env={**os.environ, "PORT": str(port)},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        load_seconds = 0.0
        try:
            _wait_for_server(port, time.time() + 30)
            inserted, load_seconds = _bulk_insert(
                base_url,
                base,
                min_effective_qps=benchmark.baseline_effective_qps,
            )
            if inserted != N_BASE:
                return _invalid(f"bulk_insert inserted {inserted}; expected {N_BASE}")
            results, latencies, candidate_seconds = _run_queries(
                base_url,
                queries,
                load_seconds=load_seconds,
                min_effective_qps=benchmark.baseline_effective_qps,
            )
        except QueryTooSlow as exc:
            metrics = {
                "baseline_qps": benchmark.baseline_qps,
                "baseline_effective_qps": benchmark.baseline_effective_qps,
                "baseline_seconds": benchmark.baseline_seconds,
                "baseline_load_seconds": benchmark.baseline_load_seconds,
                "qps": 0.0,
                "qps_upper_bound": exc.optimistic_qps,
                "effective_qps": 0.0,
                "effective_qps_upper_bound": exc.optimistic_effective_qps,
                "candidate_seconds": exc.elapsed_seconds,
                "load_seconds": load_seconds,
                "load_penalty_weight": LOAD_PENALTY_WEIGHT,
                "recall_at_10": 0.0,
                "completed_queries": float(exc.completed_queries),
                "early_stopped": 1.0,
            }
            return _invalid(str(exc), metrics)
        except LoadTooSlow as exc:
            metrics = {
                "baseline_qps": benchmark.baseline_qps,
                "baseline_effective_qps": benchmark.baseline_effective_qps,
                "baseline_seconds": benchmark.baseline_seconds,
                "baseline_load_seconds": benchmark.baseline_load_seconds,
                "qps": 0.0,
                "effective_qps": 0.0,
                "effective_qps_upper_bound": exc.optimistic_effective_qps,
                "candidate_seconds": 0.0,
                "load_seconds": exc.load_seconds,
                "load_penalty_weight": LOAD_PENALTY_WEIGHT,
                "recall_at_10": 0.0,
                "inserted_vectors": float(exc.inserted),
                "early_stopped": 1.0,
            }
            return _invalid(str(exc), metrics)
        except Exception as exc:
            stderr = b""
            if process.poll() is not None and process.stderr is not None:
                stderr = process.stderr.read()[-800:]
            metrics = {
                "baseline_qps": benchmark.baseline_qps,
                "baseline_effective_qps": benchmark.baseline_effective_qps,
                "baseline_seconds": benchmark.baseline_seconds,
                "baseline_load_seconds": benchmark.baseline_load_seconds,
                "qps": 0.0,
                "effective_qps": 0.0,
                "candidate_seconds": 0.0,
                "load_seconds": 0.0,
                "recall_at_10": 0.0,
            }
            detail = stderr.decode("utf-8", errors="replace")
            suffix = f": {detail}" if detail else ""
            return _invalid(f"candidate benchmark failed: {exc}{suffix}", metrics)
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

    recall = _recall_at_k(results, benchmark.truth)
    qps = N_QUERIES / candidate_seconds
    effective_seconds = candidate_seconds + (LOAD_PENALTY_WEIGHT * load_seconds)
    effective_qps = N_QUERIES / max(effective_seconds, 1e-9)
    if recall < TARGET_RECALL or effective_qps <= benchmark.baseline_effective_qps:
        score = 0.0
    else:
        score = 100.0 * (
            1.0 - math.sqrt(benchmark.baseline_effective_qps) / math.sqrt(effective_qps)
        )

    metrics = {
        "qps": qps,
        "effective_qps": effective_qps,
        "baseline_qps": benchmark.baseline_qps,
        "baseline_effective_qps": benchmark.baseline_effective_qps,
        "recall_at_10": recall,
        "candidate_seconds": candidate_seconds,
        "load_seconds": load_seconds,
        "baseline_seconds": benchmark.baseline_seconds,
        "baseline_load_seconds": benchmark.baseline_load_seconds,
        "load_penalty_weight": LOAD_PENALTY_WEIGHT,
        "avg_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "p50_latency_ms": float(np.percentile(latencies, 50)) if latencies else 0.0,
        "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "p99_latency_ms": float(np.percentile(latencies, 99)) if latencies else 0.0,
        "concurrency": float(CONCURRENCY),
        "n_base": float(N_BASE),
        "n_queries": float(N_QUERIES),
        "top_k": float(TOP_K),
    }
    message = (
        f"N={N_BASE}; Q={N_QUERIES}; top_k={TOP_K}; "
        f"recall_at_10={recall:.6f}; qps={qps:.6f}; "
        f"effective_qps={effective_qps:.6f}; "
        f"baseline_effective_qps={benchmark.baseline_effective_qps:.6f}; "
        f"load_seconds={load_seconds:.6f}; score={score:.6f}"
    )
    return score, score, message, metrics


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 4 and sys.argv[1] == "--reference-server":
        _run_reference_server(int(sys.argv[2]), sys.argv[3])
        raise SystemExit(0)

    if len(sys.argv) != 2:
        print("usage: evaluator.py /path/to/rust/project", file=sys.stderr)
        raise SystemExit(2)
    bounded, unbounded, detail, metrics = evaluate(sys.argv[1])
    print(detail)
    print(json.dumps(metrics, indent=2))
    print(f"{bounded:.12f} {unbounded:.12f}")
