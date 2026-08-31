#!/usr/bin/env python3
"""Evaluator for the Frontier-CS 2.0 Vector DB ANN Disk task."""

from __future__ import annotations

import json
import math
import os
import shutil
import socket
import struct
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
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
N_BASE = _config_int("FRONTIER_VECTOR_DB_N", 100_000_000)
CONCURRENCY = _config_int("FRONTIER_VECTOR_DB_CONCURRENCY", CONFIG_CONCURRENCY)
QUERIES_PER_WORKER = _config_int(
    "FRONTIER_VECTOR_DB_QUERIES_PER_WORKER", CONFIG_QUERIES_PER_WORKER
)
N_QUERIES = _config_int(
    "FRONTIER_VECTOR_DB_Q", CONCURRENCY * QUERIES_PER_WORKER
)
# Iterative (agent `submit.sh`) feedback times only this many queries for a fast
# turnaround; the final verifier (FRONTIER_SUBMISSION_ROLE=final) always times
# the full N_QUERIES set for the authoritative score. See evaluate().
ITER_QUERIES = _config_int("FRONTIER_VECTOR_DB_ITER_Q", 2000)
TOP_K = _config_int("FRONTIER_VECTOR_DB_TOP_K", 10)
SEED = _config_int("FRONTIER_VECTOR_DB_SEED", 20260528)
GRAPH_DEGREE = _config_int("FRONTIER_VECTOR_DB_GRAPH_DEGREE", 64)
TARGET_RECALL = float(os.environ.get("FRONTIER_VECTOR_DB_TARGET_RECALL", "0.95"))
QUERY_NOISE = float(os.environ.get("FRONTIER_VECTOR_DB_QUERY_NOISE", "0.02"))
BUILD_TIMEOUT_SECONDS = _config_int("FRONTIER_VECTOR_DB_BUILD_TIMEOUT", 3600)
LOAD_TIMEOUT_SECONDS = _config_int("FRONTIER_VECTOR_DB_LOAD_TIMEOUT", 600)
WARMUP = _config_int("FRONTIER_VECTOR_DB_WARMUP", 32)
REFERENCE_BATCH_SIZE = _config_int("FRONTIER_VECTOR_DB_REFERENCE_BATCH_SIZE", 50_000)
LOCAL_GENERATION_LIMIT = _config_int(
    "FRONTIER_VECTOR_DB_LOCAL_GENERATION_LIMIT", 2_000_000
)
CACHE_DIR = Path(
    os.environ.get("FRONTIER_VECTOR_DB_CACHE", "/tmp/frontier_vector_db_ann_disk")
)
SMOKE_ONLY = os.environ.get("FRONTIER_VECTOR_DB_SMOKE_ONLY", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

DEFAULT_SIFT_DIR = Path(os.path.expanduser("~/nvme7/sift/index_100M"))
DEFAULT_SIFT_VECTOR_PATH = DEFAULT_SIFT_DIR / "100M.u8bin"
DEFAULT_SIFT_INDEX_PATH = DEFAULT_SIFT_DIR / "100M_disk.index"
DEFAULT_SIFT_PQ_PIVOTS_PATH = DEFAULT_SIFT_DIR / "100M_pq_pivots.bin"
DEFAULT_SIFT_PQ_COMPRESSED_PATH = DEFAULT_SIFT_DIR / "100M_pq_compressed.bin"
DEFAULT_SIFT_QUERY_PATH = DEFAULT_SIFT_DIR / "query.bin"
DEFAULT_SIFT_TRUTH_PATH = DEFAULT_SIFT_DIR / "truth.bin"
DEFAULT_SIFT_BASELINE_PATH = DEFAULT_SIFT_DIR / "baseline.json"
SUPPORTED_VECTOR_DTYPES = {
    "f32": np.dtype(np.float32),
    "float": np.dtype(np.float32),
    "float32": np.dtype(np.float32),
    "u8": np.dtype(np.uint8),
    "uint8": np.dtype(np.uint8),
    "i8": np.dtype(np.int8),
    "int8": np.dtype(np.int8),
}

_BENCHMARK: "Benchmark | None" = None


@dataclass
class MatrixSpec:
    path: Path
    rows: int
    dim: int
    dtype: np.dtype
    offset: int
    format: str


@dataclass
class Benchmark:
    index_path: Path
    vector_path: Path
    vector_dtype: str
    pq_vector_path: Path | None
    pq_pivots_path: Path | None
    queries_path: Path
    truth: np.ndarray | None
    baseline_qps: float
    baseline_seconds: float
    baseline_load_seconds: float


def prepare() -> dict:
    print(
        f"[vector-db-ann-disk] preparing benchmark n_base={N_BASE} "
        f"n_queries={N_QUERIES} top_k={TOP_K} graph_degree={GRAPH_DEGREE}",
        flush=True,
    )
    benchmark = _ensure_benchmark()
    print(
        f"[vector-db-ann-disk] benchmark ready baseline_qps="
        f"{benchmark.baseline_qps:.6f} baseline_seconds="
        f"{benchmark.baseline_seconds:.6f} baseline_load_seconds="
        f"{benchmark.baseline_load_seconds:.6f}",
        flush=True,
    )
    return {
        "n_base": N_BASE,
        "n_queries": N_QUERIES,
        "top_k": TOP_K,
        "graph_degree": GRAPH_DEGREE,
        "baseline_qps": benchmark.baseline_qps,
        "baseline_seconds": benchmark.baseline_seconds,
        "baseline_load_seconds": benchmark.baseline_load_seconds,
    }


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
    opener = request.build_opener(request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _wait_for_server(
    port: int, deadline: float, process: subprocess.Popen | None = None
) -> None:
    last_error: Exception | None = None
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            stderr = b""
            if process.stderr is not None:
                stderr = process.stderr.read()[-800:]
            detail = stderr.decode("utf-8", errors="replace")
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(f"server exited before becoming ready{suffix}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                pass
            return
        except Exception as exc:
            last_error = exc
            time.sleep(0.25)
    raise RuntimeError(f"server did not become ready: {last_error}")


def _coerce_matrix_dtype(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if dtype == np.dtype(np.float32):
        return values.astype(np.float32, copy=False)
    if dtype == np.dtype(np.uint8):
        info = np.iinfo(np.uint8)
        return np.rint(values).clip(info.min, info.max).astype(np.uint8)
    if dtype == np.dtype(np.int8):
        info = np.iinfo(np.int8)
        return np.rint(values).clip(info.min, info.max).astype(np.int8)
    raise ValueError(f"unsupported vector dtype: {dtype}")


def _write_vectors(
    path: Path, values: np.ndarray, dtype: np.dtype | None = None
) -> None:
    if dtype is None:
        dtype = np.dtype(np.float32)
    _coerce_matrix_dtype(values, dtype).tofile(path)


def _expand_path(value: str | os.PathLike[str]) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(os.fspath(value))))


def _env_path(name: str, default: Path) -> Path:
    return _expand_path(os.environ.get(name, str(default)))


def _env_path_any(names: tuple[str, ...], default: Path | None = None) -> Path | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return _expand_path(value)
    return default


def _using_default_sift_files() -> bool:
    return (
        N_BASE == 100_000_000
        and DEFAULT_SIFT_VECTOR_PATH.exists()
        and DEFAULT_SIFT_INDEX_PATH.exists()
    )


def _read_matrix_header(path: Path) -> tuple[int, int] | None:
    if path.stat().st_size < 8:
        return None
    with path.open("rb") as handle:
        raw = handle.read(8)
    rows, dim = struct.unpack("<II", raw)
    if rows <= 0 or dim <= 0:
        return None
    return int(rows), int(dim)


def _normalize_dtype(value: str) -> np.dtype:
    key = value.strip().lower()
    try:
        return SUPPORTED_VECTOR_DTYPES[key]
    except KeyError as exc:
        supported = ", ".join(sorted(SUPPORTED_VECTOR_DTYPES))
        raise ValueError(f"unsupported vector dtype {value!r}; supported: {supported}") from exc


def _dtype_hint(label: str) -> np.dtype | None:
    env_names = []
    normalized_label = label.lower()
    if normalized_label.startswith(("query", "queries")):
        env_names.append("FRONTIER_VECTOR_DB_QUERY_DTYPE")
        env_names.append("FRONTIER_VECTOR_DB_VECTOR_DTYPE")
    elif normalized_label.startswith(("vector", "vectors")):
        env_names.append("FRONTIER_VECTOR_DB_VECTOR_DTYPE")
    env_names.append("FRONTIER_VECTOR_DB_DTYPE")

    for name in env_names:
        value = os.environ.get(name)
        if value:
            return _normalize_dtype(value)
    return None


def _matrix_spec(
    path: Path, rows: int, dtype: np.dtype, offset: int, format_name: str
) -> MatrixSpec:
    return MatrixSpec(
        path=path,
        rows=rows,
        dim=DIM,
        dtype=dtype,
        offset=offset,
        format=format_name,
    )


def _rows_match(rows: int, expected_rows: int, allow_prefix: bool) -> bool:
    return rows == expected_rows or (allow_prefix and rows >= expected_rows)


def _detect_matrix_file(
    path: Path, expected_rows: int, label: str, allow_prefix: bool = False
) -> MatrixSpec:
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")

    size = path.stat().st_size
    hint = _dtype_hint(label)

    header = _read_matrix_header(path)
    if header is not None:
        rows, dim = header
        if dim == DIM and _rows_match(rows, expected_rows, allow_prefix):
            payload_bytes = size - 8
            if hint is not None:
                if payload_bytes == rows * dim * hint.itemsize:
                    return _matrix_spec(path, rows, hint, 8, f"{hint.name}_bin")
                raise RuntimeError(
                    f"{label} file size does not match dtype {hint.name}: {path}"
                )
            if payload_bytes == rows * dim:
                raise RuntimeError(
                    f"{label} file stores one byte per component. Set "
                    "FRONTIER_VECTOR_DB_VECTOR_DTYPE or "
                    "FRONTIER_VECTOR_DB_QUERY_DTYPE to uint8 or int8."
                )
            if payload_bytes == rows * dim * 4:
                return _matrix_spec(path, rows, np.dtype(np.float32), 8, "float32_bin")

    if hint is not None:
        row_bytes = DIM * hint.itemsize
        rows = size // row_bytes
        if size % row_bytes == 0 and _rows_match(rows, expected_rows, allow_prefix):
            return _matrix_spec(path, int(rows), hint, 0, f"raw_{hint.name}")
        raise RuntimeError(f"{label} file size does not match dtype {hint.name}: {path}")

    dtype = np.dtype(np.float32)
    row_bytes = DIM * dtype.itemsize
    rows = size // row_bytes
    if size % row_bytes == 0 and _rows_match(rows, expected_rows, allow_prefix):
        return _matrix_spec(path, int(rows), dtype, 0, "raw_float32")

    one_byte_rows = size // DIM
    if size % DIM == 0 and _rows_match(one_byte_rows, expected_rows, allow_prefix):
        raise RuntimeError(
            f"{label} file stores one byte per component but no dtype was provided. "
            "Set FRONTIER_VECTOR_DB_VECTOR_DTYPE or FRONTIER_VECTOR_DB_QUERY_DTYPE "
            "to uint8 or int8."
        )

    prefix = "at least " if allow_prefix else ""
    raise RuntimeError(
        f"{label} file has unexpected format or size: {path}; expected "
        f"{prefix}{expected_rows} rows with dim={DIM}. For non-float32 data, "
        "provide FRONTIER_VECTOR_DB_VECTOR_DTYPE or FRONTIER_VECTOR_DB_QUERY_DTYPE."
    )


def _load_vectors(path: Path, rows: int, label: str = "vectors") -> np.memmap:
    spec = _detect_matrix_file(path, rows, label, allow_prefix=True)
    return np.memmap(
        path,
        dtype=spec.dtype,
        mode="r",
        offset=spec.offset,
        shape=(rows, DIM),
    )


def _graph_vector_query_paths() -> tuple[
    Path, Path, Path | None, Path | None, Path, Path, Path
]:
    default_dir = CACHE_DIR / (
        f"n{N_BASE}_d{DIM}_deg{GRAPH_DEGREE}_q{N_QUERIES}_k{TOP_K}_seed{SEED}"
    )
    benchmark_dir = _env_path("FRONTIER_VECTOR_DB_BENCHMARK_DIR", default_dir)

    default_index_path = (
        DEFAULT_SIFT_INDEX_PATH if _using_default_sift_files() else benchmark_dir / "graph.bin"
    )
    default_vector_path = (
        DEFAULT_SIFT_VECTOR_PATH
        if _using_default_sift_files()
        else benchmark_dir / "vectors.bin"
    )

    index_path = _env_path_any(
        ("FRONTIER_VECTOR_DB_INDEX_PATH", "FRONTIER_VECTOR_DB_GRAPH_PATH"),
        default_index_path,
    )
    assert index_path is not None
    vector_path = _env_path("FRONTIER_VECTOR_DB_VECTOR_PATH", default_vector_path)

    default_pq_vector = (
        DEFAULT_SIFT_PQ_COMPRESSED_PATH
        if _using_default_sift_files() and DEFAULT_SIFT_PQ_COMPRESSED_PATH.exists()
        else None
    )
    default_pq_pivots = (
        DEFAULT_SIFT_PQ_PIVOTS_PATH
        if _using_default_sift_files() and DEFAULT_SIFT_PQ_PIVOTS_PATH.exists()
        else None
    )
    pq_vector_path = _env_path_any(
        ("FRONTIER_VECTOR_DB_PQ_VECTOR_PATH", "FRONTIER_VECTOR_DB_PQ_COMPRESSED_PATH"),
        default_pq_vector,
    )
    pq_pivots_path = _env_path_any(
        ("FRONTIER_VECTOR_DB_PQ_PIVOTS_PATH",), default_pq_pivots
    )

    default_queries_path = (
        DEFAULT_SIFT_QUERY_PATH
        if _using_default_sift_files() and DEFAULT_SIFT_QUERY_PATH.exists()
        else benchmark_dir / "queries.bin"
    )
    default_truth_path = (
        DEFAULT_SIFT_TRUTH_PATH
        if _using_default_sift_files() and DEFAULT_SIFT_TRUTH_PATH.exists()
        else benchmark_dir / "truth.u32"
    )
    default_meta_path = (
        DEFAULT_SIFT_BASELINE_PATH
        if _using_default_sift_files() and DEFAULT_SIFT_BASELINE_PATH.exists()
        else benchmark_dir / "baseline.json"
    )

    queries_path = _env_path("FRONTIER_VECTOR_DB_QUERY_PATH", default_queries_path)
    truth_path = _env_path("FRONTIER_VECTOR_DB_TRUTH_PATH", default_truth_path)
    meta_path = _env_path("FRONTIER_VECTOR_DB_BASELINE_PATH", default_meta_path)
    return (
        index_path,
        vector_path,
        pq_vector_path,
        pq_pivots_path,
        queries_path,
        truth_path,
        meta_path,
    )


def _generate_vectors(vector_path: Path, queries_path: Path) -> None:
    rng = np.random.default_rng(SEED)
    chunk = 50_000
    vector_dtype = _dtype_hint("vectors") or np.dtype(np.float32)
    query_dtype = _dtype_hint("queries") or vector_dtype
    base = np.memmap(vector_path, dtype=vector_dtype, mode="w+", shape=(N_BASE, DIM))
    for start in range(0, N_BASE, chunk):
        end = min(start + chunk, N_BASE)
        if vector_dtype == np.dtype(np.float32):
            base[start:end] = rng.standard_normal(
                (end - start, DIM), dtype=np.float32
            )
        elif vector_dtype == np.dtype(np.uint8):
            base[start:end] = rng.integers(
                0, 256, size=(end - start, DIM), dtype=np.uint16
            ).astype(np.uint8)
        elif vector_dtype == np.dtype(np.int8):
            base[start:end] = rng.integers(
                -128, 128, size=(end - start, DIM), dtype=np.int16
            ).astype(np.int8)
        else:
            raise ValueError(f"unsupported vector dtype: {vector_dtype}")
    base.flush()

    ids = rng.integers(0, N_BASE, size=N_QUERIES)
    selected = np.asarray(base[ids], dtype=np.float32)
    noise = rng.standard_normal((N_QUERIES, DIM), dtype=np.float32) * QUERY_NOISE
    _write_vectors(queries_path, selected + noise, query_dtype)


def _generate_queries_from_vectors(vector_path: Path, queries_path: Path) -> None:
    rng = np.random.default_rng(SEED)
    vector_spec = _detect_matrix_file(vector_path, N_BASE, "vectors")
    query_dtype = _dtype_hint("queries") or vector_spec.dtype
    base = _load_vectors(vector_path, N_BASE, "vectors")
    ids = rng.integers(0, N_BASE, size=N_QUERIES)
    selected = np.asarray(base[ids], dtype=np.float32)
    noise = rng.standard_normal((N_QUERIES, DIM), dtype=np.float32) * QUERY_NOISE
    queries_path.parent.mkdir(parents=True, exist_ok=True)
    _write_vectors(queries_path, selected + noise, query_dtype)


def _generate_local_graph(index_path: Path) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    degree = min(GRAPH_DEGREE, max(0, N_BASE - 1))
    with index_path.open("wb") as handle:
        handle.write(b"FCANNDK1")
        np.asarray([N_BASE], dtype=np.uint64).tofile(handle)
        np.asarray([DIM, degree], dtype=np.uint32).tofile(handle)
        if degree == 0:
            return
        chunk = 50_000
        offsets = np.arange(1, degree + 1, dtype=np.uint64)
        for start in range(0, N_BASE, chunk):
            end = min(start + chunk, N_BASE)
            ids = np.arange(start, end, dtype=np.uint64).reshape(-1, 1)
            neighbors = ((ids + offsets) % N_BASE).astype(np.uint32)
            degrees = np.full((end - start, 1), degree, dtype=np.uint32)
            np.concatenate([degrees, neighbors], axis=1).tofile(handle)


def _ensure_local_generation_allowed() -> None:
    if N_BASE <= LOCAL_GENERATION_LIMIT:
        return
    raise RuntimeError(
        "disk benchmark files were not found. Provide index/vector/query/truth "
        "paths with FRONTIER_VECTOR_DB_* environment variables, or set "
        "FRONTIER_VECTOR_DB_N to a small value for local smoke testing."
    )


def _ensure_data_files(
    index_path: Path,
    vector_path: Path,
    pq_vector_path: Path | None,
    pq_pivots_path: Path | None,
    queries_path: Path,
    truth_path: Path,
    meta_path: Path,
) -> None:
    has_vectors = vector_path.exists()
    has_queries = queries_path.exists()

    if has_vectors:
        _detect_matrix_file(vector_path, N_BASE, "vectors")
    else:
        _ensure_local_generation_allowed()
        print("[vector-db-ann-disk] generating local vectors", flush=True)
        vector_path.parent.mkdir(parents=True, exist_ok=True)
        queries_path.parent.mkdir(parents=True, exist_ok=True)
        _generate_vectors(vector_path, queries_path)
        truth_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
        has_queries = True

    if has_queries:
        _detect_matrix_file(queries_path, N_QUERIES, "queries", allow_prefix=True)
    else:
        print("[vector-db-ann-disk] generating local queries from vectors", flush=True)
        _generate_queries_from_vectors(vector_path, queries_path)
        truth_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)

    if not index_path.exists():
        _ensure_local_generation_allowed()
        print("[vector-db-ann-disk] generating local index graph", flush=True)
        _generate_local_graph(index_path)

    for label, path in (
        ("pq_vector_path", pq_vector_path),
        ("pq_pivots_path", pq_pivots_path),
    ):
        if path is not None and not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")


def _run_reference_server(port: int) -> None:
    import faiss

    index: faiss.IndexIDMap | None = None

    class ReferenceHandler(BaseHTTPRequestHandler):
        server_version = "FrontierVectorDiskReference/1.0"

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
            nonlocal index
            if self.path == "/load":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    payload = json.loads(self.rfile.read(length).decode("utf-8"))
                    index_path = Path(payload.get("index_path") or payload["graph_path"])
                    vector_path = Path(payload["vector_path"])
                    if not index_path.exists():
                        raise FileNotFoundError(f"index_path not found: {index_path}")
                    if not vector_path.exists():
                        raise FileNotFoundError(f"vector_path not found: {vector_path}")

                    base = _load_vectors(vector_path, N_BASE, "vectors")
                    new_index = faiss.IndexIDMap(faiss.IndexFlatL2(DIM))
                    for start in range(0, N_BASE, REFERENCE_BATCH_SIZE):
                        end = min(start + REFERENCE_BATCH_SIZE, N_BASE)
                        vectors = np.asarray(base[start:end], dtype=np.float32)
                        ids = np.arange(start, end, dtype=np.int64)
                        new_index.add_with_ids(vectors, ids)
                    index = new_index
                    self._write_json(200, {"status": "ok"})
                except Exception as exc:
                    self._write_json(400, {"status": "error", "error": str(exc)})
                return

            if self.path != "/search":
                self._write_json(404, {"status": "error", "error": "not found"})
                return

            try:
                if index is None:
                    raise RuntimeError("reference index has not been loaded")
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


def _load_payload(
    index_path: Path,
    vector_path: Path,
    vector_dtype: str,
    pq_vector_path: Path | None,
    pq_pivots_path: Path | None,
) -> dict:
    payload = {
        "index_path": str(index_path),
        "graph_path": str(index_path),
        "vector_path": str(vector_path),
        "vector_dtype": vector_dtype,
    }
    if pq_vector_path is not None:
        payload["pq_vector_path"] = str(pq_vector_path)
        payload["pq_compressed_path"] = str(pq_vector_path)
    if pq_pivots_path is not None:
        payload["pq_pivots_path"] = str(pq_pivots_path)
    return payload


def _load_service(
    base_url: str,
    index_path: Path,
    vector_path: Path,
    vector_dtype: str,
    pq_vector_path: Path | None,
    pq_pivots_path: Path | None,
) -> float:
    start = time.perf_counter()
    response = _post_json(
        f"{base_url}/load",
        _load_payload(
            index_path, vector_path, vector_dtype, pq_vector_path, pq_pivots_path
        ),
        timeout=LOAD_TIMEOUT_SECONDS,
    )
    load_seconds = max(time.perf_counter() - start, 1e-9)
    if response.get("status") != "ok":
        raise ValueError(f"load response did not report ok: {response}")
    if load_seconds > LOAD_TIMEOUT_SECONDS:
        raise TimeoutError(f"load timed out after {LOAD_TIMEOUT_SECONDS}s")
    return load_seconds


def _measure_reference_baseline(
    index_path: Path,
    vector_path: Path,
    vector_dtype: str,
    pq_vector_path: Path | None,
    pq_pivots_path: Path | None,
    queries: np.ndarray,
) -> tuple[np.ndarray, list[float], float, float]:
    port = _free_port()
    process = subprocess.Popen(
        ["python3", str(Path(__file__).resolve()), "--reference-server", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        _wait_for_server(port, time.time() + 120, process)
        load_seconds = _load_service(
            f"http://127.0.0.1:{port}",
            index_path,
            vector_path,
            vector_dtype,
            pq_vector_path,
            pq_pivots_path,
        )
        results, latencies, baseline_seconds = _run_queries(
            f"http://127.0.0.1:{port}", queries, N_QUERIES
        )
        return results, latencies, baseline_seconds, load_seconds
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def _load_truth(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype=np.uint32)
    required = N_QUERIES * TOP_K
    if values.size >= required + 2 and values[0] == N_QUERIES and values[1] == TOP_K:
        values = values[2:]
    if values.size < required:
        raise RuntimeError(
            f"truth file has too few values: {path}; expected at least {required}"
        )
    return values[:required].reshape(N_QUERIES, TOP_K)


def _ensure_benchmark() -> Benchmark:
    global _BENCHMARK
    if _BENCHMARK is not None:
        return _BENCHMARK

    (
        index_path,
        vector_path,
        pq_vector_path,
        pq_pivots_path,
        queries_path,
        truth_path,
        meta_path,
    ) = _graph_vector_query_paths()
    index_path.parent.mkdir(parents=True, exist_ok=True)
    vector_path.parent.mkdir(parents=True, exist_ok=True)
    queries_path.parent.mkdir(parents=True, exist_ok=True)
    truth_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    _ensure_data_files(
        index_path,
        vector_path,
        pq_vector_path,
        pq_pivots_path,
        queries_path,
        truth_path,
        meta_path,
    )

    vector_spec = _detect_matrix_file(vector_path, N_BASE, "vectors")
    _detect_matrix_file(queries_path, N_QUERIES, "queries", allow_prefix=True)

    queries = _load_vectors(queries_path, N_QUERIES, "queries")

    if truth_path.exists() and meta_path.exists():
        truth = _load_truth(truth_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        baseline_seconds = float(meta["baseline_seconds"])
        baseline_qps = float(meta["baseline_qps"])
        baseline_load_seconds = float(meta["baseline_load_seconds"])
        # Anti-cheat: the candidate service is built and run by this judge in the
        # same container and could otherwise read the ground truth / baseline
        # straight off disk and fabricate perfect results. They are now held in
        # memory (truth + baseline_* above and the cached _BENCHMARK), so remove
        # the files before any candidate runs. _ensure_benchmark caches globally,
        # so later submissions reuse the in-memory copy and never need the files.
        # Only for the real baked benchmark (small-N local CI keeps its cache and
        # has no candidate-isolation concern). Opt out with FRONTIER_VECTOR_DB_KEEP_TRUTH.
        keep = os.environ.get("FRONTIER_VECTOR_DB_KEEP_TRUTH", "").lower() in {
            "1",
            "true",
            "yes",
        }
        if N_BASE > LOCAL_GENERATION_LIMIT and not keep:
            for secret_path in (truth_path, meta_path):
                try:
                    secret_path.unlink()
                except OSError:
                    pass
    elif SMOKE_ONLY:
        truth = None
        baseline_seconds = 0.0
        baseline_qps = 0.0
        baseline_load_seconds = 0.0
    else:
        if N_BASE > LOCAL_GENERATION_LIMIT:
            raise RuntimeError(
                "truth/baseline files were not found and the local exact reference "
                "is disabled for this benchmark size. Set "
                "FRONTIER_VECTOR_DB_SMOKE_ONLY=1 to run endpoint/load/throughput "
                "testing without recall scoring."
            )
        print("[vector-db-ann-disk] running Faiss HTTP exact baseline", flush=True)
        truth, _, baseline_seconds, baseline_load_seconds = _measure_reference_baseline(
            index_path,
            vector_path,
            vector_spec.dtype.name,
            pq_vector_path,
            pq_pivots_path,
            queries,
        )
        truth.astype(np.uint32, copy=False).tofile(truth_path)
        baseline_qps = N_QUERIES / baseline_seconds
        meta_path.write_text(
            json.dumps(
                {
                    "baseline_seconds": baseline_seconds,
                    "baseline_qps": baseline_qps,
                    "baseline_load_seconds": baseline_load_seconds,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    _BENCHMARK = Benchmark(
        index_path=index_path,
        vector_path=vector_path,
        vector_dtype=vector_spec.dtype.name,
        pq_vector_path=pq_vector_path,
        pq_pivots_path=pq_pivots_path,
        queries_path=queries_path,
        truth=truth,
        baseline_qps=baseline_qps,
        baseline_seconds=baseline_seconds,
        baseline_load_seconds=baseline_load_seconds,
    )
    return _BENCHMARK


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
    base_url: str, queries: np.ndarray, n_eval: int
) -> tuple[np.ndarray, list[float], float]:
    for i in range(min(WARMUP, n_eval)):
        try:
            _search_one(base_url, i, queries[i])
        except Exception:
            pass

    results = np.zeros((n_eval, TOP_K), dtype=np.uint32)
    latencies: list[float] = []
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = [
            pool.submit(_search_one, base_url, i, queries[i])
            for i in range(n_eval)
        ]
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
    duration = max(time.perf_counter() - start, 1e-9)
    return results, latencies, duration


def _recall_at_k(results: np.ndarray, truth: np.ndarray) -> float:
    hits = 0
    n = len(results)
    for got, expected in zip(results, truth):
        hits += len(set(int(x) for x in got) & set(int(x) for x in expected))
    return hits / float(max(n, 1) * TOP_K)


def evaluate(solution_path: str):
    root = Path(solution_path)
    if not root.is_dir():
        return _invalid("submission path must be a Rust project directory")
    if not (root / "Cargo.toml").exists():
        return _invalid("Cargo.toml not found in submission directory")

    try:
        benchmark = _ensure_benchmark()
    except Exception as exc:
        return _invalid(f"benchmark preparation failed: {exc}")

    # The judge sets FRONTIER_SUBMISSION_ROLE per submission: "final" for the
    # verifier (authoritative, full query set) and "agent" for iterative
    # `submit.sh` feedback (fast subset). Timing only n_eval queries keeps the
    # iterative loop responsive so agents get a real score without cancelling.
    is_final = (
        os.environ.get("FRONTIER_SUBMISSION_ROLE", "agent").strip().lower() == "final"
    )
    n_eval = N_QUERIES if is_final else max(1, min(N_QUERIES, ITER_QUERIES))

    with tempfile.TemporaryDirectory(prefix="frontier_vector_db_ann_disk_") as tmp:
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

        queries = _load_vectors(benchmark.queries_path, N_QUERIES, "queries")
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
            _wait_for_server(port, time.time() + 30, process)
            load_seconds = _load_service(
                base_url,
                benchmark.index_path,
                benchmark.vector_path,
                benchmark.vector_dtype,
                benchmark.pq_vector_path,
                benchmark.pq_pivots_path,
            )
            results, latencies, candidate_seconds = _run_queries(
                base_url, queries, n_eval
            )
        except Exception as exc:
            stderr = b""
            if process.poll() is not None and process.stderr is not None:
                stderr = process.stderr.read()[-800:]
            metrics = {
                "baseline_qps": benchmark.baseline_qps,
                "baseline_seconds": benchmark.baseline_seconds,
                "baseline_load_seconds": benchmark.baseline_load_seconds,
                "qps": 0.0,
                "candidate_seconds": 0.0,
                "load_seconds": load_seconds,
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

    recall = (
        _recall_at_k(results, benchmark.truth)
        if benchmark.truth is not None
        else None
    )
    qps = n_eval / candidate_seconds
    if recall is None:
        score = 0.0
    elif recall < TARGET_RECALL or qps <= benchmark.baseline_qps:
        score = 0.0
    else:
        score = 100.0 * (1.0 - math.sqrt(benchmark.baseline_qps) / math.sqrt(qps))

    metrics = {
        "qps": qps,
        "baseline_qps": benchmark.baseline_qps,
        "recall_at_10": recall,
        "candidate_seconds": candidate_seconds,
        "load_seconds": load_seconds,
        "baseline_seconds": benchmark.baseline_seconds,
        "baseline_load_seconds": benchmark.baseline_load_seconds,
        "avg_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "p50_latency_ms": float(np.percentile(latencies, 50)) if latencies else 0.0,
        "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "p99_latency_ms": float(np.percentile(latencies, 99)) if latencies else 0.0,
        "concurrency": float(CONCURRENCY),
        "n_base": float(N_BASE),
        "n_queries": float(n_eval),
        "n_queries_full": float(N_QUERIES),
        "submission_role": "final" if is_final else "agent",
        "top_k": float(TOP_K),
        "graph_degree": float(GRAPH_DEGREE),
        "smoke_only": bool(recall is None),
    }
    recall_text = "skipped" if recall is None else f"{recall:.6f}"
    mode = "final" if is_final else "iterative"
    message = (
        f"[{mode}] N={N_BASE}; Q={n_eval}; top_k={TOP_K}; "
        f"recall_at_10={recall_text}; qps={qps:.6f}; "
        f"baseline_qps={benchmark.baseline_qps:.6f}; "
        f"load_seconds={load_seconds:.6f}; score={score:.6f}"
    )
    if not is_final:
        message += (
            f" (fast estimate over {n_eval} of {N_QUERIES} queries; "
            f"the final verifier scores all {N_QUERIES})"
        )
    return score, score, message, metrics


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3 and sys.argv[1] == "--reference-server":
        _run_reference_server(int(sys.argv[2]))
        raise SystemExit(0)

    if len(sys.argv) != 2:
        print("usage: evaluator.py /path/to/rust/project", file=sys.stderr)
        raise SystemExit(2)
    bounded, unbounded, detail, metrics = evaluate(sys.argv[1])
    print(detail)
    print(json.dumps(metrics, indent=2))
    print(f"{bounded:.12f} {unbounded:.12f}")
