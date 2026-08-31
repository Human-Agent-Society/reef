"""Evaluator for the experimental DuckDB E2E query optimization task."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import os
import posixpath
import random
import re
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_PATCH_BYTES = 1_500_000
MAX_CHANGED_FILES = 80
TASK_CONFIG_PATH = Path("/judge/task_config.json")


def _load_task_config() -> dict[str, Any]:
    try:
        payload = json.loads(TASK_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


TASK_CONFIG = _load_task_config()
EVALUATION_CONFIG = TASK_CONFIG.get("evaluation", {}) if isinstance(TASK_CONFIG.get("evaluation"), dict) else {}


def _config_int(env_name: str, config_name: str, default: int) -> int:
    del env_name
    raw = EVALUATION_CONFIG.get(config_name, default)
    try:
        return int(raw)
    except Exception:
        return default


def _config_str(env_name: str, config_name: str, default: str) -> str:
    del env_name
    raw = EVALUATION_CONFIG.get(config_name, default)
    return str(raw)


def _config_bool(env_name: str, config_name: str, default: bool) -> bool:
    del env_name
    raw = EVALUATION_CONFIG.get(config_name, default)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return bool(raw)


BUILD_TIMEOUT_SECONDS = _config_int("FRONTIER_DUCKDB_BUILD_TIMEOUT", "build_timeout_seconds", 3600)
QUERY_TIMEOUT_SECONDS = _config_int("FRONTIER_DUCKDB_QUERY_TIMEOUT", "query_timeout_seconds", 300)
SQLLOGICTEST_TIMEOUT_SECONDS = _config_int(
    "FRONTIER_DUCKDB_SQLLOGICTEST_TIMEOUT",
    "sqllogictest_timeout_seconds",
    600,
)
DUCKDB_MEMORY_LIMIT = _config_str("FRONTIER_DUCKDB_MEMORY_LIMIT", "duckdb_memory_limit", "6GB")
DUCKDB_TEMP_LIMIT = _config_str("FRONTIER_DUCKDB_TEMP_LIMIT", "duckdb_temp_limit", "2GB")
CHILD_MEMORY_LIMIT_MB = _config_int("FRONTIER_DUCKDB_CHILD_MEMORY_MB", "child_memory_mb", 12288)
USE_BENCHMARK_RUNNER = _config_bool("FRONTIER_DUCKDB_USE_BENCHMARK_RUNNER", "use_benchmark_runner", False)

DEFAULT_CLEAN_SOURCE = Path("/opt/duckdb-clean")
DEFAULT_VANILLA_SOURCE = Path("/opt/duckdb-vanilla")
PUBLIC_SCALE_FACTOR = _config_str("FRONTIER_DUCKDB_TPCH_SF", "scale_factor", "1")
DEFAULT_HIDDEN_SCALE_FACTORS = ("0.1", "2")
QUERY_ORDER_SEED = _config_int("FRONTIER_DUCKDB_QUERY_ORDER_SEED", "query_order_seed", 20260604)

BENCHMARK_RUNNER_REL = Path("build/release/benchmark/benchmark_runner")
DUCKDB_SHELL_REL = Path("build/release/duckdb")
DUCKDB_UNITTEST_REL = Path("build/release/test/unittest")
DEFAULT_CORRECTNESS_QUERIES = tuple(range(1, 23))
FINAL_SQLLOGICTEST_FILES = (
    "test/sql/join/inner/test_join.test",
    "test/sql/join/inner/test_inner_join_filter_pushdown.test",
    "test/sql/join/semianti/semijoin.test",
    "test/sql/filter/test_transitive_filters.test",
    "test/sql/aggregate/distinct/test_distinct.test",
)

STRONGLY_ALLOWED_PATTERNS = (
    "src/optimizer/**",
    "src/include/duckdb/optimizer/**",
    "src/execution/operator/join/**",
    "src/include/duckdb/execution/operator/join/**",
    "src/execution/operator/filter/**",
    "src/include/duckdb/execution/operator/filter/**",
    "src/planner/operator/logical_join.cpp",
    "src/planner/operator/logical_comparison_join.cpp",
    "src/include/duckdb/planner/operator/logical_join.hpp",
    "src/include/duckdb/planner/operator/logical_comparison_join.hpp",
)

CONDITIONALLY_ALLOWED_PATTERNS = (
    "src/planner/**",
    "src/include/duckdb/planner/**",
    "src/execution/physical_plan/**",
    "src/include/duckdb/execution/physical_plan/**",
    "src/common/**",
    "src/include/duckdb/common/**",
)

BUILD_ALLOWED_PATTERNS = (
    "CMakeLists.txt",
    "src/CMakeLists.txt",
    "src/optimizer/CMakeLists.txt",
    "src/optimizer/**/CMakeLists.txt",
    "src/planner/CMakeLists.txt",
    "src/planner/**/CMakeLists.txt",
    "src/planner/operator/CMakeLists.txt",
    "src/common/CMakeLists.txt",
    "src/common/**/CMakeLists.txt",
    "src/execution/CMakeLists.txt",
    "src/execution/operator/CMakeLists.txt",
    "src/execution/operator/join/CMakeLists.txt",
    "src/execution/operator/join/**/CMakeLists.txt",
    "src/execution/operator/filter/CMakeLists.txt",
    "src/execution/operator/filter/**/CMakeLists.txt",
    "src/execution/physical_plan/CMakeLists.txt",
    "src/execution/physical_plan/**/CMakeLists.txt",
    "extension_config.cmake",
)

DENIED_PATTERNS = (
    "benchmark/**",
    "test/**",
    "tools/**",
    "scripts/**",
    "extension/tpch/**",
    "third_party/**",
    "examples/**",
    "docs/**",
    ".github/**",
    "src/main/**",
    "src/storage/**",
    "src/catalog/**",
    "src/parser/**",
    "src/function/**",
    "src/common/file_system.cpp",
    "src/main/client_context.cpp",
    "src/main/database.cpp",
    "CMakePresets.json",
    "Makefile",
    "package.json",
    "requirements.txt",
)

ENVIRONMENT_TOKENS = (
    "getenv",
    "std::getenv",
    "secure_getenv",
    "setenv",
    "putenv",
    "unsetenv",
    "environ",
    "GetEnvironmentVariable",
    "FileSystem::GetEnv",
)

CMAKE_DENY_TOKENS = (
    "add_custom_command",
    "add_custom_target",
    "execute_process",
    "FetchContent",
    "ExternalProject",
    "file(DOWNLOAD",
    "configure_file",
    "target_compile_options",
    "target_link_options",
    "set(CMAKE_",
    "install(",
)

HARD_CODE_TOKENS = (
    "tpch",
    "lineitem",
    "orders",
    "customer",
    "supplier",
    "part",
    "partsupp",
    "nation",
    "region",
    "q01",
    "q02",
    "q03",
    "q04",
    "q05",
    "q06",
    "q07",
    "q08",
    "q09",
    "q10",
    "q11",
    "q12",
    "q13",
    "q14",
    "q15",
    "q16",
    "q17",
    "q18",
    "q19",
    "q20",
    "q21",
    "q22",
    "benchmark",
)

HARD_CODE_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_])("
    + "|".join(re.escape(token) for token in HARD_CODE_TOKENS)
    + r")(?![A-Za-z0-9_])",
    re.IGNORECASE,
)

SOURCE_EXTENSIONS = (".cpp", ".hpp", ".h", ".hh", ".cc", ".cxx")


@dataclass(frozen=True)
class PatchFile:
    old_path: str
    new_path: str
    added_lines: tuple[str, ...]
    removed_lines: tuple[str, ...]

    @property
    def path(self) -> str:
        return self.new_path if self.new_path != "/dev/null" else self.old_path


def _match(path: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def _is_build_file(path: str) -> bool:
    return _match(path, BUILD_ALLOWED_PATTERNS)


def _is_allowed_source_path(path: str) -> bool:
    return _match(path, STRONGLY_ALLOWED_PATTERNS) or _match(
        path, CONDITIONALLY_ALLOWED_PATTERNS
    )


def _invalid(message: str, metrics: dict[str, Any] | None = None):
    payload = metrics or {}
    payload.setdefault("valid_patch", 0)
    return 0.0, 0.0, message, payload


def _parse_patch(text: str) -> list[PatchFile]:
    files: list[PatchFile] = []
    current_old = ""
    current_new = ""
    added: list[str] = []
    removed: list[str] = []
    in_file = False

    for line in text.splitlines():
        if line.startswith("diff --git "):
            if in_file:
                files.append(PatchFile(current_old, current_new, tuple(added), tuple(removed)))
            in_file = True
            current_old = ""
            current_new = ""
            added = []
            removed = []
            continue
        if not in_file:
            continue
        if line.startswith("--- "):
            current_old = line[4:].strip()
            if current_old.startswith("a/"):
                current_old = current_old[2:]
            continue
        if line.startswith("+++ "):
            current_new = line[4:].strip()
            if current_new.startswith("b/"):
                current_new = current_new[2:]
            continue
        if line.startswith("+") and not line.startswith("+++ "):
            added.append(line[1:])
            continue
        if line.startswith("-") and not line.startswith("--- "):
            removed.append(line[1:])

    if in_file:
        files.append(PatchFile(current_old, current_new, tuple(added), tuple(removed)))
    return files


def _validate_patch_path(path: str, metrics: dict[str, Any]) -> tuple[bool, str, bool, bool]:
    if not path or path == "/dev/null":
        return True, "", False, False
    if path.startswith("/") or ".." in Path(path).parts:
        return False, f"unsafe patch path: {path}", False, False
    if _match(path, DENIED_PATTERNS):
        return False, f"changed file is outside task boundary: {path}", False, False

    build_file = _is_build_file(path)
    allowed_source = _is_allowed_source_path(path)
    if not build_file and not allowed_source:
        return False, f"changed file is not allowlisted: {path}", False, False
    return True, "", build_file, allowed_source


def _validate_build_diff(patch_file: PatchFile) -> str | None:
    if any(line.strip() for line in patch_file.removed_lines):
        return "build-system changes may add allowed source files but may not delete or rewrite existing build rules"
    added_text = "\n".join(patch_file.added_lines)
    for token in CMAKE_DENY_TOKENS:
        if token in added_text:
            return f"build-system change contains forbidden token: {token}"
    for line in patch_file.added_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ".cpp" in stripped:
            cpp_paths = re.findall(r"[A-Za-z0-9_./+-]+\.cpp", stripped)
            if not cpp_paths:
                return "CMake change mentions .cpp without a parseable source path"
            for cpp_path in cpp_paths:
                normalized = cpp_path.strip("./")
                if not normalized.startswith("src/"):
                    normalized = posixpath.normpath(
                        posixpath.join(posixpath.dirname(patch_file.path), normalized)
                    )
                if normalized.startswith("../") or not _is_allowed_source_path(normalized):
                    return f"CMake change wires source outside allowlist: {cpp_path}"
            continue
        if stripped in {"(", ")", "set(", "duckdb_sources("}:
            continue
        if stripped.endswith("(") or stripped.endswith(")"):
            continue
        return "CMake changes may only add allowed .cpp files to existing targets"
    return None


def validate_patch(patch_path: Path) -> tuple[bool, str, dict[str, Any]]:
    if not patch_path.exists():
        return False, "solution patch does not exist", {}
    size = patch_path.stat().st_size
    if size > MAX_PATCH_BYTES:
        return False, f"patch is too large ({size} bytes > {MAX_PATCH_BYTES})", {}
    text = patch_path.read_text(encoding="utf-8", errors="replace")
    patch_hash = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    files = _parse_patch(text)
    metrics: dict[str, Any] = {
        "patch_bytes": size,
        "patch_sha256": patch_hash,
        "changed_files": len(files),
    }
    if len(files) > MAX_CHANGED_FILES:
        return False, f"too many changed files ({len(files)} > {MAX_CHANGED_FILES})", metrics

    for patch_file in files:
        path = patch_file.path
        if patch_file.new_path == "/dev/null":
            return False, f"deleting source files is outside task boundary: {patch_file.old_path}", metrics
        if patch_file.old_path != "/dev/null" and patch_file.old_path != patch_file.new_path:
            old_ok, old_error, _, _ = _validate_patch_path(patch_file.old_path, metrics)
            if not old_ok:
                return False, f"rename/copy source is outside task boundary: {old_error}", metrics

        ok, error, build_file, allowed_source = _validate_patch_path(path, metrics)
        if not ok:
            return False, error, metrics
        if not path or path == "/dev/null":
            return False, "could not determine changed path from patch", metrics

        if build_file:
            error = _validate_build_diff(patch_file)
            if error:
                return False, f"{path}: {error}", metrics

        if path.endswith(SOURCE_EXTENSIONS):
            added_text = "\n".join(patch_file.added_lines)
            for token in ENVIRONMENT_TOKENS:
                if token in added_text:
                    return False, f"{path}: environment access is forbidden ({token})", metrics
            match = HARD_CODE_TOKEN_RE.search(added_text)
            if match:
                token = match.group(1)
                return False, f"{path}: benchmark-specific token is forbidden ({token})", metrics

    metrics["valid_patch"] = 1
    return True, "patch accepted by static policy", metrics


def clean_env(tmp_root: Path) -> dict[str, str]:
    home = tmp_root / "home"
    tmp = tmp_root / "tmp"
    home.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    return {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": str(home),
        "TMPDIR": str(tmp),
        "LC_ALL": "C",
        "LANG": "C",
        "CCACHE_DIR": str(tmp_root / "ccache"),
    }


def run_checked(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: int,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    def limit_child() -> None:
        if os.name != "posix":
            return
        try:
            os.setsid()
        except Exception:
            pass
        try:
            import resource

            limit_bytes = CHILD_MEMORY_LIMIT_MB * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
        except Exception:
            pass

    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=limit_child if os.name == "posix" else None,
    )
    try:
        stdout, stderr = process.communicate(input=input_text, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except Exception:
                process.kill()
        else:
            process.kill()
        stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(
            cmd=cmd,
            timeout=timeout,
            output=stdout,
            stderr=stderr,
        ) from exc

    completed = subprocess.CompletedProcess(
        cmd,
        process.returncode,
        stdout=stdout,
        stderr=stderr,
    )
    if process.returncode != 0:
        raise subprocess.CalledProcessError(
            process.returncode,
            cmd,
            output=stdout,
            stderr=stderr,
        )
    return completed


def patch_is_empty(metrics: dict[str, Any]) -> bool:
    return int(metrics.get("changed_files", 0)) == 0


def _parse_csv_or_list(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, list):
        return tuple(str(item).strip() for item in raw if str(item).strip())
    if isinstance(raw, str) and raw.strip():
        return tuple(item.strip() for item in raw.split(",") if item.strip())
    return ()


def evaluation_scale_factors(*, final_role: bool) -> tuple[str, ...]:
    config_key = "scale_factors" if final_role else "agent_scale_factors"
    configured = _parse_csv_or_list(EVALUATION_CONFIG.get(config_key))
    if configured:
        raw_values = configured
    elif final_role:
        raw_values = (PUBLIC_SCALE_FACTOR, *DEFAULT_HIDDEN_SCALE_FACTORS)
    else:
        raw_values = (PUBLIC_SCALE_FACTOR,)
    seen: set[str] = set()
    result: list[str] = []
    for value in raw_values:
        normalized = str(value).strip()
        if not normalized:
            continue
        try:
            if float(normalized) <= 0:
                continue
        except Exception:
            continue
        if normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return tuple(result) or (PUBLIC_SCALE_FACTOR,)


def shuffled(items: list[Any], salt: str) -> list[Any]:
    rng = random.Random(f"{QUERY_ORDER_SEED}:{salt}")
    rng.shuffle(items)
    return items


def restore_prebuilt_source(source_dir: Path, env: dict[str, str]) -> bool:
    """Undo the previous submission while preserving ignored prebuilt artifacts."""
    status_before = run_checked(
        ["git", "status", "--porcelain"],
        cwd=source_dir,
        env=env,
        timeout=60,
    ).stdout
    run_checked(["git", "reset", "--hard", "HEAD"], cwd=source_dir, env=env, timeout=60)
    clean = run_checked(["git", "clean", "-fd", "src"], cwd=source_dir, env=env, timeout=60).stdout
    return bool(status_before.strip()) or bool(clean.strip())


def build_duckdb(source_dir: Path, env: dict[str, str], *, include_unittest: bool = False) -> None:
    build_env = dict(env)
    build_env["GEN"] = "ninja"
    build_env["BUILD_BENCHMARK"] = "1"
    build_env["BUILD_EXTENSIONS"] = "tpch"
    build_env.setdefault("BUILD_UNITTESTS", "1" if include_unittest else "0")
    build_env.setdefault("DISABLE_UNITY", "1")
    build_env.setdefault("DISABLE_PARQUET", "1")
    build_env.setdefault("BUILD_JEMALLOC", "0")
    build_env.setdefault("CMAKE_BUILD_PARALLEL_LEVEL", "1")
    targets = ["duckdb", "benchmark_runner"]
    if include_unittest:
        targets.append("unittest")
    run_checked(
        [
            "cmake",
            "--build",
            "build/release",
            "--config",
            "Release",
            "--target",
            *targets,
        ],
        cwd=source_dir,
        env=build_env,
        timeout=BUILD_TIMEOUT_SECONDS,
    )


def benchmark_runner(source_dir: Path) -> Path:
    runner = source_dir / BENCHMARK_RUNNER_REL
    if not runner.exists():
        raise FileNotFoundError(f"benchmark_runner not found at {runner}")
    return runner


def duckdb_shell(source_dir: Path) -> Path:
    shell = source_dir / DUCKDB_SHELL_REL
    if not shell.exists():
        raise FileNotFoundError(f"DuckDB shell not found at {shell}")
    return shell


def duckdb_unittest(source_dir: Path) -> Path:
    runner = source_dir / DUCKDB_UNITTEST_REL
    if not runner.exists():
        raise FileNotFoundError(f"DuckDB unittest runner not found at {runner}")
    return runner


def benchmark_env(base_env: dict[str, str], tmp_root: Path) -> dict[str, str]:
    env = dict(base_env)
    env["DUCKDB_BENCHMARK_MEMORY_LIMIT"] = DUCKDB_MEMORY_LIMIT
    env["DUCKDB_BENCHMARK_TEMP_LIMIT"] = DUCKDB_TEMP_LIMIT
    env["DUCKDB_BENCHMARK_TEMP_DIR"] = str(tmp_root / "duckdb_tmp")
    return env


def settings_sql(temp_dir: Path) -> str:
    temp_dir.mkdir(parents=True, exist_ok=True)
    return "\n".join(
        (
            "SET threads = 1;",
            f"SET memory_limit = '{DUCKDB_MEMORY_LIMIT}';",
            f"SET max_temp_directory_size = '{DUCKDB_TEMP_LIMIT}';",
            f"SET temp_directory = '{temp_dir.as_posix()}';",
            "SET preserve_insertion_order = false;",
        )
    )


def is_final_submission_role() -> bool:
    return os.environ.get("FRONTIER_SUBMISSION_ROLE", "agent") == "final"


def run_final_sqllogictest_smoke(
    source_dir: Path,
    *,
    env: dict[str, str],
    metrics: dict[str, Any],
) -> None:
    runner = duckdb_unittest(source_dir)
    start = time.perf_counter()
    for test_file in FINAL_SQLLOGICTEST_FILES:
        run_checked(
            [str(runner), test_file],
            cwd=source_dir,
            env=env,
            timeout=SQLLOGICTEST_TIMEOUT_SECONDS,
        )
    metrics["final_sqllogictest"] = 1
    metrics["final_sqllogictest_count"] = len(FINAL_SQLLOGICTEST_FILES)
    metrics["final_sqllogictest_seconds"] = round(time.perf_counter() - start, 3)


def run_duckdb_sql(
    shell: Path,
    database: Path,
    sql: str,
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: int,
) -> str:
    proc = run_checked(
        [str(shell), str(database), "-csv", "-c", sql],
        cwd=cwd,
        env=env,
        timeout=timeout,
    )
    return proc.stdout


def prepare_tpch_database(
    shell: Path,
    database: Path,
    *,
    cwd: Path,
    env: dict[str, str],
    tmp_root: Path,
    scale_factor: str,
) -> None:
    sql = "\n".join(
        (
            settings_sql(tmp_root / "duckdb_tmp"),
            "LOAD tpch;",
            f"CALL dbgen(sf = {scale_factor});",
        )
    )
    run_duckdb_sql(shell, database, sql, cwd=cwd, env=env, timeout=QUERY_TIMEOUT_SECONDS)


def correctness_queries() -> tuple[int, ...]:
    raw = EVALUATION_CONFIG.get("correctness_queries")
    if isinstance(raw, list):
        return tuple(int(item) for item in raw)
    if isinstance(raw, str) and raw.strip():
        return tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    return DEFAULT_CORRECTNESS_QUERIES


def output_digest(text: str) -> str:
    normalized = "\n".join(line.rstrip() for line in text.splitlines())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def check_correctness(
    vanilla_source: Path,
    patched_source: Path,
    *,
    env: dict[str, str],
    tmp_root: Path,
    scale_factor: str,
    queries: tuple[int, ...],
) -> dict[str, str]:
    vanilla_shell = duckdb_shell(vanilla_source)
    patched_shell = duckdb_shell(patched_source)
    scale_label = scale_factor.replace(".", "_")
    vanilla_db = tmp_root / f"vanilla_tpch_sf{scale_label}.duckdb"
    patched_db = tmp_root / f"patched_tpch_sf{scale_label}.duckdb"
    prepare_tpch_database(
        vanilla_shell,
        vanilla_db,
        cwd=vanilla_source,
        env=env,
        tmp_root=tmp_root / f"vanilla_setup_sf{scale_label}",
        scale_factor=scale_factor,
    )
    prepare_tpch_database(
        patched_shell,
        patched_db,
        cwd=patched_source,
        env=env,
        tmp_root=tmp_root / f"patched_setup_sf{scale_label}",
        scale_factor=scale_factor,
    )

    mismatches: dict[str, str] = {}
    for query_nr in queries:
        query_sql = "\n".join(
            (
                settings_sql(tmp_root / f"query_{query_nr}_tmp"),
                "LOAD tpch;",
                f"PRAGMA tpch({query_nr});",
            )
        )
        vanilla_out = run_duckdb_sql(
            vanilla_shell,
            vanilla_db,
            query_sql,
            cwd=vanilla_source,
            env=env,
            timeout=QUERY_TIMEOUT_SECONDS,
        )
        patched_out = run_duckdb_sql(
            patched_shell,
            patched_db,
            query_sql,
            cwd=patched_source,
            env=env,
            timeout=QUERY_TIMEOUT_SECONDS,
        )
        if output_digest(vanilla_out) != output_digest(patched_out):
            mismatches[f"sf{scale_factor}:q{query_nr:02d}"] = "result differs from vanilla DuckDB"
    return mismatches


def run_benchmark(
    runner: Path,
    benchmark: str,
    *,
    cwd: Path,
    env: dict[str, str],
    tmp_root: Path,
) -> float:
    tmp_root.mkdir(parents=True, exist_ok=True)
    out_file = tmp_root / (benchmark.replace("/", "_") + ".out")
    command = [
        str(runner),
        benchmark,
        f"--out={out_file}",
    ]
    run_checked(command, cwd=cwd, env=env, timeout=QUERY_TIMEOUT_SECONDS)
    timings = []
    for raw in out_file.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        if raw.upper() == "ERROR":
            raise RuntimeError(f"benchmark_runner reported ERROR for {benchmark}")
        timings.append(float(raw))
    if not timings:
        raise RuntimeError(f"no timings produced for {benchmark}")
    return min(timings)


def sanitize_error_text(text: str) -> str:
    text = re.sub(r"/tmp/[A-Za-z0-9_./-]+", "<tmp>", text)
    text = re.sub(r"duckdb_[A-Za-z0-9_./-]+", "duckdb_<redacted>", text)
    text = re.sub(r"sf[0-9]+(?:_[0-9]+)?", "sf<hidden>", text)
    text = re.sub(r"q[0-9]{1,2}(?=\\.benchmark|\\b)", "q<hidden>", text)
    return text[-800:]


def query_number_from_benchmark(benchmark: str) -> int:
    match = re.search(r"q(\d{1,2})\.benchmark$", benchmark)
    if not match:
        raise ValueError(f"could not infer TPC-H query number from {benchmark}")
    query_nr = int(match.group(1))
    if query_nr < 1 or query_nr > 22:
        raise ValueError(f"TPC-H query number out of range in {benchmark}")
    return query_nr


def run_shell_tpch_timing(
    shell: Path,
    database: Path,
    query_nr: int,
    *,
    cwd: Path,
    env: dict[str, str],
    tmp_root: Path,
    repetitions: int,
) -> float:
    timings: list[float] = []
    sql = "\n".join(
        (
            settings_sql(tmp_root / f"query_{query_nr}_tmp"),
            "LOAD tpch;",
            f"PRAGMA tpch({query_nr});",
        )
    )
    for _ in range(max(1, repetitions)):
        start = time.perf_counter()
        run_duckdb_sql(shell, database, sql, cwd=cwd, env=env, timeout=QUERY_TIMEOUT_SECONDS)
        timings.append(time.perf_counter() - start)
    return min(timings)


def run_paired_shell_tpch_timing(
    vanilla_shell: Path,
    patched_shell: Path,
    vanilla_database: Path,
    patched_database: Path,
    query_nr: int,
    *,
    vanilla_source: Path,
    patched_source: Path,
    env: dict[str, str],
    tmp_root: Path,
    repetitions: int,
    patched_first: bool,
) -> tuple[float, float]:
    vanilla_timings: list[float] = []
    patched_timings: list[float] = []
    sql = "\n".join(
        (
            settings_sql(tmp_root / f"query_{query_nr}_tmp"),
            "LOAD tpch;",
            f"PRAGMA tpch({query_nr});",
        )
    )
    order = ("patched", "vanilla") if patched_first else ("vanilla", "patched")
    for _ in range(max(1, repetitions)):
        for engine in order:
            if engine == "vanilla":
                start = time.perf_counter()
                run_duckdb_sql(
                    vanilla_shell,
                    vanilla_database,
                    sql,
                    cwd=vanilla_source,
                    env=env,
                    timeout=QUERY_TIMEOUT_SECONDS,
                )
                vanilla_timings.append(time.perf_counter() - start)
            else:
                start = time.perf_counter()
                run_duckdb_sql(
                    patched_shell,
                    patched_database,
                    sql,
                    cwd=patched_source,
                    env=env,
                    timeout=QUERY_TIMEOUT_SECONDS,
                )
                patched_timings.append(time.perf_counter() - start)
        order = tuple(reversed(order))
    return min(vanilla_timings), min(patched_timings)


def load_benchmark_list() -> tuple[str, ...]:
    raw = EVALUATION_CONFIG.get("benchmarks")
    if isinstance(raw, list):
        return tuple(str(item) for item in raw)
    if isinstance(raw, str) and raw.strip():
        return tuple(item.strip() for item in raw.split(",") if item.strip())
    sf_label = str(PUBLIC_SCALE_FACTOR).replace(".", "_")
    return tuple(f"benchmark/tpch/sf{sf_label}/q{index:02d}.benchmark" for index in range(1, 23))


def benchmark_query_numbers() -> tuple[int, ...]:
    query_numbers: list[int] = []
    for benchmark in load_benchmark_list():
        query_nr = query_number_from_benchmark(benchmark)
        if query_nr not in query_numbers:
            query_numbers.append(query_nr)
    return tuple(query_numbers)


def benchmark_name(scale_factor: str, query_nr: int) -> str:
    sf_label = str(scale_factor).replace(".", "_")
    return f"benchmark/tpch/sf{sf_label}/q{query_nr:02d}.benchmark"


def geometric_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return math.exp(sum(math.log(max(value, 1e-9)) for value in values) / len(values))


def score_from_speedup(speedup: float) -> float:
    if speedup <= 0:
        return 0.0
    raw = 100.0 * math.log(speedup, 2)
    return max(0.0, min(100.0, raw))


def safe_failed_command(cmd: Any) -> str:
    if not isinstance(cmd, list) or not cmd:
        return "subprocess"
    executable = Path(str(cmd[0])).name
    if executable == "git":
        return " ".join(str(part) for part in cmd[:3])
    if executable == "cmake":
        return "cmake build"
    if executable == "unittest":
        return "duckdb sqllogictest"
    if executable in {"duckdb", "benchmark_runner"}:
        return executable
    return executable or "subprocess"


def safe_exception(exc: Exception) -> str:
    if isinstance(exc, (FileNotFoundError, RuntimeError, ValueError)):
        return sanitize_error_text(str(exc))
    return type(exc).__name__


def full_evaluation(patch_path: Path, metrics: dict[str, Any]):
    final_role = is_final_submission_role()
    metrics["submission_role"] = "final" if final_role else "agent"
    scale_factors = evaluation_scale_factors(final_role=final_role)
    benchmark_queries = benchmark_query_numbers()
    correctness_query_set = correctness_queries()
    correctness_cases = shuffled(
        [(scale_factor, query_nr) for scale_factor in scale_factors for query_nr in correctness_query_set],
        "correctness",
    )
    timing_cases = shuffled(
        [(scale_factor, query_nr) for scale_factor in scale_factors for query_nr in benchmark_queries],
        "timing",
    )
    repetitions = _config_int("FRONTIER_DUCKDB_REPETITIONS", "benchmark_repetitions", 3)
    if not DEFAULT_CLEAN_SOURCE.exists():
        metrics["full_benchmark"] = 0
        return (
            1.0,
            1.0,
            "patch policy smoke passed; DuckDB clean source is not configured in this environment",
            metrics,
        )

    with tempfile.TemporaryDirectory(prefix="duckdb_e2e_query_opt_eval_") as tmp:
        tmp_root = Path(tmp)
        env = clean_env(tmp_root)
        patched_source = DEFAULT_CLEAN_SOURCE
        restored_source = restore_prebuilt_source(patched_source, env)
        if patch_is_empty(metrics):
            metrics["used_prebuilt_empty_patch"] = 1
            if restored_source:
                metrics["rebuilt_after_source_restore"] = 1
                build_duckdb(patched_source, env, include_unittest=final_role)
            elif final_role:
                build_duckdb(patched_source, env, include_unittest=True)
        else:
            run_checked(["git", "apply", "--check", str(patch_path)], cwd=patched_source, env=env, timeout=60)
            run_checked(["git", "apply", str(patch_path)], cwd=patched_source, env=env, timeout=60)
            build_duckdb(patched_source, env, include_unittest=final_role)

        vanilla_source = DEFAULT_VANILLA_SOURCE if DEFAULT_VANILLA_SOURCE.exists() else DEFAULT_CLEAN_SOURCE
        mismatches: dict[str, str] = {}
        for scale_factor in scale_factors:
            scale_queries = tuple(query_nr for current_scale, query_nr in correctness_cases if current_scale == scale_factor)
            mismatches.update(
                check_correctness(
                    vanilla_source,
                    patched_source,
                    env=env,
                    tmp_root=tmp_root / "correctness",
                    scale_factor=scale_factor,
                    queries=scale_queries,
                )
            )
        metrics["scale_factor_count"] = len(scale_factors)
        metrics["correctness_queries"] = len(correctness_query_set)
        metrics["correctness_case_count"] = len(correctness_cases)
        if mismatches:
            metrics["correctness_mismatch_count"] = len(mismatches)
            return _invalid("patched DuckDB produced incorrect TPC-H results", metrics)

        if final_role:
            run_final_sqllogictest_smoke(patched_source, env=env, metrics=metrics)

        if USE_BENCHMARK_RUNNER:
            vanilla_runner = benchmark_runner(vanilla_source)
            patched_runner = benchmark_runner(patched_source)
            use_benchmark_runner = True
        else:
            vanilla_runner = patched_runner = None
            use_benchmark_runner = False
        bench_env = benchmark_env(env, tmp_root)
        if use_benchmark_runner:
            metrics["timing_harness"] = "benchmark_runner"
        else:
            metrics["timing_harness"] = "duckdb_shell_tpch"
            vanilla_shell = duckdb_shell(vanilla_source)
            patched_shell = duckdb_shell(patched_source)
            timing_dbs: dict[str, tuple[Path, Path]] = {}
            for scale_factor in scale_factors:
                scale_label = scale_factor.replace(".", "_")
                vanilla_timing_db = tmp_root / f"vanilla_timing_sf{scale_label}.duckdb"
                patched_timing_db = tmp_root / f"patched_timing_sf{scale_label}.duckdb"
                prepare_tpch_database(
                    vanilla_shell,
                    vanilla_timing_db,
                    cwd=vanilla_source,
                    env=env,
                    tmp_root=tmp_root / f"vanilla_timing_setup_sf{scale_label}",
                    scale_factor=scale_factor,
                )
                prepare_tpch_database(
                    patched_shell,
                    patched_timing_db,
                    cwd=patched_source,
                    env=env,
                    tmp_root=tmp_root / f"patched_timing_setup_sf{scale_label}",
                    scale_factor=scale_factor,
                )
                timing_dbs[scale_factor] = (vanilla_timing_db, patched_timing_db)

        speedups: list[float] = []
        per_query: dict[str, dict[str, float]] = {}
        for scale_factor, query_nr in timing_cases:
            benchmark = benchmark_name(scale_factor, query_nr)
            if use_benchmark_runner:
                assert vanilla_runner is not None
                assert patched_runner is not None
                vanilla_time = run_benchmark(
                    vanilla_runner,
                    benchmark,
                    cwd=vanilla_source,
                    env=bench_env,
                    tmp_root=tmp_root / "vanilla",
                )
                patched_time = run_benchmark(
                    patched_runner,
                    benchmark,
                    cwd=patched_source,
                    env=bench_env,
                    tmp_root=tmp_root / "patched",
                )
            else:
                vanilla_timing_db, patched_timing_db = timing_dbs[scale_factor]
                vanilla_time, patched_time = run_paired_shell_tpch_timing(
                    vanilla_shell,
                    patched_shell,
                    vanilla_timing_db,
                    patched_timing_db,
                    query_nr,
                    vanilla_source=vanilla_source,
                    patched_source=patched_source,
                    env=bench_env,
                    tmp_root=tmp_root / "paired_shell",
                    repetitions=repetitions,
                    patched_first=random.Random(f"{QUERY_ORDER_SEED}:pair:{scale_factor}:{query_nr}").choice(
                        [False, True]
                    ),
                )
            speedup = vanilla_time / patched_time if patched_time > 0 else 0.01
            speedups.append(max(speedup, 0.01))
            per_query[benchmark] = {
                "vanilla_time": vanilla_time,
                "patched_time": patched_time,
                "speedup": speedup,
            }

        gm_speedup = geometric_mean(speedups)
        bounded = score_from_speedup(gm_speedup)
        metrics.update(
            {
                "full_benchmark": 1,
                "benchmark_count": len(timing_cases),
                "geomean_speedup": gm_speedup,
            }
        )
        if _config_bool("FRONTIER_DUCKDB_EXPOSE_PER_QUERY", "expose_per_query_metrics", False):
            metrics["per_query"] = per_query
        return (
            bounded,
            bounded,
            f"TPC-H geomean speedup {gm_speedup:.4f}x over vanilla DuckDB",
            metrics,
        )


def evaluate(solution_path: str) -> tuple[float, float, str, dict[str, Any]]:
    patch_path = Path(solution_path)
    ok, message, metrics = validate_patch(patch_path)
    if not ok:
        return _invalid(message, metrics)
    try:
        return full_evaluation(patch_path, metrics)
    except subprocess.TimeoutExpired:
        return _invalid("DuckDB build or benchmark timed out", metrics)
    except subprocess.CalledProcessError as exc:
        failed_command = safe_failed_command(exc.cmd)
        metrics["failed_command"] = failed_command
        if failed_command.startswith("git "):
            metrics["stderr_tail"] = sanitize_error_text(exc.stderr or "")
        return _invalid("DuckDB build, patch apply, or benchmark command failed", metrics)
    except Exception as exc:
        metrics["error_type"] = type(exc).__name__
        metrics["error_detail"] = safe_exception(exc)
        return _invalid("evaluation failed", metrics)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Usage: evaluator.py SOLUTION_PATCH", file=sys.stderr)
        return 2
    score, score_unbounded, message, metrics = evaluate(argv[1])
    print(
        json.dumps(
            {
                "score": score,
                "score_unbounded": score_unbounded,
                "message": message,
                "metrics": metrics,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
