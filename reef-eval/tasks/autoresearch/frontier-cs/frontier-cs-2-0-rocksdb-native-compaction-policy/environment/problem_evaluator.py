"""Judge for the RocksDB native compaction-policy task."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


TASK_CONFIG_PATH = Path(os.environ.get("TASK_CONFIG_PATH", "/judge/task_config.json"))
DEFAULT_COMMIT = "4595a5e95ae8525c42e172a054435782b3479c57"
SOURCE_COMMIT_MARKER = ".frontier-upstream-commit"
MAX_PATCH_BYTES = 256 * 1024
MAX_CHANGED_FILES = 5

# Harbor keeps the best iterative artifact by (score, score_unbounded), and
# valid below-floor runs report small negative unbounded scores, so invalid
# submissions must carry a far lower sentinel than any valid run can reach.
INVALID_SCORE_UNBOUNDED = -1.0e6

ALLOWLIST = frozenset(
    {
        "db/compaction/compaction_picker.cc",
        "db/compaction/compaction_picker.h",
        "db/compaction/compaction_picker_level.cc",
        "db/compaction/compaction_picker_level.h",
        "db/version_set.cc",
    }
)

KNOWN_PROFILES = frozenset(
    {
        "smoke",
        "l0_pressure",
        "range_snapshot",
        "scanmix",
        "multi_cf",
        "time_series",
        "difficulty",
        "overlap_rewrite",
        "full",
    }
)

FINAL_PROFILES = (
    "l0_pressure",
    "range_snapshot",
    "scanmix",
    "multi_cf",
    "time_series",
    "difficulty",
    "overlap_rewrite",
)

DEFAULT_FINAL_SUITE_KEY = "rocksdb-native-compaction-v2-development"

STRUCTURAL_MARKERS = (
    "new file mode",
    "deleted file mode",
    "rename from",
    "rename to",
    "copy from",
    "copy to",
    "old mode ",
    "new mode ",
    "GIT binary patch",
    "Subproject commit",
)

FORBIDDEN_IDENTIFIERS = frozenset(
    {
        "getenv",
        "secure_getenv",
        "system",
        "popen",
        "fork",
        "execve",
        "dlopen",
        "dlsym",
        "readlink",
        "realpath",
        "getcwd",
        "clock_gettime",
        "rdtsc",
        "rdtscp",
        "thread_local",
        "argv",
        "argc",
        "GetEnv",
        "GetFileSystem",
        "NowMicros",
        "NowNanos",
        "NowSeconds",
        "SleepForMicroseconds",
        "GetCurrentTime",
        "getpid",
        "getppid",
        "syscall",
        "timespec_get",
        "gettimeofday",
        "getrusage",
        "uname",
        "sysinfo",
        "sched_getcpu",
        "gethostname",
        "getuid",
        "geteuid",
        "environ",
        "srand",
        "getrandom",
        "arc4random",
        "random_device",
    }
)

FORBIDDEN_CALL_IDENTIFIERS = frozenset({"clock"})

FORBIDDEN_FRAGMENTS = (
    "/proc",
    "/sys",
    "frontier",
    "harbor",
    "submission",
    "harness",
    "profile",
    "case-file",
    "std::chrono",
    "system_clock",
    "steady_clock",
    "high_resolution_clock",
    "std::random_device",
    "std::filesystem",
    "program_invocation_name",
    "ioptions_.clock",
    "mutable_db_options_.clock",
    "Env::",
    "FileSystem::",
    "std::ifstream",
    "std::ofstream",
)

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
PREPROCESSOR_RE = re.compile(r"^\s*(?:#|%:|\?\?=)\s*[A-Za-z_][A-Za-z0-9_]*")


def _load_config() -> dict[str, Any]:
    try:
        payload = json.loads(TASK_CONFIG_PATH.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        pass

    local = Path(__file__).with_name("config.yaml")
    if not local.exists():
        return {}
    cases = [
        {"seed": int(seed), "profile": profile}
        for seed, profile in re.findall(
            r"\{\s*seed:\s*(\d+)\s*,\s*profile:\s*([A-Za-z0-9_]+)\s*\}",
            local.read_text(encoding="utf-8"),
        )
    ]
    return {"evaluation": {"feedback_cases": cases}}


CONFIG = _load_config()
EVALUATION_CONFIG = (
    CONFIG.get("evaluation", {})
    if isinstance(CONFIG.get("evaluation"), dict)
    else {}
)


def _config_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        value = EVALUATION_CONFIG.get(name.lower(), default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


PINNED_COMMIT = str(EVALUATION_CONFIG.get("rocksdb_commit", DEFAULT_COMMIT))
VANILLA_DIR = Path(os.environ.get("ROCKSDB_VANILLA_DIR", "/opt/rocksdb-vanilla"))
WARM_DIR = Path(os.environ.get("ROCKSDB_WARM_DIR", "/opt/rocksdb-clean"))
HARNESS_SRC = Path(
    os.environ.get("HARNESS_SRC", "/judge/harness/lsm_policy_harness.cc")
)
WORK_DIR = Path(os.environ.get("WORK_DIR", "/work"))
BUILD_TIMEOUT = _config_int("BUILD_TIMEOUT_SECONDS", 7200)
RUN_TIMEOUT = _config_int("RUN_TIMEOUT_SECONDS", 1800)
BUILD_JOBS = _config_int("BUILD_JOBS", 3)
COMPRESS_LINK = ["-lz", "-lbz2", "-llz4", "-lzstd", "-lsnappy"]

_FINAL_CASES: tuple[tuple[int, str], ...] | None = None


@dataclass
class PatchFile:
    diff_old: str
    diff_new: str
    old_path: str = ""
    new_path: str = ""
    hunks: int = 0
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)

    @property
    def path(self) -> str:
        return self.new_path or self.diff_new


def _strip_prefix(path: str) -> str:
    path = path.strip().split("\t", 1)[0]
    if path.startswith("a/") or path.startswith("b/"):
        return path[2:]
    return path


def _unsafe_path(path: str) -> bool:
    return (
        not path
        or path.startswith("/")
        or "\\" in path
        or "\x00" in path
        or ".." in Path(path).parts
    )


def _diff_paths(line: str) -> tuple[str, str]:
    try:
        parts = shlex.split(line)
    except ValueError as exc:
        raise ValueError("malformed diff header") from exc
    if len(parts) != 4 or parts[:2] != ["diff", "--git"]:
        raise ValueError("malformed diff header")
    return _strip_prefix(parts[2]), _strip_prefix(parts[3])


def _parse_patch(text: str) -> list[PatchFile]:
    files: list[PatchFile] = []
    current: PatchFile | None = None
    for line in text.splitlines():
        if line.startswith("diff --git "):
            old, new = _diff_paths(line)
            current = PatchFile(old, new)
            files.append(current)
            continue
        if current is None:
            continue
        if line.startswith("--- "):
            current.old_path = _strip_prefix(line[4:])
        elif line.startswith("+++ "):
            current.new_path = _strip_prefix(line[4:])
        elif line.startswith("@@ "):
            current.hunks += 1
        elif line.startswith("+") and not line.startswith("+++ "):
            current.added.append(line[1:])
        elif line.startswith("-") and not line.startswith("--- "):
            current.removed.append(line[1:])
    return files


def _forbidden_added_line(line: str) -> str | None:
    if PREPROCESSOR_RE.search(line):
        return "preprocessor directive"
    lowered = line.lower()
    for fragment in FORBIDDEN_FRAGMENTS:
        if fragment.lower() in lowered:
            return fragment
    identifiers = set(TOKEN_RE.findall(line))
    for identifier in FORBIDDEN_IDENTIFIERS:
        if identifier in identifiers:
            return identifier
    for identifier in FORBIDDEN_CALL_IDENTIFIERS:
        if re.search(rf"\b{re.escape(identifier)}\s*\(", line):
            return f"{identifier}()"
    return None


def _validate_patch_text(
    text: str,
    *,
    allow_empty: bool,
) -> tuple[bool, str, dict[str, Any]]:
    metrics: dict[str, Any] = {
        "valid_patch": 0,
        "patch_bytes": len(text.encode("utf-8", errors="replace")),
    }
    if metrics["patch_bytes"] > MAX_PATCH_BYTES:
        return False, "patch exceeds 256 KiB", metrics
    if not text.strip():
        if not allow_empty:
            return False, "empty post-apply diff", metrics
        metrics.update({"valid_patch": 1, "empty_patch": 1, "changed_files": []})
        return True, "empty no-op patch", metrics
    for marker in STRUCTURAL_MARKERS:
        if marker in text:
            return False, f"forbidden structural change: {marker.strip()}", metrics

    try:
        files = _parse_patch(text)
    except ValueError as exc:
        return False, str(exc), metrics
    if not files:
        return False, "submission is not a git patch", metrics
    if len(files) > MAX_CHANGED_FILES:
        return False, "patch changes too many files", metrics

    changed: list[str] = []
    for patch_file in files:
        path = patch_file.path
        if patch_file.diff_old != patch_file.diff_new:
            return False, "renames are not allowed", metrics
        if patch_file.old_path != patch_file.diff_old:
            return False, f"old patch path mismatch for {path}", metrics
        if patch_file.new_path != patch_file.diff_new:
            return False, f"new patch path mismatch for {path}", metrics
        if _unsafe_path(path):
            return False, f"unsafe patch path: {path}", metrics
        if path not in ALLOWLIST:
            return False, f"changed file is outside the editable surface: {path}", metrics
        if patch_file.hunks == 0 or not (patch_file.added or patch_file.removed):
            return False, f"patch for {path} has no changes", metrics
        for line in patch_file.added:
            reason = _forbidden_added_line(line)
            if reason:
                return False, f"forbidden added code in {path}: {reason}", metrics
        changed.append(path)

    metrics.update(
        {
            "valid_patch": 1,
            "empty_patch": 0,
            "changed_files": sorted(set(changed)),
        }
    )
    return True, "patch accepted by static policy", metrics


def validate_patch(path: Path) -> tuple[bool, str, dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False, "solution patch is unreadable", {"valid_patch": 0}
    return _validate_patch_text(text, allow_empty=True)


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    timeout: int,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        timeout=timeout,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )


def _source_commit(source: Path) -> str:
    return source.joinpath(SOURCE_COMMIT_MARKER).read_text(encoding="utf-8").strip()


def _ensure_sources() -> None:
    for source in (VANILLA_DIR, WARM_DIR):
        if not source.joinpath(".git").is_dir():
            raise RuntimeError("pinned RocksDB source is missing")
        if _source_commit(source) != PINNED_COMMIT:
            raise RuntimeError("RocksDB source is not at the pinned commit")
        if not source.joinpath("librocksdb.a").is_file():
            raise RuntimeError("prebuilt RocksDB static library is missing")
    if not HARNESS_SRC.is_file():
        raise RuntimeError("judge harness source is missing")


def _compile_harness(source: Path, output: Path) -> None:
    _run(
        [
            os.environ.get("CXX", "g++"),
            "-O2",
            "-DNDEBUG",
            "-std=c++20",
            "-fno-rtti",
            f"-I{source / 'include'}",
            f"-I{source}",
            str(HARNESS_SRC),
            str(source / "librocksdb.a"),
            "-o",
            str(output),
            "-lpthread",
            "-ldl",
            *COMPRESS_LINK,
        ],
        timeout=600,
    )


def _restore_warm_source() -> None:
    dirty = _run(
        ["git", "diff", "--name-only", "--", *sorted(ALLOWLIST)],
        cwd=WARM_DIR,
        timeout=120,
    ).stdout.splitlines()
    if dirty:
        _run(
            ["git", "checkout", "HEAD", "--", *dirty],
            cwd=WARM_DIR,
            timeout=300,
        )


def _build_candidate(patch_path: Path) -> Path:
    _ensure_sources()
    _restore_warm_source()
    text = patch_path.read_text(encoding="utf-8", errors="replace")
    if text.strip():
        _run(["git", "apply", "--check", str(patch_path)], cwd=WARM_DIR, timeout=120)
        _run(["git", "apply", str(patch_path)], cwd=WARM_DIR, timeout=120)
        actual = _run(
            ["git", "diff", "--binary", "--no-color"],
            cwd=WARM_DIR,
            timeout=120,
        ).stdout
        ok, message, _ = _validate_patch_text(actual, allow_empty=False)
        if not ok:
            raise RuntimeError(message)

    build_env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": "/tmp",
        "LANG": "C",
        "LC_ALL": "C",
        "PORTABLE": "1",
        "DEBUG_LEVEL": "0",
        "DISABLE_WARNING_AS_ERROR": "1",
    }
    _run(
        ["make", f"-j{BUILD_JOBS}", "static_lib"],
        cwd=WARM_DIR,
        timeout=BUILD_TIMEOUT,
        env=build_env,
    )
    output = WORK_DIR / "candidate_harness"
    _compile_harness(WARM_DIR, output)
    return output


def _ensure_vanilla_harness() -> Path:
    _ensure_sources()
    output = WORK_DIR / "vanilla_harness"
    if not output.exists():
        _compile_harness(VANILLA_DIR, output)
    return output


def _harness_env() -> dict[str, str]:
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": "/nonexistent",
        "LANG": "C",
        "LC_ALL": "C",
    }


def _run_harness(
    binary: Path,
    trusted_inspector: Path,
    seed: int,
    profile: str,
) -> dict[str, Any]:
    run_id = secrets.token_hex(12)
    db_dir = WORK_DIR / f"db_{run_id}"
    output = WORK_DIR / f"result_{run_id}.json"
    trusted_output = WORK_DIR / f"trusted_{run_id}.json"
    case_file = WORK_DIR / f"case_{run_id}.txt"
    case_file.write_text(f"seed={seed}\nprofile={profile}\n", encoding="utf-8")
    try:
        started = time.monotonic()
        _run(
            [
                str(binary),
                "--db",
                str(db_dir),
                "--out",
                str(output),
                "--case-file",
                str(case_file),
            ],
            timeout=RUN_TIMEOUT,
            env=_harness_env(),
        )
        payload = json.loads(output.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("harness result is not an object")
        workload_seconds = time.monotonic() - started
        trusted_started = time.monotonic()
        _run(
            [
                str(trusted_inspector),
                "--trusted-residual",
                "--db",
                str(db_dir),
                "--out",
                str(trusted_output),
                "--seed",
                str(seed),
                "--profile",
                profile,
            ],
            timeout=RUN_TIMEOUT,
            env=_harness_env(),
        )
        trusted = json.loads(trusted_output.read_text(encoding="utf-8"))
        if not isinstance(trusted, dict) or not trusted.get("ok"):
            raise ValueError("trusted drain failed")
        for key in (
            "compaction_output_bytes",
            "final_table_bytes",
            "final_files",
            "l0_files",
            "upper_level_files",
            "upper_level_bytes",
            "passes",
            "digest_entries",
            "digest_bytes",
        ):
            payload[f"trusted_{key}"] = trusted.get(key)
        payload["_wall_seconds"] = workload_seconds
        payload["_trusted_wall_seconds"] = time.monotonic() - trusted_started
        return payload
    finally:
        shutil.rmtree(db_dir, ignore_errors=True)
        output.unlink(missing_ok=True)
        trusted_output.unlink(missing_ok=True)
        case_file.unlink(missing_ok=True)


def _number(run: dict[str, Any], key: str, *, minimum: float = 0.0) -> float:
    try:
        value = float(run[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"missing metric: {key}") from exc
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"invalid metric: {key}")
    return value


def _objective_metrics(run: dict[str, Any]) -> dict[str, float]:
    user_bytes = _number(run, "user_put_bytes", minimum=1.0)
    policy_drain_bytes = _number(run, "drain_compaction_bytes")
    residual_bytes = _number(run, "trusted_compaction_output_bytes")
    effective_drain_bytes = policy_drain_bytes + residual_bytes
    table_bytes = _number(run, "table_created_bytes")
    denominator = max(user_bytes, 64.0 * 1024 * 1024)
    return {
        "write_amp": (table_bytes + residual_bytes) / user_bytes,
        "read_amp": _number(run, "read_amp"),
        "space_amp": _number(run, "pre_drain_space_amp", minimum=1e-9),
        "debt": 1.0 + effective_drain_bytes / denominator,
    }


METRIC_WEIGHTS = {
    "write_amp": 0.40,
    "read_amp": 0.25,
    "space_amp": 0.20,
    "debt": 0.15,
}

METRIC_FLOORS = {
    "write_amp": 0.05,
    "read_amp": 0.05,
    "space_amp": 0.05,
    "debt": 0.0,
}

MIN_ROBUST_GAIN = 1.005
# Saturation anchor for 100 points, calibrated so the reference patch
# (robust_gain ~1.011 on the final suite) lands near 50 while keeping
# headroom above it before the bounded score saturates.
TARGET_ROBUST_GAIN = 1.017
PROFILE_IMPROVEMENT_GAIN = MIN_ROBUST_GAIN
PROFILE_REGRESSION_GAIN = 0.98
MIN_IMPROVED_PROFILE_FRACTION = 0.40


def _paired_case(
    vanilla_binary: Path,
    candidate_binary: Path,
    seed: int,
    profile: str,
) -> dict[str, Any]:
    ratio_samples = {name: [] for name in METRIC_WEIGHTS}
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            "vanilla": executor.submit(
                _run_harness, vanilla_binary, vanilla_binary, seed, profile
            ),
            "candidate": executor.submit(
                _run_harness, candidate_binary, vanilla_binary, seed, profile
            ),
        }
        runs = {label: future.result() for label, future in futures.items()}

    vanilla = runs["vanilla"]
    candidate = runs["candidate"]
    if not vanilla.get("ok") or not vanilla.get("correctness_ok"):
        raise RuntimeError("vanilla harness failed")
    if not candidate.get("ok") or not candidate.get("correctness_ok"):
        return {"ok": False, "reason": "candidate correctness or runtime failure"}
    if candidate.get("base_fingerprint") != vanilla.get("base_fingerprint"):
        return {"ok": False, "reason": "candidate changed the fixed base load"}

    vanilla_wall = _number(vanilla, "_wall_seconds", minimum=1e-6)
    candidate_wall = _number(candidate, "_wall_seconds", minimum=1e-6)
    if candidate_wall > vanilla_wall * 1.50 + 5.0:
        return {"ok": False, "reason": "candidate exceeded the runtime guard"}
    vanilla_stall = _number(vanilla, "stall_micros")
    candidate_stall = _number(candidate, "stall_micros")
    if candidate_stall > vanilla_stall * 1.25 + 10_000.0:
        return {"ok": False, "reason": "candidate exceeded the stall guard"}

    vanilla_metrics = _objective_metrics(vanilla)
    candidate_metrics = _objective_metrics(candidate)
    for name in METRIC_WEIGHTS:
        floor = METRIC_FLOORS[name]
        ratio = (vanilla_metrics[name] + floor) / (candidate_metrics[name] + floor)
        ratio_samples[name].append(ratio)

    ratios: dict[str, float] = {}
    weighted_log_gain = 0.0
    for name, weight in METRIC_WEIGHTS.items():
        ratio = math.exp(statistics.mean(math.log(value) for value in ratio_samples[name]))
        ratios[name] = ratio
        weighted_log_gain += weight * math.log(min(2.0, max(0.5, ratio)))
    return {
        "ok": True,
        "profile": profile,
        "gain": math.exp(weighted_log_gain),
        "ratios": ratios,
        "candidate_intra_l0": int(candidate.get("intra_l0_compactions", 0)),
        "vanilla_intra_l0": int(vanilla.get("intra_l0_compactions", 0)),
        "runtime_ratio": candidate_wall / vanilla_wall,
    }


def _score_cases(cases: list[dict[str, Any]]) -> tuple[float, float, dict[str, Any]]:
    if not cases:
        raise ValueError("no scored cases")
    logs = [math.log(float(case["gain"])) for case in cases]
    mean_log = statistics.mean(logs)
    stderr = statistics.stdev(logs) / math.sqrt(len(logs)) if len(logs) > 1 else 0.0
    robust_log = mean_log - 0.10 * stderr
    composite_gain = math.exp(mean_log)
    robust_gain = math.exp(robust_log)
    worst_gain = min(float(case["gain"]) for case in cases)
    component_floor = min(
        float(ratio) for case in cases for ratio in case["ratios"].values()
    )
    profile_logs: dict[str, list[float]] = {}
    for case, log_gain in zip(cases, logs):
        profile_logs.setdefault(str(case.get("profile", "")), []).append(log_gain)
    seed_spreads = [
        max(values) - min(values)
        for values in profile_logs.values()
        if len(values) > 1
    ]
    profile_gains = {
        profile: math.exp(statistics.mean(values))
        for profile, values in profile_logs.items()
    }
    required_improved_profiles = max(
        2,
        math.ceil(len(profile_gains) * MIN_IMPROVED_PROFILE_FRACTION),
    )
    improved_profile_count = sum(
        gain >= PROFILE_IMPROVEMENT_GAIN for gain in profile_gains.values()
    )
    material_regression_count = sum(
        gain < PROFILE_REGRESSION_GAIN for gain in profile_gains.values()
    )

    unbounded = 100.0 * (
        robust_log - math.log(MIN_ROBUST_GAIN)
    ) / math.log(TARGET_ROBUST_GAIN / MIN_ROBUST_GAIN)
    bounded = max(0.0, min(100.0, unbounded))
    if (
        improved_profile_count < required_improved_profiles
        or material_regression_count > 1
        or worst_gain < 0.75
        or component_floor <= 0.50
    ):
        bounded = 0.0
    elif component_floor < 0.67:
        bounded = min(bounded, 10.0)
    elif component_floor < 0.80:
        bounded = min(bounded, 25.0)
    elif worst_gain < 0.90:
        bounded = min(bounded, 25.0)

    metrics = {
        "composite_gain": round(composite_gain, 5),
        "robust_gain": round(robust_gain, 5),
        "worst_case_gain": round(worst_gain, 5),
        "component_floor": round(component_floor, 5),
        "profile_count": len(profile_gains),
        "improved_profile_count": improved_profile_count,
        "required_improved_profile_count": required_improved_profiles,
        "material_regression_count": material_regression_count,
        "mean_seed_log_spread": round(statistics.mean(seed_spreads), 5)
        if seed_spreads
        else 0.0,
        "max_seed_log_spread": round(max(seed_spreads), 5) if seed_spreads else 0.0,
        "case_count": len(cases),
        "intra_l0_delta": round(
            sum(
                case["candidate_intra_l0"] - case["vanilla_intra_l0"]
                for case in cases
            )
            / len(cases),
            3,
        ),
        "max_runtime_ratio": round(
            max(float(case.get("runtime_ratio", 1.0)) for case in cases), 3
        ),
    }
    return bounded, unbounded, metrics


def _feedback_cases() -> tuple[tuple[int, str], ...]:
    raw = EVALUATION_CONFIG.get("feedback_cases", [])
    cases: list[tuple[int, str]] = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            seed = int(item.get("seed", 0))
            profile = str(item.get("profile", ""))
            if seed > 0 and profile in KNOWN_PROFILES:
                cases.append((seed, profile))
    return tuple(cases) or (
        (1101, "smoke"),
        (1202, "l0_pressure"),
        (1303, "range_snapshot"),
        (1404, "scanmix"),
        (1505, "multi_cf"),
        (1606, "time_series"),
        (1707, "difficulty"),
        (1808, "overlap_rewrite"),
    )


def _generate_final_cases() -> tuple[tuple[int, str], ...]:
    suite_key = os.environ.get(
        "FRONTIER_LSM_FINAL_SUITE_KEY", DEFAULT_FINAL_SUITE_KEY
    )
    cases: list[tuple[int, str]] = []
    used_seeds: set[int] = set()
    for profile in FINAL_PROFILES:
        for replica in range(2):
            counter = 0
            while True:
                material = f"{suite_key}:{profile}:{replica}:{counter}".encode()
                seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
                seed &= (1 << 63) - 1
                seed = seed or 1
                if seed not in used_seeds:
                    break
                counter += 1
            used_seeds.add(seed)
            cases.append((seed, profile))
    return tuple(cases)


def _is_final_role() -> bool:
    return os.environ.get("FRONTIER_SUBMISSION_ROLE", "agent") == "final"


def _cases_for_role() -> tuple[tuple[int, str], ...]:
    global _FINAL_CASES
    if not _is_final_role():
        return _feedback_cases()
    if _FINAL_CASES is None:
        _FINAL_CASES = _generate_final_cases()
    return _FINAL_CASES


def _public_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "valid_patch",
        "empty_patch",
        "build_ok",
        "changed_files_count",
        "case_count",
        "composite_gain",
        "robust_gain",
        "worst_case_gain",
        "component_floor",
        "profile_count",
        "improved_profile_count",
        "required_improved_profile_count",
        "material_regression_count",
        "mean_seed_log_spread",
        "max_seed_log_spread",
        "intra_l0_delta",
        "max_runtime_ratio",
        "score_band",
    )
    return {key: metrics[key] for key in keys if key in metrics}


def _invalid(message: str, metrics: dict[str, Any] | None = None):
    payload = dict(metrics or {})
    payload.setdefault("valid_patch", 0)
    return 0.0, INVALID_SCORE_UNBOUNDED, message, _public_metrics(payload)


def _band(score: float) -> str:
    if score <= 0:
        return "baseline"
    if score < 25:
        return "measurable"
    if score < 60:
        return "broad"
    if score < 85:
        return "strong"
    return "frontier"


def full_evaluation(
    patch_path: Path,
    metrics: dict[str, Any],
) -> tuple[float, float, str, dict[str, Any]]:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    candidate = _build_candidate(patch_path)
    vanilla = _ensure_vanilla_harness()
    metrics["build_ok"] = 1

    results: list[dict[str, Any]] = []
    for seed, profile in _cases_for_role():
        result = _paired_case(vanilla, candidate, seed, profile)
        if not result.get("ok"):
            return _invalid(str(result.get("reason", "candidate run failed")), metrics)
        results.append(result)

    scored_results = [result for result in results if result["profile"] != "smoke"]
    score, score_unbounded, score_metrics = _score_cases(scored_results or results)
    metrics.update(score_metrics)
    metrics["score_band"] = _band(score)
    metrics["changed_files_count"] = len(metrics.get("changed_files", []))
    return (
        score,
        score_unbounded,
        f"scored; band={metrics['score_band']}",
        _public_metrics(metrics),
    )


def evaluate(solution_path: str) -> tuple[float, float, str, dict[str, Any]]:
    patch_path = Path(solution_path)
    ok, message, metrics = validate_patch(patch_path)
    if not ok:
        return _invalid(message, metrics)
    metrics["changed_files_count"] = len(metrics.get("changed_files", []))
    if metrics.get("empty_patch"):
        metrics["score_band"] = "baseline"
        return 0.0, 0.0, "empty no-op baseline", _public_metrics(metrics)
    try:
        return full_evaluation(patch_path, metrics)
    except subprocess.TimeoutExpired:
        return _invalid("build or benchmark timed out", metrics)
    except subprocess.CalledProcessError as exc:
        command = exc.cmd[0] if isinstance(exc.cmd, (list, tuple)) else str(exc.cmd)
        metrics["failed_command"] = Path(str(command)).name
        return _invalid("build, patch apply, or benchmark command failed", metrics)
    except Exception as exc:
        metrics["error_type"] = type(exc).__name__
        return _invalid("evaluation failed", metrics)


def prepare() -> dict[str, Any]:
    global _FINAL_CASES
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    _FINAL_CASES = _generate_final_cases()
    _ensure_vanilla_harness()
    suite_commitment = hashlib.sha256(
        json.dumps(_FINAL_CASES, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "vanilla_harness": "ready",
        "final_suite_id": str(
            EVALUATION_CONFIG.get("final_suite_id", "rocksdb-native-v1")
        ),
        "final_suite_commitment": suite_commitment,
        "final_case_count": len(_FINAL_CASES),
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: evaluator.py SOLUTION_PATCH", file=sys.stderr)
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
            sort_keys=True,
        ),
        file=sys.stderr,
    )
    print(f"{score:.12f} {score_unbounded:.12f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
