"""Official ALE-Bench public evaluator exposed through the Guidance-TTT judge protocol.

The service deliberately delegates compilation, sandboxing, timing, memory
accounting, reactive/batch interaction, and scoring to a pinned ALE-Bench
runner. The GRPO reward uses the normalization from TTT-Discover.
``scoreUnbounded`` contains the exact public raw score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import threading
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download

try:
    from .protocol import Judge, serve
except ImportError:  # Executed directly by a cluster job runner.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from protocol import Judge, serve

HERE = Path(__file__).resolve().parent
MANIFEST_PATH = HERE / "ale_bench_tasks.json"
MANIFEST = json.loads(MANIFEST_PATH.read_text())
TASKS = MANIFEST["tasks"]
ALE_BENCH_COMMIT = MANIFEST["ale_bench_commit"]
DATASET_REVISION = MANIFEST["dataset_revision"]
DATASET_REPO = "SakanaAI/ALE-Bench"
TTT_DISCOVER_CACHE_URL = "https://drive.google.com/uc?export=download&id=1bA044QSbhsQWLjgs467ygoCpoxH3NevD"


def configure_rootless_single_id() -> bool:
    """Map container root to the caller on clusters without subuid ranges.

    ALE-Bench normally passes the host UID/GID to Docker so bind-mounted files
    stay owned by the caller. A rootless Podman single-ID namespace can expose
    only container ID 0, which already maps to that same caller. This opt-in
    compatibility mode changes ownership plumbing only, not judge commands or
    resource limits.
    """
    enabled = os.environ.get("GUIDANCE_TTT_ALE_ROOTLESS_SINGLE_ID", "").strip() == "1"
    if enabled:
        os.getuid = lambda: 0  # type: ignore[method-assign]
        os.getgid = lambda: 0  # type: ignore[method-assign]
    return enabled


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_pinned_archive(problem_id: str, cache_dir: Path) -> Path:
    """Download and verify the exact official dataset snapshot for one task."""
    config = TASKS[problem_id]
    source = Path(
        hf_hub_download(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            filename=f"{problem_id}.zip",
            revision=DATASET_REVISION,
            cache_dir=cache_dir / "huggingface",
        )
    )
    actual = sha256(source)
    expected = config["archive_sha256"]
    if actual != expected:
        raise RuntimeError(f"{problem_id} archive SHA-256 mismatch: {actual} != {expected}")
    data_dir = cache_dir / "official-data"
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / f"{problem_id}.zip"
    if not target.is_file() or sha256(target) != expected:
        temporary = target.with_suffix(".zip.tmp")
        shutil.copy2(source, temporary)
        temporary.replace(target)
    os.environ["ALE_BENCH_DATA"] = str(data_dir)
    return target


def prepare_ttt_discover_suite(problem_id: str, cache_dir: Path) -> tuple[list[str], Path]:
    """Load the exact 150 inputs and tester binaries released by TTT-Discover."""
    config = TASKS[problem_id]
    archive = cache_dir / "ttt-discover" / "cache_ahc.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.is_file() or sha256(archive) != MANIFEST["ttt_discover_cache_sha256"]:
        temporary = archive.with_suffix(".zip.tmp")
        urllib.request.urlretrieve(TTT_DISCOVER_CACHE_URL, temporary)
        if sha256(temporary) != MANIFEST["ttt_discover_cache_sha256"]:
            temporary.unlink(missing_ok=True)
            raise RuntimeError("TTT-Discover AHC cache SHA-256 mismatch")
        temporary.replace(archive)

    extracted = archive.parent / "extracted"
    marker = extracted / ".complete"
    if not marker.is_file():
        temporary_dir = Path(tempfile.mkdtemp(prefix="ttt-ahc-extract-", dir=archive.parent))
        try:
            with zipfile.ZipFile(archive) as source:
                for member in source.infolist():
                    destination = (temporary_dir / member.filename).resolve()
                    if temporary_dir.resolve() not in destination.parents and destination != temporary_dir.resolve():
                        raise RuntimeError(f"unsafe archive member: {member.filename}")
                source.extractall(temporary_dir)
            (temporary_dir / ".complete").write_text(MANIFEST["ttt_discover_cache_sha256"] + "\n")
            if extracted.exists():
                shutil.rmtree(extracted)
            temporary_dir.replace(extracted)
        except Exception:
            shutil.rmtree(temporary_dir, ignore_errors=True)
            raise

    source_root = extracted / "cache"
    input_path = source_root / "public_inputs_150" / config["input_cache_file"]
    tester_path = source_root / "tester_binaries" / f"{problem_id}_tester"
    if sha256(input_path) != config["input_cache_sha256"]:
        raise RuntimeError(f"{problem_id} input cache SHA-256 mismatch")
    if sha256(tester_path) != config["tester_sha256"]:
        raise RuntimeError(f"{problem_id} tester SHA-256 mismatch")
    payload = json.loads(input_path.read_text())
    inputs = payload.get("inputs") or []
    if payload.get("problem_id") != problem_id or len(inputs) != config["public_cases"]:
        raise RuntimeError(f"{problem_id} TTT-Discover input cache metadata mismatch")

    # ALE-Bench's runner expects the official tester at this path. The binary
    # itself is copied byte-for-byte from the published TTT-Discover cache.
    data_root = Path(tempfile.mkdtemp(prefix=f"guidance-{problem_id}-"))
    target = data_root / "tools" / "target" / "release" / "tester"
    target.parent.mkdir(parents=True)
    shutil.copy2(tester_path, target)
    target.chmod(0o755)
    return [str(value) for value in inputs], data_root


class OfficialPublicEvaluator(Judge):
    """One initialized, immutable official public suite for a single AHC task."""

    def __init__(self, problem_id: str, *, case_workers: int, cache_dir: Path) -> None:
        if problem_id not in TASKS:
            raise ValueError(f"unsupported AHC task: {problem_id}")
        if case_workers < 1:
            raise ValueError("case_workers must be positive")
        self.problem_id = problem_id
        self.config = TASKS[problem_id]
        self.case_workers = case_workers
        self.rootless_single_id = configure_rootless_single_id()
        self.reuse_containers = os.environ.get("GUIDANCE_TTT_ALE_REUSE_CONTAINERS", "").strip() == "1"
        self._evaluation_lock = threading.BoundedSemaphore(
            int(os.environ.get("GUIDANCE_TTT_AHC_MAX_ACTIVE_EVALS", "8"))
        )
        # Import after ALE_BENCH_DATA is pinned. These are the public APIs and
        # result types from the exact dependency revision in requirements.
        if self.config.get("suite") == "ttt-discover-public-150":
            from types import SimpleNamespace
            from ale_bench.data import ProblemType, ScoreType

            self.inputs, self.data_root = prepare_ttt_discover_suite(problem_id, cache_dir)
            self.problem = SimpleNamespace(
                constraints=SimpleNamespace(
                    time_limit=self.config["time_limit_s"],
                    memory_limit=self.config["memory_limit_bytes"],
                ),
                metadata=SimpleNamespace(
                    problem_type=ProblemType(self.config["problem_type"]),
                    score_type=(ScoreType.MINIMIZE if self.config["score_direction"] == "min" else ScoreType.MAXIMIZE),
                ),
            )
        else:
            prepare_pinned_archive(problem_id, cache_dir)
            from ale_bench.data import build_rust_tools, load_problem
            from ale_bench.tool_wrappers import generate_inputs

            problem, seeds, _standings, _rank_map, data_root = load_problem(
                problem_id=problem_id,
                lite_version=False,
            )
            self.problem = problem
            self.data_root = data_root
            if len(seeds.public) != self.config["public_cases"]:
                raise RuntimeError(
                    f"{problem_id} public-case mismatch: {len(seeds.public)} != {self.config['public_cases']}"
                )
            constraints = problem.constraints
            metadata = problem.metadata
            actual_direction = "min" if metadata.score_type.value == "minimize" else "max"
            checks = {
                "time_limit_s": constraints.time_limit,
                "memory_limit_bytes": constraints.memory_limit,
                "problem_type": metadata.problem_type.value,
                "score_direction": actual_direction,
            }
            for name, actual in checks.items():
                if actual != self.config[name]:
                    raise RuntimeError(f"{problem_id} official metadata drift for {name}: {actual!r}")
            build_rust_tools(data_root / "tools")
            self.inputs = generate_inputs(seeds.public, {}, data_root)
        if len(self.inputs) != self.config["public_cases"]:
            raise RuntimeError("official generator returned the wrong number of public inputs")

    def __call__(self, pid: str, language: str, code: str) -> dict:
        if pid != self.problem_id:
            return self.rejected(f"judge serves {self.problem_id}, not {pid!r}")
        if language.lower() not in {"cpp", "cpp20", "c++", "c++20"}:
            return self.rejected("official AHC evaluator accepts C++20 only")

        from ale_bench.code_language import CodeLanguage, JudgeVersion
        from ale_bench.constants import ALLOW_SCORE_NON_AC_PUBLIC
        from ale_bench.result import JudgeResult, ResourceUsage, Result
        from ale_bench.tool_wrappers import run_cases

        with self._evaluation_lock:
            case_results = run_cases(
                inputs=self.inputs,
                code=code,
                code_language=CodeLanguage.CPP20,
                judge_version=JudgeVersion.V202301,
                time_limit=self.problem.constraints.time_limit,
                memory_limit=self.problem.constraints.memory_limit,
                problem_id=self.problem_id,
                problem_type=self.problem.metadata.problem_type,
                tool_dir=self.data_root,
                return_details=False,
                skip_local_visualization=True,
                num_workers=self.case_workers,
                reuse_containers=self.reuse_containers,
            )
        result = Result(
            allow_score_non_ac=self.problem_id in ALLOW_SCORE_NON_AC_PUBLIC,
            resource_usage=ResourceUsage(num_call_public_eval=1),
            case_results=case_results,
        )
        partial_reward = bool(self.config.get("partial_reward_on_invalid", False))
        # TTT-Discover sums accepted case scores even when another case fails.
        # Candidate validity remains a separate archive gate.
        raw_total = (
            sum(float(case.absolute_score) for case in case_results)
            if partial_reward
            else float(result.overall_absolute_score)
        )
        raw_score = raw_total / len(case_results) if self.config.get("score_aggregation") == "mean" else raw_total
        status = result.overall_judge_result.value
        accepted = sum(case.judge_result == JudgeResult.ACCEPTED for case in case_results)
        status_counts = Counter(case.judge_result.value for case in case_results)
        execution_times = [case.execution_time for case in case_results]
        memory_usages = [case.memory_usage for case in case_results]
        reward = self.reward(raw_score)
        valid = raw_score > 0 and accepted == len(case_results)
        return {
            "valid": valid,
            "score": reward if valid or partial_reward else 0.0,
            "score_unbounded": raw_score,
            "training_reward_on_invalid": partial_reward,
            "message": (
                f"ALE-Bench {self.problem_id}: {status}, {accepted}/{len(case_results)} AC, "
                f"raw {self.config.get('score_aggregation', 'sum')} score {raw_score:g}"
            ),
            "artifacts": {
                "suite": "ALE-Bench official public",
                "ale_bench_commit": ALE_BENCH_COMMIT,
                "dataset_revision": DATASET_REVISION,
                "problem_id": self.problem_id,
                "public_cases": len(case_results),
                "score_aggregation": self.config.get("score_aggregation", "sum"),
                "absolute_score_total": raw_total,
                "judge_result": status,
                "case_status_counts": dict(sorted(status_counts.items())),
                "max_execution_time_s": max(execution_times, default=0.0),
                "max_memory_bytes": max(memory_usages, default=0),
                "official_time_limit_s": self.problem.constraints.time_limit,
                "official_memory_limit_bytes": self.problem.constraints.memory_limit,
                "score_direction": self.config["score_direction"],
                "reward_divisor": self.config.get("reward_divisor"),
                "reward_divisor_source": self.config.get("reward_divisor_source"),
                "partial_reward_matches_ttt_discover": partial_reward,
                "atcoder_leaderboard_comparable": self.config.get("atcoder_leaderboard_comparable", True),
                "rootless_single_id_compatibility": self.rootless_single_id,
                "reuse_containers": self.reuse_containers,
            },
        }

    def reward(self, raw_score: float) -> float:
        if raw_score <= 0:
            return 0.0
        if divisor := self.config.get("reward_divisor"):
            return raw_score / float(divisor)
        reference = float(self.config["reference_score"])
        if self.config["score_direction"] == "min":
            return reference / raw_score
        return raw_score / reference

    @staticmethod
    def rejected(message: str) -> dict:
        return {"valid": False, "score": 0.0, "score_unbounded": 0, "message": message, "artifacts": {}}

    def close(self) -> None:
        shutil.rmtree(self.data_root, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--candidate-workers", type=int, default=8)
    parser.add_argument("--case-workers", type=int, default=4)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ.get("GUIDANCE_TTT_ALE_CACHE", "~/.cache/guidance-ttt-ale")).expanduser(),
    )
    args = parser.parse_args()
    if args.candidate_workers < 1 or args.case_workers < 1:
        parser.error("worker counts must be positive")
    if args.candidate_workers * args.case_workers > 32:
        parser.error("candidate-workers * case-workers must not exceed the measured-safe 32-slot evaluator budget")
    evaluator = OfficialPublicEvaluator(args.task, case_workers=args.case_workers, cache_dir=args.cache_dir)
    try:
        serve(evaluator, host=args.host, port=args.port, max_workers=args.candidate_workers)
    finally:
        evaluator.close()


if __name__ == "__main__":
    main()
