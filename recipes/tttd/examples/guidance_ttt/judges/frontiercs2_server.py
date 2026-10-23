"""Expose FrontierCS 2 task verifiers through the judge protocol.

A FrontierCS 2 task accepts a patch against a pristine repository and scores it
with a pinned Verifier image. The candidate here is the complete text of the
task's one editable file. The adapter turns it into that patch, packs the
payload tar the platform expects, and runs the task's own verifier once in
``train`` mode on the development suite. ``pid`` names the task;
``<task>:eval`` runs the held-out final suite instead.

Only what a FrontierCS 2 agent may read comes back: the train score and the
task-approved metrics. Reference values and per-workload decisions stay in the
verifier's evidence file. In ``eval`` mode the private verdict is returned as
``beats_reference``; only the final Harbor verifier asks for it.

Each evaluation runs ``apptainer exec`` on its own CPU set, sized by the
task's ``verifier.cpu_millis``, under the task's evaluation timeout.
"""

import argparse
import difflib
import json
import os
import queue
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

if __package__:
    from .protocol import Judge, serve
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from protocol import Judge, serve

TASKS_FILE = Path(__file__).resolve().parent / "frontiercs2_tasks.json"


@dataclass(frozen=True)
class Task:
    name: str
    directory: Path
    editable_file: str
    language: str
    image: Path
    cpus: int
    timeout_s: float

    @property
    def pristine(self) -> str:
        return (self.directory / "agent-base/frontier/task/repo" / self.editable_file).read_text()


def load_tasks(problems: Path, images: Path, names: list[str] | None = None) -> dict[str, Task]:
    tasks = {}
    for name, spec in json.loads(TASKS_FILE.read_text()).items():
        if names and name not in names:
            continue
        matches = sorted(problems.glob(f"*/{name}/policy.json"))
        if len(matches) != 1:
            raise FileNotFoundError(f"expected one FrontierCS 2 task directory for {name} under {problems}")
        policy = json.loads(matches[0].read_text())
        allowed = policy["metadata"]["runtime"]["allowed_paths"]
        if not any(spec["editable_file"] == path or spec["editable_file"].startswith(path) for path in allowed):
            raise ValueError(f"{name}: {spec['editable_file']} is outside the task's patch policy {allowed}")
        tasks[name] = Task(
            name=name,
            directory=matches[0].parent,
            editable_file=spec["editable_file"],
            language=spec["language"],
            image=images / f"{name}.sif",
            cpus=max(1, int(policy["verifier"]["cpu_millis"]) // 1000),
            timeout_s=float(policy["quota"]["eval_timeout_seconds"]),
        )
        if not tasks[name].image.is_file():
            raise FileNotFoundError(f"missing verifier image {tasks[name].image}")
    return tasks


def _normalized(info: tarfile.TarInfo) -> tarfile.TarInfo:
    """The fixed owner and time of the platform's ``tar --owner=0 --mtime=1970``."""
    info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, "", ""
    return info


def make_patch(task: Task, code: str) -> str:
    """A git-style diff of the one editable file, as the task's make_submission.sh writes it."""
    before = task.pristine.splitlines(keepends=True)
    after = (code if code.endswith("\n") else code + "\n").splitlines(keepends=True)
    path = task.editable_file
    body = "".join(difflib.unified_diff(before, after, f"a/{path}", f"b/{path}"))
    if not body:
        # The platform rejects an empty patch, so the unchanged file is sent as a one-comment change.
        marker = "// unchanged\n" if task.language == "cpp" else "# unchanged\n"
        body = "".join(difflib.unified_diff(before, [*before, marker], f"a/{path}", f"b/{path}"))
    return f"diff --git a/{path} b/{path}\n{body}"


class FrontierCS2Judge(Judge):
    def __init__(self, tasks: dict[str, Task], *, workers: int, work_dir: Path) -> None:
        self.tasks = tasks
        self.work_dir = work_dir
        cores = sorted(os.sched_getaffinity(0))
        need = max(task.cpus for task in tasks.values())
        if len(cores) < workers * need:
            raise ValueError(
                f"{workers} workers of {need} CPUs need {workers * need} CPUs; {len(cores)} are available"
            )
        self.cpu_sets: queue.Queue[list[int]] = queue.Queue()
        for index in range(workers):
            self.cpu_sets.put(cores[index * need : (index + 1) * need])

    def __call__(self, pid: str, language: str, code: str) -> dict:
        name, _, mode = pid.partition(":")
        mode = mode or "train"
        task = self.tasks.get(name)
        if task is None or mode not in ("train", "eval"):
            raise ValueError(f"unknown FrontierCS 2 task or mode: {pid!r}")
        if language != task.language:
            return {"valid": False, "score": 0, "message": f"expected a {task.language} candidate"}
        patch = make_patch(task, code)
        cpus = self.cpu_sets.get()
        try:
            return self._verify(task, mode, patch, cpus[: task.cpus])
        finally:
            self.cpu_sets.put(cpus)

    def _verify(self, task: Task, mode: str, patch: str, cpus: list[int]) -> dict:
        root = Path(tempfile.mkdtemp(prefix=f"fcs2-{task.name}-", dir=self.work_dir))
        try:
            (root / "in/payload").mkdir(parents=True)
            (root / "out").mkdir()
            (root / "scratch").mkdir()
            (root / "home").mkdir()
            (root / "in/payload/solution.patch").write_text(patch)
            with tarfile.open(root / "in/payload.tar", "w", format=tarfile.USTAR_FORMAT) as archive:
                archive.add(root / "in/payload", arcname="payload", filter=_normalized)
            dataset = "eval" if mode == "eval" else "train"
            shutil.copyfile(task.directory / f"datasets/{dataset}.json", root / "in/dataset.json")
            env = {
                "FRONTIER_PAYLOAD_PATH": "/in/payload.tar",
                "FRONTIER_DATASET_PATH": "/in/dataset.json",
                "FRONTIER_VERIFICATION_MODE": mode,
                "FRONTIER_RESULT_PATH": "/out/result.json",
                "FRONTIER_EVIDENCE_PATH": "/out/evidence.tar.zst",
                "FRONTIER_VERIFIER_SCRATCH": "/scratch",
                "FRONTIER_MAX_EVIDENCE_BYTES": str(64 << 20),
                "TMPDIR": "/scratch",
            }
            command = [
                "taskset", "-c", ",".join(map(str, cpus)),
                "apptainer", "exec", "--containall", "--cleanenv", "--no-home",
                "-B", f"{root / 'in'}:/in:ro", "-B", f"{root / 'out'}:/out", "-B", f"{root / 'scratch'}:/scratch",
                "-B", f"{root / 'home'}:/frontier",
                "--pwd", "/frontier",
                *[arg for key, value in env.items() for arg in ("--env", f"{key}={value}")],
                str(task.image), "/opt/frontier/verifier",
            ]  # fmt: skip
            started = time.monotonic()
            try:
                process = subprocess.run(command, capture_output=True, text=True, timeout=task.timeout_s)
            except subprocess.TimeoutExpired:
                return {"valid": False, "score": 0, "message": f"verification exceeded {task.timeout_s:g} seconds"}
            elapsed = time.monotonic() - started
            result_path = root / "out/result.json"
            if not result_path.is_file():
                raise RuntimeError(f"{task.name} verifier wrote no result: {process.stderr[-2000:]}")
            return self._read(task, mode, json.loads(result_path.read_text()), elapsed)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    @staticmethod
    def _read(task: Task, mode: str, result: dict, elapsed: float) -> dict:
        failure = result.get("failure") or {}
        if result.get("status") != "passed":
            # The adapter's own configuration errors are infrastructure; everything else is the candidate's.
            if str(failure.get("code", "")).startswith("verifier_"):
                raise RuntimeError(f"{task.name} verifier failed: {failure}")
            return {"valid": False, "score": 0, "message": str(result.get("summary") or "rejected")}
        metrics = result.get("metrics") or {}
        score = float(result.get("raw_score") or 0.0)
        # The task's score already counts every failed workload as zero; a zero score has no correct workload.
        valid = score > 0
        message = str(result.get("summary") or "")
        if metrics:
            message += "\nDevelopment metrics: " + json.dumps(metrics, sort_keys=True, separators=(",", ":"))
        payload = {
            "valid": valid,
            "score": score if valid else 0,
            "score_unbounded": float(metrics.get("score_unbounded", score)) if valid else 0,
            "message": message if valid else "the candidate scored zero\n" + message,
            "artifacts": {"frontiercs2_mode": mode, "verifier_seconds": round(elapsed, 1)},
        }
        if mode == "eval":
            payload["artifacts"]["beats_reference"] = (result.get("private_metrics") or {}).get("beats_reference")
        return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--problems", type=Path, required=True, help="FrontierCS-2 tasks/frontier-cs-2.0-demo/problems"
    )
    parser.add_argument("--images", type=Path, required=True, help="directory of <task>.sif verifier images")
    parser.add_argument("--work-dir", type=Path, default=Path(tempfile.gettempdir()))
    parser.add_argument("--tasks", nargs="*", help="serve only these tasks (default: all)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8082)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(args.problems, args.images, args.tasks)
    judge = FrontierCS2Judge(tasks, workers=args.workers, work_dir=args.work_dir)
    serve(judge, host=args.host, port=args.port, max_workers=args.workers)


if __name__ == "__main__":
    main()
