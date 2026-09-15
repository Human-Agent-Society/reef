"""A generated environment: a Harbor task written directly, and the checks that Harbor can solve it.

The Designer writes a Harbor task directly: the instruction the agent reads, the
files of the container image, the verifier under ``tests/`` and a reference solution under ``solution/``.
Any Harbor agent plays it; Harbor scores it. The checks follow the team's terminal task designer
(spare, ``scripts/terminal-rsi``): a structural gate refuses an untouched scaffold, the Dockerfile is
read the way the classic Docker parser reads it (a heredoc body is a parse error there), tasks are
deduplicated by a content hash that ignores ``task.toml`` and the hint, and ``oracle_check`` runs the
task twice through the ``harbor`` command line, once with Harbor's oracle agent (the reference solution)
and once with its nop agent (nothing), and accepts the task only when the first scores 1 and the second
scores below 1: solvable, and not for free. One gate is reef's own: the team requires a pytest file under
``tests/``, reef requires a file under ``tests/`` that names the reward file, since its verifiers are shell
scripts that write it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from recipes.beta.spade.designer import SKILL_PATTERN, HarborReply
from reef.core.tasks import HarborTask, TaskSplit, split_by_source

DEFAULT_VERIFIER_TIMEOUT_S = 300
CATEGORIES = (
    "debugging",
    "data_querying",
    "data_science",
    "data_processing",
    "file_operations",
    "scientific_computing",
    "security",
    "software_engineering",
    "system_administration",
)
MIN_INSTRUCTION_CHARS = 80
REWARD_PATHS = ("/logs/verifier/reward.txt", "/logs/verifier/reward.json")
DOCKER_INSTRUCTIONS = frozenset(
    {
        "FROM",
        "RUN",
        "CMD",
        "LABEL",
        "MAINTAINER",
        "EXPOSE",
        "ENV",
        "ADD",
        "COPY",
        "ENTRYPOINT",
        "VOLUME",
        "USER",
        "WORKDIR",
        "ARG",
        "ONBUILD",
        "STOPSIGNAL",
        "HEALTHCHECK",
        "SHELL",
    }
)
SETUP_INSTRUCTIONS = frozenset({"RUN", "COPY", "ADD", "ENV"})
HEREDOC_PATTERN = re.compile(r"<<[-~]?[ \t]*['\"]?[A-Za-z_][A-Za-z0-9_]*")
SCAFFOLD_LINES = (
    ("Dockerfile", "Install or copy over any environment dependencies here"),
    ("solution/solve.sh", "Use this file to solve the task"),
)
DEFAULT_AGENT_TIMEOUT_S = 900
EXCEPTION_CHARS = 300
ORACLE_TIMEOUT_S = 1800.0
STOP_GRACE_S = 30.0
JOBS_DIRECTORY = ".harbor-jobs"


@dataclass(frozen=True)
class GeneratedHarborTask:
    """One harbor environment as the Designer emitted it, with where it came from."""

    reply: HarborReply
    generation: int
    index: int
    source_record_id: str
    skill: str | None = None
    step: int = 0
    difficulty: str | None = None
    document_id: str | None = None
    category: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.reply, HarborReply):
            raise ValueError("reply must be a HarborReply")
        if self.category is not None and self.category not in CATEGORIES:
            raise ValueError(f"category must be one of {CATEGORIES}")
        if self.skill is not None and (not isinstance(self.skill, str) or not SKILL_PATTERN.fullmatch(self.skill)):
            raise ValueError(f"skill {self.skill!r} must match {SKILL_PATTERN.pattern}")
        for label, number in (("generation", self.generation), ("index", self.index), ("step", self.step)):
            if isinstance(number, bool) or not isinstance(number, int) or number < 0:
                raise ValueError(f"{label} must be a non-negative integer")
        if not isinstance(self.source_record_id, str) or not self.source_record_id:
            raise ValueError("source_record_id must name the Designer's generation record")
        for label, text in (("difficulty", self.difficulty), ("document_id", self.document_id)):
            if text is not None and (not isinstance(text, str) or not text):
                raise ValueError(f"{label} must be a non-empty string when set")

    @property
    def name(self) -> str:
        """The task directory name: unique per generation and index, readable by skill when there is one."""
        name = f"harbor-{self.generation:05d}-{self.index:03d}"
        return name if self.skill is None else f"{name}-{self.skill}"


@dataclass(frozen=True)
class OracleResult:
    """What Harbor made of a task: the oracle's reward, the nop agent's reward, and the verdict."""

    is_solvable: bool
    reason: str
    oracle_reward: float | None = None
    nop_reward: float | None = None


def dockerfile_parse_errors(text: str) -> list[str]:
    """Logical lines the classic Docker parser rejects: continuations joined, comments dropped, an instruction first."""
    errors: list[str] = []
    is_continued = False
    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        continues = line.endswith("\\")
        if is_continued:
            is_continued = continues
            continue
        keyword = stripped.split(None, 1)[0].upper()
        if keyword not in DOCKER_INSTRUCTIONS:
            hint = (
                " (a heredoc body: the classic parser reads it as instructions)"
                if HEREDOC_PATTERN.search(text)
                else ""
            )
            errors.append(f"{stripped[:90]}{hint}")
        is_continued = continues
    return errors


def reply_errors(reply: HarborReply) -> list[str]:
    """Why a harbor reply is not a substantive task, in the team's terms; empty when it is one."""
    errors: list[str] = []
    if len(reply.instruction.strip()) < MIN_INSTRUCTION_CHARS:
        errors.append(f"the instruction has fewer than {MIN_INSTRUCTION_CHARS} characters")
    dockerfile = reply.environment.get("Dockerfile", "")
    errors.extend(f"Dockerfile: {error}" for error in dockerfile_parse_errors(dockerfile))
    instructions = {
        line.strip().split(None, 1)[0].upper()
        for line in dockerfile.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if "FROM" not in instructions or not instructions & SETUP_INSTRUCTIONS:
        errors.append("the Dockerfile needs FROM and one of RUN, COPY, ADD or ENV")
    # The agent's container has no network, so the image must carry what Terminus 2 needs to run in it.
    if "tmux" not in dockerfile:
        errors.append("the Dockerfile does not install tmux, which the agent needs inside the container")
    solution = reply.solution.get("solve.sh", "")
    if not [line for line in solution.splitlines() if line.strip() and not line.lstrip().startswith("#")]:
        errors.append("solution/solve.sh has no command")
    tests_text = "".join(reply.tests.values())
    if not any(path in tests_text for path in REWARD_PATHS):
        errors.append("no file under tests/ names /logs/verifier/reward.txt or reward.json")
    for name, scaffold in SCAFFOLD_LINES:
        text = dockerfile if name == "Dockerfile" else solution
        if scaffold in text:
            errors.append(f"an untouched scaffold line in {name}: {scaffold!r}")
    return errors


def content_hash(task: HarborTask) -> str:
    """Sixteen hex digits over the instruction and the files, without task.toml or the hint, for deduplication."""
    digest = hashlib.sha256()
    digest.update(task.instruction.encode("utf-8"))
    for label, files in (("environment", task.environment), ("tests", task.tests), ("solution", task.solution)):
        for path in sorted(files):
            if label == "solution" and path == "hint.txt":
                continue
            digest.update(f"\0{label}/{path}\0".encode())
            digest.update(files[path].encode("utf-8"))
    return digest.hexdigest()[:16]


def harbor_task(task: GeneratedHarborTask, *, agent_timeout_s: int = DEFAULT_AGENT_TIMEOUT_S) -> HarborTask:
    """The Harbor task that holds ``task``: its files as written, the hint beside the reference solution."""
    if isinstance(agent_timeout_s, bool) or not isinstance(agent_timeout_s, int) or agent_timeout_s < 1:
        raise ValueError("agent_timeout_s must be a positive integer")
    errors = reply_errors(task.reply)
    if errors:
        raise ValueError("the reply is not a substantive task: " + "; ".join(errors))
    metadata: dict[str, object] = {
        "generation": task.generation,
        "step": task.step,
        "index": task.index,
    }
    if task.skill is not None:
        metadata["skill"] = task.skill
    if task.difficulty is not None:
        metadata["difficulty"] = task.difficulty
    if task.document_id is not None:
        metadata["document"] = task.document_id
    if task.category is not None:
        metadata["category"] = task.category
        metadata["tags"] = [tag for tag in (task.category, task.skill) if tag is not None]
    reply = task.reply
    return HarborTask(
        name=task.name,
        instruction=reply.instruction,
        tests=dict(reply.tests),
        environment=dict(reply.environment),
        config={
            "agent": {"timeout_sec": agent_timeout_s},
            # The verifier reads state a Dockerfile may have kept from the agent's user.
            "verifier": {"timeout_sec": DEFAULT_VERIFIER_TIMEOUT_S, "user": "root"},
            "environment": {"cpus": 1, "memory_mb": 2048, "storage_mb": 2048, "gpus": 0, "network_mode": "no-network"},
        },
        metadata=metadata,
        solution={**reply.solution, "hint.txt": reply.hint + "\n"},
        source_agent_record_ids=(task.source_record_id,),
    )


@dataclass(frozen=True)
class TrialOutcome:
    """One Harbor trial as its result file records it: the reward, and the exception that ended it when one did."""

    reward: float | None
    exception: str | None
    agent_note: str | None = None


def trial_outcome(jobs_path: Path) -> TrialOutcome:
    """The one trial Harbor wrote under ``jobs_path``: ``<job>/<trial>/result.json`` (the job level file has no verdict)."""
    results = sorted(jobs_path.glob("*/*/result.json"))
    if len(results) != 1:
        raise ValueError(f"expected one trial result under {jobs_path}, found {len(results)}")
    document = json.loads(results[0].read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError(f"{results[0]} is not a trial result")
    exception = document.get("exception_info")
    exception_text = None
    if isinstance(exception, Mapping):
        message = str(exception.get("exception_message") or exception.get("exception_type") or "unknown")
        # A build failure carries the whole docker log; the last lines name the cause.
        tail = [line.strip() for line in message.splitlines() if line.strip()][-3:]
        exception_text = " | ".join(tail)[-EXCEPTION_CHARS:]
    agent_note = None
    exit_code_path = results[0].parent / "agent" / "exit-code.txt"
    if exit_code_path.is_file() and exit_code_path.read_text(encoding="utf-8").strip() not in ("", "0"):
        lines = (
            [
                line
                for line in (results[0].parent / "agent" / "oracle.txt")
                .read_text(encoding="utf-8", errors="replace")
                .splitlines()
                if line.strip()
            ]
            if (results[0].parent / "agent" / "oracle.txt").is_file()
            else []
        )
        last = lines[-1].strip() if lines else "no output"
        agent_note = f"solve.sh exited {exit_code_path.read_text(encoding='utf-8').strip()}: {last}"
    verifier = document.get("verifier_result")
    rewards = verifier.get("rewards") if isinstance(verifier, Mapping) else None
    if not isinstance(rewards, Mapping) or not rewards:
        return TrialOutcome(reward=None, exception=exception_text, agent_note=agent_note)
    value = next(iter(rewards.values()))
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return TrialOutcome(reward=None, exception=exception_text, agent_note=agent_note)
    return TrialOutcome(reward=float(value), exception=exception_text, agent_note=agent_note)


def run_harbor_agent(task_path: Path, agent: str, jobs_path: Path, *, harbor: str, timeout_s: float) -> TrialOutcome:
    """Run ``agent`` on the task through the harbor command line, in its own process group, and read the trial."""
    command = [harbor, "run", "-p", str(task_path), "-a", agent, "-e", "docker", "-o", str(jobs_path), "-n", "1"]
    with subprocess.Popen(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, start_new_session=True
    ) as process:
        try:
            _, stderr = process.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            # Harbor's compose children are in the group too: a term first, so compose can take its containers down.
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=STOP_GRACE_S)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise RuntimeError(f"harbor run -a {agent} did not finish within {timeout_s:g} s") from None
    if process.returncode != 0:
        raise RuntimeError(f"harbor run -a {agent} exited {process.returncode}: {stderr.strip()[:500]}")
    return trial_outcome(jobs_path)


def oracle_check(task_path: Path, *, harbor: str | None = None, timeout_s: float = ORACLE_TIMEOUT_S) -> OracleResult:
    """Solvable, and not for free: the reference solution scores 1 and doing nothing scores below 1 under Harbor."""
    executable = harbor if harbor is not None else shutil.which("harbor")
    if executable is None:
        return OracleResult(is_solvable=False, reason="the harbor command line is not installed")
    jobs_path = task_path.parent / JOBS_DIRECTORY / task_path.name
    if jobs_path.exists():
        shutil.rmtree(jobs_path)
    try:
        oracle = run_harbor_agent(task_path, "oracle", jobs_path / "oracle", harbor=executable, timeout_s=timeout_s)
    except (RuntimeError, ValueError, OSError) as exc:
        return OracleResult(is_solvable=False, reason=str(exc)[:500])
    if oracle.reward is None or oracle.reward < 1.0:
        # A task the reference solution does not solve is refused here; the nop run would only cost a build.
        reason = f"the reference solution scored {oracle.reward}, not 1"
        if oracle.exception is not None:
            reason = f"{reason}; the trial ended with {oracle.exception}"
        if oracle.agent_note is not None:
            reason = f"{reason}; {oracle.agent_note}"
        return OracleResult(is_solvable=False, reason=reason, oracle_reward=oracle.reward)
    try:
        nop = run_harbor_agent(task_path, "nop", jobs_path / "nop", harbor=executable, timeout_s=timeout_s)
    except (RuntimeError, ValueError, OSError) as exc:
        return OracleResult(is_solvable=False, reason=str(exc)[:500], oracle_reward=oracle.reward)
    if nop.reward is None or nop.reward >= 1.0:
        reason = f"doing nothing scored {nop.reward}, not below 1"
        if nop.exception is not None:
            reason = f"{reason}; the trial ended with {nop.exception}"
        return OracleResult(is_solvable=False, reason=reason, oracle_reward=oracle.reward, nop_reward=nop.reward)
    return OracleResult(is_solvable=True, reason="", oracle_reward=oracle.reward, nop_reward=nop.reward)


def split_generation(tasks: Sequence[HarborTask], *, eval_fraction: float, seed: int) -> TaskSplit:
    """Split one generation's tasks so that every task of one Designer call lands in one split."""
    names = [task.name for task in tasks]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f"tasks share a name: {', '.join(repeated)}")
    return split_by_source(
        {task.name: task.source_agent_record_ids for task in tasks}, eval_fraction=eval_fraction, seed=seed
    )
