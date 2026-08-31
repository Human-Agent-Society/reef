"""Episode executors.

An executor turns an :class:`EpisodeSpec` into an :class:`EpisodeResult`.
The Lab does not care how. That indirection keeps the core testable
without Docker and lets the same orchestration drive containers today and
anything else later.

- :class:`HarborExecutor`: the benchmark run (containers, judge sidecar,
  verifier). Harbor is imported lazily so the rest of reef-eval works without it.
- :class:`LocalExecutor`: the development run, with the task's real judge
  as a local process and no containers.
- :class:`FakeExecutor`: deterministic, instant, dependency-free. Used by
  the test suite and the quickstart demo.

Executors recognize one reef-eval-level override, ``state_dir``: a host
directory that :class:`reef_eval.stream.Stream` carries across episodes.
Harbor mounts it into the agent's container at :data:`STATE_TARGET` and
sets ``$REEF_EVAL_STATE_DIR``; the local executor passes the host path itself.
The judge and verifier never see it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import tomllib
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from reef_eval.submissions import load_trace
from reef_eval.types import EpisodeResult, EpisodeSpec, TracePoint

logger = logging.getLogger("reef_eval")


class Executor(Protocol):
    async def execute(self, spec: EpisodeSpec) -> EpisodeResult: ...


STATE_TARGET = "/reef-eval/state"
"""Where a stream's state directory appears inside the agent's container."""


def apply_state_mount(
    agent: dict[str, Any], overrides: dict[str, Any], state_dir: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Add a stream's state directory to Harbor trial config.

    Mounts *state_dir* into the agent's container at :data:`STATE_TARGET`
    and sets ``$REEF_EVAL_STATE_DIR`` both at container startup and through the
    agent adapter. Existing mounts and env entries are preserved.
    """
    mount = {"type": "bind", "source": state_dir, "target": STATE_TARGET}
    env_cfg = dict(overrides.get("environment") or {})
    env_cfg["mounts"] = [*(env_cfg.get("mounts") or []), mount]
    env_cfg["env"] = {"REEF_EVAL_STATE_DIR": STATE_TARGET, **(env_cfg.get("env") or {})}
    agent = dict(agent)
    agent["env"] = {"REEF_EVAL_STATE_DIR": STATE_TARGET, **agent.get("env", {})}
    return agent, {**overrides, "environment": env_cfg}


class HarborExecutor:
    """Runs episodes as Harbor trials.

    ``trials_dir`` is where Harbor writes trial directories (one per episode);
    each stored row's ``uri`` points into it. ``agent`` dicts use Harbor's
    ``AgentConfig`` field names verbatim; ``spec.overrides`` maps onto
    ``TrialConfig`` fields (e.g. ``{"verifier": {"disable": True}}`` or
    ``{"timeout_multiplier": 2.0}``).
    """

    def __init__(self, trials_dir: Path | str):
        self.trials_dir = Path(trials_dir)

    async def execute(self, spec: EpisodeSpec) -> EpisodeResult:
        # Lazy import: Harbor's dependency tree is heavy and optional.
        from harbor.models.trial.config import TaskConfig, TrialConfig
        from harbor.trial.trial import Trial

        missing = _unset_compose_vars(Path(spec.task))
        if missing:
            return EpisodeResult(
                rewards={},
                error=(
                    f"{spec.task} needs {', '.join(missing)}, which nothing "
                    "provides. Its docker-compose.yaml builds mount paths from "
                    "them, and Compose reads an unset variable as an empty "
                    "string, so the task would run and score 0 with nothing to "
                    "show for it. Set them in the environment or in an .env "
                    "beside the compose file; the benchmark's README says "
                    "which, and for frontier-cs `python fetch.py --problems "
                    "<id>` does it for you."
                ),
            )

        task_field = (
            {"path": Path(spec.task)}
            if Path(spec.task).exists()
            else {"name": spec.task}
        )
        agent, overrides = dict(spec.agent), dict(spec.overrides)
        state_dir = overrides.pop("state_dir", None)
        if state_dir:
            Path(state_dir).mkdir(parents=True, exist_ok=True)
            agent, overrides = apply_state_mount(agent, overrides, str(state_dir))
        config = TrialConfig.model_validate(
            {
                "task": TaskConfig.model_validate(task_field).model_dump(),
                "trials_dir": self.trials_dir,
                "agent": agent,
                **overrides,
            }
        )

        trial = await Trial.create(config)
        result = await trial.run()

        rewards = result.verifier_result.rewards if result.verifier_result else {}
        error = (
            result.exception_info.exception_message
            if result.exception_info is not None
            else None
        )
        trace = tuple(load_trace(trial.paths.trial_dir))
        return EpisodeResult(
            rewards=dict(rewards),
            uri=result.trial_uri,
            trace=trace,
            error=error,
            usage=_harbor_usage(result.agent_result, len(trace)),
            agent_exit_code=_agent_exit_code(trial.paths.trial_dir),
        )


def _unset_compose_vars(task_dir: Path) -> list[str]:
    """Compose variables the task needs that nothing is going to supply.

    Harbor fills in its own set and the rest have to come from the
    environment. An unset one is not an error to Compose: it substitutes an
    empty string, the mount quietly points somewhere else, and the run ends
    at a score of 0 that reads like the agent's fault.
    """
    compose = task_dir / "environment" / "docker-compose.yaml"
    if not compose.is_file():
        return []
    try:
        from harbor.environments.docker.compose_env import ComposeInfraEnvVars

        provided = {name.upper() for name in ComposeInfraEnvVars.model_fields}
    except ImportError:  # pragma: no cover - harbor is optional
        provided = set()
    provided |= set(os.environ) | _dotenv_names(compose.parent)
    referenced = set(re.findall(r"\$\{([A-Z_][A-Z0-9_]*)[:}]", compose.read_text()))
    return sorted(referenced - provided)


def _dotenv_names(environment_dir: Path) -> set[str]:
    """Names an ``.env`` beside the compose file defines.

    Harbor points Compose at this directory, and Compose reads a ``.env``
    there, so a variable set in one is as good as an exported one and the
    check has to look in the same places Compose does.
    """
    dotenv = environment_dir / ".env"
    if not dotenv.is_file():
        return set()
    names = set()
    for line in dotenv.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            names.add(line.split("=", 1)[0].strip())
    return names


def _agent_exit_code(trial_dir: Path) -> int | None:
    """The agent's exit status, when Harbor recorded a non-zero one.

    Harbor writes ``agent/exit-code.txt`` only on a non-zero exit, and its
    trial result carries no other sign of it, so an agent that died leaves a
    perfectly ordinary row unless this is read back.
    """
    path = trial_dir / "agent" / "exit-code.txt"
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _harbor_usage(agent_result: Any, n_submissions: int) -> dict[str, float]:
    """Pull the harness's own token accounting off Harbor's AgentContext.

    These numbers come from each adapter parsing its harness's usage report
    (claude-code, codex, or qwen trajectories), so they are the counts the
    harness reported for itself. Absent fields are omitted.
    """
    usage: dict[str, float] = {"n_submissions": float(n_submissions)}
    if agent_result is None:
        return usage
    fields = {
        "n_input_tokens": "n_input_tokens",
        "n_output_tokens": "n_output_tokens",
        "n_cache_tokens": "n_cache_tokens",
    }
    for out_key, attr in fields.items():
        value = getattr(agent_result, attr, None)
        if value:
            usage[out_key] = float(value)
    tok = usage.get("n_input_tokens", 0) + usage.get("n_output_tokens", 0)
    if tok:
        usage["n_total_tokens"] = tok
    return usage


class LocalExecutor:
    """Runs episodes on this machine against the task's real judge, no Docker.

    The contract is identical to the container run: the same
    ``judge_server.py`` that runs as a sidecar in containers is started as a
    local process, your command gets ``$JUDGE_URL`` (and ``$BUDGET_SEC``)
    and submits solutions to it, and the final result is the judge's
    grade. Being killed at the deadline is a normal ending.

    Fast and dependency-free, for the development loop. Local rows carry a
    ``local://`` uri because the judge ran on a machine the agent also
    controls; report container numbers. Works for any task following the
    template layout (``environment/judge_server.py`` and ``score.py``);
    image-based tasks (EdgeBench, FrontierCS) need containers.
    """

    def __init__(self, root: Path | str | None = None):
        self.root = Path(root) if root else None
        if self.root:
            self.root.mkdir(parents=True, exist_ok=True)

    async def execute(self, spec: EpisodeSpec) -> EpisodeResult:
        task_dir = Path(spec.task)
        judge_dir = task_dir / "environment"
        if not (
            (judge_dir / "judge_server.py").is_file()
            and (judge_dir / "score.py").is_file()
        ):
            return EpisodeResult(
                rewards={},
                error=f"{spec.task} does not follow the template layout "
                "(environment/judge_server.py + score.py); run it in containers",
            )
        command = spec.agent.get("command")
        if not command:
            return EpisodeResult(
                rewards={},
                error='local runs need agent={"command": "..."} '
                "(the CLI flag is --command)",
            )

        budget = spec.agent.get("override_timeout_sec")
        if budget is None:
            config = tomllib.loads((task_dir / "task.toml").read_text())
            budget = config.get("agent", {}).get("timeout_sec", 600.0)

        state_env: dict[str, str] = {}
        state_dir = spec.overrides.get("state_dir")
        if state_dir:
            Path(state_dir).mkdir(parents=True, exist_ok=True)
            state_env["REEF_EVAL_STATE_DIR"] = str(state_dir)

        workdir = Path(tempfile.mkdtemp(prefix=f"{task_dir.name}-", dir=self.root))
        data_dir = workdir / "judge_data"
        judge_log = workdir / "judge.log"
        judge, judge_url, verifier_url = await asyncio.to_thread(
            _start_judge, judge_dir, data_dir, judge_log
        )
        try:
            error = None
            try:
                proc = await asyncio.to_thread(
                    subprocess.run,
                    command,
                    shell=True,
                    env={
                        **os.environ,
                        "JUDGE_URL": judge_url,
                        "BUDGET_SEC": str(budget),
                        **state_env,  # a stream's carried state, if any
                        # Budget signals (REEF_EVAL_MAX_TOKENS, ...) the command may pace on.
                        **spec.agent.get("env", {}),
                    },
                    timeout=float(budget),
                    capture_output=True,
                    text=True,
                )
                if proc.returncode != 0:
                    error = (
                        f"agent command exited {proc.returncode}: {proc.stderr[-500:]}"
                    )
            except subprocess.TimeoutExpired:
                pass  # budget spent: a normal ending

            token = (data_dir / ".verifier_token").read_text()
            final = await asyncio.to_thread(_get_json, f"{verifier_url}/final", token)
        finally:
            judge.kill()
            judge.wait()

        submissions = final.get("submissions", [])
        with (workdir / "submissions.jsonl").open("w") as f:
            for entry in submissions:
                f.write(json.dumps({"t": entry["t"], "score": entry["score"]}) + "\n")
        return EpisodeResult(
            rewards={"reward": float(final["reward"])},
            uri=f"local://{workdir}",
            trace=tuple(TracePoint(t=e["t"], score=e["score"]) for e in submissions),
            error=error,
            usage={"n_submissions": float(len(submissions))},
        )


def _start_judge(
    judge_dir: Path, data_dir: Path, log: Path, attempts: int = 3
) -> tuple[subprocess.Popen, str, str]:
    """Start the judge and wait for both its ports, retrying a lost race.

    Ports are chosen, released, and only then bound by the judge, so another
    process on the machine can take one in between. That shows up as the
    judge exiting on a bind error, which is worth retrying with fresh ports;
    any other startup failure is raised on the first attempt.
    """
    last: RuntimeError | None = None
    for attempt in range(attempts):
        port, verifier_port = _free_ports(2)
        with log.open("w") as stderr:
            judge = subprocess.Popen(
                [sys.executable, str(judge_dir / "judge_server.py")],
                env={
                    **os.environ,
                    "PORT": str(port),
                    "VERIFIER_PORT": str(verifier_port),
                    "JUDGE_DIR": str(judge_dir),
                    "DATA_DIR": str(data_dir),
                },
                stdout=subprocess.DEVNULL,
                stderr=stderr,
            )
        judge_url = f"http://127.0.0.1:{port}"
        verifier_url = f"http://127.0.0.1:{verifier_port}"
        try:
            _wait_healthy(judge_url, judge, log)
            _wait_healthy(verifier_url, judge, log)
            return judge, judge_url, verifier_url
        except RuntimeError as exc:
            judge.kill()
            judge.wait()
            last = exc
            if "address already in use" not in str(exc).lower():
                raise
            logger.warning("judge lost the port race, retry %d", attempt + 1)
    raise RuntimeError(f"judge could not get a free port in {attempts} tries") from last


def _free_ports(count: int) -> list[int]:
    """Pick *count* distinct free loopback ports.

    Every socket is held until all of them are bound, so no two ports can
    come back equal. Binding one at a time and releasing it can hand out the
    same port twice, and the judge needs its two ports to differ.
    """
    held: list[socket.socket] = []
    try:
        for _ in range(count):
            s = socket.socket()
            s.bind(("127.0.0.1", 0))
            held.append(s)
        return [s.getsockname()[1] for s in held]
    finally:
        for s in held:
            s.close()


def _wait_healthy(
    judge_url: str,
    process: subprocess.Popen | None = None,
    log: Path | None = None,
    timeout_sec: float = 10.0,
) -> None:
    """Block until the judge answers /health.

    A judge that fails at startup would otherwise show up as ten seconds of
    silence, so *process* is polled for an early exit and *log* is quoted in
    either failure. Both are optional to keep the function usable on a server
    this process did not start.
    """

    def _tail() -> str:
        if log is None or not log.is_file():
            return "no judge log"
        text = log.read_text().strip()
        return text[-800:] if text else "judge log is empty"

    deadline = time.monotonic() + timeout_sec
    while True:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"judge exited with code {process.returncode} before serving "
                f"{judge_url}: {_tail()}"
            )
        try:
            with urllib.request.urlopen(f"{judge_url}/health", timeout=2):
                return
        except Exception as e:
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"judge at {judge_url} never became healthy in "
                    f"{timeout_sec:g}s: {_tail()}"
                ) from e
            time.sleep(0.05)


def _get_json(url: str, token: str | None = None) -> dict:
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


class FakeExecutor:
    """Deterministic executor for tests and demos.

    ``score`` maps a spec to rewards; the default scores by a stable hash of
    the task name so demos produce varied but reproducible numbers. ``trace``
    optionally fabricates a submission log per spec.
    """

    def __init__(
        self,
        score: Callable[[EpisodeSpec], dict[str, Any]] | None = None,
        trace: Callable[[EpisodeSpec], list] | None = None,
    ):
        self._score = score or self._default_score
        self._trace = trace
        self.calls: list[EpisodeSpec] = []

    async def execute(self, spec: EpisodeSpec) -> EpisodeResult:
        self.calls.append(spec)
        return EpisodeResult(
            rewards=self._score(spec),
            uri=f"fake://{spec.task}",
            trace=tuple(self._trace(spec)) if self._trace else (),
        )

    @staticmethod
    def _default_score(spec: EpisodeSpec) -> dict[str, float]:
        digest = sum(spec.task.encode()) % 100
        return {"reward": round(digest / 100, 2)}
