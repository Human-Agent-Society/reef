"""Harbor agent that runs one CEO-Bench episode with the agent role served by Reef.

The Harbor environment holds a pinned CEO-Bench checkout
(``harbor/environment/Dockerfile``). ``run()`` starts a reef-client sidecar on
this host that stamps the scenario and token onto every model call and keeps
each exchange with its receipt, then runs the benchmark's own bash-agent
runner inside the container with the agent role pointed at that sidecar over
``/v1/chat/completions``. The two simulator roles keep the benchmark's provider
settings and never pass through Reef.

The reward is online and weekly. Every request the agent sends carries the
dashboard of the simulated week it is in (``=== Week N Dashboard (Day D) ===``
with the week's opening cash), so the sidecar's captures say which week each
turn belongs to. While the episode runs, a reporter thread watches for the
next week's dashboard; when it appears the previous week is over, and every
turn of that week is reported with the week's cash change as its score
(``harness.report``). The last week ends with the verifier's final cash,
which a watcher thread reads from Harbor's ``result.json`` after the trial.
When the episode ends, the run directory (``world.nmdb``, config, checkpoint,
logs) is copied into the trial's log directory and the receipts, with their
weeks and token counts, go into the agent context in call order.

Connection settings come from the environment set by ``run.sh``:

- ``REEF_SERVICE_URL`` (required): the Reef service as the task container
  reaches it, so a LAN address rather than ``127.0.0.1``
- ``REEF_SCENARIO``, ``REEF_TOKEN``, ``REEF_TIMEOUT_S``

Variables named ``SAAS_BENCH_*``, ``OPENAI_*``, ``ANTHROPIC_*``, and ``AWS_*``
are forwarded into the container for the simulator roles, so their credentials
and any provider override stay outside the repository.

``CEOBENCH_TRAIN_MAX_TOKENS`` (0 or unset: no limit) is the trainer's window:
the engine serves the model's full context, but a turn whose prompt and
completion together exceed this many tokens is recorded and never reported,
because the trainer could not hold it.
"""

import atexit
import json
import os
import re
import shlex
import threading
import time
from http.server import ThreadingHTTPServer
from urllib.parse import urlsplit

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from reef_client import ReefClient
from reef_client.serve import CaptureStore, ServeConfig, build_handler

from .report import post_week_reports

#: The pinned checkout and the run root inside the task container.
CEOBENCH_DIR = "/opt/ceobench"
RUNS_DIR = "/workspace/ceobench-runs"
#: Episode defaults; ``kwargs`` on the Harbor agent config override them.
DEFAULT_SEED = 42
DEFAULT_DAYS = 500
#: Environment variables the simulator roles read; forwarded verbatim.
FORWARDED_ENV_PREFIXES = ("SAAS_BENCH_", "OPENAI_", "ANTHROPIC_", "AWS_")
#: How often the reporter looks for a finished week while the episode runs.
WEEK_POLL_S = 5.0
#: The weekly dashboard header the benchmark's engine returns, with the
#: week's opening cash on the line after it.
DASHBOARD_RE = re.compile(r"=== Week (\d+) Dashboard \(Day (\d+)\) ===\s*\n\s*\nCash: (-?)\$(-?[\d,]+)")


def runner_command(base_url: str, model: str, seed: int, days: int) -> str:
    """The benchmark's bash-agent baseline, with the agent role at ``base_url``."""
    args = [
        "uv",
        "run",
        "--no-sync",
        "python",
        "-m",
        "saas_bench.agents.bash_agent.run_test",
        "--provider",
        "openai",
        "--base-url",
        base_url,
        "--api-key",
        "reef",  # the sidecar replaces it with the Reef token
        "--model",
        model,
        "--reasoning-effort",
        "none",
        "--seed",
        str(seed),
        "--days",
        str(days),
        "--workspace",
        RUNS_DIR,
    ]
    return f"mkdir -p {RUNS_DIR} && cd {CEOBENCH_DIR} && {shlex.join(args)} > {RUNS_DIR}/runner.log 2>&1"


def turn_tokens(turn: dict) -> int:
    """Prompt plus completion tokens of one captured turn (0 when unreported)."""
    usage = (turn.get("response") or {}).get("usage") or {}
    return int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0)


def turn_week(turn: dict) -> tuple[int, int, float] | None:
    """``(week, day, opening cash)`` of the latest dashboard in the turn's request.

    The runner rebuilds the conversation from the new dashboard after every
    ``next-week``; taking the latest header also covers a transcript that
    still carries an earlier week's dashboard. ``None`` when the request has
    no dashboard at all.
    """
    latest: tuple[int, int, float] | None = None
    for message in (turn.get("request") or {}).get("messages") or []:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            continue
        for match in DASHBOARD_RE.finditer(content):
            week, day = int(match.group(1)), int(match.group(2))
            sign = -1.0 if match.group(3) == "-" else 1.0
            cash = sign * float(match.group(4).replace(",", ""))
            if latest is None or week > latest[0]:
                latest = (week, day, cash)
    return latest


def forwarded_environment(environ: dict[str, str]) -> dict[str, str]:
    forwarded = {key: value for key, value in environ.items() if key.startswith(FORWARDED_ENV_PREFIXES)}
    # Reef serves /v1/chat/completions, not the Responses API the runner
    # prefers for OpenAI-compatible endpoints (see reef.patch).
    forwarded["SAAS_BENCH_OPENAI_CHAT_COMPLETIONS"] = "1"
    return forwarded


class WeekLedger:
    """The episode's turns grouped by simulated week, in call order.

    ``observe`` reads the sidecar's captures; a served turn without a
    dashboard of its own belongs to the week the previous turn was in.
    """

    def __init__(self) -> None:
        self.weeks: dict[int, dict] = {}
        self.turns: list[dict] = []  # {"receipt", "tokens", "week"} per served turn
        self.posted: set[int] = set()

    def observe(self, captured: list[dict]) -> None:
        current: int | None = None
        self.weeks = {}
        self.turns = []
        for turn in captured:
            if turn.get("status") != 200 or not turn.get("receipt"):
                continue
            seen = turn_week(turn)
            if seen is not None:
                week, day, cash = seen
                self.weeks.setdefault(week, {"day": day, "cash_start": cash, "turns": []})
                current = week
            record = {"receipt": turn["receipt"], "tokens": turn_tokens(turn), "week": current}
            self.turns.append(record)
            if current is not None:
                self.weeks[current]["turns"].append((record["receipt"], record["tokens"]))

    def finished_weeks(self, final_cash: float | None = None) -> list[tuple[int, float]]:
        """Unreported weeks with a known closing cash, as ``(week, cash_end)``.

        A week closes with the opening cash of the next week seen; the last
        week closes with ``final_cash`` when the caller has it.
        """
        ordered = sorted(self.weeks)
        finished = []
        for position, week in enumerate(ordered):
            if week in self.posted:
                continue
            if position + 1 < len(ordered):
                finished.append((week, self.weeks[ordered[position + 1]]["cash_start"]))
            elif final_cash is not None:
                finished.append((week, final_cash))
        return finished

    def summary(self, final_cash: float | None = None) -> list[dict]:
        ordered = sorted(self.weeks)
        rows = []
        for position, week in enumerate(ordered):
            entry = self.weeks[week]
            if position + 1 < len(ordered):
                cash_end: float | None = self.weeks[ordered[position + 1]]["cash_start"]
            else:
                cash_end = final_cash
            rows.append(
                {
                    "week": week,
                    "day": entry["day"],
                    "cash_start": entry["cash_start"],
                    "cash_end": cash_end,
                    "turns": len(entry["turns"]),
                    "reported": week in self.posted,
                }
            )
        return rows


class HarborAgent(BaseAgent):
    """One Harbor trial, one CEO-Bench episode through Reef."""

    def __init__(self, *args, seed: int = DEFAULT_SEED, days: int = DEFAULT_DAYS, **kwargs):
        super().__init__(*args, **kwargs)
        self._service_url = os.environ.get("REEF_SERVICE_URL", "").rstrip("/")
        if not self._service_url:
            raise ValueError("the ceobench harness requires REEF_SERVICE_URL")
        self._scenario = os.environ.get("REEF_SCENARIO", "ceobench-sao")
        self._token = os.environ.get("REEF_TOKEN", "reef-local")
        self._seed = int(seed)
        self._days = int(days)
        self._client = ReefClient(
            self._service_url, token=self._token, timeout_s=float(os.environ.get("REEF_TIMEOUT_S", "7200"))
        )
        self._init_week_reporting()
        # Harbor runs the verifier after run() returns and ends the trial by
        # writing result.json; watch for it from construction, so the last
        # week's reward still reaches Reef when the episode itself failed late.
        self._report_watch_from = time.time()
        self._reporter = threading.Thread(target=self._report_trial_result, daemon=True)
        self._reporter.start()
        atexit.register(self._reporter.join, 120.0)  # hundreds of reports; don't drop them

    def _init_week_reporting(self) -> None:
        self._capture = CaptureStore()
        self._ledger = WeekLedger()
        self._ledger_lock = threading.Lock()
        self._max_tokens = int(os.environ.get("CEOBENCH_TRAIN_MAX_TOKENS", "0") or 0)

    @staticmethod
    def name() -> str:
        return "reef-ceobench"

    def version(self) -> str | None:
        return None

    async def setup(self, environment: BaseEnvironment) -> None:
        """Nothing to install: the image carries the pinned checkout."""

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        server = self._start_sidecar()
        episode_over = threading.Event()
        weekly = threading.Thread(target=self._report_weeks_online, args=(episode_over,), daemon=True)
        weekly.start()
        try:
            sidecar_host = urlsplit(self._service_url).hostname or "172.17.0.1"
            base_url = f"http://{sidecar_host}:{server.server_address[1]}/v1"
            command = runner_command(base_url, self.model_name or "reef", self._seed, self._days)
            self.logger.info("ceobench seed=%s days=%s agent via %s", self._seed, self._days, base_url)
            result = await environment.exec(command, env=forwarded_environment(dict(os.environ)))
        finally:
            episode_over.set()
            weekly.join()
            server.shutdown()

        turns = self._capture.snapshot()
        self._post_finished_weeks()
        with self._ledger_lock:
            ledger_turns = list(self._ledger.turns)
            weeks = self._ledger.summary()
        await environment.download_dir(RUNS_DIR, self.logs_dir / "ceobench")
        context.metadata = {
            **(context.metadata or {}),
            "reef": {
                "agent_record_ids": [turn["receipt"] for turn in ledger_turns],
                "agent_record_tokens": [turn["tokens"] for turn in ledger_turns],
                "agent_record_weeks": [turn["week"] for turn in ledger_turns],
            },
            "ceobench": {
                "seed": self._seed,
                "days": self._days,
                "turns": len(turns),
                "exit_code": result.return_code,
                "weeks": weeks,
            },
        }
        usage = [((turn.get("response") or {}).get("usage") or {}) for turn in turns]
        context.n_input_tokens = sum(int(item.get("prompt_tokens") or 0) for item in usage)
        context.n_output_tokens = sum(int(item.get("completion_tokens") or 0) for item in usage)
        if result.return_code != 0:
            raise RuntimeError(f"ceobench runner exited {result.return_code}; see {self.logs_dir / 'ceobench'}")

    def _start_sidecar(self) -> ThreadingHTTPServer:
        # Override, not setdefault: the runner's OpenAI client sends its own
        # Authorization header, and the scenario is this harness's to choose.
        config = ServeConfig(
            upstream=self._service_url,
            listen_host="0.0.0.0",
            listen_port=0,  # an ephemeral port, so concurrent trials never collide
            override_headers={"x-reef-scenario": self._scenario, "authorization": f"Bearer {self._token}"},
        )
        self._capture = CaptureStore()
        server = ThreadingHTTPServer((config.listen_host, config.listen_port), build_handler(config, self._capture))
        threading.Thread(target=server.serve_forever, name="ceobench-sidecar", daemon=True).start()
        return server

    def _report_weeks_online(self, episode_over: threading.Event) -> None:
        """Report every week as soon as the next week's dashboard shows up."""
        while not episode_over.wait(WEEK_POLL_S):
            self._post_finished_weeks()

    def _post_finished_weeks(self, final_cash: float | None = None) -> int:
        """Post the unreported weeks whose closing cash is known; return how many."""
        with self._ledger_lock:
            self._ledger.observe(self._capture.snapshot())
            finished = self._ledger.finished_weeks(final_cash)
            for week, cash_end in finished:
                entry = self._ledger.weeks[week]
                posted = post_week_reports(
                    self._client,
                    self._scenario,
                    week=week,
                    day=entry["day"],
                    cash_start=entry["cash_start"],
                    cash_end=cash_end,
                    turns=entry["turns"],
                    max_tokens=self._max_tokens,
                )
                self._ledger.posted.add(week)
                self.logger.info(
                    "reported week %d (cash %.0f -> %.0f) against %d of %d turns",
                    week,
                    entry["cash_start"],
                    cash_end,
                    len(posted),
                    len(entry["turns"]),
                )
        return len(finished)

    def _report_trial_result(self) -> None:
        """Close the last week with the verifier's final cash once Harbor writes result.json."""
        result_path = self.logs_dir.parent / "result.json"
        while not (result_path.exists() and result_path.stat().st_mtime >= self._report_watch_from):
            time.sleep(1.0)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        rewards = (result.get("verifier_result") or {}).get("rewards") or {}
        if rewards.get("final_cash") is None:
            self.logger.warning(
                "trial %s ended without a verifier final cash; last week not reported", result.get("id")
            )
            return
        self._post_finished_weeks(final_cash=float(rewards["final_cash"]))
