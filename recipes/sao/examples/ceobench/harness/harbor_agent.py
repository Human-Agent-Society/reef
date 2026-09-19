"""One Harbor trial: CEO-Bench's bash agent, served by Reef and scored week by week.

``harness/`` is the benchmark's bash agent. ``agent.py`` is its loop and
``tools.py`` its tools, both the benchmark's call for call; its prompt and
tool definitions are read from the pinned checkout in the task image when
the episode starts, so they are the benchmark's byte for byte. This module
is what Reef adds around that agent:

- the task's engine session, started through the task image's
  ``/opt/ceobench-engine.py`` and read over its HTTP API (status, dashboard,
  the books);
- a reef-client proxy on the host's loopback that stamps the scenario and
  token on every model call and keeps each exchange with its receipt, so
  the agent's OpenAI client sees a plain base URL and the request body is
  the agent's own;
- the reward (``report.py``): every captured turn is filed under the
  simulated week it was played in, and when enough later weeks have opened
  the week's decision turns are reported with the week's credit, its change
  in company value (cash plus the engine's subscription run-rate over the
  weeks left) and the discounted changes of the weeks after it, scaled
  against the weeks before;
- the pace of the game: every model call waits until the batches the
  reported turns filled have committed a training release.

What the model is asked, what it may call, and what its tools return are
the benchmark's; what this module decides is where the calls go, when a
week is credited, and how long the game waits for the trainer.

Connection settings come from the environment set by ``run.sh``:

- ``REEF_SERVICE_URL`` (required), ``REEF_SCENARIO``, ``REEF_TOKEN``,
  ``REEF_TIMEOUT_S``
- ``CEOBENCH_LLM_TIMEOUT_S`` (default 1800): how long one model call may
  take, a paced hold included; the benchmark's loop retries a timeout.
- ``CEOBENCH_REASONING_EFFORT`` (default ``none``) and
  ``CEOBENCH_MAX_COMPLETION_TOKENS`` (default 16384): the request fields
  the benchmark's runner sets.
- ``CEOBENCH_TOOL_USER`` (default ``agent``): the container user the
  agent's tools run as; ``CEOBENCH_BASH_TIMEOUT_S`` (default 1200): the
  benchmark's limit on one bash command, ``next-week`` included.

Variables named ``SAAS_BENCH_*``, ``OPENAI_*``, ``ANTHROPIC_*``, and
``AWS_*`` are forwarded into the container when the engine starts, for the
simulator roles.

``CEOBENCH_TRAIN_MAX_TOKENS`` (0 or unset: no limit) is the trainer's window:
a turn whose prompt and completion together exceed this many tokens is
recorded and never reported, because the trainer could not hold it.

``CEOBENCH_VALUE_HORIZON_WEEKS`` (default 26) caps the weeks of run-rate a
week's opening state is valued at. ``CEOBENCH_CREDIT_WEEKS`` (default 4) and
``CEOBENCH_CREDIT_DISCOUNT`` (default 0.8) set how many later weeks' value
changes a week is credited with and how fast they discount; a week is
reported once that many weeks have opened after it. ``CEOBENCH_SCORE_CLIP``
(default 0.05) and ``CEOBENCH_SCORE_FLOOR`` (default 0.003) clip a week's
credit and floor the running scale it is divided by before it is posted.

Only a week's decision turns are reported: the turns whose tool call changed
the company (``report.DECISION_CALLS``). ``CEOBENCH_ENGINE_READS`` (default
on; ``0`` turns it off) values the base at the engine's own monthly recurring
revenue, read from its books at each week start; the listed-price estimate
from the dashboard is the fallback.

``CEOBENCH_REPORTS`` (default on; ``0`` turns it off) is the untrained
baseline switch: the weeks are still recorded in the trial metadata, but no
report is posted, so the stack serves the base model for the whole episode
and trains nothing. The pacer is off with it.

``CEOBENCH_PACE_BATCH`` (0 or unset: off) paces the game to the trainer. Set
to the recipe's batch size, every model call waits until each batch the
reported turns filled has committed a training release, so a week is played
by a policy trained on every week the credit window has closed, and no turn
is generated while a step publishes its adapter.
``CEOBENCH_PACE_TIMEOUT_S`` (default 1800) bounds one such wait.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shlex
import threading
import time
from dataclasses import replace
from http.server import ThreadingHTTPServer
from typing import NamedTuple, TextIO

import httpx
import openai
from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from reef_client import ReefClient
from reef_client.serve import CaptureStore, ServeConfig, build_handler

from .agent import DEFAULT_MAX_COMPLETION_TOKENS, BashAgent
from .report import (
    DEFAULT_CREDIT_DISCOUNT,
    DEFAULT_CREDIT_WEEKS,
    DEFAULT_SCORE_CLIP,
    DEFAULT_SCORE_FLOOR,
    DEFAULT_VALUE_HORIZON_WEEKS,
    INITIAL_CASH,
    ScenarioReleases,
    ScoreScale,
    TrainingPacer,
    WeekRecords,
    WeekStart,
    dashboards,
    post_week_reports,
)
from .tools import DEFAULT_BASH_TIMEOUT_S, ContainerTools
from .values import JsonValue

#: Episode defaults; ``kwargs`` on the Harbor agent config override them.
DEFAULT_SEED = 42
DEFAULT_DAYS = 500
DEFAULT_TOOL_USER = "agent"
#: The benchmark's runner takes at most this many turns before it re-reads
#: the dashboard and hands it to the agent as the last tool's output.
TURNS_PER_ROUND = 100
DAYS_PER_WEEK = 7
#: The pinned checkout, the task's engine script, and the run root inside the task container.
CHECKOUT_DIR = "/opt/ceobench"
CHECKOUT_PYTHON = "/opt/ceobench/.venv/bin/python"
ENGINE_SCRIPT = "/opt/ceobench-engine.py"
RUNS_DIR = "/workspace/ceobench-runs"
#: Environment variables the engine's simulator roles read; forwarded when the engine starts.
FORWARDED_ENV_PREFIXES = ("SAAS_BENCH_", "OPENAI_", "ANTHROPIC_", "AWS_")
#: Bounds on one command in the container, in seconds.
ENGINE_START_TIMEOUT_S = 600.0
ENGINE_READ_TIMEOUT_S = 180.0
ENGINE_STOP_TIMEOUT_S = 300.0

#: Runs with the checkout's interpreter: the agent's prompt and tools as the
#: benchmark's own classes build them, for the episode's length.
CONTRACT_SCRIPT = """
import json, sys
from saas_bench.agents.bash_agent.agent import BashAgent
from saas_bench.agents.bash_agent.tools import get_bash_agent_tool_descriptions
request = json.loads(sys.stdin.read())
tools = get_bash_agent_tool_descriptions()
agent = BashAgent(tool_descriptions=tools, client=None, model=request["model"], total_days=request["days"])
print(json.dumps({"system_prompt": agent.system_prompt, "tools": tools}))
"""
#: Runs with the container's Python: one read of the engine's HTTP API.
ENGINE_READ_SCRIPT = """
import json, sys, urllib.request
request = json.loads(sys.stdin.read())
url = f"http://127.0.0.1:{request['port']}"
try:
    if request["op"] == "query":
        body = json.dumps({"sql": request["sql"]}).encode()
        call = urllib.request.Request(url + "/query", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(call, timeout=120) as response:
            print(json.dumps(json.loads(response.read())))
    else:
        path = "/dashboard" if request["op"] == "dashboard" else "/game-status"
        with urllib.request.urlopen(urllib.request.Request(url + path), timeout=30) as response:
            print(json.dumps(json.loads(response.read())))
except (OSError, ValueError) as error:
    print(json.dumps({"error": f"{type(error).__name__}: {error}"}))
"""


#: The engine's own monthly recurring revenue: every live subscription at its
#: effective price times its seats (an individual subscription has one).
MRR_SQL = (
    "SELECT COALESCE(SUM(effective_price * COALESCE(seat_count, 1)), 0) AS mrr"
    " FROM subscriptions WHERE status = 'subscribed' AND end_day IS NULL"
)
MRR_FALLBACK_SQL = (
    "SELECT COALESCE(SUM(effective_price), 0) AS mrr FROM subscriptions"
    " WHERE status = 'subscribed' AND end_day IS NULL"
)


class Session(NamedTuple):
    """The engine session the task started for this episode."""

    run_dir: str
    workspace: str
    session_id: str
    port: int
    total_days: int
    initial_cash: float


class Contract(NamedTuple):
    """What the benchmark's agent class is built with: its prompt and its tools."""

    system_prompt: str
    tools: list[dict]


def forwarded_environment(environ: dict[str, str]) -> dict[str, str]:
    return {key: value for key, value in environ.items() if key.startswith(FORWARDED_ENV_PREFIXES)}


class TaskContainer:
    """The task container from the episode thread: its engine session and the checkout it carries."""

    def __init__(
        self,
        environment: BaseEnvironment,
        loop: asyncio.AbstractEventLoop,
        *,
        environ: dict[str, str] | None = None,
    ) -> None:
        self.environment = environment
        self.loop = loop
        self.environ = dict(os.environ if environ is None else environ)
        self.session: Session | None = None

    def exec(
        self,
        command: str,
        *,
        stdin: str | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: float,
    ) -> dict[str, JsonValue]:
        """Run ``command`` in the container (``stdin`` piped in) and return its last line as JSON."""
        if stdin is not None:
            encoded = base64.b64encode(stdin.encode("utf-8")).decode("ascii")
            command = f"printf %s {encoded} | base64 -d | {command}"
        future = asyncio.run_coroutine_threadsafe(
            self.environment.exec(command, cwd=cwd, env=env, timeout_sec=int(timeout_s)), self.loop
        )
        result = future.result(timeout=timeout_s + 60.0)
        lines = [line for line in (result.stdout or "").splitlines() if line.strip()]
        try:
            payload = json.loads(lines[-1]) if lines else None
        except json.JSONDecodeError:
            payload = None
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"the task container answered exit {result.return_code} without JSON: "
                f"{(result.stderr or result.stdout or '')[-2000:]}"
            )
        if payload.get("error"):
            raise RuntimeError(str(payload["error"]))
        return payload

    def require_session(self) -> Session:
        if self.session is None:
            raise RuntimeError("the engine has not been started")
        return self.session

    def start_engine(
        self, *, seed: int, days: int, cash: float, model: str, reasoning_effort: str | None, tool_user: str
    ) -> Session:
        """Have the task create the session and start its engine; the simulator roles' settings go along."""
        arguments = [
            "start",
            "--runs-dir",
            RUNS_DIR,
            "--seed",
            str(seed),
            "--days",
            str(days),
            "--cash",
            str(cash),
            "--model",
            model,
            "--reasoning-effort",
            reasoning_effort or "",
            "--tool-user",
            tool_user,
        ]
        answer = self.exec(
            f"{CHECKOUT_PYTHON} {ENGINE_SCRIPT} {shlex.join(arguments)}",
            cwd=CHECKOUT_DIR,
            env=forwarded_environment(self.environ),
            timeout_s=ENGINE_START_TIMEOUT_S,
        )
        self.session = Session(
            run_dir=str(answer["run_dir"]),
            workspace=str(answer["workspace"]),
            session_id=str(answer["session_id"]),
            port=int(answer["port"]),
            total_days=int(answer["total_days"]),
            initial_cash=float(answer["initial_cash"]),
        )
        return self.session

    def contract(self, days: int, model: str) -> Contract:
        """The agent's prompt and tools, built by the benchmark's classes in the checkout."""
        answer = self.exec(
            f"{CHECKOUT_PYTHON} -c {shlex.quote(CONTRACT_SCRIPT)}",
            stdin=json.dumps({"days": days, "model": model}),
            cwd=CHECKOUT_DIR,
            timeout_s=ENGINE_READ_TIMEOUT_S,
        )
        return Contract(system_prompt=str(answer["system_prompt"]), tools=list(answer["tools"]))

    def read(self, op: str, **fields: JsonValue) -> dict[str, JsonValue]:
        session = self.require_session()
        return self.exec(
            f"python3 -c {shlex.quote(ENGINE_READ_SCRIPT)}",
            stdin=json.dumps({"op": op, "port": session.port, **fields}),
            timeout_s=ENGINE_READ_TIMEOUT_S,
        )

    def status(self) -> dict[str, JsonValue]:
        """``/game-status``; the benchmark's fallback when the engine does not answer."""
        try:
            return self.read("status")
        except RuntimeError:
            return {"day": 0, "cash": 0, "subscribers": 0, "timed_out": False}

    def dashboard(self) -> str:
        try:
            return str(self.read("dashboard").get("dashboard", ""))
        except RuntimeError:
            return "(Dashboard unavailable)"

    def query(self, sql: str) -> list[dict]:
        """Rows of one read-only SQL statement against the engine's live books."""
        answer = self.read("query", sql=sql)
        if answer.get("success") is False:
            raise RuntimeError(str(answer.get("error") or "query failed"))
        data = answer.get("data", answer)
        return list((data or {}).get("rows") or []) if isinstance(data, dict) else []

    def commit_weeks(self, sim_day: int) -> None:
        """Snapshot the workspace for every week reached, as the runner does at week boundaries."""
        session = self.require_session()
        self.exec(
            f"{CHECKOUT_PYTHON} {ENGINE_SCRIPT} commit --run-dir {shlex.quote(session.run_dir)} --day {int(sim_day)}",
            cwd=CHECKOUT_DIR,
            timeout_s=ENGINE_READ_TIMEOUT_S,
        )

    def stop_engine(self) -> dict[str, JsonValue]:
        """The final status; the engine stopped and ``world.nmdb`` copied for the verifier."""
        session = self.require_session()
        return self.exec(
            f"{CHECKOUT_PYTHON} {ENGINE_SCRIPT} stop --run-dir {shlex.quote(session.run_dir)}",
            cwd=CHECKOUT_DIR,
            timeout_s=ENGINE_STOP_TIMEOUT_S,
        )


class HarborAgent(BaseAgent):
    """One Harbor trial: one CEO-Bench episode, the benchmark's bash agent served by Reef."""

    def __init__(self, *args, seed: int = DEFAULT_SEED, days: int = DEFAULT_DAYS, **kwargs):
        super().__init__(*args, **kwargs)
        environ = os.environ
        self.service_url = environ.get("REEF_SERVICE_URL", "").rstrip("/")
        if not self.service_url:
            raise ValueError("the ceobench harness requires REEF_SERVICE_URL")
        self.scenario = environ.get("REEF_SCENARIO", "ceobench-sao")
        self.token = environ.get("REEF_TOKEN", "reef-local")
        self.seed = int(seed)
        self.days = int(days)
        self.client = ReefClient(
            self.service_url, token=self.token, timeout_s=float(environ.get("REEF_TIMEOUT_S", "7200"))
        )
        self.llm_timeout_s = float(environ.get("CEOBENCH_LLM_TIMEOUT_S", "") or 1800.0)
        self.reasoning_effort = environ.get("CEOBENCH_REASONING_EFFORT", "none") or None
        self.max_completion_tokens = int(
            environ.get("CEOBENCH_MAX_COMPLETION_TOKENS", "") or DEFAULT_MAX_COMPLETION_TOKENS
        )
        self.tool_user = environ.get("CEOBENCH_TOOL_USER", "") or DEFAULT_TOOL_USER
        self.bash_timeout_s = int(environ.get("CEOBENCH_BASH_TIMEOUT_S", "") or DEFAULT_BASH_TIMEOUT_S)
        self.capture = CaptureStore()
        self.records = WeekRecords(
            total_weeks=self.days // DAYS_PER_WEEK,
            horizon_weeks=int(environ.get("CEOBENCH_VALUE_HORIZON_WEEKS", "") or DEFAULT_VALUE_HORIZON_WEEKS),
            credit_weeks=int(environ.get("CEOBENCH_CREDIT_WEEKS", "") or DEFAULT_CREDIT_WEEKS),
            discount=float(environ.get("CEOBENCH_CREDIT_DISCOUNT", "") or DEFAULT_CREDIT_DISCOUNT),
        )
        self.scale = ScoreScale(
            clip=float(environ.get("CEOBENCH_SCORE_CLIP", "") or DEFAULT_SCORE_CLIP),
            floor=float(environ.get("CEOBENCH_SCORE_FLOOR", "") or DEFAULT_SCORE_FLOOR),
        )
        self.engine_reads = (environ.get("CEOBENCH_ENGINE_READS", "1") or "1") != "0"
        self.reports = (environ.get("CEOBENCH_REPORTS", "1") or "1") != "0"
        self.max_tokens = int(environ.get("CEOBENCH_TRAIN_MAX_TOKENS", "0") or 0)
        # Nothing is reported for an untrained baseline, so nothing trains and
        # there is no batch to wait for.
        batch = int(environ.get("CEOBENCH_PACE_BATCH", "0") or 0) if self.reports else 0
        self.pacer = (
            TrainingPacer(
                batch,
                float(environ.get("CEOBENCH_PACE_TIMEOUT_S", "1800") or 1800),
                ScenarioReleases(self.service_url, self.scenario, self.token),
                self.logger,
            )
            if batch > 0
            else None
        )
        self.current_week: int | None = None

    @staticmethod
    def name() -> str:
        return "reef-ceobench"

    def version(self) -> str | None:
        return None

    async def setup(self, environment: BaseEnvironment) -> None:
        """Nothing to install: the image carries the pinned checkout."""

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        # The episode is a blocking loop of model calls and container
        # commands; it runs in a thread so this event loop keeps serving the
        # container commands it issues.
        loop = asyncio.get_running_loop()
        task = TaskContainer(environment, loop)
        proxy = self.start_proxy()
        try:
            outcome = await asyncio.to_thread(self.play, task, environment, loop, proxy.server_address[1])
        finally:
            proxy.shutdown()
        turns = self.capture.snapshot()
        weeks = self.records.summary(final_cash=outcome["final_cash"])
        if task.session is not None:
            await environment.download_dir(RUNS_DIR, self.logs_dir / "ceobench")
        context.metadata = {
            **(context.metadata or {}),
            "reef": {
                "agent_record_ids": [turn["receipt"] for turn in self.records.turns],
                "agent_record_tokens": [turn["tokens"] for turn in self.records.turns],
                "agent_record_weeks": [turn["week"] for turn in self.records.turns],
                "agent_record_decisions": [turn["decision"] for turn in self.records.turns],
            },
            "ceobench": {
                "seed": self.seed,
                "days": self.days,
                "horizon_weeks": self.records.horizon_weeks,
                "credit_weeks": self.records.credit_weeks,
                "discount": self.records.discount,
                "score_clip": self.scale.clip,
                "score_floor": self.scale.floor,
                "engine_reads": self.engine_reads,
                "reports": self.reports,
                "turns": len(turns),
                "weeks": weeks,
                **outcome,
            },
        }
        usage = [((turn.get("response") or {}).get("usage") or {}) for turn in turns]
        context.n_input_tokens = sum(int(item.get("prompt_tokens") or 0) for item in usage)
        context.n_output_tokens = sum(int(item.get("completion_tokens") or 0) for item in usage)

    def start_proxy(self) -> ThreadingHTTPServer:
        """A local reef-client proxy: the agent's OpenAI client talks to it, Reef gets the calls."""
        config = ServeConfig(
            upstream=self.service_url,
            listen_host="127.0.0.1",
            listen_port=0,  # an ephemeral port, so concurrent trials never collide
            override_headers={"x-reef-scenario": self.scenario, "authorization": f"Bearer {self.token}"},
        )
        self.capture = CaptureStore()
        server = ThreadingHTTPServer((config.listen_host, config.listen_port), build_handler(config, self.capture))
        threading.Thread(target=server.serve_forever, name="ceobench-proxy", daemon=True).start()
        return server

    def play(
        self, task: TaskContainer, environment: BaseEnvironment, loop: asyncio.AbstractEventLoop, proxy_port: int
    ) -> dict:
        """The benchmark runner's episode, with the weeks credited as they close."""
        model = self.model_name or "reef"
        session = task.start_engine(
            seed=self.seed,
            days=self.days,
            cash=INITIAL_CASH,
            model=model,
            reasoning_effort=self.reasoning_effort,
            tool_user=self.tool_user,
        )
        contract = task.contract(session.total_days, model)
        tools = ContainerTools(
            environment,
            loop,
            workspace=session.workspace,
            port=session.port,
            user=self.tool_user,
            bash_timeout_s=self.bash_timeout_s,
            logger=self.logger,
        )
        tools.install()
        self.logger.info(
            "ceobench seed=%s days=%s: session %s, engine port %d, %d tools, prompt %d chars",
            self.seed,
            session.total_days,
            session.session_id,
            session.port,
            len(contract.tools),
            len(contract.system_prompt),
        )
        client = openai.OpenAI(
            base_url=f"http://127.0.0.1:{proxy_port}/v1",
            api_key="reef",  # the proxy replaces it with the Reef token
            timeout=httpx.Timeout(self.llm_timeout_s),
        )
        agent = BashAgent(
            client,
            model,
            contract.system_prompt,
            contract.tools,
            tools,
            reasoning_effort=self.reasoning_effort,
            max_completion_tokens=self.max_completion_tokens,
            logger=self.logger,
        )
        with open(self.logs_dir / "turns.jsonl", "a", encoding="utf-8") as turn_log:
            outcome = self.rounds(task, tools, agent, session.total_days, turn_log)
        task.commit_weeks(int(outcome.get("sim_day", 0) or 0))
        final = task.stop_engine()
        status = final.get("final") or {}
        final_cash = float(status.get("cash", outcome.get("cash", 0.0)) or 0.0)
        self.post_finished_weeks(final_cash=final_cash)
        return {
            "outcome": outcome["outcome"],
            "final_cash": final_cash,
            "sim_day": int(status.get("day", outcome.get("sim_day", 0)) or 0),
            "total_days": session.total_days,
            "agent_turns": agent.total_turns,
            "agent_input_tokens": agent.total_input_tokens,
            "agent_output_tokens": agent.total_output_tokens,
        }

    def rounds(
        self, task: TaskContainer, tools: ContainerTools, agent: BashAgent, total_days: int, turn_log: TextIO
    ) -> dict:
        """The runner's outer loop: rounds of at most ``TURNS_PER_ROUND`` turns, each opened by the dashboard."""
        sim_day = 0
        cash = 0.0
        for round_index in range(1, total_days + 1):
            status = task.status()
            sim_day = int(status.get("day", sim_day) or 0)
            dashboard = task.dashboard()
            self.log_turn(turn_log, sim_day, 0, "_dashboard", {}, dashboard)
            observation = dashboard
            info = {"day": sim_day, "cash": status.get("cash", 0)}
            turns = 0
            day_ended = False
            while not day_ended and turns < TURNS_PER_ROUND:
                turns += 1
                week = sim_day // DAYS_PER_WEEK
                self.enter_week(task, week, sim_day)
                seen = len(self.capture)
                action = agent.act(observation, info)
                self.records.add_turns(week, self.capture.snapshot()[seen:])
                outcome = tools.execute(action.tool, action.arguments)
                if outcome.timeout is not None:
                    self.logger.warning("next-week timed out on sim day %d: %s", sim_day, outcome.timeout)
                    return {"outcome": "timeout", "sim_day": sim_day, "cash": cash}
                observation = outcome.result
                if action.tool == "bash":
                    agent.check_day_advanced(observation)
                self.log_turn(turn_log, sim_day, agent.total_turns, action.tool, action.arguments, observation)
                if agent.day_advanced:
                    day_ended = True
                    agent.clear_day_advanced()
                status = task.status()
                sim_day = int(status.get("day", sim_day) or 0)
                if sim_day >= total_days:
                    return {"outcome": "completed", "sim_day": sim_day, "cash": status.get("cash", cash)}
                if status.get("timed_out"):
                    self.logger.warning("the engine timed out stepping the week on sim day %d", sim_day)
                    return {"outcome": "timeout", "sim_day": sim_day, "cash": status.get("cash", cash)}
                cash = float(status.get("cash", 0) or 0)
                info = {"day": sim_day, "cash": cash}
                if cash < 0:
                    return {"outcome": "bankrupt", "sim_day": sim_day, "cash": cash}
            if not day_ended:
                self.logger.info(
                    "round %d: %d turns on sim day %d without a week advance; going on", round_index, turns, sim_day
                )
            status = task.status()
            sim_day = int(status.get("day", sim_day) or 0)
            cash = float(status.get("cash", 0) or 0)
            if sim_day >= total_days:
                return {"outcome": "completed", "sim_day": sim_day, "cash": cash}
            if cash < 0:
                return {"outcome": "bankrupt", "sim_day": sim_day, "cash": cash}
        return {"outcome": "completed" if sim_day >= total_days else "incomplete", "sim_day": sim_day, "cash": cash}

    @staticmethod
    def log_turn(turn_log: TextIO, day: int, turn: int, tool: str, arguments: dict, result: str) -> None:
        turn_log.write(
            json.dumps(
                {"at": time.time(), "day": day, "turn": turn, "tool": tool, "arguments": arguments, "result": result}
            )
            + "\n"
        )
        turn_log.flush()

    def enter_week(self, task: TaskContainer, week: int, sim_day: int) -> None:
        """Before a model call: open a new week, report the weeks it finishes, wait for the trainer."""
        if self.current_week is None or week > self.current_week:
            self.current_week = week
            if week > 0:
                task.commit_weeks(sim_day)
            start = self.week_start(task, week, sim_day)
            self.records.open_week(start)
            self.logger.info(
                "week %d starts (day %d): cash %.0f, %d subscribers, %d seats, run-rate %.0f/month (%s)",
                week,
                start.day,
                start.cash,
                start.subscribers,
                start.seats,
                start.run_rate,
                "engine MRR" if start.mrr is not None else "listed-price estimate",
            )
            self.post_finished_weeks()
        if self.pacer is None:
            return
        posted = sum(
            1
            for posted_week, entry in self.records.weeks.items()
            if posted_week in self.records.posted
            for receipt, tokens, decision in entry.turns
            if decision is not None and (not self.max_tokens or tokens <= self.max_tokens)
        )
        waited = self.pacer.wait(week, posted)
        if waited > 1.0:
            self.logger.info("week %d held %.0fs for training", week, waited)

    def week_start(self, task: TaskContainer, week: int, sim_day: int) -> WeekStart:
        """The week's opening state from the engine's dashboard, valued at its own MRR when readable."""
        parsed = dashboards(task.dashboard())
        start = parsed[0] if parsed else None
        if start is None or start.week != week:
            status = task.status()
            start = WeekStart(
                week=week,
                day=sim_day,
                cash=float(status.get("cash", 0) or 0),
                subscribers=int(status.get("subscribers", 0) or 0),
                seats=0,
                prices=(0.0, 0.0, 0.0),
            )
        if self.engine_reads:
            mrr = self.engine_mrr(task)
            if mrr is not None:
                start = replace(start, mrr=mrr)
        return start

    def engine_mrr(self, task: TaskContainer) -> float | None:
        """The engine's own monthly recurring revenue, or ``None`` when its books cannot be read."""
        for sql in (MRR_SQL, MRR_FALLBACK_SQL):
            try:
                rows = task.query(sql)
            except RuntimeError as error:
                self.logger.warning("engine read failed: %s", error)
                continue
            if not rows:
                return 0.0
            row = rows[0]
            value = row["mrr"]
            return float(value or 0.0)
        return None

    def post_finished_weeks(self, final_cash: float | None = None) -> int:
        """Post the unreported weeks whose closing state is known; return how many."""
        finished = self.records.finished_weeks(final_cash)
        for week, cash_end, value_end, credit in finished:
            entry = self.records.weeks[week]
            start: WeekStart = entry.start
            value_start = self.records.value(start)
            score = self.scale.scale(credit)
            posted = (
                post_week_reports(
                    self.client,
                    self.scenario,
                    week=week,
                    day=start.day,
                    cash_start=start.cash,
                    cash_end=cash_end,
                    value_start=value_start,
                    value_end=value_end,
                    credit=credit,
                    score=score,
                    turns=entry.turns,
                    max_tokens=self.max_tokens,
                )
                if self.reports
                else []
            )
            self.records.posted.add(week)
            self.records.scores[week] = score
            self.logger.info(
                "%s week %d (credit %.4f, score %.2f; value %.0f -> %.0f, cash %.0f -> %.0f)"
                " against %d of %d decision turns (%d turns)",
                "reported" if self.reports else "recorded",
                week,
                credit,
                score,
                value_start,
                value_end,
                start.cash,
                cash_end,
                len(posted),
                sum(1 for receipt, tokens, decision in entry.turns if decision is not None),
                len(entry.turns),
            )
        return len(finished)
