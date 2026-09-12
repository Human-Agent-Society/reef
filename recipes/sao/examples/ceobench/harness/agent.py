"""Harbor agent that runs one CEO-Bench episode with the agent role served by Reef.

The Harbor environment holds a pinned CEO-Bench checkout
(``harbor/environment/Dockerfile``). ``run()`` starts a reef-client sidecar on
this host that stamps the scenario and token onto every model call and keeps
the receipts, then runs the benchmark's own bash-agent runner inside the
container with the agent role pointed at that sidecar over
``/v1/chat/completions``. The two simulator roles keep the benchmark's provider
settings and never pass through Reef. When the episode ends, the run directory
(``world.nmdb``, config, checkpoint, logs) is copied into the trial's log
directory, the receipts go into the agent context in call order, and a watcher
thread posts the verifier's reward against every receipt once Harbor writes
``result.json``.

Connection settings come from the environment set by ``run.sh``:

- ``REEF_SERVICE_URL`` (required): the Reef service as the task container
  reaches it, so a LAN address rather than ``127.0.0.1``
- ``REEF_SCENARIO``, ``REEF_TOKEN``, ``REEF_TIMEOUT_S``

Variables named ``SAAS_BENCH_*``, ``OPENAI_*``, ``ANTHROPIC_*``, and ``AWS_*``
are forwarded into the container for the simulator roles, so their credentials
and any provider override stay outside the repository.
"""

import atexit
import json
import os
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

from .report import post_reports

#: The pinned checkout and the run root inside the task container.
CEOBENCH_DIR = "/opt/ceobench"
RUNS_DIR = "/workspace/ceobench-runs"
#: Episode defaults; ``kwargs`` on the Harbor agent config override them.
DEFAULT_SEED = 42
DEFAULT_DAYS = 500
#: Environment variables the simulator roles read; forwarded verbatim.
FORWARDED_ENV_PREFIXES = ("SAAS_BENCH_", "OPENAI_", "ANTHROPIC_", "AWS_")


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


def forwarded_environment(environ: dict[str, str]) -> dict[str, str]:
    forwarded = {key: value for key, value in environ.items() if key.startswith(FORWARDED_ENV_PREFIXES)}
    # Reef serves /v1/chat/completions, not the Responses API the runner
    # prefers for OpenAI-compatible endpoints (see reef.patch).
    forwarded["SAAS_BENCH_OPENAI_CHAT_COMPLETIONS"] = "1"
    return forwarded


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
        self._capture = CaptureStore()
        # Harbor runs the verifier after run() returns and ends the trial by
        # writing result.json; watch for it from construction, so the reward
        # still reaches Reef when the episode itself failed late.
        self._report_watch_from = time.time()
        self._reporter = threading.Thread(target=self._report_trial_result, daemon=True)
        self._reporter.start()
        atexit.register(self._reporter.join, 120.0)  # hundreds of reports; don't drop them

    @staticmethod
    def name() -> str:
        return "reef-ceobench"

    def version(self) -> str | None:
        return None

    async def setup(self, environment: BaseEnvironment) -> None:
        """Nothing to install: the image carries the pinned checkout."""

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        server = self._start_sidecar()
        try:
            sidecar_host = urlsplit(self._service_url).hostname or "172.17.0.1"
            base_url = f"http://{sidecar_host}:{server.server_address[1]}/v1"
            command = runner_command(base_url, self.model_name or "reef", self._seed, self._days)
            self.logger.info("ceobench seed=%s days=%s agent via %s", self._seed, self._days, base_url)
            result = await environment.exec(command, env=forwarded_environment(dict(os.environ)))
        finally:
            server.shutdown()

        turns = self._capture.snapshot()
        receipts = [turn["receipt"] for turn in turns if turn["status"] == 200 and turn["receipt"]]
        await environment.download_dir(RUNS_DIR, self.logs_dir / "ceobench")
        context.metadata = {
            **(context.metadata or {}),
            "reef": {"agent_record_ids": receipts},
            "ceobench": {"seed": self._seed, "days": self._days, "turns": len(turns), "exit_code": result.return_code},
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

    def _report_trial_result(self) -> None:
        """Post the verifier reward once Harbor writes result.json in the trial directory."""
        result_path = self.logs_dir.parent / "result.json"
        while not (result_path.exists() and result_path.stat().st_mtime >= self._report_watch_from):
            time.sleep(1.0)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("verifier_result") is None:
            self.logger.warning("trial %s ended without a verifier result; nothing reported", result.get("id"))
            return
        posted = post_reports(result, client=self._client, scenario=self._scenario)
        self.logger.info(
            "reported score %s to reef against %d receipts",
            result["verifier_result"]["rewards"]["reward"],
            len(posted),
        )
