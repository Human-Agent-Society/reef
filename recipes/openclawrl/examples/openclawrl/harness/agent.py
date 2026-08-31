"""The stream harness IS the Hermes agent: one homework session per position.

reef-eval invokes this once per stream position. It owns the whole session:

* a **header shim** thread (``reef_client.serve``) fronting reef with the
  stream's scenario attached — hermes cannot set ``x-reef-scenario``
  itself. It binds ``0.0.0.0`` so the in-container hermes reaches it on
  the same host the deployment already allowlists for reef;
* the **reef scenario id**, minted at position 0 and carried in
  ``$REEF_EVAL_STATE_DIR`` — one stream, one scenario, one weight chain; a
  fresh variant starts fresh;
* **hermes itself**, run inside the task container (the environment image
  installs it) with its home on the state mount, so optional agent memory
  rides reef-eval's one cross-position channel. The config is written on first
  use; compression stays off because history rewrites would break the
  processor's trace matching (and homework sessions are short);
* the **conversation loop** with the judge sidecar: student message from
  ``$JUDGE_URL/state`` → one ``hermes -z`` turn (``--resume latest`` after
  the first) → the reply to ``/reply`` — until the student is done or
  ``MAX_TURNS``. A failed hermes turn ends the session, not the trial: the
  verifier scores whatever the judge saw.

Agent configuration (``--agent-arg``): ``reef_url`` (required; reef as
reachable from inside a container, e.g. ``http://172.17.0.1:28900`` — and
allowlist that host) and ``hermes_memory`` (off by default: memory and
weights are two separate adaptation channels; the paper's numbers need
memory OFF, and memory-only / weights-only / both are variants of the
same stream). Everything else is a constant below, except ``REEF_TOKEN``:
run.sh mints one per run, because this stack listens on the LAN.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import threading
import time
import uuid
from http.server import ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from reef_client.serve import CaptureStore, ServeConfig, build_handler

TOKEN = os.environ.get("REEF_TOKEN", "")  # run.sh mints one per run: the stack listens on the LAN
MODEL = "reef"  # model name hermes sends through the shim
SCENARIO_PREFIX = "openclawrl-stream"
MAX_TURNS = 8
TURN_TIMEOUT_S = 600
SHIM_PORT = 29101
#: How often to ask the judge whether its reaction has landed.
REACTION_POLL_S = 2.0

STATE_DIR = "/reef_eval/state"
HERMES_HOME = f"{STATE_DIR}/hermes"


# context_length stays at hermes's hard 64k minimum (it refuses to start
# below it) even though Qwen3-8B serves a 40960 window — hermes only uses
# the number to size its own compaction. max_tokens is what the engine
# actually checks: unset, hermes requests a completion the size of its whole
# window, so input + 65536 blows past 40960 and every turn 400s. Pinning it
# to the trainer's --rollout-max-response-len (8192) keeps requests in the
# engine's real window and matches what the rollout can consume. Do not lower
# it: a thinking model that runs out of tokens mid-<think> returns truncated
# reasoning, which then has to be caught downstream instead of never happening.
HERMES_CONFIG = """\
model:
  provider: "custom"
  base_url: "http://{shim_host}:{shim_port}/v1"
  default: "{model}"
  context_length: 65536
  max_tokens: 8192

memory:
  memory_enabled: {memory}

# platform_toolsets is top-level — hermes reads it as config["platform_toolsets"],
# so nesting it under `tools:` is silently ignored and the agent gets the full
# default bundle (18 tools) instead of the named ones. The paper's runs had it
# nested, so they ran on that bundle; `terminal` is what it bought them, and
# the task does not work without it. The student asks for an append that must
# not overwrite: `write_file` replaces the whole file, so the `file` toolset's
# only append route is `patch` — whose addition-only hunk the model cannot
# land (observed: six consecutive "context hint not found" failures in one
# turn, with read_file's `N|` line prefixes echoed back into the patch body).
# Sessions then burn every turn on failed patches, the student's style
# complaint never reaches the judge, and the PRM reads the repeated request as
# contentment and rewards it. `file` + `terminal` is the smallest set that can
# finish the task: 5 tools, so `clarify` still cannot burn a turn asking where
# the homework is and the schemas stay a small share of the prompt.
platform_toolsets:
  cli: [file, terminal]

compression:
  enabled: false
"""


class HermesStreamAgent(BaseAgent):
    """Drive one hermes homework session against the judge sidecar."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._reef_url = str(kwargs.get("reef_url", "")).strip().rstrip("/")
        if not self._reef_url:
            raise ValueError("the hermes stream harness requires --agent-arg reef_url=http://<host>:<port>")
        self._hermes_memory = str(kwargs.get("hermes_memory", "")).strip().lower() in ("1", "true", "yes")
        self._capture = CaptureStore()  # replaced per session by _start_shim

    @staticmethod
    def name() -> str:
        return "openclawrl-hermes-stream"

    def version(self) -> str | None:
        return None

    async def setup(self, environment: BaseEnvironment) -> None:
        """Nothing to install: the environment image carries hermes."""

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        del instruction  # the judge protocol drives the session, not the prompt
        problem = json.loads(await self._exec(environment, "cat /agent/problem.json"))
        scenario = await self._ensure_scenario(environment)
        # One homework session is one conversation: the position's index makes
        # the tag unique inside the stream and identical across a resume.
        shim = self._start_shim(scenario, f"{scenario}-s{problem['index']}")
        try:
            await self._prepare(environment, problem)
            turns, failure = await self._session_loop(environment)
        finally:
            shim.shutdown()
            shim.server_close()  # the next position reuses this port
        metadata = dict(context.metadata or {})
        metadata["openclawrl"] = {"scenario": scenario, "turns": turns, "failure": failure or None}
        context.metadata = metadata

    # ------------------------------------------------------------- plumbing

    async def _exec(self, environment: BaseEnvironment, command: str, *, ok_codes: tuple[int, ...] = (0,)) -> str:
        result = await environment.exec(command)
        if result.return_code not in ok_codes:
            raise RuntimeError(
                f"exec failed ({result.return_code}): {command[:120]} :: {(result.stderr or '')[-300:]}"
            )
        return result.stdout or ""

    async def _ensure_scenario(self, environment: BaseEnvironment) -> str:
        existing = (await self._exec(environment, f"cat {STATE_DIR}/scenario 2>/dev/null || true")).strip()
        if existing:
            return existing
        minted = f"{SCENARIO_PREFIX}-{uuid.uuid4().hex[:12]}"
        await self._exec(
            environment, f"mkdir -p {STATE_DIR} && printf %s {shlex.quote(minted)} > {STATE_DIR}/scenario"
        )
        return minted

    def _upstream_health(self) -> tuple[int, str]:
        """``(successful reef calls so far, last upstream error seen)``.

        The shim proxies every model call, so its capture is the only place
        that distinguishes "the agent answered badly" from "the agent never
        reached a model" — hermes reports both as ordinary text on exit 0.
        """
        ok, last_error = 0, ""
        for turn in self._capture.snapshot():
            status = int(turn.get("status") or 0)
            if 200 <= status < 300:
                ok += 1
            else:
                response = turn.get("response")
                error = response.get("error") if isinstance(response, dict) else None
                message = error.get("message") if isinstance(error, dict) else error
                last_error = f"HTTP {status} from reef" + (f": {str(message)[:200]}" if message else "")
        return ok, last_error

    def _start_shim(self, scenario: str, session: str) -> ThreadingHTTPServer:
        # Override, not setdefault: hermes's OpenAI client sends its own
        # (dummy) Authorization header, and the stream's scenario id is
        # authoritative — nothing the agent sends may replace either.
        #
        # The session tag is what makes this method trainable here at all.
        # Hermes keeps its transcript locally and restarts each turn from
        # ``[system, user]``, so no request ever extends the previous one and
        # reef's header-free correlation binds only inside a turn — never
        # across the student reply that carries the whole reward signal.
        headers = {
            "x-reef-scenario": scenario,
            "x-reef-tag-session": session,
            "authorization": f"Bearer {TOKEN}",
        }
        config = ServeConfig(
            upstream=self._reef_url,
            listen_host="0.0.0.0",
            listen_port=SHIM_PORT,
            override_headers=headers,
        )
        self._capture = CaptureStore()
        server = ThreadingHTTPServer((config.listen_host, config.listen_port), build_handler(config, self._capture))
        threading.Thread(target=server.serve_forever, name="openclawrl-shim", daemon=True).start()
        return server

    async def _prepare(self, environment: BaseEnvironment, problem: dict) -> None:
        # The homework lands under HERMES_HOME, not /workspace: hermes pins its
        # working directory to HOME (``--in`` does not survive), so that is
        # where the judge's relative ``homework/N.txt`` resolves for the agent.
        # The directory is wiped first because HOME rides the state mount —
        # left alone, every past session's solved homework would stay readable
        # and become a third adaptation channel behind weights and memory.
        homework = f"Problem:\n{problem['question']}\n\nSolution:\n"
        await self._exec(
            environment,
            f"rm -rf {HERMES_HOME}/homework && mkdir -p {HERMES_HOME}/homework && "
            f"printf %s {shlex.quote(homework)} > {HERMES_HOME}/homework/{problem['index']}.txt",
        )
        # The shim runs on the same host as reef; hermes reaches it there.
        shim_host = urlsplit(self._reef_url).hostname or "172.17.0.1"
        hermes_config = HERMES_CONFIG.format(
            shim_host=shim_host,
            shim_port=SHIM_PORT,
            model=MODEL,
            memory="true" if self._hermes_memory else "false",
        )
        # Position 0 writes the config; later positions keep it (and memory).
        await self._exec(
            environment,
            f"[ -f {HERMES_HOME}/.hermes/config.yaml ] || (mkdir -p {HERMES_HOME}/.hermes && "
            f"printf %s {shlex.quote(hermes_config)} > {HERMES_HOME}/.hermes/config.yaml)",
        )

    async def _session_loop(self, environment: BaseEnvironment) -> tuple[int, str]:
        state = json.loads(await self._exec(environment, 'curl -s "$JUDGE_URL/state"'))
        turns = 0
        while not state.get("done") and turns < MAX_TURNS:
            served_before, _ = self._upstream_health()
            try:
                reply = await self._hermes_turn(environment, str(state.get("message", "")), resume=turns > 0)
            except RuntimeError as error:  # a dead agent ends the session, not the trial
                return turns, str(error)
            # A turn that reached no model is infrastructure, not a bad answer:
            # hermes exits 0 and hands back its own error text, which the judge
            # would otherwise score as an ordinary (zero-reward) reply. Ending
            # the session with `failure` set keeps a broken stack from reading
            # as a flat "never adapts" learning curve.
            served_after, last_error = self._upstream_health()
            if served_after == served_before:
                return turns, last_error or "hermes turn reached no reef inference call"
            turns += 1
            state = await self._deliver_reply(environment, reply, turns)
            if state is None:
                return turns, "the judge stopped answering; session ended on its record"
            state = await self._await_reaction(environment, state)
            if state is None:
                return turns, "the judge never finished reacting to the reply"
        return turns, ""

    async def _deliver_reply(self, environment: BaseEnvironment, reply: str, turns: int) -> dict | None:
        """POST one reply and return the student's next state, or None.

        The judge is authoritative about what it received, so a lost response
        is recovered by re-reading ``/state`` — never by re-POSTing. ``/reply``
        counts the turn and records the session's first reply before it starts
        composing a reaction, so a retry would burn a turn and corrupt the
        score. Both calls can also come back non-JSON (an egress proxy's own
        error page), which ``json.loads`` reports as a ValueError.
        """
        body = shlex.quote(json.dumps({"text": reply}))
        try:
            return json.loads(
                await self._exec(
                    environment,
                    f'curl -s -X POST -H "Content-Type: application/json" -d {body} "$JUDGE_URL/reply"',
                )
            )
        except (RuntimeError, ValueError):
            pass
        try:
            recovered = json.loads(await self._exec(environment, 'curl -s "$JUDGE_URL/state"'))
        except (RuntimeError, ValueError):
            return None
        return recovered if int(recovered.get("turn") or 0) >= turns else None

    async def _await_reaction(self, environment: BaseEnvironment, state: dict) -> dict | None:
        """Poll until the student's reaction has landed.

        The judge answers ``/reply`` immediately and composes off the response
        path, so the next message is not in that response — an older judge
        that reacts inline simply reports no ``ready`` field and this returns
        at once.
        """
        deadline = time.monotonic() + TURN_TIMEOUT_S
        while state.get("ready") is False:
            if time.monotonic() > deadline:
                return None
            await asyncio.sleep(REACTION_POLL_S)
            try:
                state = json.loads(await self._exec(environment, 'curl -s "$JUDGE_URL/state"'))
            except (RuntimeError, ValueError):
                return None
        return state

    async def _hermes_turn(self, environment: BaseEnvironment, message: str, *, resume: bool) -> str:
        resume_flag = " --resume latest" if resume else ""
        command = (
            f"cd /workspace && HOME={HERMES_HOME} timeout {TURN_TIMEOUT_S} "
            f"hermes -z {shlex.quote(message)} --in /workspace{resume_flag}"
        )
        return (await self._exec(environment, command)).strip()
