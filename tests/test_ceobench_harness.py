"""Host-side contracts of the CEO-Bench example (recipes/sao/examples/ceobench)."""

from __future__ import annotations

import asyncio
import base64
import importlib
import json
import logging
import runpy
import subprocess
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import httpx
import openai
import pytest

from tests.test_example_entrypoints import EXAMPLE_DIRS, _load_harness

EXAMPLE_DIR = EXAMPLE_DIRS["ceobench"]
SCORE_PATH = EXAMPLE_DIR / "harbor" / "tests" / "score.py"
ENGINE_PATH = EXAMPLE_DIR / "harbor" / "environment" / "engine.py"
PACKAGE = "_reef_ceobench_example_harness"


def load_modules(monkeypatch):
    agent, report = _load_harness(monkeypatch, "ceobench")
    return SimpleNamespace(
        agent=agent,
        report=report,
        tools=importlib.import_module(f"{PACKAGE}.tools"),
        harbor_agent=importlib.import_module(f"{PACKAGE}.harbor_agent"),
    )


def load_score_module():
    return SimpleNamespace(**runpy.run_path(str(SCORE_PATH), run_name="score"))


def dashboard(week: int, day: int, cash: int, *, subscribers: int = 0, seats: int = 0, prices=(0, 0, 0)) -> str:
    a, b, c = prices
    return (
        f"=== Week {week} Dashboard (Day {day}) ===\n\nCash: ${cash:,}\n"
        f"Individual Subscribers: {subscribers}\nEnterprise Subscribed Seats: {seats}\nOpen Issues: 0\n\n"
        f"--- Current Config ---\nPrices: A=${a}, B=${b}, C=${c}\nModel Tiers: A=1, B=1, C=1\n"
    )


def start(report, week: int, day: int, cash: float, *, subscribers=0, seats=0, prices=(0.0, 0.0, 0.0), mrr=None):
    return report.WeekStart(week, day, cash, subscribers, seats, prices, mrr)


def turn(receipt: str, prompt_tokens: int, completion_tokens: int, *, command=None, tool=None) -> dict:
    """A captured turn; by default its response calls ``next-week``, a decision."""
    if tool is not None:
        name, arguments = tool
    else:
        name, arguments = "bash", {"command": command or "./novamind-operation next-week 'rationale'"}
    message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "call-1", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
        ],
    }
    return {
        "status": 200,
        "receipt": receipt,
        "request": {"messages": []},
        "response": {
            "choices": [{"message": message}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        },
    }


def tool_call(call_id: str, name: str, arguments) -> SimpleNamespace:
    text = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=text), model_extra={})


def response(
    *calls, content: str = "", prompt_tokens: int = 10, completion_tokens: int = 3
) -> openai.types.chat.ChatCompletion:
    """A real SDK response, including its optional token detail fields."""
    return openai.types.chat.ChatCompletion.model_validate(
        {
            "id": "completion-test",
            "created": 0,
            "model": "reef",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls" if calls else "stop",
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": [
                            {
                                "id": call.id,
                                "type": "function",
                                "function": {"name": call.function.name, "arguments": call.function.arguments},
                                **call.model_extra,
                            }
                            for call in calls
                        ]
                        or None,
                    },
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "prompt_tokens_details": {"cached_tokens": 0},
                "completion_tokens_details": {"reasoning_tokens": 0},
            },
        }
    )


class Model:
    """A scripted model behind the OpenAI client interface, recording each request."""

    def __init__(self, responses, capture=None) -> None:
        self.responses = list(responses)
        self.requests: list[dict] = []
        self.capture = capture
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.requests.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        if self.capture is not None:
            message = response.choices[0].message
            self.capture.turns.append(
                {
                    "status": 200,
                    "receipt": f"r-{len(self.requests)}",
                    "request": kwargs,
                    "response": {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": message.content,
                                    "tool_calls": [
                                        {
                                            "id": c.id,
                                            "type": "function",
                                            "function": {"name": c.function.name, "arguments": c.function.arguments},
                                        }
                                        for c in (message.tool_calls or [])
                                    ],
                                }
                            }
                        ],
                        "usage": {
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                        },
                    },
                }
            )
        return response


class Capture:
    def __init__(self) -> None:
        self.turns: list[dict] = []

    def snapshot(self):
        return list(self.turns)

    def __len__(self) -> int:
        return len(self.turns)


class Notes:
    def __init__(self, text: str | None = None) -> None:
        self.text = text

    def memory(self):
        return self.text


class Client:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def report(self, scenario, payload, *, recipe=None):
        self.calls.append((scenario, payload))
        return {"accepted": True}


class Releases:
    def __init__(self, counts) -> None:
        self.counts = list(counts)

    def training_releases(self):
        return self.counts.pop(0) if len(self.counts) > 1 else self.counts[0]


class Simulation:
    """A two-week company: ``next-week`` moves the day by seven and grows the base a little."""

    def __init__(self, total_days: int = 14) -> None:
        self.day = 0
        self.cash = 1_000_000
        self.subscribers = 0
        self.total_days = total_days

    def dashboard(self) -> str:
        return dashboard(self.day // 7, self.day, self.cash, subscribers=self.subscribers, prices=(10, 39, 99))

    def status(self) -> dict:
        return {"day": self.day, "cash": self.cash, "subscribers": self.subscribers, "timed_out": False}

    def advance(self) -> None:
        self.day += 7
        self.cash -= 10_000
        self.subscribers += 5


class Task:
    """The task container as the harness drives it, over the simulation above."""

    def __init__(self, simulation: Simulation) -> None:
        self.simulation = simulation
        self.session = None
        self.queries: list[str] = []
        self.commits: list[int] = []
        self.stopped = False
        self.start_kwargs = None

    def start_engine(self, **kwargs):
        self.start_kwargs = kwargs
        self.session = SimpleNamespace(
            run_dir="/workspace/ceobench-runs/run_test",
            workspace="/workspace/ceobench-runs/run_test/agent_workspace",
            session_id="s1",
            port=4242,
            total_days=self.simulation.total_days,
            initial_cash=kwargs["cash"],
        )
        return self.session

    def contract(self, days, model):
        return SimpleNamespace(
            system_prompt=f"PROMPT for {days} days",
            tools=[{"name": "bash", "description": "run bash", "parameters": {"type": "object"}}],
        )

    def status(self):
        return self.simulation.status()

    def dashboard(self):
        return self.simulation.dashboard()

    def query(self, sql):
        self.queries.append(sql)
        return [{"mrr": self.simulation.subscribers * 10.0}]

    def commit_weeks(self, sim_day):
        self.commits.append(sim_day)

    def stop_engine(self):
        self.stopped = True
        return {"final": self.simulation.status(), "copied": True}


class Tools:
    """The agent's tools over the simulation: ``next-week`` advances it, anything else says ok."""

    def __init__(self, simulation: Simulation, memory: str | None = None) -> None:
        self.simulation = simulation
        self.notes = memory
        self.calls: list[tuple[str, dict]] = []
        self.installed = False

    def install(self):
        self.installed = True

    def execute(self, name, arguments):
        self.calls.append((name, arguments))
        if name == "bash" and "next-week" in arguments.get("command", ""):
            self.simulation.advance()
            return SimpleNamespace(result=self.simulation.dashboard(), timeout=None)
        return SimpleNamespace(result="ok", timeout=None)

    def memory(self):
        return self.notes


def create_harness(monkeypatch, modules, tmp_path, *, env=None):
    """A Harbor agent with its service settings set, week reporting initialized from ``env``."""
    for key, value in {"CEOBENCH_CREDIT_WEEKS": "1", **(env or {})}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("REEF_SERVICE_URL", "http://10.0.0.7:28900")
    monkeypatch.setenv("REEF_SCENARIO", "ceobench-host-test")
    monkeypatch.setenv("CEOBENCH_LLM_TIMEOUT_S", "30")
    monkeypatch.setenv("CEOBENCH_BASH_TIMEOUT_S", "60")

    def initialize_base(self):
        self.model_name = "reef"
        self.logs_dir = tmp_path / "agent"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("ceobench-test")

    monkeypatch.setattr(modules.harbor_agent.BaseAgent, "__init__", initialize_base)
    monkeypatch.setattr(modules.harbor_agent, "ReefClient", lambda *args, **kwargs: Client())
    return modules.harbor_agent.HarborAgent(seed=7, days=14)


def play(monkeypatch, modules, harness, task, tools, responses):
    """Run one episode of the harness against ``task`` and ``tools`` with a scripted model; return the context."""
    capture = Capture()
    model = Model(responses, capture)
    server = SimpleNamespace(server_address=("127.0.0.1", 29123), stopped=False)
    server.shutdown = lambda: setattr(server, "stopped", True)

    def start_proxy():
        harness.capture = capture
        return server

    downloads = []

    class Environment:
        async def download_dir(self, source, target):
            downloads.append((source, Path(target)))

    monkeypatch.setattr(harness, "start_proxy", start_proxy)
    monkeypatch.setattr(modules.harbor_agent, "TaskContainer", lambda environment, loop: task)
    monkeypatch.setattr(modules.harbor_agent, "ContainerTools", lambda environment, loop, **kwargs: tools)
    monkeypatch.setattr(modules.harbor_agent.openai, "OpenAI", lambda **kwargs: model)
    context = SimpleNamespace(metadata={}, n_input_tokens=0, n_output_tokens=0)
    asyncio.run(harness.run("play", Environment(), context))
    assert server.stopped and task.stopped and tools.installed
    return context, model, downloads


# ---------------------------------------------------------------------------
# The benchmark's agent loop (harness/agent.py)


@pytest.mark.unit
def test_agent_starts_without_a_system_prompt_and_rebuilds_it_each_week(monkeypatch) -> None:
    modules = load_modules(monkeypatch)
    model = Model(
        [
            response(tool_call("c1", "bash", {"command": "ls"})),
            response(tool_call("c2", "bash", {"command": "./novamind-operation next-week"})),
            response(tool_call("c3", "bash", {"command": "cat MEMORY.md"})),
        ]
    )
    notes = Notes()
    agent = modules.agent.BashAgent(
        model, "reef", "PROMPT", [{"name": "bash", "description": "d", "parameters": {}}], notes
    )

    first = agent.act(dashboard(0, 0, 1_000_000), {"day": 0, "cash": 1_000_000})
    assert first.tool == "bash" and first.arguments == {"command": "ls"}
    # Week 0 is played without a system prompt: the benchmark's conversation starts empty.
    assert [m["role"] for m in model.requests[0]["messages"]] == ["user"]

    second = agent.act("docs  novamind-operation", {"day": 0, "cash": 1_000_000})
    assert second.arguments == {"command": "./novamind-operation next-week"}
    tool_message = model.requests[1]["messages"][-1]
    assert tool_message == {
        "role": "tool",
        "content": "docs  novamind-operation",
        "tool_call_id": "c1",
        "name": "bash",
    }

    notes.text = "# Notes\nkeep prices"
    agent.act(dashboard(1, 7, 990_000), {"day": 7, "cash": 990_000})
    messages = model.requests[2]["messages"]
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"].startswith("PROMPT\n\n## Your MEMORY.md (auto-loaded)\n\n")
    assert messages[0]["content"].endswith("at the start of every day.\n\n# Notes\nkeep prices")
    assert messages[1]["content"].startswith("=== Week 1 Dashboard (Day 7) ===")
    assert agent.total_turns == 3 and agent.current_day == 7 and agent.turns_today == 1


@pytest.mark.unit
def test_agent_request_matches_the_benchmark_runner(monkeypatch) -> None:
    modules = load_modules(monkeypatch)
    tools = [{"name": "bash", "description": "Run bash", "parameters": {"type": "object", "properties": {}}}]
    model = Model([response(tool_call("c1", "bash", {"command": "ls"}), prompt_tokens=40, completion_tokens=5)])
    agent = modules.agent.BashAgent(model, "reef", "PROMPT", tools, Notes())

    agent.act("dashboard", {"day": 0})

    request = model.requests[0]
    assert set(request) == {
        "model",
        "messages",
        "tools",
        "tool_choice",
        "max_completion_tokens",
        "temperature",
        "reasoning_effort",
    }
    assert request["model"] == "reef" and request["tool_choice"] == "auto" and request["temperature"] == 1.0
    assert request["max_completion_tokens"] == 16384 and request["reasoning_effort"] == "none"
    assert request["tools"] == [{"type": "function", "function": tools[0]}]
    assert (agent.total_input_tokens, agent.total_output_tokens) == (40, 5)


@pytest.mark.unit
def test_agent_executes_the_first_tool_call_and_skips_the_rest(monkeypatch) -> None:
    modules = load_modules(monkeypatch)
    model = Model(
        [
            response(
                tool_call("c1", "read_file", {"path": "README.md"}), tool_call("c2", "glob_files", {"pattern": "*"})
            ),
            response(tool_call("c3", "bash", {"command": "ls"})),
        ]
    )
    agent = modules.agent.BashAgent(model, "reef", "PROMPT", [], Notes())

    action = agent.act("dashboard", {"day": 0})
    assert (action.tool, action.arguments) == ("read_file", {"path": "README.md"})

    agent.act("# README", {"day": 0})
    messages = model.requests[1]["messages"]
    assert messages[1]["role"] == "assistant" and [c["id"] for c in messages[1]["tool_calls"]] == ["c1", "c2"]
    assert messages[2] == {
        "role": "tool",
        "content": "[Skipped - only one tool per turn. Call glob_files again if needed.]",
        "tool_call_id": "c2",
        "name": "glob_files",
    }
    assert messages[3] == {"role": "tool", "content": "# README", "tool_call_id": "c1", "name": "read_file"}


@pytest.mark.unit
def test_agent_feeds_back_missing_tool_calls_and_invalid_json(monkeypatch) -> None:
    modules = load_modules(monkeypatch)
    model = Model(
        [
            response(content="Let me think."),
            response(tool_call("c1", "bash", '{"command": "echo \\$HOME"}')),
            response(tool_call("c2", "bash", {"command": "ls"})),
        ]
    )
    agent = modules.agent.BashAgent(model, "reef", "PROMPT", [], Notes())

    action = agent.act("dashboard", {"day": 0})

    assert action.arguments == {"command": "ls"} and agent.total_turns == 3
    messages = model.requests[2]["messages"]
    assert messages[1] == {"role": "assistant", "content": "Let me think."}
    assert messages[2] == {"role": "user", "content": modules.agent.NO_TOOL_FEEDBACK}
    assert messages[3]["role"] == "user" and messages[3]["content"].startswith(
        "Your previous response contained invalid JSON in the `bash` tool_call arguments.\nJSON decode error: "
    )
    assert "Shell-style escapes like \\$ or \\! are NOT valid JSON." in messages[3]["content"]
    # The response with the invalid call was never added to the conversation.
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "user"]


@pytest.mark.unit
def test_agent_retries_retryable_errors_and_feeds_back_the_others(monkeypatch) -> None:
    modules = load_modules(monkeypatch)
    waits = []
    monkeypatch.setattr(modules.agent.time, "sleep", waits.append)
    request = httpx.Request("POST", "http://127.0.0.1:1/v1/chat/completions")
    server_error = openai.APIStatusError("upstream", response=httpx.Response(503, request=request), body=None)
    model = Model(
        [
            server_error,
            openai.APITimeoutError(request=request),
            ValueError("context length exceeded"),
            response(tool_call("c1", "bash", {"command": "ls"})),
        ]
    )
    agent = modules.agent.BashAgent(model, "reef", "PROMPT", [], Notes())

    action = agent.act("dashboard", {"day": 0})

    assert action.arguments == {"command": "ls"}
    assert waits == [10, 20, 15]  # two retryable backoffs, then the linear wait after feedback
    fed_back = model.requests[3]["messages"][-1]
    assert fed_back["role"] == "user" and fed_back["content"].startswith(
        "The previous API request failed with a non-retryable error:\nValueError: context length exceeded\n\n"
    )
    assert agent.total_turns == 1


@pytest.mark.unit
def test_memory_is_appended_up_to_forty_thousand_characters(monkeypatch) -> None:
    modules = load_modules(monkeypatch)
    agent = modules.agent.BashAgent(Model([]), "reef", "PROMPT", [], Notes("x" * 40_010))
    prompt = agent.system_prompt_with_memory()
    assert prompt.endswith(
        "--- MEMORY.md TRUNCATED ---\nShowing first 40,000 of 40,010 characters. "
        "Use the read_file tool to see the full contents if needed."
    )
    assert (
        modules.agent.BashAgent(Model([]), "reef", "PROMPT", [], Notes("  \n")).system_prompt_with_memory() == "PROMPT"
    )


# ---------------------------------------------------------------------------
# The tools (harness/tools.py): the executor script, run here the way the container runs it


def run_tool(modules, workspace: Path, name: str, args: dict, *, bash_timeout: int = 20, port: int = 4242) -> dict:
    call = {"name": name, "args": args, "workspace": str(workspace), "port": port, "bash_timeout": bash_timeout}
    done = subprocess.run(
        [sys.executable, "-c", modules.tools.TOOL_SCRIPT],
        input=json.dumps(call),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.mark.unit
def test_tool_script_runs_bash_in_the_workspace_with_the_benchmark_output_format(monkeypatch, tmp_path) -> None:
    modules = load_modules(monkeypatch)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "hello.txt").write_text("hello\n")

    assert run_tool(modules, workspace, "bash", {"command": "cat hello.txt"}) == {"result": "hello\n"}
    assert run_tool(modules, workspace, "bash", {"command": "true"}) == {"result": "(no output)"}
    assert run_tool(modules, workspace, "bash", {"command": "echo out; echo err 1>&2; exit 3"}) == {
        "result": "out\n\n[stderr]\nerr\n\n[exit code: 3]"
    }
    env = run_tool(modules, workspace, "bash", {"command": 'echo "$HOME|$TMPDIR|$NOVAMIND_API_PORT|$PWD"'})["result"]
    assert env == f"{workspace}|{workspace}|4242|{workspace}\n"
    assert run_tool(modules, workspace, "bash", {"command": ""}) == {"result": "Error: No command provided"}
    long = run_tool(modules, workspace, "bash", {"command": "yes | head -c 40000"})["result"]
    assert len(long) < 31000 and "... (output truncated — exceeded 30,000 character limit) ..." in long


@pytest.mark.unit
def test_tool_script_times_out_bash_and_ends_the_run_only_for_next_week(monkeypatch, tmp_path) -> None:
    modules = load_modules(monkeypatch)
    workspace = tmp_path / "ws"
    workspace.mkdir()

    slow = run_tool(modules, workspace, "bash", {"command": "echo partial; sleep 5"}, bash_timeout=1)
    assert slow == {"result": "partial\n\nError: Command timed out after 1 seconds"}
    advancing = run_tool(
        modules, workspace, "bash", {"command": "./novamind-operation next-week; sleep 5"}, bash_timeout=1
    )
    assert advancing["timeout"] == "next_week timed out after 1s" and "partial_stdout" in advancing
    # The engine's own week-step timeout comes back as an exit code and a marker in the output.
    stalled = run_tool(
        modules, workspace, "bash", {"command": "./novamind-operation next-week; echo step_week_timeout; exit 1"}
    )
    assert stalled["timeout"] == "next_week engine-side timeout (step_week_timeout)"


@pytest.mark.unit
def test_tool_script_file_tools_keep_the_benchmark_semantics(monkeypatch, tmp_path) -> None:
    modules = load_modules(monkeypatch)
    workspace = tmp_path / "ws"
    (workspace / "docs").mkdir(parents=True)
    (workspace / "docs" / "cli.md").write_text("# CLI\nnext-week\nstatus\n")

    assert run_tool(modules, workspace, "write_file", {"path": "daily_scripts/w0.py", "content": "print(1)\n"}) == {
        "result": "File written: daily_scripts/w0.py (9 bytes)"
    }
    assert (
        run_tool(modules, workspace, "read_file", {"path": "docs/cli.md"})["result"]
        == "     1\t# CLI\n     2\tnext-week\n     3\tstatus\n     4\t"
    )
    assert (
        run_tool(modules, workspace, "read_file", {"path": "docs/cli.md", "offset": 2, "limit": 1})["result"]
        == "     2\tnext-week"
    )
    assert run_tool(modules, workspace, "read_file", {"path": "missing.md"}) == {
        "result": "Error: File not found: missing.md"
    }
    assert run_tool(modules, workspace, "read_file", {"path": "docs"}) == {"result": "Error: Not a file: docs"}
    assert run_tool(
        modules, workspace, "edit_file", {"path": "daily_scripts/w0.py", "old_string": "1", "new_string": "2"}
    ) == {"result": "File edited: daily_scripts/w0.py"}
    assert (workspace / "daily_scripts" / "w0.py").read_text() == "print(2)\n"
    assert run_tool(
        modules, workspace, "edit_file", {"path": "daily_scripts/w0.py", "old_string": "x", "new_string": "y"}
    ) == {"result": "Error: old_string not found in daily_scripts/w0.py"}
    assert run_tool(
        modules, workspace, "edit_file", {"path": "docs/cli.md", "old_string": "\n", "new_string": " "}
    ) == {"result": "Error: old_string found 3 times in docs/cli.md (must be unique)"}
    assert run_tool(modules, workspace, "search_files", {"pattern": "week|status", "path": "docs"})["result"] == (
        "docs/cli.md:2: next-week\ndocs/cli.md:3: status"
    )
    assert run_tool(modules, workspace, "search_files", {"pattern": "nothing"}) == {"result": "No matches found."}
    assert run_tool(modules, workspace, "search_files", {"pattern": "("}) == {
        "result": "Error: Invalid regex: missing ), unterminated subpattern at position 0"
    }
    assert run_tool(modules, workspace, "glob_files", {"pattern": "**/*.md"}) == {"result": "docs/cli.md"}
    assert run_tool(modules, workspace, "glob_files", {"pattern": "*.nope"}) == {"result": "No matching files."}
    assert run_tool(modules, workspace, "read_file", {"path": "../outside"}) == {
        "result": "Error: Path escapes workspace: ../outside"
    }
    assert run_tool(modules, workspace, "novamind", {}) == {"result": "Error: Unknown tool 'novamind'"}
    call = {"op": "memory", "workspace": str(workspace)}
    done = subprocess.run(
        [sys.executable, "-c", modules.tools.TOOL_SCRIPT], input=json.dumps(call), capture_output=True, text=True
    )
    assert json.loads(done.stdout) == {"memory": None}


@pytest.mark.unit
def test_container_tools_run_the_script_as_the_agent_user(monkeypatch) -> None:
    modules = load_modules(monkeypatch)
    calls = []

    class Environment:
        async def upload_file(self, source, target):
            calls.append(("upload", Path(source).read_text()[:60], target))

        async def exec(self, command, env=None, timeout_sec=None, user=None):
            calls.append(("exec", command, user, timeout_sec))
            if "chmod" in command:
                return SimpleNamespace(stdout="", stderr="", return_code=0)
            encoded = command.split("printf %s ", 1)[1].split(" |", 1)[0]
            call = json.loads(base64.b64decode(encoded))
            if call.get("op") == "memory":
                return SimpleNamespace(stdout='{"memory": "# notes"}\n', stderr="", return_code=0)
            if "next-week" in call["args"].get("command", ""):
                return SimpleNamespace(
                    stdout='{"timeout": "next_week timed out after 60s", "partial_stdout": "x"}\n',
                    stderr="",
                    return_code=0,
                )
            return SimpleNamespace(
                stdout=json.dumps({"result": f"ran {call['name']} in {call['workspace']} on {call['port']}"}) + "\n",
                stderr="",
                return_code=0,
            )

    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    try:
        tools = modules.tools.ContainerTools(
            Environment(), loop, workspace="/runs/run_1/agent_workspace", port=4242, user="agent", bash_timeout_s=60
        )
        tools.install()
        assert tools.execute("bash", {"command": "ls"}) == ("ran bash in /runs/run_1/agent_workspace on 4242", None)
        assert tools.execute("bash", {"command": "./novamind-operation next-week"}) == (
            "x",
            "next_week timed out after 60s",
        )
        assert tools.memory() == "# notes"
    finally:
        loop.call_soon_threadsafe(loop.stop)

    assert calls[0][0] == "upload" and calls[0][2] == "/tmp/reef-ceobench-tools.py" and "tool executor" in calls[0][1]
    assert calls[1] == ("exec", "chmod 644 /tmp/reef-ceobench-tools.py", None, None)
    _kind, command, user, timeout = calls[2]
    assert command.startswith("printf %s ") and command.endswith(
        " | base64 -d | /opt/ceobench/.venv/bin/python /tmp/reef-ceobench-tools.py"
    )
    assert user == "agent" and timeout == 180


# ---------------------------------------------------------------------------
# The task container (harness/harbor_agent.py)


@pytest.mark.unit
def test_task_container_starts_the_engine_and_reads_it_over_http(monkeypatch) -> None:
    modules = load_modules(monkeypatch)
    calls = []

    class Environment:
        async def exec(self, command, cwd=None, env=None, timeout_sec=None):
            calls.append((command, cwd, dict(env or {}), timeout_sec))
            if "ceobench-engine.py start" in command:
                return SimpleNamespace(
                    stdout='{"run_dir": "/runs/run_1", "workspace": "/runs/run_1/agent_workspace", "session_id": "s1", "port": 4242, "total_days": 14, "initial_cash": 1000000.0}\n',
                    stderr="",
                    return_code=0,
                )
            encoded = command.split("printf %s ", 1)[1].split(" |", 1)[0]
            request = json.loads(base64.b64decode(encoded))
            if "days" in request:
                return SimpleNamespace(
                    stdout=json.dumps({"system_prompt": f"P{request['days']}", "tools": []}) + "\n",
                    stderr="",
                    return_code=0,
                )
            if request["op"] == "query":
                return SimpleNamespace(
                    stdout='{"success": true, "data": {"rows": [{"mrr": 42.0}]}}\n', stderr="", return_code=0
                )
            if request["op"] == "dashboard":
                return SimpleNamespace(
                    stdout='{"dashboard": "=== Week 0 Dashboard (Day 0) ==="}\n', stderr="", return_code=0
                )
            return SimpleNamespace(stdout='{"error": "URLError: refused"}\n', stderr="", return_code=0)

    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    try:
        environ = {"SAAS_BENCH_SOCIAL_POST_LLM_PROVIDER": "openai", "OPENAI_API_KEY": "local", "HOME": "/home/x"}
        task = modules.harbor_agent.TaskContainer(Environment(), loop, environ=environ)
        session = task.start_engine(
            seed=7, days=14, cash=1e6, model="reef", reasoning_effort="none", tool_user="agent"
        )
        assert session.port == 4242 and session.total_days == 14
        assert task.contract(14, "reef").system_prompt == "P14"
        assert task.query("SELECT 1") == [{"mrr": 42.0}]
        assert task.dashboard() == "=== Week 0 Dashboard (Day 0) ==="
        assert task.status() == {"day": 0, "cash": 0, "subscribers": 0, "timed_out": False}  # the benchmark's fallback
    finally:
        loop.call_soon_threadsafe(loop.stop)

    command, cwd, env, timeout = calls[0]
    assert command == (
        "/opt/ceobench/.venv/bin/python /opt/ceobench-engine.py start --runs-dir /workspace/ceobench-runs"
        " --seed 7 --days 14 --cash 1000000.0 --model reef --reasoning-effort none --tool-user agent"
    )
    assert cwd == "/opt/ceobench" and timeout == 600
    assert env == {"SAAS_BENCH_SOCIAL_POST_LLM_PROVIDER": "openai", "OPENAI_API_KEY": "local"}
    # The prompt is built in the checkout, with the request piped straight into the interpreter.
    assert calls[1][1] == "/opt/ceobench" and calls[1][0].startswith("printf %s ")
    assert " | base64 -d | /opt/ceobench/.venv/bin/python -c " in calls[1][0]
    assert calls[2][0].endswith(
        "| python3 -c " + modules.harbor_agent.shlex.quote(modules.harbor_agent.ENGINE_READ_SCRIPT)
    )


@pytest.mark.unit
def test_engine_script_stop_reads_the_run_config_and_survives_a_missing_engine(tmp_path, monkeypatch, capsys) -> None:
    run_dir = tmp_path / "run_1"
    (run_dir / "agent_workspace").mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({"api_server_port": 1, "session_id": "s1"}))
    monkeypatch.setattr(sys, "argv", ["engine.py", "stop", "--run-dir", str(run_dir)])

    runpy.run_path(str(ENGINE_PATH), run_name="__main__")

    answer = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert answer["final"] == {"day": 0, "cash": 0, "subscribers": 0, "timed_out": False} and answer["copied"] is False


# ---------------------------------------------------------------------------
# Weeks, decisions, and the reward (harness/report.py)


@pytest.mark.unit
def test_dashboards_parse_the_week_cash_subscribers_seats_and_prices(monkeypatch) -> None:
    report = load_modules(monkeypatch).report
    (start,) = report.dashboards(dashboard(3, 21, 804_817, subscribers=110, seats=2, prices=(18, 39, 99)))
    assert start == report.WeekStart(3, 21, 804_817.0, 110, 2, (18.0, 39.0, 99.0))
    negative = "=== Week 37 Dashboard (Day 259) ===\n\nCash: -$66\nIndividual Subscribers: 1\nEnterprise Subscribed Seats: 0\n"
    assert report.dashboards(negative)[0].cash == -66.0 and report.dashboards(negative)[0].prices == (0.0, 0.0, 0.0)
    assert report.dashboards("no dashboard here") == []


@pytest.mark.unit
def test_week_start_run_rate_uses_the_engine_mrr_else_the_listed_price_estimate(monkeypatch) -> None:
    report = load_modules(monkeypatch).report
    assert start(report, 1, 7, 1.0, subscribers=30, seats=4, prices=(10.0, 39.0, 99.0)).run_rate == 30 * 10 + 4 * 99
    assert start(report, 1, 7, 1.0, subscribers=30, prices=(0.0, 39.0, 99.0)).run_rate == 30 * 39
    assert start(report, 1, 7, 1.0, subscribers=30).run_rate == 0.0
    assert start(report, 1, 7, 1.0, subscribers=30, prices=(10.0, 39.0, 99.0), mrr=207.0).run_rate == 207.0


@pytest.mark.unit
def test_valuation_counts_the_run_rate_over_the_remaining_horizon(monkeypatch) -> None:
    report = load_modules(monkeypatch).report
    assert report.valuation(100_000, 3_000, 40, 26) == 100_000 + 3_000 * 7 * 26 / 30
    assert report.valuation(100_000, 3_000, 5, 26) == 100_000 + 3_000 * 7 * 5 / 30
    assert report.valuation(100_000, 3_000, 0, 26) == 100_000 and report.valuation(100_000, 3_000, -1, 26) == 100_000


@pytest.mark.unit
def test_score_scale_clips_outliers_and_divides_by_the_running_median(monkeypatch) -> None:
    report = load_modules(monkeypatch).report
    scale = report.ScoreScale(clip=0.05, floor=0.003)
    assert scale.scale(-0.17) == -1.0  # clipped to -0.05, the only magnitude so far
    assert scale.scale(0.005) == pytest.approx(0.005 / 0.0275)  # median of 0.05 and 0.005
    assert scale.scale(-0.001) == pytest.approx(-0.001 / 0.005)
    tiny = report.ScoreScale(clip=0.05, floor=0.003)
    assert tiny.scale(0.0006) == pytest.approx(0.2)  # the floor keeps noise from blowing up to the cap
    with pytest.raises(ValueError):
        report.ScoreScale(clip=0)


@pytest.mark.unit
def test_week_credit_spans_the_following_weeks_and_is_cut_short_at_the_end(monkeypatch) -> None:
    report = load_modules(monkeypatch).report
    assert report.week_credit([-20_000, 5_000, 5_000, 5_000], 0.8) == pytest.approx(
        (-20_000 + 4_000 + 3_200 + 2_560) / 1e6
    )
    assert report.week_credit([-20_000, 5_000], 0.8) == pytest.approx((-20_000 + 4_000) / 1e6)


@pytest.mark.unit
def test_week_records_close_weeks_over_the_credit_window(monkeypatch) -> None:
    report = load_modules(monkeypatch).report
    records = report.WeekRecords(total_weeks=4, credit_weeks=2, discount=0.5)
    records.open_week(start(report, 0, 0, 1_000_000.0))
    records.add_turns(0, [turn("r-1", 10, 3, command="cat docs/cli.md"), turn("r-2", 20, 5)])
    records.open_week(start(report, 1, 7, 980_000.0, subscribers=30, prices=(10.0, 39.0, 99.0)))
    records.add_turns(1, [turn("r-3", 30, 7)])
    assert records.finished_weeks() == []  # week 0 needs week 2's opening state too
    records.open_week(start(report, 2, 14, 975_000.0, subscribers=40, prices=(10.0, 39.0, 99.0)))

    value_1 = 980_000 + 300 * 7 * 3 / 30
    value_2 = 975_000 + 400 * 7 * 2 / 30
    (finished,) = records.finished_weeks()
    assert finished == (
        0,
        980_000.0,
        pytest.approx(value_1),
        pytest.approx(((value_1 - 1e6) + 0.5 * (value_2 - value_1)) / 1e6),
    )
    assert records.turns == [
        {"receipt": "r-1", "tokens": 13, "week": 0, "decision": None},
        {"receipt": "r-2", "tokens": 25, "week": 0, "decision": "next_week"},
        {"receipt": "r-3", "tokens": 37, "week": 1, "decision": "next_week"},
    ]
    # The last weeks close with the episode's final cash, valued as cash alone.
    closed = {week: (cash_end, value_end) for week, cash_end, value_end, _credit in records.finished_weeks(970_000.0)}
    assert closed[2] == (970_000.0, 970_000.0)
    rows = records.summary(970_000.0)
    assert [(row["week"], row["turns"], row["decisions"]) for row in rows] == [(0, 2, 1), (1, 1, 1), (2, 0, 0)]


@pytest.mark.unit
def test_turn_decision_tells_company_changing_calls_from_reads(monkeypatch) -> None:
    report = load_modules(monkeypatch).report
    scripts: dict[str, str] = {}
    assert report.turn_decision(turn("r", 1, 1, command="./novamind-operation status"), scripts) is None
    assert report.turn_decision(turn("r", 1, 1, command="./novamind-operation next-week 'x'"), scripts) == "next_week"
    assert (
        report.turn_decision(
            turn("r", 1, 1, command='./novamind-operation python-c "nm.pricing.set_prices(A=18)"'), scripts
        )
        == "set_prices"
    )
    assert report.turn_decision(turn("r", 1, 1, tool=("read_file", {"path": "docs/cli.md"})), scripts) is None
    written = turn(
        "r",
        1,
        1,
        tool=(
            "write_file",
            {"path": "daily_scripts/week2.py", "content": "nm.marketing.set_daily_spend(operations=300)"},
        ),
    )
    assert report.turn_decision(written, scripts) is None and "daily_scripts/week2.py" in scripts
    assert (
        report.turn_decision(turn("r", 1, 1, command="./novamind-operation python week2.py"), scripts)
        == "set_daily_spend"
    )
    assert report.turn_decision(turn("r", 1, 1, command="python3 analysis.py"), scripts) == "script"
    assert report.turn_decision(turn("r", 1, 1, command="python3 -c 'print(1)'"), scripts) is None


@pytest.mark.unit
def test_training_pacer_waits_for_the_batches_the_reported_turns_filled(monkeypatch) -> None:
    report = load_modules(monkeypatch).report
    monkeypatch.setattr(report.time, "sleep", lambda _s: None)
    pacer = report.TrainingPacer(8, 60.0, Releases([3, 3, 4, 4]), SimpleNamespace(warning=lambda *a, **k: None))
    assert pacer.wait(1, 7) < 1.0  # no full batch yet: nothing to wait for
    pacer.wait(2, 8)  # one batch: waits until the release count moves from 3 to 4
    assert pacer.expected_releases(16) == 2


@pytest.mark.unit
def test_training_pacer_forgives_a_batch_the_trainer_never_commits(monkeypatch) -> None:
    report = load_modules(monkeypatch).report
    monkeypatch.setattr(report.time, "sleep", lambda _s: None)
    clock = iter(range(0, 10_000, 20))
    monkeypatch.setattr(report.time, "time", lambda: next(clock))
    warnings = []
    pacer = report.TrainingPacer(
        8, 30.0, Releases([0]), SimpleNamespace(warning=lambda msg, *a: warnings.append(msg % a))
    )
    pacer.wait(3, 16)
    assert warnings == ["week 3: trainer committed 0 of 2 expected releases after 30s; going on"]
    assert pacer.expected_releases(16) == 0  # the forgiven batches are not waited for again


# ---------------------------------------------------------------------------
# One episode through the Harbor agent (harness/harbor_agent.py)


@pytest.mark.unit
def test_harness_plays_the_episode_and_reports_each_week_to_reef(monkeypatch, tmp_path) -> None:
    modules = load_modules(monkeypatch)
    harness = create_harness(monkeypatch, modules, tmp_path)
    simulation = Simulation()
    task, tools = Task(simulation), Tools(simulation)
    responses = [
        response(tool_call("c1", "bash", {"command": "ls docs"}), prompt_tokens=100, completion_tokens=10),
        response(tool_call("c2", "bash", {"command": "./novamind-operation next-week 'grow' 990000 980000 970000"})),
        response(tool_call("c3", "bash", {"command": "./novamind-operation next-week 'hold' 980000 970000 960000"})),
    ]

    context, model, downloads = play(monkeypatch, modules, harness, task, tools, responses)

    assert task.start_kwargs == {
        "seed": 7,
        "days": 14,
        "cash": 1e6,
        "model": "reef",
        "reasoning_effort": "none",
        "tool_user": "agent",
    }
    assert [name for name, _args in tools.calls] == ["bash", "bash", "bash"]
    assert [[m["role"] for m in request["messages"]] for request in model.requests] == [
        ["user"],
        ["user", "assistant", "tool"],
        ["system", "user"],  # week 1: the conversation restarts from the benchmark's prompt
    ]
    assert model.requests[2]["messages"][0]["content"] == "PROMPT for 14 days"

    # Week 0 was reported when week 1 opened, week 1 with the engine's final cash; only the decision turns.
    payloads = [payload for scenario, payload in harness.client.calls]
    assert [payload["references"] for payload in payloads] == [["r-2"], ["r-3"]]
    value_1 = 990_000 + 50.0 * 7 * 1 / 30  # 5 subscribers at the engine's $10 MRR each, one week left
    assert payloads[0]["metadata"]["ceobench"]["credit"] == pytest.approx((value_1 - 1e6) / 1e6)
    assert payloads[0]["score"] == pytest.approx(-1.0)
    assert payloads[1]["metadata"]["ceobench"]["cash_end"] == 980_000.0
    assert task.queries and task.queries[0].startswith(
        "SELECT COALESCE(SUM(effective_price * COALESCE(seat_count, 1)), 0)"
    )
    assert task.commits == [7, 14]  # the workspace is snapshotted at week 1 and at the end

    ceobench = context.metadata["ceobench"]
    assert (ceobench["outcome"], ceobench["final_cash"], ceobench["sim_day"], ceobench["turns"]) == (
        "completed",
        980_000.0,
        14,
        3,
    )
    assert [(row["week"], row["cash_start"], row["run_rate"], row["decisions"]) for row in ceobench["weeks"]] == [
        (0, 1_000_000.0, 0.0, 1),
        (1, 990_000.0, 50.0, 1),
    ]
    assert context.metadata["reef"] == {
        "agent_record_ids": ["r-1", "r-2", "r-3"],
        "agent_record_tokens": [110, 13, 13],
        "agent_record_weeks": [0, 0, 1],
        "agent_record_decisions": [None, "next_week", "next_week"],
    }
    assert (context.n_input_tokens, context.n_output_tokens) == (120, 16)
    assert downloads == [("/workspace/ceobench-runs", tmp_path / "agent" / "ceobench")]
    logged = [json.loads(line) for line in (tmp_path / "agent" / "turns.jsonl").read_text().splitlines()]
    assert [(entry["day"], entry["tool"]) for entry in logged] == [
        (0, "_dashboard"),
        (0, "bash"),
        (0, "bash"),
        (7, "bash"),
    ]


@pytest.mark.unit
def test_reports_off_records_the_weeks_without_posting(monkeypatch, tmp_path) -> None:
    modules = load_modules(monkeypatch)
    harness = create_harness(monkeypatch, modules, tmp_path, env={"CEOBENCH_REPORTS": "0", "CEOBENCH_PACE_BATCH": "8"})
    assert harness.pacer is None
    simulation = Simulation()
    responses = [
        response(tool_call("c1", "bash", {"command": "./novamind-operation next-week 'a'"})),
        response(tool_call("c2", "bash", {"command": "./novamind-operation next-week 'b'"})),
    ]

    context, _model, _downloads = play(monkeypatch, modules, harness, Task(simulation), Tools(simulation), responses)

    assert harness.client.calls == []
    assert context.metadata["ceobench"]["reports"] is False
    assert [(row["week"], row["reported"], row["cash_end"]) for row in context.metadata["ceobench"]["weeks"]] == [
        (0, True, 990_000.0),
        (1, True, 980_000.0),
    ]


@pytest.mark.unit
def test_turns_longer_than_the_training_window_are_not_reported(monkeypatch, tmp_path) -> None:
    modules = load_modules(monkeypatch)
    harness = create_harness(monkeypatch, modules, tmp_path, env={"CEOBENCH_TRAIN_MAX_TOKENS": "50"})
    simulation = Simulation()
    responses = [
        response(tool_call("c1", "bash", {"command": "./novamind-operation next-week 'a'"}), prompt_tokens=60),
        response(tool_call("c2", "bash", {"command": "./novamind-operation next-week 'b'"}), prompt_tokens=20),
    ]

    play(monkeypatch, modules, harness, Task(simulation), Tools(simulation), responses)

    assert [payload["references"] for _s, payload in harness.client.calls] == [["r-2"]]


@pytest.mark.unit
def test_harness_ends_the_episode_when_the_engine_times_out(monkeypatch, tmp_path) -> None:
    modules = load_modules(monkeypatch)
    harness = create_harness(monkeypatch, modules, tmp_path)
    simulation = Simulation()
    tools = Tools(simulation)
    tools.execute = lambda name, arguments: SimpleNamespace(result="", timeout="next_week timed out after 3600s")
    responses = [response(tool_call("c1", "bash", {"command": "./novamind-operation next-week 'a'"}))]

    context, _model, _downloads = play(monkeypatch, modules, harness, Task(simulation), tools, responses)

    assert context.metadata["ceobench"]["outcome"] == "timeout"


@pytest.mark.unit
def test_gate_reads_the_engine_books_for_a_new_week_and_holds_for_training(monkeypatch, tmp_path) -> None:
    modules = load_modules(monkeypatch)
    harness = create_harness(
        monkeypatch, modules, tmp_path, env={"CEOBENCH_PACE_BATCH": "1", "CEOBENCH_PACE_TIMEOUT_S": "1"}
    )
    monkeypatch.setattr(modules.report.time, "sleep", lambda _s: None)
    harness.pacer = modules.report.TrainingPacer(1, 1.0, Releases([0, 0, 1]), harness.logger)
    simulation = Simulation()
    simulation.subscribers = 3
    task = Task(simulation)

    harness.enter_week(task, 0, 0)
    start = harness.records.weeks[0].start
    assert start.mrr == 30.0 and start.run_rate == 30.0
    task.query = lambda sql: (_ for _ in ()).throw(RuntimeError("no books"))
    harness.records.add_turns(0, [turn("r-1", 10, 3)])
    harness.enter_week(task, 1, 7)  # week 0 is reported; the pacer then waits for its batch
    assert harness.records.weeks[1].start.mrr is None
    assert [payload["references"] for _s, payload in harness.client.calls] == [["r-1"]]
    assert harness.pacer.expected_releases(1) == 1


# ---------------------------------------------------------------------------
# The task and the entry point


@pytest.mark.unit
def test_scorer_locates_the_single_run_and_prefers_the_checkpointed_database(tmp_path) -> None:
    score = load_score_module()
    runs = tmp_path / "runs"
    run_dir = runs / "run_abc123"
    live = run_dir / "agent_workspace" / "sessions" / "s1"
    live.mkdir(parents=True)
    (live / "world.nmdb").write_bytes(b"live")

    assert score.find_run_dir(runs) == run_dir
    assert score.find_world_db(run_dir) == live / "world.nmdb"
    (run_dir / "world.nmdb").write_bytes(b"checkpointed")
    assert score.find_world_db(run_dir) == run_dir / "world.nmdb"

    (runs / "run_def456").mkdir()
    with pytest.raises(RuntimeError, match="expected one run"):
        score.find_run_dir(runs)
    with pytest.raises(FileNotFoundError):
        score.find_run_dir(tmp_path / "empty")


@pytest.mark.unit
def test_entrypoint_runs_one_episode_and_drains_training(monkeypatch, capsys) -> None:
    calls = []

    class Lab:
        def __init__(self, path):
            self.path = Path(path)

        async def run(self, task, agent, tags=None):
            calls.append((self.path, Path(task), agent, tags))
            return SimpleNamespace(rewards={"reward": 0.9, "final_cash": 900000.0}, tags={}, uri="file:///trial")

    reef_eval = ModuleType("reef_eval")
    reef_eval.Lab = Lab
    monkeypatch.setitem(sys.modules, "reef_eval", reef_eval)
    for key, value in {
        "REEF_SERVICE_URL": "http://127.0.0.1:1/",  # nothing listens: the drain skips
        "REEF_SCENARIO": "ceobench-host-test",
        "REEF_TOKEN": "reef-local",
        "CEOBENCH_SEED": "43",
        "CEOBENCH_DAYS": "14",
    }.items():
        monkeypatch.setenv(key, value)

    runpy.run_path(str(EXAMPLE_DIR / "run.py"))

    # One episode: the policy adapts inside the episode it is scored on.
    (lab, task, agent, tags), *rest = calls
    assert not rest
    assert str(task.relative_to(EXAMPLE_DIR)) == "harbor"
    assert agent == {"name": "harness:HarborAgent", "model_name": "reef", "kwargs": {"seed": 43, "days": 14}}
    assert tags == {"seed": 43, "days": 14, "scenario": "ceobench-host-test"}
    assert lab == EXAMPLE_DIR / "work" / "lab"
    out = capsys.readouterr().out
    assert "reward" in out and "not reachable" in out


@pytest.mark.unit
def test_entrypoint_fails_when_harbor_reports_an_error(monkeypatch) -> None:
    class Lab:
        def __init__(self, _path):
            pass

        async def run(self, _task, _agent, tags=None):
            return SimpleNamespace(rewards={}, tags={"error": "environment failed"}, uri="file:///failed-trial")

    reef_eval = ModuleType("reef_eval")
    reef_eval.Lab = Lab
    monkeypatch.setitem(sys.modules, "reef_eval", reef_eval)
    monkeypatch.setenv("REEF_SERVICE_URL", "http://127.0.0.1:1/")

    with pytest.raises(RuntimeError, match="Harbor trial failed: environment failed"):
        runpy.run_path(str(EXAMPLE_DIR / "run.py"))


@pytest.mark.unit
def test_task_pins_the_upstream_commit_and_ships_no_credentials() -> None:
    dockerfile = (EXAMPLE_DIR / "harbor" / "environment" / "Dockerfile").read_text(encoding="utf-8")
    assert "ARG CEOBENCH_COMMIT=d2b7b32e5301a571b77f5f68bd1032adbcd5b464" in dockerfile
    assert "COPY engine.py /opt/ceobench-engine.py" in dockerfile
    patch = (EXAMPLE_DIR / "harbor" / "environment" / "reef.patch").read_text(encoding="utf-8")
    # The patch touches the engine only: the agent, its tools, and its runner live in harness/.
    assert "SAAS_BENCH_SIMULATOR_TIMEOUT_S" in patch and "agents/bash_agent" not in patch
    for path in EXAMPLE_DIR.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".sh", ".yaml", ".toml", ".md", ".patch"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            assert "sk-ant-" not in text and "AKIA" not in text, path
    serve = (EXAMPLE_DIR / "serve.yaml").read_text(encoding="utf-8")
    assert "batch-size: 8" in serve and "recipes.sao.recipe:SAORecipe" in serve


@pytest.mark.unit
def test_agent_preserves_provider_tool_fields_and_optional_sdk_usage(monkeypatch) -> None:
    modules = load_modules(monkeypatch)
    call = tool_call("c1", "bash", {"command": "ls"})
    call.model_extra = {"extra_content": {"provider": {"signature": "opaque"}}}
    completion = response(call)
    completion.usage = None
    agent = modules.agent.BashAgent(Model([completion]), "reef", "PROMPT", [], Notes())

    action = agent.act("dashboard", {"day": 0})

    assert action.tool == "bash"
    assert agent.total_input_tokens == 0
    assert agent.total_output_tokens == 0
    assert agent.conversation[-1].as_payload()["tool_calls"][0]["extra_content"] == call.model_extra["extra_content"]


@pytest.mark.unit
def test_agent_propagates_programming_errors_instead_of_retrying(monkeypatch) -> None:
    modules = load_modules(monkeypatch)
    agent = modules.agent.BashAgent(Model([AttributeError("broken client")]), "reef", "PROMPT", [], Notes())

    with pytest.raises(AttributeError, match="broken client"):
        agent.act("dashboard", {"day": 0})


@pytest.mark.unit
def test_tool_script_rejects_non_object_arguments(monkeypatch, tmp_path) -> None:
    modules = load_modules(monkeypatch)
    assert run_tool(modules, tmp_path, "bash", ["ls"]) == {"result": "Error: Tool arguments must be a JSON object"}
