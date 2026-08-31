"""OpenClaw-RL REEF Eval stream example: student sidecar, tasks, harness."""

from __future__ import annotations

import ast
import importlib
import json
import sys
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = REPO_ROOT / "recipes" / "openclawrl" / "examples" / "openclawrl"


@pytest.fixture
def harbor_runtime(monkeypatch):
    """Minimal Harbor protocol needed to exercise the example agent on CPU."""

    class BaseAgent:
        def __init__(self, *args, **kwargs):
            del args
            self.logs_dir = Path(kwargs.get("logs_dir", "/tmp"))
            self.model_name = kwargs.get("model_name")

    modules = {
        "harbor": ModuleType("harbor"),
        "harbor.agents": ModuleType("harbor.agents"),
        "harbor.agents.base": ModuleType("harbor.agents.base"),
        "harbor.environments": ModuleType("harbor.environments"),
        "harbor.environments.base": ModuleType("harbor.environments.base"),
        "harbor.models": ModuleType("harbor.models"),
        "harbor.models.agent": ModuleType("harbor.models.agent"),
        "harbor.models.agent.context": ModuleType("harbor.models.agent.context"),
    }
    modules["harbor.agents.base"].BaseAgent = BaseAgent
    modules["harbor.environments.base"].BaseEnvironment = object
    modules["harbor.models.agent.context"].AgentContext = object
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    yield


@pytest.fixture(scope="module")
def student_server():
    """Import the sidecar the way the judge image does: personas beside it."""
    sys.path.insert(0, str(EXAMPLE / "user_sim"))
    try:
        module = importlib.import_module("student_server")
        yield module
    finally:
        sys.path.remove(str(EXAMPLE / "user_sim"))


PROBLEM = {"index": 3, "question": "What is 15 - 6?", "answer": "9"}
CLEAN_REPLY = "so we take 15 minus 6 which gives 9, and to check it 9 plus 6 equals 15 again, so the answer is 9"
AI_REPLY = "**Final answer:** \n- 15 - 6 = 9\n- so the answer is 9"


def _settled(session, timeout_s: float = 5.0):
    """The state once the judge's reaction has landed.

    /reply returns before the reaction exists so that a slow persona LLM can
    never hold the HTTP response open; tests wait the same way the harness
    does.
    """
    deadline = time.monotonic() + timeout_s
    while not session.state()["ready"]:
        assert time.monotonic() < deadline, "reaction never landed"
        time.sleep(0.01)
    return session.state()


class TestStudentSession:
    def test_scripted_session_reaches_done_and_scores_first_reply(self, student_server):
        session = student_server.StudentSession(dict(PROBLEM))
        opening = session.state()
        assert "homework/3.txt" in opening["message"]
        assert not opening["done"]

        session.reply(CLEAN_REPLY)
        state = _settled(session)
        assert "append" in state["message"]  # clean first reply -> write request
        session.reply("Done - I appended the answers to homework/3.txt.")
        state = _settled(session)
        assert state["done"]
        assert state["message"] == student_server.DONE_MESSAGE

        final = session.final()
        assert final["reward"] == 1.0
        assert final["first_reply"] == CLEAN_REPLY
        assert final["violations"] == []

    def test_ai_styled_first_reply_gets_negative_reaction_and_zero_reward(self, student_server):
        session = student_server.StudentSession(dict(PROBLEM))
        session.reply(AI_REPLY)
        state = _settled(session)
        assert state["message"] == student_server.NEGATIVE_REACTION
        # A later clean rewrite continues the session but cannot rescue the
        # session score: sessions-to-adaptation reads the FIRST reply.
        session.reply(CLEAN_REPLY)
        final = session.final()
        assert final["reward"] == 0.0
        assert any("boxed" in v or "\\*\\*" in v for v in final["violations"])

    def test_missing_gold_answer_fails_strict_criterion(self, student_server):
        session = student_server.StudentSession(dict(PROBLEM))
        session.reply("we take 15 minus 6 and then 1 plus 1 makes 2, easy")
        final = session.final()
        assert final["reward"] == 0.0
        assert "no-gold-answer" in final["violations"]

    def test_dictating_student_llm_reply_is_rejected_for_the_scripted_line(self, student_server):
        # The persona LLM sometimes solves the homework itself; that dictated
        # turn must not become the next state, or the agent gets rewarded for
        # the student's solution. A reply carrying math or the gold answer
        # falls back to the decisive scripted reaction.
        session = student_server.StudentSession(dict(PROBLEM), user_llm_url="http://user-llm")
        session._react_llm = lambda: "ok so it's 15 minus 6 = 9, just write that down"  # dictation
        assert session._react(AI_REPLY) == student_server.NEGATIVE_REACTION

        session._react_llm = lambda: "nah that looks too robotic, redo it like a person wrote it"
        assert session._react(AI_REPLY).startswith("nah")  # style-only reply is kept

    def test_reply_dictates_solution_flags_math_and_the_answer(self, student_server):
        from personas import reply_dictates_solution

        assert reply_dictates_solution("it's 15 - 6 = 9", "9")
        assert reply_dictates_solution("the total is 9", "9")
        assert not reply_dictates_solution("that's too AI-looking, fix the style", "9")
        assert not reply_dictates_solution("great, now save it to the file", "9")

    def test_turn_cap_forces_done(self, student_server):
        session = student_server.StudentSession(dict(PROBLEM), max_turns=2)
        session.reply(AI_REPLY)
        state = session.reply(AI_REPLY)
        assert state["done"]

    def test_answer_present_matches_whole_numbers_only(self, student_server):
        assert student_server.answer_present("the answer is 1,234 total", "1234")
        assert not student_server.answer_present("results: 91 and 9.5", "9")
        assert student_server.answer_present("gives 9 apples", "9")

    def test_http_surface(self, student_server):
        session = student_server.StudentSession(dict(PROBLEM))
        server = ThreadingHTTPServer(("127.0.0.1", 0), student_server.build_handler(session))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://{server.server_address[0]}:{server.server_address[1]}"
        try:
            with urllib.request.urlopen(f"{base}/health", timeout=5) as response:
                assert response.status == 200
            body = json.dumps({"text": CLEAN_REPLY}).encode()
            request = urllib.request.Request(f"{base}/reply", data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=5) as response:
                reply = json.loads(response.read())
            assert "append" in reply["message"]
            with urllib.request.urlopen(f"{base}/final", timeout=5) as response:
                final = json.loads(response.read())
            assert final["reward"] == 1.0
        finally:
            server.shutdown()
            thread.join(timeout=5)


class TestStreamTasks:
    def test_the_stream_is_the_papers_exp1_length(self):
        tasks = sorted((EXAMPLE / "harbor-tasks").glob("gsm8k-s*"))
        assert len(tasks) == 72, "the committed stream is the paper's Exp. 1 length"

    def test_judge_images_build_from_the_shared_user_sim_image(self):
        """The sidecar is one image built from user_sim/, not a copy per task.

        Every task's Dockerfile.judge names openclawrl-user-sim at the tag
        run.sh derives from user_sim/'s content, so a change there without a
        re-stamp fails here instead of running a stale sidecar. The task
        environments carry only their own problem.json besides the build files.
        """
        import hashlib

        user_sim = EXAMPLE / "user_sim"
        digest = hashlib.sha256()
        for name in ("Dockerfile", "personas.py", "pyproject.toml", "student_server.py"):
            digest.update((user_sim / name).read_bytes())
        tag = digest.hexdigest()[:12]
        tasks = sorted((EXAMPLE / "harbor-tasks").glob("gsm8k-s*"))
        for task in tasks:
            env = task / "environment"
            judge = (env / "Dockerfile.judge").read_text()
            assert f"\nFROM openclawrl-user-sim:{tag}\n" in judge, task.name
            assert "COPY problem.json /judge/" in judge, task.name
            assert sorted(p.name for p in env.iterdir()) == [
                "Dockerfile",
                "Dockerfile.judge",
                "docker-compose.yaml",
                "problem.json",
            ], task.name
            compose = (env / "docker-compose.yaml").read_text()
            assert 'command: ["python", "-m", "student_server"]' in compose, task.name

    def test_tasks_are_stock_harbor_shape_with_a_judge_allowlist(self):
        for task in sorted((EXAMPLE / "harbor-tasks").glob("gsm8k-s*")):
            toml = (task / "task.toml").read_text()
            assert 'network_mode = "allowlist"' in toml
            assert '"judge"' in toml
            assert (task / "instruction.md").exists()
            assert (task / "tests" / "test.sh").exists()
            problem = json.loads((task / "environment" / "problem.json").read_text())
            assert set(problem) == {"index", "question", "answer"}
            assert f"gsm8k-s{problem['index']:03d}" == task.name
            assert "hermes-agent" in (task / "environment" / "Dockerfile").read_text()

    def test_container_scripts_are_stdlib_only(self):
        """The judge sidecar runs on python:3.12-slim: no third-party imports."""
        allowed_prefixes = ("personas",)  # baked beside the sidecar
        for path in (EXAMPLE / "user_sim" / "student_server.py",):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    modules = [node.module or ""]
                for module in modules:
                    root = module.split(".")[0]
                    assert root in sys.stdlib_module_names or root.startswith(allowed_prefixes), (
                        f"{path.name} imports non-stdlib module {module!r}"
                    )


class TestHarness:
    def test_package_imports_without_harbor(self):
        sys.path.insert(0, str(REPO_ROOT))
        try:
            module = importlib.import_module("recipes.openclawrl.examples.openclawrl.harness")
            assert "HermesStreamAgent" in module.__all__
        finally:
            sys.path.remove(str(REPO_ROOT))

    def test_agent_requires_reef_url(self, harbor_runtime):
        from recipes.openclawrl.examples.openclawrl.harness import HermesStreamAgent

        with pytest.raises(ValueError, match="reef_url"):
            HermesStreamAgent(logs_dir=Path("/tmp"))

    def test_session_loop_drives_hermes_against_the_judge(self, harbor_runtime, monkeypatch):
        import asyncio
        from types import SimpleNamespace

        from recipes.openclawrl.examples.openclawrl.harness import HermesStreamAgent

        problem = {"index": 0, "question": "What is 2+2?", "answer": "4"}
        states = iter(
            [
                {"message": "hey, do my homework", "done": False, "turn": 0},
                {"message": "HOMEWORK_DONE", "done": True, "turn": 1},
            ]
        )

        class FakeEnvironment:
            def __init__(self):
                self.commands = []

            async def exec(self, command):
                self.commands.append(command)
                if "cat /agent/problem.json" in command:
                    return SimpleNamespace(return_code=0, stdout=json.dumps(problem), stderr="")
                if "cat /reef-eval/state/scenario" in command:
                    return SimpleNamespace(return_code=0, stdout="", stderr="")
                if "$JUDGE_URL/state" in command or "$JUDGE_URL/reply" in command:
                    return SimpleNamespace(return_code=0, stdout=json.dumps(next(states)), stderr="")
                if "hermes -z" in command:
                    return SimpleNamespace(
                        return_code=0, stdout="<think>hm</think>we add 2 plus 2 equals 4, so 4", stderr=""
                    )
                return SimpleNamespace(return_code=0, stdout="", stderr="")

        agent = HermesStreamAgent(logs_dir=Path("/tmp"), reef_url="http://127.0.0.1:1")

        class FakeShim:
            def shutdown(self):
                pass

            def server_close(self):
                pass

        health = iter([(0, ""), (1, "")])
        stamped: list[str] = []
        monkeypatch.setattr(agent, "_start_shim", lambda _scenario, session: (stamped.append(session), FakeShim())[1])
        monkeypatch.setattr(agent, "_upstream_health", lambda: next(health))
        environment = FakeEnvironment()
        context = SimpleNamespace(metadata=None)
        asyncio.run(agent.run("ignored", environment, context))

        assert context.metadata["openclawrl"]["turns"] == 1
        assert context.metadata["openclawrl"]["failure"] is None
        assert context.metadata["openclawrl"]["scenario"].startswith("openclawrl-stream-")
        # One position, one tagged conversation: the processor correlates on
        # this instead of on a transcript hermes never resends.
        assert stamped == [f"{context.metadata['openclawrl']['scenario']}-s0"]
        hermes_calls = [c for c in environment.commands if "hermes -z" in c]
        assert len(hermes_calls) == 1
        assert "--resume" not in hermes_calls[0]  # first turn starts fresh
        config_writes = [c for c in environment.commands if ".hermes/config.yaml" in c]
        assert config_writes and "enabled: false" in config_writes[0]  # compression off


def test_reply_returns_before_the_reaction_exists(student_server):
    """The HTTP response must not wait on a persona-LLM generation.

    An egress proxy between the agent and this sidecar gives up on a response
    head long before a 32B finishes — harbor's gost defaults to 15s — and the
    agent then sees an empty reply it cannot tell from a crash. So /reply
    records the turn and returns; /state reports when the reaction landed.
    """
    session = student_server.StudentSession(dict(PROBLEM))
    slow = threading.Event()
    session._react = lambda text: (slow.wait(2.0), student_server.NEGATIVE_REACTION)[1]

    started = time.monotonic()
    state = session.reply(AI_REPLY)
    assert time.monotonic() - started < 0.5, "reply blocked on the reaction"
    assert state["turn"] == 1 and state["ready"] is False

    slow.set()
    assert _settled(session)["message"] == student_server.NEGATIVE_REACTION


def test_a_reaction_that_raises_still_unblocks_the_session(student_server):
    session = student_server.StudentSession(dict(PROBLEM))
    session._react = lambda text: (_ for _ in ()).throw(RuntimeError("judge down"))
    session.reply(AI_REPLY)
    # Degraded, not wedged: the turn is recorded and the session can go on.
    assert _settled(session)["ready"] is True
    assert session.final()["turns"] == 1
