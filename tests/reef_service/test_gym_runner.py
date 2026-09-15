"""The gym adapter end to end: an episode reaches reef-gym with the tree, the model sees the rules, the scorer reads the return."""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from reef.core.tasks import gym_task, write_harbor_task
from reef.harness.adapters import available_adapters, get_adapter
from reef.harness.episodes.model_binding import ModelBinding
from reef.harness.episodes.run import run_episode
from reef.harness.episodes.trajectory import reader_for
from reef.harness.runners.gym import chat_settings_from, evaluate, system_prompt_from
from reef.harness.tree.render import render_composition

GUESS = """class GuessEnv:
    def reset(self, seed=None):
        self.target = random.Random(seed).randint(1, 3)
        return "Guess a number from 1 to 3. Answer with \\\\boxed{n}.", {}

    def step(self, action):
        match = re.search(r"\\\\boxed\\{([^}]*)\\}", action)
        if not match:
            return "Use \\\\boxed{n}.", 0.0, False, False, {}
        if match.group(1).strip() == str(self.target):
            return "Right.", 1.0, True, False, {}
        return "Wrong.", 0.0, False, False, {}
"""
NODES = [
    ("rules", {"text": "Be brief."}),
    ("skill", {"name": "counting", "text": "# Counting\n\nCount from one."}),
    ("config", {"data": {"max_tokens": 64, "temperature": 0.0}}),
]


def target_for(seed: int) -> int:
    import random

    return random.Random(seed).randint(1, 3)


class _FakeModel(ThreadingHTTPServer):
    """Guesses 9 first, then the seed 7 target; a subclass may fail instead."""

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.requests: list[dict] = []
        self.thread = threading.Thread(target=self.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def status(self, body: dict) -> int:
        return 200

    def script(self, body: dict) -> dict:
        replies = [m for m in body["messages"] if m.get("role") == "assistant"]
        content = "I will try 9.\n\\boxed{9}" if not replies else f"Then \\boxed{{{target_for(7)}}}"
        return {
            "id": "fake",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
        }


class _BrokenModel(_FakeModel):
    def status(self, body: dict) -> int:
        return 500


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
        self.server.requests.append(body)  # type: ignore[attr-defined]
        payload = json.dumps(self.server.script(body)).encode()  # type: ignore[attr-defined]
        self.send_response(self.server.status(body))  # type: ignore[attr-defined]
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args) -> None:
        return None


@pytest.fixture
def fake_model():
    server = _FakeModel()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def broken_model():
    server = _BrokenModel()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def _launcher(tmp_path: Path) -> str:
    """The episode binary: this interpreter running the gym runner, like an installed reef-gym."""
    root = Path(__file__).resolve().parents[2]
    path = tmp_path / "reef-gym"
    path.write_text(
        f"#!{sys.executable}\nimport sys\nsys.path.insert(0, {str(root)!r})\n"
        "from reef.harness.runners.gym import main\nsys.exit(main())\n"
    )
    path.chmod(0o755)
    return str(path)


def _task(tmp_path: Path) -> Path:
    return write_harbor_task(gym_task(name="guess-007", code=GUESS, seed=7), tmp_path / "tasks")


def _episode(tmp_path: Path, model, task: Path, nodes=NODES):
    descriptor = get_adapter("gym")
    binding = ModelBinding(base_url=model.base_url, model="fake", api_key="dummy")
    files = render_composition([*nodes, *binding.compose_nodes(descriptor)], descriptor)
    return run_episode(descriptor, files, str(task), binary=_launcher(tmp_path), timeout=120.0)


def test_the_gym_adapter_is_bundled_and_declares_what_the_runner_reads() -> None:
    assert "gym" in available_adapters()
    descriptor = get_adapter("gym")
    assert descriptor.binary == "reef-gym" and descriptor.argv == ("--task", "{prompt}")
    assert descriptor.is_prompt_task_directory and not descriptor.self_isolating
    assert descriptor.trajectory_format == "gym-jsonl" and descriptor.trajectory_path == "gym/sessions"
    assert descriptor.writable_paths == ("gym/sessions",)
    assert type(reader_for("gym-jsonl")).__name__ == "GymSessionReader"
    files = render_composition([*NODES], descriptor)
    assert (
        files["gym/RULES.md"] == "Be brief.\n"
        and files["gym/skills/counting/SKILL.md"] == "# Counting\n\nCount from one.\n"
    )
    assert json.loads(files["gym/config.json"]) == {"max_tokens": 64, "temperature": 0.0}


def test_an_episode_plays_the_task_under_the_tree_and_the_scorer_reads_the_return(tmp_path: Path, fake_model) -> None:
    task = _task(tmp_path)
    result = _episode(tmp_path, fake_model, task)
    assert result.exit_code == 0, result.stderr
    assert [event["type"] for event in result.trajectory] == ["opening", "step", "step", "verifier"]
    opening, first, second, verifier = result.trajectory
    assert opening["system_prompt"] == "Be brief.\n\n# Skill: counting\n\n# Counting\n\nCount from one."
    assert opening["first_observation"].startswith("Guess a number")
    assert (first["action"], first["reward"], first["terminated"]) == ("\\boxed{9}", 0.0, False)
    assert second["reward"] == 1.0 and second["terminated"]
    assert verifier == {
        "type": "verifier",
        "task": str(task),
        "name": "guess-007",
        "reward": 1.0,
        "rewards": [0.0, 1.0],
        "terminated": True,
        "turns": 2,
        "failure": None,
    }
    assert evaluate(str(task), result) == 1.0
    assert result.residue == ()
    # The model saw the rules and the skill as the system prompt, then the gameplay prompt, with the config's knobs.
    request = fake_model.requests[0]
    assert request["messages"][0] == {"role": "system", "content": opening["system_prompt"]}
    assert request["messages"][1]["content"].startswith("You are playing a language game.")
    assert request["max_tokens"] == 64 and request["temperature"] == 0.0
    assert [m["role"] for m in fake_model.requests[1]["messages"]] == ["system", "user", "assistant", "user"]


def test_a_model_that_fails_makes_a_failed_episode_that_scores_zero(tmp_path: Path, broken_model) -> None:
    task = _task(tmp_path)
    result = _episode(tmp_path, broken_model, task)
    assert result.exit_code == 1 and "model:" in result.stderr
    verifier = result.trajectory[-1]
    assert verifier["type"] == "verifier" and verifier["reward"] == 0.0 and verifier["failure"].startswith("model:")
    assert evaluate(str(task), result) == 0.0


def test_an_edited_task_is_refused_and_scores_zero(tmp_path: Path, fake_model) -> None:
    task = _task(tmp_path)
    (task / "tests" / "env.py").write_text(GUESS.replace("randint(1, 3)", "randint(1, 1)"))
    result = _episode(tmp_path, fake_model, task)
    assert result.exit_code == 1 and "edited" in result.stderr
    assert [event["type"] for event in result.trajectory] == ["verifier"]
    assert result.trajectory[0]["failure"].startswith("task:") and evaluate(str(task), result) == 0.0
    assert fake_model.requests == []


def test_the_scorer_refuses_a_verifier_event_for_another_task_or_without_a_reward(tmp_path: Path, fake_model) -> None:
    task = _task(tmp_path)
    result = _episode(tmp_path, fake_model, task)
    with pytest.raises(ValueError, match="names task"):
        evaluate("another", result)
    with pytest.raises(ValueError, match="no finite reward"):
        evaluate(
            str(task),
            type(result)(
                exit_code=0,
                stdout="",
                stderr="",
                trajectory=({"type": "verifier", "task": str(task), "reward": "one"},),
                residue=(),
            ),
        )
    assert evaluate(str(task), type(result)(exit_code=1, stdout="", stderr="", trajectory=(), residue=())) == 0.0


def test_the_tree_readers_take_what_the_render_wrote(tmp_path: Path) -> None:
    assert system_prompt_from(tmp_path) is None
    (tmp_path / "RULES.md").write_text("  \n")
    (tmp_path / "skills" / "b").mkdir(parents=True)
    (tmp_path / "skills" / "a").mkdir(parents=True)
    (tmp_path / "skills" / "b" / "SKILL.md").write_text("Second.\n")
    (tmp_path / "skills" / "a" / "SKILL.md").write_text("First.\n")
    assert system_prompt_from(tmp_path) == "# Skill: a\n\nFirst.\n\n# Skill: b\n\nSecond."
    assert chat_settings_from(tmp_path / "config.json") == {"max_tokens": 4096, "timeout_s": 600.0}
    (tmp_path / "config.json").write_text(json.dumps({"max_tokens": 0, "timeout_s": 3, "temperature": 5, "other": 1}))
    assert chat_settings_from(tmp_path / "config.json") == {"max_tokens": 4096, "timeout_s": 3.0}
