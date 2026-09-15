"""One SPADE generation end to end against stand ins: propose, check, write, play both arms, split, report."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from recipes.beta.spade import OpenEnvCheck, OracleResult, PlayRecord, SmokeResult
from recipes.beta.spade.generation import (
    Checks,
    Designer,
    DesignerAnswer,
    Generation,
    GenerationError,
    GenerationRequest,
    Solver,
    experience_for,
    load_experience,
    main,
    mean_reward,
)
from reef.core.tasks import read_harbor_task, read_split_manifest
from reef.harness.client.tasks import TaskPlay

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
GYM_REPLY = (
    "Here it is.\n\n```python\n" + GUESS + "```\n\n```hint\nThe number is one of three; answer with one digit.\n```\n"
)
HARBOR_DOCUMENT = {
    "instruction": (
        "A service on this machine writes the port it listens on under /var/run. Find that file and write the "
        "port number, and nothing else, to /workspace/port.txt."
    ),
    "environment": {"Dockerfile": "FROM python:3.12-slim\nRUN echo 8471 > /var/run/app.port\nWORKDIR /workspace\n"},
    "tests": {
        "test.sh": '#!/bin/sh\nmkdir -p /logs/verifier\ntest "$(cat /workspace/port.txt)" = 8471 && echo 1 > /logs/verifier/reward.txt || echo 0 > /logs/verifier/reward.txt\n'
    },
    "solution": {"solve.sh": "#!/bin/sh\ncat /var/run/app.port > /workspace/port.txt\n"},
    "hint": "Look under /var/run for what the service left behind.",
}
HARBOR_REPLY = "```json\n" + json.dumps(HARBOR_DOCUMENT) + "\n```\n"
OPENENV_MODELS = """from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class GuessAction(Action):
    guess: int = Field(..., description="A number from 1 to 3")


class GuessObservation(Observation):
    message: str = Field("", description="What the environment said")
"""
OPENENV_ENVIRONMENT = """import random
import uuid

from openenv.core.env_server import Environment
from openenv.core.env_server.types import State

from openenv_task.models import GuessAction, GuessObservation


class GuessEnvironment(Environment):
    def __init__(self):
        self._state = State(episode_id=str(uuid.uuid4()), step_count=0)
        self.target = 0

    def reset(self, seed=None, **kwargs):
        self.target = random.Random(seed).randint(1, 3)
        self._state = State(episode_id=str(uuid.uuid4()), step_count=0)
        return GuessObservation(message="Guess a number from 1 to 3.", reward=0.0, done=False)

    def step(self, action: GuessAction):
        self._state.step_count += 1
        if action.guess == self.target:
            return GuessObservation(message="Right.", reward=1.0, done=True)
        return GuessObservation(message="Wrong.", reward=0.0, done=False)

    @property
    def state(self):
        return self._state
"""
OPENENV_DOCUMENT = {
    "instruction": "A hidden number between 1 and 3 was drawn. Find it with as few guesses as you can; each guess tells you only whether it was right.",
    "models": OPENENV_MODELS,
    "environment": OPENENV_ENVIRONMENT,
    "action_example": {"guess": 2},
    "hint": "There are only three candidates; a wrong guess rules one out.",
}
OPENENV_REPLY = "```json\n" + json.dumps(OPENENV_DOCUMENT) + "\n```\n"
REPLIES = {"gym": GYM_REPLY, "harbor": HARBOR_REPLY, "openenv": OPENENV_REPLY}


class StandInDesigner(Designer):
    """Answers each kind with a fixed reply (or a scripted one) and keeps the reports it gets."""

    def __init__(self, replies: Mapping[str, str] | None = None, scripted: Sequence[str] = ()) -> None:
        self.replies = dict(replies or REPLIES)
        self.scripted = list(scripted)
        self.calls: list[dict[str, object]] = []
        self.reports: list[dict[str, object]] = []

    def answer(self, messages, *, tags) -> DesignerAnswer:
        self.calls.append({"messages": list(messages), "tags": dict(tags)})
        text = self.scripted.pop(0) if self.scripted else self.replies[tags["kind"]]
        return DesignerAnswer(text=text, record_id=f"designer-{len(self.calls)}")

    def report(self, record_id, *, score, metadata) -> str:
        self.reports.append({"record_id": record_id, "score": score, "metadata": dict(metadata)})
        return f"report-{len(self.reports)}"


class StandInChecks(Checks):
    def __init__(self, *, refuse: str = "") -> None:
        self.refuse = refuse
        self.calls: list[tuple[str, object]] = []

    def smoke(self, code, *, seed, max_turns) -> SmokeResult:
        self.calls.append(("smoke", seed))
        return SmokeResult(is_runnable=self.refuse != "smoke", reason="broken step" if self.refuse == "smoke" else "")

    def oracle(self, task_path) -> OracleResult:
        self.calls.append(("oracle", task_path))
        if self.refuse == "oracle":
            return OracleResult(is_solvable=False, reason="the oracle scored 0", oracle_reward=0.0, nop_reward=0.0)
        return OracleResult(is_solvable=True, reason="", oracle_reward=1.0, nop_reward=0.0)

    def serving(self, task_path, action_example) -> OpenEnvCheck:
        self.calls.append(("serving", task_path))
        if self.refuse == "serving":
            return OpenEnvCheck(is_serving=False, reason="reset answered 500")
        return OpenEnvCheck(is_serving=True, reason="", first_observation="{}")


class StandInSolver(Solver):
    """Scripted rewards per kind and arm; records how each arm was asked for."""

    REWARDS = {
        ("gym", "plain"): 0.25,
        ("gym", "hint"): 0.75,
        ("harbor", "plain"): 1.0,
        ("harbor", "hint"): 1.0,
        ("openenv", "plain"): 0.0,
        ("openenv", "hint"): 0.0,
    }

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.episodes = 0

    def play(self, task_path, *, arm, plays, is_reporting, extra_instruction_paths, tags) -> tuple[TaskPlay, ...]:
        self.calls.append(
            {
                "task": task_path.name,
                "arm": arm,
                "plays": plays,
                "is_reporting": is_reporting,
                "extra": [Path(path) for path in extra_instruction_paths],
                "tags": dict(tags),
            }
        )
        kind = task_path.name.split("-")[0]
        reward = self.REWARDS[(kind, arm)]
        played = []
        for _ in range(plays):
            self.episodes += 1
            played.append(
                TaskPlay(
                    task_path=task_path,
                    name=task_path.name,
                    episode_id=f"episode-{self.episodes}",
                    reward=reward,
                    rewards={"reward": reward},
                    error="",
                    receipts=(f"rec-{self.episodes}",),
                    failed_calls=0,
                    report_agent_record_ids=(f"rep-{self.episodes}",) if is_reporting else (),
                    trial_uri=None,
                )
            )
        return tuple(played)


def request(**overrides: object) -> GenerationRequest:
    fields: dict[str, object] = {
        "description": "small deduction puzzles with hidden state",
        "skills": ("deduction",),
        "kinds": ("gym", "harbor", "openenv"),
        "count": 3,
        "generation": 4,
        "plays": 2,
        "hint_plays": 1,
        "eval_fraction": 0.3,
        "seed": 7,
    }
    fields.update(overrides)
    return GenerationRequest(**fields)  # type: ignore[arg-type]


def generation(tmp_path: Path, **parts: object) -> tuple[Generation, StandInDesigner, StandInSolver, StandInChecks]:
    designer = parts.get("designer") or StandInDesigner()
    solver = StandInSolver()
    checks = parts.get("checks") or StandInChecks()
    run = Generation(designer=designer, solver=solver, checks=checks, tasks_root=tmp_path / "tasks")  # type: ignore[arg-type]
    return run, designer, solver, checks  # type: ignore[return-value]


def test_one_generation_proposes_checks_writes_plays_splits_and_reports(tmp_path: Path) -> None:
    run, designer, solver, checks = generation(tmp_path)
    result = run.run(request())

    assert [proposal.kind for proposal in result.proposals] == ["gym", "harbor", "openenv"]
    assert all(proposal.is_written for proposal in result.proposals)
    names = [measure.name for measure in result.measures]
    assert names == ["gym-00004-000-deduction", "harbor-00004-001-deduction", "openenv-00004-002-deduction"]
    for index, measure in enumerate(result.measures):
        task = read_harbor_task(measure.task_path)
        assert task.digest == measure.digest and (measure.task_path / "solution" / "hint.txt").is_file()
        assert task.source_agent_record_ids == (
            f"designer-{index + 1}",
        ), "the Designer's receipt is the task's source"

    assert designer.calls[0]["tags"] == {
        "role": "designer",
        "generation": "4",
        "kind": "gym",
        "skill": "deduction",
    }
    assert designer.calls[0]["messages"][0]["role"] == "system"
    assert [step for step, _ in checks.calls] == ["smoke", "oracle", "serving"]
    assert checks.calls[1][1] == tmp_path / "tasks" / "harbor-00004-001-deduction"

    arms = [(call["task"], call["arm"], call["plays"], call["is_reporting"], call["extra"]) for call in solver.calls]
    assert arms[0] == ("gym-00004-000-deduction", "plain", 2, True, [])
    assert arms[1] == (
        "gym-00004-000-deduction",
        "hint",
        1,
        False,
        [tmp_path / "tasks" / "gym-00004-000-deduction" / "solution" / "hint.txt"],
    )
    assert all(
        call["tags"] == {"generation": "4", "kind": call["task"].split("-")[0], "skill": "deduction"}
        for call in solver.calls
    )

    by_name = {measure.name: measure for measure in result.measures}
    gym = by_name["gym-00004-000-deduction"]
    assert gym.plain_rewards == (0.25, 0.25) and gym.hint_rewards == (0.75,)
    assert gym.regret == 0.5 and gym.outcome == "frontier"
    assert by_name["harbor-00004-001-deduction"].outcome == "mastered"
    assert by_name["openenv-00004-002-deduction"].outcome == "out_of_reach"

    assert result.manifest_path == tmp_path / "tasks" / "manifest-00004.json"
    manifest = read_split_manifest(result.manifest_path)
    assert sorted([*manifest.train, *manifest.eval]) == names and len(manifest.eval) == 1

    assert [report["record_id"] for report in designer.reports] == ["designer-1", "designer-2", "designer-3"]
    assert [report["score"] for report in designer.reports] == [0.5, 0.0, 0.0]
    first = designer.reports[0]["metadata"]
    assert first["task"] == {"name": gym.name, "path": str(gym.task_path), "digest": gym.digest}
    assert first["outcome"] == "frontier" and first["regret"] == 0.5 and first["kind"] == "gym"
    assert [proposal.designer_report_id for proposal in result.proposals] == ["report-1", "report-2", "report-3"]

    document = json.loads(result.report_path.read_text())
    assert result.report_path == tmp_path / "tasks" / ".spade" / "generation-00004.json"
    assert document["manifest"] == str(result.manifest_path) and len(document["tasks"]) == 3
    assert document["request"]["count"] == 3 and "experience" not in document["request"]
    assert [record["name"] for record in document["experience"]] == names
    assert load_experience(result.report_path) == result.experience


def test_a_reply_the_kind_cannot_parse_is_refused_and_reported_as_zero(tmp_path: Path) -> None:
    designer = StandInDesigner(scripted=["no code here", GYM_REPLY])
    run, designer, solver, _ = generation(tmp_path, designer=designer)
    result = run.run(request(kinds=("gym",), count=2))
    first, second = result.proposals
    assert not first.is_written and first.refusal.startswith("gym reply refused:") and first.task_name is None
    assert second.is_written and second.task_name == "gym-00004-001-deduction"
    assert designer.reports[0]["score"] == 0.0 and designer.reports[0]["metadata"]["refusal"] == first.refusal
    assert not (tmp_path / "tasks" / "gym-00004-000-deduction").exists()
    assert [call["task"] for call in solver.calls] == ["gym-00004-001-deduction"] * 2


@pytest.mark.parametrize(
    ("kind", "refuse", "message"),
    [
        ("gym", "smoke", "smoke test refused: broken step"),
        ("harbor", "oracle", "oracle check refused: the oracle scored 0"),
        ("openenv", "serving", "serve check refused: reset answered 500"),
    ],
)
def test_a_task_its_check_refuses_never_stays_under_the_root(
    tmp_path: Path, kind: str, refuse: str, message: str
) -> None:
    run, designer, solver, _ = generation(tmp_path, checks=StandInChecks(refuse=refuse))
    result = run.run(request(kinds=(kind,), count=1))
    proposal = result.proposals[0]
    assert not proposal.is_written and message in proposal.refusal
    assert result.measures == () and result.manifest_path is None and solver.calls == []
    assert [entry.name for entry in (tmp_path / "tasks").iterdir() if entry.name != ".staging"] == [".spade"]
    assert designer.reports[0]["score"] == 0.0


def test_a_task_already_under_the_root_is_not_written_twice(tmp_path: Path) -> None:
    run, _, _, _ = generation(tmp_path)
    first = run.run(request(kinds=("gym",), count=1, generation=1))
    second = run.run(request(kinds=("gym",), count=1, generation=2))
    assert first.proposals[0].is_written
    assert second.proposals[0].refusal == "duplicate of a task already under the root"
    assert sorted(entry.name for entry in (tmp_path / "tasks").iterdir() if entry.name != ".staging") == [
        ".spade",
        "gym-00001-000-deduction",
        "manifest-00001.json",
    ]


def test_experience_for_puts_this_skill_and_the_frontier_first_and_caps_the_records() -> None:
    def record(name: str, skill: str, plain: float, hint: float) -> PlayRecord:
        return PlayRecord(name=name, skill=skill, return_without_hint=plain, return_with_hint=hint)

    records = [
        record("t-mastered", "deduction", 0.95, 1.0),
        record("t-other-skill", "planning", 0.5, 0.9),
        record("t-frontier-low", "deduction", 0.5, 0.6),
        record("t-frontier-high", "deduction", 0.3, 0.9),
        record("t-out", "deduction", 0.0, 0.2),
    ]
    chosen = experience_for(records, "deduction")
    assert [item.name for item in chosen] == [
        "t-frontier-high",
        "t-frontier-low",
        "t-out",
        "t-mastered",
        "t-other-skill",
    ]
    many = [record(f"t-{index:02d}", "deduction", 0.5, 0.6) for index in range(20)]
    assert len(experience_for(many, "deduction")) == 12


def test_mean_reward_counts_an_unscored_episode_as_a_loss(tmp_path: Path) -> None:
    def play(reward: float | None) -> TaskPlay:
        return TaskPlay(tmp_path, "t", "e", reward, {}, "", (), 0, (), None)

    assert mean_reward([play(1.0), play(None)]) == 0.5 and mean_reward([]) == 0.0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"description": " "}, "description must be non-empty text"),
        ({"skills": ()}, "skills must name at least one skill"),
        ({"kinds": ("gym", "board")}, "kinds must be among"),
        ({"count": 0}, "count must be at least 1"),
        ({"plays": 0}, "plays must be at least 1"),
        ({"eval_fraction": 1.0}, "eval_fraction must be in"),
    ],
)
def test_a_request_that_cannot_run_is_refused(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(GenerationError, match=message):
        request(**overrides)


def test_load_experience_refuses_a_file_that_is_not_a_report(tmp_path: Path) -> None:
    (tmp_path / "x.json").write_text("{}")
    with pytest.raises(GenerationError, match="holds no experience"):
        load_experience(tmp_path / "x.json")
    with pytest.raises(GenerationError, match="is not a generation report"):
        load_experience(tmp_path / "missing.json")


def test_main_runs_a_generation_from_the_command_line_and_prints_a_line_per_proposal(tmp_path: Path, capsys) -> None:
    designer, checks = StandInDesigner(), StandInChecks()
    solver = StandInSolver()
    status = main(
        [
            "--reef-url",
            "http://127.0.0.1:8900",
            "--scenario",
            "spade",
            "--model",
            "m",
            "--tasks-root",
            str(tmp_path / "tasks"),
            "--description",
            "small deduction puzzles with hidden state",
            "--skills",
            "deduction,planning",
            "--kinds",
            "gym,harbor",
            "--count",
            "2",
            "--generation",
            "3",
            "--plays",
            "1",
            "--hint-plays",
            "1",
        ],
        designer=designer,
        solver=solver,
        checks=checks,
    )
    assert status == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [line.get("task") for line in lines[:2]] == ["gym-00003-000-deduction", "harbor-00003-001-planning"]
    assert lines[0]["regret"] == 0.5 and lines[0]["outcome"] == "frontier"
    assert lines[2]["manifest"].endswith("manifest-00003.json") and lines[2]["report"].endswith(
        "generation-00003.json"
    )
    second = main(
        [
            "--reef-url",
            "http://127.0.0.1:8900",
            "--scenario",
            "spade",
            "--model",
            "m",
            "--tasks-root",
            str(tmp_path / "tasks"),
            "--description",
            "small deduction puzzles with hidden state",
            "--skills",
            "deduction",
            "--kinds",
            "openenv",
            "--count",
            "1",
            "--generation",
            "4",
            "--experience",
            str(tmp_path / "tasks" / ".spade" / "generation-00003.json"),
        ],
        designer=StandInDesigner(),
        solver=StandInSolver(),
        checks=StandInChecks(),
    )
    assert second == 0


def test_the_reef_designer_posts_its_request_options_and_keeps_the_receipt() -> None:
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from recipes.beta.spade.generation import ReefDesigner

    seen: list[dict[str, object]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            seen.append({"path": self.path, "body": body, "headers": {k.lower(): v for k, v in self.headers.items()}})
            if self.path == "/v1/chat/completions":
                answer = {
                    "choices": [{"message": {"role": "assistant", "content": "```python\nclass AEnv: pass\n```"}}]
                }
                receipt = "designer-rec-1"
            else:
                answer = {"agent_record_id": "designer-rep-1"}
                receipt = None
            payload = json.dumps(answer).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            if receipt:
                self.send_header("x-reef-agent-record-id", receipt)
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        designer = ReefDesigner(
            reef_url=f"http://127.0.0.1:{server.server_address[1]}",
            scenario="spade",
            model="m",
            token="tok",
            request_options={"reasoning_effort": "none"},
        )
        answer = designer.answer([{"role": "user", "content": "hi"}], tags={"role": "designer", "kind": "gym"})
        report_id = designer.report(answer.record_id, score=0.5, metadata={"kind": "gym"})
    finally:
        server.shutdown()
        server.server_close()
    assert (
        answer.text.startswith("```python") and answer.record_id == "designer-rec-1" and report_id == "designer-rep-1"
    )
    call, report = seen
    assert call["path"] == "/v1/chat/completions" and call["body"]["reasoning_effort"] == "none"
    assert call["body"]["model"] == "m" and call["body"]["messages"] == [{"role": "user", "content": "hi"}]
    assert call["headers"]["x-reef-scenario"] == "spade" and call["headers"]["x-reef-tag-kind"] == "gym"
    assert call["headers"]["authorization"] == "Bearer tok"
    assert report["path"] == "/reef/report" and report["body"]["references"] == ["designer-rec-1"]
    assert report["body"]["score"] == 0.5 and report["body"]["metadata"] == {"kind": "gym"}


def test_main_refuses_an_unknown_kind(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "--reef-url",
                "http://x",
                "--scenario",
                "s",
                "--model",
                "m",
                "--tasks-root",
                str(tmp_path),
                "--description",
                "d",
                "--skills",
                "a",
                "--kinds",
                "board",
            ],
            designer=StandInDesigner(),
            solver=StandInSolver(),
            checks=StandInChecks(),
        )
