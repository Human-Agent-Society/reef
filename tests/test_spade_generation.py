"""One SPADE generation end to end against stand ins: propose, write, check, play both arms, split, report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from spade_stand_ins import REPLY, StandInChecks, StandInDesigner, StandInReasoningAgent, reply_for

from recipes.beta.spade import PlayRecord
from recipes.beta.spade.designer import DesignerPrompt
from recipes.beta.spade.generation import (
    DesignerAnswer,
    Generation,
    GenerationError,
    GenerationRequest,
    experience_for,
    load_experience,
    main,
    mean_reward,
    reef_designer,
    reef_reasoning_agent,
)
from recipes.beta.spade.roles import REFUSAL_SCORE, HarborAgent, Role, RoleVersion, verifier_reward
from reef.core.tasks import read_harbor_task, read_split_manifest
from reef.harness.client.tasks import TaskPlay


def request(**overrides: object) -> GenerationRequest:
    fields: dict[str, object] = {
        "description": "shell tasks with hidden state under /var and /etc",
        "skills": ("inspection",),
        "count": 3,
        "generation": 4,
        "plays": 2,
        "hint_plays": 1,
        "eval_fraction": 0.3,
        "seed": 7,
    }
    fields.update(overrides)
    return GenerationRequest(**fields)  # type: ignore[arg-type]


def generation(
    tmp_path: Path, **parts: object
) -> tuple[Generation, StandInDesigner, StandInReasoningAgent, StandInChecks]:
    designer = parts.get("designer") or StandInDesigner()
    reasoning_agent = parts.get("reasoning_agent") or StandInReasoningAgent()
    checks = parts.get("checks") or StandInChecks()
    options = {name: value for name, value in parts.items() if name not in ("designer", "reasoning_agent", "checks")}
    run = Generation(designer=designer, reasoning_agent=reasoning_agent, checks=checks, tasks_root=tmp_path / "tasks", **options)  # type: ignore[arg-type]
    return run, designer, reasoning_agent, checks  # type: ignore[return-value]


def written(root: Path) -> list[str]:
    return sorted(entry.name for entry in root.iterdir() if entry.name != ".staging")


def test_one_generation_proposes_writes_checks_plays_splits_and_reports(tmp_path: Path) -> None:
    run, designer, reasoning_agent, checks = generation(tmp_path)
    result = run.run(request(skills=("inspection", "repair")))

    assert [proposal.skill for proposal in result.proposals] == ["inspection", "repair", "inspection"]
    assert all(proposal.is_written for proposal in result.proposals)
    names = [measure.name for measure in result.measures]
    assert names == ["harbor-00004-000-inspection", "harbor-00004-001-repair", "harbor-00004-002-inspection"]
    for index, measure in enumerate(result.measures):
        task = read_harbor_task(measure.task_path)
        assert task.digest == measure.digest and (measure.task_path / "solution" / "hint.txt").is_file()
        assert task.source_agent_record_ids == (
            f"designer-{index + 1}",
        ), "the Designer's receipt is the task's source"

    assert designer.calls[0]["tags"] == {"role": "designer", "generation": "4", "skill": "inspection"}
    assert designer.calls[0]["messages"][0]["role"] == "system"
    assert checks.calls == [tmp_path / "tasks" / name for name in names]

    arms = [
        (call["task"], call["arm"], call["plays"], call["is_reporting"], call["extra"])
        for call in reasoning_agent.calls
    ]
    assert arms[0] == (
        "harbor-00004-000-inspection",
        "plain",
        2,
        False,
        [],
    ), "the plain arm is reported after the play"
    assert arms[1] == (
        "harbor-00004-000-inspection",
        "hint",
        1,
        False,
        [tmp_path / "tasks" / "harbor-00004-000-inspection" / "solution" / "hint.txt"],
    )
    assert all(
        call["tags"] == {"generation": "4", "skill": call["task"].split("-")[-1]} for call in reasoning_agent.calls
    )

    first = result.measures[0]
    assert first.plain_rewards == (0.25, 0.25) and first.hint_rewards == (0.75,)
    assert first.regret == 0.5 and first.outcome == "frontier"
    assert first.record.instruction_excerpt.startswith("A service on this machine")
    reports = reasoning_agent.reports
    assert [report["plays"] for report in reports] == [plays for name in names for plays in ([name] * 2, [name])]
    assert reports[0]["scores"] == [0.25, 0.25], "the verifier reward is the Reasoning Agent's score"
    assert reports[0]["metadata"] == {"arm": "plain", "generation": 4, "designer_version": None}
    assert reports[1]["metadata"] == {"arm": "hint", "generation": 4, "designer_version": None}
    assert [play.report_agent_record_ids for play in first.plain_plays] == [("rep-1",), ("rep-2",)]
    assert [play.report_agent_record_ids for play in first.hint_plays] == [("rep-3",)], "the hint arm is reported too"

    assert result.manifest_path == tmp_path / "tasks" / "manifest-00004.json"
    manifest = read_split_manifest(result.manifest_path)
    assert sorted([*manifest.train, *manifest.eval]) == names and len(manifest.eval) == 1

    assert [report["record_id"] for report in designer.reports] == ["designer-1", "designer-2", "designer-3"]
    assert [report["score"] for report in designer.reports] == [0.5, 0.5, 0.5]
    metadata = designer.reports[0]["metadata"]
    assert metadata["task"] == {"name": first.name, "path": str(first.task_path), "digest": first.digest}
    assert metadata["outcome"] == "frontier" and metadata["regret"] == 0.5 and metadata["skill"] == "inspection"
    assert metadata["proposals"] == 3, "the Designer's group is the whole generation"
    assert designer.reports[1]["metadata"]["proposals"] == 3 and metadata["designer_version"] is None
    assert metadata["opponent"] == {"role": "reasoning_agent", "version": None}
    assert designer.reports[0]["feedback"] == {
        "task": first.name,
        "refusal": "",
        "round": {"generation": 4, "mean_regret": 0.5, "measured": 3, "refused": 0, "previous": None},
        "outcome": "frontier",
        "regret": 0.5,
        "return_without_hint": 0.25,
        "return_with_hint": 0.75,
    }
    assert [proposal.designer_report_id for proposal in result.proposals] == ["report-1", "report-2", "report-3"]

    document = json.loads(result.report_path.read_text())
    assert result.report_path == tmp_path / "tasks" / ".spade" / "generation-00004.json"
    assert document["manifest"] == str(result.manifest_path) and len(document["tasks"]) == 3
    assert document["request"]["count"] == 3 and "experience" not in document["request"]
    assert document["request"]["prompt_entries"] == ["designer-system", "designer-rules"]
    assert document["request"]["designer_version"] is None and "prompt" not in document["request"]
    assert [record["name"] for record in document["experience"]] == names
    assert load_experience(result.report_path) == result.experience


def test_a_generation_without_skills_takes_the_description_as_the_target(tmp_path: Path) -> None:
    run, designer, reasoning_agent, _ = generation(tmp_path)
    result = run.run(request(skills=(), count=2))
    assert [measure.name for measure in result.measures] == ["harbor-00004-000", "harbor-00004-001"]
    assert designer.calls[0]["tags"] == {"role": "designer", "generation": "4"}
    prompt = designer.calls[0]["messages"][1]["content"]
    assert "that tests: shell tasks with hidden state under /var and /etc." in prompt
    assert all("skill" not in call["tags"] for call in reasoning_agent.calls)
    assert "skill" not in designer.reports[0]["metadata"] and result.experience[0].skill is None
    document = json.loads(result.report_path.read_text())
    assert document["tasks"][0]["skill"] is None and load_experience(result.report_path) == result.experience


def test_a_reply_the_parser_refuses_is_reported_as_zero_and_the_generation_goes_on(tmp_path: Path) -> None:
    designer = StandInDesigner(scripted=["no json here", reply_for(2)])
    run, designer, reasoning_agent, _ = generation(tmp_path, designer=designer)
    result = run.run(request(count=2))
    first, second = result.proposals
    assert not first.is_written and first.refusal.startswith("reply refused:") and first.task_name is None
    assert second.is_written and second.task_name == "harbor-00004-001-inspection"
    assert (
        designer.reports[0]["score"] == REFUSAL_SCORE and designer.reports[0]["metadata"]["refusal"] == first.refusal
    )
    assert designer.reports[0]["feedback"]["round"] == {
        "generation": 4,
        "mean_regret": 0.5,
        "measured": 1,
        "refused": 1,
        "previous": None,
    }
    assert not (tmp_path / "tasks" / "harbor-00004-000-inspection").exists()
    assert [call["task"] for call in reasoning_agent.calls] == ["harbor-00004-001-inspection"] * 2


def test_a_task_the_oracle_check_refuses_never_stays_under_the_root(tmp_path: Path) -> None:
    run, designer, reasoning_agent, _ = generation(tmp_path, checks=StandInChecks(is_solvable=False))
    result = run.run(request(count=1))
    proposal = result.proposals[0]
    assert not proposal.is_written and proposal.refusal == "oracle check refused: the oracle scored 0"
    assert result.measures == () and result.manifest_path is None and reasoning_agent.calls == []
    assert written(tmp_path / "tasks") == [".spade"]
    assert designer.reports[0]["score"] == REFUSAL_SCORE
    assert designer.reports[0]["feedback"]["round"]["mean_regret"] is None


def test_a_task_the_reasoning_agent_could_not_play_is_refused_not_scored(tmp_path: Path) -> None:
    reasoning_agent = StandInReasoningAgent(error="Failed to start tmux session. Error: None")
    run, designer, _, _ = generation(tmp_path, reasoning_agent=reasoning_agent)
    result = run.run(request(count=1))
    proposal = result.proposals[0]
    assert not proposal.is_written
    assert proposal.refusal == "the Reasoning Agent could not play the task: Failed to start tmux session. Error: None"
    assert result.measures == () and result.manifest_path is None
    assert [call["arm"] for call in reasoning_agent.calls] == [
        "plain"
    ], "the hint arm is not played for a task that cannot run"
    assert not (tmp_path / "tasks" / "harbor-00004-000-inspection").exists()
    assert designer.reports[0]["score"] == REFUSAL_SCORE
    assert designer.reports[0]["metadata"]["refusal"] == proposal.refusal


def test_a_task_already_under_the_root_is_not_written_twice(tmp_path: Path) -> None:
    run, _, _, _ = generation(tmp_path, designer=StandInDesigner(scripted=[REPLY, REPLY]))
    first = run.run(request(count=1, generation=1))
    second = run.run(request(count=1, generation=2))
    assert first.proposals[0].is_written
    assert second.proposals[0].refusal == "duplicate of a task already under the root"
    assert written(tmp_path / "tasks") == [".spade", "harbor-00001-000-inspection", "manifest-00001.json"]


def test_experience_for_puts_this_skill_and_the_frontier_first_and_caps_the_records() -> None:
    def record(name: str, skill: str | None, plain: float, hint: float) -> PlayRecord:
        return PlayRecord(name=name, skill=skill, return_without_hint=plain, return_with_hint=hint)

    records = [
        record("t-mastered", "inspection", 0.95, 1.0),
        record("t-other-skill", "repair", 0.5, 0.9),
        record("t-frontier-low", "inspection", 0.5, 0.6),
        record("t-frontier-high", "inspection", 0.3, 0.9),
        record("t-out", "inspection", 0.0, 0.2),
    ]
    chosen = experience_for(records, "inspection")
    assert [item.name for item in chosen] == [
        "t-frontier-high",
        "t-frontier-low",
        "t-out",
        "t-mastered",
        "t-other-skill",
    ]
    many = [record(f"t-{index:02d}", "inspection", 0.5, 0.6) for index in range(20)]
    assert len(experience_for(many, "inspection")) == 12
    unskilled = [record("t-a", None, 0.5, 0.6), record("t-b", None, 0.0, 0.2), record("t-c", "repair", 0.5, 0.9)]
    assert [item.name for item in experience_for(unskilled, None)] == ["t-c", "t-a", "t-b"]


def test_mean_reward_counts_an_unscored_run_as_a_loss_and_skips_an_episode_that_never_ran(tmp_path: Path) -> None:
    def play(reward: float | None, error: str = "") -> TaskPlay:
        return TaskPlay(tmp_path, "t", "e", reward, {}, error, (), 0, (), None)

    assert mean_reward([play(1.0), play(None)]) == 0.5 and mean_reward([]) == 0.0
    assert mean_reward([play(1.0), play(None, "the trial raised")]) == 1.0
    assert mean_reward([play(None, "the trial raised")]) == 0.0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"description": " "}, "description must be non-empty text"),
        ({"skills": ("",)}, "skills must be a tuple of skill names"),
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
    reasoning_agent = StandInReasoningAgent()
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
            "shell tasks with hidden state under /var and /etc",
            "--skills",
            "inspection,repair",
            "--count",
            "2",
            "--generation",
            "3",
            "--plays",
            "1",
            "--hint-plays",
            "1",
        ],
        designer=StandInDesigner(),
        reasoning_agent=reasoning_agent,
        checks=StandInChecks(),
    )
    assert status == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [line.get("task") for line in lines[:2]] == ["harbor-00003-000-inspection", "harbor-00003-001-repair"]
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
            "shell tasks with hidden state under /var and /etc",
            "--skills",
            "inspection",
            "--count",
            "1",
            "--generation",
            "4",
            "--experience",
            str(tmp_path / "tasks" / ".spade" / "generation-00003.json"),
        ],
        designer=StandInDesigner(scripted=[reply_for(9)]),
        reasoning_agent=StandInReasoningAgent(),
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
                answer = {"choices": [{"message": {"role": "assistant", "content": REPLY}}]}
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
            timeout_s=5.0,
        )
        assert designer.client.timeout_s == 5.0
        answer = designer.answer([{"role": "user", "content": "hi"}], tags={"role": "designer", "skill": "inspection"})
        report_id = designer.report(answer.record_id, score=0.5, metadata={"skill": "inspection"})
    finally:
        server.shutdown()
        server.server_close()
    assert answer.text == REPLY and answer.record_id == "designer-rec-1" and report_id == "designer-rep-1"
    call, report = seen
    assert call["path"] == "/v1/chat/completions" and call["body"]["reasoning_effort"] == "none"
    assert call["body"]["model"] == "m" and call["body"]["messages"] == [{"role": "user", "content": "hi"}]
    assert call["headers"]["x-reef-scenario"] == "spade" and call["headers"]["x-reef-tag-skill"] == "inspection"
    assert call["headers"]["authorization"] == "Bearer tok"
    assert report["path"] == "/reef/report" and report["body"]["references"] == ["designer-rec-1"]
    assert report["body"]["score"] == 0.5 and report["body"]["metadata"] == {"skill": "inspection"}


def test_a_designer_call_that_times_out_is_a_generation_error() -> None:
    import socket

    from recipes.beta.spade.generation import ReefDesigner

    # A listener that never answers: the client's own timeout is what ends the call.
    silent = socket.socket()
    silent.bind(("127.0.0.1", 0))
    silent.listen(1)
    try:
        designer = ReefDesigner(
            reef_url=f"http://127.0.0.1:{silent.getsockname()[1]}", scenario="spade", model="m", timeout_s=0.5
        )
        with pytest.raises(GenerationError, match=r"did not complete within 0\.5 s"):
            designer.answer([{"role": "user", "content": "hi"}], tags={"skill": "inspection"})
    finally:
        silent.close()
    with pytest.raises(GenerationError, match="timeout_s must be a positive number"):
        ReefDesigner(reef_url="http://127.0.0.1:1", scenario="spade", model="m", timeout_s=0)


def test_main_refuses_a_request_it_cannot_run(tmp_path: Path) -> None:
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
                "--count",
                "0",
            ],
            designer=StandInDesigner(),
            reasoning_agent=StandInReasoningAgent(),
            checks=StandInChecks(),
        )


def test_main_points_the_designer_at_its_own_service_and_model(tmp_path: Path, monkeypatch) -> None:
    seen: list[dict[str, object]] = []

    class RecordingDesigner(StandInDesigner):
        def __init__(self, **options: object) -> None:
            seen.append(options)
            super().__init__()

    monkeypatch.setattr("recipes.beta.spade.generation.ReefDesigner", RecordingDesigner)
    shared = ["--reef-url", "http://127.0.0.1:8900", "--scenario", "spade", "--model", "m", "--token", "t"]
    request = ["--description", "shell tasks", "--count", "1", "--plays", "1", "--tasks-root"]
    own = [
        "--designer-reef-url",
        "http://127.0.0.1:8901",
        "--designer-scenario",
        "designer",
        "--designer-model",
        "strong",
    ]
    stand_ins: dict[str, object] = {"reasoning_agent": StandInReasoningAgent(), "checks": StandInChecks()}
    assert main([*shared, *request, str(tmp_path / "shared")], **stand_ins) == 0
    assert main([*shared, *request, str(tmp_path / "own"), *own, "--designer-token", "t2"], **stand_ins) == 0
    keys = ("reef_url", "scenario", "model", "token")
    assert [tuple(options[key] for key in keys) for options in seen] == [
        ("http://127.0.0.1:8900", "spade", "m", "t"),
        ("http://127.0.0.1:8901", "designer", "strong", "t2"),
    ]


def test_the_designer_score_is_the_raw_regret_so_a_hint_that_hurt_scores_below_zero(tmp_path: Path) -> None:
    run, designer, _, _ = generation(tmp_path, reasoning_agent=StandInReasoningAgent(plain=0.75, hint=0.25))
    result = run.run(request(count=1))
    assert result.measures[0].regret == -0.5 and result.measures[0].outcome == "frontier"
    assert designer.reports[0]["score"] == -0.5 and designer.reports[0]["feedback"]["regret"] == -0.5


def test_a_generation_takes_its_prompt_and_the_versions_it_is_measured_against(tmp_path: Path) -> None:
    run, designer, reasoning_agent, _ = generation(tmp_path)
    prompt = DesignerPrompt(
        system="You design Harbor tasks about /var.", rules="RULES:\n- at most {turn_limit} commands, {braces} kept"
    )
    result = run.run(
        request(
            count=1,
            prompt=prompt,
            designer_version=RoleVersion("release", "abc123"),
            agent_version=RoleVersion("runtime", "load-7"),
        )
    )
    messages = designer.calls[0]["messages"]
    assert messages[0]["content"] == "You design Harbor tasks about /var."
    assert "- at most 12 commands, {braces} kept" in messages[1]["content"]
    metadata = designer.reports[0]["metadata"]
    assert metadata["designer_version"] == {"kind": "release", "id": "abc123"}
    assert metadata["opponent"] == {"role": "reasoning_agent", "version": {"kind": "runtime", "id": "load-7"}}
    assert reasoning_agent.reports[0]["metadata"] == {
        "arm": "plain",
        "generation": 4,
        "designer_version": {"kind": "release", "id": "abc123"},
    }
    document = json.loads(result.report_path.read_text())
    assert document["request"]["designer_version"] == {"kind": "release", "id": "abc123"}
    assert document["request"]["agent_version"] == {"kind": "runtime", "id": "load-7"}


def test_a_generation_that_does_not_report_the_agent_holds_the_plain_plays(tmp_path: Path) -> None:
    run, _, reasoning_agent, _ = generation(tmp_path, is_reporting_agent=False)
    result = run.run(request(count=1))
    assert reasoning_agent.reports == []
    plays = result.measures[0].plain_plays
    assert [play.receipts for play in plays] == [("rec-1",), ("rec-2",)] and not any(
        play.is_reported for play in plays
    )
    reported = reasoning_agent.report_plays(plays, reward=verifier_reward, metadata={"arm": "plain", "generation": 4})
    assert [play.report_agent_record_ids for play in reported] == [("rep-1",), ("rep-2",)]


def test_every_designer_report_waits_for_the_end_of_the_generation_and_carries_the_last_round(tmp_path: Path) -> None:
    class CountingDesigner(StandInDesigner):
        def answer(self, messages, *, tags) -> DesignerAnswer:
            assert self.reports == [], "no report goes out while proposals are still being made"
            return super().answer(messages, tags=tags)

    designer = CountingDesigner()
    run, _, _, _ = generation(tmp_path, designer=designer)
    previous = {"generation": 3, "mean_regret": 0.1, "designer_version": None, "agent_version": None}
    result = run.run(request(count=2, previous=previous))
    assert len(designer.reports) == 2
    assert designer.reports[1]["feedback"]["round"]["previous"] == previous
    assert json.loads(result.report_path.read_text())["request"]["previous"] == previous


def test_a_request_refuses_a_prompt_or_a_version_of_the_wrong_kind() -> None:
    with pytest.raises(GenerationError, match="prompt must be a DesignerPrompt"):
        request(prompt="be creative")
    with pytest.raises(GenerationError, match="designer_version must be a RoleVersion"):
        request(designer_version="abc")
    with pytest.raises(GenerationError, match="previous must be the last round's record"):
        request(previous="round 3")


def test_the_reef_roles_build_the_designer_and_the_reasoning_agent(tmp_path: Path) -> None:
    prompt = DesignerPrompt(request_options={"reasoning_effort": "none"}, timeout_s=7.0)
    designer = reef_designer(Role.designer("http://127.0.0.1:1", "designer", "strong", prompt=prompt, token="t"))
    assert (designer.scenario, designer.model, designer.request_options) == (
        "designer",
        "strong",
        {"reasoning_effort": "none"},
    )
    assert designer.client.timeout_s == 7.0
    harness = HarborAgent(
        spec={"name": "codex", "kwargs": {"api_base": "{base_url}"}}, host="host.docker.internal", concurrency=3
    )
    agent = reef_reasoning_agent(
        Role.reasoning_agent("http://127.0.0.1:1", "spade", "m", agent=harness, token="t"), work_dir=tmp_path / "work"
    )
    assert (agent.scenario, agent.model, agent.agent_host, agent.concurrency) == (
        "spade",
        "m",
        "host.docker.internal",
        3,
    )
    assert (
        agent.agent == {"name": "codex", "kwargs": {"api_base": "{base_url}"}} and agent.work_dir == tmp_path / "work"
    )
    with pytest.raises(GenerationError, match="must be a DesignerPrompt"):
        reef_designer(Role.reasoning_agent("http://127.0.0.1:1", "spade", "m"))
    with pytest.raises(GenerationError, match="must be a HarborAgent"):
        reef_reasoning_agent(Role.designer("http://127.0.0.1:1", "designer", "m"), work_dir=tmp_path)


def test_main_hands_the_reasoning_agent_its_harbor_agent(tmp_path: Path, monkeypatch) -> None:
    seen: list[dict[str, object]] = []

    class RecordingAgent(StandInReasoningAgent):
        def __init__(self, **options: object) -> None:
            seen.append(options)
            super().__init__()

    monkeypatch.setattr("recipes.beta.spade.generation.ReefReasoningAgent", RecordingAgent)
    spec = json.dumps({"name": "codex", "kwargs": {"api_base": "{base_url}"}})
    status = main(
        [
            "--reef-url",
            "http://127.0.0.1:8900",
            "--scenario",
            "spade",
            "--model",
            "m",
            "--tasks-root",
            str(tmp_path / "t"),
            "--work-dir",
            str(tmp_path / "w"),
            "--description",
            "shell tasks",
            "--count",
            "1",
            "--plays",
            "1",
            "--agent-json",
            spec,
            "--agent-host",
            "h",
            "--concurrency",
            "3",
        ],
        designer=StandInDesigner(),
        checks=StandInChecks(),
    )
    assert status == 0 and len(seen) == 1
    options = seen[0]
    assert (options["reef_url"], options["scenario"], options["model"]) == ("http://127.0.0.1:8900", "spade", "m")
    assert options["agent"] == {"name": "codex", "kwargs": {"api_base": "{base_url}"}}
    assert (options["agent_host"], options["concurrency"], options["work_dir"]) == ("h", 3, tmp_path / "w")
