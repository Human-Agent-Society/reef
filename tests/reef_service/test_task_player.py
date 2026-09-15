"""The task player: an agent's calls through the capture proxy, the verifier reward reported against the receipts."""

from __future__ import annotations

import contextlib
import json
import threading
import urllib.error
import urllib.request
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from reef.core.tasks import HarborTask, read_split_manifest, split_by_source, write_harbor_task, write_split_manifest
from reef.harness.client.tasks import (
    DEFAULT_AGENT,
    EpisodeRow,
    TaskLab,
    TaskPlay,
    TaskPlayer,
    TaskPlayError,
    bound_agent,
    episode_reward,
    main,
    task_identity,
)


class StandInReef:
    """A Reef service that answers inference with a receipt and keeps every report it gets."""

    def __init__(self, *, refuse_reports: bool = False, refuse_inference: bool = False) -> None:
        self.inferences: list[dict[str, object]] = []
        self.reports: list[dict[str, object]] = []
        service = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                headers = {name.lower(): value for name, value in self.headers.items()}
                if self.path == "/v1/chat/completions" and refuse_inference:
                    service.inferences.append({"headers": headers, "body": body})
                    self.answer(401, {"error": {"message": "invalid service token"}})
                elif self.path == "/v1/chat/completions":
                    service.inferences.append({"headers": headers, "body": body})
                    receipt = f"rec-{len(service.inferences)}"
                    answer = {"id": receipt, "choices": [{"message": {"role": "assistant", "content": "ls"}}]}
                    self.answer(200, answer, receipt=receipt)
                elif self.path == "/reef/report":
                    service.reports.append({"headers": headers, "body": body})
                    if refuse_reports:
                        self.answer(400, {"error": "references must identify an existing inference"})
                    else:
                        self.answer(200, {"agent_record_id": f"rep-{len(service.reports)}"})
                else:
                    self.answer(404, {"error": self.path})

            def answer(self, status: int, document: dict[str, object], *, receipt: str | None = None) -> None:
                payload = json.dumps(document).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                if receipt is not None:
                    self.send_header("x-reef-agent-record-id", receipt)
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: object) -> None:
                return None

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


class StandInLab(TaskLab):
    """Plays by calling the model twice through the api_base the agent was given, then answers with fixed rewards."""

    def __init__(self, rewards: Mapping[str, float], error: str = "", turns: int = 2) -> None:
        self.rewards = dict(rewards)
        self.error = error
        self.turns = turns
        self.calls: list[dict[str, object]] = []

    async def run(self, task_path, agent, *, key, tags, overrides) -> EpisodeRow:
        self.calls.append(
            {"task_path": task_path, "agent": agent, "key": key, "tags": dict(tags), "overrides": dict(overrides)}
        )
        kwargs = agent["kwargs"]
        for turn in range(self.turns):
            body = json.dumps(
                {"model": agent["model_name"], "messages": [{"role": "user", "content": f"turn {turn}"}]}
            )
            request = urllib.request.Request(
                f"{kwargs['api_base']}/chat/completions",
                data=body.encode(),
                headers={"Content-Type": "application/json", "Authorization": "Bearer agent-side"},
                method="POST",
            )
            # An agent that is refused keeps going, like terminus after its retries; the episode still ends.
            with contextlib.suppress(urllib.error.HTTPError), urllib.request.urlopen(request, timeout=10) as response:
                response.read()
        return EpisodeRow(rewards=self.rewards, error=self.error, trial_uri=f"trials/{task_path.name}")


@pytest.fixture
def reef():
    service = StandInReef()
    yield service
    service.close()


def written_task(root: Path, name: str, record_id: str = "rec-source") -> Path:
    task = HarborTask(
        name=name,
        instruction="List the files in the working directory and write their count to /app/count.txt.",
        tests={"test.sh": "#!/bin/sh\nmkdir -p /logs/verifier\necho 1 > /logs/verifier/reward.txt\n"},
        environment={"Dockerfile": "FROM python:3.12-slim\nWORKDIR /app\n"},
        source_agent_record_ids=(record_id,),
    )
    return write_harbor_task(task, root)


def player(reef: StandInReef, tmp_path: Path, lab: TaskLab, **overrides: object) -> TaskPlayer:
    fields: dict[str, object] = {
        "reef_url": reef.url,
        "scenario": "guess",
        "model": "qwen3.8:27b",
        "work_dir": tmp_path / "work",
        "token": "tok",
        "lab": lab,
    }
    fields.update(overrides)
    return TaskPlayer(**fields)  # type: ignore[arg-type]


# ------------------------------------------------------------------------------------------------ pure parts


def test_bound_agent_fills_the_placeholders_wherever_they_sit() -> None:
    agent = {
        "name": "claude-code",
        "model_name": "{model}",
        "env": {"ANTHROPIC_BASE_URL": "{base_url}", "ANTHROPIC_API_KEY": "{api_key}"},
        "kwargs": {"flags": ["--model", "{model}"], "retries": 2},
    }
    bound = bound_agent(agent, model="m", base_url="http://127.0.0.1:1", api_key="k")
    assert bound == {
        "name": "claude-code",
        "model_name": "m",
        "env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:1", "ANTHROPIC_API_KEY": "k"},
        "kwargs": {"flags": ["--model", "m"], "retries": 2},
    }
    default = bound_agent(DEFAULT_AGENT, model="m", base_url="http://h:2", api_key="k")
    assert default["model_name"] == "openai/m", "LiteLLM honours api_base only under a provider prefix"
    assert default["kwargs"] == {"api_base": "http://h:2/v1", "llm_kwargs": {"api_key": "k"}}


@pytest.mark.parametrize(
    ("rewards", "expected"),
    [
        ({"reward": 0.5}, 0.5),
        ({"accuracy": 1, "reward": 0.0}, 0.0),
        ({"accuracy": 1}, 1.0),
        ({}, None),
        ({"reward": float("nan")}, None),
        ({"reward": True}, None),
    ],
)
def test_episode_reward_reads_the_reward_entry_first(rewards, expected) -> None:
    assert episode_reward(rewards) == expected


def test_task_identity_carries_the_reef_digest_only_when_the_task_has_one(tmp_path: Path) -> None:
    written = written_task(tmp_path, "t-written")
    identity = task_identity(written)
    assert identity["name"] == "t-written" and identity["path"] == str(written) and len(identity["digest"]) == 64
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "task.toml").write_text('version = "1.0"\n')
    assert task_identity(foreign) == {"name": "foreign", "path": str(foreign)}


# ------------------------------------------------------------------------------------------------ playing


def test_a_scored_episode_is_reported_against_its_receipts(reef: StandInReef, tmp_path: Path) -> None:
    task_path = written_task(tmp_path / "tasks", "t1")
    hint = tmp_path / "hint.md"
    hint.write_text("Count with ls | wc -l.\n")
    lab = StandInLab({"reward": 1.0})
    played = player(reef, tmp_path, lab, labels={"arm": "hint"}, extra_instruction_paths=[hint]).play(task_path)

    assert played.reward == 1.0 and played.receipts == ("rec-1", "rec-2") and played.error == ""
    assert played.report_agent_record_ids == ("rep-1",) and played.is_reported
    assert played.trial_uri == "trials/t1" and len(played.episode_id) == 32

    call = lab.calls[0]
    assert call["task_path"] == task_path and call["key"] == played.episode_id
    assert call["tags"] == {"task": "t1", "episode": played.episode_id, "arm": "hint"}
    assert call["overrides"] == {"environment": {"type": "docker"}, "extra_instruction_paths": [str(hint)]}
    agent = call["agent"]
    assert agent["name"] == "terminus-2" and agent["model_name"] == "openai/qwen3.8:27b"
    assert agent["kwargs"]["api_base"].startswith("http://127.0.0.1:") and agent["kwargs"]["api_base"].endswith("/v1")
    assert agent["kwargs"]["llm_kwargs"] == {"api_key": "reef"}, "the proxy injects the token; no trial dir holds it"
    assert played.failed_calls == 0

    headers = reef.inferences[0]["headers"]
    assert headers["x-reef-scenario"] == "guess" and headers["authorization"] == "Bearer tok"
    assert headers["x-reef-tag-task"] == "t1" and headers["x-reef-tag-arm"] == "hint"
    assert headers["x-reef-tag-episode"] == played.episode_id

    report = reef.reports[0]
    assert report["headers"]["x-reef-scenario"] == "guess" and report["headers"]["authorization"] == "Bearer tok"
    body = report["body"]
    assert body["score"] == 1.0 and body["references"] == ["rec-1", "rec-2"]
    assert body["feedback"] == "verifier reward 1.0 on t1"
    assert body["metadata"]["task"] == task_identity(task_path)
    assert body["metadata"]["episode"] == {
        "id": played.episode_id,
        "agent": "terminus-2",
        "labels": {"arm": "hint"},
        "rewards": {"reward": 1.0},
        "trial_uri": "trials/t1",
    }


def test_a_held_episode_is_reported_after_the_fact_with_extra_metadata(reef: StandInReef, tmp_path: Path) -> None:
    task_path = written_task(tmp_path / "tasks", "t1")
    held = player(reef, tmp_path, StandInLab({"reward": 1.0}), labels={"arm": "plain"}, is_reporting=False).play(
        task_path
    )
    assert not held.is_reported and held.receipts == ("rec-1", "rec-2") and held.labels == {"arm": "plain"}
    assert reef.reports == []

    later = player(reef, tmp_path, StandInLab({"reward": 1.0}))
    record_ids = later.report_play(held, score=0.5, metadata={"generation": 4, "task": "not this one"})
    assert record_ids == ("rep-1",)
    body = reef.reports[0]["body"]
    assert body["score"] == 0.5 and body["references"] == ["rec-1", "rec-2"]
    assert body["feedback"] == "verifier reward 0.5 on t1" and body["metadata"]["generation"] == 4
    assert body["metadata"]["task"] == task_identity(task_path), "extra keys never replace the task"
    assert (
        body["metadata"]["episode"]["labels"] == {"arm": "plain"}
        and body["metadata"]["episode"]["id"] == held.episode_id
    )

    unplayed = TaskPlay(task_path, "t1", "e", 1.0, {"reward": 1.0}, "", (), 0, (), None)
    with pytest.raises(TaskPlayError, match="holds no receipt"):
        later.report_play(unplayed, score=1.0)


def test_an_unscored_episode_is_kept_but_not_reported(reef: StandInReef, tmp_path: Path) -> None:
    task_path = written_task(tmp_path / "tasks", "t1")
    played = player(reef, tmp_path, StandInLab({}, error="the container died")).play(task_path)
    assert played.reward is None and played.error == "the container died"
    assert played.receipts == ("rec-1", "rec-2") and not played.is_reported and reef.reports == []


def test_an_episode_without_model_calls_is_not_reported(reef: StandInReef, tmp_path: Path) -> None:
    task_path = written_task(tmp_path / "tasks", "t1")
    played = player(reef, tmp_path, StandInLab({"reward": 1.0}, turns=0)).play(task_path)
    assert played.reward == 1.0 and played.receipts == () and not played.is_reported and reef.reports == []


def test_per_receipt_sends_one_report_per_model_call(reef: StandInReef, tmp_path: Path) -> None:
    task_path = written_task(tmp_path / "tasks", "t1")
    played = player(reef, tmp_path, StandInLab({"reward": 0.25}), per_receipt=True).play(task_path)
    assert played.report_agent_record_ids == ("rep-1", "rep-2")
    assert [report["body"]["references"] for report in reef.reports] == [["rec-1"], ["rec-2"]]
    assert {report["body"]["score"] for report in reef.reports} == {0.25}


def test_a_refused_report_is_an_error_that_names_the_task(tmp_path: Path) -> None:
    reef = StandInReef(refuse_reports=True)
    try:
        task_path = written_task(tmp_path / "tasks", "t1")
        with pytest.raises(TaskPlayError, match=r"the report for t1 was refused \(400\)"):
            player(reef, tmp_path, StandInLab({"reward": 1.0})).play(task_path)
    finally:
        reef.close()


def test_the_agents_own_authorization_passes_through_without_a_token(reef: StandInReef, tmp_path: Path) -> None:
    task_path = written_task(tmp_path / "tasks", "t1")
    lab = StandInLab({"reward": 1.0})
    played = player(reef, tmp_path, lab, token=None).play(task_path)
    assert played.is_reported
    assert lab.calls[0]["agent"]["kwargs"]["llm_kwargs"] == {"api_key": "reef"}
    assert reef.inferences[0]["headers"]["authorization"] == "Bearer agent-side"
    assert "authorization" not in reef.reports[0]["headers"]


def test_a_measurement_run_keeps_its_episodes_out_of_the_training_data(reef: StandInReef, tmp_path: Path) -> None:
    task_path = written_task(tmp_path / "tasks", "t1")
    played = player(reef, tmp_path, StandInLab({"reward": 1.0}), is_reporting=False).play(task_path)
    assert played.reward == 1.0 and played.receipts == ("rec-1", "rec-2")
    assert not played.is_reported and reef.reports == []


def test_concurrent_plays_each_get_their_own_proxy_and_report(reef: StandInReef, tmp_path: Path) -> None:
    paths = [written_task(tmp_path / "tasks", name) for name in ("t1", "t2", "t3")]
    lab = StandInLab({"reward": 1.0})
    plays = player(reef, tmp_path, lab).play_concurrently(paths, concurrency=2)
    assert [play.name for play in plays] == ["t1", "t2", "t3"] and all(play.is_reported for play in plays)
    assert len({play.episode_id for play in plays}) == 3 and len(reef.reports) == 3
    ports = {call["agent"]["kwargs"]["api_base"] for call in lab.calls}
    assert len(ports) == 3, "every episode talks to its own proxy, so its receipts are its own"
    assert sorted(receipt for play in plays for receipt in play.receipts) == [f"rec-{n}" for n in range(1, 7)]
    with pytest.raises(TaskPlayError, match="concurrency must be a positive integer"):
        player(reef, tmp_path, lab).play_concurrently(paths, concurrency=0)


def test_two_plays_are_two_episodes(reef: StandInReef, tmp_path: Path) -> None:
    task_path = written_task(tmp_path / "tasks", "t1")
    lab = StandInLab({"reward": 1.0})
    first, second = player(reef, tmp_path, lab).play_all([task_path, task_path])
    assert first.episode_id != second.episode_id and lab.calls[0]["key"] != lab.calls[1]["key"]
    assert [report["body"]["references"] for report in reef.reports] == [["rec-1", "rec-2"], ["rec-3", "rec-4"]]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"reef_url": ""}, "reef_url must be a non-empty string"),
        ({"scenario": " "}, "scenario must be a non-empty string"),
        ({"agent": {"kwargs": {}}}, "agent must carry a Harbor agent name or an import_path"),
        (
            {"extra_instruction_paths": [Path("/nowhere/hint.md")]},
            "extra instruction file /nowhere/hint.md does not exist",
        ),
    ],
)
def test_a_player_refuses_an_incomplete_configuration(reef: StandInReef, tmp_path: Path, overrides, message) -> None:
    with pytest.raises(TaskPlayError, match=message):
        player(reef, tmp_path, StandInLab({"reward": 1.0}), **overrides)


def test_a_directory_without_task_toml_is_refused(reef: StandInReef, tmp_path: Path) -> None:
    with pytest.raises(TaskPlayError, match="not a Harbor task directory"):
        player(reef, tmp_path, StandInLab({"reward": 1.0})).play(tmp_path)


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        ({"my arm": "x"}, "label name 'my arm' must be letters"),
        ({"arm/v2": "x"}, "label name 'arm/v2' must be letters"),
        ({"task": "other"}, "label 'task' is the player's own tag"),
        ({"arm": "提示"}, "needs a non-empty printable ASCII value"),
        ({"arm": " "}, "needs a non-empty printable ASCII value"),
    ],
)
def test_a_label_that_cannot_ride_as_a_header_is_refused(reef: StandInReef, tmp_path: Path, labels, message) -> None:
    with pytest.raises(TaskPlayError, match=message):
        player(reef, tmp_path, StandInLab({"reward": 1.0}), labels=labels)


def test_labels_are_stored_the_way_the_service_stores_tags(reef: StandInReef, tmp_path: Path) -> None:
    task_path = written_task(tmp_path / "tasks", "t1")
    played = player(reef, tmp_path, StandInLab({"reward": 1.0}), labels={"Arm": " Hint "}).play(task_path)
    assert reef.inferences[0]["headers"]["x-reef-tag-arm"] == "Hint"
    assert reef.reports[0]["body"]["metadata"]["episode"]["labels"] == {"arm": "Hint"} and played.is_reported


def test_refused_model_calls_are_named_in_the_error(tmp_path: Path) -> None:
    reef = StandInReef(refuse_inference=True)
    try:
        task_path = written_task(tmp_path / "tasks", "t1")
        played = player(reef, tmp_path, StandInLab({"reward": 1.0})).play(task_path)
    finally:
        reef.close()
    assert played.reward == 1.0 and played.receipts == () and not played.is_reported
    assert played.failed_calls == 2
    assert played.error == "2 model calls failed; the first answered 401: invalid service token"
    assert played.as_line()["failed_calls"] == 2


def test_an_agent_inside_the_container_gets_a_reachable_proxy_address(reef: StandInReef, tmp_path: Path) -> None:
    task_path = written_task(tmp_path / "tasks", "t1")
    lab = StandInLab({"reward": 1.0})
    played = player(reef, tmp_path, lab, agent_host="localhost").play(task_path)
    assert played.is_reported and played.receipts == ("rec-1", "rec-2")
    assert lab.calls[0]["agent"]["kwargs"]["api_base"].startswith("http://localhost:")


def test_a_report_that_does_not_reach_reef_is_an_error_that_names_the_task(
    reef: StandInReef, tmp_path: Path, monkeypatch
) -> None:
    task_path = written_task(tmp_path / "tasks", "t1")
    playing = player(reef, tmp_path, StandInLab({"reward": 1.0}))

    def refused(*args: object, **kwargs: object) -> dict[str, object]:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(playing.client, "report", refused)
    with pytest.raises(TaskPlayError, match="the report for t1 did not reach"):
        playing.play(task_path)


# ------------------------------------------------------------------------------------------------ the command


def test_main_plays_one_side_of_a_manifest_and_prints_a_line_per_task(
    reef: StandInReef, tmp_path: Path, capsys, monkeypatch
) -> None:
    root = tmp_path / "tasks"
    for name, record in (("t1", "r1"), ("t2", "r2"), ("t3", "r3"), ("t4", "r4")):
        written_task(root, name, record)
    split = split_by_source({"t1": ["r1"], "t2": ["r2"], "t3": ["r3"], "t4": ["r4"]}, eval_fraction=0.5, seed=3)
    manifest = tmp_path / "manifest.json"
    write_split_manifest(manifest, split)
    train_names = list(read_split_manifest(manifest).train)
    monkeypatch.setenv("REEF_TOKEN", "env-token")
    lab = StandInLab({"reward": 1.0})
    status = main(
        [
            "--manifest",
            str(manifest),
            "--tasks-root",
            str(root),
            "--side",
            "train",
            "--reef-url",
            reef.url,
            "--scenario",
            "guess",
            "--model",
            "m",
            "--work-dir",
            str(tmp_path / "work"),
            "--label",
            "arm=plain",
        ],
        lab=lab,
    )
    assert status == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [line["task"] for line in lines] == train_names and len(train_names) == 2
    assert all(line["reward"] == 1.0 and line["receipts"] == 2 and len(line["reports"]) == 1 for line in lines)
    assert reef.inferences[0]["headers"]["authorization"] == "Bearer env-token"
    assert reef.reports[0]["body"]["metadata"]["episode"]["labels"] == {"arm": "plain"}
    assert [call["task_path"] for call in lab.calls] == [root / name for name in train_names]


def test_main_returns_one_when_a_task_was_not_reported(reef: StandInReef, tmp_path: Path, capsys) -> None:
    task_path = written_task(tmp_path / "tasks", "t1")
    status = main(
        [
            str(task_path),
            "--reef-url",
            reef.url,
            "--scenario",
            "guess",
            "--model",
            "m",
            "--work-dir",
            str(tmp_path / "w"),
        ],
        lab=StandInLab({}, error="no verifier"),
    )
    assert status == 1
    line = json.loads(capsys.readouterr().out.strip())
    assert line == {
        "task": "t1",
        "reward": None,
        "receipts": 2,
        "failed_calls": 0,
        "reports": [],
        "error": "no verifier",
    }


def test_main_prints_each_line_as_it_finishes_and_keeps_going_after_a_refused_report(tmp_path: Path, capsys) -> None:
    reef = StandInReef(refuse_reports=True)
    try:
        first = written_task(tmp_path / "tasks", "t1")
        second = written_task(tmp_path / "tasks", "t2")
        lab = StandInLab({"reward": 1.0})
        status = main(
            [
                str(first),
                str(second),
                "--reef-url",
                reef.url,
                "--scenario",
                "guess",
                "--model",
                "m",
                "--work-dir",
                str(tmp_path / "w"),
            ],
            lab=lab,
        )
    finally:
        reef.close()
    assert status == 1
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [line["task"] for line in lines] == ["t1", "t2"] and len(lab.calls) == 2
    assert all(line["error"].startswith("the report for t") for line in lines)


def test_main_checks_every_task_directory_before_the_first_play(reef: StandInReef, tmp_path: Path, capsys) -> None:
    task_path = written_task(tmp_path / "tasks", "t1")
    lab = StandInLab({"reward": 1.0})
    with pytest.raises(SystemExit):
        main(
            [str(task_path), str(tmp_path / "nope"), "--reef-url", reef.url, "--scenario", "s", "--model", "m"],
            lab=lab,
        )
    assert lab.calls == [] and "not a Harbor task directory" in capsys.readouterr().err


def test_main_refuses_a_manifest_beside_task_directories(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                str(tmp_path),
                "--manifest",
                str(tmp_path / "m.json"),
                "--reef-url",
                "http://x",
                "--scenario",
                "s",
                "--model",
                "m",
            ]
        )
