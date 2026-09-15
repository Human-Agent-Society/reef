"""Rounds: a deployment's kind, version and prompt, the wait for its commit, and a round over stand ins."""

from __future__ import annotations

import json
from collections.abc import Mapping

import pytest
from reef_client.client import ReefClientError

from recipes.beta.spade.designer import DesignerPrompt
from recipes.beta.spade.roles import HarborAgent, Role, RoleVersion, regret, verifier_reward
from recipes.beta.spade.rounds import (
    HARNESS_PATH,
    KIND_FIXED,
    KIND_HARNESS,
    KIND_WEIGHTS,
    SCENARIOS_PATH,
    STATUS_PATH,
    TREE_FILE,
    Deployment,
    RoundError,
    Timer,
    parsed_status,
)


class StandInClient:
    """A Reef service as a round sees it: a status document, an optional harness manifest, created scenarios."""

    def __init__(
        self, *, harness: dict[str, object] | None = None, step: int | None = None, load_id: str | None = None
    ) -> None:
        self.harness = harness
        self.step = step
        self.load_id = load_id
        self.head_state = "synchronized"
        self.metrics: dict[str, object] = {}
        self.error: str | None = None
        self.created: list[str] = []
        self.reads = 0

    def status_document(self) -> dict[str, object]:
        committed = (
            None if self.step is None else {"step": self.step, "recorded_at": 1.0, "metrics": dict(self.metrics)}
        )
        return {
            "error": self.error,
            "scenarios": {
                "spade": {
                    "scenario_step": self.step or 0,
                    "last_committed_step": committed,
                    "artifact_head_sync": {"state": self.head_state, "release_id": "r1", "error": None},
                    "current_runtime_load_id": self.load_id,
                    "batch_ready": False,
                    "processor": {"groups": 1},
                }
            },
        }

    def get(self, path: str, *, extra_headers: Mapping[str, str] | None = None) -> dict[str, object]:
        self.reads += 1
        if path == STATUS_PATH:
            return self.status_document()
        if path == HARNESS_PATH:
            assert extra_headers == {"x-reef-scenario": "spade"}
            if self.harness is None:
                raise ReefClientError(404, "scenario 'spade' serves no files")
            return dict(self.harness)
        raise AssertionError(path)

    def post(
        self, path: str, scenario: str, payload: Mapping[str, object]
    ) -> tuple[dict[str, object], dict[str, str]]:
        assert path == SCENARIOS_PATH and payload == {"name": scenario}
        self.created.append(scenario)
        return {"scenario": scenario}, {}


def designer_role() -> Role:
    return Role(reef_url="http://127.0.0.1:8901", scenario="spade", model="m", harness=DesignerPrompt(), reward=regret)


def agent_role() -> Role:
    return Role(
        reef_url="http://127.0.0.1:8900", scenario="spade", model="m", harness=HarborAgent(), reward=verifier_reward
    )


def harness_manifest(system: str = "evolved system", release_id: str = "rel-2") -> dict[str, object]:
    entries = [{"id": "designer-system", "name": "skill", "config": {"name": "designer-system", "text": system}}]
    return {
        "release_id": release_id,
        "parent_release_id": "rel-1",
        "content_id": "c",
        "files": {TREE_FILE: json.dumps(entries)},
    }


class StandInTimer(Timer):
    """A clock that advances by each sleep and lets the service commit its first step after one poll."""

    def __init__(self, client: StandInClient, *, commits: bool = True) -> None:
        self.client = client
        self.commits = commits
        self.elapsed = 0.0

    def now(self) -> float:
        return self.elapsed

    def sleep(self, seconds: float) -> None:
        self.elapsed += seconds
        if self.commits and self.client.step is not None:
            self.client.step += 1


def deployment(
    client: StandInClient, role: Role | None = None, *, commits: bool = True, **options: float
) -> Deployment:
    timer = StandInTimer(client, commits=commits)
    return Deployment(role or designer_role(), client=client, timer=timer, **options)  # type: ignore[arg-type]


def test_parsed_status_reads_the_scenario_block_and_refuses_an_unknown_scenario() -> None:
    client = StandInClient(step=3, load_id="load-3")
    status = parsed_status(client.status_document(), "spade")
    assert (status.step, status.runtime_load_id, status.head_state, status.release_id) == (
        3,
        "load-3",
        "synchronized",
        "r1",
    )
    with pytest.raises(RoundError, match="no scenario 'other'"):
        parsed_status(client.status_document(), "other")


def test_a_deployment_is_a_harness_when_it_serves_a_tree() -> None:
    client = StandInClient(harness=harness_manifest(), step=0)
    dep = deployment(client)
    assert dep.kind() == KIND_HARNESS
    assert dep.version() == RoleVersion("release", "rel-2")
    prompt = dep.prompt(DesignerPrompt())
    assert prompt.system == "evolved system" and prompt.rules == DesignerPrompt().rules


def test_a_deployment_is_weights_when_it_serves_no_tree_but_a_runtime_load() -> None:
    client = StandInClient(step=0, load_id="load-7")
    dep = deployment(client)
    assert dep.kind() == KIND_WEIGHTS
    assert dep.version() == RoleVersion("runtime", "load-7")
    assert dep.prompt(DesignerPrompt()) == DesignerPrompt(), "no tree, the role's own prompt"


def test_a_deployment_that_evolves_nothing_is_fixed_and_named_by_its_model() -> None:
    dep = deployment(StandInClient())
    assert dep.kind() == KIND_FIXED
    assert dep.version() == RoleVersion("model", "m")


def test_ensure_scenario_posts_the_name_once_per_call() -> None:
    client = StandInClient()
    deployment(client).ensure_scenario()
    assert client.created == ["spade"]


def test_wait_for_step_returns_at_the_first_later_commit() -> None:
    client = StandInClient(step=0, load_id="load-1")
    dep = deployment(client, poll_s=1.0, wait_timeout_s=10.0)
    status = dep.wait_for_step(after=0)
    assert status.step == 1


def test_wait_for_step_on_a_harness_also_waits_for_the_head_to_synchronize() -> None:
    client = StandInClient(harness=harness_manifest(), step=1)
    client.head_state = "publishing"
    dep = deployment(client, poll_s=1.0, wait_timeout_s=3.0)
    with pytest.raises(RoundError, match="committed no step after 0 within 3 s"):
        dep.wait_for_step(after=0)
    synchronized = StandInClient(harness=harness_manifest(), step=0)
    assert deployment(synchronized, poll_s=1.0, wait_timeout_s=3.0).wait_for_step(after=0).step == 1


def test_wait_for_step_raises_on_a_service_error_and_on_the_deadline_with_the_processor_state() -> None:
    client = StandInClient(step=2, load_id="load-2")
    client.error = "spade: ValueError: training data cannot assemble"
    with pytest.raises(RoundError, match="stopped training: spade: ValueError"):
        deployment(client, poll_s=1.0, wait_timeout_s=5.0).wait_for_step(after=2)
    client.error = None
    with pytest.raises(RoundError, match=r'batch_ready=False, processor=\{"groups": 1\}'):
        deployment(client, commits=False, poll_s=1.0, wait_timeout_s=2.0).wait_for_step(after=2)


def test_a_served_tree_that_is_not_a_prompt_is_refused() -> None:
    broken = harness_manifest()
    broken["files"] = {TREE_FILE: "not json"}
    with pytest.raises(RoundError, match="is not JSON"):
        deployment(StandInClient(harness=broken, step=0)).prompt(DesignerPrompt())
    with pytest.raises(RoundError, match=r"carries no native/tree\.json"):
        deployment(StandInClient(harness={"release_id": "r", "files": {}}, step=0)).prompt(DesignerPrompt())


from spade_stand_ins import StandInChecks, StandInDesigner, StandInReasoningAgent

from recipes.beta.spade.generation import GenerationRequest
from recipes.beta.spade.rounds import ROUNDS_FILE, SpadeRun, main


def run(
    tmp_path, designer_client: StandInClient, agent_client: StandInClient, **options: object
) -> tuple[SpadeRun, StandInDesigner, StandInReasoningAgent]:
    designer_impl, agent_impl = StandInDesigner(), StandInReasoningAgent()
    spade_run = SpadeRun(
        designer_role(),
        agent_role(),
        checks=StandInChecks(),
        tasks_root=tmp_path / "tasks",
        work_dir=tmp_path / "work",
        designer_impl=designer_impl,
        agent_impl=agent_impl,
        designer_deployment=deployment(designer_client, designer_role(), poll_s=1.0, wait_timeout_s=30.0),
        agent_deployment=deployment(agent_client, agent_role(), poll_s=1.0, wait_timeout_s=30.0),
        **options,  # type: ignore[arg-type]
    )
    return spade_run, designer_impl, agent_impl


def request(**overrides: object) -> GenerationRequest:
    fields: dict[str, object] = {
        "description": "shell tasks",
        "count": 2,
        "generation": 1,
        "plays": 2,
        "hint_plays": 1,
    }
    fields.update(overrides)
    return GenerationRequest(**fields)  # type: ignore[arg-type]


def test_a_round_pulls_the_prompt_measures_waits_for_the_designer_then_reports_the_held_plays(tmp_path) -> None:
    designer_client = StandInClient(harness=harness_manifest(), step=0)
    designer_client.metrics = {"traces": 2}
    agent_client = StandInClient(step=0, load_id="load-1")
    spade_run, designer, agent = run(tmp_path, designer_client, agent_client)
    result = spade_run.round(request())

    assert designer.calls[0]["messages"][0]["content"] == "evolved system", "the served prompt reached the Designer"
    assert (result.designer_kind, result.agent_kind) == (KIND_HARNESS, KIND_WEIGHTS)
    assert result.designer_version_before == RoleVersion("release", "rel-2")
    assert result.agent_version_before == RoleVersion("runtime", "load-1")
    assert (result.designer_step, result.agent_step, result.measured, result.reported_plays) == (1, 1, 2, 4)
    assert result.mean_regret == 0.5
    assert [report["metadata"]["opponent"]["version"] for report in designer.reports] == [
        {"kind": "runtime", "id": "load-1"}
    ] * 2
    plain = [report for report in agent.reports if report["metadata"]["arm"] == "plain"]
    hint = [report for report in agent.reports if report["metadata"]["arm"] == "hint"]
    assert len(plain) == 2 and len(hint) == 2
    assert all(
        report["metadata"]["round"] == "generation-00001" and report["metadata"]["round_plays"] == 4
        for report in plain
    )
    assert all(report["metadata"]["designer_version"] == {"kind": "release", "id": "rel-2"} for report in plain + hint)
    lines = [json.loads(line) for line in (tmp_path / "tasks" / ".spade" / ROUNDS_FILE).read_text().splitlines()]
    assert lines[0]["generation"] == 1 and lines[0]["designer_version_after"] == {"kind": "release", "id": "rel-2"}
    following = spade_run.next_request(request(), result)
    assert following.generation == 2 and following.previous == result.summary() and len(following.experience) == 2


def test_the_agent_round_runs_every_n_designer_rounds_over_all_held_plays(tmp_path) -> None:
    designer_client = StandInClient(step=0, load_id="designer-load")
    agent_client = StandInClient(step=0, load_id="load-1")
    spade_run, _, agent = run(tmp_path, designer_client, agent_client, agent_round_every=2)
    first = spade_run.round(request())
    assert (first.reported_plays, first.agent_step) == (0, None) and agent.reports == []
    second = spade_run.round(spade_run.next_request(request(), first))
    assert (second.reported_plays, second.agent_step) == (8, 1)
    plain = [report for report in agent.reports if report["metadata"]["arm"] == "plain"]
    assert sorted({report["metadata"]["generation"] for report in plain}) == [1, 2]
    assert all(
        report["metadata"]["round"] == "generation-00001" and report["metadata"]["round_plays"] == 8
        for report in plain
    )


def test_a_fixed_role_is_measured_against_and_never_waited_on(tmp_path) -> None:
    spade_run, designer, agent = run(tmp_path, StandInClient(), StandInClient())
    result = spade_run.round(request())
    assert (result.designer_kind, result.agent_kind) == (KIND_FIXED, KIND_FIXED)
    assert result.designer_step is None and result.agent_step is None and result.reported_plays == 0
    assert designer.reports == [] and agent.reports == []
    assert result.designer_version_after == RoleVersion("model", "m")


def test_a_harness_step_over_fewer_reports_than_the_generation_sent_is_an_error(tmp_path) -> None:
    designer_client = StandInClient(harness=harness_manifest(), step=0)
    designer_client.metrics = {"traces": 1}
    spade_run, _, _ = run(tmp_path, designer_client, StandInClient())
    with pytest.raises(RoundError, match="consumed 1 reports, not the 2 of one generation"):
        spade_run.round(request())


def test_main_prints_one_line_per_round(tmp_path, capsys) -> None:
    designer_client, agent_client = StandInClient(), StandInClient()
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
            "--work-dir",
            str(tmp_path / "work"),
            "--description",
            "shell tasks",
            "--count",
            "1",
            "--plays",
            "1",
            "--rounds",
            "2",
        ],
        checks=StandInChecks(),
        designer_impl=StandInDesigner(),
        agent_impl=StandInReasoningAgent(),
        designer_deployment=deployment(designer_client, designer_role()),
        agent_deployment=deployment(agent_client, agent_role()),
    )
    assert status == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [line["generation"] for line in lines] == [0, 1] and lines[1]["designer_kind"] == KIND_FIXED
