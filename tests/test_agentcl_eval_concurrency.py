"""Bounded AgentCL evaluation fixtures; no models, containers, or provider calls."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
from pathlib import Path
from types import ModuleType

import pytest
from tests import test_sdft_agentcl as sdft_workflow
from tests import test_sdpo_agentcl as sdpo_workflow

from recipes.sdft.examples.agentcl.report import JsonObject


@pytest.fixture(params=[sdft_workflow, sdpo_workflow], ids=["sdft", "sdpo"])
def workflow(request: pytest.FixtureRequest) -> ModuleType:
    return request.param


class ControlledEvaluation(sdft_workflow.FakeEpisodes, sdpo_workflow.run.EpisodeBackend):
    def __init__(self, arguments: argparse.Namespace, automatic: bool = False) -> None:
        super().__init__()
        self.arguments = arguments
        self.automatic = automatic
        self.started = []
        self.finished = []
        self.owned_tasks = []
        self.active_positions = []
        self.max_active = 0
        self.errors: dict[int, BaseException] = {}
        self.faults = set()
        self.truncated = set()
        self.scores: dict[int, float | None] = {}
        self.runtime_changes: dict[int, str | None] = {}
        self.release_events = [asyncio.Event() for _ in range(120)]
        self.start_events = [asyncio.Event() for _ in range(120)]
        self.finish_events = [asyncio.Event() for _ in range(120)]
        self.cleanup_started = asyncio.Event()
        self.cleanup_released = asyncio.Event()
        self.cleanup_released.set()

    async def run(self, task: JsonObject, episode_id: str, phase: str, release: JsonObject) -> JsonObject:
        position = task["campaign_position"]
        state_key = json.dumps([task["category"], task["id"]], separators=(",", ":"))
        cursor = sdft_workflow.report.read_object(self.arguments.run_root / "cursor.json")
        state = cursor["phases"][phase]["tasks"][state_key]
        assert episode_id in state["started_episodes"]
        assert state["sampled_release"] == release
        if phase != "train":
            assert state["campaign_position"] == position
            assert cursor["phases"][phase]["sampled_release"] == release
        else:
            assert not self.active_positions or set(self.active_positions) == {position}
        self.started.append((position, phase, episode_id, task["sampling_seed"], copy.deepcopy(release)))
        self.owned_tasks.append(asyncio.current_task())
        self.active_positions.append(position)
        self.max_active = max(self.max_active, len(self.active_positions))
        self.start_events[position].set()
        try:
            if self.automatic:
                await asyncio.sleep(0)
            else:
                await self.release_events[position].wait()
            if position in self.errors:
                raise self.errors[position]
            result = await super().run(task, episode_id, phase, release)
            result["score"] = self.scores.get(position, float(position % 2))
            result["runtime_load_id"] = release["runtime_load_id"]
            result["turns"][0]["runtime_load_id"] = release["runtime_load_id"]
            if position in self.truncated:
                result.update(outcome="truncated", fault="student response or episode token window was exhausted")
            if position in self.runtime_changes:
                result["turns"][0]["runtime_load_id"] = self.runtime_changes[position]
            if position in self.faults:
                result.update(outcome="fault", score=None)
            self.results[episode_id] = copy.deepcopy(result)
            return result
        except asyncio.CancelledError:
            self.cleanup_started.set()
            await self.cleanup_released.wait()
            raise
        finally:
            self.active_positions.remove(position)
            self.finished.append(position)
            self.finish_events[position].set()


def prepare(workflow: ModuleType, tmp_path: Path, phase: str, full: bool = True, automatic: bool = False) -> tuple[
    argparse.Namespace,
    JsonObject,
    sdft_workflow.FakeApi | sdpo_workflow.FakeApi,
    ControlledEvaluation,
    sdft_workflow.run.Campaign | sdpo_workflow.run.Campaign,
]:
    arguments = workflow.arguments_fixture(tmp_path)
    if full:
        arguments.profile = "full"
        arguments.steps = 96
        arguments.attempts = 4 if workflow.run.METHOD == "sdpo" else 1
    manifest = workflow.manifest_fixture(arguments.data_root)
    api = workflow.FakeApi(arguments.attempts)
    backend = ControlledEvaluation(arguments, automatic=automatic)
    campaign = workflow.run.Campaign(arguments, api, backend, manifest)
    if phase in ("frozen-repeat", "independent"):
        api.current.update(release_id="frozen", runtime_load_id="runtime-frozen", operation="training")
        campaign.cursor.update(training_step=arguments.steps, training_release=copy.deepcopy(api.current))
        campaign.save()
    return arguments, manifest, api, backend, campaign


@pytest.mark.parametrize("missing_field", [False, True])
def test_creation_release_without_runtime_binds_exact_receipt_runtime_across_phases(
    workflow: ModuleType, tmp_path: Path, missing_field: bool
) -> None:
    async def check() -> None:
        arguments = workflow.arguments_fixture(tmp_path)
        manifest = workflow.manifest_fixture(arguments.data_root)
        api = workflow.FakeApi(arguments.attempts)
        if missing_field:
            api.current.pop("runtime_load_id")
        else:
            api.current["runtime_load_id"] = None

        class NativeRuntimeEpisodes(sdft_workflow.FakeEpisodes, workflow.run.EpisodeBackend):
            async def run(self, task, episode_id, phase, release):
                row = await super().run(task, episode_id, phase, release)
                row["runtime_load_id"] = "native-base-runtime"
                row["turns"][0]["runtime_load_id"] = "native-base-runtime"
                return row

        backend = NativeRuntimeEpisodes()
        campaign = workflow.run.Campaign(arguments, api, backend, manifest)
        await campaign.execute_phase("baseline")
        await campaign.execute_phase("baseline-independent")
        assert campaign.cursor["evaluation_runtime_load_ids"] == {"base": "native-base-runtime"}
        for phase in ("baseline", "baseline-independent"):
            assert campaign.cursor["phases"][phase]["complete"] is True
        assert api.current.get("runtime_load_id") is None
        assert not api.reports

    asyncio.run(check())


@pytest.mark.parametrize("phase", ["baseline", "baseline-independent", "frozen-repeat", "independent"])
def test_four_evaluations_overlap_bounded_and_finish_in_supplied_order(
    workflow: ModuleType, tmp_path: Path, phase: str
) -> None:
    async def check() -> None:
        arguments, manifest, api, backend, campaign = prepare(workflow, tmp_path, phase)
        expected = manifest["independent" if "independent" in phase else "training"]
        execution = asyncio.create_task(campaign.execute_phase(phase))
        try:
            await asyncio.wait_for(backend.start_events[3].wait(), 1)
            assert len(backend.active_positions) == 4
            assert not backend.start_events[4].is_set()
            assert not api.reports
            for position in (3, 2, 1):
                backend.release_events[position].set()
                await asyncio.wait_for(backend.finish_events[position].wait(), 1)
            assert campaign.cursor["phases"][phase]["complete"] is False
            assert not execution.done()
            for event in backend.release_events:
                event.set()
            await asyncio.wait_for(execution, 5)
        finally:
            execution.cancel()
            await asyncio.gather(execution, return_exceptions=True)
        assert backend.finished[:3] == [3, 2, 1]
        assert backend.max_active == 4
        assert not backend.active_positions
        assert all(child.done() for child in backend.owned_tasks)
        assert [row[0] for row in backend.started] == list(range(len(expected)))
        states = list(campaign.cursor["phases"][phase]["tasks"].values())
        assert [state["campaign_position"] for state in states] == list(range(len(expected)))
        assert all(state["complete"] for state in states)
        assert campaign.cursor["phases"][phase]["complete"] is True
        assert not api.reports
        assert not api.history
        assert not (arguments.run_root / "reports").exists()
        assert not (arguments.run_root / "commits").exists()
        records = [workflow.report.read_object(path) for path in (arguments.run_root / "episodes").glob("*.json")]
        records.sort(key=lambda row: row["campaign_position"])
        assert len(records) == len(expected)
        for record, task, started in zip(records, expected, backend.started, strict=True):
            assert (record["category"], record["task_id"], record["position"]) == (
                task["category"],
                task["id"],
                task["position"],
            )
            assert record["attempt"] == 0
            assert record["episode_id"] == workflow.report.stable_id(
                arguments.run_id, phase, task["category"], task["id"], 0, "episode"
            )
            assert record["report_id"] == workflow.report.stable_id(
                arguments.run_id, phase, task["category"], task["id"], 0, "report"
            )
            seed_id = workflow.report.stable_id("sampling", "all", task["category"], task["id"], 0, "seed")
            assert started[3] == (arguments.seed + int(seed_id.replace("-", "")[:8], 16)) % (2**31)
        summary = workflow.metrics.build_summary(records)["phases"][phase]
        assert summary["one_attempt"] == {
            "episodes": len(expected),
            "valid": len(expected),
            "faulted": 0,
            "truncated": 0,
            "accuracy": 0.5,
            "prompt_tokens": 3 * len(expected),
            "completion_tokens": 2 * len(expected),
            "elapsed_seconds": sum(0.1 for _ in expected),
        }
        assert summary["mean_at_k"] is None
        resumed = workflow.run.Campaign(arguments, api, backend, manifest)
        await resumed.execute_phase(phase)
        assert len(backend.started) == len(expected)

    asyncio.run(check())


def test_full_training_tasks_remain_sequential(workflow: ModuleType, tmp_path: Path) -> None:
    async def check() -> None:
        arguments, _, api, backend, campaign = prepare(workflow, tmp_path, "train", automatic=True)
        await campaign.execute_phase("train")
        assert [row[0] for row in backend.started] == [
            position for position in range(96) for _ in range(arguments.attempts)
        ]
        assert backend.max_active == arguments.attempts
        assert campaign.cursor["training_step"] == 96
        assert len(api.history) == 96
        assert len(api.reports) == 96 * arguments.attempts
        assert not backend.active_positions
        assert all(child is asyncio.current_task() or child.done() for child in backend.owned_tasks)

    asyncio.run(check())


@pytest.mark.parametrize("failure", ["exceptions", "fault"])
def test_failures_wait_for_all_evaluations_and_never_leave_orphans(
    workflow: ModuleType, tmp_path: Path, failure: str
) -> None:
    async def check() -> None:
        arguments, manifest, api, backend, campaign = prepare(workflow, tmp_path, "baseline-independent")
        first_error = TimeoutError("unknown remote result")
        if failure == "exceptions":
            backend.errors = {0: first_error, 1: ValueError("later task failure")}
        else:
            backend.faults.add(0)
        execution = asyncio.create_task(campaign.execute_phase("baseline-independent"))
        try:
            await asyncio.wait_for(backend.start_events[3].wait(), 1)
            for position in (1, 0, 2):
                backend.release_events[position].set()
                await asyncio.wait_for(backend.finish_events[position].wait(), 1)
                assert not execution.done()
            for event in backend.release_events:
                event.set()
            if failure == "exceptions":
                with pytest.raises(TimeoutError) as raised:
                    await asyncio.wait_for(execution, 5)
                assert raised.value is first_error
            else:
                with pytest.raises(RuntimeError, match="fault or truncation"):
                    await asyncio.wait_for(execution, 5)
        finally:
            execution.cancel()
            await asyncio.gather(execution, return_exceptions=True)
        assert len(backend.finished) == 120
        assert backend.max_active == 4
        assert all(child.done() for child in backend.owned_tasks)
        assert not backend.active_positions
        assert not api.reports
        assert campaign.cursor["phases"]["baseline-independent"]["complete"] is False
        resumed = workflow.run.Campaign(arguments, api, backend, manifest)
        with pytest.raises(RuntimeError, match=r"never replay|fault or truncation"):
            await resumed.execute_phase("baseline-independent")
        assert len(backend.started) == 120
        assert not (arguments.run_root / "reports").exists()

    asyncio.run(check())


@pytest.mark.parametrize("prior", ["unknown", "persisted-fault", "recovered-fault", "persisted", "recovered"])
def test_resume_reconciles_late_prior_intent_before_any_missing_requests(
    workflow: ModuleType, tmp_path: Path, prior: str
) -> None:
    async def check() -> None:
        arguments, manifest, api, backend, campaign = prepare(
            workflow, tmp_path, "baseline-independent", automatic=True
        )
        task = {**manifest["independent"][80], "campaign_position": 80}
        key = json.dumps([task["category"], task["id"]], separators=(",", ":"))
        release = campaign.cursor["initial_release"]
        identifier = workflow.report.stable_id(
            arguments.run_id, "baseline-independent", "new", task["id"], 0, "episode"
        )
        state = {"started_episodes": [], "report_attempted": [], "complete": False, "sampled_release": release}
        campaign.cursor["phases"]["baseline-independent"] = {
            "tasks": {key: state},
            "complete": False,
            "release_id": release["release_id"],
        }
        if prior == "unknown":
            state["started_episodes"].append(identifier)
            campaign.save()
        else:
            state["campaign_position"] = 80
            campaign.cursor["phases"]["baseline-independent"]["sampled_release"] = release
            if "fault" in prior:
                backend.faults.add(80)
            await campaign.episode(task, "baseline-independent", 0, release, state)
            if prior.startswith("recovered"):
                (arguments.run_root / "episodes" / f"{identifier}.json").unlink()
        before = len(backend.started)
        resumed = workflow.run.Campaign(arguments, api, backend, manifest)
        if prior in ("unknown", "persisted-fault", "recovered-fault"):
            with pytest.raises(RuntimeError, match=r"never replay|fault or truncation"):
                await resumed.execute_phase("baseline-independent")
            assert len(backend.started) == before
            assert resumed.cursor["phases"]["baseline-independent"]["complete"] is False
        else:
            await resumed.execute_phase("baseline-independent")
            assert len(backend.started) == 120
            assert sum(row[0] == 80 for row in backend.started) == 1
        assert not api.reports

    asyncio.run(check())


@pytest.mark.parametrize("when", ["before", "after", "all-finished"])
@pytest.mark.parametrize("change", ["release_id", "runtime_load_id", "pending"])
def test_release_changes_fail_closed_before_after_and_at_phase_finish(
    workflow: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, when: str, change: str
) -> None:
    async def check() -> None:
        _, _, api, backend, campaign = prepare(workflow, tmp_path, "baseline", full=False, automatic=True)
        original = api.current_release
        call_count = 0
        mutation_call = {"before": 1, "after": 4, "all-finished": 6}[when]

        def _current_release() -> JsonObject:
            nonlocal call_count
            call_count += 1
            if call_count == mutation_call:
                api.current[change] = True if change == "pending" else "changed"
            return original()

        monkeypatch.setattr(api, "current_release", _current_release)
        with pytest.raises(RuntimeError, match="release changed"):
            await campaign.execute_phase("baseline")
        assert campaign.cursor["phases"]["baseline"]["complete"] is False
        assert not api.reports
        assert not backend.active_positions
        assert all(child.done() for child in backend.owned_tasks)
        if when == "before":
            assert not backend.started
        elif when == "all-finished":
            assert len(backend.finished) == 2

    asyncio.run(check())


@pytest.mark.parametrize("cleanup", ["complete", "timeout", "recancel"])
def test_outer_cancellation_drains_active_and_queued_evaluations(
    workflow: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cleanup: str
) -> None:
    async def check() -> None:
        arguments, manifest, api, backend, campaign = prepare(workflow, tmp_path, "baseline-independent")
        backend.cleanup_released.clear()
        execution = asyncio.create_task(campaign.execute_phase("baseline-independent"))
        try:
            await asyncio.wait_for(backend.start_events[3].wait(), 1)
            if cleanup == "timeout":
                original_timeout = asyncio.timeout

                def _cleanup_timeout(timeout_seconds: float) -> asyncio.Timeout:
                    assert timeout_seconds == 30
                    return original_timeout(0)

                monkeypatch.setattr(asyncio, "timeout", _cleanup_timeout)
            execution.cancel("stop evaluation")
            await asyncio.wait_for(backend.cleanup_started.wait(), 1)
            if cleanup == "complete":
                assert not execution.done()
                backend.cleanup_released.set()
            elif cleanup == "recancel":
                execution.cancel("stop cleanup")
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(execution, 1)
        finally:
            backend.cleanup_released.set()
            execution.cancel()
            await asyncio.gather(execution, return_exceptions=True)
        assert len(backend.started) == 4
        assert all(child.done() for child in backend.owned_tasks)
        assert not backend.active_positions
        assert not api.reports
        assert campaign.cursor["phases"]["baseline-independent"]["complete"] is False
        assert not (arguments.run_root / "episodes").exists()
        resumed = workflow.run.Campaign(arguments, api, backend, manifest)
        with pytest.raises(RuntimeError, match="never replay"):
            await resumed.execute_phase("baseline-independent")
        assert len(backend.started) == 4

    asyncio.run(check())


@pytest.mark.parametrize("stored_change", ["phase-release", "task-runtime", "episode-position", "turn-release"])
def test_persisted_mutations_stop_before_fresh_evaluation(
    workflow: ModuleType, tmp_path: Path, stored_change: str
) -> None:
    async def check() -> None:
        arguments, manifest, api, backend, campaign = prepare(
            workflow, tmp_path, "baseline-independent", automatic=True
        )
        task = {**manifest["independent"][80], "campaign_position": 80}
        key = json.dumps([task["category"], task["id"]], separators=(",", ":"))
        release = copy.deepcopy(campaign.cursor["initial_release"])
        state = {
            "started_episodes": [],
            "report_attempted": [],
            "complete": False,
            "sampled_release": copy.deepcopy(release),
            "campaign_position": 80,
        }
        phase_state = {
            "tasks": {key: state},
            "complete": False,
            "release_id": release["release_id"],
            "sampled_release": copy.deepcopy(release),
        }
        campaign.cursor["phases"]["baseline-independent"] = phase_state
        if stored_change == "phase-release":
            phase_state["sampled_release"]["release_id"] = "changed"
            campaign.save()
        elif stored_change == "task-runtime":
            state["sampled_release"]["runtime_load_id"] = "changed"
            campaign.save()
        else:
            episode = await campaign.episode(task, "baseline-independent", 0, release, state)
            if stored_change == "episode-position":
                episode["campaign_position"] = 0
            else:
                episode["turns"][0]["release_id"] = "changed"
            workflow.report.write_object(arguments.run_root / "episodes" / f"{episode['episode_id']}.json", episode)
        before = len(backend.started)
        resumed = workflow.run.Campaign(arguments, api, backend, manifest)
        with pytest.raises(RuntimeError, match="sampled release changed"):
            await resumed.execute_phase("baseline-independent")
        assert len(backend.started) == before
        assert not api.reports
        assert resumed.cursor["phases"]["baseline-independent"]["complete"] is False

    asyncio.run(check())


@pytest.mark.parametrize("phase", ["baseline", "baseline-independent", "frozen-repeat", "independent"])
def test_known_zero_score_truncations_complete_evaluation_and_are_retained(
    workflow: ModuleType, tmp_path: Path, phase: str
) -> None:
    async def check() -> None:
        arguments, manifest, api, backend, campaign = prepare(workflow, tmp_path, phase, full=False, automatic=True)
        backend.truncated.add(0)
        backend.scores[0] = 0.0
        await campaign.execute_phase(phase)
        records = [workflow.report.read_object(path) for path in (arguments.run_root / "episodes").glob("*.json")]
        records.sort(key=lambda row: row["campaign_position"])
        assert [row["outcome"] for row in records] == ["truncated", "completed"]
        assert [row["score"] for row in records] == [0.0, 1.0]
        primary = workflow.metrics.build_summary(records)["phases"][phase]["one_attempt"]
        assert primary["valid"] == 2
        assert primary["episodes"] == 2
        assert primary["truncated"] == 1
        assert primary["faulted"] == 0
        assert primary["accuracy"] == 0.5
        assert all(row["runtime_load_id"] == api.current["runtime_load_id"] for row in records)
        assert campaign.cursor["phases"][phase]["complete"] is True
        assert len(backend.started) == 2
        assert not api.reports
        assert not api.history
        resumed = workflow.run.Campaign(arguments, api, backend, manifest)
        await resumed.execute_phase(phase)
        assert len(backend.started) == 2

    asyncio.run(check())


@pytest.mark.parametrize("bad_score", [None, True, 1.0, float("nan"), float("inf")])
def test_truncation_requires_zero_binary_score_and_waits_for_all(
    workflow: ModuleType, tmp_path: Path, bad_score: float | None
) -> None:
    async def check() -> None:
        _, _, api, backend, campaign = prepare(workflow, tmp_path, "baseline-independent", full=False, automatic=True)
        backend.truncated.add(0)
        backend.scores[0] = bad_score
        with pytest.raises((RuntimeError, ValueError)):
            await campaign.execute_phase("baseline-independent")
        assert len(backend.finished) == 2
        assert not backend.active_positions
        assert all(child.done() for child in backend.owned_tasks)
        assert not api.reports
        assert campaign.cursor["phases"]["baseline-independent"]["complete"] is False

    asyncio.run(check())


@pytest.mark.parametrize("runtime_load_id", [None, "changed"])
def test_truncated_runtime_identity_must_match_every_turn(
    workflow: ModuleType, tmp_path: Path, runtime_load_id: str | None
) -> None:
    async def check() -> None:
        _, _, api, backend, campaign = prepare(workflow, tmp_path, "baseline-independent", full=False, automatic=True)
        backend.truncated.add(0)
        backend.scores[0] = 0.0
        backend.runtime_changes[0] = runtime_load_id
        with pytest.raises(RuntimeError, match="runtime load changed or is missing"):
            await campaign.execute_phase("baseline-independent")
        assert len(backend.finished) == 2
        assert not backend.active_positions
        assert not api.reports
        assert campaign.cursor["phases"]["baseline-independent"]["complete"] is False

    asyncio.run(check())


def test_training_still_rejects_zero_score_truncation(workflow: ModuleType, tmp_path: Path) -> None:
    async def check() -> None:
        _, _, api, backend, campaign = prepare(workflow, tmp_path, "train", full=False, automatic=True)
        backend.truncated.add(0)
        backend.scores[0] = 0.0
        with pytest.raises(RuntimeError, match="fault or truncation"):
            await campaign.execute_phase("train")
        assert not api.reports
        assert not api.history
        assert campaign.cursor["phases"]["train"]["complete"] is False

    asyncio.run(check())


@pytest.mark.parametrize(
    "reason",
    [
        "student response or episode token window was exhausted",
        "student code tool timed out",
        "maximum student turns reached without a final submission",
    ],
)
def test_zero_score_evaluation_limits_count_in_accuracy_and_gains(workflow: ModuleType, reason: str) -> None:
    records = []
    for phase in ("baseline", "train", "frozen-repeat", "baseline-independent", "independent"):
        for position in range(2):
            truncated = phase in ("baseline", "frozen-repeat", "baseline-independent") and position == 0
            records.append(
                {
                    "phase": phase,
                    "outcome": "truncated" if truncated else "completed",
                    "fault": reason if truncated else None,
                    "score": 0.0 if truncated else 1.0,
                    "attempt": 0,
                    "category": "new",
                    "task_id": str(position),
                    "position": position,
                    "campaign_position": position,
                    "release_id": "base",
                }
            )
    summary = workflow.metrics.build_summary(records)
    assert summary["gains"] == {"PG": 50.0, "SG": -50.0, "GG": 50.0}
    assert summary["phases"]["baseline"]["one_attempt"]["accuracy"] == 0.5
    assert summary["phases"]["baseline"]["one_attempt"]["valid"] == 2
    assert summary["phases"]["baseline"]["one_attempt"]["truncated"] == 1
    failed_training = copy.deepcopy(records)
    failed_training[2].update(outcome="truncated", fault=reason, score=0.0)
    rejected = workflow.metrics.build_summary(failed_training)
    assert rejected["gains"]["PG"] is None
    assert rejected["gains"]["SG"] is None
    assert rejected["phases"]["train"]["one_attempt"]["valid"] == 1


@pytest.mark.parametrize(
    "outcome,score,reason",
    [
        ("fault", 0.0, "student response or episode token window was exhausted"),
        ("truncated", None, "student response or episode token window was exhausted"),
        ("truncated", 0.0, "unknown backend failure"),
    ],
)
def test_faults_and_unknown_limits_never_become_zero_score_attempts(
    workflow: ModuleType, outcome: str, score: float | None, reason: str
) -> None:
    record = {
        "phase": "independent",
        "outcome": outcome,
        "score": score,
        "fault": reason,
        "attempt": 0,
        "category": "new",
        "task_id": "0",
    }
    summary = workflow.metrics.build_summary([record])["phases"]["independent"]["one_attempt"]
    assert summary["valid"] == 0
    assert summary["accuracy"] is None
    assert summary["faulted"] == int(outcome == "fault")
    assert summary["truncated"] == int(outcome == "truncated")


@pytest.mark.parametrize(
    "mutation", ["none", "last-receipt", "last-release", "last-runtime", "unknown-limit", "train"]
)
def test_independent_verification_accepts_only_intact_known_evaluation_limits(
    workflow: ModuleType, tmp_path: Path, mutation: str
) -> None:
    async def check() -> None:
        arguments, _, api, backend, campaign = prepare(workflow, tmp_path, "baseline", full=False, automatic=True)
        backend.truncated.add(0)
        backend.scores[0] = 0.0
        for phase in ("baseline", "baseline-independent"):
            await campaign.execute_phase(phase)
        backend.truncated.clear()
        await campaign.execute_phase("train")
        backend.truncated.add(0)
        for phase in ("frozen-repeat", "independent"):
            await campaign.execute_phase(phase)
        paths = list((arguments.run_root / "episodes").glob("*.json"))
        records = [workflow.report.read_object(path) for path in paths]
        if mutation != "none":
            phase = "train" if mutation == "train" else "independent"
            index = next(
                index for index, row in enumerate(records) if row["phase"] == phase and row["campaign_position"] == 0
            )
            record = records[index]
            if mutation == "last-receipt":
                record["turns"][-1]["receipt"] = "changed"
            elif mutation == "last-release":
                record["turns"][-1]["release_id"] = "changed"
            elif mutation == "last-runtime":
                record["turns"][-1]["runtime_load_id"] = "changed"
            elif mutation == "unknown-limit":
                record["fault"] = "unknown backend failure"
            else:
                record.update(
                    outcome="truncated", score=0.0, fault="student response or episode token window was exhausted"
                )
            workflow.report.write_object(paths[index], record)
        workflow.report.write_object(arguments.run_root / "summary.json", workflow.metrics.build_summary(records))
        verified = workflow.verification.verify_run(arguments.run_root)
        assert verified["local_checks_passed"] is (mutation == "none")
        assert verified["complete"] is False
        if mutation == "none":
            assert verified["summary"]["phases"]["independent"]["one_attempt"]["accuracy"] == 0.5
        assert len(api.history) == 2
        assert not backend.active_positions

    asyncio.run(check())
