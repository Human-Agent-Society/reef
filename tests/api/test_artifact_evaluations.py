"""A client evaluates an artifact of a snapshot: Reef runs the rollouts and stores their policy evaluation."""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import math
import tarfile
import time
import urllib.error
import urllib.request
import uuid
import weakref
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from urllib.parse import urlencode

import pytest
from aiohttp.test_utils import TestClient, TestServer
from tests.support.task_service_stubs import task_for
from tests.support.trajectory_uploads import dispatcher_for

from reef.api.app import create_app
from reef.artifact import InMemoryRepositoryBackend
from reef.artifact.artifact import Artifact
from reef.artifact.components import ComponentSpec, ScenarioComponents
from reef.control.dispatcher import Dispatcher
from reef.control.operations.artifact_evaluations import (
    CANCELLED_ERROR,
    FAILURE_CHECK_ROLLOUTS,
    FAILURES_ERROR,
    READ_CALLS,
    STOPPED_ERROR,
    ArtifactEvaluations,
    NewArtifact,
    NewArtifacts,
    StoredCalls,
    attempt_trajectories,
    iter_attempt_trajectories,
    rollout_work,
)
from reef.control.operations.data import DataAccess
from reef.control.scenario.live_objects import LiveObjects
from reef.control.scenario.snapshots import ScenarioSnapshots
from reef.core.artifact_selection import ArtifactSelection
from reef.core.errors import ReefError, UpstreamLimit, UpstreamStatusError
from reef.core.evaluation import (
    ArtifactEvaluationRequest,
    ArtifactEvaluationSettings,
    EvaluationCall,
    EvaluationCallSummary,
    PolicyModel,
    RolloutScore,
)
from reef.core.served_models import ServedArtifact
from reef.core.snapshot import SnapshotRef
from reef.core.tasks import write_harbor_task
from reef.core.trajectories import CONVERSATION_FIELDS
from reef.evaluation.activation import AlwaysActivate, EvaluatedActivation, NeverActivate, activation_reference
from reef.evaluation.rollout_worker import (
    EVALUATION_ROLLOUT_KIND,
    EVALUATIONS_QUEUE,
    TASK_TEMPLATE_KIND,
    EvaluationRollout,
    EvaluationRolloutWorker,
    LocalCandidateEndpoints,
    TaskTemplateWorker,
    rollout_item_id,
    template_item_id,
)
from reef.evaluation.task_templates import TaskTemplateBuilder
from reef.harness.adapters import get_adapter
from reef.harness.harbor.agent import TreeAgent
from reef.harness.harbor.rollouts import RolloutRow, TaskLab
from reef.harness.sync import HARNESS_COMPONENT, HarnessFiles
from reef.harness.tree import HarnessTree
from reef.inference.interfaces import InferenceHandler, InferenceRuntime
from reef.recipe import Recipe
from reef.recipe.composite import CompositeRecipe
from reef.storage.coordination.work_queue import NewWork, WorkItem
from reef.storage.data.store import MemoryDataStore, NewItem
from reef.storage.data.tasks import LocalTaskFileStore, files_digest
from reef.storage.scenario.commits import CommitRecord
from reef.storage.scenario.evaluations import EvaluationStore
from reef.storage.scenario.observer import ObservedScenarioStorage
from reef.storage.scenario.record_items import evaluation_call_item
from reef.storage.sql.sqlite import SQLiteScenarioDatabase, SQLiteScenarioStorage

TOKEN = "artifact-evaluation-test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


class ArtifactHandler(InferenceHandler):
    """Answers every chat call, and records the artifact that it was asked to answer from."""

    def __init__(self) -> None:
        self.artifact_ids: list[str] = []
        self.models: list[str] = []

    async def inference(self, artifact: Artifact, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.artifact_ids.append(artifact.artifact_id)
        self.models.append(payload["model"])
        return {"choices": [{"message": {"role": "assistant", "content": "ls"}, "finish_reason": "stop"}]}


class ArtifactRuntime(InferenceRuntime):
    """A runtime that answers each call from the snapshot that the call names, as a sampler for each checkpoint.

    Its engine serves every scenario under one model name, so control writes the ``served_models`` rows that the
    inference routes answer from."""

    def __init__(self, *, any_artifact: bool = True) -> None:
        super().__init__(base_url="http://runtime.invalid")
        self.handler = ArtifactHandler()
        self.served: list[str] = []
        #: The scenario of each served artifact, and each serving config that inference applied.
        self.serve_scenarios: list[str] = []
        self.applied: list[str] = []
        self.serve_configs: list[dict] = []
        self.serve_error: str | None = None
        self.any_artifact = any_artifact

    @property
    def inference_handler(self) -> InferenceHandler:
        return self.handler

    @property
    def served_weight_kind(self):
        return "model_adapter"

    def served_model_name(self, scenario: str) -> str:
        return "base"

    @property
    def serves_any_artifact(self) -> bool:
        return self.any_artifact

    def artifact_model(self, artifact: Artifact) -> PolicyModel:
        return PolicyModel("base", f"sampler:{artifact.artifact_id}")

    def apply_serving_config(self, scenario, override):
        self.applied.append(scenario)

    def serve_artifact(self, artifact: Artifact, *, scenario: str, serving_config) -> ServedArtifact:
        self.served.append(artifact.artifact_id)
        self.serve_configs.append(dict(serving_config))
        self.serve_scenarios.append(scenario)
        if self.serve_error is not None:
            raise ValueError(self.serve_error)
        return ServedArtifact(artifact.artifact_id, self.base_url, f"engine:{artifact.artifact_id}")

    def activate_checkpoint(self, artifact, *, scenario=None):
        return "load-0"

    def served_artifact(self, artifact_id: str, scenario: str) -> ServedArtifact:
        """The test binding: the test engine answers every scenario's active artifact under ``base``."""
        return ServedArtifact(artifact_id, self.base_url, self.served_model_name(scenario))


@dataclass(frozen=True, kw_only=True)
class WeightRecipe(Recipe):
    """A record-only recipe that serves weights on its runtime."""

    def build_components(self, scenario: str) -> ScenarioComponents:
        return ScenarioComponents({"weights": ComponentSpec(kind="model_adapter")})


@dataclass(frozen=True, kw_only=True)
class HarnessRecipe(Recipe):
    """A record-only recipe that serves a ``pi`` harness tree."""

    def build_components(self, scenario: str) -> ScenarioComponents:
        return ScenarioComponents(
            {HARNESS_COMPONENT: ComponentSpec(files=HarnessFiles(get_adapter("pi")), kind="harness_plugin")}
        )


class ModelCallingLab(TaskLab):
    """Calls the model once through the agent's api_base, then scores the rollout with the task's reward."""

    def __init__(self, rewards: dict[str, float], failing: frozenset[str] = frozenset()) -> None:
        self.rewards = rewards
        self.failing = failing

    async def run(self, task_path, agent, *, key, tags, overrides) -> RolloutRow:
        if task_path.name in self.failing:
            return RolloutRow(rewards={}, error="the sandbox did not start")
        body = json.dumps({"model": "m", "messages": [{"role": "user", "content": f"solve {task_path.name}"}]})
        request = urllib.request.Request(
            f"{agent['kwargs']['api_base']}/chat/completions",
            data=body.encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        await asyncio.to_thread(lambda: urllib.request.urlopen(request, timeout=10).read())
        return RolloutRow(rewards={"reward": self.rewards[task_path.name]})

    async def build_environment(self, task_path, *, overrides) -> str:
        return ""


def archive(root: Path, *names: str) -> bytes:
    """A tar.gz of one Harbor task directory for each name."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for port, name in enumerate(names, start=9000):
            tar.add(write_harbor_task(task_for(name, "record-1", port), root / str(port)), arcname=name)
    return buffer.getvalue()


def evaluating_server(dispatcher: Dispatcher, lab: TaskLab, **options: Any) -> tuple[TestServer, ArtifactEvaluations]:
    """The service and its evaluations. As in one process, the evaluation workers serve each candidate themselves;
    ``options`` go to ``ArtifactEvaluations``."""
    evaluations = ArtifactEvaluations(dispatcher, **options)
    dispatcher.workers.add(
        EvaluationRolloutWorker(
            dispatcher.queue, LocalCandidateEndpoints(dispatcher), worker_id="evaluation-test", lab=lab
        )
    )
    return TestServer(create_app(dispatcher, tokens=TOKEN, evaluations=evaluations)), evaluations


async def finished(client: TestClient, evaluation_id: str) -> dict[str, Any]:
    for _ in range(600):
        answer = await (await client.get(f"/reef/scenarios/s/evaluations/{evaluation_id}", headers=AUTH)).json()
        if answer["state"] not in ("queued", "running"):
            return answer
        await asyncio.sleep(0.05)
    raise AssertionError("the evaluation did not end")


async def created(
    client: TestClient, tmp_path: Path, *names: str, owner: str | None = None, scenario_name: str = "s"
) -> str:
    """Create ``scenario_name`` with eval tasks ``names`` in the data of ``owner`` (None: the scenario's own name),
    and return its active artifact."""
    scenario: dict[str, str] = {"name": scenario_name} if owner is None else {"name": scenario_name, "owner": owner}
    await client.post("/reef/scenarios", headers=AUTH, json=scenario)
    metadata = urlencode({"metadata": json.dumps({"scenarios": {scenario_name: "eval"}})})
    upload = await client.post(
        f"/reef/namespaces/{owner or scenario_name}/data/tasks?{metadata}",
        headers={**AUTH, "Content-Type": "application/gzip"},
        data=archive(tmp_path, *names),
    )
    assert upload.status == 201, await upload.text()
    artifacts = await (await client.get(f"/reef/scenarios/{scenario_name}/artifacts", headers=AUTH)).json()
    return next(row["artifact_id"] for row in artifacts["artifacts"] if row["active"])


@pytest.mark.integration
def test_evaluation_start_id_replays_one_evaluation_and_rejects_a_different_request(tmp_path: Path) -> None:
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(), live=LiveObjects(runtime=ArtifactRuntime()))
    route = "/reef/scenarios/s/evaluations"

    async def run() -> None:
        async with TestClient(TestServer(create_app(dispatcher, tokens=TOKEN))) as client:
            artifact_id = await created(client, tmp_path, "add")
            capability = await client.get("/reef/scenarios/s/evaluation-capabilities", headers=AUTH)
            assert capability.status == 200 and await capability.json() == {"evaluation_start_id": True}
            body = {"artifact_id": artifact_id, "harness": {"name": "terminus-2"}, "start_id": "request-1"}
            responses = await asyncio.gather(
                client.post(route, headers=AUTH, json=body),
                client.post(route, headers=AUTH, json=dict(reversed(tuple(body.items())))),
            )
            assert [response.status for response in responses] == [202, 202]
            first, second = [await response.json() for response in responses]
            evaluation_id = first["evaluation_id"]
            assert second["evaluation_id"] == evaluation_id
            conflict = await client.post(route, headers=AUTH, json={**body, "rollouts_per_task": 1})
            assert conflict.status == 409
            for invalid_id in ("", " \t"):
                invalid = await client.post(route, headers=AUTH, json={**body, "start_id": invalid_id})
                assert invalid.status == 400
            cancelled = await client.post(f"{route}/{evaluation_id}/cancel", headers=AUTH)
            assert cancelled.status == 200
            replay = await client.post(route, headers=AUTH, json=body)
            assert replay.status == 202
            accepted = await replay.json()
            assert accepted["evaluation_id"] == evaluation_id and accepted["state"] == "failed"
            assert len(dispatcher.queue.items(EVALUATIONS_QUEUE, scenario="s")) == 1
            other_artifact = await created(client, tmp_path / "other", "sort", scenario_name="other")
            separate = await client.post(
                "/reef/scenarios/other/evaluations", headers=AUTH, json={**body, "artifact_id": other_artifact}
            )
            assert separate.status == 202
            assert (await separate.json())["evaluation_id"] != evaluation_id

    asyncio.run(run())


@pytest.mark.integration
def test_a_store_without_keyed_starts_keeps_unkeyed_evaluations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluation_store = SQLiteScenarioDatabase.evaluation_store

    def open_legacy_store(database: SQLiteScenarioDatabase) -> EvaluationStore:
        store = evaluation_store(database)

        class LegacyStore(EvaluationStore):
            create = staticmethod(store.create)
            begin = staticmethod(store.begin)
            finish = staticmethod(store.finish)
            fail = staticmethod(store.fail)
            reopen = staticmethod(store.reopen)
            limit_rollouts = staticmethod(store.limit_rollouts)
            read = staticmethod(store.read)
            newest = staticmethod(store.newest)

        return LegacyStore()

    monkeypatch.setattr(SQLiteScenarioDatabase, "evaluation_store", open_legacy_store)
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(), live=LiveObjects(runtime=ArtifactRuntime()))

    async def run() -> None:
        async with TestClient(TestServer(create_app(dispatcher, tokens=TOKEN))) as client:
            artifact_id = await created(client, tmp_path, "add")
            route = "/reef/scenarios/s/evaluations"
            capability = await client.get("/reef/scenarios/s/evaluation-capabilities", headers=AUTH)
            assert capability.status == 200 and await capability.json() == {"evaluation_start_id": False}
            body = {"artifact_id": artifact_id, "harness": {"name": "terminus-2"}}
            unsupported = await client.post(route, headers=AUTH, json={**body, "start_id": "request-1"})
            assert unsupported.status == 501
            assert dispatcher.queue.items(EVALUATIONS_QUEUE, scenario="s") == ()
            unkeyed = await client.post(route, headers=AUTH, json=body)
            assert unkeyed.status == 202
            accepted = await unkeyed.json()
            cancelled = await client.post(f"{route}/{accepted['evaluation_id']}/cancel", headers=AUTH)
            assert cancelled.status == 200
            resumed = await client.post(f"{route}/{accepted['evaluation_id']}/resume", headers=AUTH)
            assert resumed.status == 202

    asyncio.run(run())


@pytest.mark.integration
@pytest.mark.parametrize("interruption", ["before_append", "after_append", "cancelled", "expired"])
def test_a_pending_evaluation_start_recovers_after_an_interrupted_append(
    tmp_path: Path, monkeypatch, interruption: str
) -> None:
    factory = InMemoryRepositoryBackend.factory(directory=tmp_path / "repository")
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(), factory, live=LiveObjects(runtime=ArtifactRuntime()))
    evaluations = ArtifactEvaluations(dispatcher, pin_task_files=True)
    original_append = dispatcher.queue.append
    restored_body: dict[str, object] = {}
    accepted_digest = ""

    async def interrupted() -> str:
        nonlocal accepted_digest
        async with TestClient(TestServer(create_app(dispatcher, tokens=TOKEN, evaluations=evaluations))) as client:
            artifact_id = await created(client, tmp_path, "add")
            original = dispatcher.data_store.read("s", "add")
            assert original is not None
            accepted_digest = original.digest
            restored_body.update(artifact_id=artifact_id, harness={"name": "terminus-2"}, start_id="request-1")

            def interrupt(work: NewWork) -> WorkItem:
                if work.queue != EVALUATIONS_QUEUE:
                    return original_append(work)
                if interruption in ("after_append", "expired"):
                    original_append(work)
                raise RuntimeError("control stopped before it confirmed the append")

            with monkeypatch.context() as patch:
                patch.setattr(dispatcher.queue, "append", interrupt)
                with pytest.raises(RuntimeError, match="control stopped"):
                    evaluations.start("s", restored_body)
            [stored] = evaluations.newest("s")
            if interruption == "after_append":
                replay = evaluations.start("s", restored_body)
                assert replay.evaluation_id == stored.evaluation_id
                assert len(dispatcher.queue.items(EVALUATIONS_QUEUE, scenario="s")) == 1
            if interruption == "cancelled":
                cancelled = await client.post(
                    f"/reef/scenarios/s/evaluations/{stored.evaluation_id}/cancel", headers=AUTH
                )
                assert cancelled.status == 200
            if interruption == "expired":
                with dispatcher.locked_scenario("s"):
                    worker = "first-worker"
                    first = dispatcher.queue.take(EVALUATIONS_QUEUE, worker, [EVALUATION_ROLLOUT_KIND])
                    assert first is not None
                    dispatcher.queue.start(EVALUATIONS_QUEUE, first.id, worker)
                    dispatcher.queue.finish(
                        EVALUATIONS_QUEUE, first.id, worker, {"score": RolloutScore(1.0).to_json()}
                    )
                    with monkeypatch.context() as patch:
                        patch.setattr(dispatcher, "work_retention_seconds", 0.000001)
                        dispatcher.queue.delete_final(finished_before=time.time())
                        assert dispatcher.queue.read(EVALUATIONS_QUEUE, first.id) is None
                        replay = evaluations.start("s", restored_body)
                        assert replay.evaluation_id == stored.evaluation_id
                        assert replay.state == "failed" and replay.error == STOPPED_ERROR
                        assert not dispatcher.queue.items(EVALUATIONS_QUEUE, scenario="s")
            if interruption == "before_append":
                replacement = write_harbor_task(task_for("add", "record-2", 9000), tmp_path / "changed")
                changed = dispatcher.data_store.replace_task("s", "add", replacement)
                assert changed.digest != accepted_digest
            return stored.evaluation_id

    evaluation_id = asyncio.run(interrupted())
    if interruption in ("after_append", "expired"):
        return

    restarted = dispatcher_for(tmp_path, WeightRecipe(), factory, live=LiveObjects(runtime=ArtifactRuntime()))

    async def after_restart() -> None:
        server, restored_evaluations = evaluating_server(restarted, ModelCallingLab({"add": 1.0}))
        async with TestClient(server) as client:
            if interruption == "cancelled":
                restored_evaluations.results.fail_stopped()
                assert not restarted.queue.items(EVALUATIONS_QUEUE, scenario="s")
                resumed = await client.post(f"/reef/scenarios/s/evaluations/{evaluation_id}/resume", headers=AUTH)
                assert resumed.status == 202, await resumed.text()
            answer = await finished(client, evaluation_id)
            assert answer["state"] == "finished", answer
            replay = await client.post("/reef/scenarios/s/evaluations", headers=AUTH, json=restored_body)
            assert replay.status == 202
            assert (await replay.json())["evaluation_id"] == evaluation_id
            [rollout] = restarted.queue.items(EVALUATIONS_QUEUE, scenario="s")
            path = LocalCandidateEndpoints(restarted).task_directory(EvaluationRollout.from_item(rollout))
            assert files_digest(path) == accepted_digest

    asyncio.run(after_restart())


@pytest.mark.integration
def test_a_model_artifact_is_evaluated_with_the_configured_harness_on_the_snapshot_it_names(tmp_path: Path) -> None:
    runtime = ArtifactRuntime()
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(), live=LiveObjects(runtime=runtime))
    lab = ModelCallingLab({"add": 1.0, "sort": 0.0}, failing=frozenset({"broken"}))

    async def run() -> None:
        server, _ = evaluating_server(dispatcher, lab)
        async with TestClient(server) as client:
            artifact_id = await created(client, tmp_path, "add", "broken", "sort")
            body = {"artifact_id": artifact_id, "harness": {"name": "terminus-2"}, "rollouts_per_task": 2}
            response = await client.post("/reef/scenarios/s/evaluations", headers=AUTH, json=body)
            assert response.status == 202, await response.text()
            accepted = await response.json()
            assert accepted["state"] == "running" and accepted["request"]["component"] == "weights"
            assert (dispatcher.candidate_task("s", "add") / "task.toml").is_file()
            with pytest.raises(ReefError, match="uses no task"):
                dispatcher.candidate_task("s", "missing")
            # A worker in another Pod finds each task by its name: a path here would mean nothing there.
            queued = dispatcher.queue.items(EVALUATIONS_QUEUE, scenario="s")
            assert queued and all("task_path" not in item.payload for item in queued)

            answer = await finished(client, accepted["evaluation_id"])
            assert answer["state"] == "finished", answer
            assert answer["policy_evaluation"] == {
                "harness": {"base": "terminus-2", "plugin": {}},
                "model": {"name": "base", "adapter": f"sampler:{artifact_id}"},
                "scores": [
                    {"task": "add", "scores": [1.0, 1.0]},
                    {"task": "broken", "scores": ["failed", "failed"]},
                    {"task": "sort", "scores": [0.0, 0.0]},
                ],
            }
            assert answer["mean_score"] == 0.5
            assert runtime.handler.artifact_ids == [artifact_id] * 4, "every call samples the evaluated artifact"
            assert runtime.handler.models == [f"engine:{artifact_id}"] * 4, "every call names the engine's model"
            # Each broken rollout's sandbox does not start: it reaches no model, so all 3 of its attempts run.
            assert runtime.served == [artifact_id] * 10, "inference serves the candidate before each trial"
            newest = dispatcher.coordination.served_models.current_request("s").payload.get("serving_config", {})
            assert (
                runtime.serve_scenarios == ["s"] * 10 and runtime.serve_configs == [newest] * 10
            ), "inference serves the candidate with the config of the scenario's newest request"
            routes = dispatcher.coordination.served_models.read_scenario("s")
            assert set(routes) == {
                "active",
                artifact_id,
            }, "the active artifact's two names, and no route of the evaluation"
            rollouts_route = f"/reef/scenarios/s/evaluations/{accepted['evaluation_id']}/rollouts"
            rollouts = (await (await client.get(rollouts_route, headers=AUTH)).json())["rollouts"]
            assert [(rollout["task"], rollout["repeat"], rollout["score"]) for rollout in rollouts] == [
                ("add", 0, 1.0),
                ("add", 1, 1.0),
                ("broken", 0, None),
                ("broken", 1, None),
                ("sort", 0, 0.0),
                ("sort", 1, 0.0),
            ]
            assert [rollout["state"] for rollout in rollouts] == ["finished"] * 2 + ["failed"] * 2 + ["finished"] * 2
            assert [rollout["rewards"] for rollout in rollouts] == [{"reward": 1.0}] * 2 + [None] * 2 + [
                {"reward": 0.0}
            ] * 2
            assert [rollout["error"] for rollout in rollouts if rollout["task"] == "broken"] == [
                "ReefError: the rollout reached no model: the sandbox did not start"
            ] * 2
            assert all(rollout["started_at"] <= rollout["ended_at"] for rollout in rollouts)
            one_call = {"count": 1, "failed": 0, "prompt_tokens": 0, "completion_tokens": 0}
            assert [rollout["calls"] for rollout in rollouts] == [one_call, one_call, None, None, one_call, one_call]
            calls_route = f"{rollouts_route}/0/1/calls"
            # The polled routes read the calls' metadata only: every query is without content, and nothing is read.
            store = dispatcher.data_store
            reads: list[tuple[str, bool]] = []
            query, read_many = store.query, store.read_many

            def spied_query(*args: Any, **kwargs: Any) -> Any:
                reads.append(("query", kwargs.get("content", True)))
                return query(*args, **kwargs)

            def spied_read_many(*args: Any, **kwargs: Any) -> Any:
                reads.append(("read_many", kwargs.get("content", True)))
                return read_many(*args, **kwargs)

            store.query, store.read_many = spied_query, spied_read_many
            try:
                for route in (rollouts_route, calls_route):
                    assert (await client.get(route, headers=AUTH)).status == 200
            finally:
                del store.query, store.read_many
            assert reads and set(reads) == {("query", False)}
            listed = await (await client.get(calls_route, headers=AUTH)).json()
            assert (listed["task_index"], listed["repeat"]) == (0, 1)
            [summary] = listed["calls"]
            assert (summary["attempt"], summary["index"], summary["status"], summary["finish_reason"]) == (
                1,
                0,
                200,
                "stop",
            )
            assert summary["duration_s"] >= 0 and "request" not in summary
            kept = await (await client.get(f"{calls_route}/1/0", headers=AUTH)).json()
            assert kept["request"]["messages"] == [{"role": "user", "content": "solve add"}]
            assert kept["response"]["choices"][0]["finish_reason"] == "stop"
            broken = await (await client.get(f"{rollouts_route}/1/0/calls", headers=AUTH)).json()
            assert broken["calls"] == [], "a rollout that reached no model keeps no call"
            for route in (f"{calls_route}/1/1", "/reef/scenarios/s/evaluations/missing/rollouts/0/0/calls"):
                assert (await client.get(route, headers=AUTH)).status == 404
            assert (await client.get(f"{rollouts_route}/x/0/calls", headers=AUTH)).status == 400
            drawer = await (await client.get(f"{rollouts_route}/0/1/trajectories", headers=AUTH)).json()
            [trajectory] = drawer["trajectories"]
            assert (drawer["task_index"], drawer["repeat"]) == (0, 1)
            assert trajectory["extra"]["reef"]["reward"] == 1.0
            assert trajectory["extra"]["reef"]["evaluation"]["scored"] is True
            broken = await (await client.get(f"{rollouts_route}/1/0/trajectories", headers=AUTH)).json()
            assert broken["trajectories"] == []
            export_route = f"/reef/scenarios/s/evaluations/{accepted['evaluation_id']}/trajectories"
            for query in ("", "?attempts=all"):
                exported = await client.get(f"{export_route}{query}", headers=AUTH)
                assert exported.headers["Content-Type"] == "application/x-ndjson"
                lines = [json.loads(line) for line in (await exported.text()).splitlines()]
                assert [
                    (line["extra"]["reef"]["evaluation"]["task"], line["extra"]["reef"]["reward"]) for line in lines
                ] == [("add", 1.0), ("add", 1.0), ("sort", 0.0), ("sort", 0.0)]
            assert (await client.get(f"{export_route}?attempts=last", headers=AUTH)).status == 400
            uploaded = await client.post(
                "/reef/namespaces/s/data/trajectories",
                headers={**AUTH, "Content-Type": "application/x-ndjson"},
                data=await (await client.get(export_route, headers=AUTH)).read(),
            )
            assert uploaded.status == 201, await uploaded.text()
            for route in (
                "/reef/scenarios/s/evaluations/missing/trajectories",
                "/reef/scenarios/s/evaluations/missing/rollouts/0/0/trajectories",
            ):
                assert (await client.get(route, headers=AUTH)).status == 404
            missing = await client.get("/reef/scenarios/s/evaluations/missing/rollouts", headers=AUTH)
            assert missing.status == 404
            listing = await (await client.get("/reef/scenarios/s/evaluations", headers=AUTH)).json()
            assert [item["evaluation_id"] for item in listing["evaluations"]] == [accepted["evaluation_id"]]
            # The evaluation keeps no record and changes nothing the scenario serves.
            records = await (await client.get("/reef/scenarios/s/records", headers=AUTH)).json()
            assert records["records"] == []

    asyncio.run(run())


class JudgingLab(ModelCallingLab):
    """As ``ModelCallingLab``, and the verifier of each trial writes two judge calls, a bad line and a trajectory."""

    def __init__(self, rewards: dict[str, float], trials: Path) -> None:
        super().__init__(rewards)
        self.trials = trials

    async def run(self, task_path, agent, *, key, tags, overrides) -> RolloutRow:
        row = await super().run(task_path, agent, key=key, tags=tags, overrides=overrides)
        judge = self.trials / key / "verifier" / "judge"
        judge.mkdir(parents=True)
        asked = [{"role": "user", "content": "rate the answer"}]
        lines = [
            {
                "request": {"model": "judge", "messages": asked},
                "response": {"choices": [{"message": {"role": "assistant", "content": "4"}, "finish_reason": "stop"}]},
                "started_at": 5.0,
                "duration_s": 0.5,
                "label": "judge:quality",
            },
            {"request": {"model": "judge", "messages": asked}, "response": {"error": "busy"}, "status": 429,
             "started_at": 6.0, "duration_s": 0.1},
        ]  # fmt: skip
        (judge / "calls.jsonl").write_text("\n".join([*map(json.dumps, lines), "not json"]) + "\n")
        document = {"schema_version": "ATIF-v1.4", "agent": {"name": "judge"}, "steps": [{"source": "user"}]}
        (judge / "trajectory.json").write_text(json.dumps(document))
        rewards = {**row.rewards, "rule": 1.0, "judge": 0.5, "unbounded": math.inf}
        return replace(row, rewards=rewards, trial_uri=(self.trials / key).as_uri())


@pytest.mark.integration
def test_an_evaluation_keeps_the_calls_and_trajectories_that_its_verifier_wrote(tmp_path: Path) -> None:
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(), live=LiveObjects(runtime=ArtifactRuntime()))
    lab = JudgingLab({"add": 1.0}, tmp_path / "trials")

    async def run() -> None:
        server, _ = evaluating_server(dispatcher, lab)
        async with TestClient(server) as client:
            artifact_id = await created(client, tmp_path, "add")
            body = {"artifact_id": artifact_id, "harness": {"name": "terminus-2"}}
            accepted = await (await client.post("/reef/scenarios/s/evaluations", headers=AUTH, json=body)).json()
            assert (await finished(client, accepted["evaluation_id"]))["mean_score"] == 1.0
            rollouts_route = f"/reef/scenarios/s/evaluations/{accepted['evaluation_id']}/rollouts"
            [rollout] = (await (await client.get(rollouts_route, headers=AUTH)).json())["rollouts"]
            assert rollout["calls"]["count"] == 1, "the agent's totals leave out the verifier's calls"
            assert rollout["score"] == 1.0, "the score is the primary reward"
            assert rollout["rewards"] == {"reward": 1.0, "rule": 1.0, "judge": 0.5}, "only finite rewards stay"
            assert rollout["verifier_calls"] == {"count": 2, "failed": 1, "prompt_tokens": 0, "completion_tokens": 0}
            without_calls = (await (await client.get(f"{rollouts_route}?calls=false", headers=AUTH)).json())[
                "rollouts"
            ]
            assert without_calls == [
                {key: value for key, value in rollout.items() if key not in ("calls", "verifier_calls")}
            ], "a read without the call totals keeps every other field"
            assert (await client.get(f"{rollouts_route}?calls=no", headers=AUTH)).status == 400
            calls_route = f"{rollouts_route}/0/0/calls"
            agent_calls = (await (await client.get(calls_route, headers=AUTH)).json())["calls"]
            assert [call["caller"] for call in agent_calls] == ["agent"]
            listed = (await (await client.get(f"{calls_route}?caller=verifier", headers=AUTH)).json())["calls"]
            assert [(call["index"], call["status"], call["label"], call["finish_reason"]) for call in listed] == [
                (0, 200, "judge:quality", "stop"),
                (1, 429, None, None),
            ]
            judged = await (await client.get(f"{calls_route}/1/0?caller=verifier", headers=AUTH)).json()
            assert judged["caller"] == "verifier" and judged["response"]["choices"][0]["message"]["content"] == "4"
            agent_call = await (await client.get(f"{calls_route}/1/0", headers=AUTH)).json()
            assert agent_call["request"]["messages"] == [{"role": "user", "content": "solve add"}]
            assert (await client.get(f"{calls_route}/1/2?caller=verifier", headers=AUTH)).status == 404
            assert (await client.get(f"{calls_route}?caller=judge", headers=AUTH)).status == 400
            trajectories_route = f"{rollouts_route}/0/0/trajectories"
            [agent] = (await (await client.get(trajectories_route, headers=AUTH)).json())["trajectories"]
            assert agent["extra"]["reef"]["reward"] == 1.0
            written, stitched = (
                await (await client.get(f"{trajectories_route}?caller=verifier", headers=AUTH)).json()
            )["trajectories"]
            assert written["agent"] == {"name": "judge"}
            assert written["extra"]["reef"]["evaluation"] == {
                "evaluation_id": accepted["evaluation_id"],
                "task": "add",
                "task_index": 0,
                "repeat": 0,
                "attempt": 1,
                "caller": "verifier",
                "file": "trajectory.json",
                "scored": False,
            }
            assert stitched["extra"]["reef"]["reward"] is None
            assert stitched["extra"]["reef"]["evaluation"]["caller"] == "verifier"
            assert [record["agent_record_id"] for record in stitched["extra"]["reef"]["records"]] == [
                f"{accepted['evaluation_id']}:0:0:1:verifier:0"
            ]
            export_route = f"/reef/scenarios/s/evaluations/{accepted['evaluation_id']}/trajectories"
            exported = (await (await client.get(export_route, headers=AUTH)).text()).splitlines()
            assert len(exported) == 1, "the export holds the agent's trajectories only"

            dispatcher.queue.delete_final(finished_before=time.time())
            item_id = rollout_item_id(accepted["evaluation_id"], 0, 0)
            assert dispatcher.queue.read(EVALUATIONS_QUEUE, item_id) is None
            [unpaged] = (await (await client.get(rollouts_route, headers=AUTH)).json())["rollouts"]
            assert unpaged["calls"] == rollout["calls"]
            assert unpaged["verifier_calls"] == rollout["verifier_calls"]
            page = await (await client.get(f"{rollouts_route}?offset=0&limit=1", headers=AUTH)).json()
            assert (page["offset"], page["total"], page["rollouts"]) == (0, 1, [unpaged])

    asyncio.run(run())


@pytest.mark.integration
def test_a_rollout_page_materializes_only_its_range_of_a_large_waiting_evaluation(tmp_path: Path) -> None:
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(), live=LiveObjects(runtime=ArtifactRuntime()))

    async def run() -> None:
        async with TestClient(TestServer(create_app(dispatcher, tokens=TOKEN))) as client:
            artifact_id = await created(client, tmp_path, "add")
            request = ArtifactEvaluationRequest(
                artifact_id,
                "weights",
                harness={"name": "terminus-2"},
                task_names=tuple(f"task-{index}" for index in range(10_000)),
                rollouts_per_task=16,
                max_concurrent_rollouts=1,
            )
            with dispatcher.locked_scenario("s") as current:
                current.store.evaluations.create(
                    evaluation_id="large", owner="test", request=request, created_at=time.time()
                )
            route = "/reef/scenarios/s/evaluations/large/rollouts"
            first = await client.get(f"{route}?limit=96&calls=false", headers=AUTH)
            assert first.status == 200
            page = await first.json()
            assert (page["offset"], page["total"], len(page["rollouts"])) == (0, 160_000, 96)
            assert all(row["state"] == "waiting" and "calls" not in row for row in page["rollouts"])
            last = await client.get(f"{route}?limit=96&offset=159999", headers=AUTH)
            tail = await last.json()
            assert [(row["task"], row["repeat"], row["calls"]) for row in tail["rollouts"]] == [
                ("task-9999", 15, None)
            ]
            assert (await client.get(f"{route}?limit=501", headers=AUTH)).status == 400
            filtered = await (await client.get(f"{route}?limit=1&benchmark=full", headers=AUTH)).json()
            assert (filtered["total"], filtered["rollouts"], filtered["task_means"]) == (0, [], {})

    asyncio.run(run())


@pytest.mark.integration
def test_filtered_rollout_pages_keep_numeric_repeat_order(tmp_path: Path) -> None:
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(), live=LiveObjects(runtime=ArtifactRuntime()))
    queue = dispatcher.queue

    async def run() -> None:
        async with TestClient(TestServer(create_app(dispatcher, tokens=TOKEN))) as client:
            artifact_id = await created(client, tmp_path, "add")
            harness = {"name": "terminus-2"}
            request = ArtifactEvaluationRequest(
                artifact_id,
                "weights",
                harness=harness,
                task_names=("add",),
                rollouts_per_task=100_001,
                max_concurrent_rollouts=2,
            )
            evaluation_id = "large"
            # Keep only the two repeats at the width boundary in the queue.
            for repeat in (100_000, 99_999):
                rollout = EvaluationRollout(
                    "s",
                    evaluation_id,
                    artifact_id,
                    "weights",
                    0,
                    "add",
                    repeat,
                    PolicyModel("m"),
                    harness,
                    "docker",
                    {},
                )
                queue.append(rollout_work(rollout))
            taken = [queue.take(EVALUATIONS_QUEUE, "w", [EVALUATION_ROLLOUT_KIND]) for _ in range(2)]
            with dispatcher.locked_scenario("s") as current:
                current.store.evaluations.create(
                    evaluation_id=evaluation_id, owner="test", request=request, created_at=time.time()
                )
            calls = dispatcher.data_store
            assert calls is not None
            for repeat in (99_998, 99_999):
                summary = EvaluationCallSummary(
                    0, repeat, 1, 0, "/v1/chat/completions", 200, 10.0, 0.5, 3, 2, "stop", 0
                )
                calls.create(
                    "s", [evaluation_call_item("s", artifact_id, evaluation_id, EvaluationCall(summary, None, None))]
                )
            route = f"/reef/scenarios/s/evaluations/{evaluation_id}/rollouts"
            unfiltered = await client.get(f"{route}?offset=99999&limit=2&calls=false", headers=AUTH)
            assert unfiltered.status == 200, await unfiltered.text()
            assert [row["repeat"] for row in (await unfiltered.json())["rollouts"]] == [99_999, 100_000]
            for state in ("running", "failed"):
                for offset, repeat in enumerate((99_999, 100_000)):
                    response = await client.get(f"{route}?offset={offset}&limit=1&state={state}", headers=AUTH)
                    assert response.status == 200, await response.text()
                    page = await response.json()
                    [row] = page["rollouts"]
                    assert (page["total"], row["repeat"]) == (2, repeat)
                    if repeat == 99_999:
                        assert row["calls"] == {"count": 1, "failed": 0, "prompt_tokens": 3, "completion_tokens": 2}
                    else:
                        assert row["calls"] is None
                if state == "running":
                    for item in taken:
                        assert item is not None
                        queue.fail(EVALUATIONS_QUEUE, item.id, "w", "the sandbox stopped", retryable=False)

    asyncio.run(run())


@pytest.mark.integration
def test_an_evaluation_keeps_at_most_its_max_concurrent_rollouts_in_the_queue(tmp_path: Path) -> None:
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(), live=LiveObjects(runtime=ArtifactRuntime()))
    queue = dispatcher.queue

    def scored(item: WorkItem, value: float) -> None:
        queue.start(EVALUATIONS_QUEUE, item.id, "w")
        queue.finish(EVALUATIONS_QUEUE, item.id, "w", {"score": RolloutScore(value).to_json()})

    async def queued_rollouts(count: int) -> list[WorkItem]:
        for _ in range(200):
            items = [item for item in queue.items(EVALUATIONS_QUEUE, scenario="s") if not item.final]
            if len(items) == count:
                return sorted(items, key=lambda item: item.id)
            await asyncio.sleep(0.05)
        raise AssertionError(f"the queue did not hold {count} rollouts that are not final")

    async def run() -> None:
        # No worker runs: the test takes and ends each rollout itself.
        server = TestServer(create_app(dispatcher, tokens=TOKEN, evaluations=ArtifactEvaluations(dispatcher)))
        async with TestClient(server) as client:
            artifact_id = await created(client, tmp_path, "add", "sort")
            body = {
                "artifact_id": artifact_id,
                "harness": {"name": "terminus-2"},
                "rollouts_per_task": 2,
                "max_concurrent_rollouts": 2,
            }
            response = await client.post("/reef/scenarios/s/evaluations", headers=AUTH, json=body)
            assert response.status == 202, await response.text()
            accepted = await response.json()
            assert accepted["request"]["task_names"] == ["add", "sort"]
            assert accepted["request"]["max_concurrent_rollouts"] == 2
            rollouts_route = f"/reef/scenarios/s/evaluations/{accepted['evaluation_id']}/rollouts"

            first, second = await queued_rollouts(2)
            listed = (await (await client.get(rollouts_route, headers=AUTH)).json())["rollouts"]
            assert [(row["task"], row["repeat"], row["state"]) for row in listed] == [
                ("add", 0, "queued"),
                ("add", 1, "queued"),
                ("sort", 0, "waiting"),
                ("sort", 1, "waiting"),
            ]
            assert queue.take(EVALUATIONS_QUEUE, "w", [EVALUATION_ROLLOUT_KIND]).id == first.id
            running = await client.get(f"{rollouts_route}?limit=2&state=running&calls=false", headers=AUTH)
            page = await running.json()
            assert (page["total"], [row["state"] for row in page["rollouts"]]) == (1, ["assigned"])
            scored(first, 1.0)
            still_queued, third = await queued_rollouts(2)
            assert still_queued.id == second.id and third.id.endswith(":00001:00000")
            assert queue.take(EVALUATIONS_QUEUE, "w", [EVALUATION_ROLLOUT_KIND]).id == second.id
            running = await client.get(f"{rollouts_route}?limit=2&state=running&calls=false", headers=AUTH)
            page = await running.json()
            assert len(page["rollouts"]) == 1 and page["rollouts"][0]["score"] is None
            assert page["task_means"] == {"0": 1.0}
            scored(second, 0.0)
            assert queue.take(EVALUATIONS_QUEUE, "w", [EVALUATION_ROLLOUT_KIND]).id == third.id
            scored(third, 0.0)
            [last] = await queued_rollouts(1)
            assert queue.take(EVALUATIONS_QUEUE, "w", [EVALUATION_ROLLOUT_KIND]).id == last.id
            scored(last, 1.0)

            answer = await finished(client, accepted["evaluation_id"])
            assert answer["state"] == "finished", answer
            assert answer["policy_evaluation"]["scores"] == [
                {"task": "add", "scores": [1.0, 0.0]},
                {"task": "sort", "scores": [0.0, 1.0]},
            ]
            assert "score_levels" not in answer["progress"]
            view = await client.get(f"{rollouts_route.removesuffix('/rollouts')}?score_levels=true", headers=AUTH)
            summary = (await view.json())["progress"]
            assert summary["score_levels"] == {"full": 2, "part": 0, "zero": 2, "failed": 0, "pending": 0}

    asyncio.run(run())


@pytest.mark.integration
def test_an_evaluation_on_e2b_adds_the_rollouts_of_each_task_whose_template_item_is_final(tmp_path: Path) -> None:
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(), live=LiveObjects(runtime=ArtifactRuntime()))
    queue = dispatcher.queue

    def scored(item: WorkItem, value: float) -> None:
        assert queue.take(EVALUATIONS_QUEUE, "w", [EVALUATION_ROLLOUT_KIND]).id == item.id
        queue.finish(EVALUATIONS_QUEUE, item.id, "w", {"score": RolloutScore(value).to_json()})

    async def open_items(kind: str, count: int) -> list[WorkItem]:
        for _ in range(200):
            items = [item for item in queue.items(EVALUATIONS_QUEUE, scenario="s", kind=kind) if not item.final]
            if len(items) == count:
                return sorted(items, key=lambda item: item.id)
            await asyncio.sleep(0.05)
        raise AssertionError(f"the queue did not hold {count} {kind} items that are not final")

    async def run() -> None:
        # No worker runs: the test takes and ends each item itself.
        server = TestServer(create_app(dispatcher, tokens=TOKEN, evaluations=ArtifactEvaluations(dispatcher)))
        async with TestClient(server) as client:
            artifact_id = await created(client, tmp_path, "add", "sort")
            body = {
                "artifact_id": artifact_id,
                "harness": {"name": "terminus-2"},
                "environment": "e2b",
                "rollouts_per_task": 2,
                "max_concurrent_rollouts": 2,
            }
            response = await client.post("/reef/scenarios/s/evaluations", headers=AUTH, json=body)
            assert response.status == 202, await response.text()
            evaluation_id = (await response.json())["evaluation_id"]
            route = f"/reef/scenarios/s/evaluations/{evaluation_id}"

            # Each task has a template item first, and no rollout runs while the templates build.
            add_template, sort_template = await open_items(TASK_TEMPLATE_KIND, 2)
            assert [item.payload["task_name"] for item in (add_template, sort_template)] == ["add", "sort"]
            assert queue.items(EVALUATIONS_QUEUE, scenario="s", kind=EVALUATION_ROLLOUT_KIND) == ()
            # A ready template of a later task starts its rollouts while the template of an earlier task builds.
            assert queue.take(EVALUATIONS_QUEUE, "w", [TASK_TEMPLATE_KIND]).id == add_template.id
            assert queue.take(EVALUATIONS_QUEUE, "w", [TASK_TEMPLATE_KIND]).id == sort_template.id
            queue.finish(EVALUATIONS_QUEUE, sort_template.id, "w", {})
            first, second = await open_items(EVALUATION_ROLLOUT_KIND, 2)
            assert [(item.payload["task_name"], item.payload["repeat"]) for item in (first, second)] == [
                ("sort", 0),
                ("sort", 1),
            ]
            listed = (await (await client.get(f"{route}/rollouts", headers=AUTH)).json())["rollouts"]
            assert [(row["task"], row["repeat"], row["state"]) for row in listed] == [
                ("add", 0, "waiting"),
                ("add", 1, "waiting"),
                ("sort", 0, "queued"),
                ("sort", 1, "queued"),
            ]
            # The rollouts of the earlier task start once its template is ready and the limit lets them.
            queue.finish(EVALUATIONS_QUEUE, add_template.id, "w", {})
            await asyncio.sleep(0.3)
            assert len(await open_items(EVALUATION_ROLLOUT_KIND, 2)) == 2
            for item in (first, second):
                scored(item, 0.0)
            for item in await open_items(EVALUATION_ROLLOUT_KIND, 2):
                scored(item, 1.0)

            answer = await finished(client, evaluation_id)
            assert answer["state"] == "finished", answer
            assert answer["policy_evaluation"]["scores"] == [
                {"task": "add", "scores": [1.0, 1.0]},
                {"task": "sort", "scores": [0.0, 0.0]},
            ]

    asyncio.run(run())


@pytest.mark.integration
def test_a_failed_template_item_still_adds_its_tasks_rollouts_and_a_cancel_ends_the_others(tmp_path: Path) -> None:
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(), live=LiveObjects(runtime=ArtifactRuntime()))
    queue = dispatcher.queue

    async def run() -> None:
        server = TestServer(create_app(dispatcher, tokens=TOKEN, evaluations=ArtifactEvaluations(dispatcher)))
        async with TestClient(server) as client:
            artifact_id = await created(client, tmp_path, "add", "sort")
            body = {"artifact_id": artifact_id, "harness": {"name": "terminus-2"}, "environment": "e2b"}
            response = await client.post("/reef/scenarios/s/evaluations", headers=AUTH, json=body)
            assert response.status == 202, await response.text()
            evaluation_id = (await response.json())["evaluation_id"]
            add_template = queue.take(EVALUATIONS_QUEUE, "w", [TASK_TEMPLATE_KIND])
            assert add_template.payload["task_name"] == "add"
            # The trial of a task whose build failed builds the template itself, so its failure is the rollout's.
            queue.fail(EVALUATIONS_QUEUE, add_template.id, "w", "aenv build of add failed", retryable=False)
            for _ in range(200):
                rollouts = queue.items(EVALUATIONS_QUEUE, scenario="s", kind=EVALUATION_ROLLOUT_KIND)
                if rollouts:
                    break
                await asyncio.sleep(0.05)
            assert [item.payload["task_name"] for item in rollouts] == ["add"]

            route = f"/reef/scenarios/s/evaluations/{evaluation_id}"
            assert (await client.post(f"{route}/cancel", headers=AUTH)).status == 200
            sort_template = queue.read(EVALUATIONS_QUEUE, template_item_id(evaluation_id, 1))
            assert (sort_template.state, sort_template.error) == ("failed", CANCELLED_ERROR)

            # A resume builds the cancelled template again; the failed build of "add" stays failed.
            assert (await client.post(f"{route}/resume", headers=AUTH)).status == 202
            assert queue.read(EVALUATIONS_QUEUE, template_item_id(evaluation_id, 1)).state == "queued"
            assert queue.read(EVALUATIONS_QUEUE, add_template.id).state == "failed"

    asyncio.run(run())


@pytest.mark.integration
@pytest.mark.parametrize("rollout_kept", [True, False])
def test_a_resume_of_an_e2b_evaluation_needs_the_rollouts_of_the_tasks_before_its_first_template_item(
    tmp_path: Path, rollout_kept: bool
) -> None:
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(), live=LiveObjects(runtime=ArtifactRuntime()))
    queue = dispatcher.queue

    async def run() -> None:
        server = TestServer(create_app(dispatcher, tokens=TOKEN, evaluations=ArtifactEvaluations(dispatcher)))
        async with TestClient(server) as client:
            artifact_id = await created(client, tmp_path, "add", "sort")
            body = {"artifact_id": artifact_id, "harness": {"name": "terminus-2"}, "environment": "e2b"}
            response = await client.post("/reef/scenarios/s/evaluations", headers=AUTH, json=body)
            assert response.status == 202, await response.text()
            evaluation_id = (await response.json())["evaluation_id"]
            route = f"/reef/scenarios/s/evaluations/{evaluation_id}"
            add_template = queue.take(EVALUATIONS_QUEUE, "w", [TASK_TEMPLATE_KIND])
            queue.finish(EVALUATIONS_QUEUE, add_template.id, "w", {})
            for _ in range(200):
                if queue.items(EVALUATIONS_QUEUE, scenario="s", kind=EVALUATION_ROLLOUT_KIND):
                    break
                await asyncio.sleep(0.05)
            assert (await client.post(f"{route}/cancel", headers=AUTH)).status == 200
            # The template item of "add" goes, as after its retention or in an evaluation that ran "add" before Reef
            # had template items. Its rollout goes too when the retention deleted it.
            deleted = {add_template.id} if rollout_kept else {add_template.id, rollout_item_id(evaluation_id, 0, 0)}
            kept = [item.id for item in queue.items(EVALUATIONS_QUEUE, scenario="s") if item.id not in deleted]
            queue.delete_final(finished_before=time.time() + 1, keep={EVALUATIONS_QUEUE: kept})
            resumed = await client.post(f"{route}/resume", headers=AUTH)
            if rollout_kept:
                assert resumed.status == 202, await resumed.text()
                assert queue.read(EVALUATIONS_QUEUE, template_item_id(evaluation_id, 1)).state == "queued"
            else:
                assert resumed.status == 409, await resumed.text()

    asyncio.run(run())


@pytest.mark.integration
def test_an_evaluation_on_e2b_builds_the_templates_of_the_next_tasks_ahead(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("reef.control.operations.artifact_evaluations.TEMPLATE_TASKS_AHEAD", 1)
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(), live=LiveObjects(runtime=ArtifactRuntime()))
    queue = dispatcher.queue

    async def run() -> None:
        server = TestServer(create_app(dispatcher, tokens=TOKEN, evaluations=ArtifactEvaluations(dispatcher)))
        async with TestClient(server) as client:
            artifact_id = await created(client, tmp_path, "add", "sort")
            body = {"artifact_id": artifact_id, "harness": {"name": "terminus-2"}, "environment": "e2b"}
            response = await client.post("/reef/scenarios/s/evaluations", headers=AUTH, json=body)
            assert response.status == 202, await response.text()
            evaluation_id = (await response.json())["evaluation_id"]
            [add_template] = queue.items(EVALUATIONS_QUEUE, scenario="s", kind=TASK_TEMPLATE_KIND)
            assert queue.take(EVALUATIONS_QUEUE, "w", [TASK_TEMPLATE_KIND]).id == add_template.id
            queue.finish(EVALUATIONS_QUEUE, add_template.id, "w", {})
            # The rollout of "add" is added, and the window moves on to the template of "sort".
            for _ in range(200):
                if queue.read(EVALUATIONS_QUEUE, template_item_id(evaluation_id, 1)) is not None:
                    break
                await asyncio.sleep(0.05)
            [rollout] = queue.items(EVALUATIONS_QUEUE, scenario="s", kind=EVALUATION_ROLLOUT_KIND)
            assert rollout.payload["task_name"] == "add"
            assert queue.read(EVALUATIONS_QUEUE, template_item_id(evaluation_id, 1)).state == "queued"

    asyncio.run(run())


class OwnerTemplates(TaskTemplateBuilder):
    """A builder that finds a task's image by owner and task name, as a registry builder does: it keeps
    ``(owner, scenario, task_name)`` of each build in ``builds`` and builds nothing."""

    def __init__(self) -> None:
        self.builds: list[tuple[str | None, str, str]] = []

    async def build(self, scenario: str, task_name: str, task_directory: Path) -> None:
        raise AssertionError("the worker calls build_for_owner")

    async def build_for_owner(self, owner: str | None, scenario: str, task_name: str, task_directory: Path) -> None:
        assert (task_directory / "task.toml").is_file()
        self.builds.append((owner, scenario, task_name))


@pytest.mark.integration
def test_the_template_worker_builds_for_the_owner_of_the_scenarios_record_not_its_name(tmp_path: Path) -> None:
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(), live=LiveObjects(runtime=ArtifactRuntime()))
    queue = dispatcher.queue
    builder = OwnerTemplates()
    dispatcher.workers.add(
        TaskTemplateWorker(queue, LocalCandidateEndpoints(dispatcher), builder, worker_id="evaluation-templates")
    )

    async def run() -> None:
        server = TestServer(create_app(dispatcher, tokens=TOKEN, evaluations=ArtifactEvaluations(dispatcher)))
        async with TestClient(server) as client:
            # The scenario's name says nothing of the owner whose data holds its tasks.
            artifact_id = await created(client, tmp_path, "add", owner="workspace-7")
            body = {"artifact_id": artifact_id, "harness": {"name": "terminus-2"}, "environment": "e2b"}
            response = await client.post("/reef/scenarios/s/evaluations", headers=AUTH, json=body)
            assert response.status == 202, await response.text()
            evaluation_id = (await response.json())["evaluation_id"]
            for _ in range(200):
                if queue.items(EVALUATIONS_QUEUE, scenario="s", kind=EVALUATION_ROLLOUT_KIND):
                    break
                await asyncio.sleep(0.05)
            template = queue.read(EVALUATIONS_QUEUE, template_item_id(evaluation_id, 0))
            assert (template.state, template.payload["owner"]) == ("finished", "workspace-7")
            assert builder.builds == [("workspace-7", "s", "add")]

    asyncio.run(run())


class EnvironmentCallingLab(ModelCallingLab):
    """A ``ModelCallingLab`` that keeps the task name and the Harbor environment config of each trial in ``trials``."""

    def __init__(self, rewards: dict[str, float]) -> None:
        super().__init__(rewards)
        self.trials: list[tuple[str, Mapping[str, Any]]] = []

    async def run(self, task_path, agent, *, key, tags, overrides) -> RolloutRow:
        self.trials.append((task_path.name, overrides["environment"]))
        return await super().run(task_path, agent, key=key, tags=tags, overrides=overrides)


@pytest.mark.integration
def test_with_prepared_templates_required_a_failed_template_runs_no_trial_and_a_retry_builds_it_first(
    tmp_path: Path,
) -> None:
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(), live=LiveObjects(runtime=ArtifactRuntime()))
    queue = dispatcher.queue
    lab = EnvironmentCallingLab({"add": 1.0, "sort": 0.5})
    unprepared = "the templates of task 'add' were not prepared: the registry holds no image of task add"

    async def rollouts(client: TestClient, route: str) -> list[dict[str, Any]]:
        return (await (await client.get(f"{route}/rollouts", headers=AUTH)).json())["rollouts"]

    async def run() -> None:
        # No template worker runs: the test takes and ends each template item itself.
        server, _ = evaluating_server(dispatcher, lab, requires_prepared_templates=True)
        async with TestClient(server) as client:
            artifact_id = await created(client, tmp_path, "add", "sort")
            body = {
                "artifact_id": artifact_id,
                "harness": {"name": "terminus-2"},
                "environment": "e2b",
                "rollouts_per_task": 2,
            }
            response = await client.post("/reef/scenarios/s/evaluations", headers=AUTH, json=body)
            assert response.status == 202, await response.text()
            evaluation_id = (await response.json())["evaluation_id"]
            route = f"/reef/scenarios/s/evaluations/{evaluation_id}"
            add_template = queue.take(EVALUATIONS_QUEUE, "builder", [TASK_TEMPLATE_KIND])
            sort_template = queue.take(EVALUATIONS_QUEUE, "builder", [TASK_TEMPLATE_KIND])
            assert [add_template.payload["task_name"], sort_template.payload["task_name"]] == ["add", "sort"]
            error = "the registry holds no image of task add"
            queue.fail(EVALUATIONS_QUEUE, add_template.id, "builder", error, retryable=False)
            queue.finish(EVALUATIONS_QUEUE, sort_template.id, "builder", {})

            answer = await finished(client, evaluation_id)
            assert answer["state"] == "finished", answer
            assert answer["policy_evaluation"]["scores"] == [
                {"task": "add", "scores": ["failed", "failed"]},
                {"task": "sort", "scores": [0.5, 0.5]},
            ]
            progress = answer["progress"]
            assert (progress["ended"], progress["scored"], progress["failed"]) == (4, 2, 2)
            # Each rollout of "add" reports the template's error and ran no trial, so nothing built its template.
            # Each trial of "sort" tells Reef's e2b class to build no template.
            add_rollouts = [view for view in await rollouts(client, route) if view["task"] == "add"]
            assert [(view["state"], view["attempt_count"], view["error"]) for view in add_rollouts] == [
                ("failed", 1, unprepared)
            ] * 2
            assert [task for task, _ in lab.trials] == ["sort", "sort"]
            assert all(environment["kwargs"]["builds_templates"] is False for _, environment in lab.trials)

            # A retry builds the template of "add" again; its rollouts wait until that build ends.
            assert (await client.post(f"{route}/retry", headers=AUTH)).status == 202
            assert queue.read(EVALUATIONS_QUEUE, add_template.id).state == "queued"
            add_rollouts = [view for view in await rollouts(client, route) if view["task"] == "add"]
            assert [view["state"] for view in add_rollouts] == ["waiting", "waiting"]
            await asyncio.sleep(0.3)
            assert len(lab.trials) == 2
            assert queue.take(EVALUATIONS_QUEUE, "builder", [TASK_TEMPLATE_KIND]).id == add_template.id
            queue.finish(EVALUATIONS_QUEUE, add_template.id, "builder", {})

            answer = await finished(client, evaluation_id)
            assert answer["state"] == "finished", answer
            assert answer["policy_evaluation"]["scores"] == [
                {"task": "add", "scores": [1.0, 1.0]},
                {"task": "sort", "scores": [0.5, 0.5]},
            ]
            assert sorted(task for task, _ in lab.trials) == ["add", "add", "sort", "sort"]
            assert all(environment["kwargs"]["builds_templates"] is False for _, environment in lab.trials)

    asyncio.run(run())


@pytest.mark.integration
def test_with_prepared_templates_required_a_retry_that_waits_for_its_template_holds_back_no_ready_task(
    tmp_path: Path,
) -> None:
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(), live=LiveObjects(runtime=ArtifactRuntime()))
    queue = dispatcher.queue

    async def rollout_of(task: str) -> WorkItem:
        for _ in range(200):
            found = [
                item
                for item in queue.items(EVALUATIONS_QUEUE, scenario="s", kind=EVALUATION_ROLLOUT_KIND)
                if item.payload["task_name"] == task
            ]
            if found:
                return found[0]
            await asyncio.sleep(0.05)
        raise AssertionError(f"the queue holds no rollout of {task!r}")

    async def states(client: TestClient, route: str) -> list[tuple[str, str]]:
        listed = (await (await client.get(f"{route}/rollouts", headers=AUTH)).json())["rollouts"]
        return [(row["task"], row["state"]) for row in listed]

    async def run() -> None:
        # No worker runs: the test takes and ends each item itself.
        evaluations = ArtifactEvaluations(dispatcher, requires_prepared_templates=True)
        async with TestClient(TestServer(create_app(dispatcher, tokens=TOKEN, evaluations=evaluations))) as client:
            artifact_id = await created(client, tmp_path, "add", "sort")
            body = {
                "artifact_id": artifact_id,
                "harness": {"name": "terminus-2"},
                "environment": "e2b",
                "max_concurrent_rollouts": 1,
            }
            response = await client.post("/reef/scenarios/s/evaluations", headers=AUTH, json=body)
            assert response.status == 202, await response.text()
            evaluation_id = (await response.json())["evaluation_id"]
            route = f"/reef/scenarios/s/evaluations/{evaluation_id}"
            add_template = queue.take(EVALUATIONS_QUEUE, "w", [TASK_TEMPLATE_KIND])
            sort_template = queue.take(EVALUATIONS_QUEUE, "w", [TASK_TEMPLATE_KIND])
            assert [add_template.payload["task_name"], sort_template.payload["task_name"]] == ["add", "sort"]
            error = "the registry holds no image of task add"
            queue.fail(EVALUATIONS_QUEUE, add_template.id, "w", error, retryable=False)
            # The worker fails the rollout of a task whose templates were not prepared, as ``score`` does.
            add = await rollout_of("add")
            assert queue.take(EVALUATIONS_QUEUE, "w", [EVALUATION_ROLLOUT_KIND]).id == add.id
            unprepared = f"the templates of task 'add' were not prepared: {error}"
            queue.fail(EVALUATIONS_QUEUE, add.id, "w", unprepared, retryable=False)

            # The retry builds the template of "add" again, and its rollout waits for that build.
            assert (await client.post(f"{route}/retry", headers=AUTH)).status == 202
            assert queue.read(EVALUATIONS_QUEUE, add_template.id).state == "queued"
            # The template of "sort" is ready meanwhile: its rollout takes the one place, which the waiting retry
            # does not keep.
            queue.finish(EVALUATIONS_QUEUE, sort_template.id, "w", {})
            sort = await rollout_of("sort")
            assert await states(client, route) == [("add", "waiting"), ("sort", "queued")]

            # Once the template of "add" is ready, its retry waits for the place, and then runs.
            assert queue.take(EVALUATIONS_QUEUE, "w", [TASK_TEMPLATE_KIND]).id == add_template.id
            queue.finish(EVALUATIONS_QUEUE, add_template.id, "w", {})
            await asyncio.sleep(0.3)
            assert await states(client, route) == [("add", "waiting"), ("sort", "queued")]
            assert queue.take(EVALUATIONS_QUEUE, "w", [EVALUATION_ROLLOUT_KIND]).id == sort.id
            queue.finish(EVALUATIONS_QUEUE, sort.id, "w", {"score": RolloutScore(0.5).to_json()})
            for _ in range(200):
                retried = queue.take(EVALUATIONS_QUEUE, "w", [EVALUATION_ROLLOUT_KIND])
                if retried is not None:
                    break
                await asyncio.sleep(0.05)
            assert retried is not None and retried.id == add.id
            queue.finish(EVALUATIONS_QUEUE, add.id, "w", {"score": RolloutScore(1.0).to_json()})

            answer = await finished(client, evaluation_id)
            assert answer["state"] == "finished", answer
            assert answer["policy_evaluation"]["scores"] == [
                {"task": "add", "scores": [1.0]},
                {"task": "sort", "scores": [0.5]},
            ]

    asyncio.run(run())


def test_an_evaluation_on_a_collection_runs_its_eval_tasks_and_scores_each_benchmark(tmp_path: Path) -> None:
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(), live=LiveObjects(runtime=ArtifactRuntime()))
    queue = dispatcher.queue

    async def run() -> None:
        # No worker runs: the test takes and ends each rollout itself.
        server = TestServer(create_app(dispatcher, tokens=TOKEN, evaluations=ArtifactEvaluations(dispatcher)))
        async with TestClient(server) as client:
            artifact_id = await created(client, tmp_path, "add", "sort", "sum")
            for item_id, benchmarks in (("add", ["lite", "full"]), ("sort", ["full"])):
                patch = {"ids": [item_id], "patch": {"benchmarks": benchmarks}}
                assert (await client.post("/reef/namespaces/s/data/metadata", headers=AUTH, json=patch)).status == 200
            # The benchmark "other" has no task in the eval split, so the evaluation leaves it out.
            collection = {"benchmarks": ["lite", "full", "other"], "average": "benchmark"}
            stored = "/reef/namespaces/s/collections/coding"
            assert (await client.put(stored, headers=AUTH, json=collection)).status == 200
            body = {
                "artifact_id": artifact_id,
                "harness": {"name": "terminus-2"},
                "collection": "coding",
                "rollouts_per_task": 2,
            }
            for refused in ({**body, "task_names": ["add"]}, {**body, "collection": "missing"}):
                response = await client.post("/reef/scenarios/s/evaluations", headers=AUTH, json=refused)
                assert response.status == 400, await response.text()
            response = await client.post("/reef/scenarios/s/evaluations", headers=AUTH, json=body)
            assert response.status == 202, await response.text()
            accepted = await response.json()
            assert accepted["request"]["task_names"] == ["add", "sort"]
            assert accepted["request"]["collection"] == {
                "name": "coding",
                "average": "benchmark",
                "benchmarks": [
                    {"benchmark": "lite", "task_names": ["add"]},
                    {"benchmark": "full", "task_names": ["add", "sort"]},
                ],
            }
            route = f"/reef/scenarios/s/evaluations/{accepted['evaluation_id']}"
            # The first repeat of "add" has ended. The other repeats have not ended.
            item = queue.take(EVALUATIONS_QUEUE, "w", [EVALUATION_ROLLOUT_KIND])
            queue.finish(EVALUATIONS_QUEUE, item.id, "w", {"score": RolloutScore(1.0).to_json()})
            running = (await (await client.get(route, headers=AUTH)).json())["progress"]
            ended = {"ended": 1, "scored": 1, "failed": 0, "cancelled": 0, "running": 0, "mean_score": 1.0}
            assert running["benchmarks"] == [
                {"benchmark": "lite", "total": 2, **ended},
                {"benchmark": "full", "total": 4, **ended},
            ]
            for _ in range(3):
                item = queue.take(EVALUATIONS_QUEUE, "w", [EVALUATION_ROLLOUT_KIND])
                queue.finish(EVALUATIONS_QUEUE, item.id, "w", {"score": RolloutScore(0.0).to_json()})
            answer = await finished(client, accepted["evaluation_id"])
            assert answer["state"] == "finished", answer
            assert [entry["mean_score"] for entry in answer["progress"]["benchmarks"]] == [0.5, 0.25]
            assert answer["benchmark_scores"] == [
                {"benchmark": "lite", "task_count": 1, "mean_score": 0.5},
                {"benchmark": "full", "task_count": 2, "mean_score": 0.25},
            ]
            assert answer["mean_score"] == 0.375

            response = await client.get(f"{route}/rollouts?limit=1&benchmark=lite&calls=false", headers=AUTH)
            assert response.status == 200
            page = await response.json()
            assert page["total"] == 2
            [rollout] = page["rollouts"]
            assert (rollout["task"], rollout["repeat"], rollout["score"]) == ("add", 0, 1.0)
            assert page["task_means"] == {"0": 0.5}

    asyncio.run(run())


class RolloutMeanActivation(EvaluatedActivation):
    """A deployment plugin that inherits the legacy rollout-mean rule."""


@pytest.mark.parametrize(
    ("average", "scores", "threshold", "rule_type", "expected_score", "activates"),
    [
        ("benchmark", (0.0, 1.0, 1.0), 0.6, EvaluatedActivation, 0.5, False),
        ("benchmark", (1.0, 0.0, 0.0), 0.4, EvaluatedActivation, 0.5, True),
        ("task", (0.0, 1.0, 1.0), 0.6, EvaluatedActivation, 2 / 3, True),
        ("benchmark", (None, 1.0, 1.0), 0.6, EvaluatedActivation, 1.0, True),
        ("benchmark", (None, None, None), 0.6, EvaluatedActivation, None, False),
        ("benchmark", (0.0, 1.0, 1.0), 0.6, RolloutMeanActivation, 0.5, True),
    ],
)
def test_collection_average_controls_progress_and_activation(
    tmp_path: Path,
    average: str,
    scores: tuple[float | None, ...],
    threshold: float,
    rule_type: type[EvaluatedActivation],
    expected_score: float | None,
    activates: bool,
) -> None:
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(), live=LiveObjects(runtime=ArtifactRuntime()))
    queue = dispatcher.queue

    async def run() -> None:
        evaluations = ArtifactEvaluations(dispatcher)
        activated: list[str] = []
        evaluations.results.activate = lambda scenario, request: activated.append(request.artifact_id)
        async with TestClient(TestServer(create_app(dispatcher, tokens=TOKEN, evaluations=evaluations))) as client:
            artifact_id = await created(client, tmp_path, "add", "sort", "sum", "wait")
            for task, benchmark in (("add", "lite"), ("sort", "full"), ("sum", "full"), ("wait", "lite")):
                patch = {"ids": [task], "patch": {"benchmarks": [benchmark]}}
                assert (await client.post("/reef/namespaces/s/data/metadata", headers=AUTH, json=patch)).status == 200
            collection = {"benchmarks": ["lite", "full"], "average": average}
            route = "/reef/namespaces/s/collections/coding"
            assert (await client.put(route, headers=AUTH, json=collection)).status == 200
            with dispatcher.locked_scenario("s") as scenario:
                scenario.set_activation_rule(rule_type(threshold))
            body = {"artifact_id": artifact_id, "harness": {"name": "terminus-2"}, "collection": "coding"}
            response = await client.post("/reef/scenarios/s/evaluations", headers=AUTH, json=body)
            assert response.status == 202, await response.text()
            evaluation_id = (await response.json())["evaluation_id"]
            for task, score in zip(("add", "sort", "sum"), scores, strict=True):
                item = queue.take(EVALUATIONS_QUEUE, "w", [EVALUATION_ROLLOUT_KIND])
                assert item.payload["task_name"] == task
                queue.finish(EVALUATIONS_QUEUE, item.id, "w", {"score": RolloutScore(score).to_json()})
            route = f"/reef/scenarios/s/evaluations/{evaluation_id}"
            running = await (await client.get(route, headers=AUTH)).json()
            assert running["state"] in ("queued", "running")
            item = queue.take(EVALUATIONS_QUEUE, "w", [EVALUATION_ROLLOUT_KIND])
            queue.finish(EVALUATIONS_QUEUE, item.id, "w", {"score": RolloutScore(None).to_json()})
            answer = await finished(client, evaluation_id)
            assert answer["state"] == "finished"
            assert (running["progress"]["mean_score"], answer["progress"]["mean_score"], activated) == (
                expected_score,
                expected_score,
                [artifact_id] if activates else [],
            )
            assert answer["mean_score"] == expected_score

    asyncio.run(run())


def test_an_evaluation_with_num_samples_runs_that_many_tasks_chosen_by_their_names(tmp_path: Path) -> None:
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(), live=LiveObjects(runtime=ArtifactRuntime()))

    async def run() -> None:
        # No worker runs: the test reads the accepted requests only.
        server = TestServer(create_app(dispatcher, tokens=TOKEN, evaluations=ArtifactEvaluations(dispatcher)))
        async with TestClient(server) as client:
            artifact_id = await created(client, tmp_path, "add", "sort", "sum")
            body = {"artifact_id": artifact_id, "harness": {"name": "terminus-2"}, "rollouts_per_task": 2}

            async def _accepted(settings: dict[str, object]) -> dict:
                response = await client.post("/reef/scenarios/s/evaluations", headers=AUTH, json={**body, **settings})
                assert response.status == 202, await response.text()
                return await response.json()

            for refused in (0, -1, 1.5, True, "2"):
                response = await client.post(
                    "/reef/scenarios/s/evaluations", headers=AUTH, json={**body, "num_samples": refused}
                )
                assert response.status == 400, await response.text()
            # By the SHA-256 digests of their names, the tasks rank sum, sort, add. The request keeps the split's order.
            two = await _accepted({"num_samples": 2})
            assert two["request"]["task_names"] == ["sort", "sum"]
            rollouts = await (
                await client.get(f"/reef/scenarios/s/evaluations/{two['evaluation_id']}/rollouts", headers=AUTH)
            ).json()
            assert [(view["task"], view["repeat"]) for view in rollouts["rollouts"]] == [
                ("sort", 0),
                ("sort", 1),
                ("sum", 0),
                ("sum", 1),
            ]
            assert (await _accepted({"num_samples": 2}))["request"]["task_names"] == ["sort", "sum"]
            assert (await _accepted({"num_samples": 1}))["request"]["task_names"] == ["sum"]
            assert (await _accepted({"num_samples": 5}))["request"]["task_names"] == ["add", "sort", "sum"]
            named = await _accepted({"num_samples": 1, "task_names": ["add", "sort"]})
            assert named["request"]["task_names"] == ["sort"]

            # A seed changes the choice, and the same seed gives the same choice again.
            response = await client.post(
                "/reef/scenarios/s/evaluations", headers=AUTH, json={**body, "num_samples": 1, "sample_seed": "7"}
            )
            assert response.status == 400, await response.text()
            seeded = [
                (await _accepted({"num_samples": 1, "sample_seed": seed}))["request"]["task_names"]
                for seed in (0, 0, 1, 2, 3, 4, 5, 6, 7)
            ]
            assert seeded[0] == seeded[1]
            assert {name for names in seeded for name in names} == {"add", "sort", "sum"}
            # A seed alone runs every task, in the random order of the seed: the choice with num_samples is its start.
            orders = [
                (await _accepted({"sample_seed": seed}))["request"]["task_names"]
                for seed in (0, 0, 1, 2, 3, 4, 5, 6, 7)
            ]
            assert orders[0] == orders[1]
            assert all(sorted(order) == ["add", "sort", "sum"] for order in orders)
            assert [order[:1] for order in orders] == seeded
            assert len({tuple(order) for order in orders}) > 1

            # A task-average collection can omit a benchmark with no chosen task.
            for item_id, benchmarks in (
                ("add", ["lite", "shared"]),
                ("sort", ["full", "shared"]),
                ("sum", ["full", "shared"]),
            ):
                patch = {"ids": [item_id], "patch": {"benchmarks": benchmarks}}
                assert (await client.post("/reef/namespaces/s/data/metadata", headers=AUTH, json=patch)).status == 200
            stored = "/reef/namespaces/s/collections/coding"
            assert (await client.put(stored, headers=AUTH, json={"benchmarks": ["lite", "full"]})).status == 200
            sampled = await _accepted({"num_samples": 1, "collection": "coding"})
            assert sampled["request"]["task_names"] == ["sum"]
            assert sampled["request"]["collection"]["benchmarks"] == [{"benchmark": "full", "task_names": ["sum"]}]

            assert (
                await client.put(stored, headers=AUTH, json={"benchmarks": ["lite", "full"], "average": "benchmark"})
            ).status == 200
            covered = await _accepted({"num_samples": 2, "collection": "coding"})
            assert covered["request"]["task_names"] == ["add", "sum"]
            assert covered["request"]["collection"]["benchmarks"] == [
                {"benchmark": "lite", "task_names": ["add"]},
                {"benchmark": "full", "task_names": ["sum"]},
            ]
            seeded_coverage = await _accepted({"num_samples": 2, "sample_seed": 1, "collection": "coding"})
            assert seeded_coverage["request"]["task_names"] == ["sort", "add"]
            response = await client.post(
                "/reef/scenarios/s/evaluations",
                headers=AUTH,
                json={**body, "num_samples": 1, "collection": "coding"},
            )
            assert response.status == 400, await response.text()
            assert "num_samples" in await response.text()

            assert (
                await client.put(stored, headers=AUTH, json={"benchmarks": ["lite", "shared"], "average": "benchmark"})
            ).status == 200
            shared = await _accepted({"num_samples": 1, "collection": "coding"})
            assert shared["request"]["task_names"] == ["add"]
            assert shared["request"]["collection"]["benchmarks"] == [
                {"benchmark": "lite", "task_names": ["add"]},
                {"benchmark": "shared", "task_names": ["add"]},
            ]
            filled = await _accepted({"num_samples": 2, "collection": "coding"})
            assert filled["request"]["task_names"] == ["add", "sum"]
            assert filled["request"]["collection"]["benchmarks"] == [
                {"benchmark": "lite", "task_names": ["add"]},
                {"benchmark": "shared", "task_names": ["add", "sum"]},
            ]

    asyncio.run(run())


def test_an_evaluation_with_a_harbor_agent_starts_without_reading_a_task_file(tmp_path: Path, monkeypatch) -> None:
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(), live=LiveObjects(runtime=ArtifactRuntime()))

    def refuse(item) -> Path:
        raise AssertionError(f"the start read the files of {item.item_id}")

    async def run() -> None:
        # No worker runs, so only the start can read a task file.
        server = TestServer(create_app(dispatcher, tokens=TOKEN, evaluations=ArtifactEvaluations(dispatcher)))
        async with TestClient(server) as client:
            artifact_id = await created(client, tmp_path, "add", "sort")
            monkeypatch.setattr(DataAccess.of(dispatcher).require_store(), "task_path", refuse)
            body = {
                "artifact_id": artifact_id,
                "harness": {"name": "terminus-2"},
                "task_names": ["sort"],
                "log_probs": True,
            }
            response = await client.post("/reef/scenarios/s/evaluations", headers=AUTH, json=body)
            assert response.status == 202, await response.text()
            request = (await response.json())["request"]
            assert (request["task_names"], request["log_probs"]) == (["sort"], True)
            [rollout] = dispatcher.queue.items(EVALUATIONS_QUEUE, scenario="s")
            assert EvaluationRollout.from_item(rollout).log_probs is True

    asyncio.run(run())


def test_a_cancelled_evaluation_fails_with_its_rollouts_and_starts_no_waiting_rollout(tmp_path: Path) -> None:
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(), live=LiveObjects(runtime=ArtifactRuntime()))
    queue = dispatcher.queue

    async def run() -> None:
        # No worker runs: the test takes the one rollout in the queue, and three rollouts wait.
        server = TestServer(create_app(dispatcher, tokens=TOKEN, evaluations=ArtifactEvaluations(dispatcher)))
        async with TestClient(server) as client:
            artifact_id = await created(client, tmp_path, "add", "sort")
            body = {
                "artifact_id": artifact_id,
                "harness": {"name": "terminus-2"},
                "rollouts_per_task": 2,
                "max_concurrent_rollouts": 1,
            }
            accepted = await (await client.post("/reef/scenarios/s/evaluations", headers=AUTH, json=body)).json()
            route = f"/reef/scenarios/s/evaluations/{accepted['evaluation_id']}"
            taken = queue.take(EVALUATIONS_QUEUE, "w", [EVALUATION_ROLLOUT_KIND])
            queue.start(EVALUATIONS_QUEUE, taken.id, "w")

            answer = await (await client.post(f"{route}/cancel", headers=AUTH)).json()
            assert (answer["state"], answer["error"]) == ("failed", CANCELLED_ERROR)
            assert [(item.state, item.error) for item in queue.items(EVALUATIONS_QUEUE, scenario="s")] == [
                ("failed", CANCELLED_ERROR)
            ]
            listed = (await (await client.get(f"{route}/rollouts", headers=AUTH)).json())["rollouts"]
            assert [row["state"] for row in listed] == ["failed", "waiting", "waiting", "waiting"]
            # A second cancel changes nothing; an unknown evaluation is 404.
            again = await (await client.post(f"{route}/cancel", headers=AUTH)).json()
            assert again == answer
            unknown = await client.post("/reef/scenarios/s/evaluations/unknown/cancel", headers=AUTH)
            assert unknown.status == 404

    asyncio.run(run())


@pytest.mark.parametrize("pin_task_files", [False, True])
def test_a_retry_after_a_task_replace_uses_the_configured_task_version(tmp_path: Path, pin_task_files: bool) -> None:
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(), live=LiveObjects(runtime=ArtifactRuntime()))
    queue = dispatcher.queue
    endpoints = LocalCandidateEndpoints(dispatcher)

    async def run() -> None:
        server = TestServer(
            create_app(
                dispatcher, tokens=TOKEN, evaluations=ArtifactEvaluations(dispatcher, pin_task_files=pin_task_files)
            )
        )
        async with TestClient(server) as client:
            artifact_id = await created(client, tmp_path, "add")
            body = {"artifact_id": artifact_id, "harness": {"name": "terminus-2"}}
            accepted = await (await client.post("/reef/scenarios/s/evaluations", headers=AUTH, json=body)).json()
            route = f"/reef/scenarios/s/evaluations/{accepted['evaluation_id']}"
            item = queue.take(EVALUATIONS_QUEUE, "w", [EVALUATION_ROLLOUT_KIND])
            first, first_digest = endpoints.task_version(EvaluationRollout.from_item(item))
            queue.fail(
                EVALUATIONS_QUEUE, item.id, "w", "the task's compose file reads an unset variable", retryable=False
            )

            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
                tar.add(write_harbor_task(task_for("add", "record-2", 9000), tmp_path / "fixed"), arcname="add")
            gzip = {**AUTH, "Content-Type": "application/gzip"}
            put = await client.put("/reef/namespaces/s/data/tasks/add", headers=gzip, data=buffer.getvalue())
            assert put.status == 200, await put.text()
            assert (await client.post(f"{route}/retry", headers=AUTH)).status == 202

            item = queue.take(EVALUATIONS_QUEUE, "w", [EVALUATION_ROLLOUT_KIND])
            second, second_digest = endpoints.task_version(EvaluationRollout.from_item(item))
            assert item.attempt_count == 2 and first.is_dir()
            assert (second == first) is pin_task_files
            assert (second_digest == first_digest) is pin_task_files

            new = await client.post("/reef/scenarios/s/evaluations", headers=AUTH, json=body)
            assert new.status == 202, await new.text()
            next_item = queue.take(EVALUATIONS_QUEUE, "new", [EVALUATION_ROLLOUT_KIND])
            assert next_item is not None
            latest, latest_digest = endpoints.task_version(EvaluationRollout.from_item(next_item))
            assert latest != first
            assert latest_digest == files_digest(latest) == dispatcher.data_store.read("s", "add").digest

    asyncio.run(run())


@pytest.mark.integration
@pytest.mark.parametrize("environment", ["docker", "e2b"])
def test_a_deferred_evaluation_task_uses_the_files_from_its_keyed_start(tmp_path: Path, environment: str) -> None:
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(), live=LiveObjects(runtime=ArtifactRuntime()))
    queue = dispatcher.queue
    endpoints = LocalCandidateEndpoints(dispatcher)
    service = ArtifactEvaluations(dispatcher, pin_task_files=True)
    route = "/reef/scenarios/s/evaluations"

    async def take(kind: str, task_name: str) -> WorkItem:
        for _ in range(200):
            item = queue.take(EVALUATIONS_QUEUE, "w", [kind])
            if item is not None:
                assert item.payload["task_name"] == task_name
                return item
            await asyncio.sleep(0.05)
        raise AssertionError(f"the queue did not hold {kind} for {task_name}")

    async def run() -> None:
        async with TestClient(TestServer(create_app(dispatcher, tokens=TOKEN, evaluations=service))) as client:
            artifact_id = await created(client, tmp_path, "add", "sort")
            original = dispatcher.data_store.read("s", "sort")
            assert original is not None
            original_path = dispatcher.data_store.task_path(original)
            body = {
                "artifact_id": artifact_id,
                "harness": {"name": "terminus-2"},
                "environment": environment,
                "max_concurrent_rollouts": 1,
                "start_id": "stable-tasks",
            }
            accepted = await client.post(route, headers=AUTH, json=body)
            assert accepted.status == 202, await accepted.text()
            evaluation = await accepted.json()
            assert "task_digests" not in evaluation["request"]
            detail = await client.get(f"{route}/{evaluation['evaluation_id']}", headers=AUTH)
            assert detail.status == 200
            assert "task_digests" not in (await detail.json())["request"]
            listing = await client.get(route, headers=AUTH)
            assert listing.status == 200
            [listed] = (await listing.json())["evaluations"]
            assert "task_digests" not in listed["request"]

            replacement = write_harbor_task(task_for("sort", "record-2", 9000), tmp_path / "changed")
            changed = dispatcher.data_store.replace_task("s", "sort", replacement)
            assert changed.digest != original.digest
            replay = await client.post(route, headers=AUTH, json=body)
            assert replay.status == 202 and (await replay.json())["evaluation_id"] == evaluation["evaluation_id"]

            if environment == "e2b":
                assert dispatcher.data_store.delete("s", "sort")
                assert dispatcher.data_store.purge("s", "sort")
                first_template = await take(TASK_TEMPLATE_KIND, "add")
                queue.finish(EVALUATIONS_QUEUE, first_template.id, "w", {})
            first = await take(EVALUATION_ROLLOUT_KIND, "add")
            queue.finish(EVALUATIONS_QUEUE, first.id, "w", {"score": 1.0})

            if environment == "e2b":
                template = await take(TASK_TEMPLATE_KIND, "sort")
                assert endpoints.task_directory(EvaluationRollout.from_item(template)) == original_path
                queue.finish(EVALUATIONS_QUEUE, template.id, "w", {})
            deferred = await take(EVALUATION_ROLLOUT_KIND, "sort")
            assert endpoints.task_directory(EvaluationRollout.from_item(deferred)) == original_path

    asyncio.run(run())


def test_a_retry_runs_the_rollouts_without_a_score_again_and_ends_the_evaluation_again(tmp_path: Path) -> None:
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(), live=LiveObjects(runtime=ArtifactRuntime()))
    queue = dispatcher.queue

    async def run() -> None:
        # No worker runs: the test takes each rollout and ends it.
        server = TestServer(create_app(dispatcher, tokens=TOKEN, evaluations=ArtifactEvaluations(dispatcher)))
        async with TestClient(server) as client:
            artifact_id = await created(client, tmp_path, "add", "sort")
            body = {"artifact_id": artifact_id, "harness": {"name": "terminus-2"}, "rollouts_per_task": 2}
            accepted = await (await client.post("/reef/scenarios/s/evaluations", headers=AUTH, json=body)).json()
            route = f"/reef/scenarios/s/evaluations/{accepted['evaluation_id']}"
            assert (await client.post(f"{route}/retry", headers=AUTH)).status == 409, "no rollout ended yet"

            # While the evaluation runs, a rollout that failed goes back to the queue, and the others stay.
            first_item = queue.take(EVALUATIONS_QUEUE, "w", [EVALUATION_ROLLOUT_KIND])
            queue.fail(EVALUATIONS_QUEUE, first_item.id, "w", "the provider answered 503", retryable=False)
            during = await client.post(f"{route}/retry", headers=AUTH)
            assert during.status == 202 and (await during.json())["state"] in ("queued", "running")
            states = [(item.state, item.attempt_count) for item in queue.items(EVALUATIONS_QUEUE, scenario="s")]
            assert states == [("queued", 1), ("queued", 0), ("queued", 0), ("queued", 0)]

            # One rollout scores, one fails in its sandbox, one finishes with no score, and one scores.
            for end in (1.0, "the sandbox stopped", None, 0.0):
                item = queue.take(EVALUATIONS_QUEUE, "w", [EVALUATION_ROLLOUT_KIND])
                if isinstance(end, str):
                    queue.fail(EVALUATIONS_QUEUE, item.id, "w", end, retryable=False)
                else:
                    queue.finish(EVALUATIONS_QUEUE, item.id, "w", {"score": RolloutScore(end).to_json()})
            first = await finished(client, accepted["evaluation_id"])
            assert (first["state"], first["mean_score"]) == ("finished", 0.5)

            retried = await client.post(f"{route}/retry", headers=AUTH)
            assert retried.status == 202, await retried.text()
            answer = await retried.json()
            assert (answer["state"], answer["policy_evaluation"], answer["finished_at"]) == ("running", None, None)
            listed = (await (await client.get(f"{route}/rollouts", headers=AUTH)).json())["rollouts"]
            assert [(row["state"], row["attempt_count"]) for row in listed] == [
                ("finished", 2),
                ("queued", 1),
                ("queued", 1),
                ("finished", 1),
            ]
            assert listed[0]["score"] == RolloutScore(1.0).to_json(), "a scored rollout keeps its score"

            # The two rollouts run their second attempt, and the evaluation ends with every score.
            for value in (1.0, 1.0):
                item = queue.take(EVALUATIONS_QUEUE, "w", [EVALUATION_ROLLOUT_KIND])
                assert item.attempt_count == 2
                queue.finish(EVALUATIONS_QUEUE, item.id, "w", {"score": RolloutScore(value).to_json()})
            second = await finished(client, accepted["evaluation_id"])
            assert (second["state"], second["mean_score"]) == ("finished", 0.75)
            assert second["finished_at"] >= first["finished_at"]

            # Every rollout has a score now. An unknown evaluation is 404.
            assert (await client.post(f"{route}/retry", headers=AUTH)).status == 409
            assert (await client.post("/reef/scenarios/s/evaluations/unknown/retry", headers=AUTH)).status == 404

    asyncio.run(run())


def test_a_retry_keeps_max_concurrent_rollouts_and_queues_the_others_as_rollouts_end(tmp_path: Path) -> None:
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(), live=LiveObjects(runtime=ArtifactRuntime()))
    queue = dispatcher.queue

    async def taken() -> WorkItem:
        # A rollout enters the queue when the results thread settles the evaluation after a rollout ended.
        for _ in range(200):
            item = queue.take(EVALUATIONS_QUEUE, "w", [EVALUATION_ROLLOUT_KIND])
            if item is not None:
                assert sum(not other.final for other in queue.items(EVALUATIONS_QUEUE, scenario="s")) <= 2
                return item
            await asyncio.sleep(0.05)
        raise AssertionError("no rollout entered the queue")

    async def run() -> None:
        # No worker runs: the test takes each rollout and ends it.
        server = TestServer(create_app(dispatcher, tokens=TOKEN, evaluations=ArtifactEvaluations(dispatcher)))
        async with TestClient(server) as client:
            artifact_id = await created(client, tmp_path, "add", "sort")
            body = {
                "artifact_id": artifact_id,
                "harness": {"name": "terminus-2"},
                "rollouts_per_task": 2,
                "max_concurrent_rollouts": 2,
            }
            accepted = await (await client.post("/reef/scenarios/s/evaluations", headers=AUTH, json=body)).json()
            route = f"/reef/scenarios/s/evaluations/{accepted['evaluation_id']}"
            for _ in range(4):
                item = await taken()
                queue.fail(EVALUATIONS_QUEUE, item.id, "w", "agent setup timed out", retryable=False)
            assert (await finished(client, accepted["evaluation_id"]))["progress"]["failed"] == 4

            # Two rollouts go back to the queue; the other two wait, and no rollout counts as ended.
            assert (await client.post(f"{route}/retry", headers=AUTH)).status == 202
            listed = (await (await client.get(f"{route}/rollouts", headers=AUTH)).json())["rollouts"]
            assert [(row["state"], row["attempt_count"], row["error"]) for row in listed] == [
                ("queued", 1, "agent setup timed out"),
                ("queued", 1, "agent setup timed out"),
                ("waiting", 1, None),
                ("waiting", 1, None),
            ]
            progress = (await (await client.get(route, headers=AUTH)).json())["progress"]
            assert (progress["ended"], progress["failed"], progress["cancelled"]) == (0, 0, 0)

            # Each rollout that ends lets one that waits go back to the queue, and the evaluation ends with all four.
            for value in (1.0, 0.0, 1.0, 1.0):
                item = await taken()
                assert item.attempt_count == 2
                queue.finish(EVALUATIONS_QUEUE, item.id, "w", {"score": RolloutScore(value).to_json()})
            answer = await finished(client, accepted["evaluation_id"])
            assert (answer["state"], answer["mean_score"]) == ("finished", 0.75)
            assert not any(
                item.metadata.get("waits_for_retry") for item in queue.items(EVALUATIONS_QUEUE, scenario="s")
            )

    asyncio.run(run())


def test_a_retry_runs_the_failed_rollouts_and_a_resume_runs_the_cancelled_and_unstarted_ones(tmp_path: Path) -> None:
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(), live=LiveObjects(runtime=ArtifactRuntime()))
    queue = dispatcher.queue

    async def taken() -> WorkItem:
        # A rollout enters the queue when the results thread settles the evaluation after a rollout ended.
        for _ in range(200):
            item = queue.take(EVALUATIONS_QUEUE, "w", [EVALUATION_ROLLOUT_KIND])
            if item is not None:
                return item
            await asyncio.sleep(0.05)
        raise AssertionError("no rollout entered the queue")

    async def run() -> None:
        # No worker runs: the test takes each rollout and ends it.
        server = TestServer(create_app(dispatcher, tokens=TOKEN, evaluations=ArtifactEvaluations(dispatcher)))
        async with TestClient(server) as client:
            artifact_id = await created(client, tmp_path, "add", "sort")
            body = {
                "artifact_id": artifact_id,
                "harness": {"name": "terminus-2"},
                "rollouts_per_task": 2,
                "max_concurrent_rollouts": 1,
            }
            accepted = await (await client.post("/reef/scenarios/s/evaluations", headers=AUTH, json=body)).json()
            route = f"/reef/scenarios/s/evaluations/{accepted['evaluation_id']}"
            # The first rollout fails in its sandbox, the cancel stops the second in its run, and two never start.
            first = await taken()
            queue.fail(EVALUATIONS_QUEUE, first.id, "w", "the sandbox stopped", retryable=False)
            second = await taken()
            queue.start(EVALUATIONS_QUEUE, second.id, "w")
            assert (await (await client.post(f"{route}/cancel", headers=AUTH)).json())["state"] == "failed"

            page = await (await client.get(f"{route}/rollouts?limit=4&state=failed&calls=false", headers=AUTH)).json()
            assert page["total"] == 1
            assert [(row["task"], row["repeat"]) for row in page["rollouts"]] == [("add", 0)]

            # A retry runs the failed rollout only: the cancelled rollout stays, and no other rollout starts.
            assert (await (await client.post(f"{route}/retry", headers=AUTH)).json())["state"] == "running"
            item = await taken()
            assert (item.id, item.attempt_count) == (first.id, 2)
            queue.finish(EVALUATIONS_QUEUE, item.id, "w", {"score": RolloutScore(1.0).to_json()})
            answer = await finished(client, accepted["evaluation_id"])
            assert (answer["state"], answer["error"]) == ("failed", CANCELLED_ERROR)
            progress = answer["progress"]
            assert (progress["ended"], progress["scored"], progress["cancelled"]) == (1, 1, 1)
            assert len(queue.items(EVALUATIONS_QUEUE, scenario="s")) == 2
            assert (await client.post(f"{route}/retry", headers=AUTH)).status == 409, "no rollout failed"

            # A resume runs the cancelled rollout again and starts the two others, one at a time.
            assert (await (await client.post(f"{route}/resume", headers=AUTH)).json())["state"] == "running"
            for value in (0.0, 1.0, 0.0):
                item = await taken()
                assert sum(not other.final for other in queue.items(EVALUATIONS_QUEUE, scenario="s")) == 1
                queue.finish(EVALUATIONS_QUEUE, item.id, "w", {"score": RolloutScore(value).to_json()})
            answer = await finished(client, accepted["evaluation_id"])
            assert (answer["state"], answer["mean_score"], answer["error"]) == ("finished", 0.5, None)
            assert (await client.post(f"{route}/resume", headers=AUTH)).status == 409
            assert (await client.post("/reef/scenarios/s/evaluations/unknown/resume", headers=AUTH)).status == 404

    asyncio.run(run())


@pytest.mark.parametrize("limit", [3, None])
def test_a_resume_can_change_max_concurrent_rollouts(tmp_path: Path, limit: int | None) -> None:
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(), live=LiveObjects(runtime=ArtifactRuntime()))
    queue = dispatcher.queue

    def not_final() -> list[WorkItem]:
        return [item for item in queue.items(EVALUATIONS_QUEUE, scenario="s") if not item.final]

    async def queued_rollouts(count: int) -> list[WorkItem]:
        # A rollout enters the queue when the results thread settles the evaluation after a rollout ended.
        for _ in range(200):
            if len(not_final()) == count:
                return not_final()
            await asyncio.sleep(0.05)
        raise AssertionError(f"the queue did not hold {count} rollouts that are not final")

    async def taken() -> WorkItem:
        for _ in range(200):
            item = queue.take(EVALUATIONS_QUEUE, "w", [EVALUATION_ROLLOUT_KIND])
            if item is not None:
                return item
            await asyncio.sleep(0.05)
        raise AssertionError("no rollout entered the queue")

    async def run() -> None:
        # No worker runs: the test takes each rollout and ends it.
        server = TestServer(create_app(dispatcher, tokens=TOKEN, evaluations=ArtifactEvaluations(dispatcher)))
        async with TestClient(server) as client:
            artifact_id = await created(client, tmp_path, "add", "sort")
            body = {
                "artifact_id": artifact_id,
                "harness": {"name": "terminus-2"},
                "rollouts_per_task": 2,
                "max_concurrent_rollouts": 1,
            }
            accepted = await (await client.post("/reef/scenarios/s/evaluations", headers=AUTH, json=body)).json()
            route = f"/reef/scenarios/s/evaluations/{accepted['evaluation_id']}"
            await queued_rollouts(1)
            first = await taken()
            queue.start(EVALUATIONS_QUEUE, first.id, "w")
            assert (await (await client.post(f"{route}/cancel", headers=AUTH)).json())["state"] == "failed"

            # A refused body changes nothing.
            for refused in (
                {"max_concurrent_rollouts": 0},
                {"max_concurrent_rollouts": True},
                {"rollouts_per_task": 1},
            ):
                assert (await client.post(f"{route}/resume", headers=AUTH, json=refused)).status == 400
            answer = await (await client.get(route, headers=AUTH)).json()
            assert (answer["state"], answer["request"]["max_concurrent_rollouts"]) == ("failed", 1)

            # The cancelled rollout and the three that never started run at the new limit.
            resumed = await client.post(f"{route}/resume", headers=AUTH, json={"max_concurrent_rollouts": limit})
            assert resumed.status == 202, await resumed.text()
            assert (await resumed.json())["request"]["max_concurrent_rollouts"] == limit
            running = 4 if limit is None else limit
            await queued_rollouts(running)
            for value in (1.0, 0.0, 1.0, 1.0):
                item = await taken()
                assert len(not_final()) <= running
                queue.finish(EVALUATIONS_QUEUE, item.id, "w", {"score": RolloutScore(value).to_json()})
            answer = await finished(client, accepted["evaluation_id"])
            assert (answer["state"], answer["mean_score"]) == ("finished", 0.75)
            assert answer["request"]["max_concurrent_rollouts"] == limit

    asyncio.run(run())


def test_a_retry_of_an_evaluation_that_too_many_failures_stopped_runs_on(tmp_path: Path, monkeypatch) -> None:
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(), live=LiveObjects(runtime=ArtifactRuntime()))
    queue = dispatcher.queue
    queue_retry = queue.retry
    slow = iter([0.5])

    def retry_with_a_pause(queue_name: str, item_id: str) -> WorkItem | None:
        # The pause after the first rollout lets the results thread settle while the other rollouts are still failed.
        item = queue_retry(queue_name, item_id)
        time.sleep(next(slow, 0.0))
        return item

    async def run() -> None:
        server = TestServer(create_app(dispatcher, tokens=TOKEN, evaluations=ArtifactEvaluations(dispatcher)))
        async with TestClient(server) as client:
            artifact_id = await created(client, tmp_path, "add", "sort")
            body = {
                "artifact_id": artifact_id,
                "harness": {"name": "terminus-2"},
                # Exactly ``FAILURE_CHECK_ROLLOUTS`` rollouts: the stop comes after the last one ends, so it cancels none
                # that the loop below has not failed yet.
                "rollouts_per_task": FAILURE_CHECK_ROLLOUTS // 2,
                "max_failed_fraction": 0.5,
            }
            accepted = await (await client.post("/reef/scenarios/s/evaluations", headers=AUTH, json=body)).json()
            route = f"/reef/scenarios/s/evaluations/{accepted['evaluation_id']}"
            for _ in range(FAILURE_CHECK_ROLLOUTS):
                item = queue.take(EVALUATIONS_QUEUE, "w", [EVALUATION_ROLLOUT_KIND])
                queue.fail(EVALUATIONS_QUEUE, item.id, "w", "the sandbox refused the task", retryable=False)
            assert (await finished(client, accepted["evaluation_id"]))["state"] == "failed"

            monkeypatch.setattr(queue, "retry", retry_with_a_pause)
            assert (await client.post(f"{route}/retry", headers=AUTH)).status == 202
            await asyncio.sleep(0.5)
            answer = await (await client.get(route, headers=AUTH)).json()
            assert (answer["state"], answer["error"]) == ("running", None)
            states = {item.state for item in queue.items(EVALUATIONS_QUEUE, scenario="s")}
            assert states == {"queued"}

    asyncio.run(run())


def test_after_a_failure_stop_a_retry_runs_the_failed_rollouts_and_a_resume_the_stopped_ones(tmp_path: Path) -> None:
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(), live=LiveObjects(runtime=ArtifactRuntime()))
    queue = dispatcher.queue

    def ended_all(error: str | None) -> None:
        for _ in range(len(queue.items(EVALUATIONS_QUEUE, scenario="s"))):
            item = queue.take(EVALUATIONS_QUEUE, "w", [EVALUATION_ROLLOUT_KIND])
            if item is None:
                return
            if error is None:
                queue.finish(EVALUATIONS_QUEUE, item.id, "w", {"score": RolloutScore(1.0).to_json()})
            else:
                queue.fail(EVALUATIONS_QUEUE, item.id, "w", error, retryable=False)

    async def run() -> None:
        server = TestServer(create_app(dispatcher, tokens=TOKEN, evaluations=ArtifactEvaluations(dispatcher)))
        async with TestClient(server) as client:
            artifact_id = await created(client, tmp_path, "add", "sort")
            body = {
                "artifact_id": artifact_id,
                "harness": {"name": "terminus-2"},
                "rollouts_per_task": 6,
                "max_failed_fraction": 0.5,
            }
            accepted = await (await client.post("/reef/scenarios/s/evaluations", headers=AUTH, json=body)).json()
            route = f"/reef/scenarios/s/evaluations/{accepted['evaluation_id']}"
            # Ten of the twelve rollouts fail, and the failure check stops the other two.
            for item in queue.items(EVALUATIONS_QUEUE, scenario="s")[:10]:
                queue.take(EVALUATIONS_QUEUE, "w", [EVALUATION_ROLLOUT_KIND])
                queue.fail(EVALUATIONS_QUEUE, item.id, "w", "the sandbox refused the task", retryable=False)
            stopped = (await finished(client, accepted["evaluation_id"]))["progress"]
            assert (stopped["failed"], stopped["cancelled"]) == (10, 2)

            # The retried rollouts score; the two stopped rollouts do not count as failed, and do not run.
            assert (await client.post(f"{route}/retry", headers=AUTH)).status == 202
            ended_all(None)
            answer = await finished(client, accepted["evaluation_id"])
            assert answer["state"] == "failed"
            assert answer["error"] == "too many rollouts failed: the stop ended 2 rollouts, which wait for a resume"
            assert (answer["progress"]["scored"], answer["progress"]["cancelled"]) == (10, 2)

            assert (await client.post(f"{route}/resume", headers=AUTH)).status == 202
            ended_all(None)
            answer = await finished(client, accepted["evaluation_id"])
            assert (answer["state"], answer["mean_score"]) == ("finished", 1.0)

    asyncio.run(run())


@pytest.mark.integration
def test_max_failed_fraction_stops_an_evaluation_when_402_refusals_are_followed_by_auxiliary_calls(
    tmp_path: Path,
) -> None:
    class RefusingToolsHandler(ArtifactHandler):
        """An ``ArtifactHandler`` that refuses tool-bearing chat calls with HTTP 402 and answers tool-less calls.

        Values: inherits ``artifact_ids: list[str]`` and ``models: list[str]`` from ``ArtifactHandler`` (both initially
        empty). Invariant: only answered (tool-less) calls are recorded in ``artifact_ids`` and ``models``; refused
        calls raise before ``super().inference`` runs.

        Purpose: used by this circuit-breaker test to model a provider that rejects main-loop calls with payment
        refusals while answering auxiliary title calls. ``inference(artifact, path, payload)`` raises
        ``UpstreamStatusError(..., status=402, limit=UpstreamLimit.PAYMENT)`` iff ``bool(payload.get("tools"))`` is
        True, and otherwise delegates to ``ArtifactHandler.inference`` to record the call and return a 200 reply.
        """

        async def inference(self, artifact: Artifact, path: str, payload: dict[str, Any]) -> dict[str, Any]:
            if payload.get("tools"):
                raise UpstreamStatusError("insufficient credits", status=402, limit=UpstreamLimit.PAYMENT)
            return await super().inference(artifact, path, payload)

    class RefusedThenTitleLab(TaskLab):
        """A test ``TaskLab`` that sends a refused tool-bearing call followed by a tool-less call and exits cleanly.

        Values: ``runs`` is a non-negative int (initially 0) counting how many times ``run`` was invoked.

        Purpose: used by this circuit-breaker test to reproduce a harness (such as OpenCode) whose main-loop call hits
        HTTP 402 and whose trailing title call returns 200 before the trial finishes without an error. ``run`` requires
        ``agent["kwargs"]["api_base"]`` as a base URL string, increments ``runs``, posts two ``/chat/completions``
        requests (the first with a non-empty ``tools`` list, the second without ``tools``) while suppressing
        ``urllib.error.HTTPError``, and always returns ``RolloutRow(rewards={"reward": 0.0}, error=None)``.
        ``build_environment`` starts no sandbox and returns ``""``.
        """

        def __init__(self) -> None:
            self.runs = 0

        async def run(self, task_path, agent, *, key, tags, overrides) -> RolloutRow:
            self.runs += 1
            api_base = agent["kwargs"]["api_base"]
            payloads = [
                {
                    "model": "m",
                    "messages": [{"role": "user", "content": "solve"}],
                    "tools": [{"type": "function", "function": {"name": "bash"}}],
                },
                {"model": "m", "messages": [{"role": "user", "content": "title"}]},
            ]
            for payload in payloads:
                request = urllib.request.Request(
                    f"{api_base}/chat/completions",
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with contextlib.suppress(urllib.error.HTTPError):
                    await asyncio.to_thread(lambda req=request: urllib.request.urlopen(req, timeout=10).read())
            return RolloutRow(rewards={"reward": 0.0}, error=None)

        async def build_environment(self, task_path, *, overrides) -> str:
            return ""

    runtime = ArtifactRuntime()
    runtime.handler = RefusingToolsHandler()
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(), live=LiveObjects(runtime=runtime))
    lab = RefusedThenTitleLab()

    async def run() -> None:
        server, _ = evaluating_server(dispatcher, lab)
        async with TestClient(server) as client:
            artifact_id = await created(client, tmp_path, "add", "sort")
            body = {
                "artifact_id": artifact_id,
                "harness": {"name": "terminus-2"},
                "rollouts_per_task": 6,
                "max_concurrent_rollouts": 1,
                "max_failed_fraction": 0.5,
            }
            accepted = await (await client.post("/reef/scenarios/s/evaluations", headers=AUTH, json=body)).json()
            answer = await finished(client, accepted["evaluation_id"])
            assert answer["state"] == "failed"
            assert answer["error"] == "too many rollouts failed: 10 of 10 ended (100%), over the limit of 50%"
            assert (
                answer["progress"]["total"],
                answer["progress"]["ended"],
                answer["progress"]["scored"],
                answer["progress"]["failed"],
                answer["progress"]["cancelled"],
            ) == (12, 10, 0, 10, 0)
            assert lab.runs == 10

    asyncio.run(run())


def chat_call(attempt: int, index: int, messages: list[dict[str, str]], reply: str, status: int = 200):
    """A kept chat call of rollout (2, 0) that sent ``messages`` and got ``reply``, or an error when not 200."""
    response = (
        {"choices": [{"message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}]}
        if status == 200
        else {"error": {"message": reply}}
    )
    summary = EvaluationCallSummary(
        2, 0, attempt, index, "/v1/chat/completions", status, 10.0 * attempt + index, 0.5, None, None,
        "stop" if status == 200 else None, 0,
    )  # fmt: skip
    return EvaluationCall(summary, {"model": "m", "messages": messages}, response)


@pytest.mark.unit
def test_the_calls_of_each_attempt_stitch_into_one_trajectory_and_the_scored_attempt_carries_the_reward() -> None:
    rollout = EvaluationRollout(
        "s", "e1", "a1", "weights", 2, "add", 0, PolicyModel("m"), {"name": "terminus-2"}, "docker", {}
    )
    item = WorkItem(
        EVALUATIONS_QUEUE,
        rollout.item_id(),
        "s",
        EVALUATION_ROLLOUT_KIND,
        rollout.to_payload(),
        state="finished",
        started_at=1.0,
        ended_at=2.0,
        attempt_count=3,
        result={"score": 0.75},
    )
    first = [{"role": "user", "content": "solve add"}]
    second = [*first, {"role": "assistant", "content": "ls"}, {"role": "user", "content": "a.txt"}]
    calls = [
        chat_call(1, 0, first, "ls"),
        chat_call(2, 0, first, "the engine stopped", status=502),
        chat_call(3, 0, first, "ls"),
        chat_call(3, 1, first, "the engine stopped", status=502),
        chat_call(3, 2, second, "cat a.txt"),
    ]
    earlier, scored = attempt_trajectories("s", "e1", item, calls)
    assert earlier["extra"]["reef"]["reward"] is None and scored["extra"]["reef"]["reward"] == 0.75
    assert scored["extra"]["reef"]["evaluation"] == {
        "evaluation_id": "e1",
        "task": "add",
        "task_index": 2,
        "repeat": 0,
        "attempt": 3,
        "caller": "agent",
        "conversation": 0,
        "scored": True,
    }
    # Attempt 2 has no 200 call, so it has no trajectory.
    assert (earlier["extra"]["reef"]["evaluation"]["attempt"], earlier["extra"]["reef"]["evaluation"]["scored"]) == (
        1,
        False,
    )
    records = scored["extra"]["reef"]["records"]
    assert [record["agent_record_id"] for record in records] == ["e1:2:0:3:0", "e1:2:0:3:2"]
    # Each record is the call's summary: its time and its settings, without the conversation that the steps hold.
    assert records[0]["created_at"] == 30.0
    assert not CONVERSATION_FIELDS & set(records[0]["payload"])
    # The second call repeats the first exchange, so the stitched steps hold each message once.
    assert [step["message"] for step in scored["steps"]] == ["solve add", "ls", "a.txt", "cat a.txt"]
    running = replace(item, state="running", ended_at=None, result=None, worker_id="w", lease_expires_at=9.0)
    assert [
        trajectory["extra"]["reef"]["reward"] for trajectory in attempt_trajectories("s", "e1", running, calls)
    ] == [
        None,
        None,
    ]


class Messages(list):
    """A request's message list that a weak reference can follow."""


class CountedCalls(Sequence[EvaluationCall]):
    """Calls whose bodies are made each time they are asked for, as ``StoredCalls`` reads them, with the most
    request message lists that were alive at once: what a reader of a rollout keeps of its calls."""

    def __init__(self, calls: list[EvaluationCall]) -> None:
        self.calls, self.live, self.most = calls, 0, 0

    def __len__(self) -> int:
        return len(self.calls)

    def __getitem__(self, index):
        call = self.calls[index]
        messages = Messages(call.request["messages"])
        self.live += 1
        self.most = max(self.most, self.live)
        weakref.finalize(messages, self.release)
        return replace(call, request={**call.request, "messages": messages})

    def release(self) -> None:
        self.live -= 1


@pytest.mark.unit
def test_an_attempt_is_stitched_with_one_call_at_a_time_and_can_leave_out_the_raw_bodies() -> None:
    rollout = EvaluationRollout(
        "s", "e1", "a1", "weights", 2, "add", 0, PolicyModel("m"), {"name": "terminus-2"}, "docker", {}
    )
    item = WorkItem(
        EVALUATIONS_QUEUE,
        rollout.item_id(),
        "s",
        EVALUATION_ROLLOUT_KIND,
        rollout.to_payload(),
        state="finished",
        started_at=1.0,
        ended_at=2.0,
        attempt_count=1,
        result={"score": 1.0},
    )
    # Each call sends the whole conversation again, as an agent does.
    messages: list[dict[str, str]] = [{"role": "user", "content": "solve add"}]
    calls = []
    for index in range(50):
        calls.append(chat_call(1, index, list(messages), f"step {index}"))
        messages += [{"role": "assistant", "content": f"step {index}"}, {"role": "user", "content": f"out {index}"}]
    counted = CountedCalls(calls)
    (lean,) = attempt_trajectories("s", "e1", item, counted, include_records=False)
    # A few calls at once (the one read now and the one before it, in each loop), not one per call: 50 before.
    assert counted.most <= 4, counted.most
    (full,) = attempt_trajectories("s", "e1", item, calls)
    assert lean["steps"] == full["steps"] and len(full["extra"]["reef"]["records"]) == 50
    assert "records" not in lean["extra"]["reef"] and lean["extra"]["reef"]["reward"] == 1.0


class CountedReads(MemoryDataStore):
    """A memory data store that counts its reads of items."""

    reads = 0

    def read(self, owner, item_id):
        self.reads += 1
        return super().read(owner, item_id)

    def read_many(self, owner, item_ids, *, content=True):
        self.reads += 1
        return super().read_many(owner, item_ids, content=content)


@pytest.mark.unit
def test_stored_calls_are_read_in_batches_and_stitched_like_calls_in_memory() -> None:
    rollout = EvaluationRollout(
        "s", "e1", "a1", "weights", 2, "add", 0, PolicyModel("m"), {"name": "terminus-2"}, "docker", {}
    )
    item = WorkItem(
        EVALUATIONS_QUEUE, rollout.item_id(), "s", EVALUATION_ROLLOUT_KIND, rollout.to_payload(),
        state="finished", started_at=1.0, ended_at=2.0, attempt_count=1, result={"score": 1.0},
    )  # fmt: skip
    # Two roles take turns for 20 calls each, so the second read adds the calls of both conversations at once.
    roles = {name: [{"role": "system", "content": f"You are {name}."}] for name in ("Ann", "Ben")}
    calls = []
    for index in range(40):
        name = ("Ann", "Ben")[index % 2]
        calls.append(chat_call(1, index, list(roles[name]), f"{name} {index}"))
        roles[name] += [{"role": "assistant", "content": f"{name} {index}"}, {"role": "user", "content": "go on"}]
    store = CountedReads(LocalTaskFileStore())
    stored = store.create("o", [evaluation_call_item("s", "a1", "e1", call) for call in calls])
    store.reads = 0
    read = attempt_trajectories("s", "e1", item, StoredCalls(store, "o", stored), include_records=False)
    # Two reads of the 40 calls, 16 calls a read: 3 reads each, not one read for each call.
    assert store.reads == 2 * math.ceil(40 / READ_CALLS)
    in_memory = attempt_trajectories("s", "e1", item, calls, include_records=False)
    assert [trajectory["steps"] for trajectory in read] == [trajectory["steps"] for trajectory in in_memory]
    # Each role: its system message, 20 replies and the 19 turns between them.
    assert [len(trajectory["steps"]) for trajectory in read] == [40, 40]


@pytest.mark.unit
def test_an_attempt_that_plays_two_roles_gives_one_trajectory_for_each_role_without_replaced_calls() -> None:
    rollout = EvaluationRollout(
        "s", "e1", "a1", "weights", 2, "add", 0, PolicyModel("m"), {"name": "terminus-2"}, "docker", {}
    )
    item = WorkItem(
        EVALUATIONS_QUEUE,
        rollout.item_id(),
        "s",
        EVALUATION_ROLLOUT_KIND,
        rollout.to_payload(),
        state="finished",
        started_at=1.0,
        ended_at=2.0,
        attempt_count=1,
        result={"score": 0.5},
    )
    ann = [{"role": "system", "content": "You are Ann."}, {"role": "user", "content": "Start."}]
    ben = [{"role": "system", "content": "You are Ben."}, {"role": "user", "content": "Ann: Hello."}]
    ann_again = [*ann, {"role": "assistant", "content": "Hello."}, {"role": "user", "content": "Ben: Hi."}]
    # The roles take turns. Ann's second call has an empty answer, and the agent asks again with the same messages.
    calls = [
        chat_call(1, 0, ann, "Hello."),
        chat_call(1, 1, ben, "Hi."),
        chat_call(1, 2, ann_again, ""),
        chat_call(1, 3, ann_again, "How are you?"),
    ]
    first, second = attempt_trajectories("s", "e1", item, calls)
    assert [step["message"] for step in first["steps"]] == [
        "You are Ann.",
        "Start.",
        "Hello.",
        "Ben: Hi.",
        "How are you?",
    ]
    assert [step["message"] for step in second["steps"]] == ["You are Ben.", "Ann: Hello.", "Hi."]
    assert [record["agent_record_id"] for record in first["extra"]["reef"]["records"]] == ["e1:2:0:1:0", "e1:2:0:1:3"]
    assert [trajectory["extra"]["reef"]["evaluation"]["conversation"] for trajectory in (first, second)] == [0, 1]
    assert first["extra"]["reef"]["reward"] == second["extra"]["reef"]["reward"] == 0.5


@pytest.mark.unit
def test_streaming_releases_finished_conversations_and_preserves_shared_branches() -> None:
    rollout = EvaluationRollout("s", "e1", "a1", "weights", 2, "add", 0, PolicyModel("m"), {}, "docker", {})
    item = WorkItem(
        EVALUATIONS_QUEUE,
        rollout.item_id(),
        "s",
        EVALUATION_ROLLOUT_KIND,
        rollout.to_payload(),
        state="finished",
        started_at=1.0,
        ended_at=2.0,
        attempt_count=1,
        result={"score": 0.5},
    )
    question = [{"role": "user", "content": "question"}]
    shared = [*question, {"role": "assistant", "content": "shared reply"}]
    calls = [
        chat_call(1, 0, question, "shared reply"),
        chat_call(1, 1, [*shared, {"role": "user", "content": "left"}], "answer left"),
        chat_call(1, 2, [*shared, {"role": "user", "content": "right"}], "answer right"),
        *(chat_call(1, i, [{"role": "user", "content": f"unrelated {i}"}], "answer") for i in range(3, 70)),
    ]
    counted = CountedCalls(calls)
    documents = iter_attempt_trajectories("s", "e1", item, counted)
    left, right = next(documents), next(documents)
    assert [s["message"] for s in left["steps"]] == ["question", "shared reply", "left", "answer left"]
    assert [s["message"] for s in right["steps"]] == ["question", "shared reply", "right", "answer right"]
    assert [r["agent_record_id"] for r in left["extra"]["reef"]["records"]] == ["e1:2:0:1:1"]
    assert [r["agent_record_id"] for r in right["extra"]["reef"]["records"]] == ["e1:2:0:1:0", "e1:2:0:1:2"]
    for ordinal, doc in enumerate(documents, start=2):
        assert doc["extra"]["reef"]["evaluation"]["conversation"] == ordinal
        assert doc["extra"]["reef"]["reward"] == 0.5
    assert counted.most <= 4


class TreeCallingLab(TaskLab):
    """Builds the agent as Harbor does, keeps what it rendered, and calls the model with the rendered binding."""

    def __init__(self, logs: Path) -> None:
        self.logs = logs
        self.rendered: dict[str, str] = {}
        self.overrides: Mapping[str, object] = {}
        self.agent: TreeAgent | None = None

    async def run(self, task_path, agent, *, key, tags, overrides) -> RolloutRow:
        import_path = str(agent["import_path"])
        assert import_path == "reef.harness.harbor.agent:TreeAgent"
        kwargs = cast(dict[str, Any], agent["kwargs"])
        self.overrides = overrides
        # Harbor gives the agent the config's env as extra_env.
        self.agent = TreeAgent(self.logs, extra_env=cast(dict[str, str], agent.get("env") or {}), **kwargs)
        files = Path(kwargs["files_dir"])
        self.rendered = {str(path.relative_to(files)): path.read_text() for path in files.rglob("*") if path.is_file()}
        body = json.dumps({"model": "m", "messages": [{"role": "user", "content": "solve"}]}).encode()
        request = urllib.request.Request(
            f"{kwargs['base_url']}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        await asyncio.to_thread(lambda: urllib.request.urlopen(request, timeout=10).read())
        return RolloutRow(rewards={"reward": 1.0})

    async def build_environment(self, task_path, *, overrides) -> str:
        return ""


def harness_tree(root: Path) -> str:
    """The URI of a directory that holds a harness tree for pi, whose skill holds the text of a placeholder."""
    HarnessTree({"skills/check/SKILL.md": "---\nname: check\ndescription: Check.\n---\nCall {model} once.\n"}).write(
        root
    )
    return f"file://{root}/"


def tree_item(
    dispatcher: Dispatcher, uri: str | None, *, kind: str = "harness_plugin", adapter: str | None = "pi"
) -> str:
    """Store an artifact item of a harness in another scenario of the owner of ``s``, and return its ID."""
    item_id = f"other/artifact:{uuid.uuid4().hex}"
    metadata = {
        "kind": "artifact",
        "origin": {"scenario": "other", "artifact_id": item_id.removeprefix("other/")},
        **({"adapter": adapter} if adapter else {}),
    }
    content = {"source_uri": uri, "kind": kind, "component": "harness"}
    owner = cast(str, dispatcher.scenario_owner("s"))
    DataAccess.of(dispatcher).require_store().create(owner, [NewItem(item_id, content, metadata)])
    return item_id


@pytest.mark.integration
def test_a_model_artifact_is_evaluated_with_a_harness_tree_from_an_artifact_item(tmp_path: Path) -> None:
    runtime = ArtifactRuntime()
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(), live=LiveObjects(runtime=runtime))
    lab = TreeCallingLab(tmp_path / "logs")
    dispatcher.workers.add(
        EvaluationRolloutWorker(
            dispatcher.queue, LocalCandidateEndpoints(dispatcher), worker_id="trees", lab=lab, agent_host="127.0.0.1"
        )
    )

    async def run() -> None:
        evaluations = ArtifactEvaluations(dispatcher)
        async with TestClient(TestServer(create_app(dispatcher, tokens=TOKEN, evaluations=evaluations))) as client:
            artifact_id = await created(client, tmp_path, "add")
            await client.post("/reef/scenarios", headers=AUTH, json={"name": "other", "owner": "s"})
            rule = {
                "activation": {"module": "evaluated", "config": {"min_mean_score": 0.5}},
                "artifact_evaluation": {"harness": {"name": "terminus-2"}},
            }
            response = await client.post("/reef/scenarios/s/update", headers=AUTH, json=rule)
            assert response.status == 200, await response.text()
            activated: list[str] = []
            evaluations.results.activate = lambda scenario, request: activated.append(request.artifact_id)
            item_id = tree_item(dispatcher, harness_tree(tmp_path / "tree"))
            harness = {
                "item": item_id,
                "env": {"REEF_MAX_OUTPUT_TOKENS": "8192", "TASK_MODE": "fast"},
                "config": {"primary": {"defaultTools": []}},
            }
            body = {"artifact_id": artifact_id, "harness": harness}
            response = await client.post("/reef/scenarios/s/evaluations", headers=AUTH, json=body)
            assert response.status == 202, await response.text()
            answer = await finished(client, (await response.json())["evaluation_id"])
            assert activated == [], "a score with a harness from an item activates nothing"

        assert answer["state"] == "finished", answer
        assert answer["request"]["harness"] == harness
        assert lab.agent is not None and lab.agent.extra_env["TASK_MODE"] == "fast", "the agent's commands get the env"
        assert '"maxTokens": 8192' in lab.rendered["pi-agent/models.json"], "the env sets the binding's reply budget"
        assert json.loads(lab.rendered["plugin/settings.json"]) == {"defaultTools": []}, "the config merges in"
        evaluation = answer["policy_evaluation"]
        assert evaluation["harness"]["base"] == "pi"
        assert evaluation["scores"] == [{"task": "add", "scores": [1.0]}]
        assert runtime.handler.artifact_ids == [artifact_id]
        rendered = "".join(lab.rendered.values())
        assert "Call {model} once." in rendered, "the runner fills no placeholder in the tree's own text"
        assert "http://127.0.0.1:" in rendered, "the rendered binding names the address of the capture proxy"
        assert "127.0.0.1" not in json.dumps(evaluation["harness"]["plugin"]), "the result keeps no endpoint"

    asyncio.run(run())


@pytest.mark.integration
def test_a_tree_in_a_remote_environment_bridges_its_container_to_the_proxy(tmp_path: Path) -> None:
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(), live=LiveObjects(runtime=ArtifactRuntime()))
    lab = TreeCallingLab(tmp_path / "logs")

    async def run() -> None:
        server, _ = evaluating_server(dispatcher, lab)
        async with TestClient(server) as client:
            artifact_id = await created(client, tmp_path, "add")
            await client.post("/reef/scenarios", headers=AUTH, json={"name": "other", "owner": "s"})
            item_id = tree_item(dispatcher, harness_tree(tmp_path / "tree"))
            body = {"artifact_id": artifact_id, "harness": {"item": item_id}, "environment": "modal"}
            response = await client.post("/reef/scenarios/s/evaluations", headers=AUTH, json=body)
            assert response.status == 202, await response.text()
            answer = await finished(client, (await response.json())["evaluation_id"])
            assert answer["state"] == "finished", answer

    asyncio.run(run())
    assert lab.overrides["environment"] == {"type": "modal", "kwargs": {"sandbox_timeout_secs": 7200}}
    assert lab.agent is not None and lab.agent.bridge_port is not None
    assert f"http://127.0.0.1:{lab.agent.bridge_port}" in "".join(lab.rendered.values())


@pytest.mark.integration
@pytest.mark.parametrize(
    ("options", "reason"),
    [
        ({"kind": "model"}, "not an artifact of the kind harness_plugin"),
        ({"uri": None}, "only a published artifact"),
        ({"adapter": None}, "names no adapter"),
        ({"adapter": "missing"}, "unknown harness adapter"),
        ({"adapter": "native"}, "cannot run in a task's container"),
        ({"adapter": "claude"}, "cannot call this runtime's API dialect"),
        ({"env": {"REEF_CONTEXT_WINDOW": "big"}}, "REEF_CONTEXT_WINDOW must be a positive number of tokens"),
        ({"env": {"TOKEN": "${REEF_TOKEN}"}}, "text values without ${"),
        ({"env": {"1BAD": "x"}}, "maps variable names"),
        ({"config": {"primary": "x"}}, "maps config target names to JSON objects"),
        ({"config": {"missing": {}}}, "declares no config target 'missing'"),
        ({"config": {"models": {"providers": {"evil": {}}}}}, "the harness config fails validation"),
        ({"config": {"primary": {"defaultTools": ["bash"]}}}, "the harness config fails validation"),
    ],
)
def test_an_artifact_item_that_cannot_run_is_refused_before_any_rollout(
    tmp_path: Path, options: dict[str, Any], reason: str
) -> None:
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(), live=LiveObjects(runtime=ArtifactRuntime()))

    async def run() -> None:
        server, _ = evaluating_server(dispatcher, TreeCallingLab(tmp_path / "logs"))
        async with TestClient(server) as client:
            artifact_id = await created(client, tmp_path, "add")
            await client.post("/reef/scenarios", headers=AUTH, json={"name": "other", "owner": "s"})
            uri = options.pop("uri", harness_tree(tmp_path / "tree"))
            environment = options.pop("environment", "docker")
            extra = {key: options.pop(key) for key in ("env", "config") if key in options}
            body = {
                "artifact_id": artifact_id,
                "harness": {"item": tree_item(dispatcher, uri, **options), **extra},
                "environment": environment,
            }
            response = await client.post("/reef/scenarios/s/evaluations", headers=AUTH, json=body)
            assert response.status == 400 and reason in await response.text()

    asyncio.run(run())


@pytest.mark.integration
def test_an_item_of_another_owner_and_automatic_settings_are_refused(tmp_path: Path) -> None:
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(), live=LiveObjects(runtime=ArtifactRuntime()))

    async def run() -> None:
        server, _ = evaluating_server(dispatcher, TreeCallingLab(tmp_path / "logs"))
        async with TestClient(server) as client:
            artifact_id = await created(client, tmp_path, "add")
            await client.post("/reef/scenarios", headers=AUTH, json={"name": "other", "owner": "s"})
            item_id = tree_item(dispatcher, harness_tree(tmp_path / "tree"))

            body = {"artifact_id": artifact_id, "harness": {"item": "another/artifact:1"}}
            response = await client.post("/reef/scenarios/s/evaluations", headers=AUTH, json=body)
            assert response.status == 400 and "stores no item" in await response.text()

            settings = {"artifact_evaluation": {"harness": {"item": item_id}}}
            response = await client.post("/reef/scenarios/s/update", headers=AUTH, json=settings)
            assert response.status == 400 and "not an artifact item" in await response.text()

    asyncio.run(run())


def test_an_evaluation_stops_when_too_many_of_its_ended_rollouts_fail_and_reports_its_progress(
    tmp_path: Path,
) -> None:
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(), live=LiveObjects(runtime=ArtifactRuntime()))
    queue = dispatcher.queue

    async def end(score: float | None) -> None:
        for _ in range(200):
            item = queue.take(EVALUATIONS_QUEUE, "w", [EVALUATION_ROLLOUT_KIND])
            if item is not None:
                break
            await asyncio.sleep(0.05)
        else:
            raise AssertionError("no rollout was queued")
        queue.start(EVALUATIONS_QUEUE, item.id, "w")
        queue.finish(EVALUATIONS_QUEUE, item.id, "w", {"score": RolloutScore(score).to_json()})

    async def status(client: TestClient, route: str, ended: int) -> dict[str, Any]:
        for _ in range(200):
            answer = await (await client.get(route, headers=AUTH)).json()
            if answer["progress"]["ended"] == ended:
                return answer
            await asyncio.sleep(0.05)
        raise AssertionError(f"the evaluation did not count {ended} ended rollouts")

    async def run() -> None:
        # No worker runs: the test takes and ends each rollout itself, 12 rollouts of 2 tasks, 2 at once.
        server = TestServer(create_app(dispatcher, tokens=TOKEN, evaluations=ArtifactEvaluations(dispatcher)))
        async with TestClient(server) as client:
            artifact_id = await created(client, tmp_path, "add", "sort")
            body = {
                "artifact_id": artifact_id,
                "harness": {"name": "terminus-2"},
                "rollouts_per_task": 6,
                "max_concurrent_rollouts": 2,
                "max_failed_fraction": 0.5,
            }
            accepted = await (await client.post("/reef/scenarios/s/evaluations", headers=AUTH, json=body)).json()
            assert accepted["request"]["max_failed_fraction"] == 0.5
            route = f"/reef/scenarios/s/evaluations/{accepted['evaluation_id']}"

            # 9 ended, 6 of them failed: more than half, but fewer than 10 ended, so the evaluation runs on.
            for score in (1.0, 0.5, 0.0, None, None, None, None, None, None):
                await end(score)
            answer = await status(client, route, 9)
            assert answer["state"] == "running"
            progress = answer["progress"]
            assert {key: progress[key] for key in ("total", "ended", "scored", "failed", "cancelled")} == {
                "total": 12,
                "ended": 9,
                "scored": 3,
                "failed": 6,
                "cancelled": 0,
            }
            # The mean leaves out the failed rollouts.
            assert progress["mean_score"] == 0.5
            assert progress["eta_seconds"] is not None and progress["eta_seconds"] > 0
            assert progress["benchmarks"] == []
            listed = (await (await client.get("/reef/scenarios/s/evaluations", headers=AUTH)).json())["evaluations"]
            assert listed[0]["progress"] == progress | {"eta_seconds": listed[0]["progress"]["eta_seconds"]}

            # The tenth: 7 of 10 failed, over 50%. The evaluation fails, and the queued rollout is stopped.
            await end(None)
            answer = await finished(client, accepted["evaluation_id"])
            assert answer["state"] == "failed"
            assert answer["error"] == "too many rollouts failed: 7 of 10 ended (70%), over the limit of 50%"
            assert answer["progress"]["ended"] == 10
            assert answer["progress"]["eta_seconds"] is None
            # The rollouts that did not end are stopped, and the ones that wait never start.
            rows = (await (await client.get(f"{route}/rollouts", headers=AUTH)).json())["rollouts"]
            stopped = [row for row in rows if row["error"] == FAILURES_ERROR]
            assert answer["progress"]["cancelled"] == len(stopped)
            assert len(stopped) + [row["state"] for row in rows].count("waiting") == 2
            page = await (await client.get(f"{route}/rollouts?limit=12&state=failed&calls=false", headers=AUTH)).json()
            assert page["total"] == 7
            assert all(row["error"] != FAILURES_ERROR for row in page["rollouts"])
            progress = (await (await client.get(f"{route}?score_levels=true", headers=AUTH)).json())["progress"]
            assert progress["score_levels"] == {
                "full": 1,
                "part": 1,
                "zero": 1,
                "failed": 7,
                "pending": 2 - len(stopped),
            }

    asyncio.run(run())


def test_the_list_reads_the_rollouts_of_an_ended_evaluation_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(), live=LiveObjects(runtime=ArtifactRuntime()))
    queue = dispatcher.queue
    evaluations = ArtifactEvaluations(dispatcher)
    read_rollouts = evaluations.results.rollouts
    reads: list[str] = []

    def counted_rollouts(evaluation_id: str, scenario: str) -> tuple[WorkItem, ...]:
        reads.append(evaluation_id)
        return read_rollouts(evaluation_id, scenario)

    monkeypatch.setattr(evaluations.results, "rollouts", counted_rollouts)

    async def run() -> None:
        server = TestServer(create_app(dispatcher, tokens=TOKEN, evaluations=evaluations))
        async with TestClient(server) as client:
            artifact_id = await created(client, tmp_path, "add", "sort")
            body = {"artifact_id": artifact_id, "harness": {"name": "terminus-2"}}
            accepted = await (await client.post("/reef/scenarios/s/evaluations", headers=AUTH, json=body)).json()
            for _ in range(2):
                item = queue.take(EVALUATIONS_QUEUE, "w", [EVALUATION_ROLLOUT_KIND])
                queue.finish(EVALUATIONS_QUEUE, item.id, "w", {"score": RolloutScore(1.0).to_json()})
            ended = await finished(client, accepted["evaluation_id"])
            assert ended["progress"]["scored"] == 2

            reads.clear()
            for _ in range(2):
                listed = (await (await client.get("/reef/scenarios/s/evaluations", headers=AUTH)).json())[
                    "evaluations"
                ]
                assert listed[0]["progress"] == ended["progress"]
            assert reads == []

    asyncio.run(run())


def test_ended_progress_keeps_nothing_while_a_rollout_has_not_ended(tmp_path: Path) -> None:
    # A cancel stores the failed evaluation before it ends the rollouts; a reader in between sees a running rollout.
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(), live=LiveObjects(runtime=ArtifactRuntime()))
    queue = dispatcher.queue
    evaluations = ArtifactEvaluations(dispatcher)

    async def run() -> None:
        server = TestServer(create_app(dispatcher, tokens=TOKEN, evaluations=evaluations))
        async with TestClient(server) as client:
            artifact_id = await created(client, tmp_path, "add")
            body = {"artifact_id": artifact_id, "harness": {"name": "terminus-2"}}
            accepted = await (await client.post("/reef/scenarios/s/evaluations", headers=AUTH, json=body)).json()
            queue.take(EVALUATIONS_QUEUE, "w", [EVALUATION_ROLLOUT_KIND])
            running = evaluations.results.rollouts(accepted["evaluation_id"], "s")
            stored = replace(
                evaluations.get("s", accepted["evaluation_id"]), state="failed", error="x", finished_at=1e10
            )
            evaluations.ended_progress.add("s", stored, running, {"ended": 0})
            assert evaluations.ended_progress.read("s", stored) is None

    asyncio.run(run())


class NoCallLab(TaskLab):
    """A trial whose verifier writes a reward, though the agent reached no model: the endpoint went away."""

    async def run(self, task_path, agent, *, key, tags, overrides) -> RolloutRow:
        return RolloutRow(rewards={"reward": 0.0})

    async def build_environment(self, task_path, *, overrides) -> str:
        return ""


@pytest.mark.integration
def test_a_rollout_that_reached_no_model_fails_though_its_verifier_wrote_a_reward(tmp_path: Path) -> None:
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(), live=LiveObjects(runtime=ArtifactRuntime()))

    async def run() -> None:
        server, _ = evaluating_server(dispatcher, NoCallLab())
        async with TestClient(server) as client:
            artifact_id = await created(client, tmp_path, "add")
            body = {"artifact_id": artifact_id, "harness": {"name": "terminus-2"}}
            response = await client.post("/reef/scenarios/s/evaluations", headers=AUTH, json=body)
            assert response.status == 202, await response.text()
            answer = await finished(client, (await response.json())["evaluation_id"])
            rollouts = await (
                await client.get(f"/reef/scenarios/s/evaluations/{answer['evaluation_id']}/rollouts", headers=AUTH)
            ).json()

        assert answer["policy_evaluation"]["scores"] == [{"task": "add", "scores": ["failed"]}]
        assert answer["mean_score"] is None
        [row] = rollouts["rollouts"]
        assert row["attempt_count"] > 1, "a rollout that reached no model is retried"
        assert row["error"] == "ReefError: the rollout reached no model: the agent made no model call"

    asyncio.run(run())


@pytest.mark.integration
def test_a_candidate_that_inference_cannot_serve_fails_its_rollouts_and_scores_nothing(tmp_path: Path) -> None:
    runtime = ArtifactRuntime()
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(), live=LiveObjects(runtime=runtime))
    lab = ModelCallingLab({"add": 1.0})

    async def run() -> None:
        server, _ = evaluating_server(dispatcher, lab)
        async with TestClient(server) as client:
            artifact_id = await created(client, tmp_path, "add")
            runtime.serve_error = "the weights are gone"
            body = {"artifact_id": artifact_id, "harness": {"name": "terminus-2"}}
            response = await client.post("/reef/scenarios/s/evaluations", headers=AUTH, json=body)
            assert response.status == 202, await response.text()
            answer = await finished(client, (await response.json())["evaluation_id"])

        assert answer["policy_evaluation"]["scores"] == [{"task": "add", "scores": ["failed"]}]
        assert runtime.handler.artifact_ids == [], "no trial ran"

    asyncio.run(run())


@pytest.mark.integration
def test_an_artifact_that_is_not_published_is_refused_before_any_rollout(tmp_path: Path, monkeypatch) -> None:
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(), live=LiveObjects(runtime=ArtifactRuntime()))

    async def run() -> None:
        server, _ = evaluating_server(dispatcher, ModelCallingLab({"add": 1.0}))
        async with TestClient(server) as client:
            artifact_id = await created(client, tmp_path, "add")
            monkeypatch.setattr(
                ScenarioSnapshots, "find_snapshot", lambda self, revision: (SnapshotRef(revision, None), False)
            )
            body = {"artifact_id": artifact_id, "harness": {"name": "terminus-2"}}
            response = await client.post("/reef/scenarios/s/evaluations", headers=AUTH, json=body)
            assert response.status == 400 and "no durable bytes" in await response.text()
            monkeypatch.undo()
            listing = await (await client.get("/reef/scenarios/s/evaluations", headers=AUTH)).json()
            assert listing["evaluations"] == []

    asyncio.run(run())


@pytest.mark.integration
def test_a_request_that_the_scenario_cannot_run_is_refused_before_any_rollout(tmp_path: Path) -> None:
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(), live=LiveObjects(runtime=ArtifactRuntime()))

    async def run() -> None:
        app = create_app(dispatcher, tokens=TOKEN, evaluations=ArtifactEvaluations(dispatcher))
        async with TestClient(TestServer(app)) as client:
            artifact_id = await created(client, tmp_path, "add")
            agent = {"name": "terminus-2"}
            for body, status, message in (
                ({"artifact_id": artifact_id, "model": "served"}, 400, "configure the harness"),
                ({"artifact_id": artifact_id, "harness": agent, "task_names": ["sort"]}, 400, "holds no tasks"),
                ({"artifact_id": artifact_id, "harness": agent, "component": "harness"}, 400, "no component"),
                ({"artifact_id": artifact_id, "harness": agent, "rollouts": 2}, 400, "take no"),
                ({"artifact_id": "missing", "harness": agent}, 404, "missing"),
            ):
                response = await client.post("/reef/scenarios/s/evaluations", headers=AUTH, json=body)
                assert (response.status, message in await response.text()) == (status, True), await response.text()
            body = {"artifact_id": artifact_id, "harness": agent}
            unknown = await client.post("/reef/scenarios/other/evaluations", headers=AUTH, json=body)
            assert unknown.status == 404
            assert (await client.get("/reef/scenarios/s/evaluations/missing", headers=AUTH)).status == 404
            assert (await client.post("/reef/scenarios/s/evaluations", json={})).status == 401

    asyncio.run(run())


@pytest.mark.unit
def test_a_rollout_settles_its_evaluation_when_its_state_changes_and_not_for_a_lease_or_progress(
    tmp_path: Path, monkeypatch
) -> None:
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(), live=LiveObjects(runtime=ArtifactRuntime()))
    queue, results = dispatcher.queue, ArtifactEvaluations(dispatcher).results
    settled: list[str] = []
    monkeypatch.setattr(results, "settle", lambda scenario, evaluation_id: settled.append(evaluation_id))

    def settles(count: int) -> None:
        deadline = time.monotonic() + 10
        while len(settled) < count and time.monotonic() < deadline:
            time.sleep(0.02)
        # Three polls of the results thread, so that a settle too many would show.
        time.sleep(0.6)
        assert len(settled) == count, settled

    item_id = "evaluation_rollout:e1:00000:00000"
    queue.append(NewWork(EVALUATIONS_QUEUE, item_id, "s", EVALUATION_ROLLOUT_KIND, {"evaluation_id": "e1"}))
    results.open()
    try:
        settles(1)
        assert queue.take(EVALUATIONS_QUEUE, "w", [EVALUATION_ROLLOUT_KIND]).id == item_id
        settles(2)
        queue.extend_lease(EVALUATIONS_QUEUE, item_id, "w")
        queue.append_progress(EVALUATIONS_QUEUE, item_id, "w", {"step": 1}, number=0)
        settles(2)
        queue.start(EVALUATIONS_QUEUE, item_id, "w")
        settles(3)
        queue.finish(EVALUATIONS_QUEUE, item_id, "w", {"score": RolloutScore(1.0).to_json()})
        settles(4)
        assert set(settled) == {"e1"}
    finally:
        results.close()
        dispatcher.close()


def test_a_changed_rollout_of_a_deleted_scenario_is_no_failure(tmp_path: Path, caplog) -> None:
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(), live=LiveObjects(runtime=ArtifactRuntime()))
    try:
        evaluations = ArtifactEvaluations(dispatcher)
        with caplog.at_level(logging.ERROR, logger="reef.control.operations.artifact_evaluations"):
            evaluations.results.settle_changed("deleted-scenario", "evaluation-1")
        assert caplog.records == []
    finally:
        dispatcher.close()


@pytest.mark.integration
def test_a_runtime_that_serves_only_its_current_artifact_or_a_scenario_with_a_harness_is_refused(
    tmp_path: Path,
) -> None:
    async def run(dispatcher: Dispatcher, root: Path, message: str, *, harness: bool) -> None:
        evaluations = ArtifactEvaluations(dispatcher)
        async with TestClient(TestServer(create_app(dispatcher, tokens=TOKEN, evaluations=evaluations))) as client:
            artifact_id = await created(client, root, "add")
            body = {"artifact_id": artifact_id, "harness": {"name": "terminus-2"}, "component": "weights"}
            response = await client.post("/reef/scenarios/s/evaluations", headers=AUTH, json=body)
            assert response.status == 400 and message in await response.text(), await response.text()
            if harness:
                body = {"artifact_id": artifact_id, "model": "served", "component": "harness"}
                refused = await client.post("/reef/scenarios/s/evaluations", headers=AUTH, json=body)
                assert refused.status == 400 and "harness artifact yet" in await refused.text()

    current = ArtifactRuntime(any_artifact=False)
    asyncio.run(
        run(
            dispatcher_for(
                tmp_path / "current",
                WeightRecipe(),
                live=LiveObjects(runtime=current),
            ),
            tmp_path / "current",
            "serves only its current",
            harness=False,
        )
    )
    composed = CompositeRecipe(components={"weights": WeightRecipe(), HARNESS_COMPONENT: HarnessRecipe()})
    asyncio.run(
        run(
            dispatcher_for(
                tmp_path / "harness",
                composed,
                live=LiveObjects(runtime=ArtifactRuntime()),
            ),
            tmp_path / "harness",
            "artifacts hold no harness",
            harness=True,
        )
    )


@pytest.mark.integration
def test_a_started_evaluation_service_fails_the_evaluations_whose_rollouts_the_queue_lost(tmp_path: Path) -> None:
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(), live=LiveObjects(runtime=ArtifactRuntime()))

    async def run() -> None:
        evaluations = ArtifactEvaluations(dispatcher)
        async with TestClient(TestServer(create_app(dispatcher, tokens=TOKEN, evaluations=evaluations))) as client:
            artifact_id = await created(client, tmp_path, "add")
            request = ArtifactEvaluationRequest(artifact_id, "weights", harness={"name": "terminus-2"})
            with dispatcher.locked_scenario("s") as current:
                current.store.evaluations.create(
                    evaluation_id="stale", owner="a-stopped-service", request=request, created_at=time.time()
                )
            # A read changes nothing.
            answer = await (await client.get("/reef/scenarios/s/evaluations/stale", headers=AUTH)).json()
            assert answer["state"] == "running"
            restarted = ArtifactEvaluations(dispatcher)
            restarted.open()
            try:
                for _ in range(100):
                    answer = await (await client.get("/reef/scenarios/s/evaluations/stale", headers=AUTH)).json()
                    if answer["state"] == "failed":
                        break
                    await asyncio.sleep(0.05)
            finally:
                restarted.close()
            assert (answer["state"], answer["error"]) == ("failed", STOPPED_ERROR)

    asyncio.run(run())


@pytest.mark.integration
def test_reef_evaluates_each_new_artifact_with_the_scenarios_settings_through_the_queue(tmp_path: Path) -> None:
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(), live=LiveObjects(runtime=ArtifactRuntime()))
    new_artifacts = NewArtifacts()

    async def run() -> None:
        server, evaluations = evaluating_server(dispatcher, ModelCallingLab({"add": 1.0}), new_artifacts=new_artifacts)
        async with TestClient(server) as client:
            artifact_id = await created(client, tmp_path, "add")
            settings = {"harness": {"name": "terminus-2"}, "rollouts_per_task": 2}
            response = await client.post(
                "/reef/scenarios/s/update", headers=AUTH, json={"artifact_evaluation": settings}
            )
            assert response.status == 200, await response.text()
            assert (await response.json())["artifact_evaluation"]["rollouts_per_task"] == 2
            refused = await client.post(
                "/reef/scenarios/s/update", headers=AUTH, json={"artifact_evaluation": {"model": "served"}}
            )
            assert refused.status == 400 and "configure the harness" in await refused.text()

            # A commit that publishes a checkpoint is a new artifact: Reef adds its rollouts to the evaluations
            # queue, and the evaluation workers run them.
            with dispatcher.locked_scenario("s") as scenario:
                snapshot = next(
                    row["snapshot"] for row in scenario.artifact_rows() if row["artifact_id"] == artifact_id
                )
            commit = CommitRecord(
                scenario="s",
                commit_number=1,
                snapshot=snapshot,
                selection=ArtifactSelection({"weights": artifact_id}),
                checkpoint=True,
                algorithm_state={},
                high_water_sequence=0,
                high_water_offset=0,
            )
            new_artifacts.record_committed(commit)
            new_artifacts.record_committed(replace(commit, checkpoint=False))
            queued = await until(client, lambda listing: len(listing) == 1)
            assert queued[0]["request"]["artifact_id"] == artifact_id
            rollouts = evaluations.results.rollouts(queued[0]["evaluation_id"], "s")
            assert [EvaluationRollout.from_item(item).repeat for item in rollouts] == [0, 1]
            answer = await finished(client, queued[0]["evaluation_id"])
            assert answer["state"] == "finished" and answer["mean_score"] == 1.0
            assert len(await until(client, lambda listing: True)) == 1, "a step that published nothing is no artifact"

            # Without settings, a new artifact is not evaluated.
            off = await client.post("/reef/scenarios/s/update", headers=AUTH, json={"artifact_evaluation": None})
            assert (await off.json())["artifact_evaluation"] is None
            assert evaluations.evaluate_new(NewArtifact("s", artifact_id, None, "r")) is None

    asyncio.run(run())


async def settings_of(client: TestClient) -> dict[str, Any]:
    response = await client.get("/reef/scenarios/s/settings", headers=AUTH)
    assert response.status == 200, await response.text()
    return await response.json()


@pytest.mark.integration
def test_a_scenario_activation_overrides_the_deployment_rule_and_applies_again_after_a_restart(
    tmp_path: Path,
) -> None:
    recipe = WeightRecipe(activation=activation_reference({"module": "never"}))
    live = LiveObjects(runtime=ArtifactRuntime())
    (tmp_path / "initial").mkdir()
    # The in-memory repository keeps its registrations in its factory: a restart reuses it.
    factory = InMemoryRepositoryBackend.factory(directory=tmp_path / "repository")
    dispatcher = dispatcher_for(tmp_path, recipe, factory, live=live)

    async def run() -> None:
        async with TestClient(TestServer(create_app(dispatcher, tokens=TOKEN))) as client:
            active = await created(client, tmp_path, "add")
            assert await settings_of(client) == {
                "scenario": "s",
                "training_mode": "auto",
                "artifact_evaluation": None,
                "activation": None,
                "effective_activation": {"module": "never", "config": {}},
                "serving_config": {},
                "effective_serving_config": {"weights": {}},
                "served": {"weights": {active: 100}},
                "record2sample_config": {},
                "job_parameters": {"trajectory": {}, "task": {}},
            }
            rule = {"module": "always", "config": {"review_kinds": ["skill"]}}
            response = await client.post("/reef/scenarios/s/update", headers=AUTH, json={"activation": rule})
            assert response.status == 200, await response.text()
            assert await response.json() == {"scenario": "s", "activation": rule}
            settings = await settings_of(client)
            assert settings["activation"] == rule and settings["effective_activation"] == rule
            assert isinstance(dispatcher.registry.get_loaded("s").activation_rule, AlwaysActivate)

    asyncio.run(run())
    dispatcher.close()

    restarted = dispatcher_for(tmp_path, recipe, factory, live=live)

    async def after_restart() -> None:
        async with TestClient(TestServer(create_app(restarted, tokens=TOKEN))) as client:
            assert (await settings_of(client))["effective_activation"]["module"] == "always"
            assert isinstance(restarted.registry.get_loaded("s").activation_rule, AlwaysActivate)
            # null follows the deployment's rule again.
            response = await client.post("/reef/scenarios/s/update", headers=AUTH, json={"activation": None})
            assert (await response.json())["activation"] is None
            settings = await settings_of(client)
            assert settings["activation"] is None and settings["effective_activation"]["module"] == "never"
            assert isinstance(restarted.registry.get_loaded("s").activation_rule, NeverActivate)
            missing = await client.get("/reef/scenarios/other/settings", headers=AUTH)
            assert missing.status == 404

    asyncio.run(after_restart())
    restarted.close()


@pytest.mark.integration
def test_the_evaluated_activation_needs_artifact_evaluation_in_either_order_and_a_plugin_rule_is_refused(
    tmp_path: Path,
) -> None:
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(), live=LiveObjects(runtime=ArtifactRuntime()))
    evaluated = {"module": "evaluated", "config": {"min_mean_score": 0.5}}
    evaluation = {"harness": {"name": "terminus-2"}}

    async def update(client: TestClient, body: dict[str, Any]) -> tuple[int, str]:
        response = await client.post("/reef/scenarios/s/update", headers=AUTH, json=body)
        return response.status, await response.text()

    async def run() -> None:
        async with TestClient(TestServer(create_app(dispatcher, tokens=TOKEN))) as client:
            await created(client, tmp_path, "add")
            status, text = await update(client, {"activation": evaluated})
            assert status == 400 and "set artifact_evaluation" in text
            # The request is refused as a whole: a bad artifact_evaluation stores no activation either.
            status, _ = await update(client, {"activation": {"module": "never"}, "artifact_evaluation": {"x": 1}})
            assert status == 400
            assert (await settings_of(client))["activation"] is None

            # Both in one request, or artifact_evaluation first.
            assert (await update(client, {"activation": evaluated, "artifact_evaluation": evaluation}))[0] == 200
            assert (await update(client, {"activation": None, "artifact_evaluation": None}))[0] == 200
            assert (await update(client, {"artifact_evaluation": evaluation}))[0] == 200
            assert (await update(client, {"activation": evaluated}))[0] == 200
            assert isinstance(dispatcher.registry.get_loaded("s").activation_rule, EvaluatedActivation)

            status, text = await update(client, {"artifact_evaluation": None})
            assert status == 400 and "set artifact_evaluation" in text
            settings = await settings_of(client)
            assert settings["artifact_evaluation"]["harness"]["name"] == "terminus-2"
            assert settings["activation"] == settings["effective_activation"] == evaluated

            # A scenario names built-in rules only: the client does not choose which code the service imports.
            plugin = {"module": "tests.support.activation_rule_plugin:ComponentRule", "config": {"component": "w"}}
            for refused in (plugin, "never", {"module": "evaluated", "config": {}}):
                status, _ = await update(client, {"activation": refused})
                assert status == 400, refused
            assert (await settings_of(client))["activation"] == evaluated

    asyncio.run(run())


@pytest.mark.unit
def test_a_scenario_under_a_deployment_evaluated_rule_without_artifact_evaluation_loads_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    evaluated = activation_reference({"module": "evaluated", "config": {"min_mean_score": 0.5}})
    dispatcher = dispatcher_for(
        tmp_path,
        WeightRecipe(activation=evaluated),
        live=LiveObjects(runtime=ArtifactRuntime()),
    )
    with caplog.at_level("WARNING", logger="reef.control.scenario.registry"):
        dispatcher.get_or_create_scenario("s", allow_implicit_creation=True)
    assert "activates no new artifact" in caplog.text
    with pytest.raises(ValueError, match="set artifact_evaluation"):
        dispatcher.set_evaluation_and_activation("s", artifact_evaluation=None, activation=None)
    dispatcher.close()


async def until(client: TestClient, done) -> list[dict[str, Any]]:
    for _ in range(200):
        listing = (await (await client.get("/reef/scenarios/s/evaluations", headers=AUTH)).json())["evaluations"]
        if done(listing):
            return listing
        await asyncio.sleep(0.05)
    raise AssertionError("the evaluations did not reach the expected state")


@pytest.mark.unit
def test_a_commit_that_publishes_a_checkpoint_reports_a_new_artifact(tmp_path: Path) -> None:
    new_artifacts = NewArtifacts()
    storage = ObservedScenarioStorage(SQLiteScenarioStorage(tmp_path / "records"), new_artifacts)
    try:
        session = storage.open("s")
        step = {"algorithm_state": {}, "high_water_sequence": 0, "high_water_offset": 0, "scenario": "s"}
        published = CommitRecord(
            commit_number=1,
            snapshot=SnapshotRef("a1", None),
            checkpoint=True,
            selection=ArtifactSelection({"model": "a1"}),
            **step,
        )
        kept = CommitRecord(commit_number=2, snapshot=SnapshotRef("a1", None), checkpoint=False, **step)
        session.append_commit(expected_commit_number=0, commit=published)
        session.append_commit(expected_commit_number=1, commit=kept)
        assert new_artifacts.queue.get_nowait() == NewArtifact("s", "a1", None, "a1")
        assert new_artifacts.queue.empty(), "a step that publishes nothing reports no artifact"
        # An activation, even one that publishes a composed snapshot, and a new training start make no artifact.
        operation = {"high_water_sequence": 0, "high_water_offset": 0, "scenario": "s", "algorithm_state": None}
        activation = CommitRecord(
            commit_number=3,
            snapshot=SnapshotRef("a2", "a1"),
            checkpoint=False,
            operation="activate",
            component="harness",
            components={"harness": "c1", "weights": "c0"},
            **operation,
        )
        start = CommitRecord(
            commit_number=4, snapshot=SnapshotRef("a1", None), checkpoint=False, operation="train_from", **operation
        )
        session.append_commit(expected_commit_number=2, commit=activation)
        session.append_commit(expected_commit_number=3, commit=start)
        assert new_artifacts.queue.empty(), "an activation or a new training start reports no artifact"
        session.close()
    finally:
        storage.close()
    assert new_artifacts.queue.get_nowait() is None, "the observer closes with the storage"


@pytest.mark.parametrize(
    ("component", "selected", "expected"),
    [
        (None, {"weights": "artifact:weights"}, [NewArtifact("s", "artifact:weights", None, "storage:snapshot")]),
        (
            "harness",
            {"weights": "artifact:carried", "harness": "artifact:new"},
            [NewArtifact("s", "artifact:new", "harness", "storage:snapshot")],
        ),
        (
            None,
            {"weights": "artifact:weights", "harness": "artifact:harness"},
            [
                NewArtifact("s", "artifact:weights", "weights", "storage:snapshot"),
                NewArtifact("s", "artifact:harness", "harness", "storage:snapshot"),
            ],
        ),
    ],
)
def test_automatic_evaluation_uses_published_artifact_ids(component, selected, expected) -> None:
    observer = NewArtifacts()
    commit = CommitRecord(
        scenario="s",
        commit_number=1,
        snapshot=SnapshotRef("storage:snapshot", None),
        checkpoint=True,
        algorithm_state={},
        high_water_sequence=0,
        high_water_offset=0,
        component=component,
        selection=ArtifactSelection(selected),
    )
    observer.record_committed(commit)
    assert [observer.queue.get_nowait() for _ in expected] == expected
    assert observer.queue.empty()


def test_automatic_evaluation_skips_a_component_that_an_earlier_commit_made(monkeypatch) -> None:
    catalog = (
        {"artifact_id": "artifact:new", "snapshot": SnapshotRef("storage:2", "storage:1")},
        {"artifact_id": "artifact:carried", "snapshot": SnapshotRef("storage:1", None)},
    )

    class Catalog:
        def artifact_rows(self) -> tuple[dict[str, object], ...]:
            return catalog

    class Dispatcher:
        queue = None

        def stored_settings(self, scenario: str) -> SimpleNamespace:
            return SimpleNamespace(artifact_evaluation={"settings": "stored"})

        @contextlib.contextmanager
        def locked_scenario(self, scenario: str) -> Iterator[Catalog]:
            yield Catalog()

    evaluations = ArtifactEvaluations(Dispatcher())  # type: ignore[arg-type]
    accepted: list[str] = []
    monkeypatch.setattr(
        ArtifactEvaluationSettings, "from_json", staticmethod(lambda stored: SimpleNamespace(component=None))
    )
    monkeypatch.setattr(
        evaluations, "accept", lambda scenario, artifact_id, settings, state: accepted.append(artifact_id)
    )
    evaluations.evaluate_new(NewArtifact("s", "artifact:carried", None, "storage:2"))
    evaluations.evaluate_new(NewArtifact("s", "artifact:new", None, "storage:2"))
    assert accepted == ["artifact:new"]
