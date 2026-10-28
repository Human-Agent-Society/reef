"""A client evaluates an artifact of a release: Reef runs the rollouts and stores their policy evaluation."""

from __future__ import annotations

import asyncio
import io
import json
import tarfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from aiohttp.test_utils import TestClient, TestServer
from tests.support.task_service_stubs import task_for

from reef.artifact import InMemoryRepositoryBackend
from reef.artifact.artifact import Artifact
from reef.core.evaluation import ArtifactEvaluationRequest, PolicyModel
from reef.core.tasks import write_harbor_task
from reef.dispatcher import Dispatcher
from reef.harness.client.tasks import RolloutRow, TaskLab
from reef.recipe import Recipe
from reef.runtime.interfaces import InferenceHandler, InferenceRuntime
from reef.service.app import create_app
from reef.service.artifact_evaluations import STOPPED_ERROR, ArtifactEvaluations
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.surface import Surface, create_harness_surface, create_weight_surface
from reef.train.trainer import ComponentTrainer

TOKEN = "artifact-evaluation-test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


class ReleaseHandler(InferenceHandler):
    """Answers every chat call with the release that it was asked to answer from."""

    def __init__(self) -> None:
        self.releases: list[str] = []

    async def inference(self, artifact: Artifact, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.releases.append(artifact.ref.release_id)
        return {"choices": [{"message": {"role": "assistant", "content": "ls"}, "finish_reason": "stop"}]}


class ReleaseRuntime(InferenceRuntime):
    """A runtime that answers each call from the release that the call names, as a sampler for each checkpoint."""

    def __init__(self, *, serves_any_release: bool = True) -> None:
        super().__init__(base_url="http://runtime.invalid")
        self.handler = ReleaseHandler()
        self.serves_any_release = serves_any_release

    @property
    def inference_handler(self) -> InferenceHandler:
        return self.handler

    @property
    def served_weight_kind(self):
        return "model_adapter"

    def release_model(self, artifact: Artifact) -> PolicyModel | None:
        return PolicyModel("base", f"sampler:{artifact.ref.release_id}") if self.serves_any_release else None

    def activate_checkpoint(self, artifact, *, scenario=None):
        return "load-0"

    def current_runtime_load_id(self) -> str:
        return "load-0"


@dataclass(frozen=True, kw_only=True)
class WeightRecipe(Recipe):
    """A record-only recipe that serves weights on its runtime."""

    harness: bool = False

    def build_surface(self, scenario: str) -> Surface:
        weights = create_weight_surface(kind=self.runtime.served_weight_kind)
        if not self.harness:
            return weights
        return Surface({**weights.components, **create_harness_surface().components})

    def build_trainers(
        self, scenario, records, *, surface, algorithm_states, experiment_logger=None, task_storage=None
    ):
        return tuple(
            ComponentTrainer(
                name,
                self.build(scenario, records, algorithm_state=algorithm_states.get(name), task_storage=task_storage),
            )
            for name in surface.names
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


def dispatcher_for(tmp_path: Path, recipe: Recipe) -> Dispatcher:
    initial = tmp_path / "initial"
    initial.mkdir(parents=True, exist_ok=True)
    return Dispatcher(
        recipe,
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
    )


def archive(root: Path, *names: str) -> bytes:
    """A tar.gz of one Harbor task directory for each name."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for port, name in enumerate(names, start=9000):
            tar.add(write_harbor_task(task_for(name, "record-1", port), root / str(port)), arcname=name)
    return buffer.getvalue()


async def finished(client: TestClient, evaluation_id: str) -> dict[str, Any]:
    for _ in range(200):
        answer = await (await client.get(f"/reef/scenarios/s/evaluations/{evaluation_id}", headers=AUTH)).json()
        if answer["state"] != "running":
            return answer
        await asyncio.sleep(0.05)
    raise AssertionError("the evaluation did not end")


async def created(client: TestClient, tmp_path: Path, *names: str) -> str:
    """Create the scenario with eval tasks ``names``, and return its current release."""
    upload = await client.post(
        "/reef/scenarios/s/tasks?split=eval",
        headers={**AUTH, "Content-Type": "application/gzip"},
        data=archive(tmp_path, *names),
    )
    assert upload.status == 200, await upload.text()
    releases = await (await client.get("/reef/scenarios/s/releases", headers=AUTH)).json()
    return releases["releases"][0]["release_id"]


@pytest.mark.integration
def test_a_model_artifact_is_evaluated_with_the_configured_harness_on_the_release_it_names(tmp_path: Path) -> None:
    runtime = ReleaseRuntime()
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(runtime=runtime))
    lab = ModelCallingLab({"add": 1.0, "sort": 0.0}, failing=frozenset({"broken"}))

    async def run() -> None:
        app = create_app(dispatcher, tokens=TOKEN, evaluations=ArtifactEvaluations(dispatcher, lab=lab))
        async with TestClient(TestServer(app)) as client:
            release_id = await created(client, tmp_path, "add", "broken", "sort")
            body = {"release_id": release_id, "harness": {"name": "terminus-2"}, "rollouts_per_task": 2}
            response = await client.post("/reef/scenarios/s/evaluations", headers=AUTH, json=body)
            assert response.status == 202, await response.text()
            accepted = await response.json()
            assert accepted["state"] == "running" and accepted["request"]["component"] == "weights"

            answer = await finished(client, accepted["evaluation_id"])
            assert answer["state"] == "finished", answer
            assert answer["policy_evaluation"] == {
                "harness": {"base": "terminus-2", "plugin": {}},
                "model": {"name": "base", "adapter": f"sampler:{release_id}"},
                "scores": [
                    {"task": "add", "scores": [1.0, 1.0]},
                    {"task": "broken", "scores": ["failed", "failed"]},
                    {"task": "sort", "scores": [0.0, 0.0]},
                ],
            }
            assert answer["mean_score"] == 0.5
            assert runtime.handler.releases == [release_id] * 4, "every call samples the evaluated release"
            listing = await (await client.get("/reef/scenarios/s/evaluations", headers=AUTH)).json()
            assert [item["evaluation_id"] for item in listing["evaluations"]] == [accepted["evaluation_id"]]
            # The evaluation keeps no record and changes nothing the scenario serves.
            records = await (await client.get("/reef/scenarios/s/records", headers=AUTH)).json()
            assert records["records"] == []

    asyncio.run(run())


@pytest.mark.integration
def test_a_request_that_the_scenario_cannot_run_is_refused_before_any_rollout(tmp_path: Path) -> None:
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(runtime=ReleaseRuntime()))
    lab = ModelCallingLab({"add": 1.0})

    async def run() -> None:
        app = create_app(dispatcher, tokens=TOKEN, evaluations=ArtifactEvaluations(dispatcher, lab=lab))
        async with TestClient(TestServer(app)) as client:
            release_id = await created(client, tmp_path, "add")
            agent = {"name": "terminus-2"}
            for body, status, message in (
                ({"release_id": release_id, "model": "served"}, 400, "configure the harness"),
                ({"release_id": release_id, "harness": agent, "task_names": ["sort"]}, 400, "holds no tasks"),
                ({"release_id": release_id, "harness": agent, "component": "harness"}, 400, "no component"),
                ({"release_id": release_id, "harness": agent, "rollouts": 2}, 400, "takes no"),
                ({"release_id": "missing", "harness": agent}, 404, "missing"),
            ):
                response = await client.post("/reef/scenarios/s/evaluations", headers=AUTH, json=body)
                assert (response.status, message in await response.text()) == (status, True), await response.text()
            unknown = await client.post("/reef/scenarios/other/evaluations", headers=AUTH, json={})
            assert unknown.status == 404
            assert (await client.get("/reef/scenarios/s/evaluations/missing", headers=AUTH)).status == 404
            assert (await client.post("/reef/scenarios/s/evaluations", json={})).status == 401

    asyncio.run(run())


@pytest.mark.integration
def test_a_runtime_that_serves_only_its_current_release_or_a_scenario_with_a_harness_is_refused(
    tmp_path: Path,
) -> None:
    async def run(dispatcher: Dispatcher, root: Path, message: str, *, harness: bool) -> None:
        evaluations = ArtifactEvaluations(dispatcher, lab=ModelCallingLab({}))
        async with TestClient(TestServer(create_app(dispatcher, tokens=TOKEN, evaluations=evaluations))) as client:
            release_id = await created(client, root, "add")
            body = {"release_id": release_id, "harness": {"name": "terminus-2"}, "component": "weights"}
            response = await client.post("/reef/scenarios/s/evaluations", headers=AUTH, json=body)
            assert response.status == 400 and message in await response.text(), await response.text()
            if harness:
                body = {"release_id": release_id, "model": "served", "component": "harness"}
                refused = await client.post("/reef/scenarios/s/evaluations", headers=AUTH, json=body)
                assert refused.status == 400 and "harness artifact yet" in await refused.text()

    current = ReleaseRuntime(serves_any_release=False)
    asyncio.run(
        run(
            dispatcher_for(tmp_path / "current", WeightRecipe(runtime=current)),
            tmp_path / "current",
            "serves only its current",
            harness=False,
        )
    )
    composed = WeightRecipe(runtime=ReleaseRuntime(), harness=True)
    asyncio.run(
        run(dispatcher_for(tmp_path / "harness", composed), tmp_path / "harness", "hold no harness", harness=True)
    )


@pytest.mark.integration
def test_an_evaluation_that_another_service_left_running_reads_as_failed_and_the_limit_is_two(tmp_path: Path) -> None:
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(runtime=ReleaseRuntime()))

    async def run() -> None:
        evaluations = ArtifactEvaluations(dispatcher, lab=ModelCallingLab({"add": 1.0}))
        async with TestClient(TestServer(create_app(dispatcher, tokens=TOKEN, evaluations=evaluations))) as client:
            release_id = await created(client, tmp_path, "add")
            request = ArtifactEvaluationRequest(release_id, "weights", harness={"name": "terminus-2"})
            with dispatcher.locked_scenario("s") as current:
                current.store.evaluations.create(
                    evaluation_id="stale", owner="a-stopped-service", request=request, created_at=time.time()
                )
            answer = await (await client.get("/reef/scenarios/s/evaluations/stale", headers=AUTH)).json()
            assert (answer["state"], answer["error"]) == ("failed", STOPPED_ERROR)

            evaluations.running.update({"one", "two"})
            body = {"release_id": release_id, "harness": {"name": "terminus-2"}}
            busy = await client.post("/reef/scenarios/s/evaluations", headers=AUTH, json=body)
            assert busy.status == 429

    asyncio.run(run())
