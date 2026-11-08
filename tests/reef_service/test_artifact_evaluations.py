"""A client evaluates an artifact of a snapshot: Reef runs the rollouts and stores their policy evaluation."""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import tarfile
import time
import urllib.request
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from urllib.parse import urlencode

import pytest
from aiohttp.test_utils import TestClient, TestServer
from tests.support.task_service_stubs import task_for
from tests.support.trajectory_uploads import dispatcher_for

from reef.artifact import InMemoryRepositoryBackend
from reef.artifact.artifact import Artifact
from reef.artifact.components import ComponentSpec, ScenarioComponents
from reef.core.artifact_selection import ArtifactSelection
from reef.core.errors import ReefError
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
from reef.dispatcher import Dispatcher
from reef.evaluation.activation import AlwaysActivate, EvaluatedActivation, NeverActivate, activation_reference
from reef.evaluation.rollout_worker import (
    EVALUATION_ROLLOUT_KIND,
    EVALUATIONS_QUEUE,
    EvaluationRollout,
    EvaluationRolloutWorker,
    LocalCandidateEndpoints,
)
from reef.harness.adapters import get_adapter
from reef.harness.harbor.agent import TreeAgent
from reef.harness.harbor.rollouts import RolloutRow, TaskLab
from reef.harness.sync import HARNESS_COMPONENT, HarnessFiles
from reef.harness.tree import HarnessTree
from reef.inference.interfaces import InferenceHandler, InferenceRuntime
from reef.recipe import Recipe
from reef.scenario.snapshots import ScenarioSnapshots
from reef.service.app import create_app
from reef.service.artifact_evaluations import (
    STOPPED_ERROR,
    ArtifactEvaluations,
    NewArtifact,
    NewArtifacts,
    attempt_trajectories,
)
from reef.service.data import DataAccess
from reef.storage.commits import CommitRecord
from reef.storage.data import NewItem
from reef.storage.observer import ObservedScenarioStorage
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.storage.work_queue import WorkItem
from reef.train.trainer import ComponentTrainer

TOKEN = "artifact-evaluation-test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


class ArtifactHandler(InferenceHandler):
    """Answers every chat call, and records the artifact that it was asked to answer from."""

    def __init__(self) -> None:
        self.releases: list[str] = []
        self.models: list[str] = []

    async def inference(self, artifact: Artifact, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.releases.append(artifact.artifact_id)
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

    def serve_artifact(self, artifact: Artifact, *, scenario: str) -> ServedArtifact:
        self.served.append(artifact.artifact_id)
        self.serve_scenarios.append(scenario)
        if self.serve_error is not None:
            raise ValueError(self.serve_error)
        return ServedArtifact(artifact.artifact_id, self.base_url, f"engine:{artifact.artifact_id}")

    def activate_checkpoint(self, artifact, *, scenario=None):
        return "load-0"


@dataclass(frozen=True, kw_only=True)
class WeightRecipe(Recipe):
    """A record-only recipe that serves weights on its runtime."""

    harness: bool = False

    def build_components(self, scenario: str) -> ScenarioComponents:
        components = {"weights": ComponentSpec(kind=self.runtime.served_weight_kind)}
        if self.harness:
            components[HARNESS_COMPONENT] = ComponentSpec(files=HarnessFiles(get_adapter("pi")), kind="harness_plugin")
        return ScenarioComponents(components)

    def build_trainers(
        self,
        scenario,
        records,
        *,
        components,
        algorithm_states,
        experiment_logger=None,
        task_storage=None,
        artifacts=None,
        samples=None,
        root=None,
    ):
        return tuple(
            ComponentTrainer(
                name,
                self.build(
                    scenario,
                    records,
                    algorithm_state=algorithm_states.get(name),
                    task_storage=task_storage,
                    samples=samples,
                    root=root,
                ),
            )
            for name in components.names
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
    for _ in range(200):
        answer = await (await client.get(f"/reef/scenarios/s/evaluations/{evaluation_id}", headers=AUTH)).json()
        if answer["state"] not in ("queued", "running"):
            return answer
        await asyncio.sleep(0.05)
    raise AssertionError("the evaluation did not end")


async def created(client: TestClient, tmp_path: Path, *names: str) -> str:
    """Create the scenario with eval tasks ``names``, and return its active artifact."""
    await client.post("/reef/scenarios", headers=AUTH, json={"name": "s"})
    metadata = urlencode({"metadata": json.dumps({"scenarios": {"s": "eval"}})})
    upload = await client.post(
        f"/reef/namespaces/s/data/tasks?{metadata}",
        headers={**AUTH, "Content-Type": "application/gzip"},
        data=archive(tmp_path, *names),
    )
    assert upload.status == 201, await upload.text()
    artifacts = await (await client.get("/reef/scenarios/s/artifacts", headers=AUTH)).json()
    return next(row["artifact_id"] for row in artifacts["artifacts"] if row["active"])


@pytest.mark.integration
def test_a_model_artifact_is_evaluated_with_the_configured_harness_on_the_release_it_names(tmp_path: Path) -> None:
    runtime = ArtifactRuntime()
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(runtime=runtime))
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
            assert runtime.handler.releases == [artifact_id] * 4, "every call samples the evaluated release"
            assert runtime.handler.models == [f"engine:{artifact_id}"] * 4, "every call names the engine's model"
            assert runtime.served == [artifact_id] * 6, "inference serves the candidate before each trial"
            assert (
                runtime.serve_scenarios == ["s"] * 6 and runtime.applied.count("s") >= 6
            ), "inference serves the candidate with the scenario's serving config"
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
                ("broken", 0, "failed"),
                ("broken", 1, "failed"),
                ("sort", 0, 0.0),
                ("sort", 1, 0.0),
            ]
            assert {rollout["state"] for rollout in rollouts} == {"finished"}
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


@pytest.mark.integration
def test_an_evaluation_keeps_at_most_its_max_concurrent_rollouts_in_the_queue(tmp_path: Path) -> None:
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(runtime=ArtifactRuntime()))
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
            scored(first, 1.0)
            still_queued, third = await queued_rollouts(2)
            assert still_queued.id == second.id and third.id.endswith(":00001:00000")
            for item in (second, third):
                assert queue.take(EVALUATIONS_QUEUE, "w", [EVALUATION_ROLLOUT_KIND]).id == item.id
                scored(item, 0.0)
            [last] = await queued_rollouts(1)
            assert queue.take(EVALUATIONS_QUEUE, "w", [EVALUATION_ROLLOUT_KIND]).id == last.id
            scored(last, 1.0)

            answer = await finished(client, accepted["evaluation_id"])
            assert answer["state"] == "finished", answer
            assert answer["policy_evaluation"]["scores"] == [
                {"task": "add", "scores": [1.0, 0.0]},
                {"task": "sort", "scores": [0.0, 1.0]},
            ]

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
        "scored": True,
    }
    # Attempt 2 has no 200 call, so it has no trajectory.
    assert (earlier["extra"]["reef"]["evaluation"]["attempt"], earlier["extra"]["reef"]["evaluation"]["scored"]) == (
        1,
        False,
    )
    records = scored["extra"]["reef"]["records"]
    assert [record["agent_record_id"] for record in records] == ["e1:2:0:3:0", "e1:2:0:3:2"]
    assert records[0]["created_at"] == 30.0 and records[0]["payload"]["response"]["choices"][0]["message"]["content"]
    # The second call repeats the first exchange, so the stitched steps hold each message once.
    assert [step["message"] for step in scored["steps"]] == ["solve add", "ls", "a.txt", "cat a.txt"]
    running = replace(item, state="running", ended_at=None, result=None, worker_id="w", lease_expires_at=9.0)
    assert [
        trajectory["extra"]["reef"]["reward"] for trajectory in attempt_trajectories("s", "e1", running, calls)
    ] == [
        None,
        None,
    ]


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
        self.agent = TreeAgent(self.logs, **kwargs)
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
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(runtime=runtime))
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
            body = {"artifact_id": artifact_id, "harness": {"item": item_id}}
            response = await client.post("/reef/scenarios/s/evaluations", headers=AUTH, json=body)
            assert response.status == 202, await response.text()
            answer = await finished(client, (await response.json())["evaluation_id"])
            assert activated == [], "a score with a harness from an item activates nothing"

        assert answer["state"] == "finished", answer
        assert answer["request"]["harness"] == {"item": item_id}
        evaluation = answer["policy_evaluation"]
        assert evaluation["harness"]["base"] == "pi"
        assert evaluation["scores"] == [{"task": "add", "scores": [1.0]}]
        assert runtime.handler.releases == [artifact_id]
        rendered = "".join(lab.rendered.values())
        assert "Call {model} once." in rendered, "the runner fills no placeholder in the tree's own text"
        assert "http://127.0.0.1:" in rendered, "the rendered binding names the address of the capture proxy"
        assert "127.0.0.1" not in json.dumps(evaluation["harness"]["plugin"]), "the result keeps no endpoint"

    asyncio.run(run())


@pytest.mark.integration
def test_a_tree_in_a_remote_environment_bridges_its_container_to_the_proxy(tmp_path: Path) -> None:
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(runtime=ArtifactRuntime()))
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
    assert lab.overrides["environment"] == {"type": "modal"}
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
        ({"adapter": "hermes"}, "cannot run in a task's container"),
        ({"adapter": "claude"}, "cannot call this runtime's API dialect"),
    ],
)
def test_an_artifact_item_that_cannot_run_is_refused_before_any_rollout(
    tmp_path: Path, options: dict[str, Any], reason: str
) -> None:
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(runtime=ArtifactRuntime()))

    async def run() -> None:
        server, _ = evaluating_server(dispatcher, TreeCallingLab(tmp_path / "logs"))
        async with TestClient(server) as client:
            artifact_id = await created(client, tmp_path, "add")
            await client.post("/reef/scenarios", headers=AUTH, json={"name": "other", "owner": "s"})
            uri = options.pop("uri", harness_tree(tmp_path / "tree"))
            environment = options.pop("environment", "docker")
            body = {
                "artifact_id": artifact_id,
                "harness": {"item": tree_item(dispatcher, uri, **options)},
                "environment": environment,
            }
            response = await client.post("/reef/scenarios/s/evaluations", headers=AUTH, json=body)
            assert response.status == 400 and reason in await response.text()

    asyncio.run(run())


@pytest.mark.integration
def test_an_item_of_another_owner_a_task_with_tools_and_automatic_settings_are_refused(
    tmp_path: Path, monkeypatch
) -> None:
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(runtime=ArtifactRuntime()))

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

            monkeypatch.setattr("reef.service.artifact_evaluations.task_agent_tools", lambda task: ("mcp_servers",))
            body = {"artifact_id": artifact_id, "harness": {"item": item_id}}
            response = await client.post("/reef/scenarios/s/evaluations", headers=AUTH, json=body)
            assert response.status == 400 and "mcp_servers" in await response.text()

    asyncio.run(run())


class NoCallLab(TaskLab):
    """A trial whose verifier writes a reward, though the agent reached no model: the endpoint went away."""

    async def run(self, task_path, agent, *, key, tags, overrides) -> RolloutRow:
        return RolloutRow(rewards={"reward": 0.0})

    async def build_environment(self, task_path, *, overrides) -> str:
        return ""


@pytest.mark.integration
def test_a_rollout_that_reached_no_model_fails_though_its_verifier_wrote_a_reward(tmp_path: Path) -> None:
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(runtime=ArtifactRuntime()))

    async def run() -> None:
        server, _ = evaluating_server(dispatcher, NoCallLab())
        async with TestClient(server) as client:
            artifact_id = await created(client, tmp_path, "add")
            body = {"artifact_id": artifact_id, "harness": {"name": "terminus-2"}}
            response = await client.post("/reef/scenarios/s/evaluations", headers=AUTH, json=body)
            assert response.status == 202, await response.text()
            answer = await finished(client, (await response.json())["evaluation_id"])

        assert answer["policy_evaluation"]["scores"] == [{"task": "add", "scores": ["failed"]}]
        assert answer["mean_score"] is None

    asyncio.run(run())


@pytest.mark.integration
def test_a_candidate_that_inference_cannot_serve_fails_its_rollouts_and_scores_nothing(tmp_path: Path) -> None:
    runtime = ArtifactRuntime()
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(runtime=runtime))
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
        assert runtime.handler.releases == [], "no trial ran"

    asyncio.run(run())


@pytest.mark.integration
def test_an_artifact_that_is_not_published_is_refused_before_any_rollout(tmp_path: Path, monkeypatch) -> None:
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(runtime=ArtifactRuntime()))

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
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(runtime=ArtifactRuntime()))

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


@pytest.mark.integration
def test_a_runtime_that_serves_only_its_current_release_or_a_scenario_with_a_harness_is_refused(
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
            dispatcher_for(tmp_path / "current", WeightRecipe(runtime=current)),
            tmp_path / "current",
            "serves only its current",
            harness=False,
        )
    )
    composed = WeightRecipe(runtime=ArtifactRuntime(), harness=True)
    asyncio.run(
        run(
            dispatcher_for(tmp_path / "harness", composed),
            tmp_path / "harness",
            "artifacts hold no harness",
            harness=True,
        )
    )


@pytest.mark.integration
def test_a_started_evaluation_service_fails_the_evaluations_whose_rollouts_the_queue_lost(tmp_path: Path) -> None:
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(runtime=ArtifactRuntime()))

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
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(runtime=ArtifactRuntime()))
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
                step=1,
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
    recipe = WeightRecipe(runtime=ArtifactRuntime(), activation=activation_reference({"module": "never"}))
    (tmp_path / "initial").mkdir()
    # The in-memory repository keeps its registrations in its factory: a restart reuses it.
    factory = InMemoryRepositoryBackend.factory(directory=tmp_path / "repository")
    dispatcher = dispatcher_for(tmp_path, recipe, factory)

    async def run() -> None:
        async with TestClient(TestServer(create_app(dispatcher, tokens=TOKEN))) as client:
            await created(client, tmp_path, "add")
            assert await settings_of(client) == {
                "scenario": "s",
                "training_mode": "auto",
                "artifact_evaluation": None,
                "activation": None,
                "effective_activation": {"module": "never", "config": {}},
                "serving_config": {},
                "effective_serving_config": {"weights": {}},
                "traffic_weights": {},
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

    restarted = dispatcher_for(tmp_path, recipe, factory)

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
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(runtime=ArtifactRuntime()))
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
            plugin = {"module": "tests.support._activation_rule_plugin:ComponentRule", "config": {"component": "w"}}
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
    dispatcher = dispatcher_for(tmp_path, WeightRecipe(runtime=ArtifactRuntime(), activation=evaluated))
    with caplog.at_level("WARNING", logger="reef.scenario.registry"):
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
def test_a_committed_step_that_publishes_a_checkpoint_reports_a_new_artifact(tmp_path: Path) -> None:
    new_artifacts = NewArtifacts()
    storage = ObservedScenarioStorage(SQLiteScenarioStorage(tmp_path / "records"), new_artifacts)
    try:
        session = storage.open("s")
        step = {"algorithm_state": {}, "high_water_sequence": 0, "high_water_offset": 0, "scenario": "s"}
        published = CommitRecord(
            step=1,
            snapshot=SnapshotRef("a1", None),
            checkpoint=True,
            selection=ArtifactSelection({"model": "a1"}),
            **step,
        )
        kept = CommitRecord(step=2, snapshot=SnapshotRef("a1", None), checkpoint=False, **step)
        session.commit_step(expected_step=0, commit=published)
        session.commit_step(expected_step=1, commit=kept)
        assert new_artifacts.queue.get_nowait() == NewArtifact("s", "a1", None, "a1")
        assert new_artifacts.queue.empty(), "a step that publishes nothing reports no artifact"
        # An activation, even one that publishes a composed snapshot, and a new training start make no artifact.
        operation = {"high_water_sequence": 0, "high_water_offset": 0, "scenario": "s", "algorithm_state": None}
        activation = CommitRecord(
            step=3,
            snapshot=SnapshotRef("a2", "a1"),
            checkpoint=False,
            operation="activate",
            component="harness",
            components={"harness": "c1", "weights": "c0"},
            **operation,
        )
        start = CommitRecord(
            step=4, snapshot=SnapshotRef("a1", None), checkpoint=False, operation="train_from", **operation
        )
        session.commit_step(expected_step=2, commit=activation)
        session.commit_step(expected_step=3, commit=start)
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
        step=1,
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
