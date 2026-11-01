"""Read-channel guarantees of the harness routes: the release catalog with each step's evaluation metrics,
version-addressed pulls, served files that are byte for byte the evaluated source tree rendered for the
adapter, the source form, and the routes the wrapper used that are gone. Hermetic: tasks run a fake pi
binary through the real adapter path, and the reads go through the real HTTP app."""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from threading import Event

import pytest
from aiohttp.test_utils import TestClient, TestServer
from tests.support.harness_render import descriptor_data, write_descriptor
from tests.support.harness_service import EchoInference, harness_dispatcher, job_backend
from tests.support.harness_steps import changed_rules, harness_recipe

from reef.core.evaluation import EvaluationResult, UpdateCandidate
from reef.dispatcher import Dispatcher
from reef.harness.adapters import get_adapter
from reef.harness.adapters.descriptor import DescriptorError, load_descriptor
from reef.harness.tasks.run import TaskResult
from reef.harness.tree import HarnessTree, TreePatch, render_tree
from reef.recipe.harness import HarnessEvolutionRecipe
from reef.service.app import create_app
from reef.train.pipeline import PipelineConfig

#: Each patch strictly raises the marker count, so every evaluated step wins and publishes.
PATCHES = (TreePatch(changed_rules("marker rules\n")), TreePatch(changed_rules("marker marker rules\n")))
TREE_V1 = HarnessTree(changed_rules("marker rules\n"))
TREE_V2 = HarnessTree(changed_rules("marker marker rules\n"))
#: The default seed: no files, so nothing is served before the first publish.
EMPTY_SEED = HarnessTree({})
PI = get_adapter("pi")
HEADERS = {"x-reef-scenario": "delivery"}
UPDATE_TIMEOUT_S = 5.0
UPDATE_POLL_S = 0.01


def evaluate(task: str, result: TaskResult) -> float:
    return float(result.trajectory[-1]["rules"].count("marker"))


def channel_dispatcher(
    tmp_path: Path,
    patches,
    *,
    seed: HarnessTree = EMPTY_SEED,
    inference_stages: bool = False,
    job_samples: int = 1,
) -> Dispatcher:
    """A ``delivery`` scenario whose method makes ``patches`` in order, then nothing; its root release is
    ``seed``, and each step takes ``job_samples`` traces."""
    candidates = iter(patches)
    stages: dict = (
        {
            "record2sample": {"module": "reef.record2sample.traces:InferenceSamples"},
        }
        if inference_stages
        else {}
    )
    stages["sample2job"] = {"config": {"job_samples": job_samples}}
    recipe = replace(
        harness_recipe(
            tmp_path,
            lambda workspace, services: next(candidates, None),
            seed=seed,
            score=evaluate,
        ),
        pipeline_stages=PipelineConfig.from_setting(stages, spec=HarnessEvolutionRecipe.pipeline_spec(), environ={}),
    )
    dispatcher = harness_dispatcher(tmp_path, recipe)
    dispatcher.get_or_create_scenario("delivery")
    return dispatcher


@asynccontextmanager
async def served(dispatcher: Dispatcher) -> AsyncIterator[TestClient]:
    """A started test client for the real app over ``dispatcher``, closed on exit."""
    client = TestClient(TestServer(create_app(dispatcher, inference_handler=EchoInference())))
    await client.start_server()
    try:
        yield client
    finally:
        await client.close()


async def scored_exchange(client: TestClient) -> None:
    """One inference and its failing report: a batch of one, which wakes one step."""
    body = {"messages": [{"role": "user", "content": "hi"}]}
    response = await client.post("/v1/chat/completions", headers=HEADERS, json=body)
    assert response.status == 200
    receipt = response.headers["x-reef-agent-record-id"]
    response = await client.post("/reef/report", headers=HEADERS, json={"score": 0.0, "references": [receipt]})
    assert response.status == 200


async def committed_step(client: TestClient) -> dict:
    """The scenario's latest committed step, once one is committed: a commit of the step's job after its submission."""
    deadline = asyncio.get_running_loop().time() + UPDATE_TIMEOUT_S
    while True:
        scenario = (await (await client.get("/reef/status")).json())["scenarios"]["delivery"]
        committed = scenario["last_committed_step"]
        if committed is not None and committed["metrics"]["training_job"]["event"] != "submitted":
            return scenario
        if asyncio.get_running_loop().time() >= deadline:
            pytest.fail("the step did not commit")
        await asyncio.sleep(UPDATE_POLL_S)


async def evaluated_step(client: TestClient) -> dict:
    """Drive one evaluated step through the wire; the manifest of the release it published."""
    response = await client.get("/reef/harness", headers=HEADERS)
    previous = (await response.json())["release_id"] if response.status == 200 else None
    await scored_exchange(client)
    deadline = asyncio.get_running_loop().time() + UPDATE_TIMEOUT_S
    while True:
        response = await client.get("/reef/harness", headers=HEADERS)
        if response.status == 200:
            manifest = await response.json()
            if manifest["release_id"] != previous:
                assert manifest["evaluation"]["selected"] is True
                return manifest
        else:
            assert response.status == 404
            await response.read()
        if asyncio.get_running_loop().time() >= deadline:
            pytest.fail("the harness update did not commit")
        await asyncio.sleep(UPDATE_POLL_S)


@pytest.mark.unit
def test_status_reports_a_committed_step_that_published_no_harness(tmp_path) -> None:
    async def run() -> None:
        async with served(channel_dispatcher(tmp_path, ())) as client:
            status = await (await client.get("/reef/status")).json()
            assert status["scenarios"]["delivery"]["last_committed_step"] is None
            await scored_exchange(client)
            scenario = await committed_step(client)
            committed = scenario["last_committed_step"]
            # The step's job committed its submission, then its end.
            assert scenario["scenario_step"] == committed["step"] == 2
            assert isinstance(committed["recorded_at"], float)
            assert committed["metrics"]["skipped"] == "the method left the head tree unchanged"
            # The empty seed serves nothing until a step publishes a tree.
            response = await client.get("/reef/harness", headers=HEADERS)
            assert response.status == 404 and "holds no harness tree" in await response.text()

    asyncio.run(run())


@pytest.mark.unit
def test_status_reports_a_committed_rejection(tmp_path) -> None:
    async def run() -> None:
        async with served(channel_dispatcher(tmp_path, (TreePatch(changed_rules("no help\n")),))) as client:
            await scored_exchange(client)
            scenario = await committed_step(client)
            committed = scenario["last_committed_step"]
            assert scenario["scenario_step"] == committed["step"] == 2
            assert committed["metrics"]["selected"] is False
            assert committed["metrics"]["selection"]["selected"] is False
            assert (await client.get("/reef/harness", headers=HEADERS)).status == 404

    asyncio.run(run())


@pytest.mark.unit
def test_adapters_endpoint_lists_bundled_adapters_with_install_pins(tmp_path) -> None:
    async def run() -> None:
        async with served(channel_dispatcher(tmp_path, ())) as client:
            response = await client.get("/reef/harness/adapters")
            assert response.status == 200
            by_name = {entry["name"]: entry for entry in (await response.json())["adapters"]}
            # Bundled adapters must all appear with the fields the endpoint promises.
            assert {"claude", "opencode", "pi"}.issubset(by_name.keys())
            pi = by_name["pi"]
            assert pi["binary"] == "pi" and pi["trajectory_format"] == "pi-session-jsonl"
            assert pi["install"]["package"].startswith("@")
            assert isinstance(pi["model_bindings"], list) and pi["model_bindings"]

    asyncio.run(run())


@pytest.mark.unit
def test_a_published_step_reaches_the_served_files_and_the_catalog(tmp_path) -> None:
    """One evaluated step: the served files are exactly the evaluated tree rendered for the adapter, the source is
    one query away, and the catalog row repeats the manifest's evaluation."""

    async def run() -> None:
        async with served(channel_dispatcher(tmp_path, PATCHES[:1])) as client:
            rows = (await (await client.get("/reef/scenarios/delivery/artifacts")).json())["artifacts"]
            current = next(row["release_id"] for row in rows if row["active"])
            manifest = await evaluated_step(client)
            assert manifest["release_id"] != current

            # The served files are exactly the tree the evaluation measured, rendered without a model binding.
            assert manifest["files"] == render_tree(TREE_V1, PI)
            source = await client.get("/reef/harness", params={"form": "source"}, headers=HEADERS)
            assert source.status == 200
            pulled = await source.json()
            assert pulled["files"] == dict(TREE_V1.files) and pulled["release_id"] == manifest["release_id"]
            invalid = await client.get("/reef/harness", params={"form": "zip"}, headers=HEADERS)
            assert invalid.status == 400 and "form must be 'rendered', 'source' or 'client'" in await invalid.text()

            response = await client.get("/reef/harness/releases", headers=HEADERS)
            assert response.status == 200
            catalog = await response.json()
            assert catalog["scenario"] == "delivery"
            rows = catalog["releases"]
            # Newest last: the creation release opens the catalog, the published head closes it with the
            # manifest's evaluation.
            assert rows[0]["operation"] == "creation"
            head = rows[-1]
            assert head["release_id"] == manifest["release_id"] and head["active"] is True
            assert head["metrics"] == manifest["evaluation"]
            assert head["metrics"]["selected"] is True and head["metrics"]["wins"] == 1

    asyncio.run(run())


@pytest.mark.unit
def test_version_addressed_pull_returns_the_superseded_tree(tmp_path) -> None:
    async def run() -> None:
        async with served(channel_dispatcher(tmp_path, PATCHES)) as client:
            first = await evaluated_step(client)
            second = await evaluated_step(client)
            assert first["release_id"] != second["release_id"]
            response = await client.get("/reef/harness", params={"release_id": first["release_id"]}, headers=HEADERS)
            assert response.status == 200 and response.headers["x-reef-release-id"] == first["release_id"]
            manifest = await response.json()
            assert manifest["release_id"] == first["release_id"]
            assert manifest["files"] == render_tree(TREE_V1, PI)
            assert second["files"] == render_tree(TREE_V2, PI)
            source = await client.get(
                "/reef/harness", params={"release_id": first["release_id"], "form": "source"}, headers=HEADERS
            )
            assert (await source.json())["files"] == dict(TREE_V1.files)

    asyncio.run(run())


@pytest.mark.unit
def test_an_unknown_version_is_a_404_naming_it_and_the_install_route_is_gone(tmp_path) -> None:
    """A client renders or pulls the tree itself: the route that served the wrapper's install script is gone."""

    async def run() -> None:
        async with served(channel_dispatcher(tmp_path, ())) as client:
            response = await client.get("/reef/harness", params={"release_id": "no-such-version"}, headers=HEADERS)
            assert response.status == 404 and "no-such-version" in await response.text()
            response = await client.get("/reef/harness/install", params={"adapter": "pi"}, headers=HEADERS)
            assert response.status == 404

    asyncio.run(run())


@pytest.mark.unit
def test_release_reads_stay_responsive_during_evaluation(tmp_path, monkeypatch) -> None:
    evaluation_started = Event()
    finish_evaluation = Event()

    async def run() -> None:
        dispatcher = channel_dispatcher(tmp_path, PATCHES)
        backend = job_backend(dispatcher.get_or_create_scenario("delivery").trainer)
        evaluate_candidate = backend.evaluate

        def blocked_evaluate(candidate: UpdateCandidate, **options) -> EvaluationResult:
            result = evaluate_candidate(candidate, **options)
            evaluation_started.set()
            assert finish_evaluation.wait(UPDATE_TIMEOUT_S), "evaluation was not released"
            return result

        second_step = None
        async with served(dispatcher) as client:
            try:
                first = await evaluated_step(client)
                monkeypatch.setattr(backend, "evaluate", blocked_evaluate)
                second_step = asyncio.create_task(evaluated_step(client))
                assert await asyncio.to_thread(evaluation_started.wait, UPDATE_TIMEOUT_S)

                async def read_release(path: str, release_id: str | None = None) -> dict:
                    params = {} if release_id is None else {"release_id": release_id}
                    response = await client.get(path, params=params, headers=HEADERS)
                    assert response.status == 200
                    return await response.json()

                # A resident client keeps seeing the committed tree and its evaluation while the next candidate runs.
                catalog, current, pinned = await asyncio.wait_for(
                    asyncio.gather(
                        read_release("/reef/harness/releases"),
                        read_release("/reef/harness"),
                        read_release("/reef/harness", first["release_id"]),
                    ),
                    timeout=1.0,
                )
                assert current == pinned == first
                head = catalog["releases"][-1]
                assert head["active"] is True and head["release_id"] == first["release_id"]
                assert head["content_id"] == first["content_id"] and head["metrics"] == first["evaluation"]

                finish_evaluation.set()
                second = await asyncio.wait_for(second_step, UPDATE_TIMEOUT_S)
                updated = await read_release("/reef/harness/releases")
                assert len(updated["releases"]) == len(catalog["releases"]) + 1
                assert updated["releases"][-2]["active"] is False
                head = updated["releases"][-1]
                assert head["active"] is True and head["release_id"] == second["release_id"] != first["release_id"]
                assert head["content_id"] == second["content_id"] and head["metrics"] == second["evaluation"]
                assert second["files"] == render_tree(TREE_V2, PI)
                assert await read_release("/reef/harness", first["release_id"]) == first
            finally:
                finish_evaluation.set()
                if second_step is not None:
                    await asyncio.gather(second_step, return_exceptions=True)

    asyncio.run(run())


@pytest.mark.unit
def test_crlf_content_survives_the_manifest_byte_exact(tmp_path) -> None:
    """No newline translation ever rewrites served bytes."""
    crlf = TreePatch(changed_rules("marker win\r\nrules\n"))

    async def run() -> None:
        async with served(channel_dispatcher(tmp_path, (crlf,))) as client:
            manifest = await evaluated_step(client)
            assert "marker win\r\nrules" in manifest["files"]["plugin/rules.md"]
            source = await (await client.get("/reef/harness", params={"form": "source"}, headers=HEADERS)).json()
            assert source["files"]["rules.md"] == "marker win\r\nrules\n"

    asyncio.run(run())


@pytest.mark.unit
def test_a_seeded_recipe_serves_a_fresh_scenario_before_any_step(tmp_path) -> None:
    """A fresh scenario against a recipe with a seed answers the manifest at once with the seed rendered for the
    adapter, without a model binding; no step is needed."""
    seed = HarnessTree({"skills/answer-style/SKILL.md": "# seed skill\n"})
    dispatcher = channel_dispatcher(tmp_path, (), seed=seed)

    async def run() -> None:
        async with served(dispatcher) as client:
            response = await client.get("/reef/harness", headers=HEADERS)
            assert response.status == 200
            files = (await response.json())["files"]
            # The seed skill's text plus the frontmatter pi requires, made from its first line.
            assert files["plugin/skills/answer-style/SKILL.md"] == (
                "---\nname: answer-style\ndescription: seed skill\n---\n# seed skill\n"
            )
            assert "pi-agent/models.json" in files and "reef" not in files["pi-agent/models.json"]
            source = await (await client.get("/reef/harness", params={"form": "source"}, headers=HEADERS)).json()
            assert source["files"] == dict(seed.files)

    asyncio.run(run())


@pytest.mark.unit
def test_record_only_traffic_fires_a_step_and_publishes(tmp_path) -> None:
    """The report-free policy end to end through the real service: recorded inference traffic alone fills the
    batch, one step runs on the unscored samples, and the read channel serves the winner."""

    async def run() -> None:
        async with served(channel_dispatcher(tmp_path, PATCHES[:1], inference_stages=True, job_samples=2)) as client:
            for prompt in ("first", "second"):
                response = await client.post(
                    "/v1/chat/completions", headers=HEADERS, json={"messages": [{"role": "user", "content": prompt}]}
                )
                assert response.status == 200
                await response.read()
            deadline = asyncio.get_running_loop().time() + UPDATE_TIMEOUT_S
            while True:
                response = await client.get("/reef/harness", headers=HEADERS)
                if response.status == 200:
                    manifest = await response.json()
                    assert manifest["evaluation"]["selected"] is True
                    assert "marker rules" in manifest["files"]["plugin/rules.md"]
                    break
                assert response.status == 404
                await response.read()
                if asyncio.get_running_loop().time() >= deadline:
                    pytest.fail("record-driven step did not publish")
                await asyncio.sleep(UPDATE_POLL_S)

    asyncio.run(run())


def descriptor_with(tmp_path: Path, adapter: str, change) -> Path:
    """A copy of a bundled adapter's ``adapter.yaml``, changed by ``change``."""
    data = descriptor_data(adapter)
    change(data)
    return write_descriptor(tmp_path, data)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("adapter", "field", "value"),
    [
        ("pi", "package", "bad package; rm -rf /"),
        ("pi", "version", "0.84.2$(touch pwned)"),
        ("hermes", "repository", "git@github.com:acme/agent.git"),
        ("hermes", "ref", "v1$(touch pwned)"),
    ],
)
def test_descriptor_install_fields_constrain_their_charset(tmp_path, adapter: str, field: str, value: str) -> None:
    """Install fields reach a vendor command line, so the descriptor parse pins their charsets and a violation
    names the offending field."""
    target = descriptor_with(tmp_path, adapter, lambda data: data["install"].__setitem__(field, value))
    with pytest.raises(DescriptorError, match=f"install '{field}'"):
        load_descriptor(target)


@pytest.mark.unit
@pytest.mark.parametrize("path", ["/tmp/state", "../state", "workspace", "workspace/state", "."])
def test_descriptor_dirs_stay_in_managed_state(tmp_path, path: str) -> None:
    target = descriptor_with(tmp_path, "pi", lambda data: data.__setitem__("dirs", {path: None}))
    with pytest.raises(DescriptorError, match=r"'dirs'.*run root"):
        load_descriptor(target)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("layer", "message"),
    [
        ({"args": [["--key", "{token}"]]}, "must not hold {token}"),
        ({"register": [["login", "{token}"]]}, "must not hold {token}"),
        ({"args": ["-e"]}, "a list of argument groups"),
        ({"files": {}}, "an object of args, env, register and unregister"),
    ],
)
def test_a_descriptor_layer_keeps_the_token_off_the_command_line(tmp_path, layer, message: str) -> None:
    target = descriptor_with(tmp_path, "pi", lambda data: data["run"].__setitem__("layer", layer))
    with pytest.raises(DescriptorError, match=re.escape(message)):
        load_descriptor(target)
