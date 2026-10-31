"""A real HTTP provider, method, judge and subprocess rollout share a scenario model setting.

A scenario's model names the engine only (``url``, ``model``, ``api``); every call to it carries the deployment's
provider key, which never reaches stored state or an API view.
"""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest
from tests.support._strategies import resolve_method, resolve_task_scorer
from tests.support._trajectories import recorded_trajectory
from tests.support.harness_service import harness_repository, job_backend

from reef.artifact import Artifact, InMemoryRepositoryBackend
from reef.dispatcher import Dispatcher
from reef.evaluation.harness import ComparePolicy, EvaluationSettings
from reef.harness.tasks.model_binding import ModelBinding
from reef.harness.tree import HarnessTree, TreePatch
from reef.inference.http import InferenceProxyRuntime
from reef.inference.model_config import ModelConfig
from reef.recipe.harness import EvolutionSettings, HarnessEvolutionRecipe
from reef.runtime.executor.config import ExecutorSettings
from reef.storage.scenario import ScenarioSettings
from reef.storage.sql_coordination import memory_coordination
from reef.storage.sqlite import SQLiteRecordStore, SQLiteScenarioStorage
from reef.train.backend import ReleaseFiles
from reef.train.harness_backend.state import HarnessState
from reef.train.jobs import JobLoopState
from reef.train.types import JobData


@pytest.fixture
def platform():
    state = {"calls": [], "fail_models": False}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            state["calls"].append((self.path, dict(self.headers), body))
            if state["fail_models"]:
                self.send_error(401)
                return
            if self.path.endswith("/messages") or self.path.endswith("/count_tokens"):
                answer = {
                    "content": [{"type": "text", "text": "OK"}],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            else:
                answer = {
                    "choices": [{"message": {"content": "OK"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(answer).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    state["url"] = f"http://127.0.0.1:{server.server_port}"
    # A short poll interval lets shutdown() return at once instead of after half a second.
    thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


#: The deployment's provider key: the key of the recipe's provider runtime.
PLATFORM_KEY = "platform-key"


def configure(platform, name, api, version="1"):
    return {
        "url": f"{platform['url']}/{name}/{version}",
        "model": f"custom-{name}",
        "api": api,
    }


def engine_url(platform, name, version="1"):
    return f"{platform['url']}/{name}/{version}"


FAKE_AGENT = """#!/usr/bin/env python3
import json, os, sys, urllib.request
from pathlib import Path
assert not os.environ.get("OPENAI_API_KEY")
assert not os.environ.get("ANTHROPIC_API_KEY")
root = Path(os.environ["PI_CODING_AGENT_DIR"])
provider = next(iter(json.loads((root / "models.json").read_text())["providers"].values()))
anthropic = provider["api"] == "anthropic-messages"
url = provider["baseUrl"] + ("/v1/messages" if anthropic else "/chat/completions")
headers = {"content-type": "application/json"}
headers["x-api-key" if anthropic else "authorization"] = provider["apiKey"] if anthropic else "Bearer " + provider["apiKey"]
if anthropic: headers["anthropic-version"] = "2023-06-01"
body = {"model": provider["models"][0]["id"], "max_tokens": 10, "messages": [{"role": "user", "content": "rollout"}]}
with urllib.request.urlopen(urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)) as response:
    json.load(response)
sessions = Path(os.environ["PI_CODING_AGENT_SESSION_DIR"])
sessions.mkdir(parents=True, exist_ok=True)
rules = Path(sys.argv[sys.argv.index("-e") + 1]) / "rules.md"
(sessions / "session.jsonl").write_text(json.dumps({"type": "agent_end", "rules": rules.read_text() if rules.exists() else ""}) + "\\n")
"""


def judge(task, result, *, models):
    assert models["judge"].chat([{"role": "user", "content": "judge"}]) == "OK"
    return 1.0 if "marker" in result.trajectory[-1]["rules"] else 0.0


def make_recipe(platform, tmp_path):
    binary = tmp_path / "agent"
    binary.write_text(FAKE_AGENT)
    binary.chmod(0o755)

    def method(workspace, services):
        assert services.models["teacher"].chat([{"role": "user", "content": "method"}]) == "OK"
        return TreePatch({"rules.md": "marker\n"})

    managed = ModelBinding(base_url=f"{platform['url']}/managed", api_key=PLATFORM_KEY, model="platform-model")
    return HarnessEvolutionRecipe(
        method=resolve_method(method),
        evolution=EvolutionSettings(binary=str(binary), state_dir=str(tmp_path / "harness")),
        evaluation=EvaluationSettings(tasks=("test task",)),
        scorer=resolve_task_scorer(judge),
        runtime=InferenceProxyRuntime(base_url=managed.base_url, api_key=managed.api_key, model_path=managed.model),
        models={"teacher": managed, "judge": managed},
        seed=HarnessTree({"skills/notes/SKILL.md": "notes\n"}),
    )


# Each API and each executor runs once: the API decides the headers, the executor where the rollouts run.
@pytest.mark.parametrize("executor, api", [("uni", "anthropic"), ("mp", "openai")])
def test_full_evolution_uses_only_custom_binding(platform, tmp_path, monkeypatch, api, executor):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-rollouts")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-reach-rollouts")
    recipe = replace(make_recipe(platform, tmp_path), worker_executor=ExecutorSettings(backend=executor))
    dispatcher = Dispatcher(
        recipe,
        harness_repository(tmp_path, recipe.base_artifact_source().files),
        scenario_storage=SQLiteScenarioStorage(),
        coordination=memory_coordination(),
    )
    try:
        scenario = dispatcher.configure_scenario_model("alpha", configure(platform, "alpha", api), create=True)
        artifact = Artifact(scenario.repository.require_current_artifact(), scenario.repository)
        path = "/v1/messages" if api == "anthropic" else "/v1/chat/completions"
        asyncio.run(
            scenario.runtime.inference_handler.inference(artifact, path, {"model": "custom-alpha", "messages": []})
        )
        if api == "anthropic":
            asyncio.run(
                scenario.runtime.inference_handler.inference(
                    artifact, "/v1/messages/count_tokens", {"model": "custom-alpha", "messages": []}
                )
            )
        # A job carries the scenario's model without its key; the training component binds the deployment's key.
        selected = scenario.trainer.job_loop_required().submitted.parameters(
            JobData("batch"), JobLoopState.from_dict(scenario.trainer.state)
        )["scenario_model"]
        assert selected == configure(platform, "alpha", api)
        backend = job_backend(scenario.trainer, "harness", selected)
        prepared = backend.prepare_step(
            JobData("batch", (recorded_trajectory("record", {"messages": []}, 0.0),)),
            HarnessState().to_json(),
            0,
            head=ReleaseFiles("base", dict(recipe.seed.files)),
        )
        assert prepared.candidate is not None
        # Updates between the method and the evaluation must not mix providers within the step.
        dispatcher.configure_scenario_model("alpha", configure(platform, "alpha", api, "2"))
        policy = ComparePolicy(backend)
        evaluation = policy.evaluate(prepared.candidate)
        result = backend.settle_step(prepared, policy.decide(prepared.candidate, evaluation))
        assert result.metrics["selected"] is True
        calls = platform["calls"]
        assert len(calls) == (7 if api == "anthropic" else 6)  # inference, method, two rollouts, two judges (+ count)
        for path, raw_headers, body in calls:
            headers = {k.lower(): v for k, v in raw_headers.items()}
            assert path.startswith("/alpha/1/")
            assert body["model"] == "custom-alpha"
            if api == "anthropic":
                assert headers["x-api-key"] == PLATFORM_KEY
                assert "authorization" not in headers
            else:
                assert headers["authorization"] == f"Bearer {PLATFORM_KEY}"
        assert PLATFORM_KEY not in json.dumps(dict(prepared.candidate.candidate_tree.files))
        assert PLATFORM_KEY not in json.dumps(result.state)
        result.publication.artifact.discard()
        asyncio.run(
            scenario.runtime.inference_handler.inference(
                artifact,
                "/v1/messages" if api == "anthropic" else "/v1/chat/completions",
                {"model": "custom-alpha", "messages": []},
            )
        )
        assert platform["calls"][-1][0].startswith("/alpha/2/")
    finally:
        dispatcher.close()


def test_concurrent_scenarios_rotation_and_fail_closed(platform, tmp_path):
    alpha_config = ModelConfig.from_value(configure(platform, "alpha", "openai"), api_key=PLATFORM_KEY)
    beta_config = ModelConfig.from_value(configure(platform, "beta", "anthropic"), api_key=PLATFORM_KEY)
    recipe = make_recipe(platform, tmp_path)
    alpha = recipe.with_model_config(alpha_config)
    beta = recipe.with_model_config(beta_config)
    with ThreadPoolExecutor(2) as executor:
        list(
            executor.map(
                lambda r: r.model_bindings().served.chat([{"role": "user", "content": "parallel"}]), (alpha, beta)
            )
        )
    assert {path.split("/")[1] for path, _, _ in platform["calls"]} == {"alpha", "beta"}
    first = alpha.model_bindings()
    alpha_config.runtime = ModelConfig.from_value(
        configure(platform, "alpha", "anthropic", "2"), api_key=PLATFORM_KEY
    ).runtime
    second = alpha.model_bindings()
    assert first.served.base_url == engine_url(platform, "alpha", "1")
    assert second.served.base_url == engine_url(platform, "alpha", "2")
    assert all(binding.api == "anthropic" for binding in second.values())
    assert beta.model_bindings().served.base_url == engine_url(platform, "beta", "1")
    with pytest.raises(ValueError):
        alpha_config.runtime = ModelConfig.from_value({"url": "bad", "model": "bad"}).runtime
    assert alpha.model_bindings().served.base_url == engine_url(platform, "alpha", "2")
    assert not any("/managed/" in path for path, _, _ in platform["calls"])


def test_recipe_uses_model_config_without_a_persistence_interface(platform, tmp_path):
    config = ModelConfig.from_value(configure(platform, "alpha", "openai"), api_key=PLATFORM_KEY)
    recipe = make_recipe(platform, tmp_path).with_model_config(config)
    first = recipe.model_bindings()
    config.runtime = InferenceProxyRuntime.from_model_config(
        configure(platform, "alpha", "anthropic", "2"), api_key=PLATFORM_KEY
    )
    second = recipe.model_bindings()
    assert first.served.base_url == engine_url(platform, "alpha", "1")
    assert all(binding.api == "anthropic" for binding in second.values())
    assert all(binding.base_url == engine_url(platform, "alpha", "2") for binding in second.values())
    assert all(binding.api_key == PLATFORM_KEY for binding in second.values())
    config.runtime = None
    served = recipe.model_bindings().served
    assert (served.base_url, served.api_key) == (f"{platform['url']}/managed", PLATFORM_KEY)


def test_model_settings_persist_privately_and_invalid_values_fail_recovery(platform, tmp_path):
    value = configure(platform, "alpha", "openai")
    storage = SQLiteScenarioStorage(tmp_path)
    storage.write_settings("alpha", ScenarioSettings(model_config=value))
    database = next(tmp_path.glob("*.sqlite3"))
    assert database.stat().st_mode & 0o777 == 0o600
    assert SQLiteScenarioStorage(tmp_path).read_settings("alpha").model_config == value
    recovered = ModelConfig.from_value(storage.read_settings("alpha").model_config, api_key=PLATFORM_KEY)
    assert recovered.runtime.base_url == engine_url(platform, "alpha", "1")
    assert recovered.runtime.provider_key == PLATFORM_KEY
    assert recovered.view() == {"url": engine_url(platform, "alpha", "1"), "model": "custom-alpha", "api": "openai"}
    assert PLATFORM_KEY not in repr(recovered)
    # A scenario names its engine only: a key of its own is refused.
    with pytest.raises(ValueError, match="url, model and api"):
        ModelConfig.from_value({**value, "api_key": "scoped-alpha"})
    storage.write_settings("alpha", ScenarioSettings())
    assert ModelConfig.from_value(storage.read_settings("alpha").model_config).runtime is None
    storage.write_settings("alpha", ScenarioSettings(model_config={"url": "http://user:secret@host", "model": "bad"}))
    with pytest.raises(ValueError):
        ModelConfig.from_value(storage.read_settings("alpha").model_config)


def test_model_auth_failure_never_falls_back(platform, tmp_path):
    config = ModelConfig.from_value(configure(platform, "alpha", "openai"))
    recipe = make_recipe(platform, tmp_path).with_model_config(config)
    bindings = recipe.model_bindings()
    platform["fail_models"] = True
    with pytest.raises(Exception, match="401"):
        bindings["teacher"].chat([{"role": "user", "content": "method"}])
    assert len(platform["calls"]) == 1
    assert platform["calls"][0][0].startswith("/alpha/")


def test_http_contract_updates_the_model_and_serves_no_provider(platform, tmp_path):
    from aiohttp.test_utils import TestClient, TestServer

    from reef.service.app import create_app

    recipe = make_recipe(platform, tmp_path)
    dispatcher = Dispatcher(
        recipe,
        harness_repository(tmp_path, recipe.base_artifact_source().files),
        scenario_storage=SQLiteScenarioStorage(),
        coordination=memory_coordination(),
    )

    async def run():
        client = TestClient(TestServer(create_app(dispatcher, tokens="deployment-secret")))
        await client.start_server()
        headers = {
            "authorization": "Bearer deployment-secret",
            "x-reef-scenario": "alpha",
        }
        try:
            payload = {"name": "alpha", "model": configure(platform, "alpha", "openai")}
            unauthorized = await client.post("/reef/scenarios", json=payload)
            assert unauthorized.status == 401
            created = await client.post("/reef/scenarios", headers=headers, json=payload)
            assert created.status == 201, await created.text()
            view = await created.json()
            assert view["model"] == {
                "url": engine_url(platform, "alpha", "1"),
                "model": "custom-alpha",
                "api": "openai",
            }
            assert PLATFORM_KEY not in json.dumps(view)
            # Repeated creation must never overwrite a configuration changed elsewhere.
            existing = await client.post("/reef/scenarios", headers=headers, json={"name": "alpha", "model": None})
            assert existing.status == 200
            assert (await existing.json())["model"] == view["model"]
            missing = await client.post(
                "/reef/scenarios/missing/update", headers=headers, json={"model": payload["model"]}
            )
            assert missing.status == 404
            response = await client.post(
                "/v1/chat/completions", headers=headers, json={"model": "custom-alpha", "messages": []}
            )
            assert response.status == 200, await response.text()
            assert response.headers["x-reef-agent-record-id"]
            invalid = await client.post(
                "/reef/scenarios/alpha/update", headers=headers, json={"model": {"url": "bad"}}
            )
            assert invalid.status == 400
            updated = await client.post(
                "/reef/scenarios/alpha/update",
                headers=headers,
                json={"model": configure(platform, "alpha", "anthropic", "2")},
            )
            assert updated.status == 200, await updated.text()
            assert (await updated.json())["model"]["api"] == "anthropic"
            fresh = await client.post("/v1/messages", headers=headers, json={"model": "custom-alpha", "messages": []})
            assert fresh.status == 200, await fresh.text()
            scenario = dispatcher.get_or_create_scenario("alpha")
            # The served tree carries no provider: the deployment's key never reaches a client.
            for form in ("rendered", "source"):
                served = await client.get("/reef/harness", headers=headers, params={"form": form})
                assert served.status == 200, await served.text()
                files = json.dumps((await served.json())["files"])
                assert PLATFORM_KEY not in files
            reset = await client.post("/reef/scenarios/alpha/update", headers=headers, json={"model": None})
            assert reset.status == 200
            assert (await reset.json())["model"] is None
            assert scenario.runtime.provider_key == PLATFORM_KEY
        finally:
            await client.close()

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


def test_scenario_recovery_and_deletion_keep_model_lifecycle(platform, tmp_path):
    initial = tmp_path / "initial"
    initial.mkdir()
    records = tmp_path / "records"
    factory = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    recipe = make_recipe(platform, tmp_path)
    first = Dispatcher(
        recipe, factory, scenario_storage=SQLiteScenarioStorage(records), coordination=memory_coordination()
    )
    first.configure_scenario_model("alpha", configure(platform, "alpha", "anthropic"), create=True)
    first.close()
    recovered = Dispatcher(
        recipe, factory, scenario_storage=SQLiteScenarioStorage(records), coordination=memory_coordination()
    )
    try:
        scenario = recovered.get_or_create_scenario("alpha")
        assert (scenario.runtime.base_url, scenario.runtime.provider_key) == (
            engine_url(platform, "alpha", "1"),
            PLATFORM_KEY,
        )
        selected = scenario.trainer.job_loop_required().submitted.parameters(
            JobData("batch"), JobLoopState.from_dict(scenario.trainer.state)
        )["scenario_model"]
        assert job_backend(scenario.trainer, "harness", selected).models["judge"].base_url == engine_url(
            platform, "alpha", "1"
        )
        config = scenario.model_config
        previous_runtime = config.runtime
        reloaded = recovered.registry.reload("alpha")
        assert reloaded is not scenario
        assert reloaded.model_config is config
        recovered.configure_scenario_model("alpha", configure(platform, "alpha", "openai", "2"))
        assert reloaded.runtime.base_url == engine_url(platform, "alpha", "2")
        assert previous_runtime.base_url == engine_url(platform, "alpha", "1")
        assert not list(records.glob("*-model.json"))
        recovered.delete_scenario("alpha")
        assert recovered.storage.read_settings("alpha") == ScenarioSettings()
        with closing(SQLiteRecordStore(next((records / "archived").glob("*/*.sqlite3")))) as archived:
            assert archived.read_settings().model_config == configure(platform, "alpha", "openai", "2")
        recreated = recovered.get_or_create_scenario("alpha")
        assert recreated.model_config is not config
        assert (recreated.runtime.base_url, recreated.runtime.provider_key) == (
            f"{platform['url']}/managed",
            PLATFORM_KEY,
        )
    finally:
        recovered.close()


def test_model_update_write_failure_keeps_active_config_and_saved_value(platform, tmp_path, monkeypatch):
    initial = tmp_path / "initial"
    initial.mkdir()
    records = tmp_path / "records"
    factory = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    dispatcher = Dispatcher(
        make_recipe(platform, tmp_path),
        factory,
        scenario_storage=SQLiteScenarioStorage(records),
        coordination=memory_coordination(),
    )
    try:
        original = configure(platform, "alpha", "openai")
        scenario = dispatcher.configure_scenario_model("alpha", original, create=True)
        runtime = scenario.runtime

        def fail_write(storage, scenario, settings):
            raise OSError("injected settings write failure")

        with monkeypatch.context() as patch:
            patch.setattr(SQLiteScenarioStorage, "write_settings", fail_write)
            with pytest.raises(OSError, match="injected settings write failure"):
                dispatcher.configure_scenario_model("alpha", configure(platform, "alpha", "anthropic", "2"))
        assert scenario.runtime is runtime
        assert SQLiteScenarioStorage(records).read_settings("alpha").model_config == original
        with pytest.raises(ValueError):
            dispatcher.configure_scenario_model("alpha", {"url": "http://user:secret@host", "model": "bad"})
        assert scenario.runtime is runtime
        assert SQLiteScenarioStorage(records).read_settings("alpha").model_config == original
    finally:
        dispatcher.close()
