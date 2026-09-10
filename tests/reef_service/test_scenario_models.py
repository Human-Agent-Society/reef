"""A real HTTP provider, proposer, judge and subprocess episode share a scenario model setting."""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from reef.artifact import Artifact, InMemoryRepositoryBackend
from reef.dispatcher import Dispatcher
from reef.harness.episodes.model_binding import ModelBinding
from reef.runtime.adapters.inference_proxy import InferenceProxyRuntime
from reef.runtime.executor.config import ExecutorSettings
from reef.scenario.model_config import ScenarioModelConfig
from reef.train.cordis_backend import CordisRecipe, Mutation, ScoreComparisonSelector
from reef.train.cordis_backend.strategies import resolve_episode_scorer, resolve_proposer
from reef.train.evaluation import DefaultCandidateEvaluationPlugin
from reef.train.types import TraceBatch, TraceSample


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
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def configure(platform, name, api, version="1"):
    return {
        "url": f"{platform['url']}/{name}/{version}",
        "api_key": f"scoped-{name}-{version}",
        "model": f"custom-{name}",
        "api": api,
    }


FAKE_AGENT = """#!/usr/bin/env python3
import json, os, urllib.request
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
body = {"model": provider["models"][0]["id"], "max_tokens": 10, "messages": [{"role": "user", "content": "episode"}]}
with urllib.request.urlopen(urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)) as response:
    json.load(response)
sessions = Path(os.environ["PI_CODING_AGENT_SESSION_DIR"])
sessions.mkdir(parents=True, exist_ok=True)
rules = root / "AGENTS.md"
(sessions / "session.jsonl").write_text(json.dumps({"type": "agent_end", "rules": rules.read_text() if rules.exists() else ""}) + "\\n")
"""


def judge(task, result, *, models):
    assert models["judge"].chat([{"role": "user", "content": "judge"}]) == "OK"
    return 1.0 if "marker" in result.trajectory[-1]["rules"] else 0.0


def make_recipe(platform, tmp_path):
    binary = tmp_path / "agent"
    binary.write_text(FAKE_AGENT)
    binary.chmod(0o755)

    def propose(nodes, samples, models):
        assert models["teacher"].chat([{"role": "user", "content": "proposer"}]) == "OK"
        return Mutation("create", "improvement", {"name": "rules", "config": {"text": "marker"}})

    managed = ModelBinding(base_url=f"{platform['url']}/managed", api_key="platform-key", model="platform-model")
    return CordisRecipe(
        resolve_proposer(propose),
        resolve_episode_scorer(judge),
        ("test task",),
        runtime=InferenceProxyRuntime(base_url=managed.base_url, api_key=managed.api_key, model_path=managed.model),
        models={"teacher": managed, "judge": managed},
        binary=str(binary),
        client_models=("managed-alternative",),
        seed=({"id": "initial", "name": "skill", "config": {"name": "notes", "text": "notes"}},),
        proposals_dir=str(tmp_path / "proposals"),
    )


@pytest.mark.parametrize("api", ["openai", "anthropic"])
@pytest.mark.parametrize("executor", ["uni", "mp"])
def test_full_evolution_uses_only_custom_binding(platform, tmp_path, monkeypatch, api, executor):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-episodes")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-reach-episodes")
    recipe = replace(make_recipe(platform, tmp_path), worker_executor=ExecutorSettings(backend=executor))
    initial = tmp_path / "initial"
    initial.mkdir()
    dispatcher = Dispatcher(recipe, InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repo"))
    try:
        scenario = dispatcher.configure_scenario_model("alpha", configure(platform, "alpha", api), create=True)
        artifact = Artifact(scenario.repository.require_current_artifact(), scenario.repository)
        path = "/v1/messages" if api == "anthropic" else "/v1/chat/completions"
        asyncio.run(scenario.inference_backend.inference(artifact, path, {"model": "custom-alpha", "messages": []}))
        if api == "anthropic":
            asyncio.run(
                scenario.inference_backend.inference(
                    artifact, "/v1/messages/count_tokens", {"model": "custom-alpha", "messages": []}
                )
            )
        backend = scenario.trainer.training_backend
        prepared = backend.prepare_step(
            TraceBatch("batch", (TraceSample("record", {"messages": []}, 0.0),)), backend.initial_state(), 0
        )
        assert prepared.candidate is not None
        # Updates between proposal and evaluation must not mix providers within the step.
        dispatcher.configure_scenario_model("alpha", configure(platform, "alpha", api, "2"))
        plugin = DefaultCandidateEvaluationPlugin(backend, ScoreComparisonSelector())
        evaluation = plugin.evaluate(prepared.candidate)
        result = backend.settle_step(prepared, plugin.decide(prepared.candidate, evaluation))
        assert result.metrics["selected"] is True
        calls = platform["calls"]
        assert len(calls) == (
            7 if api == "anthropic" else 6
        )  # inference, proposer, two episodes, two judges (+ count)
        for path, raw_headers, body in calls:
            headers = {k.lower(): v for k, v in raw_headers.items()}
            assert path.startswith("/alpha/1/")
            assert body["model"] == "custom-alpha"
            if api == "anthropic":
                assert headers["x-api-key"] == "scoped-alpha-1"
                assert "authorization" not in headers
            else:
                assert headers["authorization"] == "Bearer scoped-alpha-1"
            assert "platform-key" not in json.dumps(raw_headers)
        assert "scoped-alpha" not in json.dumps(prepared.candidate.candidate_files)
        assert "scoped-alpha" not in json.dumps(result.state)
        result.publication.artifact.discard()
        asyncio.run(
            scenario.inference_backend.inference(
                artifact,
                "/v1/messages" if api == "anthropic" else "/v1/chat/completions",
                {"model": "custom-alpha", "messages": []},
            )
        )
        assert platform["calls"][-1][0].startswith("/alpha/2/")
    finally:
        dispatcher.close()


def test_concurrent_scenarios_rotation_and_fail_closed(platform, tmp_path):
    alpha_config = ScenarioModelConfig(tmp_path / "alpha.json")
    beta_config = ScenarioModelConfig(tmp_path / "beta.json")
    alpha_config.save(configure(platform, "alpha", "openai"))
    beta_config.save(configure(platform, "beta", "anthropic"))
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
    alpha_config.save(configure(platform, "alpha", "anthropic", "2"))
    second = alpha.model_bindings()
    assert first.served.api_key == "scoped-alpha-1"
    assert second.served.api_key == "scoped-alpha-2"
    assert all(binding.api == "anthropic" for binding in second.values())
    assert beta.model_bindings().served.api_key == "scoped-beta-1"
    with pytest.raises(ValueError):
        alpha_config.save({"url": "bad", "model": "bad"})
    assert alpha.model_bindings().served.api_key == "scoped-alpha-2"
    assert not any("/managed/" in path for path, _, _ in platform["calls"])


def test_model_settings_persist_privately_and_corruption_fails_recovery(platform, tmp_path):
    path = tmp_path / "model.json"
    config = ScenarioModelConfig(path)
    config.save(configure(platform, "alpha", "openai"))
    assert path.stat().st_mode & 0o777 == 0o600
    recovered = ScenarioModelConfig(path)
    assert recovered.runtime.api_key == "scoped-alpha-1"
    assert "scoped-alpha-1" not in json.dumps(recovered.view())
    with pytest.raises(ValueError):
        config.save({"url": "http://user:secret@host", "model": "bad"})
    assert ScenarioModelConfig(path).runtime.api_key == "scoped-alpha-1"
    config.save(None)
    assert ScenarioModelConfig(path).runtime is None
    path.write_text("{broken")
    with pytest.raises(ValueError):
        ScenarioModelConfig(path)


def test_model_auth_failure_never_falls_back(platform, tmp_path):
    config = ScenarioModelConfig()
    config.save(configure(platform, "alpha", "openai"))
    recipe = make_recipe(platform, tmp_path).with_model_config(config)
    bindings = recipe.model_bindings()
    platform["fail_models"] = True
    with pytest.raises(Exception, match="401"):
        bindings["teacher"].chat([{"role": "user", "content": "proposer"}])
    assert len(platform["calls"]) == 1
    assert platform["calls"][0][0].startswith("/alpha/")


def test_http_contract_updates_model_and_installs_selected_protocol(platform, tmp_path):
    from aiohttp.test_utils import TestClient, TestServer

    from reef.harness.adapters import get_adapter
    from reef.service.app import create_app
    from reef.service.request_service import RequestService

    recipe = make_recipe(platform, tmp_path)
    initial = tmp_path / "initial"
    initial.mkdir()
    for path, content in recipe.base_artifact_files().items():
        output = initial / path
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content)
    dispatcher = Dispatcher(recipe, InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repo"))

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
            assert view["model"]["model"] == "custom-alpha"
            assert view["model"]["has_api_key"] is True
            assert "scoped-alpha" not in json.dumps(view)
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
            service = RequestService(dispatcher)
            scenario = dispatcher.get_or_create_scenario("alpha")
            # The live config takes precedence over the old release's model dialect.
            binding = service._install_binding(
                scenario,
                {"release_id": scenario.repository.require_current_artifact().release_id},
                get_adapter("pi"),
                {"host": "platform.example", "x-forwarded-proto": "https"},
            )
            rendered = json.dumps(binding)
            assert "anthropic-messages" in rendered
            assert "https://platform.example" in rendered
            assert "scoped-alpha" not in rendered
            assert "managed-alternative" not in rendered
            reset = await client.post("/reef/scenarios/alpha/update", headers=headers, json={"model": None})
            assert reset.status == 200
            assert (await reset.json())["model"] is None
            assert scenario.runtime.api_key == "platform-key"
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
    first = Dispatcher(recipe, factory, agent_record_dir=records)
    first.configure_scenario_model("alpha", configure(platform, "alpha", "anthropic"), create=True)
    first.close()
    recovered = Dispatcher(recipe, factory, agent_record_dir=records)
    try:
        scenario = recovered.get_or_create_scenario("alpha")
        assert scenario.runtime.api_key == "scoped-alpha-1"
        assert scenario.trainer.training_backend._models["judge"].api_key == "scoped-alpha-1"
        model_file = next(records.glob("*-model.json"))
        assert model_file.stat().st_mode & 0o777 == 0o600
        recovered.delete_scenario("alpha")
        assert not model_file.exists()
        assert list((records / "archived").glob("*/*-model.json"))
        recreated = recovered.get_or_create_scenario("alpha")
        assert recreated.runtime.api_key == "platform-key"
    finally:
        recovered.close()
