"""Tinker training in the control plane, served by engines on an inference component, end to end on CPU.

The component (``python -m reef.inference.remote --engine sglang``) runs in its
own process with its own Ray session that declares two pretend GPUs. Its
receiver, which pins each request to its weights, and Reef's SGLang service,
control actor and backend are those of a deployment. Only the serving executor
is the CPU stand-in of ``tests.support.stub_engine``, which the deployment
selects as its ``rollout`` executor. The test process is the control
plane: it builds Tinker's training component and the serving runtime of its
adapters as the HTTP service does and drives one
TTTD step through the Dispatcher, then restarts. It sends generations through
the runtime's request handler, as the inference service does, and checks which
weights each published route reaches. The Tinker SDK and the
cookbook converter are faked unless ``REEF_TEST_TINKER_REAL=1`` and
``TINKER_API_KEY`` are set, in which case the trainer, the checkpoint download
and the PEFT conversion are real.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import textwrap
import time
from contextlib import contextmanager, suppress
from pathlib import Path

import pytest

from tests.support.config_helpers import connect_training_objects
from tests.support.stub_engine import adapter_weights_id

from reef.core.errors import UpstreamStatusError
from reef.core.served_models import ScenarioModel, ServedArtifact
from reef.dispatcher import Dispatcher
from reef.inference.interfaces import InferenceArtifact, InferenceOwnerLostError, RuntimeLoadId, full_weight_name
from reef.inference.residency import adapter_name
from reef.inference.serving import AdapterServingRuntime
from reef.scenario.live_objects import LiveObjects
from reef.service.deploy.orchestrator import resolve_deployment_config
from reef.service.deploy.service_config import service_config_from_mapping
from reef.train.jobs import JobLoopState

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL = "Qwen/Qwen3-8B"
REAL = os.environ.get("REEF_TEST_TINKER_REAL") == "1" and bool(os.environ.get("TINKER_API_KEY"))
ENGINE = "sglang"
#: The CPU stand-in for the native engines: the deployment's ``rollout`` executor, which Reef's SGLang control actor
#: makes in the component.
SERVING_EXECUTOR = "tests.support.stub_engine:StubEngineExecutor"
TOKENS = {"REEF_REMOTE_CONTROL_TOKEN": "control-secret", "REEF_REMOTE_INFERENCE_TOKEN": "inference-secret"}
SCENARIO = "erdos"

FAKE_SDK = textwrap.dedent(
    '''
    """Offline Tinker SDK: immutable checkpoints as tinker:// paths, checkpoint archives as local tar files."""
    import io, json, os, tarfile, uuid
    from types import SimpleNamespace

    # Settings live beside the module: Ray workers share the import path, not the driver's environment.
    _SETTINGS = json.load(open(os.path.join(os.path.dirname(__file__), "settings.json")))
    ROOT = _SETTINGS["root"]
    MODEL = _SETTINGS["model"]

    class _Future:
        def __init__(self, value): self.value = value
        def result(self, timeout=None): return self.value

    def _log(event):
        with open(os.path.join(ROOT, "events.jsonl"), "a") as stream:
            stream.write(json.dumps(event) + "\\n")

    class ModelInput:
        @staticmethod
        def from_ints(tokens): return SimpleNamespace(length=len(tokens), tokens=list(tokens))

    class Datum(SimpleNamespace): pass
    class TensorData(SimpleNamespace): pass
    class AdamParams(SimpleNamespace): pass
    class SamplingParams(SimpleNamespace): pass

    class _Trainer:
        def __init__(self, origin): self.origin = origin
        def forward(self, data, loss_fn, loss_fn_config=None):
            # The log probabilities of the job's start, for the rendered rows that serving recorded none for.
            _log({"event": "forward", "rows": len(data), "loss_fn": loss_fn, "from": self.origin})
            outputs = [{"logprobs": TensorData(data=[-0.5] * len(datum.loss_fn_inputs["target_tokens"].data))}
                       for datum in data]
            return _Future(SimpleNamespace(loss_fn_outputs=outputs))
        def forward_backward(self, data, loss_fn, loss_fn_config=None):
            _log({"event": "forward_backward", "rows": len(data), "loss_fn": loss_fn, "from": self.origin})
            return _Future(SimpleNamespace(metrics={"loss:sum": -1.0}))
        def optim_step(self, params): return _Future(None)
        def _save(self, name, kind):
            path = f"tinker://fake:train:0/{kind}/{name}"
            return _Future(SimpleNamespace(path=path))
        def save_state(self, name, ttl_seconds=None): return self._save(name, "weights")
        def save_weights_for_sampler(self, name, ttl_seconds=None): return self._save(name, "sampler_weights")

    class _Tokenizer:
        # A chat template: each message is its role ID and one token per character; the generation prompt is the
        # assistant role, 9. The component renders the served trajectories, which carry no tokens.
        ROLES = {"system": 1, "user": 2, "assistant": 9, "tool": 4}
        def apply_chat_template(self, messages, tools=None, tokenize=True, add_generation_prompt=False, **kwargs):
            tokens = [token for message in messages
                      for token in [self.ROLES[message["role"]], *(100 + ord(c) for c in message["content"])]]
            return tokens + ([9] if add_generation_prompt else [])
        def decode(self, *args, **kwargs): raise AssertionError("a job of trajectories decodes nothing")

    class _Sampler:
        def get_base_model(self): return MODEL
        def get_tokenizer(self): return _Tokenizer()
        def compute_logprobs(self, model_input): return _Future([None] + [-0.5] * (model_input.length - 1))

    class _Rest:
        def get_weights_info_by_tinker_path(self, path):
            return _Future(SimpleNamespace(base_model=MODEL, is_lora=True, lora_rank=32))
        def get_checkpoint_archive_url_from_tinker_path(self, path):
            archive = os.path.join(ROOT, uuid.uuid4().hex + ".tar")
            with tarfile.open(archive, "w") as tar:
                for name, body in (("adapter_config.json", json.dumps({"r": 32, "lora_alpha": 32}).encode()),
                                   ("adapter_model.safetensors", path.encode())):
                    info = tarfile.TarInfo(name)
                    info.size = len(body)
                    tar.addfile(info, io.BytesIO(body))
            return _Future(SimpleNamespace(url="file://" + archive))

    class ServiceClient:
        def __init__(self, **kwargs): _log({"event": "session", "keys": sorted(kwargs)})
        def create_sampling_client(self, base_model=None, model_path=None): return _Sampler()
        def create_rest_client(self): return _Rest()
        def get_server_capabilities(self): return SimpleNamespace(supported_models=[])
        def create_lora_training_client(self, base_model, rank, seed):
            _log({"event": "initialize", "base_model": base_model, "rank": rank}); return _Trainer("seed")
        def create_training_client_from_state_with_optimizer(self, path):
            _log({"event": "restore", "path": path}); return _Trainer(path)
        def close(self, status): return _Future(None)
    '''
)

FAKE_COOKBOOK = textwrap.dedent(
    """
    import json, shutil, os
    def build_lora_adapter(*, base_model, adapter_path, output_path, trust_remote_code=None):
        os.makedirs(output_path, exist_ok=True)
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            shutil.copy(os.path.join(adapter_path, name), os.path.join(output_path, name))
        config = json.load(open(os.path.join(output_path, "adapter_config.json")))
        config["base_model_name_or_path"] = base_model
        json.dump(config, open(os.path.join(output_path, "adapter_config.json"), "w"))
    """
)


def write_fake_sdk(root: Path, model: str) -> Path:
    site = root / "fake_sdk"
    (site / "tinker").mkdir(parents=True, exist_ok=True)
    (site / "tinker" / "__init__.py").write_text(FAKE_SDK)
    (site / "tinker" / "settings.json").write_text(json.dumps({"root": str(root), "model": model}))
    (site / "tinker_cookbook").mkdir(exist_ok=True)
    (site / "tinker_cookbook" / "__init__.py").write_text("")
    (site / "tinker_cookbook" / "weights.py").write_text(FAKE_COOKBOOK)
    return site


def served_call(version: str, index: int):
    """Rollout ``index``: a served call without tokens, as the inference service records it with the artifact
    ``version`` that answered."""
    from reef.core import AgentRecord, RequestType

    return AgentRecord.create(
        scenario=SCENARIO,
        request_type=RequestType.INFERENCE,
        payload={
            "messages": [{"role": "user", "content": f"bound {index}"}],
            "response": {"choices": [{"message": {"role": "assistant", "content": f"answer {index}"}}]},
            "artifact_id": version,
        },
        agent_record_id=f"i{index}",
    )


def stop_process(process) -> None:
    """Stop the component, then anything left in its process group.

    The signal goes to the component alone: its ``ray.shutdown`` stops the Ray processes it started, which takes
    about a second. A signal to the whole group kills those processes first, and the shutdown then waits for
    them for several seconds."""
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    with suppress(subprocess.TimeoutExpired):
        process.wait(timeout=60)
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)


@contextmanager
def inference_component(tmp_path):
    """Run the component for Reef's SGLang stack on a free loopback port; yield its URL and a control client.

    The client carries no serving-owner sequence. The receiver accepts its calls only until a control plane starts
    engines with request pinning; after that, the test gives the client the owner's sequence."""
    from reef.inference.remote.service import RemoteControl

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    log_path = tmp_path / "component.log"
    env = {key: value for key, value in os.environ.items() if not key.startswith(("REEF_", "RAY_"))}
    # The component's Ray session declares two pretend GPUs and the stand-in runs no CUDA code. Ray refuses that
    # count under an inherited device mask that names fewer devices, so the component gets no mask.
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.update(TOKENS, PYTHONPATH=os.pathsep.join([str(REPO_ROOT), env.get("PYTHONPATH", "")]))
    command = [sys.executable, "-m", "reef.inference.remote", "--engine", ENGINE, "--host", "127.0.0.1"]
    command += ["--port", str(port), "--work-dir", str(tmp_path / "remote-work"), "--ray-num-gpus", "2"]
    with log_path.open("w") as log:
        component = subprocess.Popen(
            command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
    control = RemoteControl(url, TOKENS["REEF_REMOTE_CONTROL_TOKEN"])
    try:
        deadline = time.monotonic() + 120
        while True:
            assert component.poll() is None, log_path.read_text()[-4000:]
            try:
                if control.call("GET", "health", timeout_s=5)["engines"] is False:
                    break  # serving, with no engines started yet
            except RuntimeError:
                pass  # not listening yet
            assert time.monotonic() < deadline, "component did not start:\n" + log_path.read_text()[-4000:]
            time.sleep(0.1)
        yield url, control
    finally:
        stop_process(component)


@contextmanager
def control_plane(tmp_path, monkeypatch, url, repository):
    """Build Tinker's component, its serving runtime and a Dispatcher over them, as the HTTP service does, and close both after.

    ``repository`` is the artifact backend factory; a restart passes the same one, which holds the snapshots.
    """
    from recipes.tttd.recipe import TTTDRecipe
    from reef.dispatcher import Dispatcher
    from reef.storage.sql_coordination import open_coordination
    from reef.storage.sqlite import SQLiteScenarioStorage
    from reef.train.pipeline import PipelineSettings

    if not REAL:
        monkeypatch.syspath_prepend(str(write_fake_sdk(tmp_path, MODEL)))
        for module in ("tinker", "tinker_cookbook", "tinker_cookbook.weights"):
            monkeypatch.delitem(sys.modules, module, raising=False)
    api_key = os.environ["TINKER_API_KEY"] if REAL else "fake-key"
    for name, value in {**TOKENS, "TINKER_API_KEY": api_key}.items():
        monkeypatch.setenv(name, value)
    raw = {
        "recipe": {"implementation": "recipes.tttd.recipe:TTTDRecipe"},
        # The serving executor of the component's SGLang stack. The engine, receiver and handler stay built in.
        "execution": {"rollout": SERVING_EXECUTOR},
        "pipeline": {
            "record2sample": {"config": {"groups_per_step": 1, "rollouts_per_group": 4}},
            "sample2job": {"config": {"job_samples": 4}},
        },
        "inference": {
            "model-path": MODEL,
            "backend": ENGINE,
            "num-gpus": 2,
            "handler-factory": "reef.inference.sglang.chat:SGLangInferenceHandler",
            "remote": {"url": url},
        },
        "training": {
            "backend": "tinker",
            "options": {
                "state-dir": str(tmp_path / "tinker"),
                "lora-rank": 32,
                "batch-size": 4,
                "kl-coef": 0.0,
            },
        },
    }
    config, _ = resolve_deployment_config(raw, None, tmp_path / "serve.yaml")
    assert [service["name"] for service in config["services"]] == ["reef"]
    component, inference = connect_training_objects(
        service_config_from_mapping(config), model_path=MODEL, default_parameters={}
    )
    dispatcher = None
    try:
        dispatcher = Dispatcher(
            TTTDRecipe(
                pipeline_stages=PipelineSettings.from_setting(
                    raw["pipeline"], spec=TTTDRecipe.pipeline_spec(), environ={}
                ),
            ),
            repository,
            live=LiveObjects(runtime=inference, training_component=component, training_facts=component.facts),
            local_artifact_dir=tmp_path / "staged",
            scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
            # A file, as a deployment keeps it: a restarted control plane reads what the engines serve.
            coordination=open_coordination(
                "sqlite", agent_record_dir=str(tmp_path / "records"), record_database_url=None
            ),
        )
        yield dispatcher, component, inference
    finally:
        if dispatcher is not None:
            dispatcher.close()
        component.close()
        inference.shutdown()


def loop_state(scenario) -> JobLoopState:
    """The job trainer's committed state: where the next job starts and the jobs that ended."""
    return JobLoopState.from_dict(scenario.trainer.state)


def train_one_step(dispatcher, inference) -> None:
    """Feed one TTTD group through the Dispatcher and wait until the scenario commits the job's end."""
    from reef.core import AgentRecord, RequestType

    scenario = dispatcher.get_or_create_scenario(SCENARIO)
    # The staleness check keeps only data of the artifact that the scenario serves.
    version = scenario.serving_artifact_id()
    for index in range(4):
        dispatcher.accept_record(served_call(version, index))
        dispatcher.accept_record(
            AgentRecord.create(
                scenario=SCENARIO,
                request_type=RequestType.REPORT,
                payload={
                    "score": float(index),
                    "references": [f"i{index}"],
                    "metadata": {
                        "algorithm": "tttd",
                        "step": 0,
                        "group": 0,
                        "rollout": index,
                        "groups_per_step": 1,
                        "rollouts_per_group": 4,
                        "comparison_set": "tttd-step-0-group-0",
                    },
                },
                references=(f"i{index}",),
                agent_record_id=f"r{index}",
            )
        )
    deadline = time.monotonic() + (1500 if REAL else 300)
    while loop_state(scenario).finished_jobs < 1 and time.monotonic() < deadline:
        time.sleep(0.05)
    state = loop_state(scenario)
    assert state.finished_jobs == 1, json.dumps(dispatcher.build_training_status(), default=str)[:3000]
    assert state.last_job.event == "finished", state.last_job


def served_adapters(tmp_path) -> list[Path]:
    return [path for path in (tmp_path / "remote-work").iterdir() if path.is_dir()]


def wait_for_serving(
    dispatcher: Dispatcher, inference: AdapterServingRuntime, *, adapter: bool = True
) -> ServedArtifact:
    """Wait for the selected artifact and return its published route: with ``adapter``, also wait for this
    runtime's recovery of its adapter. A base root has no adapter. The route counts when it names what the runtime
    serves now, which a route of an earlier control plane does not."""
    routes = dispatcher.coordination.served_models
    deadline = time.monotonic() + (1500 if REAL else 120)
    while time.monotonic() < deadline:
        item = routes.current_request(SCENARIO)
        selected = dispatcher.get_or_create_scenario(SCENARIO).artifact_selection().to_dict()
        if item is not None and item.payload.get("selection") == selected:
            # A restart can read the prior worker's completed request before recovering its adapter.
            recovered = inference.serving_adapter_runtime_load_id(SCENARIO) is not None
            route = routes.read(ScenarioModel(SCENARIO))
            published = route is not None and route.model_name == inference.served_model_name(SCENARIO)
            if item.state == "finished" and (recovered or not adapter) and published:
                return route
            assert item.state != "failed", item.error
        time.sleep(0.05)
    pytest.fail("the selected adapter did not start serving")


def owner_sequence(inference: AdapterServingRuntime) -> int:
    """The serving-owner sequence that this control plane's runtime sends with each control call to the component."""
    return inference.engines.service.control.owner_sequence


def generate(inference: AdapterServingRuntime, served: ServedArtifact) -> dict:
    """Send one native generation to the route ``served`` and return the engines' answer.

    The request is the one that the inference service makes for a route: the runtime's handler addresses it to the
    route's model name and names that route to the receiver. ``UpstreamStatusError`` when the receiver or the
    engines refuse it."""
    handler = inference.inference_handler
    request = handler.address_model({}, served.model_name)
    payload = handler.client.payload(request, [1, 2, 3], {"max_new_tokens": 1}, stream=False)
    artifact = InferenceArtifact(served.artifact_id, served_model=served.model_name)
    return asyncio.run(handler.post_json(handler.client.generate_path, payload, artifact=artifact))


def refusal_status(inference: AdapterServingRuntime, served: ServedArtifact) -> int:
    """The HTTP status with which the component refuses a generation for the route ``served``."""
    with pytest.raises(UpstreamStatusError) as refused:
        generate(inference, served)
    return refused.value.status


@pytest.mark.integration
def test_tinker_trains_in_the_control_plane_and_serves_on_an_inference_component(tmp_path, monkeypatch):
    from reef.artifact import InMemoryRepositoryBackend

    pytest.importorskip("ray")  # the component process starts its own Ray session
    # The default start is a base root, which control writes from the backend's facts: the engines serve the base
    # model for it and load no adapter.
    repository = InMemoryRepositoryBackend.factory(directory=tmp_path / "repository")
    for key in list(os.environ):
        if key.startswith(("REEF_", "RAY_")) and key != "REEF_TEST_TINKER_REAL":
            monkeypatch.delenv(key)
    with inference_component(tmp_path) as (url, control):
        with control_plane(tmp_path, monkeypatch, url, repository) as (dispatcher, _, inference):
            scenario = dispatcher.get_or_create_scenario(SCENARIO)
            base_route = wait_for_serving(dispatcher, inference, adapter=False)
            assert inference.serving_adapter_runtime_load_id(SCENARIO) is None
            initial = inference.base_version
            # The route of the base root names the base model under the load ID that the owner gave it. The receiver
            # admits a request only under such a bound name.
            assert inference.served_model_name(SCENARIO) == full_weight_name(MODEL, initial)
            assert base_route == ServedArtifact(scenario.serving_artifact_id(), url, full_weight_name(MODEL, initial))
            assert served_adapters(tmp_path) == []
            assert inference.base_url == url
            # The receiver pins requests, so it accepts a control call only with the owner's sequence.
            with pytest.raises(InferenceOwnerLostError):
                control.call("GET", "runtime-load-ids")
            first_owner = owner_sequence(inference)
            control.owner_sequence = first_owner
            health = control.call("GET", "health", timeout_s=30)
            assert (health["engines"], health["request_pinning"], health["binding_uncertain"]) == (True, True, False)
            assert control.call("GET", "runtime-load-ids")["runtime_load_ids"] == [initial] * 2
            # A request under the base route reaches the base model. A request without a bound name reaches nothing.
            answer = generate(inference, base_route)
            assert (answer["text"], answer["meta_info"]["weight_version"]) == (MODEL, initial)
            assert refusal_status(inference, ServedArtifact(base_route.artifact_id, url, MODEL)) == 421
            train_one_step(dispatcher, inference)
            adapter_route = wait_for_serving(dispatcher, inference)
            published = inference.serving_adapter_runtime_load_id(SCENARIO)
            assert published is not None and published != initial
            assert inference.served_model_name(SCENARIO) == adapter_name(SCENARIO, published)
            assert adapter_route == ServedArtifact(
                scenario.serving_artifact_id(), url, adapter_name(SCENARIO, published)
            )
            assert adapter_route.artifact_id != base_route.artifact_id
            assert control.call("GET", "runtime-load-ids")["runtime_load_ids"] == [published] * 2
            (adapter,) = served_adapters(tmp_path)
            assert (adapter / "adapter_config.json").is_file() and (adapter / "adapter_model.safetensors").is_file()
            # A request under the adapter route reaches the weights of the files that the component received.
            weights = adapter_weights_id(adapter)
            answer = generate(inference, adapter_route)
            assert (answer["text"], answer["meta_info"]["weight_version"]) == (weights, published)
            assert inference.inference_admission_status["open"]
            start = Path(loop_state(dispatcher.get_or_create_scenario(SCENARIO)).start)
            assert start.parent.parent == tmp_path / "tinker" / "jobs"
            if not REAL:
                events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
                # The first job makes a new LoRA of the model on the service; nothing saved an untrained adapter.
                assert [event["event"] for event in events if event["event"] in ("initialize", "restore")] == [
                    "initialize"
                ]
                # Serving downloads the adapter after training, in a session of its own.
                assert [event["rows"] for event in events if event["event"] == "forward_backward"][-1] == 4
                # The served rollouts carry no log probabilities: the trainer computes them at the job's start.
                assert [event["rows"] for event in events if event["event"] == "forward"] == [4]
        # Closing the control plane leaves the engines serving what they served: the published route still reaches
        # the adapter. ``inference`` is here the runtime of the closed control plane, of which only the handler acts.
        assert control.call("GET", "runtime-load-ids")["runtime_load_ids"] == [published] * 2
        assert generate(inference, adapter_route)["text"] == weights
        # A restarted control plane is a new serving owner. It takes over no engine: it starts a new stack on the
        # component and loads the committed adapter into it once, under the first load ID of its own incarnation.
        with control_plane(tmp_path, monkeypatch, url, repository) as (dispatcher, _, inference):
            scenario = dispatcher.get_or_create_scenario(SCENARIO)
            restored_route = wait_for_serving(dispatcher, inference)
            # The receiver now refuses the control calls of the first owner.
            with pytest.raises(InferenceOwnerLostError):
                control.call("GET", "runtime-load-ids")
            assert owner_sequence(inference) > first_owner
            control.owner_sequence = owner_sequence(inference)
            restored = inference.serving_adapter_runtime_load_id(SCENARIO)
            base = RuntimeLoadId.parse(inference.base_version)
            assert base.incarnation != RuntimeLoadId.parse(initial).incarnation
            assert restored == str(RuntimeLoadId(base.incarnation, base.sequence + 1))
            assert restored != published
            assert control.call("GET", "runtime-load-ids")["runtime_load_ids"] == [restored] * 2
            # The new route binds the same artifact under the new adapter name.
            assert restored_route == ServedArtifact(adapter_route.artifact_id, url, adapter_name(SCENARIO, restored))
            assert inference.served_model_name(SCENARIO) == adapter_name(SCENARIO, restored)
            (adapter,) = served_adapters(tmp_path)
            assert (adapter / "adapter_config.json").is_file()
            # The new stack serves the same weights under the new route only: the routes of the first owner name
            # bindings that its engines held, and those engines are gone.
            assert adapter_weights_id(adapter) == weights
            answer = generate(inference, restored_route)
            assert (answer["text"], answer["meta_info"]["weight_version"]) == (weights, restored)
            assert refusal_status(inference, adapter_route) == 421
            assert refusal_status(inference, base_route) == 421
            health = control.call("GET", "health", timeout_s=30)
            assert (health["engines"], health["request_pinning"], health["binding_uncertain"]) == (True, True, False)
            assert loop_state(scenario).start == str(start)
