"""Tinker training behind Reef's model driver with a local engine, end to end on CPU.

A local Ray cluster declares two pretend GPUs; the driver process reserves
them, starts the stub engine (``tests.support._stub_engine``), the Tinker
trainer and Reef's coordinator; the test process connects like the HTTP
service does and drives one TTTD step through the Dispatcher. The Tinker SDK
and the cookbook converter are faked on the driver's path unless
``REEF_TEST_TINKER_REAL=1`` and ``TINKER_API_KEY`` are set, in which case the
trainer, the checkpoint download and the PEFT conversion are real.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import textwrap
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from reef.runtime.deployment import RuntimeRegistry
from reef.service.deploy.orchestrator import resolve_deployment_config
from reef.surface.adapter import adapter_name

REPO_ROOT = Path(__file__).resolve().parents[2]
NAMESPACE = "reef-local-engine-test"
ACTOR = "reef-test-bridge"
MODEL = "Qwen/Qwen3-8B"
REAL = os.environ.get("REEF_TEST_TINKER_REAL") == "1" and bool(os.environ.get("TINKER_API_KEY"))

FAKE_SDK = textwrap.dedent(
    '''
    """Offline Tinker SDK: immutable checkpoints as tinker:// paths, archives as local tar files."""
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
    class AdamParams(SimpleNamespace): pass
    class SamplingParams(SimpleNamespace): pass

    class _Trainer:
        def __init__(self, origin): self.origin = origin
        def forward_backward(self, data, loss_fn):
            _log({"event": "forward_backward", "rows": len(data), "loss_fn": loss_fn, "from": self.origin})
            return _Future(SimpleNamespace(metrics={"loss:sum": -1.0}))
        def optim_step(self, params): return _Future(None)
        def _save(self, name, kind):
            path = f"tinker://fake:train:0/{kind}/{name}"
            return _Future(SimpleNamespace(path=path))
        def save_state(self, name, ttl_seconds=None): return self._save(name, "weights")
        def save_weights_for_sampler(self, name, ttl_seconds=None): return self._save(name, "sampler_weights")

    class _Sampler:
        def get_base_model(self): return MODEL
        def get_tokenizer(self): raise AssertionError("the trainer never renders")
        def compute_logprobs(self, model_input): return _Future([None] + [-0.5] * (model_input.length - 1))

    class _Rest:
        def get_weights_info_by_tinker_path(self, path):
            return _Future(SimpleNamespace(base_model=MODEL, is_lora=True, lora_rank=32))
        def get_checkpoint_archive_url_from_tinker_path(self, path):
            archive = os.path.join(ROOT, uuid.uuid4().hex + ".tar")
            with tarfile.open(archive, "w") as tar:
                for name, body in (("adapter_config.json", json.dumps({"r": 32, "lora_alpha": 32}).encode()),
                                   ("adapter_model.safetensors", path.encode())):
                    info = tarfile.TarInfo(name); info.size = len(body); tar.addfile(info, io.BytesIO(body))
            return _Future(SimpleNamespace(url="file://" + archive))

    class ServiceClient:
        def __init__(self, **kwargs): _log({"event": "session", "keys": sorted(kwargs)})
        def create_sampling_client(self, base_model=None, model_path=None): return _Sampler()
        def create_rest_client(self): return _Rest()
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


def _write_fake_sdk(root: Path, model: str) -> Path:
    site = root / "fake_sdk"
    (site / "tinker").mkdir(parents=True, exist_ok=True)
    (site / "tinker" / "__init__.py").write_text(FAKE_SDK)
    (site / "tinker" / "settings.json").write_text(json.dumps({"root": str(root), "model": model}))
    (site / "tinker_cookbook").mkdir(exist_ok=True)
    (site / "tinker_cookbook" / "__init__.py").write_text("")
    (site / "tinker_cookbook" / "weights.py").write_text(FAKE_COOKBOOK)
    return site


def _training_item(version: str, index: int):
    from reef.core.batches import TrajectoryItem

    return TrajectoryItem(
        {
            "schema_version": "ATIF-v1.6",
            "agent": {"name": "test"},
            "steps": [{"step_id": 1}],
            "extra": {
                "reef": {
                    "training": {
                        "tokens": [10, 11, 20 + index, 21],
                        "loss_mask": [1, 1],
                        "rollout_log_probs": [-0.2, -0.4],
                        "runtime_load_id": version,
                    }
                }
            },
        }
    )


@contextmanager
def model_driver(tmp_path, monkeypatch, inference, runtime, *, ray_gpus, driver_env=None):
    """Run the model driver for a Tinker deployment with this ``inference`` section; connect like the HTTP service."""
    ray = pytest.importorskip("ray")
    for key in list(os.environ):
        if key.startswith("REEF_") and key != "REEF_TEST_TINKER_REAL":
            monkeypatch.delenv(key)
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    # The HTTP service side of this process reads the same variables as the driver.
    for key, value in (driver_env or {}).items():
        monkeypatch.setenv(key, value)
    site = _write_fake_sdk(tmp_path, inference["model-path"])
    api_key = os.environ["TINKER_API_KEY"] if REAL else "fake-key"
    monkeypatch.setenv("TINKER_API_KEY", api_key)
    raw = {
        "reef": {"run-dir": str(tmp_path / "stack")},
        "recipe": {"implementation": "recipes.tttd.recipe:TTTDRecipe"},
        "pipeline": {"record2record": {"config": {"groups_per_step": 1, "rollouts_per_group": 4}}},
        "inference": inference,
        "training": {
            "backend": "tinker",
            "options": {"state-dir": str(tmp_path / "tinker"), "lora-rank": 32, "batch-size": 1, "kl-coef": 0.0},
        },
        "storage": {
            "artifact-repository": str(tmp_path / "artifacts.git"),
            "artifact-work-dir": str(tmp_path / "artifact-work"),
            "artifact-cache-dir": str(tmp_path / "artifact-cache"),
            "agent-record-dir": str(tmp_path / "agent-record"),
        },
    }
    config, _ = resolve_deployment_config(raw, None, tmp_path / "serve.yaml")
    assert [service["name"] for service in config["services"]] == ["tinker-driver", "reef"]
    config_path = tmp_path / "reef-config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    ray.init(address="local", num_cpus=8, num_gpus=ray_gpus, include_dashboard=False, namespace=NAMESPACE)
    driver = None
    pair = None
    log_path = tmp_path / "driver.log"
    try:
        address = ray.get_runtime_context().gcs_address
        ready = tmp_path / "ready"
        ready.unlink(missing_ok=True)  # a restarted driver writes it again
        # Ray 2.59 starts a local cluster with token auth and records the mode in RAY_AUTH_MODE;
        # a driver that loses it cannot join the cluster.
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("REEF_", "RAY_")) or key.startswith("RAY_AUTH_") or key == "REEF_TEST_TINKER_REAL"
        }
        env.update(
            REEF_CONFIG=str(config_path),
            RAY_ADDRESS=address,
            REEF_RAY_NAMESPACE=NAMESPACE,
            REEF_RAY_ACTOR_NAME=ACTOR,
            REEF_BRIDGE_READY_FILE=str(ready),
            TINKER_API_KEY=api_key,
            PYTHONPATH=os.pathsep.join(([] if REAL else [str(site)]) + [str(REPO_ROOT), env.get("PYTHONPATH", "")]),
            **(driver_env or {}),
        )
        with log_path.open("w") as log:
            driver = subprocess.Popen(
                [sys.executable, "-m", "reef.service.training_driver"],
                cwd=REPO_ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        deadline = time.monotonic() + (900 if REAL else 300)
        while not ready.is_file():
            assert driver.poll() is None, log_path.read_text()[-4000:]
            assert time.monotonic() < deadline, "model driver did not become ready:\n" + log_path.read_text()[-4000:]
            time.sleep(1)

        pair = RuntimeRegistry().build(
            {
                "type": "coordinator_training",
                "actor_name": ACTOR,
                "namespace": NAMESPACE,
                "ray_address": address,
                "inference_timeout_s": 600.0,
                "train_timeout_s": 1800.0,
                **runtime,
            },
            model_path=inference["model-path"],
        )
        yield SimpleNamespace(ray=ray, training=pair[0], inference=pair[1], driver=driver, log_path=log_path, raw=raw)
    finally:
        if pair is not None:
            pair[0].shutdown()
            pair[1].shutdown()
        stop_process(driver)
        ray.shutdown()


def stop_process(process) -> None:
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)


@contextmanager
def one_trained_step(stack, tmp_path, scenario_name: str):
    """Feed one TTTD group through the Dispatcher, wait for its publication, and keep the Dispatcher open."""
    from tests.support._threshold_processor import threshold_pipeline  # noqa: F401  (import side effects none)

    from recipes.tttd.recipe import TTTDRecipe
    from reef.artifact import InMemoryRepositoryBackend
    from reef.core import AgentRecord, RequestType
    from reef.dispatcher import Dispatcher
    from reef.storage.sqlite import SQLiteScenarioStorage
    from reef.train.pipeline import PipelineConfig

    initial = tmp_path / "initial"
    initial.mkdir()
    dispatcher = Dispatcher(
        TTTDRecipe(
            stack.training,
            runtime=stack.inference,
            pipeline_stages=PipelineConfig.from_setting(
                stack.raw["pipeline"], spec=TTTDRecipe.pipeline_spec(), environ={}
            ),
        ),
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "staged",
        scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
    )
    try:
        scenario = dispatcher.get_or_create_scenario(scenario_name)
        version = stack.inference.serving_runtime_load_id()
        for index in range(4):
            item = _training_item(version, index)
            dispatcher.accept_record(
                AgentRecord.create(
                    scenario=scenario_name,
                    request_type=RequestType.INFERENCE,
                    payload={"response": {"training": dict(item.training)}},
                    agent_record_id=f"i{index}",
                )
            )
            dispatcher.accept_record(
                AgentRecord.create(
                    scenario=scenario_name,
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
        while scenario.scenario_step < 1 and time.monotonic() < deadline:
            assert stack.driver.poll() is None, stack.log_path.read_text()[-4000:]
            time.sleep(1)
        assert scenario.scenario_step == 1, json.dumps(dispatcher.build_training_status(), default=str)[:3000]
        yield
    finally:
        dispatcher.close()


def assert_tinker_trained_once(tmp_path, scenario_name: str, published: str) -> None:
    incumbent = json.loads((tmp_path / "tinker" / "scenarios" / scenario_name / "incumbent.json").read_text())
    assert incumbent["runtime_load_id"] == published and incumbent["rollout_id"] == 0
    if not REAL:
        events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
        assert [event["event"] for event in events if event["event"] in ("initialize", "restore")] == [
            "initialize",
            "restore",
        ]
        assert events[-1]["event"] == "forward_backward" and events[-1]["rows"] == 4


@pytest.mark.integration
def test_tinker_trains_behind_the_model_driver_with_a_local_engine(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    inference = {
        "model-path": MODEL if REAL else str(model_dir),
        "backend": "tests.support._stub_engine:create_inference",
        "num-gpus": 2,
    }
    with model_driver(tmp_path, monkeypatch, inference, {"inference_runtime": "sglang"}, ray_gpus=2) as stack:
        ray, training, inference_runtime = stack.ray, stack.training, stack.inference
        assert training.concurrent_training_scenarios  # per-scenario adapters, as the coordinator reports
        engine = ray.get_actor("reef-test-engine", namespace=NAMESPACE)
        assert ray.get(engine.state.remote())["versions"] == [inference_runtime.serving_runtime_load_id()] * 2
        scenario_name = "erdos"
        with one_trained_step(stack, tmp_path, scenario_name):
            loads = ray.get(engine.loads.remote())
            published = inference_runtime.current_runtime_load_id()
            assert loads == [(adapter_name(scenario_name, published), loads[0][1], published)]
            adapter = Path(loads[0][1])
            assert (adapter / "adapter_config.json").is_file() and (adapter / "adapter_model.safetensors").is_file()
            assert adapter.parent.name == "rollout_0" and adapter.parent.parent == tmp_path / "tinker" / "checkpoints"
            state = ray.get(engine.state.remote())
            assert state["versions"] == [published] * 2 and not state["paused"] and state["terminated"] == 0
            assert inference_runtime.inference_admission_status["open"]
            assert_tinker_trained_once(tmp_path, scenario_name, published)


@pytest.mark.integration
def test_tinker_trains_behind_the_model_driver_with_a_remote_engine(tmp_path, monkeypatch):
    """The stub engines run in a separate component process with its own Ray session; adapters travel over HTTP."""
    pytest.importorskip("ray")  # the component process starts its own Ray session
    from reef.inference.remote.service import RemoteControl
    from reef.inference.sglang.chat import SGLangInferenceHandler

    tokens = {"REEF_REMOTE_CONTROL_TOKEN": "control-secret", "REEF_REMOTE_INFERENCE_TOKEN": "inference-secret"}
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    work_dir = tmp_path / "remote-work"
    component_log = tmp_path / "component.log"
    component_env = {key: value for key, value in os.environ.items() if not key.startswith(("REEF_", "RAY_"))}
    component_env.update(tokens, PYTHONPATH=os.pathsep.join([str(REPO_ROOT), component_env.get("PYTHONPATH", "")]))
    engine = "tests.support._stub_engine:create_inference"
    with component_log.open("w") as log:
        component = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "reef.inference.remote",
                "--engine",
                engine,
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--work-dir",
                str(work_dir),
                "--ray-num-gpus",
                "2",
            ],
            cwd=REPO_ROOT,
            env=component_env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    control = RemoteControl(url, tokens["REEF_REMOTE_CONTROL_TOKEN"])
    try:
        deadline = time.monotonic() + 120
        while True:
            assert component.poll() is None, component_log.read_text()[-4000:]
            try:
                control.call("GET", "health", timeout_s=5)
            except RuntimeError as exc:
                if "409" in str(exc):
                    break  # serving, with no engines started yet
            assert time.monotonic() < deadline, "component did not start:\n" + component_log.read_text()[-4000:]
            time.sleep(0.5)

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        inference = {
            "model-path": MODEL if REAL else str(model_dir),
            "backend": engine,
            "num-gpus": 2,
            "remote": {"url": url},
        }
        runtime = {
            "inference_runtime": "remote",
            "inference_runtime_options": {"engine": engine},
            "inference_handler_factory": SGLangInferenceHandler,
        }
        with model_driver(tmp_path, monkeypatch, inference, runtime, ray_gpus=0, driver_env=tokens) as stack:
            initial = stack.inference.serving_runtime_load_id()
            assert control.call("GET", "runtime-load-ids")["runtime_load_ids"] == [initial] * 2
            assert stack.inference.base_url == url
            scenario_name = "erdos"
            with one_trained_step(stack, tmp_path, scenario_name):
                published = stack.inference.current_runtime_load_id()
                assert published != initial
                assert control.call("GET", "runtime-load-ids")["runtime_load_ids"] == [published] * 2
                (adapter,) = [path for path in work_dir.iterdir() if path.is_dir()]
                assert (adapter / "adapter_config.json").is_file()
                assert (adapter / "adapter_model.safetensors").is_file()
                assert stack.inference.inference_admission_status["open"]
                assert_tinker_trained_once(tmp_path, scenario_name, published)
        # Closing the deployment stops the remote engines; the component keeps serving.
        with pytest.raises(RuntimeError, match="409"):
            control.call("GET", "runtime-load-ids")
        assert [path for path in work_dir.iterdir() if path.is_dir()] == []
        # A restarted driver starts the engines again and reloads the committed adapter before it serves.
        with model_driver(tmp_path, monkeypatch, inference, runtime, ray_gpus=0, driver_env=tokens) as stack:
            assert stack.inference.serving_runtime_load_id() == published
            assert control.call("GET", "runtime-load-ids")["runtime_load_ids"] == [published] * 2
            (adapter,) = [path for path in work_dir.iterdir() if path.is_dir()]
            assert (adapter / "adapter_config.json").is_file()
    finally:
        stop_process(component)
