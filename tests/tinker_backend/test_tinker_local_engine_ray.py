"""Tinker training in the control plane, served by engines on an inference component, end to end on CPU.

The component (``python -m reef.inference.remote``) runs in its own process
with its own Ray session that declares two pretend GPUs, and starts the stub
engine (``tests.support._stub_engine``). The test process is the control
plane: it builds Tinker's runtime pair as the HTTP service does and drives one
TTTD step through the Dispatcher, then restarts. The Tinker SDK and the
cookbook converter are faked unless ``REEF_TEST_TINKER_REAL=1`` and
``TINKER_API_KEY`` are set, in which case the trainer, the checkpoint download
and the PEFT conversion are real.
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

import pytest

from reef.runtime.deployment import RuntimeRegistry
from reef.service.deploy.orchestrator import resolve_deployment_config
from reef.service.deploy.service_config import service_config_from_mapping
from reef.service.deploy.training import training_deployment_for

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL = "Qwen/Qwen3-8B"
REAL = os.environ.get("REEF_TEST_TINKER_REAL") == "1" and bool(os.environ.get("TINKER_API_KEY"))
ENGINE = "tests.support._stub_engine:create_inference"
TOKENS = {"REEF_REMOTE_CONTROL_TOKEN": "control-secret", "REEF_REMOTE_INFERENCE_TOKEN": "inference-secret"}
SCENARIO = "erdos"

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


def stop_process(process) -> None:
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)


@contextmanager
def inference_component(tmp_path):
    """Run the component with the stub engine on a free loopback port; yield its URL and its control client."""
    from reef.inference.remote.service import RemoteControl

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    log_path = tmp_path / "component.log"
    env = {key: value for key, value in os.environ.items() if not key.startswith(("REEF_", "RAY_"))}
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
                control.call("GET", "health", timeout_s=5)
            except RuntimeError as exc:
                if "409" in str(exc):
                    break  # serving, with no engines started yet
            assert time.monotonic() < deadline, "component did not start:\n" + log_path.read_text()[-4000:]
            time.sleep(0.5)
        yield url, control
    finally:
        stop_process(component)


@contextmanager
def control_plane(tmp_path, monkeypatch, url, repository):
    """Build Tinker's runtime pair and a Dispatcher over it, as the HTTP service does, and close both after.

    ``repository`` is the artifact backend factory; a restart passes the same one, which holds the releases.
    """
    from recipes.tttd.recipe import TTTDRecipe
    from reef.dispatcher import Dispatcher
    from reef.storage.sqlite import SQLiteScenarioStorage
    from reef.train.pipeline import PipelineConfig

    if not REAL:
        monkeypatch.syspath_prepend(str(_write_fake_sdk(tmp_path, MODEL)))
        for module in ("tinker", "tinker_cookbook", "tinker_cookbook.weights"):
            monkeypatch.delitem(sys.modules, module, raising=False)
    api_key = os.environ["TINKER_API_KEY"] if REAL else "fake-key"
    monkeypatch.setenv("TINKER_API_KEY", api_key)
    raw = {
        "recipe": {"implementation": "recipes.tttd.recipe:TTTDRecipe"},
        "pipeline": {"record2record": {"config": {"groups_per_step": 1, "rollouts_per_group": 4}}},
        "inference": {
            "model-path": MODEL,
            "backend": ENGINE,
            "num-gpus": 2,
            "handler-factory": "reef.inference.sglang.chat.SGLangInferenceHandler",
            "remote": {"url": url},
        },
        "training": {
            "backend": "tinker",
            "options": {"state-dir": str(tmp_path / "tinker"), "lora-rank": 32, "batch-size": 4, "kl-coef": 0.0},
        },
    }
    config, _ = resolve_deployment_config(raw, None, tmp_path / "serve.yaml")
    assert [service["name"] for service in config["services"]] == ["reef"]
    runtime_config = training_deployment_for("tinker").runtime_config(service_config_from_mapping(config).__dict__)
    training, inference = RuntimeRegistry().build(
        runtime_config, model_path=MODEL, environ={**TOKENS, "TINKER_API_KEY": api_key}
    )
    dispatcher = None
    try:
        dispatcher = Dispatcher(
            TTTDRecipe(
                training,
                runtime=inference,
                pipeline_stages=PipelineConfig.from_setting(
                    raw["pipeline"], spec=TTTDRecipe.pipeline_spec(), environ={}
                ),
            ),
            repository,
            local_artifact_dir=tmp_path / "staged",
            scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
        )
        yield dispatcher, training, inference
    finally:
        if dispatcher is not None:
            dispatcher.close()
        training.shutdown()
        inference.shutdown()


def train_one_step(dispatcher, inference) -> None:
    """Feed one TTTD group through the Dispatcher and wait until the scenario commits it."""
    from reef.core import AgentRecord, RequestType

    scenario = dispatcher.get_or_create_scenario(SCENARIO)
    version = inference.serving_runtime_load_id()
    for index in range(4):
        item = _training_item(version, index)
        dispatcher.accept_record(
            AgentRecord.create(
                scenario=SCENARIO,
                request_type=RequestType.INFERENCE,
                payload={"response": {"training": dict(item.training)}},
                agent_record_id=f"i{index}",
            )
        )
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
    while scenario.scenario_step < 1 and time.monotonic() < deadline:
        time.sleep(1)
    assert scenario.scenario_step == 1, json.dumps(dispatcher.build_training_status(), default=str)[:3000]


def served_adapters(tmp_path) -> list[Path]:
    return [path for path in (tmp_path / "remote-work").iterdir() if path.is_dir()]


@pytest.mark.integration
def test_tinker_trains_in_the_control_plane_and_serves_on_an_inference_component(tmp_path, monkeypatch):
    from reef.artifact import InMemoryRepositoryBackend

    pytest.importorskip("ray")  # the component process starts its own Ray session
    initial = tmp_path / "initial"
    initial.mkdir()
    repository = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    for key in list(os.environ):
        if key.startswith(("REEF_", "RAY_")) and key != "REEF_TEST_TINKER_REAL":
            monkeypatch.delenv(key)
    with inference_component(tmp_path) as (url, control):
        with control_plane(tmp_path, monkeypatch, url, repository) as (dispatcher, training, inference):
            initial = inference.serving_runtime_load_id()
            assert control.call("GET", "runtime-load-ids")["runtime_load_ids"] == [initial] * 2
            assert inference.base_url == url and training.concurrent_training_scenarios
            train_one_step(dispatcher, inference)
            published = inference.serving_adapter_runtime_load_id(SCENARIO)
            assert published is not None and published != initial
            assert control.call("GET", "runtime-load-ids")["runtime_load_ids"] == [published] * 2
            (adapter,) = served_adapters(tmp_path)
            assert (adapter / "adapter_config.json").is_file() and (adapter / "adapter_model.safetensors").is_file()
            assert inference.inference_admission_status["open"]
            start = Path(training.start(SCENARIO))
            assert start.parent == tmp_path / "tinker" / "jobs"
            if not REAL:
                events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
                assert [event["event"] for event in events if event["event"] in ("initialize", "restore")] == [
                    "initialize",
                    "restore",
                ]
                assert events[-1]["event"] == "forward_backward" and events[-1]["rows"] == 4
        # Closing the control plane stops the engines; the component keeps serving.
        with pytest.raises(RuntimeError, match="409"):
            control.call("GET", "runtime-load-ids")
        assert served_adapters(tmp_path) == []
        # A restarted control plane starts the engines again and loads the committed adapter before it serves.
        with control_plane(tmp_path, monkeypatch, url, repository) as (dispatcher, training, inference):
            dispatcher.get_or_create_scenario(SCENARIO)
            served = inference.serving_adapter_runtime_load_id(SCENARIO)
            assert served is not None and served != published
            assert control.call("GET", "runtime-load-ids")["runtime_load_ids"] == [served] * 2
            (adapter,) = served_adapters(tmp_path)
            assert (adapter / "adapter_config.json").is_file()
            assert training.start(SCENARIO) == str(start)
