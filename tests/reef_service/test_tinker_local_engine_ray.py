"""Tinker training behind Reef's model driver with a local engine, end to end on CPU.

A local Ray cluster declares two pretend GPUs; the driver process reserves
them, starts the stub engine (``reef_service._stub_engine``), the Tinker
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
import subprocess
import sys
import textwrap
import time
from pathlib import Path

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
    (site / "tinker").mkdir(parents=True)
    (site / "tinker" / "__init__.py").write_text(FAKE_SDK)
    (site / "tinker" / "settings.json").write_text(json.dumps({"root": str(root), "model": model}))
    (site / "tinker_cookbook").mkdir()
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


@pytest.mark.integration
def test_tinker_trains_behind_the_model_driver_with_a_local_engine(tmp_path, monkeypatch):
    ray = pytest.importorskip("ray")
    for key in list(os.environ):
        if key.startswith("REEF_") and key != "REEF_TEST_TINKER_REAL":
            monkeypatch.delenv(key)
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    site = _write_fake_sdk(tmp_path, MODEL if REAL else str(model_dir))
    api_key = os.environ["TINKER_API_KEY"] if REAL else "fake-key"
    monkeypatch.setenv("TINKER_API_KEY", api_key)
    raw = {
        "schema-version": 2,
        "reef": {"run-dir": str(tmp_path / "stack")},
        "recipe": {
            "implementation": "recipes.tttd.recipe:TTTDRecipe",
            "config": {"groups-per-step": 1, "rollouts-per-group": 4, "checkpoint-every-n-versions": 1},
        },
        "inference": {
            "model-path": MODEL if REAL else str(model_dir),
            "backend": "reef_service._stub_engine:create_inference",
            "num-gpus": 2,
        },
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

    ray.init(address="local", num_cpus=8, num_gpus=2, include_dashboard=False, namespace=NAMESPACE)
    driver = None
    dispatcher = None
    pair = None
    log_path = tmp_path / "driver.log"
    try:
        address = ray.get_runtime_context().gcs_address
        ready = tmp_path / "ready"
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("REEF_", "RAY_")) or key == "REEF_TEST_TINKER_REAL"
        }
        env.update(
            REEF_CONFIG=str(config_path),
            RAY_ADDRESS=address,
            REEF_RAY_NAMESPACE=NAMESPACE,
            REEF_RAY_ACTOR_NAME=ACTOR,
            REEF_BRIDGE_READY_FILE=str(ready),
            TINKER_API_KEY=api_key,
            PYTHONPATH=os.pathsep.join(
                ([] if REAL else [str(site)]) + [str(REPO_ROOT), str(REPO_ROOT / "tests"), env.get("PYTHONPATH", "")]
            ),
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
                "inference_runtime": "sglang",
                "actor_name": ACTOR,
                "namespace": NAMESPACE,
                "ray_address": address,
                "inference_timeout_s": 600.0,
                "train_timeout_s": 1800.0,
            },
            model_path=raw["inference"]["model-path"],
        )
        training, inference = pair
        assert training.concurrent_training_scenarios  # per-scenario adapters, as the coordinator reports
        engine = ray.get_actor("reef-test-engine", namespace=NAMESPACE)
        assert ray.get(engine.state.remote())["versions"] == [inference.serving_runtime_load_id()] * 2

        from reef_service._threshold_processor import ThresholdProcessor  # noqa: F401  (import side effects none)

        from recipes.tttd.recipe import TTTDRecipe
        from reef.artifact import InMemoryRepositoryBackend
        from reef.core import AgentRecord, RequestType
        from reef.dispatcher import Dispatcher
        from reef.storage.sqlite import SQLiteScenarioStorage

        initial = tmp_path / "initial"
        initial.mkdir()
        dispatcher = Dispatcher(
            TTTDRecipe(training, runtime=inference, groups_per_step=1, rollouts_per_group=4),
            InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
            local_artifact_dir=tmp_path / "staged",
            agent_record_dir=tmp_path / "records",
            scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
        )
        scenario_name = "erdos"
        scenario = dispatcher.get_or_create_scenario(scenario_name)
        version = inference.serving_runtime_load_id()
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
            assert driver.poll() is None, log_path.read_text()[-4000:]
            time.sleep(1)
        assert scenario.scenario_step == 1, json.dumps(dispatcher.build_training_status(), default=str)[:3000]

        loads = ray.get(engine.loads.remote())
        published = inference.current_runtime_load_id()
        assert loads == [(adapter_name(scenario_name, published), loads[0][1], published)]
        adapter = Path(loads[0][1])
        assert (adapter / "adapter_config.json").is_file() and (adapter / "adapter_model.safetensors").is_file()
        assert adapter.parent.name == "rollout_0" and adapter.parent.parent == tmp_path / "tinker" / "checkpoints"
        state = ray.get(engine.state.remote())
        assert state["versions"] == [published] * 2 and not state["paused"] and state["terminated"] == 0
        assert inference.inference_admission_status["open"]
        incumbent = json.loads((tmp_path / "tinker" / "scenarios" / scenario_name / "incumbent.json").read_text())
        assert incumbent["runtime_load_id"] == published and incumbent["rollout_id"] == 0
        if not REAL:
            events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
            assert [event["event"] for event in events if event["event"] in ("initialize", "restore")] == [
                "initialize",
                "restore",
            ]
            assert events[-1]["event"] == "forward_backward" and events[-1]["rows"] == 4
    finally:
        if dispatcher is not None:
            dispatcher.close()
        if pair is not None:
            pair[0].shutdown()
            pair[1].shutdown()
        if driver is not None and driver.poll() is None:
            os.killpg(driver.pid, signal.SIGTERM)
            try:
                driver.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(driver.pid, signal.SIGKILL)
        ray.shutdown()
