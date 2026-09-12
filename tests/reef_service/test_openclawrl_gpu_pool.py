"""Opt-in Ray resource allocation with logical GPUs, without model kernels."""

import json
import os
import sys
from pathlib import Path

import pytest
import yaml
from reef_service.config_helpers import deployment_layout

from reef.runtime.executor.ray import RayExecutor
from reef.service.deploy.execution import validate_services
from reef.service.deploy.orchestrator import _Stack

pytestmark = pytest.mark.skipif(os.environ.get("REEF_TEST_RAY") != "1", reason="opt-in real Ray integration")


@pytest.mark.parametrize(
    "recipe,capacity",
    [
        ("sao/examples/sao", 2),
        ("tttd/examples/tttd", 2),
        ("tttd/examples/guidance_ttt", 2),
        ("openclawrl/examples/openclawrl", 5),
    ],
)
@pytest.mark.parametrize("external", [False, True], ids=["managed-local", "external"])
def test_local_training_controllers_leave_all_model_gpus_available(tmp_path, monkeypatch, recipe, capacity, external):
    ray = pytest.importorskip("ray")
    from ray.cluster_utils import Cluster

    root = Path(__file__).resolve().parents[2]
    config = deployment_layout(yaml.safe_load((root / "recipes" / recipe / "serve.yaml").read_text()))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(str(gpu) for gpu in range(capacity)))
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    real_init = ray.init

    def init(**options):
        if options["address"] == "local":
            options.update(num_cpus=capacity + 2, num_gpus=capacity, include_dashboard=False)
        return real_init(**options)

    monkeypatch.setattr(ray, "init", init)
    cluster = Cluster()
    stack = None
    try:
        if external:
            cluster.add_node(num_cpus=capacity + 2, num_gpus=capacity, include_dashboard=False)
            monkeypatch.setenv("RAY_ADDRESS", cluster.address)
        for service in config["services"]:
            output = tmp_path / f"{service['name']}.json"
            # The Slime driver joins the supplied runtime and reserves its
            # complete model group. These are logical GPUs, not model kernels.
            body = (
                "import json,os,time,ray; from pathlib import Path; "
                "ray.init(address=os.environ['RAY_ADDRESS']); "
                "from ray.util.placement_group import placement_group; "
                "available=ray.available_resources().get('GPU',0); "
                f"group=placement_group([{{'CPU':1,'GPU':1}}]*{capacity}); "
                "ray.get(group.ready(),timeout=30); "
                f"Path({str(output)!r}).write_text(json.dumps({{'available':available,"
                "'address':ray.get_runtime_context().gcs_address})); time.sleep(120)"
                if service["name"] == "slime-driver"
                else "import json,os,time; from pathlib import Path; "
                f"Path({str(output)!r}).write_text(json.dumps(os.environ['RAY_ADDRESS'])); time.sleep(120)"
            )
            service.update(command=[sys.executable, "-c", body], ready=f"test -f {output}", ready_timeout=60)
        stack = _Stack(config, validate_services(config, "test.yaml"), tmp_path, 60, tmp_path / "input.yaml")
        stack.start()
        driver = json.loads((tmp_path / "slime-driver.json").read_text())
        address = stack.config["reef"]["ray_address"]
        assert driver == {"available": capacity, "address": address}
        assert json.loads((tmp_path / "reef.json").read_text()) == address
        assert not any(isinstance(executor, RayExecutor) for executor in stack._executors.values())
        for name in ("slime-driver", "reef"):
            snapshot = yaml.safe_load((tmp_path / name / "runtime.yaml").read_text())
            assert snapshot["reef"]["ray_address"] == address
    finally:
        if stack is not None:
            stack.shutdown(grace=1)
        assert not ray.is_initialized()
        if external:
            real_init(address=cluster.address)
            assert ray.cluster_resources()["GPU"] == capacity
        ray.shutdown()
        cluster.shutdown()
