"""Opt-in Ray resource allocation with logical GPUs, without model kernels."""

import json
import os
import sys
from pathlib import Path

import pytest
import yaml

from reef.runtime.executor.ray import RayExecutor
from reef.service.deploy.config import validate_services
from reef.service.deploy.orchestrator import _Stack

pytestmark = pytest.mark.skipif(os.environ.get("REEF_TEST_RAY") != "1", reason="opt-in real Ray integration")


@pytest.mark.parametrize("recipe", ["sao/examples/sao", "tttd/examples/tttd", "tttd/examples/guidance_ttt"])
@pytest.mark.parametrize("external", [False, True], ids=["managed-local", "external"])
def test_local_training_controllers_leave_all_model_gpus_available(tmp_path, monkeypatch, recipe, external):
    ray = pytest.importorskip("ray")
    from ray.cluster_utils import Cluster

    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / "recipes" / recipe / "serve.yaml").read_text())
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    real_init = ray.init

    def init(**options):
        if options["address"] == "local":
            options.update(num_cpus=4, num_gpus=2, include_dashboard=False)
        return real_init(**options)

    monkeypatch.setattr(ray, "init", init)
    cluster = Cluster()
    stack = None
    try:
        if external:
            cluster.add_node(num_cpus=4, num_gpus=2, include_dashboard=False)
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
                "group=placement_group([{'CPU':1,'GPU':1}]*2); "
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
        assert driver == {"available": 2, "address": address}
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
            assert ray.cluster_resources()["GPU"] == 2
        ray.shutdown()
        cluster.shutdown()


@pytest.mark.parametrize("external", [False, True], ids=["managed-local", "external"])
def test_inference_services_leave_five_disjoint_gpus_for_slime(tmp_path, monkeypatch, external):
    ray = pytest.importorskip("ray")
    from ray._private.accelerators import get_accelerator_manager_for_resource
    from ray.cluster_utils import Cluster
    from ray.util.placement_group import placement_group, remove_placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / "recipes/openclawrl/examples/openclawrl/serve.yaml").read_text())
    gpu_pool = [str(gpu) for gpu in range(1, 8)]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(gpu_pool))
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    real_init = ray.init

    def init(**options):
        if options["address"] == "local":
            options.update(num_cpus=8, num_gpus=7, include_dashboard=False)
        return real_init(**options)

    monkeypatch.setattr(ray, "init", init)
    cluster = Cluster()
    stack = None
    group = None
    actors = []
    try:
        if external:
            cluster.add_node(num_cpus=8, num_gpus=7, include_dashboard=False)
            monkeypatch.setenv("RAY_ADDRESS", cluster.address)
        assert not ray.is_initialized()
        services = []
        outputs = []
        for original in config["services"]:
            if original["name"] not in ("prm-sglang", "user-llm-sglang"):
                continue
            output = tmp_path / f"{original['name']}.json"
            outputs.append(output)
            # Keep the shipped resource declarations; replace model loading
            # with a subprocess that records its inherited CUDA visibility.
            service = {key: value for key, value in original.items() if key not in ("depends_on", "endpoint")}
            service.update(
                command=[
                    sys.executable,
                    "-c",
                    "import json,os,time; from pathlib import Path; "
                    f"Path({str(output)!r}).write_text(json.dumps(os.environ['CUDA_VISIBLE_DEVICES'])); "
                    "time.sleep(120)",
                ],
                ready=f"test -f {output}",
                ready_timeout=30,
            )
            services.append(service)
        driver_output = tmp_path / "driver-address"
        services.append(
            {
                "name": "slime-driver",
                "depends_on": ["prm-sglang", "user-llm-sglang"],
                "command": [
                    sys.executable,
                    "-c",
                    "import os,time; from pathlib import Path; "
                    f"Path({str(driver_output)!r}).write_text(os.environ['RAY_ADDRESS']); time.sleep(120)",
                ],
                "ready": f"test -f {driver_output}",
            }
        )
        config["services"] = services
        stack = _Stack(config, validate_services(config, "test.yaml"), tmp_path, 30, tmp_path / "input.yaml")
        stack.start()
        assert (
            driver_output.read_text() == ray.get_runtime_context().gcs_address == stack.config["reef"]["ray_address"]
        )

        def assigned_gpu(worker):
            return str(ray.get_gpu_ids()[0])

        service_gpus = [
            executor.rpc(0, "__ray_call__", args=(assigned_gpu,))
            for executor in stack._executors.values()
            if isinstance(executor, RayExecutor)
        ]
        assert len(set(service_gpus)) == 2
        cuda_managed = (
            get_accelerator_manager_for_resource("GPU").get_visible_accelerator_ids_env_var() == "CUDA_VISIBLE_DEVICES"
        )
        if cuda_managed:
            assert [json.loads(output.read_text()) for output in outputs] == service_gpus
        assert ray.available_resources()["GPU"] == 5

        # Slime reserves one bundle per model GPU (four train + one rollout).
        group = placement_group([{"GPU": 1, "CPU": 1} for _ in range(5)], strategy="PACK")
        ray.get(group.ready(), timeout=30)

        @ray.remote(num_gpus=1, num_cpus=1)
        class ModelRank:
            def gpu_id(self):
                return str(ray.get_gpu_ids()[0])

        for rank in range(5):
            actors.append(
                ModelRank.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=group, placement_group_bundle_index=rank
                    )
                ).remote()
            )
        model_gpus = ray.get([actor.gpu_id.remote() for actor in actors], timeout=30)
        assert len(set(model_gpus)) == 5
        assert set(model_gpus).isdisjoint(service_gpus)
        # Apple Ray has logical GPU reservations but no per-process CUDA mask.
        expected = gpu_pool if cuda_managed else [str(rank) for rank in range(7)]
        assert set(model_gpus + service_gpus) == set(expected)
    finally:
        for actor in actors:
            ray.kill(actor)
        if group is not None:
            remove_placement_group(group)
        if stack is not None:
            stack.shutdown(grace=1)
        assert not ray.is_initialized()
        if external:
            # The driver disconnected, but the external cluster survived.
            real_init(address=cluster.address)
            assert ray.cluster_resources()["GPU"] == 7
        ray.shutdown()
        cluster.shutdown()
