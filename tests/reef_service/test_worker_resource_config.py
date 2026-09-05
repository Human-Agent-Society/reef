"""Worker sizing is generic; only component capability checks know the role."""

import pytest

from reef.recipe.errors import RecipeConfigError
from reef.runtime.adapters.inference_proxy import InferenceProxyRuntime
from reef.runtime.executor import WorkerSpec
from reef.runtime.executor.config import ExecutorSettings, WorkerResources, executor_settings, select_executor
from reef.runtime.executor.requirements import ExecutionRequirements
from reef.service.deploy.execution import service_executor_config
from reef.train.cordis_backend.execution import evaluation_executor_config, evaluation_selection
from reef.train.cordis_backend.recipe import CordisRecipe
from reef.train.cordis_backend.strategies import resolve_episode_scorer


def scorer(task, result):
    return 1.0


def recipe_config(execution, **legacy):
    return {
        "execution": {"evolution": execution},
        "evolution": {"propose": lambda *args: None, "evaluate": scorer, "tasks": ["one"], **legacy},
    }


@pytest.mark.parametrize("workers, backend", [(1, "uni"), (8, "mp"), ("8", "mp")])
def test_generic_sizing_drives_real_recipe(workers, backend):
    config = recipe_config({"workers": workers, "resources": {"cpus_per_worker": "2", "gpus_per_worker": "0"}})
    recipe = CordisRecipe.from_environment(
        {}, config=config, runtime=InferenceProxyRuntime(model_path="test", base_url="http://unused", api_key="dummy")
    )
    selected, needs = evaluation_selection(recipe.score_episode, None, recipe.worker_executor)
    assert selected.settings.backend == backend
    assert needs.workers == int(workers)
    assert needs.cpus_per_worker == 2
    assert recipe.worker_executor.workers == int(workers)
    assert "episode_workers" not in recipe._backend_kwargs()


@pytest.mark.parametrize("workers", [0, -1, True, 1.5, "many", None])
def test_invalid_generic_worker_count(workers):
    with pytest.raises(ValueError, match="workers"):
        executor_settings({}, {"workers": workers})


@pytest.mark.parametrize("field", ["cpus_per_worker", "gpus_per_worker"])
@pytest.mark.parametrize("value", [-1, True, float("nan"), float("inf"), "nan", None, "many"])
def test_invalid_generic_resources(field, value):
    with pytest.raises(ValueError, match=field):
        executor_settings({}, {"resources": {field: value}})


@pytest.mark.parametrize("resources", [None, [], {"num_gpus": 1}, {"memory": 8}])
def test_unknown_resource_schema_rejected(resources):
    with pytest.raises(ValueError, match="resources"):
        executor_settings({}, {"resources": resources})


def test_gpu_resources_and_cpu_reservations_reach_ray_without_initializing_it():
    settings = executor_settings(
        {}, {"backend": "ray", "workers": 3, "resources": {"cpus_per_worker": 2, "gpus_per_worker": 0.5}}
    )
    selection, needs = evaluation_selection(resolve_episode_scorer(scorer), None, settings)
    config = evaluation_executor_config(selection, needs, WorkerSpec(dict), needs.workers)
    assert selection.settings.backend == "ray"
    assert len(config.workers) == 3
    assert config.options == {"num_cpus": 2, "num_gpus": 0.5}


def test_omitted_gpu_request_preserves_component_requirement(monkeypatch):
    monkeypatch.setattr("reef.runtime.executor.config.visible_cuda_devices", lambda: ("0", "1"))
    selected = select_executor(
        executor_settings({}, {"workers": 2}), role="evolution", requirements=ExecutionRequirements(gpus_per_worker=1)
    )
    assert selected.settings.backend == "mp"
    with pytest.raises(ValueError, match="below"):
        select_executor(
            executor_settings({}, {"resources": {"gpus_per_worker": 0}}),
            role="evolution",
            requirements=ExecutionRequirements(gpus_per_worker=1),
        )


@pytest.mark.parametrize("resource, option", [("cpus_per_worker", "num_cpus"), ("gpus_per_worker", "num_gpus")])
def test_conflicting_raw_actor_options_are_rejected(resource, option):
    settings = executor_settings({}, {"resources": {resource: 1}, "options": {option: 2}})
    with pytest.raises(ValueError, match="resources"):
        evaluation_selection(resolve_episode_scorer(scorer), None, settings)


def test_legacy_aliases_warn_and_normalize_into_generic_settings(monkeypatch):
    monkeypatch.setattr("reef.runtime.executor.config.visible_cuda_devices", lambda: ("0", "1", "2", "3"))
    config = recipe_config("auto", episode_workers="4", worker_resources={"num_gpus": 1})
    with pytest.warns(DeprecationWarning, match="deprecated"):
        recipe = CordisRecipe.from_environment({}, config=config)
    assert recipe.worker_executor == ExecutorSettings(workers=4, resources=WorkerResources(gpus_per_worker=1))


class VisibleDeviceWorker:
    def __init__(self):
        import os

        self.devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        self.pid = os.getpid()

    def placement(self):
        return self.pid, self.devices


def test_mp_assigns_disjoint_gpu_masks_before_worker_construction(monkeypatch):
    import os

    from reef.runtime.executor import Executor

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-first,GPU-second")
    monkeypatch.setattr("reef.runtime.executor.config.visible_cuda_devices", lambda: ("GPU-first", "GPU-second"))
    settings = executor_settings({}, {"workers": 2, "resources": {"gpus_per_worker": 1}})
    selection, needs = evaluation_selection(resolve_episode_scorer(scorer), None, settings)
    config = evaluation_executor_config(selection, needs, WorkerSpec(VisibleDeviceWorker), needs.workers)
    executor = Executor.create(config)
    try:
        placements = executor.collective_rpc("placement")
        assert [devices for _, devices in placements] == ["GPU-first", "GPU-second"]
        assert len({pid for pid, _ in placements}) == 2
        assert all(pid != os.getpid() for pid, _ in placements)
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-first,GPU-second"
    finally:
        executor.shutdown()


@pytest.mark.parametrize(
    "execution, legacy",
    [
        ({"workers": 4}, {"episode_workers": 2}),
        ({"resources": {"gpus_per_worker": 1}}, {"worker_resources": {"num_gpus": 2}}),
        ({"workers": 4}, {"worker_executor": "mp"}),
    ],
)
def test_mixed_configs_never_silently_drop_resource_requests(execution, legacy):
    with pytest.raises(RecipeConfigError, match=r"conflicts|remove deprecated"):
        CordisRecipe.from_environment({}, config=recipe_config(execution, **legacy))


@pytest.mark.parametrize("role", ["training", "rollout"])
@pytest.mark.parametrize("selection", [{"workers": 1}, {"resources": {"gpus_per_worker": 1}}])
def test_slime_does_not_silently_ignore_generic_topology(role, selection):
    with pytest.raises(ValueError, match="model topology"):
        select_executor(executor_settings({}, selection), role=role)


def test_services_map_resource_requests_but_do_not_create_replicas(tmp_path):
    execution = {"services": {"workers": 1, "resources": {"cpus_per_worker": 2, "gpus_per_worker": 1}}}
    config = service_executor_config({"execution": execution}, {}, tmp_path, 30, tmp_path / "config.yaml")
    assert config.workers[0].options == {"num_cpus": 2, "num_gpus": 1}
    execution["services"]["workers"] = 2
    with pytest.raises(ValueError, match="replication"):
        service_executor_config({"execution": execution}, {}, tmp_path, 30, tmp_path / "config.yaml")
