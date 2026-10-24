"""AdapterServingRuntime: the control plane loads each scenario's saved adapter into the serving engines."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reef.artifact.artifact import Artifact, ArtifactRef
from reef.core.errors import ReefError
from reef.runtime.interfaces import InferenceBackend, InferenceHandler, ModelCandidate, TrainingRuntimeError
from reef.runtime.serving import ADAPTER_DIR, AdapterServingRuntime
from reef.surface.adapter import adapter_name


class Engines(InferenceBackend):
    """Serving engines that record each control call."""

    def __init__(self) -> None:
        self.events: list = []
        self.versions = ["boot"]

    def load_weight_files(self, files, *, adapter_name, runtime_load_id):
        self.events.append(("load", files.layout, adapter_name, runtime_load_id, files.path))
        self.versions = [runtime_load_id]

    def initialize_version(self, runtime_load_id):
        self.events.append(("initialize", runtime_load_id))
        self.versions = [runtime_load_id]

    def inference_url(self):
        return "http://component.example"

    def runtime_load_ids(self):
        return list(self.versions)

    def pause(self):
        self.events.append("pause")

    def resume(self):
        self.events.append("resume")

    def recover(self):
        raise AssertionError("unused")

    def abort(self):
        raise AssertionError("unused")

    def unload_adapter(self, name):
        self.events.append(("unload", name))


class Handler(InferenceHandler):
    async def inference(self, artifact, path, payload):
        raise AssertionError("unused")


def saved_checkpoint(path: Path) -> Path:
    (path / ADAPTER_DIR).mkdir(parents=True)
    (path / ADAPTER_DIR / "adapter_config.json").write_text(json.dumps({"r": 16}))
    (path / "tinker-checkpoint.json").write_text("{}")
    return path


def candidate(job_id: str, checkpoint: Path, scenario: str = "math") -> ModelCandidate:
    return ModelCandidate(
        candidate_id=job_id,
        training_job_id=job_id,
        checkpoint_path=str(checkpoint),
        current_runtime_load_id=None,
        metadata={"scenario_step": 0, "scenario": scenario},
    )


def release(path: Path, content_id: str, runtime_load_id: str | None = None) -> Artifact:
    metadata = {} if runtime_load_id is None else {"runtime_load_id": runtime_load_id}
    return Artifact(ArtifactRef(content_id, f"release-{content_id}", None), None, local_path=path, metadata=metadata)


@pytest.fixture
def engines():
    return Engines()


@pytest.fixture
def serving(engines):
    return AdapterServingRuntime(engines, Handler(), adapter_capacity=2)


def test_the_engines_start_serving_the_base_model_under_a_fresh_version(serving, engines):
    [(_, version), resumed] = engines.events
    assert (engines.events[0][0], resumed) == ("initialize", "resume")
    assert serving.serving_runtime_load_id() == version
    assert serving.serving_adapter_runtime_load_id("math") is None
    assert serving.base_url == "http://component.example"


def test_a_selected_candidate_loads_its_adapter_and_its_release_loads_nothing(serving, engines, tmp_path):
    checkpoint = saved_checkpoint(tmp_path / "job-1")
    activated = serving.activate_candidate(candidate("job-1", checkpoint))
    version = activated.runtime_load_id
    assert engines.events[-3:] == [
        "pause",
        ("load", "adapter", adapter_name("math", version), version, checkpoint / ADAPTER_DIR),
        "resume",
    ]
    assert serving.serving_adapter_runtime_load_id("math") == version
    # A retry of the same activation loads nothing again.
    assert serving.activate_candidate(candidate("job-1", checkpoint)) == activated
    # The committer activates the staged release, then the published one: the engines already serve both.
    loads = len(engines.events)
    assert serving.activate_checkpoint(release(checkpoint, "c1", version), scenario="math") == version
    assert serving.activate_checkpoint(release(checkpoint, "c1"), scenario="math") == version
    assert len(engines.events) == loads
    with pytest.raises(TrainingRuntimeError, match="not the committed job 'job-2'"):
        serving.acknowledge_publication("job-2")
    serving.acknowledge_publication("job-1")
    serving.acknowledge_publication("job-1")


def test_scenarios_serve_their_own_adapters_and_a_restart_loads_each_release_again(engines, tmp_path):
    serving = AdapterServingRuntime(engines, Handler(), adapter_capacity=2)
    math = serving.activate_checkpoint(release(saved_checkpoint(tmp_path / "math"), "m"), scenario="math")
    code = serving.activate_checkpoint(release(saved_checkpoint(tmp_path / "code"), "c"), scenario="code")
    assert (serving.serving_adapter_runtime_load_id("math"), serving.serving_adapter_runtime_load_id("code")) == (
        math,
        code,
    )
    assert serving.adapter_residency_status()["counters"]["loads"] == 2
    restarted = AdapterServingRuntime(Engines(), Handler(), adapter_capacity=2)
    again = restarted.activate_checkpoint(release(tmp_path / "math", "m", math), scenario="math")
    # A new incarnation never reuses a version for other weights.
    assert again.split(":")[0] != math.split(":")[0]


def test_a_rollback_loads_another_release_and_a_release_without_weights_serves_the_base(serving, tmp_path):
    first = serving.activate_candidate(candidate("job-1", saved_checkpoint(tmp_path / "job-1"))).runtime_load_id
    # Another release replaces the candidate that Reef did not commit.
    older = serving.restore_checkpoint(release(saved_checkpoint(tmp_path / "older"), "o"), scenario="math")
    assert older != first and serving.serving_adapter_runtime_load_id("math") == older
    serving.acknowledge_publication("job-2")
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "reef-artifact.json").write_text("{}")
    serving.restore_checkpoint(release(empty, "base"), scenario="math")
    assert serving.serving_adapter_runtime_load_id("math") is None


def test_only_a_saved_adapter_of_a_named_scenario_is_served(serving, tmp_path):
    with pytest.raises(ReefError, match="named scenario"):
        serving.activate_checkpoint(release(saved_checkpoint(tmp_path / "a"), "a"))
    with pytest.raises(TrainingRuntimeError, match="must name its scenario"):
        serving.activate_candidate(
            ModelCandidate(
                candidate_id="j",
                training_job_id="j",
                checkpoint_path=str(tmp_path / "a"),
                current_runtime_load_id=None,
            )
        )
    manifest_only = tmp_path / "manifest-only"
    manifest_only.mkdir()
    (manifest_only / "tinker-checkpoint.json").write_text("{}")
    with pytest.raises(ReefError, match="holds no PEFT adapter"):
        serving.activate_checkpoint(release(manifest_only, "m"), scenario="math")
    assert serving.serving_adapter_runtime_load_id("math") is None
