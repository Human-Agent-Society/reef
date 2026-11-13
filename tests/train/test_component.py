"""The training-job values: what a job, a saved checkpoint, a job status and a artifact format may hold."""

from __future__ import annotations

import json

import pytest
from tests.support.job_data import job_data_ref
from tests.support.runtime_stubs import StubFacts, StubTrainingComponent

from reef.artifact.base_root import BASE_FILE, BaseRoot
from reef.artifact.files import DirectoryFiles
from reef.core.errors import InternalError, RuntimeConfigError
from reef.core.weight_layout import ADAPTER_DIR, ArtifactFormat
from reef.train.component import (
    JobStatus,
    SavedCheckpoint,
    TrainingArtifactLayout,
    TrainingBackendFacts,
    TrainingComponent,
    TrainingComponentFactory,
    TrainingJob,
    job_id_for,
    training_backend_facts,
)
from reef.train.slime_backend.signals import PreparedJob
from reef.train.types import JobData


@pytest.mark.unit
@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"job_id": ""}, ValueError, "job_id must be a non-empty string"),
        ({"backend": ""}, ValueError, "backend must be a non-empty string"),
        ({"start": ""}, ValueError, "start must be a non-empty string"),
        ({"start": None}, ValueError, "start must be a non-empty string"),
        ({"data": ()}, TypeError, "data must be a JobDataRef"),
        ({"parameters": [("lr", 1.0)]}, TypeError, "parameters must be a mapping"),
        ({"parameters": {"lr": float("nan")}}, ValueError, "Out of range float"),
    ],
)
def test_a_training_job_checks_its_values_at_construction(changes, error, message):
    values = {
        "job_id": "job",
        "backend": "tinker",
        "start": "/base",
        "data": job_data_ref(JobData("b")),
        "parameters": {},
    }
    TrainingJob(**values, scenario="s")
    with pytest.raises(error, match=message):
        TrainingJob(**{**values, **changes}, scenario="s")


@pytest.mark.unit
def test_a_job_id_names_the_backend_start_data_and_parameters():
    job_id = job_id_for("tinker", "/base", ["row"], {"lr": 1.0})
    assert job_id == job_id_for("tinker", "/base", ["row"], {"lr": 1.0})
    assert job_id != job_id_for("slime", "/base", ["row"], {"lr": 1.0})
    assert job_id != job_id_for("tinker", "/weights", ["row"], {"lr": 1.0})
    assert job_id != job_id_for("tinker", "/base", ["other"], {"lr": 1.0})
    assert job_id != job_id_for("tinker", "/base", ["row"], {"lr": 2.0})


@pytest.mark.unit
@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"state": "paused"}, "running, finished or failed"),
        ({"state": "failed"}, "error must be set exactly when the job failed"),
        ({"state": "running", "error": "boom"}, "error must be set exactly when the job failed"),
        (
            {"state": "running", "checkpoints": (SavedCheckpoint(2, "/b", "/b"), SavedCheckpoint(1, "/a", "/a"))},
            "increasing step order",
        ),
        ({"state": "failed", "error": ""}, "error must be a non-empty string"),
        ({"state": "finished", "metrics": {"loss": float("nan")}}, "Out of range float"),
    ],
)
def test_a_job_status_is_consistent(values, message):
    with pytest.raises(ValueError, match=message):
        JobStatus("job", **values)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("step", "path", "release"), [(-1, "/a", "/a"), (True, "/a", "/a"), (0, "", "/a"), (0, "/a", "")]
)
def test_a_saved_checkpoint_has_a_step_a_path_and_a_release(step, path, release):
    with pytest.raises(ValueError):
        SavedCheckpoint(step, path, release)


@pytest.mark.unit
def test_a_job_status_names_its_job_and_checkpoint_metrics_encode_as_json():
    with pytest.raises(ValueError, match="job_id must be a non-empty string"):
        JobStatus("", "running")
    with pytest.raises(ValueError, match="Out of range float"):
        SavedCheckpoint(0, "/a", "/a", {"loss": float("inf")})


@pytest.mark.unit
@pytest.mark.parametrize(
    ("values", "error", "message"),
    [
        ({"action": "retry"}, ValueError, "must be 'train' or 'skip'"),
        ({"action": "skip", "payload": {"rows": []}}, ValueError, "only a train job"),
        ({"action": "train", "payload": None}, ValueError, "only a train job"),
        ({"metrics": None}, TypeError, "must be mappings"),
    ],
)
def test_a_prepared_training_step_is_a_train_step_with_a_payload_or_a_skip(values, error, message):
    defaults = {"action": "train", "next_algorithm_state": {}, "metrics": {}, "payload": {"rows": []}}
    PreparedJob(**defaults)
    PreparedJob("skip", {}, {})
    with pytest.raises(error, match=message):
        PreparedJob(**{**defaults, **values})


@pytest.mark.unit
@pytest.mark.parametrize(
    ("values", "message"),
    [
        (("shards",), "layout must be 'full' or 'adapter'"),
        (("full", 8), "no adapter rank or target modules"),
        (("full", None, ("all",)), "no adapter rank or target modules"),
        (("adapter", None, ("all",)), "positive int adapter_rank"),
        (("adapter", 0, ("all",)), "positive int adapter_rank"),
        (("adapter", True, ("all",)), "positive int adapter_rank"),
        (("adapter", 8), "non-empty tuple of module names"),
        (("adapter", 8, ["q_proj"]), "non-empty tuple of module names"),
        (("adapter", 8, ("q_proj", "")), "non-empty tuple of module names"),
    ],
)
def test_a_release_format_sets_the_adapter_rank_and_modules_exactly_for_adapters(values, message):
    with pytest.raises(ValueError, match=message):
        ArtifactFormat(*values)


@pytest.mark.unit
@pytest.mark.parametrize("value", [ArtifactFormat("full"), ArtifactFormat("adapter", 16, ("q_proj", "v_proj"))])
def test_a_release_format_reads_back_what_it_writes(value):
    assert ArtifactFormat.from_dict(value.to_dict()) == value
    assert ArtifactFormat.from_dict({"layout": "full"}) == ArtifactFormat("full")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({"layout": "full", "dtype": "bf16"}, "an object of layout"),
        ({"layout": "adapter", "adapter_rank": 8, "adapter_target_modules": "all"}, "a list of module names"),
        ({"adapter_rank": 8}, "an object of layout"),
        ({"layout": "lora"}, "layout must be 'full' or 'adapter'"),
    ],
)
def test_a_release_format_refuses_another_shape(value, message):
    with pytest.raises(ValueError, match=message):
        ArtifactFormat.from_dict(value)


@pytest.mark.unit
def test_an_imported_release_must_have_the_layout_of_the_components_releases(tmp_path):
    full, adapter = tmp_path / "full", tmp_path / "adapter"
    full.mkdir()
    (adapter / ADAPTER_DIR).mkdir(parents=True)
    full_weights = StubTrainingComponent(artifact_format=ArtifactFormat("full"))
    with pytest.raises(ValueError, match=r"config\.json and weight files"):
        StubFacts(full_weights).check_artifact(DirectoryFiles(full))
    (full / "config.json").write_text("{}")
    (full / "model-00001-of-00002.safetensors").write_bytes(b"w")
    StubFacts(full_weights).check_artifact(DirectoryFiles(full))

    rank_8 = StubTrainingComponent(artifact_format=ArtifactFormat("adapter", 8, ("q_proj", "v_proj")))
    with pytest.raises(ValueError, match="holds a PEFT adapter in adapter/"):
        StubFacts(rank_8).check_artifact(DirectoryFiles(adapter))
    (adapter / ADAPTER_DIR / "adapter_model.safetensors").write_bytes(b"a")
    for config, message in (
        ({"r": 16, "target_modules": ["q_proj"]}, "at most 8"),
        ({"r": 4, "target_modules": ["q_proj", "o_proj"]}, "within"),
        ({"r": 4, "target_modules": ".*proj"}, "within"),
    ):
        (adapter / ADAPTER_DIR / "adapter_config.json").write_text(json.dumps(config))
        with pytest.raises(ValueError, match=message):
            StubFacts(rank_8).check_artifact(DirectoryFiles(adapter))
    (adapter / ADAPTER_DIR / "adapter_config.json").write_text(json.dumps({"r": 4, "target_modules": ["q_proj"]}))
    StubFacts(rank_8).check_artifact(DirectoryFiles(adapter))
    # ``("all",)`` takes any target modules.
    StubFacts(StubTrainingComponent(artifact_format=ArtifactFormat("adapter", 8, ("all",)))).check_artifact(
        DirectoryFiles(adapter)
    )
    with pytest.raises(ValueError, match="takes no imported snapshots"):
        StubFacts(StubTrainingComponent(artifact_format=None)).check_artifact(DirectoryFiles(full))


@pytest.mark.unit
def test_the_layout_asks_the_component_for_what_the_facts_leave_to_the_service(tmp_path):
    class NamingFacts(StubFacts):
        def check_artifact(self, files):
            if "manifest.json" not in files.paths():
                raise ValueError("an artifact holds manifest.json")
            return {"path": files.read_bytes("manifest.json").decode()}

    class Service(StubTrainingComponent):
        def check_remote_artifact(self, names):
            if names["path"] != "known":
                raise ValueError(f"the service holds no {names['path']}")

    component = Service()
    facts = NamingFacts(component)
    files = DirectoryFiles(tmp_path)
    # The facts refuse at once what the files show; nothing is asked then.
    with pytest.raises(ValueError, match=r"holds manifest\.json"):
        TrainingArtifactLayout(facts).check(files)
    (tmp_path / "manifest.json").write_text("known")
    TrainingArtifactLayout(facts, component).check(files)
    # A process with no way to the training service cannot finish the check, and says so with no refusal.
    with pytest.raises(InternalError, match="the control service checks an import"):
        TrainingArtifactLayout(facts).check(files)
    (tmp_path / "manifest.json").write_text("lost")
    with pytest.raises(ValueError, match="the service holds no lost"):
        TrainingArtifactLayout(facts, component).check(files)
    # A component whose facts name nothing is never asked.
    with pytest.raises(InternalError, match="checks no artifact on a service"):
        StubTrainingComponent().check_remote_artifact({})


@pytest.mark.unit
@pytest.mark.parametrize("artifact_format", [ArtifactFormat("full"), ArtifactFormat("adapter", 8, ("q_proj",))])
def test_a_base_root_is_admitted_exactly_for_the_default_start_that_it_names(tmp_path, artifact_format):
    component = StubTrainingComponent(artifact_format=artifact_format)
    assert StubFacts(component).default_start_inputs == {"model": "stub"}
    (tmp_path / BASE_FILE).write_text(BaseRoot({"model": "stub"}).to_text())
    StubFacts(component).check_artifact(DirectoryFiles(tmp_path))
    (tmp_path / BASE_FILE).write_text(BaseRoot({"model": "stub", "lora_rank": "8"}).to_text())
    with pytest.raises(ValueError, match=r"another default start than this deployment's \(lora_rank: '8', not None"):
        StubFacts(component).check_artifact(DirectoryFiles(tmp_path))
    (tmp_path / BASE_FILE).write_text(BaseRoot({"model": "other"}).to_text())
    with pytest.raises(ValueError, match="model: 'other', not 'stub'"):
        StubFacts(component).check_artifact(DirectoryFiles(tmp_path))
    (tmp_path / BASE_FILE).write_text("{}")
    with pytest.raises(ValueError, match=r"the object \{format, inputs\}"):
        StubFacts(component).check_artifact(DirectoryFiles(tmp_path))


#: The factory of the stand-in backend of ``tests.support.training_deployment``.
LOCAL_FACTORY = "tests.support.training_deployment:component_factory"


@pytest.mark.unit
def test_every_factory_states_its_facts_and_stating_them_builds_no_component(tmp_path, monkeypatch):
    """``TrainingComponentFactory.facts`` is required: the facts are the one source of a default start."""

    class NoFacts(TrainingComponentFactory):
        kind = "no-facts"

        def __call__(self, config, model_path, environ):
            raise AssertionError("not built")

    with pytest.raises(TypeError, match="abstract"):
        NoFacts()
    assert not hasattr(TrainingComponent, "default_start")
    assert not hasattr(TrainingComponent, "default_start_inputs")
    marker = tmp_path / "built"
    facts = training_backend_facts({"type": LOCAL_FACTORY, "marker": str(marker)}, model_path="m", environ={})
    assert isinstance(facts, TrainingBackendFacts) and facts.default_start().inputs == facts.default_start_inputs
    assert not marker.exists()
    with pytest.raises(RuntimeConfigError, match="dotted reference as type"):
        training_backend_facts({}, model_path="m", environ={})
    from tests.support.training_deployment import component_factory

    monkeypatch.setattr(type(component_factory), "facts", lambda *args: {"name": "local"})
    with pytest.raises(TypeError, match="must state TrainingBackendFacts"):
        training_backend_facts({"type": LOCAL_FACTORY}, model_path="m", environ={})
