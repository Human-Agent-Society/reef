"""CPU contract tests for Spindle's full-parameter training component, its serving, and its deployment (RFC #996)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.support.hugging_face_hub import COMMIT, MemoryHub

from reef.artifact import REFERENCE_FILE, Artifact, DirectoryFiles, HuggingFaceRevision, read_reference
from reef.core.evaluation import PolicyModel
from reef.core.job_data import JobData, TrajectorySample
from reef.inference.tinker import TinkerInferenceRuntime
from reef.runtime.interfaces import TrainingMethod
from reef.runtime.jobs import ReleaseFormat, TrainingJob, job_id_for
from reef.service.deploy.training import training_deployment_for
from reef.tinker.checkpoint import MANIFEST, FreshStart, TinkerCheckpoint
from reef.tinker.sampling import SampleResult, TinkerSampler
from reef.train.algos import TrainingSignal
from reef.train.algos.objective import TrainingObjective
from reef.train.algos.registry import register_objective, unregister_objective
from reef.train.tinker_backend import client as client_module
from reef.train.tinker_backend.client import TinkerClient, TinkerSDKClient
from reef.train.tinker_backend.config import SpindleFullConfig
from reef.train.tinker_backend.full import SpindleFullTrainingComponent
from reef.train.tinker_backend.launch import SpindleFullDeployment, spindle_full_settings
from reef.train.tinker_backend.losses import ImportanceSamplingLoss, TokenRow
from reef.train.tinker_backend.spindle_export import CheckpointExporter, checkpoint_directory, checkpoint_upload

MODEL = "org/model"
SPINDLE = "https://spindle.example"
BASE = f"hf://{MODEL}@{COMMIT}"
RELEASES = "org/model-releases"
FILES = {"config.json": b"{}", "model.safetensors": b"base-weights"}


def settings(tmp_path: Path, **options: str) -> SpindleFullConfig:
    values = {"state_dir": str(tmp_path / "state"), "base_url": SPINDLE, "base": BASE, "release_repo": RELEASES}
    return SpindleFullConfig(**{**values, **options})


class Exporter(CheckpointExporter):
    """Exports each checkpoint to the next commit of the release repo, and keeps what it exported."""

    def __init__(self) -> None:
        self.exported: list[tuple[str, HuggingFaceRevision]] = []

    def export(self, state_path, base):
        self.exported.append((state_path, base))
        return HuggingFaceRevision(RELEASES, f"{len(self.exported):040x}")


class SpindleClient(TinkerClient, TinkerSampler):
    """The SDK boundaries of a Spindle full-parameter engine as an offline double."""

    def __init__(self) -> None:
        self.starts: list[TinkerCheckpoint | FreshStart] = []
        self.sampled: list[str | None] = []
        self.checked: list[TinkerCheckpoint] = []

    def initialize(self) -> TinkerCheckpoint:
        raise AssertionError("a full-parameter component saves no base checkpoint")

    def train(self, start, steps, loss, learning_rates, *, kl_coef):
        self.starts.append(start)
        number = len(self.starts)
        return (
            TinkerCheckpoint(
                MODEL, None, f"tinker://full-{number}/state", f"tinker://full-{number}/sampler", service=SPINDLE
            ),
            {},
        )

    def check_remote(self, checkpoint):
        self.checked.append(checkpoint)

    def render(self, messages, *, template_kwargs):
        return [101]

    def decode(self, tokens):
        return "hello"

    def sample(self, sampler_path, prompt, params, *, topk_logprobs):
        self.sampled.append(sampler_path)
        return SampleResult((201,), (-0.2,), "stop")

    def close(self) -> None:
        return


class Objective(TrainingObjective):
    loss_family = "importance_sampling"
    name = "spindle-full-test-objective"

    def prepare(self, batch, state):
        return TrainingSignal("train", {}, advantages=tuple(1.0 for _ in batch.samples))


@pytest.fixture
def objective():
    value = Objective()
    register_objective(value)
    yield value
    unregister_objective(value.name)


@pytest.fixture
def hub(tmp_path, monkeypatch):
    value = MemoryHub(tmp_path / "hub-cache", MODEL, dict(FILES))
    monkeypatch.setattr("reef.artifact.sources.HfApi", lambda: value)
    return value


@pytest.fixture
def client():
    return SpindleClient()


@pytest.fixture
def exporter():
    return Exporter()


@pytest.fixture
def component(tmp_path, client, exporter):
    value = SpindleFullTrainingComponent(MODEL, settings(tmp_path), client, exporter)
    value.start_jobs()
    value.start_jobs()
    yield value
    value.close()


def train(component, start=None):
    """Run one job of one trajectory from ``start`` (the base when ``None``) to its end; return its status."""
    data = JobData(
        "batch",
        (
            TrajectorySample(
                {
                    "schema_version": "ATIF-v1.6",
                    "agent": {"name": "test"},
                    "steps": [{"step_id": 1}],
                    "extra": {
                        "reef": {
                            "training": {
                                "tokens": [10, 11, 20, 21],
                                "loss_mask": [1, 0],
                                "rollout_log_probs": [-0.2, -0.4],
                                "artifact_id": "v",
                            }
                        }
                    },
                },
                group_id="g",
            ),
        ),
    )
    parameters = {"method": TrainingMethod(Objective.name).to_dict()}
    start = component.base_start if start is None else start
    job_id = job_id_for(component.name, start, data.to_dict()["samples"], parameters)
    component.submit(TrainingJob(job_id, component.name, start, data, parameters, scenario="s"))
    deadline = time.monotonic() + 10
    while (status := component.status(job_id)).state == "running":
        assert time.monotonic() < deadline, status
        time.sleep(0.01)
    assert status.state == "finished", status
    return status


@pytest.mark.unit
def test_the_base_is_a_reference_to_the_base_revision_and_a_job_from_it_starts_fresh(
    component, client, exporter, objective
):
    assert component.release_format == ReleaseFormat("full")
    base = component.base_artifact_source()
    assert read_reference(base.model_files.directory()) == HuggingFaceRevision(MODEL, COMMIT)
    assert base.inputs == {"model": BASE, "service": SPINDLE}
    assert component.start_at(Path(component.base_start)) == FreshStart(MODEL)

    status = train(component)
    assert client.starts == [FreshStart(MODEL)]
    release = Path(status.checkpoints[0].release_path)
    saved = TinkerCheckpoint.read(release)
    assert saved.lora_rank is None and saved.service == SPINDLE
    # The release refers to the checkpoint's weights, which the exporter put in the release repo.
    assert exporter.exported == [(saved.state_path, HuggingFaceRevision(MODEL, COMMIT))]
    assert read_reference(release) == HuggingFaceRevision(RELEASES, f"{1:040x}")

    # A job from a saved release restores its checkpoint, whatever its reference names.
    train(component, start=str(release))
    assert client.starts[-1] == saved


@pytest.mark.unit
def test_a_new_base_commit_replaces_the_base_reference_in_the_state_dir(tmp_path, client, exporter):
    # The deployment restarts with a later commit of the same repo in its config.
    first = SpindleFullTrainingComponent(MODEL, settings(tmp_path), client, exporter)
    first.prepared_base()
    first.close_resources()
    later = "f" * 40
    component = SpindleFullTrainingComponent(MODEL, settings(tmp_path, base=f"hf://{MODEL}@{later}"), client, exporter)
    component.start_jobs()
    files = component.base_artifact_source().model_files
    assert read_reference(files.directory()) == HuggingFaceRevision(MODEL, later)
    assert component.start_at(Path(component.base_start)) == FreshStart(MODEL)


@pytest.mark.unit
def test_an_import_is_a_full_parameter_manifest_or_the_bases_files(component, client, hub, tmp_path):
    full = TinkerCheckpoint(MODEL, None, "tinker://f/state", "tinker://f/sampler", service=SPINDLE)
    full.write(tmp_path / "full")
    with pytest.raises(ValueError, match=f"holds {REFERENCE_FILE} beside its manifest"):
        component.check_release(DirectoryFiles(tmp_path / "full"))
    # A release's reference must resolve to full weights.
    (tmp_path / "full" / REFERENCE_FILE).write_text(json.dumps(HuggingFaceRevision(MODEL, COMMIT).to_dict()))
    component.check_release(DirectoryFiles(tmp_path / "full"))
    assert client.checked[-1] == full
    del hub.files["config.json"]
    with pytest.raises(ValueError, match="holds no full weights"):
        component.check_release(DirectoryFiles(tmp_path / "full"))
    hub.files["config.json"] = b"{}"
    TinkerCheckpoint(MODEL, 32, "tinker://l/state", "tinker://l/sampler", service=SPINDLE).write(tmp_path / "lora")
    with pytest.raises(ValueError, match="rank 32 LoRA, not all the parameters"):
        component.check_release(DirectoryFiles(tmp_path / "lora"))

    copy = tmp_path / "copy"
    copy.mkdir()
    for path, data in FILES.items():
        (copy / path).write_bytes(data)
    component.check_release(DirectoryFiles(copy))
    (copy / "model.safetensors").write_bytes(b"other-weights")
    with pytest.raises(ValueError, match=f"trains only its base {BASE}"):
        component.check_release(DirectoryFiles(copy))


@pytest.mark.unit
def test_train_from_takes_a_manifest_or_the_base(component, tmp_path):
    TinkerCheckpoint(MODEL, None, "tinker://f/state", "tinker://f/sampler", service=SPINDLE).write(tmp_path / "full")
    assert Path(component.import_start(tmp_path / "full")).parent == component.state_dir / "imported"
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / REFERENCE_FILE).write_text(json.dumps(HuggingFaceRevision(MODEL, COMMIT).to_dict()))
    assert component.import_start(reference) == component.base_start
    (reference / REFERENCE_FILE).write_text(json.dumps(HuggingFaceRevision(MODEL, "f" * 40).to_dict()))
    with pytest.raises(ValueError, match="starts only from its base"):
        component.import_start(reference)


@pytest.mark.unit
def test_the_component_trains_the_repo_of_its_base_on_a_spindle_deployment(tmp_path, client):
    with pytest.raises(ValueError, match="not the repo of the base"):
        SpindleFullTrainingComponent("other/model", settings(tmp_path), client, Exporter())
    with pytest.raises(ValueError, match="requires base_url"):
        SpindleFullConfig(state_dir=str(tmp_path), base=BASE, release_repo=RELEASES)
    with pytest.raises(ValueError, match="40-digit commit SHA"):
        settings(tmp_path, base=f"hf://{MODEL}@main")


@pytest.mark.unit
def test_spindles_sampler_serves_the_base_as_the_base_model_and_a_release_by_its_sampler(client, tmp_path):
    runtime = TinkerInferenceRuntime(client, base_model=MODEL, service=SPINDLE, weight_kind="model")
    lora_runtime = TinkerInferenceRuntime(client, base_model=MODEL, service=SPINDLE)
    try:
        assert runtime.served_weight_kind == "model"
        reference = tmp_path / "reference"
        reference.mkdir()
        (reference / REFERENCE_FILE).write_text(json.dumps(HuggingFaceRevision(MODEL, COMMIT).to_dict()))
        assert runtime.artifact_model(Artifact.local(reference)) == PolicyModel(MODEL)
        full = TinkerCheckpoint(MODEL, None, "tinker://f/state", "tinker://f/sampler", service=SPINDLE)
        full.write(tmp_path / "full")
        assert runtime.artifact_model(Artifact.local(tmp_path / "full")) == PolicyModel("tinker://f/sampler")
        # A LoRA's runtime serves manifests only, and a full-parameter runtime only its base model's weights.
        with pytest.raises(ValueError, match=r"missing tinker-checkpoint\.json"):
            lora_runtime.artifact_model(Artifact.local(reference))
        (reference / REFERENCE_FILE).write_text(json.dumps(HuggingFaceRevision("other/model", COMMIT).to_dict()))
        with pytest.raises(ValueError, match="not to the base model"):
            runtime.artifact_model(Artifact.local(reference))
    finally:
        runtime.shutdown()
        lora_runtime.shutdown()


@pytest.mark.unit
def test_the_sdk_client_creates_a_full_parameter_model_and_restores_only_full_parameter_checkpoints(
    tmp_path, monkeypatch
):
    created = []
    trainer = SimpleNamespace(
        forward_backward=lambda data, loss_fn, loss_fn_config=None: SimpleNamespace(
            result=lambda timeout: SimpleNamespace(metrics={})
        ),
        optim_step=lambda params: SimpleNamespace(result=lambda timeout: None),
        save_state=lambda name, ttl_seconds: SimpleNamespace(
            result=lambda timeout: SimpleNamespace(path="tinker://s/state")
        ),
        save_weights_for_sampler=lambda name, ttl_seconds: SimpleNamespace(
            result=lambda timeout: SimpleNamespace(path="tinker://s/sampler")
        ),
    )
    monkeypatch.setattr(
        client_module, "create_full_training_client", lambda sdk, service, model: created.append(model) or trainer
    )
    unloaded = []
    monkeypatch.setattr(client_module, "unload_model", lambda sdk, model: unloaded.append(model))
    infos = []
    service = SimpleNamespace(
        create_rest_client=lambda: SimpleNamespace(
            get_weights_info_by_tinker_path=lambda path: SimpleNamespace(result=lambda timeout: infos[-1])
        ),
        create_training_client_from_state_with_optimizer=lambda path: trainer,
        close=lambda status: SimpleNamespace(result=lambda timeout: None),
    )
    sdk = SimpleNamespace(
        Datum=lambda **fields: fields,
        ModelInput=SimpleNamespace(from_ints=list),
        AdamParams=lambda **fields: fields,
        ServiceClient=lambda **fields: service,
        NotFoundError=LookupError,
        BadRequestError=ValueError,
    )
    monkeypatch.setattr(client_module, "tinker_sdk", lambda: sdk)
    config = settings(tmp_path)
    full = TinkerSDKClient(MODEL, config, "key", lora_rank=None)
    with pytest.raises(ValueError, match="saves no base checkpoint"):
        full.initialize()

    row = TokenRow((10, 11, 20, 21), (1, 1), (-0.25, -0.5), 2)
    checkpoint, _ = full.train(FreshStart(MODEL), [[row]], ImportanceSamplingLoss(), [1e-6], kl_coef=0.0)
    # The model holds its engine alone, so the job unloads it when it ends.
    assert created == [MODEL] and unloaded == [trainer]
    assert checkpoint == TinkerCheckpoint(MODEL, None, "tinker://s/state", "tinker://s/sampler", service=SPINDLE)
    infos.append(SimpleNamespace(base_model=MODEL, is_lora=False, lora_rank=None))
    full.train(checkpoint, [[row]], ImportanceSamplingLoss(), [1e-6], kl_coef=0.0)
    infos.append(SimpleNamespace(base_model=MODEL, is_lora=True, lora_rank=32))
    with pytest.raises(ValueError, match="is no full-parameter checkpoint"):
        full.train(checkpoint, [[row]], ImportanceSamplingLoss(), [1e-6], kl_coef=0.0)


@pytest.mark.unit
def test_the_deployment_trains_on_spindle_and_serves_through_its_sampler() -> None:
    assert isinstance(training_deployment_for("spindle_full"), SpindleFullDeployment)
    options = {
        "state-dir": "/tmp/state",
        "base-url": SPINDLE,
        "base": BASE,
        "release-repo": RELEASES,
        "learning-rate": "1e-6",
    }
    settings = spindle_full_settings(options)
    assert (settings.base_revision, settings.learning_rate) == (HuggingFaceRevision(MODEL, COMMIT), 1e-6)
    serving = SpindleFullDeployment().trainer_serving_config({"training_backend_options": options})
    assert (serving["type"], serving["base_url"], serving["parameterization"]) == ("tinker", SPINDLE, "full")
    assert MANIFEST == "tinker-checkpoint.json"


@pytest.mark.unit
def test_the_export_commits_exactly_the_checkpoints_weights_and_the_bases_tokenizer(tmp_path, monkeypatch):
    assert checkpoint_directory("tinker://model-7/weights/step-2") == "step-2/model-7"
    with pytest.raises(ValueError, match="no Spindle checkpoint path"):
        checkpoint_directory("tinker://model-7/sampler_weights/step-2")
    checkpoint = tmp_path / "volume" / "step-2" / "model-7"
    checkpoint.mkdir(parents=True)
    for name, data in {
        "model.safetensors": b"trained",
        "config.json": b"{}",
        "checkpoint_rank0.pt": b"state",
        "metadata.json": b"{}",
    }.items():
        (checkpoint / name).write_bytes(data)
    base = tmp_path / "base"
    base.mkdir()
    (base / "tokenizer.json").write_text("{}")
    (base / "config.json").write_text('{"base": true}')
    (base / "generation_config.json").write_text('{"eos_token_id": [1, 8]}')
    uploads = []

    class Hub:
        def create_repo(self, repo, *, private, exist_ok):
            assert private and exist_ok

        def upload_folder(self, *, repo_id, folder_path, commit_message, delete_patterns):
            uploads.append((repo_id, sorted(path.name for path in Path(folder_path).iterdir()), delete_patterns))
            assert (
                Path(folder_path) / "config.json"
            ).read_text() == "{}", "the checkpoint's config replaces the base's"
            assert (Path(folder_path) / "generation_config.json").read_text() == '{"eos_token_id": [1, 8]}'
            return SimpleNamespace(oid="c" * 40)

    monkeypatch.setattr("huggingface_hub.HfApi", Hub)
    patterns = []

    def snapshot_download(repo, revision, allow_patterns):
        patterns.extend(allow_patterns)
        return str(base)

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)
    upload = checkpoint_upload()
    commit = upload(
        str(tmp_path / "volume"), "step-2/model-7", RELEASES, MODEL, COMMIT, "tinker://model-7/weights/step-2"
    )
    assert commit == "c" * 40
    # The training state stays behind, the base's generation config comes along since Spindle saves none, and the
    # commit replaces every file of the previous one.
    assert "generation_config.json" in patterns
    assert uploads == [
        (RELEASES, ["config.json", "generation_config.json", "model.safetensors", "tokenizer.json"], ["*"])
    ]
    with pytest.raises(FileNotFoundError, match="holds no Hugging Face weights"):
        upload(str(tmp_path / "volume"), "step-3/model-7", RELEASES, MODEL, COMMIT, "tinker://model-7/weights/step-3")
