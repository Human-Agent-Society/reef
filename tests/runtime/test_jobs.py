"""The training-job values: what a job, a saved checkpoint, a job status and a release format may hold."""

from __future__ import annotations

import json

import pytest
from tests.support.runtime_stubs import StubTrainingComponent

from reef.core.weight_layout import ADAPTER_DIR
from reef.runtime.jobs import JobStatus, ReleaseFormat, SavedCheckpoint, TrainingJob, job_id_for
from reef.train.algos.signals import PreparedJob
from reef.train.types import JobData


@pytest.mark.unit
@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"job_id": ""}, ValueError, "job_id must be a non-empty string"),
        ({"backend": ""}, ValueError, "backend must be a non-empty string"),
        ({"start": ""}, ValueError, "start must be a non-empty string"),
        ({"start": None}, ValueError, "start must be a non-empty string"),
        ({"data": ()}, TypeError, "data must be a JobData"),
        ({"parameters": [("lr", 1.0)]}, TypeError, "parameters must be a mapping"),
        ({"parameters": {"lr": float("nan")}}, ValueError, "Out of range float"),
    ],
)
def test_a_training_job_checks_its_values_at_construction(changes, error, message):
    values = {"job_id": "job", "backend": "tinker", "start": "/base", "data": JobData("b"), "parameters": {}}
    TrainingJob(**values)
    with pytest.raises(error, match=message):
        TrainingJob(**{**values, **changes})


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
        ReleaseFormat(*values)


@pytest.mark.unit
@pytest.mark.parametrize("value", [ReleaseFormat("full"), ReleaseFormat("adapter", 16, ("q_proj", "v_proj"))])
def test_a_release_format_reads_back_what_it_writes(value):
    assert ReleaseFormat.from_dict(value.to_dict()) == value
    assert ReleaseFormat.from_dict({"layout": "full"}) == ReleaseFormat("full")


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
        ReleaseFormat.from_dict(value)


@pytest.mark.unit
def test_an_imported_release_must_have_the_layout_of_the_components_releases(tmp_path):
    full, adapter = tmp_path / "full", tmp_path / "adapter"
    full.mkdir()
    (adapter / ADAPTER_DIR).mkdir(parents=True)
    full_weights = StubTrainingComponent(release_format=ReleaseFormat("full"))
    with pytest.raises(ValueError, match=r"config\.json and weight files"):
        full_weights.check_release(full)
    (full / "config.json").write_text("{}")
    (full / "model-00001-of-00002.safetensors").write_bytes(b"w")
    full_weights.check_release(full)

    rank_8 = StubTrainingComponent(release_format=ReleaseFormat("adapter", 8, ("q_proj", "v_proj")))
    with pytest.raises(ValueError, match="holds a PEFT adapter in adapter/"):
        rank_8.check_release(adapter)
    (adapter / ADAPTER_DIR / "adapter_model.safetensors").write_bytes(b"a")
    for config, message in (
        ({"r": 16, "target_modules": ["q_proj"]}, "at most 8"),
        ({"r": 4, "target_modules": ["q_proj", "o_proj"]}, "within"),
        ({"r": 4, "target_modules": ".*proj"}, "within"),
    ):
        (adapter / ADAPTER_DIR / "adapter_config.json").write_text(json.dumps(config))
        with pytest.raises(ValueError, match=message):
            rank_8.check_release(adapter)
    (adapter / ADAPTER_DIR / "adapter_config.json").write_text(json.dumps({"r": 4, "target_modules": ["q_proj"]}))
    rank_8.check_release(adapter)
    # ``("all",)`` takes any target modules.
    StubTrainingComponent(release_format=ReleaseFormat("adapter", 8, ("all",))).check_release(adapter)
    with pytest.raises(ValueError, match="takes no imported releases"):
        StubTrainingComponent(release_format=None).check_release(full)
