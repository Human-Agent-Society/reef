"""Dataset jobs: Slime rolls a prompt dataset out, rewards and trains it with its own loop, on the serving session."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("ray", reason="requires the optional Ray runtime")

from tests.slime_backend.test_slime_component import Stack, batch_for
from tests.support.slime_component import finished

from reef.core.job_data import JobData
from reef.core.training_request import TrainingRequest
from reef.runtime.durable_files import read_json, write_json
from reef.runtime.interfaces import TrainingMethod
from reef.runtime.jobs import TrainingJob
from reef.train.slime_backend.dataset import DATASET_FILE, DatasetSettings
from reef.train.slime_backend.native import DATASET_SOURCE_PATH, SlimeNativeAlgorithm
from reef.train.slime_backend.rollout_group import SlimeRolloutGroup

METHOD = TrainingMethod("reef.recipe.dataset:SlimeDatasetObjective")
FAMILY = "reef.train.slime_backend.native:SlimeNativeAlgorithm"


class DatasetRolloutGroup(SlimeRolloutGroup):
    """The serving session's engines: each rollout records when it synced, generated and released."""

    def __init__(self, events: list[str], *, rollouts_per_epoch: int, root: Path) -> None:
        super().__init__(SimpleNamespace(manager=None), colocated=True, router_url="http://router:3000")
        self.events = events
        self.per_epoch = rollouts_per_epoch
        self.root = root
        # The data source's record when each rollout generated.
        self.records: list[dict[str, Any]] = []

    def sync(self, actor_group) -> None:
        self.events.append("sync")

    def release(self) -> None:
        self.events.append("release")

    def rollouts_per_epoch(self) -> int:
        return self.per_epoch

    def generate(self, rollout_id: int) -> str:
        self.events.append(f"generate:{rollout_id}")
        self.records.append(read_json(self.root / DATASET_FILE))
        return f"rollout-data-{rollout_id}"

    def close(self) -> None:
        self.events.append("close")


def dataset_stack(tmp_path: Path, *, rollouts_per_epoch: int = 4) -> tuple[Stack, list[str], DatasetRolloutGroup]:
    events: list[str] = []
    group = DatasetRolloutGroup(events, rollouts_per_epoch=rollouts_per_epoch, root=tmp_path / "save")
    stack = Stack(
        tmp_path,
        loss_family=FAMILY,
        rollout_group=group,
        rank_metrics=[{"rollout/raw_reward": 0.25, "train/pg_loss": -0.1}],
        loss_family_override="slime",
    )
    group.root = stack.root
    return stack, events, group


def dataset_job(stack: Stack, job_id: str, *, epochs: int = 1, save_interval: int | None = 2) -> TrainingJob:
    parameters: dict[str, Any] = {
        "method": METHOD.to_dict(),
        "dataset": {"path": "/data/prompts.jsonl", "epochs": epochs},
    }
    if save_interval is not None:
        parameters["save_interval"] = save_interval
    return TrainingJob(job_id, "slime", stack.start, JobData(f"data-{job_id}", ()), parameters)


@pytest.fixture
def dataset_args(monkeypatch):
    """Stack's workers, started with the slime family instead of sft."""
    from tests.slime_backend import test_slime_component

    original = test_slime_component.worker_args

    def worker_args(tmp_path, loss_family_override=None, **overrides):
        args = original(tmp_path, **overrides)
        if loss_family_override is not None:
            args.loss_family = loss_family_override
            args.loss_type = "policy_loss"
            args.compute_advantages_and_returns = True
        return args

    monkeypatch.setattr(test_slime_component, "worker_args", worker_args)


@pytest.mark.unit
def test_dataset_settings_read_the_job_parameter() -> None:
    assert DatasetSettings.from_parameters({"dataset": {"path": "/d.jsonl"}}) == DatasetSettings(Path("/d.jsonl"), 1)
    assert DatasetSettings.from_parameters({"dataset": {"path": "/d.jsonl", "epochs": 3}}).epochs == 3
    for value, message in (
        (None, "an object with path"),
        ({"path": "/d.jsonl", "shuffle": True}, "an object with path"),
        ({"path": ""}, "non-empty string"),
        ({"path": "d.jsonl"}, "absolute path"),
        ({"path": "/d.jsonl", "epochs": 0}, "positive int"),
        ({"path": "/d.jsonl", "epochs": True}, "positive int"),
    ):
        with pytest.raises(ValueError, match=message):
            DatasetSettings.from_parameters({"dataset": value})


@pytest.mark.unit
def test_the_slime_family_serves_the_dataset_and_leaves_the_advantages_to_slime() -> None:
    family = SlimeNativeAlgorithm()
    args = Namespace(data_source_path="slime.data.data_source.RolloutDataSourceWithBuffer")
    family.configure_backend_args(args)
    assert args.data_source_path == DATASET_SOURCE_PATH
    with pytest.raises(ValueError, match="remove --data-source-path"):
        family.configure_backend_args(Namespace(data_source_path="my.source.Source"))
    family.validate_specific_args(Namespace(compute_advantages_and_returns=True), "test")
    with pytest.raises(RuntimeError, match="needs Slime's advantages"):
        family.validate_specific_args(Namespace(compute_advantages_and_returns=False), "test")


@pytest.mark.unit
def test_a_dataset_job_trains_slimes_own_rollouts_and_saves_every_interval(tmp_path, dataset_args) -> None:
    stack, events, group = dataset_stack(tmp_path)
    try:
        stack.component.submit(dataset_job(stack, "job-0"))
        status = finished(stack.component, "job-0")
        assert status.state == "finished", status.error
        # Four rollouts of Slime's own sequence: sync, generate, release, train.
        assert events == [event for rollout in range(4) for event in ("sync", f"generate:{rollout}", "release")]
        assert stack.group.train_calls == [(rollout, f"rollout-data-{rollout}") for rollout in range(4)]
        # A checkpoint every two rollouts; its step counts the job's rollouts, with the rollout's reward.
        assert [(checkpoint.step, Path(checkpoint.path).name) for checkpoint in status.checkpoints] == [
            (2, "rollout_1"),
            (4, "rollout_3"),
        ]
        assert status.checkpoints[-1].metrics["rollout/raw_reward"] == 0.25
        assert [rollout for rollout, _ in stack.group.saved] == [1, 3]
        assert status.metrics["dataset_rollouts"] == 4
        # The data source served the job's dataset, from its first prompt.
        assert {(record["job_id"], record["path"], record["skip_rollouts"]) for record in group.records} == {
            ("job-0", "/data/prompts.jsonl", 0)
        }
    finally:
        stack.component.close()


@pytest.mark.unit
def test_a_resumed_dataset_job_skips_the_rollouts_it_saved(tmp_path, dataset_args, monkeypatch) -> None:
    stack, _, group = dataset_stack(tmp_path, rollouts_per_epoch=2)
    job = dataset_job(stack, "job-0", epochs=2)
    directory = stack.component.job_directory("job-0")
    directory.mkdir(parents=True)
    try:
        # The first driver stops after its first checkpoint, two of the job's four rollouts.
        monkeypatch.setattr(stack.component, "stopping", lambda directory: len(stack.group.saved) == 1)
        stack.component.run(job, directory)
        assert [checkpoint.step for checkpoint in stack.component.saved(directory)] == [2]
        monkeypatch.setattr(stack.component, "stopping", lambda directory: False)
        group.records.clear()
        stack.group.train_calls.clear()
        metrics = stack.component.run(job, directory)
        # The data source skipped the two saved rollouts' prompts; the job trained the other two.
        assert {record["skip_rollouts"] for record in group.records} == {2}
        assert [rollout for rollout, _ in stack.group.train_calls] == [2, 3]
        assert [checkpoint.step for checkpoint in stack.component.saved(directory)] == [2, 4]
        assert metrics["rollout_id"] == 3
    finally:
        stack.component.close()


@pytest.mark.unit
def test_dataset_jobs_and_the_slime_family_go_together(tmp_path, dataset_args) -> None:
    stack, _, _ = dataset_stack(tmp_path)
    try:
        with pytest.raises(ValueError, match="trains only dataset jobs"):
            stack.component.submit(
                TrainingJob("job-0", "slime", stack.start, batch_for(0), {"method": METHOD.to_dict()})
            )
        job = dataset_job(stack, "job-1")
        with pytest.raises(ValueError, match="has no samples"):
            stack.component.submit(TrainingJob("job-1", "slime", stack.start, batch_for(0), job.parameters))
    finally:
        stack.component.close()
    stack = Stack(tmp_path / "sft")
    try:
        with pytest.raises(ValueError, match="needs the 'slime' loss family"):
            stack.component.submit(dataset_job(stack, "job-0"))
    finally:
        stack.component.close()


@pytest.mark.unit
def test_the_dataset_stage_makes_an_instructions_job_with_its_parameters() -> None:
    from reef.recipe.dataset import DatasetJobData

    stage = DatasetJobData.__new__(DatasetJobData)
    stage.context = SimpleNamespace(job_parameters={"trajectory": {"method": METHOD.to_dict(), "save_interval": 4}})
    request = TrainingRequest("train", "s", "r", id="r1", parameters={"dataset": {"path": "/d"}})
    data = JobData("scenario:instruction:r1", (), request=request)
    assert stage.make_job_data((), "scenario:instruction:r1").samples == ()
    assert stage.job_parameters(data) == {"method": METHOD.to_dict(), "save_interval": 4, "dataset": {"path": "/d"}}


@pytest.mark.unit
def test_the_data_source_loads_the_recorded_job_and_skips_its_trained_rollouts(tmp_path, monkeypatch) -> None:
    pytest.importorskip("slime")
    from reef.train.slime_backend import dataset_source

    monkeypatch.setattr(dataset_source, "load_tokenizer", lambda *args, **kwargs: None)
    monkeypatch.setattr(dataset_source, "load_processor", lambda *args, **kwargs: None)
    root = tmp_path / "save"
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text("".join(json.dumps({"input": f"q{index}", "label": str(index)}) + "\n" for index in range(6)))
    args = Namespace(
        save_hf=str(root / "hf" / "{rollout_id}"),
        prompt_data=None,
        hf_checkpoint="/models/m",
        rollout_max_prompt_len=None,
        input_key="input",
        multimodal_keys=None,
        label_key="label",
        metadata_key="metadata",
        tool_key=None,
        apply_chat_template=False,
        apply_chat_template_kwargs=None,
        rollout_seed=0,
        rollout_shuffle=False,
        rollout_batch_size=2,
        n_samples_per_prompt=2,
        buffer_filter_path=None,
        dump_details=None,
    )
    write_json(root / DATASET_FILE, {"job_id": "job-0", "path": str(prompts), "skip_rollouts": 2, "token": "a"})
    source = dataset_source.ReefDatasetSource(args)
    assert len(source) == 6
    # Two rollouts of two prompts were trained: the next prompts are q4 and q5, each in a group of two.
    groups = source.get_samples(2)
    assert [[sample.label for sample in group] for group in groups] == [["4", "4"], ["5", "5"]]
    # The same driver keeps its position; a new driver's record starts the job again.
    assert [group[0].label for group in source.get_samples(1)] == ["0"]
    write_json(root / DATASET_FILE, {"job_id": "job-1", "path": str(prompts), "skip_rollouts": 0, "token": "b"})
    assert [group[0].label for group in source.get_samples(1)] == ["0"]
