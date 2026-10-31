"""Score centering on the Spindle full-parameter component: the option, the rollouts' top-k capture and the loss."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

import pytest
from tests.tinker_backend.test_spindle_full import BASE, MODEL, RELEASES, SPINDLE, Exporter, SpindleClient

from reef.core.errors import DeployConfigError
from reef.core.evaluation import RolloutScore, RolloutScores
from reef.core.job_data import JobData, TaskSample, TrajectorySample
from reef.runtime.interfaces import TrainingMethod
from reef.runtime.jobs import TrainingJob
from reef.tinker.sampling import SampleResult, TinkerSampler
from reef.train.algos import TrainingSignal
from reef.train.algos.objective import TrainingObjective
from reef.train.algos.registry import register_objective, unregister_objective
from reef.train.algos.schedule import StepScheduling
from reef.train.task_rollouts import ChatEndpoint, TaskJobRollouts
from reef.train.tinker_backend import component as component_module
from reef.train.tinker_backend.config import ScoreCentering, SpindleFullConfig
from reef.train.tinker_backend.full import SpindleFullTrainingComponent
from reef.train.tinker_backend.launch import spindle_full_settings
from reef.train.tinker_backend.losses import (
    UNTRAINED_LOGPROB,
    CrossEntropyLoss,
    MatrixInput,
    ScoreCenteredLoss,
    TokenRow,
)
from reef.train.tinker_backend.preparation import prepare_tinker_job
from reef.train.tinker_backend.rollout import SamplerChat

CENTERING = ScoreCentering(top_k=2, min_tail_mass=1e-6, weight_low=1.0, weight_high=1.0)
HEADS = {"topk_indices": [[7, 3, 4], [8, 1, 2]], "topk_log_probs": [[-0.1, -2.5, -3.0], [-0.2, -2.0, -4.0]]}


def trajectory(group: str = "g", reward: float = 1.0, **training: object) -> TrajectorySample:
    values = {"tokens": [1, 2, 3, 7, 8], "loss_mask": [1, 1], "rollout_log_probs": [-0.1, -0.2], **training}
    return TrajectorySample(
        {
            "schema_version": "ATIF-v1.6",
            "agent": {"name": "a"},
            "steps": [{"step_id": 1}],
            "extra": {"reef": {"training": values, "reward": reward}},
        },
        group_id=group,
    )


class Objective(TrainingObjective):
    loss_family = "importance_sampling"
    name = "score-centering-test-objective"

    def prepare(self, batch, state):
        return TrainingSignal("train", {}, advantages=tuple(1.0 for _ in batch.samples))


class SftObjective(TrainingObjective):
    loss_family = "sft"
    name = "score-centering-test-sft"

    def prepare(self, batch, state):
        return TrainingSignal("train", {})


class TttdObjective(TrainingObjective):
    loss_family = "tttd"
    name = "score-centering-test-tttd"

    def prepare(self, batch, state):
        return TrainingSignal("train", {}, advantages=tuple(1.0 for _ in batch.samples))


@pytest.fixture
def objectives():
    import recipes.tttd  # noqa: F401 - registers the tttd loss family

    values = (Objective(), SftObjective(), TttdObjective())
    for value in values:
        register_objective(value)
    yield values
    for value in values:
        unregister_objective(value.name)


def prepare(objective: TrainingObjective, *items: TrajectorySample):
    return prepare_tinker_job(
        JobData("d", items),
        TrainingMethod(objective.name),
        {},
        StepScheduling(),
        batch_size=len(items),
        kl_coef=0.0,
        score_centering=CENTERING,
    )


def spindle(tmp_path, **options: object) -> SpindleFullConfig:
    values = {"state_dir": str(tmp_path / "state"), "base_url": SPINDLE, "base": BASE, "release_repo": RELEASES}
    return SpindleFullConfig(**{**values, **options})


@pytest.mark.unit
def test_the_option_is_off_by_default_and_parses_from_training_options(tmp_path):
    assert spindle(tmp_path).score_centering_settings is None
    parsed = spindle_full_settings(
        {
            "state-dir": str(tmp_path / "state"),
            "base-url": SPINDLE,
            "base": BASE,
            "release-repo": RELEASES,
            "score-centering": "true",
            "score-centering-top-k": "64",
            "score-centering-weight-low": "0",
            "score-centering-weight-high": "2",
        }
    )
    assert parsed.score_centering_settings == ScoreCentering(64, 1e-6, 0.0, 2.0)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"score-centering-top-k": "0"}, "top_k must be a positive int"),
        ({"score-centering-min-tail-mass": "1"}, r"min_tail_mass must be in \(0, 1\)"),
        ({"score-centering-weight-low": "2"}, "0 <= low <= high"),
        ({"score-centering-weight-high": "0", "score-centering-weight-low": "0"}, "high > 0"),
    ],
)
def test_invalid_score_centering_values_are_refused_even_when_it_is_off(tmp_path, options, message):
    values = {"state-dir": str(tmp_path / "state"), "base-url": SPINDLE, "base": BASE, "release-repo": RELEASES}
    with pytest.raises(DeployConfigError, match=message):
        spindle_full_settings({**values, **options})


@pytest.mark.unit
def test_the_loss_sends_cispo_with_the_sampler_head_of_each_trained_token():
    row = TokenRow(
        (1, 2, 3, 7, 8, 9),
        (1, 0, 1),
        (-0.1, 0.0, -0.3),
        0.5,
        ((7, 3, 4), (), (9, 5)),
        ((-0.1, -2.5, -3.0), (), (-0.3, -1.0)),
    )

    (inputs,) = ScoreCenteredLoss(CENTERING).inputs([row], [], kl_coef=0.0)

    assert ScoreCenteredLoss(CENTERING).loss_fn == "cispo"
    assert ScoreCenteredLoss(CENTERING).loss_fn_config == {
        "score_centering": 1.0,
        "score_centering_min_tail_mass": 1e-6,
        "clip_low_threshold": 1.0,
        "clip_high_threshold": 1.0,
    }
    # Two prompt positions predict the first two prompt tokens; the response's untrained token weighs 0.
    assert inputs["weights"] == [0.0, 0.0, 1.0, 0.0, 1.0]
    assert inputs["advantages"] == [0.0, 0.0, 0.5, 0.0, 0.5]
    assert inputs["sampler_topk_token_ids"] == MatrixInput(((0, 0), (0, 0), (7, 3), (0, 0), (9, 5)), "int64")
    assert inputs["sampler_topk_logprobs"].rows[2:] == (
        (-0.1, -2.5),
        (UNTRAINED_LOGPROB, UNTRAINED_LOGPROB),
        (-0.3, -1.0),
    )


@pytest.mark.unit
def test_a_trained_token_with_a_narrower_head_is_refused():
    row = TokenRow((1, 7), (1,), (-0.1,), 1.0, ((7,),), ((-0.1,),))
    with pytest.raises(ValueError, match="reads 2 sampler log probabilities per trained token"):
        ScoreCenteredLoss(CENTERING).inputs([row], [], kl_coef=0.0)


@pytest.mark.unit
def test_score_centering_trains_importance_sampling_as_cispo_and_leaves_sft_alone(objectives):
    policy, sft, tttd = objectives

    centered = prepare(policy, trajectory(**HEADS))
    assert isinstance(centered.loss, ScoreCenteredLoss)
    assert centered.steps[0][0].topk_indices == ((7, 3, 4), (8, 1, 2))
    assert isinstance(prepare(sft, trajectory()).loss, CrossEntropyLoss)
    with pytest.raises(ValueError, match="importance_sampling family only, not 'tttd'"):
        prepare(tttd, trajectory(**HEADS))
    with pytest.raises(ValueError, match="needs the sampler's top-k log probabilities of every response token"):
        prepare(policy, trajectory())


class TopkSampler(TinkerSampler):
    def __init__(self) -> None:
        self.widths: list[int] = []

    def render(self, messages, *, template_kwargs):
        return [1, 2, 3]

    def decode(self, tokens):
        return "done"

    def sample(self, sampler_path, prompt, params, *, topk_logprobs):
        self.widths.append(topk_logprobs)
        return SampleResult((7,), (-0.1,), "stop", (((7, -0.1), (3, -2.5)),))

    def close(self):
        return


def chat(endpoint, **sampling: object):
    body = json.dumps({"messages": [{"role": "user", "content": "hi"}], **sampling})
    request = urllib.request.Request(f"{endpoint.url}/v1/chat/completions", data=body.encode(), method="POST")
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


@pytest.mark.unit
def test_the_rollout_chat_records_the_sampler_head_at_temperature_one_only():
    sampler = TopkSampler()
    with ChatEndpoint(
        SamplerChat(sampler, None, model=MODEL, runtime_load_id="tinker:base", topk_logprobs=2)
    ) as endpoint:
        training = chat(endpoint, temperature=1.0)["training"]
        with pytest.raises(urllib.error.HTTPError) as refused:
            chat(endpoint, temperature=0.7)

    assert sampler.widths == [2]
    assert (training["topk_indices"], training["topk_log_probs"]) == ([[7, 3]], [[-0.1, -2.5]])
    assert refused.value.code == 400


@pytest.mark.unit
def test_a_sample_needs_one_head_row_of_one_width_per_token():
    with pytest.raises(ValueError, match="one width for every sampled token"):
        SampleResult((7, 8), (-0.1, -0.2), "stop", (((7, -0.1),),))
    with pytest.raises(ValueError, match="distinct and non-negative"):
        SampleResult((7,), (-0.1,), "stop", (((7, -0.1), (7, -2.0)),))


class Client(SpindleClient):
    def __init__(self) -> None:
        super().__init__()
        self.losses: list[object] = []

    def train(self, start, steps, loss, learning_rates, *, kl_coef):
        self.losses.append(loss)
        return super().train(start, steps, loss, learning_rates, kl_coef=kl_coef)


@pytest.mark.unit
def test_a_task_job_samples_the_head_and_trains_score_centered_cispo(tmp_path, monkeypatch):
    import reef.recipe.generated_tasks  # noqa: F401 - registers the task_group objective

    widths = []

    def fake_roll_out(tasks, settings, rollout_chat, **options):
        widths.append(rollout_chat.topk_logprobs)
        samples = (trajectory("add", 1.0, **HEADS), trajectory("add", 0.0, **HEADS))
        return TaskJobRollouts(
            samples, RolloutScores((("add", (RolloutScore.scored(1.0), RolloutScore.scored(0.0))),))
        )

    monkeypatch.setattr(component_module, "roll_out", fake_roll_out)
    client = Client()
    component = SpindleFullTrainingComponent(
        MODEL, spindle(tmp_path, score_centering=True, score_centering_top_k=3), client, Exporter(), sampler=client
    )
    component.start_jobs()
    try:
        task = tmp_path / "add"
        task.mkdir()
        (task / "task.toml").write_text('version = "1.0"\n')
        parameters = {
            "method": TrainingMethod("task_group").to_dict(),
            "rollout": {"rollouts_per_task": 2},
            "batch_size": 2,
        }
        component.submit(
            TrainingJob(
                "job-0",
                component.name,
                component.base_start,
                JobData("d", (TaskSample(task, metadata={"task": {"name": "add"}}),)),
                parameters,
                scenario="s",
            )
        )
        deadline = time.monotonic() + 10
        while (status := component.status("job-0")).state == "running":
            assert time.monotonic() < deadline, status
            time.sleep(0.01)
    finally:
        component.close()

    assert status.state == "finished", status
    assert widths == [3]
    (loss,) = client.losses
    assert isinstance(loss, ScoreCenteredLoss) and loss.sampler_topk == 3
