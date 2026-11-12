"""The shape of a Slime job's parameters, its JSON Schema, and the startup loss family from a recipe's defaults."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jsonschema
import pytest

from reef.core.errors import RuntimeConfigError
from reef.recipe.dataset import DatasetRecipe
from reef.recipe.generated_tasks import GeneratedTaskRecipe
from reef.recipe.sft import SFTRecipe
from reef.train.algos.schedule import StepScheduling
from reef.train.learning_rate import LearningRateSchedule
from reef.train.slime_backend.dataset import DatasetSettings
from reef.train.slime_backend.launch import SlimeComponentFactory, SlimeDeployment
from reef.train.slime_backend.parameters import SlimeJobParameters, slime_parameters_schema
from reef.train.task_rollouts import HarborSettings

SCHEDULE = {"name": "warmup", "peak_learning_rate": 1e-5, "warmup_steps": 2}
HARBOR = {
    "environment": "docker",
    "concurrency": 3,
    "agent": {"name": "pi"},
    "agent_host": "host.docker.internal",
    "verifier_timeout_reward": None,
    "scaffold_tolerance": 2,
}


def test_the_parameters_of_a_task_job_parse_into_their_typed_value() -> None:
    parameters = SlimeJobParameters.from_parameters(
        {
            "objective": "sft",
            "learning_rate_schedule": SCHEDULE,
            "scheduling": {"shuffle": True},
            "save_interval": 2,
            "harbor": HARBOR,
        }
    )
    assert parameters == SlimeJobParameters(
        "sft",
        LearningRateSchedule("warmup", 1e-5, warmup_steps=2),
        StepScheduling(shuffle=True),
        2,
        None,
        HarborSettings(**HARBOR),
    )


def test_the_parameters_of_a_dataset_job_parse_into_their_typed_value() -> None:
    parameters = SlimeJobParameters.from_parameters(
        {"objective": "slime_dataset", "dataset": {"path": "/d.jsonl", "epochs": 2}}
    )
    assert parameters.dataset == DatasetSettings(Path("/d.jsonl"), 2)
    assert parameters.harbor is None


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"method": {"objective": "sft"}}, r"accept no parameters \['method'\]"),
        ({"rollout": HARBOR}, r"accept no parameters \['rollout'\]"),
        ({"objective": "nothing"}, "unknown Slime objective"),
        ({"scheduling": {"unit": "sample"}}, "scheduling parameter must be an object"),
        ({"learning_rate_schedule": {**SCHEDULE, "unknown": 1}}, "unknown or missing fields"),
        ({"save_interval": 0}, "save_interval must be a positive int"),
        ({"dataset": {"path": "relative.jsonl"}}, "dataset.path must be an absolute path"),
        ({"dataset": {"path": "/d.jsonl", "shuffle": True}}, "dataset parameter is an object"),
        ({"harbor": None}, "needs the harbor parameter as an object"),
        ({"harbor": {"n_samples_per_prompt": 4}}, r"harbor parameter takes no \['n_samples_per_prompt'\]"),
        ({"harbor": {"rollouts_per_task": 4}}, r"harbor parameter takes no \['rollouts_per_task'\]"),
        ({"harbor": {"concurrency": 0}}, "concurrency must be an int of at least 1"),
    ],
)
def test_a_malformed_parameter_is_refused(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SlimeJobParameters.from_parameters({"objective": "sft", **overrides})


def test_the_schema_of_a_kind_says_what_the_parser_takes() -> None:
    for kind in ("trajectory", "task"):
        jsonschema.Draft202012Validator.check_schema(slime_parameters_schema(kind))
    task = jsonschema.Draft202012Validator(slime_parameters_schema("task"))
    trajectory = jsonschema.Draft202012Validator(slime_parameters_schema("trajectory"))

    complete = {"objective": "sft", "learning_rate_schedule": SCHEDULE, "harbor": HARBOR, "save_interval": 2}
    assert task.is_valid(complete)
    SlimeJobParameters.from_parameters(complete)
    dataset = {"objective": "slime_dataset", "dataset": {"path": "/d.jsonl", "epochs": 2}, "scheduling": {}}
    assert trajectory.is_valid(dataset)
    SlimeJobParameters.from_parameters(dataset)

    assert not task.is_valid({"objective": "sft"}), "a task job needs harbor"
    assert not trajectory.is_valid({"objective": "sft", "harbor": HARBOR}), "a trajectory job takes no harbor"
    assert not task.is_valid({**complete, "dataset": {"path": "/d.jsonl"}}), "a task job has its samples"
    assert not trajectory.is_valid({"objective": "sft", "method": {}}), "no key but the backend's"
    assert not task.is_valid({**complete, "harbor": {**HARBOR, "unknown": 1}})
    assert not task.is_valid({**complete, "harbor": {"rollouts_per_task": 4}}), "n-samples-per-prompt gives the count"
    assert task.is_valid({"objective": "slime_dataset", "harbor": {}}), "every harbor key has a default"
    assert not trajectory.is_valid({"save_interval": 2}), "a job needs its objective"


@pytest.mark.parametrize(
    ("parameters", "loss_family"),
    [
        (SFTRecipe.configured_parameters(None), "sft"),
        (DatasetRecipe.configured_parameters(None), "slime"),
        # The recipe names no objective: the deployment's config does.
        (GeneratedTaskRecipe.configured_parameters({"objective": "slime_dataset"}), "slime"),
    ],
)
def test_the_backend_reads_the_startup_loss_family_from_the_recipes_parameters(parameters, loss_family) -> None:
    settings = {
        "recipe": "r",
        "ray_address": "",
        "ray_namespace": "n",
        "execution_config": {},
        "train_timeout_s": None,
        "model_path": "m",
        "training_backend_options": {},
        "teacher_model_path": None,
        "teacher_timeout": None,
    }
    config = SlimeDeployment().component_config(settings, default_parameters=parameters)
    assert config["loss_family"] == loss_family


def test_harbor_and_the_objective_are_independent() -> None:
    for objective in ("sft", "slime_dataset"):
        harbor = SlimeJobParameters.from_parameters({"objective": objective, "harbor": HARBOR}).harbor
        assert harbor == HarborSettings(**HARBOR)


def test_a_recipe_without_a_default_objective_cannot_start_slime() -> None:
    config = {"loss_family": "", "settings": {}, "ray_address": "ray://host", "train_timeout_s": None}
    with pytest.raises(RuntimeConfigError, match="default objective"):
        SlimeComponentFactory()(config, "m", {})
