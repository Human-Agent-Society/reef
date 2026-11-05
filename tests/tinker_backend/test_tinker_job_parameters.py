"""The JSON Schema of a Tinker job's parameters."""

from __future__ import annotations

import jsonschema

from reef.train.tinker_backend.parameters import tinker_parameters_schema

ROLLOUT = {"rollouts_per_task": 4, "environment": "docker", "concurrency": 2}


def test_the_schema_of_a_kind_says_what_the_component_takes() -> None:
    for kind in ("trajectory", "task"):
        jsonschema.Draft202012Validator.check_schema(tinker_parameters_schema(kind))
    task = jsonschema.Draft202012Validator(tinker_parameters_schema("task"))
    trajectory = jsonschema.Draft202012Validator(tinker_parameters_schema("trajectory"))

    assert task.is_valid({"objective": "sft", "rollout": ROLLOUT, "batch_size": 2, "learning_rate": 1e-5})
    assert trajectory.is_valid({"objective": "sft", "kl_coef": 0.1, "save_interval": 2})

    assert not task.is_valid({"objective": "sft"}), "a task job needs rollout"
    assert not trajectory.is_valid({"objective": "sft", "rollout": ROLLOUT}), "a trajectory job takes no rollout"
    assert not trajectory.is_valid({"objective": "sft", "harbor": ROLLOUT}), "no key but the backend's"
    assert not task.is_valid({"objective": "sft", "rollout": {"rollouts_per_task": 1}})
    assert not trajectory.is_valid({"batch_size": 2}), "a job needs its objective"
