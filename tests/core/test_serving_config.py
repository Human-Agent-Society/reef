"""``ServingConfig``: a scenario's serving fields by component, changed field by field (RFC #1097)."""

from __future__ import annotations

import math

import pytest

from reef.core.serving_config import ServingConfig


@pytest.mark.unit
def test_an_update_sets_and_removes_fields_and_keeps_the_rest():
    config = ServingConfig({"weights": {"temperature": 0.6, "min_replicas": 1}})
    changed = config.updated({"weights": {"temperature": None, "top_p": 0.9}, "harness": {"turns": 3}})
    assert changed.to_json() == {"weights": {"min_replicas": 1, "top_p": 0.9}, "harness": {"turns": 3}}
    # A component left with no field has no override; the original config does not change.
    assert changed.updated({"harness": {"turns": None}}).to_json() == {"weights": {"min_replicas": 1, "top_p": 0.9}}
    assert config.to_json() == {"weights": {"temperature": 0.6, "min_replicas": 1}}
    assert ServingConfig.from_json(changed.to_json()) == changed
    assert changed.component("weights") == {"min_replicas": 1, "top_p": 0.9}
    assert changed.component("unknown") == {}


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides",
    [
        {"weights": {}},
        {"weights": {"temperature": None}},
        {"": {"temperature": 0.6}},
        {"weights": {"temperature": math.inf}},
        {"weights": {"stop": [object()]}},
        {"weights": "fast"},
    ],
)
def test_a_config_holds_only_non_empty_overrides_of_finite_json_values(overrides):
    with pytest.raises(ValueError):
        ServingConfig(overrides)


@pytest.mark.unit
@pytest.mark.parametrize("update", ["fast", {"weights": "fast"}, {"weights": {"": 1}}, {"weights": {"t": math.nan}}])
def test_a_malformed_update_is_refused(update):
    with pytest.raises(ValueError):
        ServingConfig().updated(update)
