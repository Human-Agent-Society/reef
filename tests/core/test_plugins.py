"""The one plugin form: a class reference, an optional config mapping, and ``from_config``."""

from __future__ import annotations

import pytest

from reef.core.plugins import ConfiguredPlugin, PluginConfigError, PluginReference
from reef.evaluation.harness import TaskScorer, VerifierRewardScorer

MODULE = "tests.core.test_plugins"


class FixedScore(TaskScorer):
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    @classmethod
    def from_config(cls, config):
        return cls(float(config.get("value", 0.0)))

    def score(self, task, result):
        return self.value


class NeedsArgument(TaskScorer):
    def __init__(self, value: float) -> None:
        self.value = value

    def score(self, task, result):
        return self.value


class AbstractScorer(TaskScorer):
    pass


def plain_function(task, result):
    return 1.0


@pytest.mark.unit
def test_the_string_form_builds_with_no_config() -> None:
    scorer = TaskScorer.from_setting(f"{MODULE}:FixedScore", setting="evolution.evaluation.scorer")
    assert isinstance(scorer, FixedScore) and scorer.value == 0.0


@pytest.mark.unit
def test_the_mapping_form_hands_its_config_to_from_config() -> None:
    setting = {"module": f"{MODULE}:FixedScore", "config": {"value": 0.5}}
    scorer = TaskScorer.from_setting(setting, setting="evolution.evaluation.scorer")
    assert isinstance(scorer, FixedScore) and scorer.value == 0.5


@pytest.mark.unit
def test_a_reference_is_checked_without_building_the_plugin() -> None:
    reference = TaskScorer.reference(f"{MODULE}:NeedsArgument", setting="evolution.evaluation.scorer")
    assert reference == PluginReference(NeedsArgument)


@pytest.mark.unit
def test_python_callers_may_pass_a_built_instance() -> None:
    scorer = VerifierRewardScorer()
    assert TaskScorer.from_setting(scorer, setting="evolution.evaluation.scorer") is scorer


@pytest.mark.unit
def test_the_default_from_config_refuses_config() -> None:
    with pytest.raises(PluginConfigError, match="VerifierRewardScorer takes no config, got threshold"):
        VerifierRewardScorer.from_config({"threshold": 1})
    assert isinstance(VerifierRewardScorer.from_config({}), ConfiguredPlugin)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("setting", "match"),
    [
        (None, r"must be 'package\.module:ClassName' or a mapping"),
        ("no-colon", r"must be 'package\.module:ClassName'"),
        (f"{MODULE}:Missing", r"cannot import evolution\.evaluation\.scorer"),
        (f"{MODULE}:plain_function", r"must name a class that inherits TaskScorer"),
        ("reef.train.harness_backend:Method", r"must name a class that inherits TaskScorer"),
        (f"{MODULE}:AbstractScorer", r"names an abstract class"),
        ({"module": f"{MODULE}:FixedScore", "extra": 1}, r"unknown key\(s\) extra"),
        ({"module": f"{MODULE}:FixedScore", "config": []}, r"evolution\.evaluation\.scorer\.config must be a mapping"),
    ],
)
def test_an_unusable_setting_is_refused(setting, match) -> None:
    with pytest.raises(PluginConfigError, match=match):
        TaskScorer.from_setting(setting, setting="evolution.evaluation.scorer")
