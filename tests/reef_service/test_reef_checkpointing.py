import pytest

import reef.scenario.checkpoint_strategy as checkpointing
from reef.scenario.checkpoint_strategy import EveryNVersions
from reef.train.types import TrainStepResult


def test_every_n_versions_one_selects_every_positive_version() -> None:
    result = TrainStepResult(state=None)
    strategy = EveryNVersions(1)

    assert strategy.should_checkpoint("math", 1, result)
    assert strategy.should_checkpoint("math", 2, result)


def test_every_n_versions_selects_only_multiples() -> None:
    strategy = EveryNVersions(3)
    result = TrainStepResult(state=None)

    assert not strategy.should_checkpoint("math", 1, result)
    assert not strategy.should_checkpoint("math", 2, result)
    assert strategy.should_checkpoint("math", 3, result)


@pytest.mark.parametrize("n", [0, -1])
def test_every_n_versions_rejects_non_positive_interval(n: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        EveryNVersions(n)


def test_redundant_every_version_symbol_is_removed() -> None:
    assert not hasattr(checkpointing, "EveryVersion")
