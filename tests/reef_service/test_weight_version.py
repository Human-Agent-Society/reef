from __future__ import annotations

import pytest

from reef.train.slime_backend.reef_adapters.weight_version import WeightVersion, new_weight_version_incarnation


@pytest.mark.unit
def test_weight_version_namespaces_the_sequence_by_incarnation() -> None:
    assert str(WeightVersion("engine-a", 7)) == "engine-a:7"
    assert WeightVersion("engine-a", 1) != WeightVersion("engine-b", 1)


@pytest.mark.unit
def test_weight_version_round_trips_through_canonical_parser() -> None:
    version = WeightVersion.parse("engine-a:17")

    assert version == WeightVersion("engine-a", 17)
    assert str(version) == "engine-a:17"


@pytest.mark.unit
@pytest.mark.parametrize("value", ["", "engine-a", "engine-a:", ":1", "engine-a:-1", "engine-a:1.0"])
def test_weight_version_parser_rejects_malformed_tokens(value: str) -> None:
    with pytest.raises(ValueError):
        WeightVersion.parse(value)


@pytest.mark.unit
def test_weight_version_incarnations_do_not_repeat_across_restarts() -> None:
    first = new_weight_version_incarnation()
    second = new_weight_version_incarnation()

    assert first != second
    assert len(first) == len(second) == 32
    assert ":" not in first


@pytest.mark.unit
@pytest.mark.parametrize(
    ("incarnation", "sequence"),
    [
        ("", 0),
        ("invalid:incarnation", 0),
        ("engine", -1),
        ("engine", True),
    ],
)
def test_weight_version_rejects_ambiguous_or_invalid_parts(incarnation, sequence) -> None:
    with pytest.raises(ValueError):
        WeightVersion(incarnation, sequence)
