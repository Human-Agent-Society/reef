from __future__ import annotations

import pytest

from reef.train.slime_backend.reef_adapters.runtime_load_id import RuntimeLoadId, new_runtime_load_id_incarnation


@pytest.mark.unit
def test_runtime_load_id_namespaces_the_sequence_by_incarnation() -> None:
    assert str(RuntimeLoadId("engine-a", 7)) == "engine-a:7"
    assert RuntimeLoadId("engine-a", 1) != RuntimeLoadId("engine-b", 1)


@pytest.mark.unit
def test_runtime_load_id_round_trips_through_canonical_parser() -> None:
    runtime_load_id = RuntimeLoadId.parse("engine-a:17")

    assert runtime_load_id == RuntimeLoadId("engine-a", 17)
    assert str(runtime_load_id) == "engine-a:17"


@pytest.mark.unit
@pytest.mark.parametrize("value", ["", "engine-a", "engine-a:", ":1", "engine-a:-1", "engine-a:1.0"])
def test_runtime_load_id_parser_rejects_malformed_tokens(value: str) -> None:
    with pytest.raises(ValueError):
        RuntimeLoadId.parse(value)


@pytest.mark.unit
def test_runtime_load_id_incarnations_do_not_repeat_across_restarts() -> None:
    first = new_runtime_load_id_incarnation()
    second = new_runtime_load_id_incarnation()

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
def test_runtime_load_id_rejects_ambiguous_or_invalid_parts(incarnation, sequence) -> None:
    with pytest.raises(ValueError):
        RuntimeLoadId(incarnation, sequence)
