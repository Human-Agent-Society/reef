"""The values of the native qualification's checks that need no engine, model or GPU."""

import pytest
from tests.smoke.inference_features import ExpectedToken


@pytest.mark.unit
@pytest.mark.parametrize(
    ("token_id", "log_probability", "alternatives"),
    [
        # The generated token is the most likely one, with that pair's score.
        (5, -0.5, ((5, -0.5), (7, -1.0))),
        # Tied alternatives in either order, and a generated token outside them that ties with the last one.
        (9, -1.0, ((5, -1.0), (7, -1.0))),
        (9, -1.0, ((7, -1.0), (5, -1.0))),
    ],
)
def test_an_expected_token_accepts_consistent_scores(token_id, log_probability, alternatives) -> None:
    position = ExpectedToken(token_id, log_probability, alternatives)
    assert position.known() == {**dict(alternatives), token_id: log_probability}


@pytest.mark.unit
@pytest.mark.parametrize(
    ("token_id", "log_probability", "alternatives", "match"),
    [
        # A listed generated token with another score than its pair's.
        (0, -10.0, ((0, -1.0), (1, -2.0)), "disagrees"),
        # A generated token outside the most likely ones that scores above them.
        (3, 0.0, ((0, -1.0), (1, -2.0)), "disagrees"),
        (-1, -3.0, ((0, -1.0), (1, -2.0)), "non-negative integers"),
        (3, -3.0, ((1, -2.0), (0, -1.0)), "most likely token comes first"),
        (3, -3.0, ((0, -1.0), (0, -1.0)), "distinct most likely tokens"),
        (3, -3.0, ((0, -1.0),), "distinct most likely tokens"),
        (3, float("nan"), ((0, -1.0), (1, -2.0)), "finite and at most 0"),
    ],
)
def test_an_expected_token_refuses_inconsistent_scores(token_id, log_probability, alternatives, match) -> None:
    with pytest.raises(ValueError, match=match):
        ExpectedToken(token_id, log_probability, alternatives)
