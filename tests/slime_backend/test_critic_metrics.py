"""Global EV must not depend on microbatch or distributed partitioning."""

import pytest

from reef.train.slime_backend.reef_adapters.critic_metrics import finalize_critic_metrics


@pytest.mark.parametrize("factor", [1.0, 0.5, 0.125])
def test_global_ev_from_constant_return_microbatches(factor):
    # Returns [0, 0] and [1, 1] each have zero local variance. Together,
    # errors [0, .2, -.2, 0] have variance .02 and returns have variance .25.
    moments = {
        "count": 4,
        "return_sum": 2,
        "return_square_sum": 2,
        "error_sum": 0,
        "error_square_sum": 0.08,
    }
    metrics = {f"train/critic-ev_{key}": value * factor for key, value in moments.items()}
    result = finalize_critic_metrics(metrics)
    assert result["train/critic-explained_variance"] == pytest.approx(0.92)
    assert result["train/critic-explained_variance_valid"] == 1
    assert not any("ev_" in key for key in result)


def test_constant_returns_are_explicitly_undefined():
    result = finalize_critic_metrics(
        {
            "ev_count": 2,
            "ev_return_sum": 2,
            "ev_return_square_sum": 2,
            "ev_error_sum": 1,
            "ev_error_square_sum": 1,
        }
    )
    assert result["explained_variance_valid"] == 0
    assert "explained_variance" not in result
