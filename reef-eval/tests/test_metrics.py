import pandas as pd
import pytest

from reef_eval import metrics


def test_anytime_best_so_far_and_groups():
    df = pd.DataFrame(
        {
            "task": ["a", "a", "a", "b", "b"],
            "t": [1, 2, 3, 1, 2],
            "score": [0.2, 0.1, 0.5, 0.4, 0.3],
        }
    )
    out = metrics.anytime(df, by=["task"])
    a = out[out.task == "a"]["best_so_far"].tolist()
    b = out[out.task == "b"]["best_so_far"].tolist()
    assert a == [0.2, 0.2, 0.5]
    assert b == [0.4, 0.4]


def test_auc_left_riemann():
    curve = pd.DataFrame({"t": [0, 10], "best_so_far": [0.0, 1.0]})
    assert metrics.auc(curve) == 0.0  # holds 0.0 until t=10
    curve = pd.DataFrame({"t": [0, 5, 10], "best_so_far": [0.0, 1.0, 1.0]})
    assert metrics.auc(curve) == pytest.approx(0.5)


def test_scaling_groups_by_budget():
    df = pd.DataFrame(
        {
            "budget": [2, 2, 4, 4],
            "reward": [0.1, 0.3, 0.5, 0.7],
        }
    )
    out = metrics.scaling(df)
    assert out.set_index("budget")["reward"].to_dict() == {2: 0.2, 4: 0.6}


def test_time_to_threshold():
    df = pd.DataFrame({"t": [1, 2, 3], "score": [0.2, 0.5, 0.9]})
    assert metrics.time_to(df, 0.5).iloc[0]["time_to"] == 2
    assert pd.isna(metrics.time_to(df, 0.95).iloc[0]["time_to"])


def test_time_to_grouped_includes_never_reached():
    df = pd.DataFrame(
        {
            "task": ["a", "a", "b", "b"],
            "t": [1, 2, 1, 2],
            "score": [0.3, 0.8, 0.1, 0.2],  # b never reaches 0.5
        }
    )
    out = metrics.time_to(df, 0.5, by=["task"]).set_index("task")
    assert out.loc["a", "time_to"] == 2
    assert pd.isna(out.loc["b", "time_to"])


def test_improvements_counts_and_rate():
    df = pd.DataFrame(
        {
            "t": [1, 2, 3, 4, 5],
            "score": [0.2, 0.1, 0.5, 0.5, 0.6],  # first, 0.5, 0.6 improve
        }
    )
    out = metrics.improvements(df)
    assert out.iloc[0]["evals"] == 5
    assert out.iloc[0]["improvements"] == 3
    assert out.iloc[0]["improvement_rate"] == pytest.approx(0.6)


def test_improvements_grouped():
    df = pd.DataFrame(
        {
            "task": ["a", "a", "b", "b", "b"],
            "t": [1, 2, 1, 2, 3],
            "score": [0.3, 0.4, 0.5, 0.2, 0.5],  # a: 2/2 · b: 1/3 (0.5 ties, not >)
        }
    )
    out = metrics.improvements(df, by=["task"]).set_index("task")
    assert out.loc["a", "improvement_rate"] == pytest.approx(1.0)
    assert out.loc["b", "improvements"] == 1
    assert out.loc["b", "improvement_rate"] == pytest.approx(1 / 3)


def test_learning_curve_cum_and_rolling_mean():
    df = pd.DataFrame(
        {
            "stream": ["s", "s", "s", "t", "t"],
            "position": [0, 1, 2, 0, 1],
            "reward": [0.0, 1.0, 1.0, 1.0, 0.0],
        }
    )
    out = metrics.learning_curve(df, by=["stream"], window=2)
    s = out[out.stream == "s"]
    assert s["cum_mean"].tolist() == pytest.approx([0.0, 0.5, 2 / 3])
    assert s["rolling_mean"].tolist() == pytest.approx([0.0, 0.5, 1.0])
    assert out[out.stream == "t"]["cum_mean"].tolist() == pytest.approx([1.0, 0.5])


def test_transfer_against_isolated_baseline():
    stream = pd.DataFrame({"task": ["a", "b", "c"], "reward": [0.8, 0.4, 0.5]})
    isolated = pd.DataFrame(
        {"task": ["a", "a", "b"], "reward": [0.5, 0.7, 0.4]}  # no baseline for c
    )
    out = metrics.transfer(stream, isolated).set_index("task")
    assert out.loc["a", "transfer"] == pytest.approx(0.2)  # 0.8 - mean(0.5, 0.7)
    assert out.loc["b", "transfer"] == pytest.approx(0.0)
    assert pd.isna(out.loc["c", "transfer"])


def test_forgetting_on_revisits():
    df = pd.DataFrame(
        {
            "task": ["a", "b", "a", "b", "c"],
            "position": [0, 1, 2, 3, 4],
            "reward": [0.9, 0.3, 0.6, 0.8, 1.0],  # a degraded, b improved
        }
    )
    out = metrics.forgetting(df).set_index("task")
    assert out.loc["a", "forgetting"] == pytest.approx(0.3)
    assert out.loc["b", "forgetting"] == pytest.approx(-0.5)
    assert "c" not in out.index  # visited once, excluded
    assert (out.loc["a", "first"], out.loc["a", "last"]) == (0, 2)


def test_rescale_linear_and_anchored():
    s = pd.Series([0.0, 5.0, 10.0, 15.0])
    lin = metrics.rescale_linear(s, lo=0, hi=10)
    assert lin.tolist() == [0.0, 50.0, 100.0, 100.0]  # clipped

    anch = metrics.rescale_anchored(s, baseline=0, top=10, super_anchor=20)
    assert anch.tolist() == [0.0, 50.0, 100.0, 150.0]  # beyond top stretches

    with pytest.raises(ValueError):
        metrics.rescale_linear(s, lo=1, hi=1)


def test_auc_over_a_fixed_window_rewards_reaching_a_score_early():
    """The default window is first-to-last submission, which is per-run and so
    not comparable across runs. Over a fixed window, getting there early wins."""
    early = metrics.anytime(pd.DataFrame({"t": [10, 1700], "score": [0.9, 0.9]}))
    late = metrics.anytime(pd.DataFrame({"t": [1690, 1700], "score": [0.9, 0.9]}))

    # Same number by default: both spans are flat at 0.9.
    assert metrics.auc(early) == pytest.approx(metrics.auc(late))

    # Over the run's actual budget they separate.
    assert metrics.auc(early, start=0, end=1700) > 0.85
    assert metrics.auc(late, start=0, end=1700) < 0.05
