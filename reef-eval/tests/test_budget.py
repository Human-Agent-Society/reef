"""Budget: the four dimensions map to timeout, env signals, tags, and the
usage that comes back is recorded as used_* columns."""

import asyncio

import pytest

from reef_eval import Budget, Lab
from reef_eval.types import EpisodeResult


class RecordingExecutor:
    """Captures the spec it was handed and returns a fixed usage dict."""

    def __init__(self, usage=None):
        self.spec = None
        self.usage = usage or {}

    async def execute(self, spec):
        self.spec = spec
        return EpisodeResult(rewards={"reward": 1.0}, uri="fake://x", usage=self.usage)


# --------------------------------------------------------------- Budget object


def test_bare_number_is_hours():
    assert Budget.coerce(2) == Budget(time_h=2.0)
    assert Budget.coerce(2.5).timeout_sec() == 2.5 * 3600


def test_duration_strings():
    from reef_eval.budget import parse_duration_hours

    assert parse_duration_hours("2h") == 2.0
    assert parse_duration_hours("30m") == pytest.approx(0.5)
    assert parse_duration_hours("90s") == pytest.approx(90 / 3600)
    assert parse_duration_hours("1d") == 24.0
    assert parse_duration_hours("0.01") == 0.01  # bare = hours (back-compat)
    with pytest.raises(ValueError, match="unknown duration unit"):
        parse_duration_hours("5x")


def test_coerce_duration_string():
    assert Budget.coerce("30m") == Budget(time_h=0.5)
    assert Budget.coerce("90s").timeout_sec() == pytest.approx(90)


def test_dict_coercion_and_unknown_fields():
    assert Budget.coerce({"max_tokens": 100}) == Budget(max_tokens=100)
    with pytest.raises(ValueError, match="unknown budget fields"):
        Budget.coerce({"nope": 1})


def test_none_and_passthrough():
    assert Budget.coerce(None) is None
    b = Budget(time_h=1)
    assert Budget.coerce(b) is b


def test_non_positive_rejected():
    for bad in (dict(time_h=0), dict(max_tokens=-5), dict(max_submissions=0)):
        with pytest.raises(ValueError, match="must be positive"):
            Budget(**bad)


def test_is_empty():
    assert Budget().is_empty
    assert not Budget(max_tokens=1).is_empty


def test_to_env_only_set_dimensions():
    env = Budget(max_tokens=500_000, max_submissions=25).to_env()
    assert env == {"REEF_EVAL_MAX_TOKENS": "500000", "REEF_EVAL_MAX_SUBMISSIONS": "25"}
    assert "REEF_EVAL_BUDGET_SEC" not in env  # time not set


def test_to_tags_keeps_budget_hours_for_backcompat():
    tags = Budget(time_h=3, max_tokens=1000).to_tags()
    assert tags["budget"] == 3  # existing scaling queries still work
    assert tags["budget_time_h"] == 3
    assert tags["budget_max_tokens"] == 1000


# ------------------------------------------------------------------ Lab wiring


def test_budget_sets_timeout_env_and_tags(tmp_path):
    ex = RecordingExecutor()
    lab = Lab(tmp_path / "lab", executor=ex)
    row = asyncio.run(
        lab.run(
            "some/task",
            {"name": "a"},
            budget=Budget(time_h=2, max_tokens=500_000, max_submissions=50),
        )
    )
    # hard time -> container timeout
    assert ex.spec.agent["override_timeout_sec"] == 7200.0
    # signals delivered to agent (local/exec) AND container startup env
    assert ex.spec.agent["env"]["REEF_EVAL_MAX_TOKENS"] == "500000"
    assert ex.spec.overrides["environment"]["env"]["REEF_EVAL_MAX_SUBMISSIONS"] == "50"
    # budget recorded as grouping tags
    assert row.tags["budget"] == 2
    assert row.tags["budget_max_tokens"] == 500_000


def test_usage_recorded_as_used_columns(tmp_path):
    ex = RecordingExecutor(usage={"n_output_tokens": 800, "n_submissions": 7})
    lab = Lab(tmp_path / "lab", executor=ex)
    row = asyncio.run(lab.run("some/task", {"name": "a"}, budget=1))
    assert row.tags["used_n_output_tokens"] == 800
    assert row.tags["used_n_submissions"] == 7


def test_different_budgets_are_different_episodes(tmp_path):
    ex = RecordingExecutor()
    lab = Lab(tmp_path / "lab", executor=ex)
    r1 = asyncio.run(lab.run("some/task", {"name": "a"}, budget=Budget(max_tokens=100)))
    r2 = asyncio.run(lab.run("some/task", {"name": "a"}, budget=Budget(max_tokens=200)))
    assert r1.key != r2.key


def test_no_budget_is_unchanged(tmp_path):
    ex = RecordingExecutor()
    lab = Lab(tmp_path / "lab", executor=ex)
    asyncio.run(lab.run("some/task", {"name": "a"}))
    assert "env" not in ex.spec.agent
    assert "override_timeout_sec" not in ex.spec.agent


def test_max_evals_on_a_task_without_a_judge_warns(tmp_path, caplog):
    task = tmp_path / "verifier-only"
    (task / "environment").mkdir(parents=True)
    lab = Lab(tmp_path / "lab", executor=RecordingExecutor())
    with caplog.at_level("WARNING", logger="reef_eval"):
        asyncio.run(lab.run(str(task), {"name": "a"}, budget=Budget(max_submissions=5)))
        # once per Lab, however many episodes hit the same condition
        asyncio.run(lab.run(str(task), {"name": "b"}, budget=Budget(max_submissions=5)))
    assert sum("has no judge" in r.message for r in caplog.records) == 1


def test_no_warning_when_the_task_has_a_judge(tmp_path, caplog):
    task = tmp_path / "autoresearch-task"
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "judge_config.json").write_text('{"max_submissions": 100}')
    lab = Lab(tmp_path / "lab", executor=RecordingExecutor())
    with caplog.at_level("WARNING", logger="reef_eval"):
        asyncio.run(lab.run(str(task), {"name": "a"}, budget=Budget(max_submissions=5)))
    assert not [r for r in caplog.records if "has no judge" in r.message]


def test_no_warning_for_time_and_token_budgets(tmp_path, caplog):
    task = tmp_path / "verifier-only"
    (task / "environment").mkdir(parents=True)
    lab = Lab(tmp_path / "lab", executor=RecordingExecutor())
    with caplog.at_level("WARNING", logger="reef_eval"):
        asyncio.run(
            lab.run(str(task), {"name": "a"}, budget=Budget(time_h=1, max_tokens=1000))
        )
    assert not [r for r in caplog.records if "has no judge" in r.message]


# ------------------------------------------------------------- usage extraction


def test_harbor_usage_from_agent_context():
    from reef_eval.executors import _harbor_usage

    class Ctx:
        n_input_tokens = 1200
        n_output_tokens = 800
        n_cache_tokens = 0

    usage = _harbor_usage(Ctx(), n_submissions=5)
    assert usage["n_input_tokens"] == 1200
    assert usage["n_total_tokens"] == 2000  # input + output
    assert usage["n_submissions"] == 5
    assert "n_cache_tokens" not in usage  # zero is omitted


def test_harbor_usage_handles_missing_context():
    from reef_eval.executors import _harbor_usage

    assert _harbor_usage(None, n_submissions=3) == {"n_submissions": 3.0}


# ------------------------------------------------------------------ efficiency


def test_efficiency_reward_per_unit():
    import pandas as pd

    from reef_eval import metrics

    df = pd.DataFrame(
        [
            {"model": "a", "reward": 1.2, "used_n_total_tokens": 100_000},
            {"model": "a", "reward": 1.4, "used_n_total_tokens": 140_000},
            {"model": "b", "reward": 1.6, "used_n_total_tokens": 400_000},
        ]
    )
    out = metrics.efficiency(df, by=["model"], per=1000).set_index("model")
    # model a: mean reward 1.3 over mean 120k tokens -> per 1k = 1.3/120
    assert out.loc["a", "reward_per_unit"] == pytest.approx(1.3 / 120)
    assert out.loc["b", "reward_per_unit"] == pytest.approx(1.6 / 400)


def test_efficiency_says_so_when_the_spend_column_is_absent():
    """An empty frame here would read as "no efficiency to report" when the
    real answer is that the column asked for was never recorded."""
    import pandas as pd

    from reef_eval import metrics

    df = pd.DataFrame([{"reward": 1.0, "used_n_submissions": 4}])
    with pytest.raises(KeyError, match="used_n_submissions"):
        metrics.efficiency(df)  # defaults to used_n_total_tokens


def test_efficiency_drops_runs_that_reported_no_spend():
    """A harness that reports nothing is ordinary, and not the same thing
    as a column that does not exist."""
    import pandas as pd

    from reef_eval import metrics

    df = pd.DataFrame(
        [
            {"reward": 1.0, "used_n_total_tokens": 1000.0},
            {"reward": 9.0, "used_n_total_tokens": None},
        ]
    )
    out = metrics.efficiency(df)
    assert out.loc[0, "reward"] == pytest.approx(1.0)
    assert out.loc[0, "reward_per_unit"] == pytest.approx(0.001)
