"""Campaign budget configuration rejects unbounded or ambiguous work."""

import pytest

from recipes.meta_harness.examples.terminal_bench.run import make_recipe

from .test_meta_harness_driver import arguments


@pytest.mark.parametrize("cap", [0, -1, float("inf"), float("nan")])
def test_cap_must_be_finite_and_positive(tmp_path, cap):
    args = arguments(tmp_path)
    args.max_observed_cost_usd = cap
    with pytest.raises(ValueError, match="finite and positive"):
        make_recipe(args, fingerprint={})


@pytest.mark.parametrize("timeout", [1800, float("inf"), float("nan")])
def test_outer_timeout_preserves_phase_budgets(tmp_path, timeout):
    args = arguments(tmp_path)
    args.episode_timeout_s = timeout
    with pytest.raises(ValueError, match="episode timeout"):
        make_recipe(args, fingerprint={})


def test_shared_account_concurrency_is_bounded(tmp_path):
    args = arguments(tmp_path)
    args.concurrency = 33
    with pytest.raises(ValueError, match="shared 32-sandbox"):
        make_recipe(args, fingerprint={})


def test_task_duplicates_cannot_reuse_trial_identities(tmp_path):
    args = arguments(tmp_path)
    args.tasks = "terminal-bench/a,terminal-bench/a"
    with pytest.raises(ValueError, match="unique"):
        make_recipe(args, fingerprint={})
