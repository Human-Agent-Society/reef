"""The observed-cost guard for the Terminal-Bench reproduction."""

from __future__ import annotations

from pathlib import Path

import pytest

from recipes.meta_harness.examples.terminal_bench.budget import ObservedCostLedger, SpendCapReached


@pytest.mark.unit
def test_a_cap_stops_the_next_trial_rather_than_the_running_one(tmp_path: Path) -> None:
    ledger = ObservedCostLedger(tmp_path / "spend.json", 1.0)
    ledger.before_trial("t1")
    ledger.record_trial("t1", 0.9)
    ledger.before_trial("t2")  # under cap, still allowed
    ledger.record_trial("t2", 0.5)  # overshoots by the cost of one in-flight trial
    with pytest.raises(SpendCapReached):
        ledger.before_trial("t3")


@pytest.mark.unit
def test_a_restart_re_accounts_the_same_trials_instead_of_doubling_them(tmp_path: Path) -> None:
    path = tmp_path / "spend.json"
    first = ObservedCostLedger(path, 10.0)
    first.record_trial("t1", 2.0)
    first.record_trial("t1", 2.0)  # idempotent
    assert first.observed_cost_usd == 2.0

    resumed = ObservedCostLedger(path, 10.0)
    assert resumed.observed_cost_usd == 2.0
    resumed.record_trial("t2", 3.0)
    assert resumed.observed_cost_usd == 5.0


@pytest.mark.unit
def test_a_trial_whose_cost_changed_across_restart_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "spend.json"
    ObservedCostLedger(path, 10.0).record_trial("t1", 2.0)
    with pytest.raises(RuntimeError, match="cost changed across restart"):
        ObservedCostLedger(path, 10.0).record_trial("t1", 4.0)


@pytest.mark.unit
def test_a_cap_below_recorded_spend_is_refused_at_open(tmp_path: Path) -> None:
    path = tmp_path / "spend.json"
    ObservedCostLedger(path, 10.0).record_trial("t1", 6.0)
    with pytest.raises(ValueError, match="below already recorded spend"):
        ObservedCostLedger(path, 5.0)


@pytest.mark.unit
@pytest.mark.parametrize("cap", [0.0, -1.0, float("inf"), float("nan")])
def test_an_unusable_cap_is_refused(tmp_path: Path, cap: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        ObservedCostLedger(tmp_path / "spend.json", cap)


class _Result:
    def __init__(self, trajectory) -> None:
        self.trajectory = trajectory


@pytest.mark.unit
def test_every_episode_is_accounted_even_with_no_verifier_event(tmp_path: Path) -> None:
    # An episode missing from the ledger is one the run cannot later tell it
    # attempted; a run recorded 577 trials for 600 episodes that way.
    from recipes.meta_harness.examples.terminal_bench.run import EpisodeScorer

    ledger = ObservedCostLedger(tmp_path / "spend.json", 10.0)
    scorer = EpisodeScorer(ledger)

    assert scorer("t", _Result([{"type": "verifier", "reward": 1.0, "cost_usd": 0.02}])) == 1.0
    assert scorer("t", _Result([{"type": "step"}])) == 0.0
    assert scorer("t", _Result([])) == 0.0
    assert scorer("t", _Result([{"type": "verifier", "reward": None, "cost_usd": None}])) == 0.0

    accounted, free = ledger.trial_tally()
    assert accounted == 4
    assert free == 3
