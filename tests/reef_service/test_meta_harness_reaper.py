"""The sandbox reaper's kill list.

An e2b account can be shared. Reaping on age alone once killed 97 sandboxes,
most belonging to an unrelated benchmark, so ownership is checked first.
"""

from __future__ import annotations

import datetime

import pytest

from recipes.meta_harness.examples.terminal_bench.reap_sandboxes import select_doomed

# older_than is in seconds; 14400s is the shipped default (12000s longest task + margin).
NOW = datetime.datetime(2026, 9, 4, 12, 0, tzinfo=datetime.timezone.utc)
OURS = {"password-recovery", "bn-fit-modify"}


class _Sandbox:
    def __init__(self, environment: str | None, age_minutes: float) -> None:
        self.sandbox_id = f"sb-{environment}-{age_minutes}"
        self.metadata = {"environment_name": environment} if environment else None
        self.started_at = NOW - datetime.timedelta(minutes=age_minutes)


@pytest.mark.unit
def test_a_sandbox_from_another_benchmark_is_never_reaped() -> None:
    foreign = _Sandbox("dynamodb-toolbox-conditional-attribute-requirements", age_minutes=600)
    assert select_doomed([foreign], OURS, NOW, older_than=14400) == []


@pytest.mark.unit
def test_ignoring_age_still_does_not_reach_another_benchmark() -> None:
    foreign = _Sandbox("actionlint-action-pinning-lint", age_minutes=600)
    assert select_doomed([foreign], OURS, NOW, older_than=14400, ignore_age=True) == []


@pytest.mark.unit
def test_a_sandbox_with_no_metadata_is_left_alone() -> None:
    unlabelled = _Sandbox(None, age_minutes=600)
    assert select_doomed([unlabelled], OURS, NOW, older_than=14400) == []


@pytest.mark.unit
def test_our_own_stale_sandbox_is_reaped() -> None:
    stale = _Sandbox("password-recovery", age_minutes=600)
    assert select_doomed([stale], OURS, NOW, older_than=14400) == [stale]


@pytest.mark.unit
def test_our_own_young_sandbox_survives_because_an_episode_may_still_be_running() -> None:
    running = _Sandbox("bn-fit-modify", age_minutes=30)
    assert select_doomed([running], OURS, NOW, older_than=14400) == []


@pytest.mark.unit
def test_no_flag_reaches_a_foreign_sandbox() -> None:
    # The account is shared; a foreign sandbox is someone's running job, so
    # neither ignoring age nor any other option may select it.
    foreign = _Sandbox("some-other-suite", age_minutes=600)
    assert select_doomed([foreign], OURS, NOW, older_than=14400) == []
    assert select_doomed([foreign], OURS, NOW, older_than=14400, ignore_age=True) == []


@pytest.mark.unit
def test_live_ours_counts_only_this_benchmarks_sandboxes() -> None:
    from recipes.meta_harness.examples.terminal_bench.reap_sandboxes import live_ours

    class _Module:
        @staticmethod
        def list():
            class _Page:
                def __init__(self):
                    self._sent = False
                    self.has_next = False

                def next_items(self):
                    if self._sent:
                        return []
                    self._sent = True
                    return [
                        _Sandbox("password-recovery", 5),
                        _Sandbox("bn-fit-modify", 5),
                        _Sandbox("some-other-suite", 5),
                    ]

            return _Page()

    assert live_ours(_Module, OURS) == 2


@pytest.mark.unit
def test_a_run_refuses_to_start_without_headroom(monkeypatch: pytest.MonkeyPatch) -> None:
    # Harbor swallows a failed sandbox delete, so leftovers from a previous run
    # spend this run's share before it starts.
    import sys
    import types

    from recipes.meta_harness.examples.terminal_bench import reap_sandboxes as reaper
    from recipes.meta_harness.examples.terminal_bench import run as driver

    fake = types.ModuleType("e2b")
    fake.Sandbox = object
    monkeypatch.setitem(sys.modules, "e2b", fake)
    monkeypatch.setattr(reaper, "live_ours", lambda _module, _ours: 20)

    with pytest.raises(SystemExit) as excinfo:
        driver.check_sandbox_headroom(concurrency=16, cap=32)
    assert "over the 32 share" in str(excinfo.value)

    # 20 live plus 10 more is inside the share, so it proceeds.
    driver.check_sandbox_headroom(concurrency=10, cap=32)


@pytest.mark.unit
def test_the_headroom_check_is_skipped_without_e2b(monkeypatch: pytest.MonkeyPatch) -> None:
    # The render path and CI have no e2b; the check must not become a hard
    # dependency of running the driver at all.
    import sys

    from recipes.meta_harness.examples.terminal_bench import run as driver

    monkeypatch.setitem(sys.modules, "e2b", None)
    driver.check_sandbox_headroom(concurrency=16, cap=32)
