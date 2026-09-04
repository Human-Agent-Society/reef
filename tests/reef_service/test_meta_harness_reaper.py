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
