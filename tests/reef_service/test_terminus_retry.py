"""Retrying an episode that never got a sandbox.

The e2b account is shared. When another team saturates it, an episode fails to
start and Harbor records no reward, which scores zero and enters the mean as
though the candidate had failed the task. One run read as a collapse from
0.350 to 0.017 that way, with 92% of its episodes never starting.
"""

from __future__ import annotations

import pytest

from reef.harness.terminus.runner import (
    BACKOFF_CEILING_SECONDS,
    DEFAULT_MAX_ATTEMPTS,
    MAX_ATTEMPTS_ENV,
    backoff_seconds,
    is_transient,
    max_attempts,
)
from reef.harness.terminus.tree import TerminusTreeError


@pytest.mark.unit
@pytest.mark.parametrize(
    "error",
    [
        "RateLimitException: 429: Rate limit exceeded, please try again later.",
        "you have reached the maximum number of concurrent E2B sandboxes (100)",
        "litellm.RateLimitException",
    ],
)
def test_an_unavailable_environment_is_transient(error: str) -> None:
    assert is_transient(error)


@pytest.mark.unit
@pytest.mark.parametrize(
    "error",
    [
        "",
        "AgentTimeoutError: the agent ran past its limit",
        "VerifierOutputParseError",
        "BadRequestError: Function tools with reasoning_effort are not supported",
    ],
)
def test_a_real_failure_is_not_retried(error: str) -> None:
    # A task the agent genuinely failed, or a request the provider will refuse
    # every time, is a result. Retrying it would spend the budget to get the
    # same answer.
    assert not is_transient(error)


@pytest.mark.unit
def test_attempts_default_and_can_be_overridden(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(MAX_ATTEMPTS_ENV, raising=False)
    assert max_attempts() == DEFAULT_MAX_ATTEMPTS
    monkeypatch.setenv(MAX_ATTEMPTS_ENV, "7")
    assert max_attempts() == 7
    monkeypatch.setenv(MAX_ATTEMPTS_ENV, "0")
    assert max_attempts() == 1


@pytest.mark.unit
def test_a_non_numeric_attempt_count_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MAX_ATTEMPTS_ENV, "lots")
    with pytest.raises(TerminusTreeError):
        max_attempts()


@pytest.mark.unit
def test_backoff_grows_and_is_capped_and_jittered() -> None:
    first = [backoff_seconds(1) for _ in range(50)]
    later = [backoff_seconds(4) for _ in range(50)]
    assert min(later) > max(first)
    assert max(backoff_seconds(20) for _ in range(50)) <= BACKOFF_CEILING_SECONDS
    # Jitter keeps 16 concurrent episodes from retrying in lockstep.
    assert len(set(first)) > 1
