import asyncio

import pytest

from reef_eval import FakeExecutor, Lab
from reef_eval.types import EpisodeResult, TracePoint


@pytest.fixture()
def fake():
    return FakeExecutor(score=lambda spec: {"reward": 1.0})


@pytest.fixture()
def lab(tmp_path, fake):
    return Lab(tmp_path / "lab", executor=fake)


async def test_run_stores_episode_row(lab):
    row = await lab.run("tasks/demo", {"name": "nop"}, tags={"phase": 0})
    assert row.kind == "episode"
    assert row.rewards == {"reward": 1.0}
    df = lab.df("episode")
    assert len(df) == 1
    assert df["phase"].iloc[0] == 0
    assert df["uri"].iloc[0] == "fake://tasks/demo"


async def test_idempotency_skips_and_returns_cached(lab, fake):
    r1 = await lab.run("tasks/demo", {"name": "nop"}, key="fixed")
    r2 = await lab.run("tasks/demo", {"name": "nop"}, key="fixed")
    assert r1.key == r2.key
    assert len(fake.calls) == 1  # second call never reached the executor


async def test_default_key_is_stable_and_tag_sensitive(lab, fake):
    await lab.run("tasks/demo", {"name": "nop"}, tags={"phase": 1})
    await lab.run("tasks/demo", {"name": "nop"}, tags={"phase": 1})
    await lab.run("tasks/demo", {"name": "nop"}, tags={"phase": 2})
    assert len(fake.calls) == 2  # same tags dedup, different tags run


async def test_trace_rows_recorded(tmp_path):
    fake = FakeExecutor(
        score=lambda s: {"reward": 0.8},
        trace=lambda s: [
            TracePoint(t=1.0, score=0.2),
            TracePoint(t=5.0, score=0.6, data={"snapshot": "a.json"}),
        ],
    )
    lab = Lab(tmp_path / "lab", executor=fake)
    await lab.run("tasks/ar", {"name": "x"}, key="ep")
    trace = lab.df("trace").sort_values("t")
    assert list(trace["score"]) == [0.2, 0.6]
    assert trace["snapshot"].iloc[1] == "a.json"
    assert list(trace["key"]) == ["ep#t0", "ep#t1"]


async def test_run_many_bounded_concurrency(tmp_path):
    peak = 0
    running = 0

    class SlowExec:
        async def execute(self, spec):
            nonlocal peak, running
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.01)
            running -= 1

            return EpisodeResult(rewards={"reward": 1.0})

    lab = Lab(tmp_path / "lab", executor=SlowExec(), concurrency=2)
    rows = await lab.run_many(
        [{"task": f"t/{i}", "agent": {"name": "nop"}} for i in range(6)]
    )
    assert len(rows) == 6
    assert peak <= 2


async def test_error_recorded_in_tags(tmp_path):
    class FailingExec:
        async def execute(self, spec):
            return EpisodeResult(rewards={}, error="AgentTimeoutError: 900s")

    lab = Lab(tmp_path / "lab", executor=FailingExec())
    row = await lab.run("t/x", {"name": "nop"})
    assert "AgentTimeoutError" in row.tags["error"]


async def test_usage_is_recorded_as_episode_metrics(tmp_path):
    class MeteredExec:
        async def execute(self, spec):
            return EpisodeResult(
                rewards={"reward": 0.8},
                usage={
                    "n_input_tokens": 1_000,
                    "n_cache_tokens": 600,
                    "n_output_tokens": 200,
                },
            )

    lab = Lab(tmp_path / "lab", executor=MeteredExec())
    row = await lab.run("t/x", {"name": "metered"})
    assert row.rewards == {"reward": 0.8}
    assert row.tags["used_n_input_tokens"] == 1_000
    assert row.tags["used_n_cache_tokens"] == 600
    assert row.tags["used_n_output_tokens"] == 200


async def test_concurrent_identical_calls_share_one_execution(tmp_path):
    # Deterministic overlap: the leader blocks inside execute, so the second
    # call is guaranteed to observe the in-flight claim rather than a miss.
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    class GatedExec:
        async def execute(self, spec):
            nonlocal calls
            calls += 1
            started.set()
            await release.wait()
            return EpisodeResult(rewards={"reward": 1.0}, uri="fake://gated")

    lab = Lab(tmp_path / "lab", executor=GatedExec())
    leader = asyncio.create_task(lab.run("tasks/demo", {"name": "nop"}, key="ep"))
    await started.wait()  # the leader is parked inside execute
    waiter = asyncio.create_task(lab.run("tasks/demo", {"name": "nop"}, key="ep"))
    release.set()
    r1, r2 = await asyncio.gather(leader, waiter)

    assert calls == 1  # the executor ran exactly once
    assert r1 is r2  # both callers received the same completed Row
    assert {r.key for r in (r1, r2)} == {"ep"}
    assert lab._inflight == {}  # the in-flight claim was released


async def test_concurrent_waiters_receive_executor_failure(tmp_path):
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    class GatedFailingExec:
        async def execute(self, spec):
            nonlocal calls
            calls += 1
            started.set()
            await release.wait()
            raise RuntimeError("boom")

    lab = Lab(tmp_path / "lab", executor=GatedFailingExec())
    leader = asyncio.create_task(lab.run("tasks/demo", {"name": "nop"}, key="ep"))
    await started.wait()  # the leader is parked inside execute
    waiter = asyncio.create_task(lab.run("tasks/demo", {"name": "nop"}, key="ep"))
    await asyncio.sleep(0.05)  # let the waiter park on the in-flight future
    release.set()

    r1, r2 = await asyncio.gather(leader, waiter, return_exceptions=True)
    assert calls == 1  # only the leader called the executor
    assert all(isinstance(r, RuntimeError) and str(r) == "boom" for r in (r1, r2))
    assert lab._inflight == {}  # the claim was released despite the failure


async def test_distinct_keys_still_run_concurrently(tmp_path):
    # Both distinct keys must enter execute before either can finish: the gate
    # is only opened when the second execution starts. If distinct keys were
    # serialized through a shared claim, this would deadlock (time out).
    gate = asyncio.Event()
    reached = 0

    class GatedExec:
        async def execute(self, spec):
            nonlocal reached
            reached += 1
            if reached == 2:
                gate.set()
            await gate.wait()
            return EpisodeResult(rewards={"reward": 1.0})

    lab = Lab(tmp_path / "lab", executor=GatedExec(), concurrency=4)
    await asyncio.wait_for(
        asyncio.gather(
            lab.run("t/a", {"name": "nop"}, key="a"),
            lab.run("t/b", {"name": "nop"}, key="b"),
        ),
        timeout=5,
    )
    assert reached == 2  # both keys executed, neither was deduplicated


async def test_inflight_claim_released_after_executor_failure(tmp_path):
    attempts = 0

    class FailOnceExec:
        async def execute(self, spec):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("boom")
            return EpisodeResult(rewards={"reward": 1.0})

    lab = Lab(tmp_path / "lab", executor=FailOnceExec())
    with pytest.raises(RuntimeError, match="boom"):
        await lab.run("tasks/demo", {"name": "nop"}, key="ep")
    assert lab._inflight == {}  # claim released so a later call can retry

    # A later call for the same key retries from scratch and succeeds.
    row = await lab.run("tasks/demo", {"name": "nop"}, key="ep")
    assert row.rewards == {"reward": 1.0}
    assert attempts == 2
    assert lab._inflight == {}


async def test_inflight_claim_released_after_cancellation(tmp_path):
    entered = asyncio.Event()

    class BlockingExec:
        async def execute(self, spec):
            entered.set()
            await asyncio.sleep(60)

    lab = Lab(tmp_path / "lab", executor=BlockingExec())
    task = asyncio.create_task(lab.run("tasks/demo", {"name": "nop"}, key="ep"))
    await entered.wait()  # the leader is parked inside execute
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert lab._inflight == {}  # cancellation released the claim

    # A later call becomes a fresh leader rather than finding a stale claim.
    fresh = asyncio.Event()

    class FreshExec:
        async def execute(self, spec):
            fresh.set()
            return EpisodeResult(rewards={"reward": 1.0})

    lab.executor = FreshExec()
    row = await lab.run("tasks/demo", {"name": "nop"}, key="ep")
    assert fresh.is_set()
    assert row.rewards == {"reward": 1.0}


async def test_run_many_dedups_identical_calls(tmp_path):
    fake = FakeExecutor(score=lambda spec: {"reward": 1.0})
    lab = Lab(tmp_path / "lab", executor=fake)
    rows = await lab.run_many(
        [
            {"task": "tasks/demo", "agent": {"name": "nop"}, "key": "same"},
            {"task": "tasks/demo", "agent": {"name": "nop"}, "key": "same"},
        ]
    )
    assert len(fake.calls) == 1  # the second call never reached the executor
    assert len(rows) == 2
    assert rows[0] == rows[1]  # same completed episode (equal, maybe not identical)


async def test_a_dead_agent_is_visible_in_its_row(tmp_path):
    """A reward earned by an agent that exited non-zero reads exactly like one
    it worked for, unless the exit is recorded. It is a tag, not an error:
    spending the whole time budget also ends the container non-zero."""

    class DiedExecutor:
        async def execute(self, spec):
            return EpisodeResult(rewards={"reward": 0.0}, agent_exit_code=1)

    lab = Lab(tmp_path / "lab", executor=DiedExecutor())
    row = await lab.run("tasks/x", {"name": "oracle"})

    assert row.rewards == {"reward": 0.0}
    assert row.tags["agent_exit_code"] == 1
    assert "error" not in row.tags  # the score still stands


async def test_a_clean_agent_adds_no_exit_tag(tmp_path):
    lab = Lab(tmp_path / "lab", executor=FakeExecutor())
    row = await lab.run("tasks/x", {"name": "oracle"})
    assert "agent_exit_code" not in row.tags
