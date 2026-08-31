"""Streams: state carried across episodes, deterministic resets, honest resume."""

from pathlib import Path

import pytest

from reef_eval import FakeExecutor, Lab, Stream
from reef_eval.executors import STATE_TARGET, apply_state_mount

AGENT = {"name": "nop"}


def learning_executor():
    """A fake agent that 'learns': it appends the task it saw to a memory file
    in the carried state dir and scores by how much memory it has, so rewards
    over a working stream are 1, 2, 3, ... and any reset/restore bug shows up
    as a wrong count."""

    def score(spec):
        state = Path(spec.overrides["state_dir"])
        memory = state / "memory.txt"
        seen = memory.read_text().splitlines() if memory.exists() else []
        seen.append(spec.task)
        memory.write_text("\n".join(seen) + "\n")
        return {"reward": float(len(seen))}

    return FakeExecutor(score=score)


def memory_lines(path: Path) -> list[str]:
    return (path / "memory.txt").read_text().splitlines()


def snapshot(snapshots: Path, position: int) -> Path:
    """The snapshot for a position. Its name carries the prefix digest that
    the episode key uses, so look it up rather than assuming one."""
    hits = sorted(snapshots.glob(f"{position:03d}-*"))
    assert len(hits) == 1, f"expected one snapshot for {position}, got {hits}"
    return hits[0]


async def test_state_carries_across_positions(tmp_path):
    lab = Lab(tmp_path / "lab", executor=learning_executor())
    stream = Stream("s1", ["tasks/a", "tasks/b", "tasks/c"])

    rows = await stream.run(lab, AGENT)

    assert [r.rewards["reward"] for r in rows] == [1.0, 2.0, 3.0]
    assert [(r.tags["stream"], r.tags["position"]) for r in rows] == [
        ("s1", 0),
        ("s1", 1),
        ("s1", 2),
    ]
    # Each position's ending state is snapshotted, so every input is auditable.
    snapshots = stream.state_root(lab, AGENT) / "snapshots"
    assert memory_lines(snapshot(snapshots, 0)) == ["tasks/a"]
    assert memory_lines(snapshot(snapshots, 2)) == ["tasks/a", "tasks/b", "tasks/c"]


async def test_rerun_skips_everything(tmp_path):
    executor = learning_executor()
    lab = Lab(tmp_path / "lab", executor=executor)
    stream = Stream("s1", ["tasks/a", "tasks/b"])

    first = await stream.run(lab, AGENT)
    again = await stream.run(lab, AGENT)

    assert len(executor.calls) == 2  # nothing re-ran
    assert [r.key for r in again] == [r.key for r in first]


async def test_appending_tasks_resumes_with_restored_state(tmp_path):
    executor = learning_executor()
    lab = Lab(tmp_path / "lab", executor=executor)
    await Stream("s1", ["tasks/a", "tasks/b"]).run(lab, AGENT)

    # Dirty the live state between sessions: the reset must discard this,
    # because position 2's input is snapshot 001, not whatever live holds.
    live = Stream("s1", ["tasks/a"]).state_root(lab, AGENT) / "state"
    (live / "memory.txt").write_text("junk\njunk\njunk\n")

    rows = await Stream("s1", ["tasks/a", "tasks/b", "tasks/c"]).run(lab, AGENT)

    assert len(executor.calls) == 3  # only the appended position ran
    assert rows[2].rewards["reward"] == 3.0  # state restored from snapshot 001


async def test_crash_mid_stream_resumes_from_snapshot(tmp_path):
    def crash_on_b(spec):
        if spec.task == "tasks/b":
            raise RuntimeError("container died")
        return learning_executor()._score(spec)

    crashing = learning_executor()
    crashing._score = crash_on_b
    lab = Lab(tmp_path / "lab", executor=crashing)
    stream = Stream("s1", ["tasks/a", "tasks/b", "tasks/c"])

    with pytest.raises(RuntimeError, match="container died"):
        await stream.run(lab, AGENT)
    assert len(lab.df("episode")) == 1  # position 0 recorded, the crash not

    healthy = learning_executor()
    lab2 = Lab(tmp_path / "lab", executor=healthy)
    rows = await stream.run(lab2, AGENT)

    assert len(healthy.calls) == 2  # positions 1 and 2 only
    assert [r.rewards["reward"] for r in rows] == [1.0, 2.0, 3.0]


async def test_editing_history_invalidates_the_suffix(tmp_path):
    executor = learning_executor()
    lab = Lab(tmp_path / "lab", executor=executor)
    await Stream("s1", ["tasks/a", "tasks/b", "tasks/c"]).run(lab, AGENT)

    rows = await Stream("s1", ["tasks/a", "tasks/X", "tasks/c"]).run(lab, AGENT)

    # Position 0 is untouched history; 1 changed, so 1 and 2 are re-measured.
    assert len(executor.calls) == 3 + 2
    assert [r.rewards["reward"] for r in rows] == [1.0, 2.0, 3.0]
    snapshots = Stream("s1", ["tasks/a"]).state_root(lab, AGENT) / "snapshots"
    # Position 2 now has two snapshots, one per history; take the new one.
    latest = max(snapshots.glob("002-*"), key=lambda p: p.stat().st_mtime)
    assert memory_lines(latest) == ["tasks/a", "tasks/X", "tasks/c"]


async def test_distinct_agents_get_distinct_state_and_keys(tmp_path):
    executor = learning_executor()
    lab = Lab(tmp_path / "lab", executor=executor)
    stream = Stream("s1", ["tasks/a", "tasks/b"])

    rows_a = await stream.run(lab, {"name": "agent-a"})
    rows_b = await stream.run(lab, {"name": "agent-b"})

    assert len(executor.calls) == 4  # no cross-agent resume
    assert {r.key for r in rows_a}.isdisjoint({r.key for r in rows_b})
    # ... and each stream learned only from its own episodes.
    assert [r.rewards["reward"] for r in rows_b] == [1.0, 2.0]


async def test_same_name_different_tasks_keep_separate_state(tmp_path):
    """The state directory is per (name, setup, task list). Two streams that
    share a name and an agent but run different tasks must not inherit or
    overwrite each other's snapshots."""
    executor = learning_executor()
    lab = Lab(tmp_path / "lab", executor=executor)
    first = Stream("s1", ["tasks/a", "tasks/b"])
    second = Stream("s1", ["tasks/c", "tasks/d"])

    await first.run(lab, AGENT)
    rows = await second.run(lab, AGENT)

    # The second stream starts from empty memory, not from the first's.
    assert [r.rewards["reward"] for r in rows] == [1.0, 2.0]
    snapshots = first.state_root(lab, AGENT) / "snapshots"
    assert len(list(snapshots.glob("000-*"))) == 2  # one per task list


async def test_seed_state_is_visible_and_captured(tmp_path):
    executor = learning_executor()
    lab = Lab(tmp_path / "lab", executor=executor)
    stream = Stream("s1", ["tasks/a"])

    live = stream.state_root(lab, AGENT) / "state"
    live.mkdir(parents=True)
    (live / "memory.txt").write_text("pretrained\n")

    rows = await stream.run(lab, AGENT)

    assert rows[0].rewards["reward"] == 2.0  # the seed counted as memory
    snapshots = stream.state_root(lab, AGENT) / "snapshots"
    assert memory_lines(snapshots / "init") == ["pretrained"]


async def test_empty_stream_is_rejected():
    with pytest.raises(ValueError, match="at least one task"):
        Stream("s1", [])


def test_apply_state_mount_preserves_existing_config():
    agent, overrides = apply_state_mount(
        {"name": "nop", "env": {"KEEP": "1"}},
        {"environment": {"env": {"ALSO": "2"}}, "timeout_multiplier": 2.0},
        "/host/state",
    )
    assert agent["env"] == {"REEF_EVAL_STATE_DIR": STATE_TARGET, "KEEP": "1"}
    assert overrides["timeout_multiplier"] == 2.0
    env_cfg = overrides["environment"]
    assert env_cfg["env"] == {"REEF_EVAL_STATE_DIR": STATE_TARGET, "ALSO": "2"}
    assert env_cfg["mounts"] == [
        {"type": "bind", "source": "/host/state", "target": STATE_TARGET}
    ]
