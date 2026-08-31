import pytest

from reef_eval.store import Store
from reef_eval.types import Row


@pytest.fixture()
def store(tmp_path):
    return Store(tmp_path / "results.sqlite")


def row(key="k1", **kw):
    defaults = dict(
        key=key,
        kind="episode",
        task="t/x",
        tags={"phase": 1},
        rewards={"reward": 0.5},
        uri="fake://x",
    )
    defaults.update(kw)
    return Row(**defaults)


def test_put_get_roundtrip(store):
    store.put(row())
    got = store.get("k1")
    assert got.task == "t/x"
    assert got.tags == {"phase": 1}
    assert got.rewards == {"reward": 0.5}


def test_duplicate_key_is_loud(store):
    import sqlite3

    store.put(row())
    assert store.get("k1") is not None
    assert store.get("k2") is None
    with pytest.raises(sqlite3.IntegrityError):
        store.put(row())


def test_df_expands_tags_and_rewards(store):
    store.put(row(key="a", tags={"phase": 1, "arm": "fresh"}))
    store.put(
        row(
            key="b",
            tags={"phase": 2, "arm": "stateful"},
            rewards={"reward": 0.9, "steps": 12},
        )
    )
    df = store.df()
    assert set(df.columns) >= {"key", "kind", "task", "phase", "arm", "reward", "steps"}
    assert df.loc[df.key == "b", "steps"].iloc[0] == 12


def test_df_kind_filter_and_empty(store):
    assert store.df().empty
    store.put(row(key="e1", kind="episode"))
    store.put(row(key="e1#t0", kind="trace", rewards={"score": 0.1}))
    assert list(store.df("trace")["key"]) == ["e1#t0"]


def test_reward_column_collision_prefixed(store):
    store.put(row(key="c", tags={"reward": "tag-wins"}, rewards={"reward": 1.0}))
    df = store.df()
    assert df["reward"].iloc[0] == "tag-wins"
    assert df["reward_reward"].iloc[0] == 1.0


def test_tag_colliding_with_base_column_is_prefixed(store):
    store.put(row(key="x", tags={"task": "short-name", "phase": 1}))
    df = store.df()
    assert df["task"].iloc[0] == "t/x"  # base column wins
    assert df["tag_task"].iloc[0] == "short-name"
    assert df["phase"].iloc[0] == 1


def test_delete_prefix(store):
    store.put(row(key="e1"))
    store.put(row(key="e1#t0", kind="trace"))
    store.put(row(key="e1#t1", kind="trace"))
    store.put(row(key="e2"))
    assert store.delete_prefix("e1#") == 2
    assert store.get("e1") is not None and store.get("e2") is not None
    assert store.get("e1#t0") is None
