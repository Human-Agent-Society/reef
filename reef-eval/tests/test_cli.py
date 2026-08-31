from pathlib import Path

import pytest

from reef_eval.cli import main, resolve_targets

TASKS_ROOT = Path(__file__).parent.parent / "tasks"


def test_resolve_single_task_dir():
    target = str(TASKS_ROOT / "autoresearch" / "first-party" / "tsp-tour")
    assert resolve_targets([target], TASKS_ROOT) == [target]


def test_resolve_category_expands_and_skips_template():
    resolved = resolve_targets(["autoresearch/first-party"], TASKS_ROOT)
    names = [Path(r).name for r in resolved]
    assert "tsp-tour" in names and "circle-packing" in names
    assert len(names) == 6
    assert not any("_template" in r for r in resolved)


def test_resolve_under_a_dot_directory_root(tmp_path):
    """Benchmarks download to ~/.cache/reef-eval, so a tasks root under a
    dot-directory must still resolve. Only the parts below the root decide
    whether a task is skipped."""
    root = tmp_path / ".cache" / "reef-eval" / "tasks" / "main" / "edgebench"
    (root / "real-task").mkdir(parents=True)
    (root / "real-task" / "task.toml").write_text('name = "real-task"\n')
    (root / "_template").mkdir()
    (root / "_template" / "task.toml").write_text('name = "template"\n')

    resolved = resolve_targets([str(root)], None)
    assert [Path(r).name for r in resolved] == ["real-task"]


def test_resolve_registry_id_passthrough():
    assert resolve_targets(["some-benchmark/some-task"], TASKS_ROOT) == [
        "some-benchmark/some-task"
    ]


def test_resolve_unknown_bare_name_is_loud(tmp_path):
    """Harbor registry ids are always org/name, so a bare word that matched
    nothing local cannot be one. Say so instead of letting Harbor fail on a
    schema error."""
    with pytest.raises(SystemExit, match="not a task directory"):
        resolve_targets(["definitely-not-a-real-task"], TASKS_ROOT)


def test_resolve_empty_folder_is_loud(tmp_path):
    (tmp_path / "emptybench").mkdir()
    with pytest.raises(SystemExit, match="no task.toml"):
        resolve_targets(["emptybench"], tmp_path)


def test_run_fake_end_to_end(tmp_path, capsys):
    lab = str(tmp_path / "lab")
    code = main(
        [
            "--tasks-dir",
            str(TASKS_ROOT),
            "run",
            "autoresearch/first-party/tsp-tour",
            "autoresearch/first-party/bin-packing",
            "--agent",
            "oracle",
            "--fake",
            "--lab",
            lab,
            "-n",
            "2",
            "--tag",
            "suite=smoke",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "OK  tsp-tour" in out and "OK  bin-packing" in out

    code = main(["report", "--lab", lab])
    out = capsys.readouterr().out
    assert code == 0
    assert "first-party/tsp-tour" in out
    assert " 2 " in out  # count column: 2 attempts each


def test_run_is_idempotent_across_invocations(tmp_path, capsys):
    lab = str(tmp_path / "lab")
    argv = [
        "--tasks-dir",
        str(TASKS_ROOT),
        "run",
        "autoresearch/first-party/tsp-tour",
        "--agent",
        "oracle",
        "--fake",
        "--lab",
        lab,
    ]
    assert main(argv) == 0
    assert main(argv) == 0  # second run resumes (skips) rather than duplicating
    capsys.readouterr()
    main(["report", "--lab", lab])
    out = capsys.readouterr().out
    assert " 1 " in out  # still one row


def test_stream_fake_end_to_end(tmp_path, capsys):
    lab = str(tmp_path / "lab")
    argv = [
        "--tasks-dir",
        str(TASKS_ROOT),
        "stream",
        "autoresearch/first-party/tsp-tour",
        "autoresearch/first-party/bin-packing",
        "--name",
        "week1",
        "--agent",
        "oracle",
        "--fake",
        "--lab",
        lab,
    ]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "stream 'week1'" in out
    assert "[  0] tsp-tour" in out and "[  1] bin-packing" in out

    assert main(argv) == 0  # re-running a stream resumes (skips) it
    from reef_eval import Lab

    df = Lab(lab).df("episode")
    assert len(df) == 2
    assert df["stream"].tolist() == ["week1", "week1"]
    assert sorted(df["position"].tolist()) == [0, 1]


def test_stream_name_defaults_to_the_targets(tmp_path, capsys):
    """`reef-eval stream` takes targets positionally, like `reef-eval run`. Without
    --name the label comes from what was asked for, so no target can be
    silently swallowed as a name."""
    lab = str(tmp_path / "lab")
    assert (
        main(
            [
                "--tasks-dir",
                str(TASKS_ROOT),
                "stream",
                "autoresearch/first-party/tsp-tour",
                "autoresearch/first-party/bin-packing",
                "--agent",
                "oracle",
                "--fake",
                "--lab",
                lab,
            ]
        )
        == 0
    )
    from reef_eval import Lab

    df = Lab(lab).df("episode")
    assert len(df) == 2  # both targets ran; neither became the name
    assert set(df["stream"]) == {"tsp-tour+1more"}


@pytest.mark.parametrize("bad", ["my/stream", "my stream", "  "])
def test_stream_name_that_would_break_the_state_dir_is_rejected(tmp_path, bad):
    with pytest.raises(SystemExit, match="name"):
        main(
            [
                "--tasks-dir",
                str(TASKS_ROOT),
                "stream",
                "autoresearch/first-party/tsp-tour",
                "--name",
                bad,
                "--agent",
                "oracle",
                "--fake",
                "--lab",
                str(tmp_path / "lab"),
            ]
        )


def test_stream_shuffle_is_deterministic_and_seed_scoped(tmp_path, capsys):
    lab = str(tmp_path / "lab")
    base = [
        "--tasks-dir",
        str(TASKS_ROOT),
        "stream",
        "autoresearch/first-party",
        "--name",
        "mix",
        "--agent",
        "oracle",
        "--fake",
        "--lab",
        lab,
    ]
    from reef_eval import Lab

    assert main([*base, "--shuffle", "7"]) == 0
    assert main([*base, "--shuffle", "7"]) == 0  # same seed = same stream: resumes
    df = Lab(lab).df("episode")
    assert len(df) == 6  # nothing re-ran
    assert set(df["shuffle_seed"]) == {7}

    assert main([*base, "--shuffle", "8"]) == 0  # new seed = its own stream
    df = Lab(lab).df("episode")
    assert len(df) == 12
    by_seed = {
        seed: df[df.shuffle_seed == seed].sort_values("position")["task"].tolist()
        for seed in (7, 8)
    }
    assert sorted(by_seed[7]) == sorted(by_seed[8])  # task set fixed...
    assert by_seed[7] != by_seed[8]  # ...order differs


def test_list_shows_downloaded_benchmarks(tmp_path, capsys, monkeypatch):
    """A pip install has no tasks/ folder, so `reef-eval list` has to look in the
    download cache too, or it reports nothing right after `reef-eval fetch`."""
    cache = tmp_path / ".cache" / "reef-eval"
    bench = cache / "tasks" / "v9.9.9" / "demo-bench" / "a-task"
    bench.mkdir(parents=True)
    (bench / "task.toml").write_text('name = "a-task"\n')
    monkeypatch.setenv("REEF_EVAL_CACHE", str(cache))

    assert main(["--tasks-dir", str(tmp_path / "no-such-tasks"), "list"]) == 0
    out = capsys.readouterr().out
    assert "demo-bench (1 tasks)" in out
    assert "reef-eval run demo-bench/<task>" in out


def test_list_with_nothing_available_points_at_fetch(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("REEF_EVAL_CACHE", str(tmp_path / "empty-cache"))
    assert main(["--tasks-dir", str(tmp_path / "no-such-tasks"), "list"]) == 1
    out = capsys.readouterr().out
    assert "reef-eval fetch" in out and "cl-bench" in out


def test_list_and_fetch_errors(capsys):
    assert main(["--tasks-dir", str(TASKS_ROOT), "list"]) == 0
    out = capsys.readouterr().out
    assert "autoresearch/first-party/circle-packing" in out
    with pytest.raises(SystemExit, match="known"):
        main(["--tasks-dir", str(TASKS_ROOT), "fetch", "nonsense-bench"])


# ------------------------------------------- the agent/--local flag contract


@pytest.mark.parametrize("command", ["run", "stream"])
def test_local_without_a_command_says_what_is_missing(command):
    """--local has no agent to run: the command is the whole point of it."""
    with pytest.raises(SystemExit, match="--command"):
        main([command, "autoresearch/first-party/tsp-tour", "--local"])


@pytest.mark.parametrize("command", ["run", "stream"])
def test_a_command_without_local_is_rejected(command):
    """A container run ignores --command, and silently ignoring it would
    leave the user waiting on a harness that never got their command."""
    with pytest.raises(SystemExit, match="only works with --local"):
        main([command, "autoresearch/first-party/tsp-tour", "--command", "true"])


@pytest.mark.parametrize("command", ["run", "stream"])
def test_no_agent_and_no_local_is_rejected(command):
    with pytest.raises(SystemExit, match="--agent is required"):
        main([command, "autoresearch/first-party/tsp-tour"])


def test_a_failed_episode_makes_the_run_exit_nonzero(tmp_path, capsys):
    """An episode with no reward has to reach the shell as a failure, or a
    CI job that scored nothing looks the same as one that scored well."""
    lab = str(tmp_path / "lab")
    code = main(
        [
            "--tasks-dir",
            str(TASKS_ROOT),
            "run",
            # A stream task has no judge to run locally, so the local
            # executor refuses it and the episode records an error.
            "continual-learning/terminal-bench/chess-best-move",
            "--local",
            "--command",
            "true",
            "--lab",
            lab,
        ]
    )
    assert code == 1
    assert "ERR" in capsys.readouterr().out
