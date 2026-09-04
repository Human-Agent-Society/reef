"""The terminus adapter through ``run_episode``, the path Reef actually uses.

The gap these close: every earlier terminus test drove the runner directly
with a hand-built environment, so none of them noticed that ``run_episode``
hands an episode only the descriptor's own env. An adapter can pass its unit
tests and still be unable to start.

A stub binary stands in for ``reef-terminus``: it records the environment and
working directory it was launched with and writes a trial file where the
descriptor says the reader will look. That exercises the descriptor, the
render, the executor, the trajectory reader, and the residue check together,
with no Harbor and no Docker.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from reef.harness.adapters import get_adapter
from reef.harness.episode import EpisodeError, run_episode
from reef.harness.executor import SandboxExecutor
from reef.harness.render import render_composition
from reef.harness.terminus.runner import SESSION_DIR_ENV, TREE_DIR_ENV, TRIALS_DIR_ENV

# Stands in for the runner: prove the episode reaches it with what it needs.
STUB = """#!/usr/bin/env python3
import json, os, sys
from pathlib import Path

root = os.environ["{tree}"]
sessions = Path(os.environ["{sessions}"])
Path(os.environ["{trials}"]).mkdir(parents=True, exist_ok=True)
task = sys.argv[sys.argv.index("--task") + 1]

# What the runner reads back: the tree, from the episode root.
tree = sorted(
    p.relative_to(root).as_posix()
    for p in Path(root).rglob("*")
    if p.is_file() and p.relative_to(root).as_posix().startswith(("terminus/", "terminus-commands/"))
)
sessions.mkdir(parents=True, exist_ok=True)
(sessions / (task + ".json")).write_text(json.dumps({{
    "task": task,
    "rewards": {{"accuracy": 1.0}},
    "reward": 1.0,
    "failed": False,
    "error": "",
    "cwd": os.getcwd(),
    "tree": tree,
    "steps": [{{"step_id": 1, "source": "agent"}}],
}}))
"""

NODES = [
    ("rules", {"text": "Be brief."}),
    ("skill", {"name": "notes", "text": "# Notes\n\nTake notes."}),
    ("agent_command", {"name": "summarize", "text": "Summarize."}),
    ("config", {"data": {"model_name": "openai/gpt-4o", "max_turns": 12}}),
]


def _stub(tmp_path: Path) -> str:
    binary = tmp_path / "reef-terminus-stub"
    binary.write_text(STUB.format(tree=TREE_DIR_ENV, sessions=SESSION_DIR_ENV, trials=TRIALS_DIR_ENV))
    binary.chmod(0o755)
    return str(binary)


@pytest.mark.unit
def test_an_episode_reaches_the_runner_with_the_tree_and_leaves_no_residue(tmp_path: Path) -> None:
    descriptor = get_adapter("terminus")
    files = render_composition(NODES, descriptor)
    result = run_episode(descriptor, files, "hello-world", binary=_stub(tmp_path), timeout=60.0)

    assert result.exit_code == 0, result.stderr
    # The trajectory reader found the trial the runner wrote.
    events = result.trajectory
    assert [event["type"] for event in events] == ["verifier", "step"]
    assert events[0]["reward"] == 1.0
    # Both skill roots reached the runner: terminus-commands is a sibling of
    # terminus, so a tree read one level down would have lost the command.
    assert events[0]["tree"] == [
        "terminus-commands/summarize/SKILL.md",
        "terminus/AGENTS.md",
        "terminus/config.json",
        "terminus/skills/notes/SKILL.md",
    ]
    # Harbor's trial tree and the session file are episode state, not drift.
    assert result.residue == ()


@pytest.mark.unit
def test_the_episode_runs_in_the_workspace(tmp_path: Path) -> None:
    descriptor = get_adapter("terminus")
    files = render_composition(NODES, descriptor)
    result = run_episode(descriptor, files, "hello-world", binary=_stub(tmp_path), timeout=60.0)
    assert Path(result.trajectory[0]["cwd"]).name == "workspace"


@pytest.mark.unit
def test_the_episode_carries_no_host_environment_beyond_the_descriptor(tmp_path: Path) -> None:
    # The reason the dataset location cannot be a host variable: run_episode
    # keeps only PATH, SYSTEMROOT and TMPDIR from the parent.
    descriptor = get_adapter("terminus")
    leaked = tmp_path / "leaked.txt"
    binary = tmp_path / "probe"
    binary.write_text(
        f"#!/usr/bin/env python3\nimport os\nopen({str(leaked)!r}, 'w').write(os.environ.get('SECRET', ''))\n"
    )
    binary.chmod(0o755)
    os.environ["SECRET"] = "must-not-reach-the-episode"
    try:
        run_episode(descriptor, render_composition(NODES, descriptor), "t", binary=str(binary), timeout=60.0)
    finally:
        os.environ.pop("SECRET", None)
    assert leaked.read_text() == ""


@pytest.mark.unit
def test_a_named_host_variable_is_inherited_and_an_unnamed_one_is_not(tmp_path: Path) -> None:
    # Episodes are hermetic by default, which is right for a harness that
    # needs nothing from the host. terminus creates sandboxes, so it names the
    # provider credential; everything else still stays out.
    descriptor = get_adapter("terminus")
    assert descriptor.inherit_env == ("E2B_API_KEY",)
    seen = tmp_path / "seen.json"
    binary = tmp_path / "probe"
    binary.write_text(
        "#!/usr/bin/env python3\nimport json, os\n"
        f"open({str(seen)!r}, 'w').write(json.dumps("
        "{'named': os.environ.get('E2B_API_KEY'), 'unnamed': os.environ.get('SECRET')}))\n"
    )
    binary.chmod(0o755)
    os.environ["E2B_API_KEY"] = "sandbox-credential"
    os.environ["SECRET"] = "must-not-reach-the-episode"
    try:
        run_episode(descriptor, render_composition(NODES, descriptor), "t", binary=str(binary), timeout=60.0)
    finally:
        os.environ.pop("E2B_API_KEY", None)
        os.environ.pop("SECRET", None)
    assert json.loads(seen.read_text()) == {"named": "sandbox-credential", "unnamed": None}


@pytest.mark.unit
def test_an_unset_named_variable_is_simply_absent(tmp_path: Path) -> None:
    descriptor = get_adapter("terminus")
    seen = tmp_path / "seen.json"
    binary = tmp_path / "probe"
    binary.write_text(
        "#!/usr/bin/env python3\nimport json, os\n"
        f"open({str(seen)!r}, 'w').write(json.dumps({{'named': os.environ.get('E2B_API_KEY')}}))\n"
    )
    binary.chmod(0o755)
    os.environ.pop("E2B_API_KEY", None)
    run_episode(descriptor, render_composition(NODES, descriptor), "t", binary=str(binary), timeout=60.0)
    assert json.loads(seen.read_text()) == {"named": None}


@pytest.mark.unit
def test_a_sandboxed_deployment_is_refused_at_the_shared_boundary(tmp_path: Path) -> None:
    # terminus isolates episodes in Harbor's container, which cannot nest in
    # bubblewrap. Every caller of run_episode is told, not just one backend.
    descriptor = get_adapter("terminus")
    with pytest.raises(EpisodeError, match=r"cannot run under evolution\.executor: sandbox"):
        run_episode(
            descriptor,
            render_composition(NODES, descriptor),
            "hello-world",
            binary=sys.executable,
            timeout=60.0,
            executor=SandboxExecutor(),
        )
