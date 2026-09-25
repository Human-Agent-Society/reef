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

import os
import sys
import tempfile
from pathlib import Path

import pytest

from reef.harness.adapters import get_adapter
from reef.harness.episodes.executor import EpisodeExecutor, ProcessOutcome, SandboxExecutor
from reef.harness.episodes.model_binding import ModelBinding
from reef.harness.episodes.run import EpisodeError, run_episode
from reef.harness.runners.terminus.runner import SESSION_DIR_ENV, TREE_DIR_ENV, TRIALS_DIR_ENV
from reef.harness.tree.render import render_composition

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
    "root": root,
    "docker": {{name: os.environ.get(name) for name in ("DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG")}},
    "tree": tree,
    "steps": [{{"step_id": 1, "source": "agent"}}],
}}))
"""

NODES = [
    ("rules", {"text": "Be brief."}),
    ("skill", {"name": "notes", "text": "# Notes\n\nTake notes."}),
    ("agent_command", {"name": "summarize", "text": "Summarize."}),
    ("config", {"data": {"max_turns": 12}}),
    # The model comes from Reef's binding: a tree cannot set model_name.
    *ModelBinding(base_url="http://127.0.0.1:9", model="openai/gpt-4o", api_key="k").compose_nodes(
        get_adapter("terminus")
    ),
]


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch) -> Path:
    """On macOS terminus roots its episodes under the home directory, so every test here gets a home of its own."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    for name in ("DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG"):
        monkeypatch.delenv(name, raising=False)
    return home


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
    # keeps only PATH, SYSTEMROOT, TMPDIR and the descriptor's host_env from
    # the parent.
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
@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_the_episode_root_is_under_the_home_directory_on_macos_only_and_removed(
    tmp_path: Path, home: Path, monkeypatch, platform: str
) -> None:
    # Harbor bind-mounts the trial directory under the root into the task
    # container. Docker on macOS runs in a VM, and colima shares the home
    # directory with it, not $TMPDIR; Linux Docker shares every path, so a
    # service user there needs no writable home.
    monkeypatch.setattr(sys, "platform", platform)
    descriptor = get_adapter("terminus")
    result = run_episode(descriptor, render_composition(NODES, descriptor), "t", binary=_stub(tmp_path), timeout=60.0)
    root = Path(result.trajectory[0]["root"])
    expected = home / ".reef" / "episodes" if platform == "darwin" else Path(tempfile.gettempdir())
    assert root.parent == expected
    assert not root.exists()
    assert (home / ".reef").exists() == (platform == "darwin")


@pytest.mark.unit
def test_the_episode_keeps_the_services_docker_settings(tmp_path: Path, home: Path, monkeypatch) -> None:
    # HOME is relocated, so without these the docker CLI finds neither
    # colima's context nor the compose plugin in the service's ~/.docker.
    descriptor = get_adapter("terminus")
    files = render_composition(NODES, descriptor)
    result = run_episode(descriptor, files, "t", binary=_stub(tmp_path), timeout=60.0)
    unset = {"DOCKER_HOST": None, "DOCKER_CONTEXT": None}
    assert result.trajectory[0]["docker"] == {**unset, "DOCKER_CONFIG": str(home / ".docker")}

    service = {
        "DOCKER_HOST": "unix:///var/run/docker.sock",
        "DOCKER_CONTEXT": "colima",
        "DOCKER_CONFIG": str(tmp_path / "docker-config"),
    }
    for name, value in service.items():
        monkeypatch.setenv(name, value)
    result = run_episode(descriptor, files, "t", binary=_stub(tmp_path), timeout=60.0)
    assert result.trajectory[0]["docker"] == service


@pytest.mark.unit
def test_an_executor_other_than_the_local_one_gets_no_service_settings(tmp_path: Path, monkeypatch) -> None:
    # A sandbox forwards only its env_from, and a remote executor runs no local container, on macOS too.
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv("DOCKER_HOST", "unix:///var/run/docker.sock")
    launched: list[tuple[Path, dict[str, str]]] = []

    class RecordingExecutor(EpisodeExecutor):
        def preflight(self) -> None:
            return None

        def launch(self, argv, *, root, workspace, env, timeout, writable_paths=(), readonly_paths=()):
            launched.append((root, dict(env)))
            return ProcessOutcome(exit_code=0, stdout="", stderr="")

    descriptor = get_adapter("terminus")
    run_episode(descriptor, render_composition(NODES, descriptor), "t", executor=RecordingExecutor())
    [(root, env)] = launched
    assert not {"DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG"} & set(env)
    assert root.parent == Path(tempfile.gettempdir())


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


@pytest.mark.unit
def test_code_extension_cannot_reach_a_local_process(tmp_path: Path) -> None:
    descriptor = get_adapter("terminus")
    files = render_composition(
        [*NODES, ("code_extension", {"name": "agent", "code": "class Agent: pass\n"})], descriptor
    )
    with pytest.raises(EpisodeError, match=r"code_extension requires evolution\.executor: sandbox"):
        run_episode(descriptor, files, "hello-world", binary="must-never-be-launched")


@pytest.mark.unit
@pytest.mark.parametrize("egress,key", [((), "dummy"), (("api.e2b.dev",), "")])
def test_remote_terminus_requires_explicit_network_and_credentials(egress, key) -> None:
    descriptor = get_adapter("terminus")
    executor = SandboxExecutor(egress_hosts=egress, env={"REEF_TERMINUS_ENVIRONMENT": "e2b", "E2B_API_KEY": key})
    with pytest.raises(EpisodeError, match="requires egress_hosts and E2B_API_KEY"):
        run_episode(descriptor, render_composition(NODES, descriptor), "task", executor=executor)
