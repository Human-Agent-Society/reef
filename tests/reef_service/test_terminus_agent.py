"""The terminus adapter against the real Harbor package.

Skipped unless the ``terminus`` extra is installed. These are the contract
checks no hermetic mapping can make: that Harbor accepts the agent spec and
the trial overrides the runner builds, and that the constructor arguments the
render quirk admits are real Terminus 2 parameters. They need Harbor, but not
Docker and not a model, so they run in any job that installs the extra.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from reef.harness.adapters import get_adapter
from reef.harness.adapters.terminus.quirks import _ALLOWED_KNOBS, _BINDING_KNOBS
from reef.harness.render import render_composition
from reef.harness.terminus import instruction_paths, skill_roots
from reef.harness.terminus.runner import AGENT_NAME, agent_spec

pytest.importorskip("harbor", reason="the terminus extra is not installed")

NODES = [
    ("rules", {"text": "Be brief."}),
    ("skill", {"name": "notes", "text": "# Notes\n\nTake notes."}),
    ("agent_command", {"name": "summarize", "text": "Summarize."}),
    ("config", {"data": {"model_name": "openai/stub", "max_turns": 12}}),
]


def _rendered(root: Path) -> dict[str, str]:
    files = render_composition(NODES, get_adapter("terminus"))
    for path, text in files.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return files


@pytest.mark.unit
def test_harbor_validates_the_trial_the_runner_builds(tmp_path: Path) -> None:
    from harbor.models.trial.config import TaskConfig, TrialConfig

    files = _rendered(tmp_path / "root")
    config = TrialConfig.model_validate(
        {
            "task": TaskConfig.model_validate({"path": Path("recipes/basic/harbor")}).model_dump(),
            "trials_dir": tmp_path / "trials",
            "agent": agent_spec(str(tmp_path / "root"), files),
            "extra_instruction_paths": instruction_paths(tmp_path / "root", files),
        }
    )
    # Harbor's own agent, configured rather than subclassed.
    assert config.agent.name == AGENT_NAME
    assert config.agent.model_name == "openai/stub"
    assert config.agent.kwargs == {"max_turns": 12}
    assert config.extra_instruction_paths == [tmp_path / "root" / "terminus/AGENTS.md"]


@pytest.mark.unit
def test_harbor_resolves_both_skill_roots_the_tree_renders(tmp_path: Path) -> None:
    from harbor.skills import resolve_skills

    files = _rendered(tmp_path / "root")
    roots = skill_roots(tmp_path / "root", files)
    resolved = resolve_skills([str(path) for path in roots])
    # One skill and one command, each discovered as its own skill directory,
    # so Harbor keeps progressive loading instead of pasting bodies inline.
    assert len(resolved) == 2


@pytest.mark.unit
def test_every_admitted_knob_is_a_real_terminus_2_argument() -> None:
    import inspect

    from harbor.agents.terminus_2.terminus_2 import Terminus2

    parameters = set(inspect.signature(Terminus2.__init__).parameters)
    unknown = sorted((_ALLOWED_KNOBS | _BINDING_KNOBS) - parameters)
    assert unknown == [], f"the quirk admits arguments Terminus 2 does not take: {unknown}"


def test_pinned_runner_reaches_real_lab_and_harbor_config_without_keyword_collision(tmp_path, monkeypatch):
    pytest.importorskip("reef_eval")
    from harbor.trial.trial import Trial
    from reef.harness.terminus.runner import run

    root = tmp_path / "root"
    _rendered(root)
    commit = "69671fbaac6d67a7ef0dfec016cc38a64ef7a77c"
    monkeypatch.setenv("REEF_TERMINUS_DIR", str(root))
    monkeypatch.setenv("REEF_TERMINUS_SESSION_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("REEF_TERMINUS_TASK_COMMIT", commit)
    monkeypatch.setenv("REEF_TERMINUS_ENVIRONMENT", "e2b")

    class ReachedTrial(Exception):
        pass

    async def create(config):
        assert str(config.task.path) == "extract-elf"
        assert config.task.git_commit_id == commit
        assert config.task.git_url == "https://github.com/laude-institute/terminal-bench-2.git"
        assert config.task.name is None
        assert config.agent.model_name == "openai/stub"
        assert config.environment.type.value == "e2b"
        raise ReachedTrial

    monkeypatch.setattr(Trial, "create", create)
    with pytest.raises(ReachedTrial):
        run("terminal-bench/extract-elf")
    # Collect reef-eval's synthetic failed future here, before later caplog assertions.
    import gc

    gc.collect()
