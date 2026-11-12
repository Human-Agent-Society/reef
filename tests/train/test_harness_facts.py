"""The harness backend's facts: what it states from its settings without a method or an executor, and that the
component agrees."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from tests.support.harness_jobs import PI_SEED
from tests.support.strategies import resolve_method

from reef.artifact.artifact import Artifact
from reef.core.errors import ConfigError
from reef.core.snapshot import RootSource
from reef.evaluation.harness import EvaluationSettings
from reef.execution.sandbox.config import build_executor
from reef.harness.tasks.harness_models import HarnessModels
from reef.harness.tree import HarnessTree
from reef.train.component import ScenarioRoot
from reef.train.harness_backend.component import HarnessTrainingComponent
from reef.train.harness_backend.facts import HarnessFacts
from reef.train.harness_backend.settings import EvolutionSettings, HarnessSettings

ITEM = RootSource("item", item="upload-1", uri="file:///upload/", version="")


def settings_of(tmp_path: Path, adapter: str, seed: HarnessTree) -> HarnessSettings:
    return HarnessSettings(
        EvolutionSettings(adapter=adapter, state_dir=str(tmp_path / "harness")),
        EvaluationSettings(policy="none"),
        seed=seed,
    )


def tree_root(directory: Path, files: dict[str, str]) -> ScenarioRoot:
    """A root whose files are ``files``, in ``directory``."""
    for relative, text in files.items():
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return ScenarioRoot(Artifact.local(directory), ITEM)


def refusal(check, root: ScenarioRoot) -> str | None:
    """The message with which ``check`` refuses ``root``, or None when it accepts it."""
    try:
        check(root)
    except ValueError as exc:
        return str(exc)
    return None


@pytest.mark.unit
@pytest.mark.parametrize(("adapter", "seed"), [("pi", PI_SEED), ("codex", HarnessTree({}))])
def test_the_facts_of_the_settings_answer_as_the_component_of_the_same_settings(tmp_path, adapter, seed) -> None:
    settings = settings_of(tmp_path, adapter, seed)
    facts = HarnessFacts.from_settings(settings)
    assert (facts.name, facts.artifact_format, facts.descriptor.name) == ("harness", None, adapter)
    assert facts.default_start_inputs == {"seed": seed.checksum()}
    component = HarnessTrainingComponent(
        settings,
        HarnessModels(None, adapter),
        method=resolve_method(lambda workspace, services: None),
        executor=build_executor(None),
    )
    try:
        assert component.facts == facts
        assert (facts.name, facts.artifact_format, facts.shares_scenarios, facts.concurrent_jobs) == (
            component.name,
            component.artifact_format,
            component.shares_scenarios,
            component.concurrent_jobs,
        )
        assert facts.default_start_inputs == component.default_start_inputs == facts.default_start().inputs
        assert facts.default_start().files == dict(seed.files)
        for kind in ("trajectory", "task"):
            assert facts.parameters_schema(kind) == component.parameters_schema(kind)
        roots = [
            tree_root(tmp_path / "pi-seed", dict(PI_SEED.files)),
            tree_root(tmp_path / "unknown", {"not-a-part/file.txt": "x\n"}),
        ]
        answers = [(refusal(facts.check_root_start, root), refusal(component.root_start, root)) for root in roots]
        assert all(mine == theirs for mine, theirs in answers)
        # The pi seed is a pi tree; a file outside every part of the layout is no tree of either adapter.
        assert [mine is None for mine, _ in answers] == [adapter == "pi", False]
    finally:
        component.close()


@pytest.mark.unit
def test_settings_that_name_no_adapter_state_no_facts(tmp_path) -> None:
    with pytest.raises(ConfigError, match="no-such-adapter"):
        HarnessFacts.from_settings(settings_of(tmp_path, "no-such-adapter", PI_SEED))


def test_the_facts_need_no_gpu_dependency(tmp_path) -> None:
    code = (
        "import sys; "
        "sys.modules.update(dict.fromkeys(('ray', 'torch', 'slime', 'sglang', 'megatron', 'tinker'))); "
        "from reef.evaluation.harness import EvaluationSettings; "
        "from reef.train.harness_backend.facts import HarnessFacts; "
        "from reef.train.harness_backend.settings import EvolutionSettings, HarnessSettings; "
        f"settings = HarnessSettings(EvolutionSettings(state_dir={str(tmp_path)!r}), EvaluationSettings(policy='none')); "
        "assert HarnessFacts.from_settings(settings).name == 'harness'"
    )
    subprocess.run([sys.executable, "-c", code], check=True, text=True, capture_output=True)
