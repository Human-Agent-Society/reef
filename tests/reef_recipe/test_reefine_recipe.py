"""Built-in Reefine configuration and checkout-independent profile contracts."""

from __future__ import annotations

import copy
import os
import subprocess
import sys
from pathlib import Path

import pytest

from reef.core.config import config_arguments
from reef.core.plugins import PluginReference
from reef.evaluation.activation import AlwaysActivate, NeverActivate
from reef.evaluation.harness import FloorPlugin, ScoreComparisonPlugin
from reef.harness.rollouts.run import RolloutResult
from reef.harness.runners.terminus.runner import trial_record
from reef.harness.tree import HarnessTree, TreeKind
from reef.recipe.errors import RecipeConfigError
from reef.recipe.reefine import AgentProposer, ReefineRecipe
from reef.recipe.reefine.recipe import SEED_DIRECTORY
from reef.recipe.reefine.scorer import HEALTH_TASK_DIRECTORY, AnswerScorer
from reef.recipe.registry import build_recipe, recipe_class_for
from reef.service.deploy.orchestrator import _prepare_profile
from reef.service.profiles import profile_path


def test_dotted_recipe_defaults_and_config_are_independent() -> None:
    config = {"evolution": {"tasks": ["[health] Run `echo reef-ok` and reply with its output."]}}
    original = copy.deepcopy(config)
    built = build_recipe("reef.recipe.reefine:ReefineRecipe", {}, config)
    assert recipe_class_for("reef.recipe.reefine:ReefineRecipe") is ReefineRecipe
    assert isinstance(built, ReefineRecipe)
    assert built.name == "reefine"
    assert built.training_mode == "manual"
    assert isinstance(built.propose, AgentProposer) and built.propose.reads_requests
    # By default a person activates a tree that changes an extension.
    rule = built.activation_rule()
    assert isinstance(rule, AlwaysActivate) and rule.review_kinds == {TreeKind.CODE_EXTENSION}
    # The floor: the candidate alone must pass every evaluation task; the current release is not run.
    assert built.selection == PluginReference(FloorPlugin) and built.floor_score == 1.0
    # Without evolution.seed the first tree is the packaged one: the starter skill.
    assert built.seed == HarnessTree.read(SEED_DIRECTORY)
    assert list(built.seed.files) == ["skills/answer-style/SKILL.md"]
    assert config == original
    assert "training_mode" in {argument.name for argument in config_arguments(ReefineRecipe)}


def test_evolution_overrides_and_training_fields_remain_available(tmp_path: Path) -> None:
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "rules.md").write_text("Answer in one line.\n")
    built = ReefineRecipe.from_environment(
        {},
        config={
            "data": {"training_mode": "hybrid"},
            "pipeline": {"sample2job": {"config": {"batch_size": 3}}},
            "evolution": {
                "tasks": ["[fib] Compute fib(90)."],
                "selection": "score_comparison",
                "seed": str(seed),
                "evaluate": "reef.recipe.reefine.scorer:AnswerScorer",
            },
            "activation": {"module": "never"},
        },
    )
    assert isinstance(built, ReefineRecipe)
    assert built.training_mode == "hybrid" and built.stages().sample2job.config == {"batch_size": 3}
    assert built.seed == HarnessTree({"rules.md": "Answer in one line.\n"})
    assert isinstance(built.activation_rule(), NeverActivate)
    assert built.selection == PluginReference(ScoreComparisonPlugin)


@pytest.mark.parametrize(
    "evolution",
    [None, [], {"tasks": []}, {"tasks": ["task"], "seed": 3}, {"tasks": ["task"], "seed": "/no/such/seed"}],
)
def test_invalid_evolution_config_is_rejected(evolution: object) -> None:
    with pytest.raises(RecipeConfigError):
        ReefineRecipe.from_environment({}, config={"evolution": evolution})


def test_a_seed_that_fails_validation_is_refused_at_startup(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("not a harness file")
    with pytest.raises(RecipeConfigError, match=r"evolution\.seed fails validation: notes\.txt"):
        ReefineRecipe.from_environment({}, config={"evolution": {"tasks": ["task"], "seed": str(tmp_path)}})


def test_profile_preparation_needs_no_checkout() -> None:
    environ: dict[str, str] = {}
    _prepare_profile("reefine", "ollama/gemma4:26b", environ)
    assert Path(environ["REEF_RECIPE_CONFIG_DIR"]) == profile_path("reefine").parent
    assert "PYTHONPATH" not in environ


def test_profile_builds_without_tutorial_or_training_imports(tmp_path: Path) -> None:
    # Run away from the repository, refuse source-tree method imports, and build
    # the packaged preset with the same upstream binding the service supplies.
    code = """
import importlib.abc
import os
import sys

class RejectSourceMethods(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'tutorials', 'recipes', 'harness', 'torch', 'slime'}:
            raise ImportError(f'unexpected dependency: {fullname}')

sys.meta_path.insert(0, RejectSourceMethods())
from reef.recipe.reefine import ReefineRecipe
from reef.recipe.registry import build_named_recipe
from reef.service.assembly import _upstream_runtime
from reef.service.deploy.config_utils import load_config
from reef.service.deploy.deployment_config import translate_layout
from reef.service.deploy.service_config import service_config_from_mapping
from reef.service.profiles import profile_path

path = profile_path('reefine')
config = translate_layout(load_config(path))
service = service_config_from_mapping(config)
recipe = build_named_recipe('reefine', os.environ, config_directory=path.parent,
                            default_runtime=_upstream_runtime(service))
assert isinstance(recipe, ReefineRecipe)
assert recipe.training_mode == 'manual' and recipe.propose.reads_requests
assert len(recipe.tasks) == 1 and recipe.tasks[0].startswith('[health] ')
assert recipe.selection.plugin_class.__name__ == 'FloorPlugin'
assert recipe.base_artifact_source().files
assert recipe.model_binding().model == 'test-model'
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=tmp_path,
        env={
            **os.environ,
            "REEF_UPSTREAM_URL": "http://127.0.0.1:8000",
            "REEF_UPSTREAM_MODEL": "test-model",
            "REEF_UPSTREAM_API_KEY": "dummy",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_on_terminus_the_health_task_runs_as_the_shipped_task_directory(tmp_path: Path) -> None:
    """reef-terminus runs a Harbor task directory, not a prompt: the profile's health prompt becomes the same check
    as the directory beside the recipe, and the bundled scorer reads that directory's verifier reward."""
    evolution = {
        "adapter": "terminus",
        "tasks": ["[health] Run `echo reef-ok` and reply with its output.", "/tasks/own-task"],
    }
    built = ReefineRecipe.from_environment({}, config={"evolution": evolution})
    assert isinstance(built, ReefineRecipe)
    assert built.tasks == (HEALTH_TASK_DIRECTORY, "/tasks/own-task")
    health = Path(HEALTH_TASK_DIRECTORY)
    for name in ("instruction.md", "task.toml", "tests/test.sh", "environment/Dockerfile"):
        assert (health / name).is_file(), name
    assert "reef-ok" in (health / "tests/test.sh").read_text()
    # A prompt adapter keeps the prompt.
    pi = ReefineRecipe.from_environment({}, config={"evolution": {**evolution, "adapter": "pi"}})
    assert isinstance(pi, ReefineRecipe) and pi.tasks[0].startswith("[health] ")

    def ran(rewards: dict[str, float], error: str = "") -> RolloutResult:
        row = trial_record(HEALTH_TASK_DIRECTORY, rewards, tmp_path, error)
        events = ({"type": "verifier", **{key: value for key, value in row.items() if key != "steps"}},)
        return RolloutResult(exit_code=0, stdout="", stderr="", trajectory=events, residue=())

    scorer = AnswerScorer()
    assert scorer.score(HEALTH_TASK_DIRECTORY, ran({"reward": 1.0})) == 1.0
    assert scorer.score(HEALTH_TASK_DIRECTORY, ran({"reward": 0.0})) == 0.0
    assert scorer.score(HEALTH_TASK_DIRECTORY, ran({}, "the container never built")) == 0.0
    # The reply's last line is not the score of a task directory: a Harbor rollout has no assistant reply.
    assert scorer.score("[health] x", ran({"reward": 1.0})) == 0.0


def test_an_unknown_adapter_is_a_config_error() -> None:
    with pytest.raises(RecipeConfigError, match="acme"):
        ReefineRecipe.from_environment({}, config={"evolution": {"adapter": "acme", "tasks": ["[health] x"]}})


def test_off_pi_no_agent_proposer_is_built_or_warned_about(caplog) -> None:
    """The agent proposer answers requests on pi alone: another adapter builds no agent executor, so an unjailed
    agent setting neither warns nor needs a sandbox there, while pi still warns."""
    config = {"evolution": {"adapter": "dsh", "tasks": ["[health] x"]}}
    with caplog.at_level("WARNING"):
        built = ReefineRecipe.from_environment({"REEF_PROPOSER_SANDBOX": "none"}, config=config)
    assert isinstance(built, ReefineRecipe) and built.agent_executor is None
    assert "sandbox is none" not in caplog.text
    with caplog.at_level("WARNING"):
        pi = ReefineRecipe.from_environment({"REEF_PROPOSER_SANDBOX": "none"}, config={"evolution": {"tasks": ["x"]}})
    assert isinstance(pi, ReefineRecipe) and pi.agent_executor is not None
    assert "sandbox is none" in caplog.text


@pytest.mark.parametrize("adapter", ["pi", "claude", "codex", "opencode", "hermes", "dsh", "terminus"])
def test_the_packaged_seed_is_the_first_tree_on_every_adapter(adapter: str) -> None:
    """The recipe validates its seed for the adapter at startup: the starter skill renders everywhere."""
    built = ReefineRecipe.from_environment({}, config={"evolution": {"adapter": adapter, "tasks": ["[health] x"]}})
    assert isinstance(built, ReefineRecipe) and built.adapter == adapter
    assert built.seed == HarnessTree.read(SEED_DIRECTORY)
