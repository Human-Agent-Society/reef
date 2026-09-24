"""Built-in Reefine configuration and checkout-independent profile contracts."""

from __future__ import annotations

import copy
import os
import subprocess
import sys
from pathlib import Path

import pytest

from reef.harness.episodes.run import EpisodeResult
from reef.harness.runners.terminus.runner import trial_record
from reef.recipe.config_fields import recipe_config_fields
from reef.recipe.errors import RecipeConfigError
from reef.recipe.reefine import ReefineRecipe
from reef.recipe.reefine.evolution import HEALTH_TASK_DIRECTORY, evaluate
from reef.recipe.registry import build_recipe, recipe_class_for
from reef.service.deploy.orchestrator import _prepare_profile
from reef.service.profiles import profile_path
from reef.train.cordis_backend.backend import FloorPluginFactory, ScoreComparisonPluginFactory


def test_dotted_recipe_defaults_and_config_are_independent() -> None:
    config = {"evolution": {"tasks": ["[health] Run `echo reef-ok` and reply with its output."]}}
    original = copy.deepcopy(config)
    built = build_recipe("reef.recipe.reefine:ReefineRecipe", {}, config)
    assert recipe_class_for("reef.recipe.reefine:ReefineRecipe") is ReefineRecipe
    assert isinstance(built, ReefineRecipe)
    assert built.name == "reefine"
    assert built.training_mode == "manual"
    assert built.propose.reads_requests
    assert built.review_kinds == ("code_extension",)
    # The floor: the candidate alone must pass every evaluation task; the current release is not run.
    assert isinstance(built.candidate_plugin, FloorPluginFactory) and built.floor_score == 1.0
    assert built.candidate_plugin.floor_score == 1.0
    assert [entry["id"] for entry in built.seed] == [
        "reef-version-check",
        "reef-requests",
        "reef-pi-extension-api",
    ]
    assert config == original
    assert "training_mode" in recipe_config_fields(ReefineRecipe)


def test_evolution_overrides_and_training_fields_remain_available() -> None:
    built = ReefineRecipe.from_environment(
        {},
        config={
            "data": {"training_mode": "hybrid", "batch_size": 3},
            "evolution": {
                "tasks": ["[fib] Compute fib(90)."],
                "selection": "score_comparison",
                "requests": False,
                "version_check": False,
                "review_kinds": [],
                "evaluate": "reef.recipe.reefine.evolution:evaluate",
            },
        },
    )
    assert isinstance(built, ReefineRecipe)
    assert built.training_mode == "hybrid" and built.batch_size == 3
    assert built.seed == () and built.review_kinds == ()
    assert isinstance(built.candidate_plugin, ScoreComparisonPluginFactory)


@pytest.mark.parametrize("evolution", [None, [], {"tasks": []}, {"tasks": ["task"], "requests": "yes"}])
def test_invalid_evolution_config_is_rejected(evolution: object) -> None:
    with pytest.raises(RecipeConfigError):
        ReefineRecipe.from_environment({}, config={"evolution": evolution})


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
assert type(recipe.candidate_plugin).__name__ == 'FloorPluginFactory'
assert recipe.base_artifact_files()
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
    """reef-terminus plays a Harbor task directory, not a prompt: the profile's health prompt becomes the same check
    as the directory beside the recipe, and the bundled scorer reads that directory's verifier reward."""
    evolution = {
        "adapter": "terminus",
        "tasks": ["[health] Run `echo reef-ok` and reply with its output.", "/tasks/own-task"],
        "requests": False,
        "version_check": False,
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

    def played(rewards: dict[str, float], error: str = "") -> EpisodeResult:
        row = trial_record(HEALTH_TASK_DIRECTORY, rewards, tmp_path, error)
        events = ({"type": "verifier", **{key: value for key, value in row.items() if key != "steps"}},)
        return EpisodeResult(exit_code=0, stdout="", stderr="", trajectory=events, residue=())

    assert evaluate(HEALTH_TASK_DIRECTORY, played({"reward": 1.0})) == 1.0
    assert evaluate(HEALTH_TASK_DIRECTORY, played({"reward": 0.0})) == 0.0
    assert evaluate(HEALTH_TASK_DIRECTORY, played({}, "the container never built")) == 0.0
    # The reply's last line is not the score of a task directory: a Harbor episode has no assistant reply.
    assert evaluate("[health] x", played({"reward": 1.0})) == 0.0


def test_an_unknown_adapter_is_a_config_error() -> None:
    with pytest.raises(RecipeConfigError, match="acme"):
        ReefineRecipe.from_environment({}, config={"evolution": {"adapter": "acme", "tasks": ["[health] x"]}})
