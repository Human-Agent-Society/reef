"""``reef serve --recipe <name> --model <provider>/<model>``: a recipe's profile starts without a config file."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from reef.recipe.reefine import ReefineRecipe
from reef.service.deploy.cli import build_parser, build_serve_parser
from reef.service.deploy.config_utils import DeployConfigError, load_config
from reef.service.deploy.execution import validate_services
from reef.service.deploy.orchestrator import _model_overrides, _prepare_profile, _resolve_config
from reef.service.deploy.service_config import service_config_from_mapping
from reef.service.profiles import PROFILES_DIR, UnknownProfileError, profile_names, profile_path
from reef.storage.sqlite import SQLiteRecordStore

REPO_ROOT = Path(__file__).resolve().parents[2]
#: What the launcher prints when the folded profile is selected by its former name.
ALIAS_NOTICE = "reef: --recipe harness-evolve is now reefine; starting the reefine profile"


@pytest.mark.unit
def test_builtin_recipe_profiles() -> None:
    assert profile_names() == ("reefine",)
    assert profile_path("reefine") == PROFILES_DIR / "reefine.yaml"
    # The folded harness-evolve profile keeps its name as an alias, not as a profile of its own.
    assert profile_path("harness-evolve") == PROFILES_DIR / "reefine.yaml"
    with pytest.raises(UnknownProfileError) as error:
        profile_path("weights")
    assert str(error.value) == (
        "no profile for recipe 'weights'; recipes with a profile: reefine (--recipe harness-evolve starts reefine)"
    )
    with pytest.raises(UnknownProfileError):
        profile_path("../reefine")


@pytest.mark.unit
def test_the_serve_parser_takes_recipe_and_model_and_the_service_parser_still_does_not() -> None:
    args, extras = build_serve_parser().parse_known_args(
        ["--recipe", "reefine", "--model", "ollama/gemma4:26b", "--port", "8901"]
    )
    assert (args.config, args.recipe, args.model, args.port, extras) == (
        None,
        "reefine",
        "ollama/gemma4:26b",
        8901,
        [],
    )
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--recipe", "reefine"])


@pytest.mark.unit
def test_only_an_explicit_config_or_profile_selects_a_file(tmp_path, monkeypatch, capsys) -> None:
    """An environment variable or nearby YAML must not silently select a deployment."""
    profile = str(PROFILES_DIR / "reefine.yaml")
    monkeypatch.setenv("REEF_CONFIG", "env.yaml")
    assert _resolve_config("mine.yaml", None) == "mine.yaml"
    assert _resolve_config(None, "reefine") == profile
    assert capsys.readouterr().err == ""
    # The former name selects the same file and says so once, on stderr.
    assert _resolve_config(None, "harness-evolve") == profile
    assert capsys.readouterr().err == ALIAS_NOTICE + "\n"
    assert _resolve_config(None, None) is None
    with pytest.raises(DeployConfigError, match="not both"):
        _resolve_config("mine.yaml", "reefine")
    with pytest.raises(DeployConfigError, match="recipes with a profile: reefine"):
        _resolve_config(None, "weights")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "reef.yaml").write_text("reef: {}\n")
    assert _resolve_config(None, None) is None


@pytest.mark.unit
def test_the_model_flag_fills_the_provider_preset_and_leaves_other_spellings_alone() -> None:
    assert _model_overrides("ollama/gemma4:26b", {}) == {
        "upstream_url": "http://127.0.0.1:11434",
        "upstream_model": "gemma4:26b",
        "upstream_api_key": "ollama",
    }
    assert _model_overrides("openai/gpt-5.6", {"REEF_UPSTREAM_API_KEY": "sk-1"}) == {
        "upstream_url": "https://api.openai.com",
        "upstream_model": "gpt-5.6",
        "upstream_api_key": "sk-1",
    }
    with pytest.raises(DeployConfigError, match="set REEF_UPSTREAM_API_KEY to the openai key"):
        _model_overrides("openai/gpt-5.6", {})
    # A prefix that is not a provider is part of the model id, and a bare id is just the id.
    assert _model_overrides("Qwen/Qwen3-8B", {}) == {"upstream_model": "Qwen/Qwen3-8B"}
    assert _model_overrides("qwen3-8b", {}) == {"upstream_model": "qwen3-8b"}
    assert _model_overrides("ollama/openai/gpt-oss-20b", {})["upstream_model"] == "openai/gpt-oss-20b"


@pytest.mark.unit
def test_a_profile_needs_a_model_and_sets_only_its_directory_for_the_service() -> None:
    with pytest.raises(DeployConfigError, match=r"pass --inference\.upstream-model MODEL"):
        _prepare_profile("reefine", None, {})
    env = {"REEF_UPSTREAM_MODEL": "qwen3-8b"}
    _prepare_profile("reefine", None, env)  # the environment names the model as before
    assert env["REEF_RECIPE_CONFIG_DIR"] == str(PROFILES_DIR)
    # The recipe ships in the wheel: no checkout goes on the service's path.
    assert set(env) == {"REEF_UPSTREAM_MODEL", "REEF_RECIPE_CONFIG_DIR"}


@pytest.mark.unit
def test_the_reefine_profile_loads_and_boots_its_recipe(monkeypatch, tmp_path) -> None:
    """The profile is the stack config and the preset in one file: it validates as a stack, the registry reads it
    back by the name it carries, and the recipe it builds is the built-in proposer over pi in manual mode."""
    from reef.recipe.registry import build_named_recipe
    from reef.service.assembly import _upstream_runtime
    from reef.service.deploy.orchestrator import resolve_deployment_config

    env: dict[str, str] = {}
    _prepare_profile("reefine", "ollama/gemma4:26b", env)
    for key, value in {
        **env,
        "REEF_UPSTREAM_URL": "http://127.0.0.1:11434",
        "REEF_UPSTREAM_MODEL": "gemma4:26b",
        "REEF_UPSTREAM_API_KEY": "ollama",
        "REEF_PYTHON": sys.executable,
    }.items():
        monkeypatch.setenv(key, value)
    path = profile_path("reefine")
    config = resolve_deployment_config(load_config(path, interpolate_env=False), None, path)[0]
    validate_services(config, path)
    assert [service["name"] for service in config["services"]] == ["reef"]
    assert config["reef"]["recipe"] == "reef.recipe.reefine:ReefineRecipe"
    assert (config["reef"]["port"], config["reef"]["token"]) == (8901, "reef-local")
    for key in ("agent_record_dir", "artifact_repository", "artifact_work_dir", "artifact_cache_dir"):
        assert config["reef"][key].startswith(".reef/reefine/")
    assert config["run_dir"].startswith(".reef/reefine/")
    service = service_config_from_mapping(config)
    monkeypatch.delenv("REEF_UPSTREAM_MODEL")
    built = build_named_recipe("reefine", dict(os.environ), default_runtime=_upstream_runtime(service))
    assert isinstance(built, ReefineRecipe) and built.adapter == "pi" and built.training_mode == "manual"
    assert built.model_binding().model == "gemma4:26b"
    records = SQLiteRecordStore()
    trainer = replace(built, binary=str(tmp_path / "fake-pi")).build("demo", records)
    assert trainer.training_mode == "manual"
    trainer.close()
    records.close()


@pytest.mark.unit
def test_serve_without_a_config_names_the_recipes_with_a_profile(tmp_path) -> None:
    # The subprocess runs away from the checkout, so reef must reach it through PYTHONPATH as the suite's own
    # interpreter has it; in the package job reef is installed and the entry is harmless.
    env = {k: v for k, v in os.environ.items() if k != "REEF_CONFIG"}
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(REPO_ROOT), env.get("PYTHONPATH", ""))))
    result = subprocess.run(
        [sys.executable, "-m", "reef.cli", "serve"], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 2
    assert "recipes with a profile: reefine" in result.stderr
    # The former name is announced before the profile's own checks run.
    result = subprocess.run(
        [sys.executable, "-m", "reef.cli", "serve", "--recipe", "harness-evolve"],
        cwd=tmp_path,
        env={k: v for k, v in env.items() if k != "REEF_UPSTREAM_MODEL"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 2
    assert ALIAS_NOTICE in result.stderr
    assert "pass --inference.upstream-model MODEL" in result.stderr


@pytest.mark.unit
def test_explicit_empty_profile_model_does_not_fall_back_to_the_environment():
    with pytest.raises(DeployConfigError, match=r"pass --inference\.upstream-model MODEL"):
        _prepare_profile("reefine", None, {"REEF_UPSTREAM_MODEL": "env-model"}, {"inference.upstream-model": ""})
