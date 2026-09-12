"""``reef serve --recipe <name> --model <provider>/<model>``: a recipe's profile starts without a config file."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest

from reef.recipe.cordis import CordisRecipe
from reef.service.deploy.cli import build_parser, build_serve_parser
from reef.service.deploy.config_utils import DeployConfigError, load_config
from reef.service.deploy.execution import validate_services
from reef.service.deploy.orchestrator import PROJECT_ROOT, _model_overrides, _prepare_profile, _resolve_config
from reef.service.deploy.service_config import service_config_from_mapping
from reef.service.profiles import PROFILES_DIR, UnknownProfileError, profile_names, profile_path
from reef.storage.sqlite import SQLiteRecordStore

REPO_ROOT = Path(__file__).resolve().parents[2]
#: The profile's proposer lives in the tutorial: the wheel alone (the package job) cannot start it.
IN_CHECKOUT = (PROJECT_ROOT / "tutorials" / "evolve-your-harness" / "harness" / "evolution.py").is_file()


@pytest.fixture
def isolated_harness_package() -> Iterator[None]:
    # Examples share this package name, but their modules belong to different
    # directories. Restore the worker's previous imports after this test.
    previous = {
        name: module for name, module in sys.modules.items() if name == "harness" or name.startswith("harness.")
    }
    for name in previous:
        sys.modules.pop(name)
    try:
        yield
    finally:
        for name in [name for name in sys.modules if name == "harness" or name.startswith("harness.")]:
            sys.modules.pop(name)
        sys.modules.update(previous)


@pytest.mark.unit
def test_the_harness_evolve_recipe_carries_the_one_profile() -> None:
    assert profile_names() == ("harness-evolve",)
    assert profile_path("harness-evolve") == PROFILES_DIR / "harness-evolve.yaml"
    with pytest.raises(UnknownProfileError, match="recipes with a profile: harness-evolve"):
        profile_path("weights")
    with pytest.raises(UnknownProfileError):
        profile_path("../harness-evolve")


@pytest.mark.unit
def test_the_serve_parser_takes_recipe_and_model_and_the_service_parser_still_does_not() -> None:
    args, extras = build_serve_parser().parse_known_args(
        ["--recipe", "harness-evolve", "--model", "ollama/gemma4:26b", "--port", "8901"]
    )
    assert (args.config, args.recipe, args.model, args.port, extras) == (
        None,
        "harness-evolve",
        "ollama/gemma4:26b",
        8901,
        [],
    )
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--recipe", "harness-evolve"])


@pytest.mark.unit
def test_only_an_explicit_config_or_profile_selects_a_file(tmp_path, monkeypatch) -> None:
    """An environment variable or nearby YAML must not silently select a deployment."""
    profile = str(PROFILES_DIR / "harness-evolve.yaml")
    monkeypatch.setenv("REEF_CONFIG", "env.yaml")
    assert _resolve_config("mine.yaml", None) == "mine.yaml"
    assert _resolve_config(None, "harness-evolve") == profile
    assert _resolve_config(None, None) is None
    with pytest.raises(DeployConfigError, match="not both"):
        _resolve_config("mine.yaml", "harness-evolve")
    with pytest.raises(DeployConfigError, match="recipes with a profile: harness-evolve"):
        _resolve_config(None, "weights")
    monkeypatch.chdir(tmp_path)
    installed = tmp_path / "installed"
    installed.mkdir()
    (installed / "reef.yaml").write_text("reef: {}\n")
    monkeypatch.setattr("reef.service.deploy.orchestrator.PROJECT_ROOT", installed)
    assert _resolve_config(None, None) is None
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
def test_a_profile_needs_a_model_and_the_checkout_before_it_loads(tmp_path, monkeypatch) -> None:
    with pytest.raises(DeployConfigError, match=r"pass --inference\.upstream-model MODEL"):
        _prepare_profile("harness-evolve", None, {})
    monkeypatch.setattr("reef.service.deploy.orchestrator.PROJECT_ROOT", tmp_path)
    with pytest.raises(DeployConfigError, match="runs from a reef checkout"):
        _prepare_profile("harness-evolve", "ollama/gemma4:26b", {})


@pytest.mark.unit
@pytest.mark.skipif(not IN_CHECKOUT, reason="the harness-evolve profile runs from a reef checkout")
def test_a_profile_sets_its_directory_and_the_checkout_for_the_service() -> None:
    env: dict[str, str] = {}
    _prepare_profile("harness-evolve", "ollama/gemma4:26b", env)
    assert env["REEF_RECIPE_CONFIG_DIR"] == str(PROFILES_DIR)
    assert env["REEF_CHECKOUT"] == str(PROJECT_ROOT)
    assert str(PROJECT_ROOT) in env["PYTHONPATH"].split(os.pathsep)
    env = {"REEF_UPSTREAM_MODEL": "qwen3-8b"}
    _prepare_profile("harness-evolve", None, env)  # the environment names the model as before


@pytest.mark.unit
@pytest.mark.skipif(not IN_CHECKOUT, reason="the harness-evolve profile runs from a reef checkout")
@pytest.mark.usefixtures("isolated_harness_package")
def test_the_harness_evolve_profile_loads_and_boots_its_recipe(monkeypatch, tmp_path) -> None:
    """The profile is the stack config and the preset in one file: it validates as a stack, the registry reads it
    back by the name it carries, and the recipe it builds is the tutorial's proposer over pi in hybrid mode."""
    from reef.recipe.registry import build_named_recipe
    from reef.service.assembly import _upstream_runtime

    env: dict[str, str] = {}
    _prepare_profile("harness-evolve", "ollama/gemma4:26b", env)
    for key, value in {
        **env,
        "REEF_UPSTREAM_URL": "http://127.0.0.1:11434",
        "REEF_UPSTREAM_MODEL": "gemma4:26b",
        "REEF_UPSTREAM_API_KEY": "ollama",
        "REEF_PYTHON": sys.executable,
    }.items():
        monkeypatch.setenv(key, value)
    path = profile_path("harness-evolve")
    from reef.service.deploy.orchestrator import resolve_deployment_config

    config = resolve_deployment_config(load_config(path, interpolate_env=False), None, path)[0]
    validate_services(config, path)
    assert [service["name"] for service in config["services"]] == ["reef"]
    assert env["REEF_RECIPE_CONFIG_DIR"] == str(PROFILES_DIR)
    method_root = Path(env["PYTHONPATH"].split(os.pathsep)[0])
    assert (method_root / "harness" / "evolution.py").is_file()
    assert config["reef"]["recipe"] == "reef.recipe.cordis:CordisRecipe" and config["reef"]["port"] == 8900
    assert "token" not in config["reef"]  # loopback only; a copy of the file adds one
    for key in ("agent_record_dir", "artifact_repository", "artifact_work_dir", "artifact_cache_dir"):
        assert config["reef"][key].startswith(".reef/harness-evolve/")
    assert config["run_dir"].startswith(".reef/harness-evolve/")
    monkeypatch.syspath_prepend(str(method_root))
    service = service_config_from_mapping(config)
    monkeypatch.delenv("REEF_UPSTREAM_MODEL")
    built = build_named_recipe("harness-evolve", dict(os.environ), default_runtime=_upstream_runtime(service))
    assert isinstance(built, CordisRecipe) and built.adapter == "pi" and built.training_mode == "hybrid"
    assert built.model_binding().model == "gemma4:26b"
    records = SQLiteRecordStore()
    trainer = replace(built, binary=str(tmp_path / "fake-pi")).build("demo", records)
    assert trainer.training_mode == "hybrid"
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
    assert "recipes with a profile: harness-evolve" in result.stderr
    result = subprocess.run(
        [sys.executable, "-m", "reef.cli", "serve", "--recipe", "harness-evolve"],
        cwd=tmp_path,
        env={k: v for k, v in env.items() if k != "REEF_UPSTREAM_MODEL"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 2
    assert "pass --inference.upstream-model MODEL" in result.stderr


@pytest.mark.unit
def test_explicit_empty_profile_model_does_not_fall_back_to_the_environment():
    with pytest.raises(DeployConfigError, match=r"pass --inference\.upstream-model MODEL"):
        _prepare_profile(
            "harness-evolve", None, {"REEF_UPSTREAM_MODEL": "env-model"}, {"inference.upstream-model": ""}
        )
