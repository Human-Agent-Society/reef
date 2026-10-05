"""Configuration and wiring for the Reefine-specific evaluator."""

import os
from collections.abc import Mapping
from pathlib import Path

from reef.recipe.cordis import _resolve_callable
from reef.recipe.errors import RecipeConfigError
from reef.recipe.reefine.prompts import BEHAVIOR_REVIEW_PROMPT, CHANGE_REVIEW_PROMPT, EVALUATION_PLAN_PROMPT
from reef.train.evaluation.reefine import (
    EvaluationInstructions,
    ReefineEvaluationSettings,
    ReefinePluginFactory,
    RequestContext,
    RequestVerifier,
)


def evaluation_factory(config: object, tasks: tuple[str, ...]) -> ReefinePluginFactory:
    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise RecipeConfigError("evolution.evaluation must be a mapping")
    if set(config) - {
        "reviewer_model",
        "request_context",
        "protected_tasks",
        "timeout_s",
        "max_output_tokens",
        "env_from",
    }:
        raise RecipeConfigError("unknown evolution.evaluation key")
    context = config.get("request_context", {})
    if not isinstance(context, Mapping) or set(context) - {
        "prompt",
        "fixture_dir",
        "initialize_git",
        "verifier",
        "version",
    }:
        raise RecipeConfigError("invalid evolution.evaluation.request_context")
    prompt = context.get("prompt", "")
    version = context.get("version", "1")
    initialize_git = context.get("initialize_git", False)
    if not isinstance(prompt, str) or not isinstance(version, str) or not isinstance(initialize_git, bool):
        raise RecipeConfigError("request_context prompt/version must be strings; initialize_git must be boolean")
    files: dict[str, str] = {}
    fixture = context.get("fixture_dir")
    if fixture is not None:
        if not isinstance(fixture, str) or not Path(fixture).is_dir():
            raise RecipeConfigError("request_context.fixture_dir must name an existing directory")
        root = Path(fixture).resolve()
        total_bytes = 0
        for path in root.rglob("*"):
            if path.is_symlink():
                raise RecipeConfigError("request fixtures must not contain symlinks")
            if path.is_file():
                if path.stat().st_size > 1024 * 1024:
                    raise RecipeConfigError("request fixture files must be at most 1 MiB")
                total_bytes += path.stat().st_size
                if len(files) >= 1000 or total_bytes > 50 * 1024 * 1024:
                    raise RecipeConfigError("request fixtures must contain at most 1000 files and 50 MiB")
                files[path.relative_to(root).as_posix()] = path.read_text(encoding="utf-8")
    verifier = None
    reference = context.get("verifier")
    if reference is not None:
        verifier = _resolve_callable(reference, "request verifier")()
        if not isinstance(verifier, RequestVerifier):
            raise RecipeConfigError("request verifier must inherit RequestVerifier")
    protected = config.get("protected_tasks", tasks)
    if not isinstance(protected, (list, tuple)) or any(
        not isinstance(task, str) or not task.strip() for task in protected
    ):
        raise RecipeConfigError("evaluation.protected_tasks must be a list of nonempty prompts")
    reviewer = config.get("reviewer_model", "served")
    timeout = config.get("timeout_s", 900)
    max_tokens = config.get("max_output_tokens", 8192)
    names = config.get("env_from", [])
    if not isinstance(names, list) or any(not isinstance(name, str) or not name for name in names):
        raise RecipeConfigError("evaluation.env_from must be a list of environment variable names")
    environment = {name: os.environ[name] for name in names if name in os.environ}
    if (
        not isinstance(reviewer, str)
        or not reviewer
        or isinstance(timeout, bool)
        or not isinstance(timeout, (float, int))
        or isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
    ):
        raise RecipeConfigError("invalid evaluation reviewer_model, timeout_s or max_output_tokens")
    try:
        settings = ReefineEvaluationSettings(
            reviewer,
            RequestContext(prompt, files, initialize_git, verifier, version),
            tuple(protected),
            float(timeout),
            max_tokens,
            environment=environment,
        )
    except ValueError as exc:
        raise RecipeConfigError(str(exc)) from exc
    return ReefinePluginFactory(
        settings, EvaluationInstructions(EVALUATION_PLAN_PROMPT, BEHAVIOR_REVIEW_PROMPT, CHANGE_REVIEW_PROMPT)
    )
