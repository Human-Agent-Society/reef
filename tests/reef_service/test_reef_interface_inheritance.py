from __future__ import annotations

import inspect
from abc import ABC

from reef import OpenClawRLRecipe
from reef.artifact import GitLFSRepositoryBackend, InMemoryRepositoryBackend, RepositoryBackend
from reef.recipe import Recipe
from reef.scenario.checkpoint_strategy import CheckpointStrategy, EveryNVersions


def test_local_interfaces_are_abstract_bases_with_explicit_implementations() -> None:
    assert issubclass(RepositoryBackend, ABC)
    assert issubclass(InMemoryRepositoryBackend, RepositoryBackend)
    assert issubclass(GitLFSRepositoryBackend, RepositoryBackend)
    for method_name in ("fork", "metadata", "current", "publish"):
        assert "scenario" not in inspect.signature(getattr(RepositoryBackend, method_name)).parameters

    assert issubclass(CheckpointStrategy, ABC)
    assert issubclass(EveryNVersions, CheckpointStrategy)

    recipe_parameters = inspect.signature(Recipe).parameters
    assert "name" in recipe_parameters
    assert "checkpoint_every_n_versions" not in recipe_parameters
    assert "checkpoint_strategy" in recipe_parameters
    assert Recipe().name == "recipe"
    assert Recipe().checkpoint_strategy == EveryNVersions(1)
    assert issubclass(OpenClawRLRecipe, Recipe)
