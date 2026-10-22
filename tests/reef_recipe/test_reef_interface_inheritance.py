from __future__ import annotations

import inspect
from abc import ABC

from recipes.openclawrl import OpenClawRLRecipe
from reef.artifact import GitLFSRepositoryBackend, InMemoryRepositoryBackend, RepositoryBackend
from reef.recipe import Recipe


def test_local_interfaces_are_abstract_bases_with_explicit_implementations() -> None:
    assert issubclass(RepositoryBackend, ABC)
    assert issubclass(InMemoryRepositoryBackend, RepositoryBackend)
    assert issubclass(GitLFSRepositoryBackend, RepositoryBackend)
    for method_name in ("fork", "metadata", "current", "publish"):
        assert "scenario" not in inspect.signature(getattr(RepositoryBackend, method_name)).parameters

    recipe_parameters = inspect.signature(Recipe).parameters
    assert "name" in recipe_parameters
    assert "checkpoint_every_n_versions" not in recipe_parameters
    assert "checkpoint_strategy" not in recipe_parameters
    assert Recipe().name == "recipe"
    assert issubclass(OpenClawRLRecipe, Recipe)
