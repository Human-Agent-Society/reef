"""The profiles ``reef serve --recipe <name>`` starts without a config file.

A profile is one versioned deployment YAML, with its selected class under
``recipe.implementation`` and component values under ``recipe.config``. It
can also be read as a named recipe preset through the public layout adapter. ``REEF_RECIPE_CONFIG_DIR``
still names a directory of the operator's own presets; a profile is only ever
selected by name on the command line.
"""

from __future__ import annotations

from pathlib import Path

from reef.core.errors import ReefError

PROFILES_DIR = Path(__file__).resolve().parent


class UnknownProfileError(ReefError):
    """``--recipe`` named a recipe that carries no profile."""


def profile_names() -> tuple[str, ...]:
    """The recipes that carry a profile, by name, sorted."""
    return tuple(sorted(path.stem for path in PROFILES_DIR.glob("*.yaml")))


def profile_path(name: str) -> Path:
    """The profile file of ``name``, or an error naming the recipes that have one."""
    path = PROFILES_DIR / f"{name}.yaml"
    if "/" in name or not path.is_file():
        raise UnknownProfileError(
            f"no profile for recipe {name!r}; recipes with a profile: {', '.join(profile_names())}"
        )
    return path
