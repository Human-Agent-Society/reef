"""The profiles ``reef serve --recipe <name>`` starts without a config file.

A profile is one versioned deployment YAML, with its selected class under
``recipe.implementation`` and component values under ``recipe.config``. It
can also be read as a named recipe preset through the public layout adapter. ``REEF_RECIPE_CONFIG_DIR``
still names a directory of the operator's own presets; a profile is only ever
selected by name on the command line.

A folded profile keeps its name as an alias of the profile that replaced it:
``--recipe harness-evolve`` starts ``reefine`` and the launcher says so.
"""

from __future__ import annotations

from pathlib import Path

from reef.core.errors import ReefError

PROFILES_DIR = Path(__file__).resolve().parent

#: Former profile names and the profile each one starts now.
PROFILE_ALIASES: dict[str, str] = {"harness-evolve": "reefine"}


class UnknownProfileError(ReefError):
    """``--recipe`` named a recipe that carries no profile."""


def profile_names() -> tuple[str, ...]:
    """The recipes that carry a profile, by name, sorted; a former name is not one of them."""
    return tuple(sorted(path.stem for path in PROFILES_DIR.glob("*.yaml")))


def profile_alias_notice(name: str) -> str | None:
    """The line the launcher prints when ``name`` is a former profile name, or None for a current one."""
    target = PROFILE_ALIASES.get(name)
    if target is None:
        return None
    return f"reef: --recipe {name} is now {target}; starting the {target} profile"


def profile_path(name: str) -> Path:
    """The profile file of ``name`` (a former name included), or an error naming the recipes that have one."""
    path = PROFILES_DIR / f"{PROFILE_ALIASES.get(name, name)}.yaml"
    if "/" in name or not path.is_file():
        aliases = "; ".join(f"--recipe {alias} starts {target}" for alias, target in PROFILE_ALIASES.items())
        raise UnknownProfileError(
            f"no profile for recipe {name!r}; recipes with a profile: {', '.join(profile_names())} ({aliases})"
        )
    return path
