"""Load and play a generated environment the way the SPADE reference code does; standard library only.

This module ships verbatim into every task's ``tests/`` directory (``env_loader.py``) beside the replay
verifier, and the smoke test and the driver import it on the host, so there is one loader, one action
rule and one return rule. It mirrors ``spade/core/envs/synthetic_game_env.py`` and
``spade/core/utils/parsing.py``: the class is the first top level class whose name ends in ``Env``, the
module namespace holds the standard library names a Designer tends to use without importing them, the
agent's text reaches ``step`` as ``\\boxed{action}`` (re-boxed after extraction, the raw text when
there is no box), an exception inside ``step`` ends the episode with -1.0, a missing reward is 0.0.
"""

from __future__ import annotations

import ast
import collections
import importlib.util
import itertools
import json
import math
import random
import re
import sys
import typing
from collections.abc import Sequence

ENV_CLASS_SUFFIX = "Env"
BASE_CLASS_NAMES = ("Env",)
TOOL_USE_BASE_NAME = "ToolUseBaseEnv"
ERROR_REWARD = -1.0
BOXED_PATTERNS = (
    re.compile(r"\\boxed\{\{\{([^{}]+)\}\}\}"),
    re.compile(r"\\boxed\{\{([^{}]+)\}\}"),
    re.compile(r"\\boxed\{([^{}]+)\}"),
)
INJECTED_NAMES: dict[str, object] = {
    "random": random,
    "re": re,
    "math": math,
    "json": json,
    "itertools": itertools,
    "collections": collections,
    "deque": collections.deque,
    "Counter": collections.Counter,
    "Any": typing.Any,
    "Optional": typing.Optional,
    "List": list,
    "Dict": dict,
    "Tuple": tuple,
}


def environment_class_name(code: str) -> str:
    """The first top level class whose name ends in ``Env``, found without running the code."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"environment code is not valid Python: {exc}") from exc
    if any(isinstance(node, ast.Name) and node.id == TOOL_USE_BASE_NAME for node in ast.walk(tree)):
        raise ValueError(
            f"environment code uses {TOOL_USE_BASE_NAME}, which no task ships; tool use games are not supported"
        )
    for node in tree.body:
        if (
            isinstance(node, ast.ClassDef)
            and node.name.endswith(ENV_CLASS_SUFFIX)
            and node.name not in BASE_CLASS_NAMES
        ):
            return node.name
    raise ValueError("environment code defines no top level class ending in 'Env'")


def load_environment_class(path: str) -> type:
    """Run the module at ``path`` with the injected names and return its environment class."""
    with open(path, encoding="utf-8") as handle:
        name = environment_class_name(handle.read())
    spec = importlib.util.spec_from_file_location("generated_env", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    vars(module).update(INJECTED_NAMES)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    value = vars(module)[name]
    if not isinstance(value, type):
        raise ValueError(f"{name} is not a class")
    return value


def extract_boxed_answer(text: str) -> str | None:
    """The content of the last ``\\boxed{...}``, nested braces included; None when there is no box."""
    for pattern in BOXED_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            return matches[-1].strip()
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    depth = 0
    for position in range(start + len("\\boxed"), len(text)):
        if text[position] == "{":
            depth += 1
        elif text[position] == "}":
            depth -= 1
            if depth == 0:
                return text[start + len("\\boxed{") : position].strip()
    return None


def normalized_action(text: str) -> str:
    """What ``step`` receives: the extracted answer re-boxed, else the raw text stripped."""
    answer = extract_boxed_answer(text)
    if answer is None:
        return text.strip()
    return f"\\boxed{{{answer}}}"


def make_environment(environment_class: type, max_turns: int) -> object:
    """Instantiate the class as the reference does and cap its own turn limit at ``max_turns``."""
    try:
        environment = environment_class()
    except TypeError:
        environment = environment_class(max_turns=max_turns)
    # Generated classes vary; the reference reads the designed limit off the instance the same way.
    designed = vars(environment).get("max_turns", vars(type(environment)).get("max_turns"))
    if isinstance(designed, int) and not isinstance(designed, bool) and designed > 0:
        environment.max_turns = min(designed, max_turns)
    return environment


def step_once(environment: object, action: str) -> tuple[float, bool, bool]:
    """One ``step``: (reward, terminated, truncated); a broken step is an error reward that terminates."""
    try:
        result = environment.step(action)  # type: ignore[attr-defined]
    except Exception:
        return ERROR_REWARD, True, False
    if not isinstance(result, tuple) or len(result) != 5:
        return ERROR_REWARD, True, False
    _, reward, terminated, truncated, _ = result
    if reward is None:
        reward = 0.0
    try:
        reward = float(reward)
    except (TypeError, ValueError):
        return ERROR_REWARD, True, False
    if not math.isfinite(reward):
        reward = 0.0
    return reward, bool(terminated), bool(truncated)


def play(environment: object, actions: Sequence[str], max_turns: int) -> tuple[list[float], bool]:
    """Replay ``actions`` (at most ``max_turns``) and return the rewards and whether the episode terminated."""
    rewards: list[float] = []
    terminated = False
    for action in list(actions)[:max_turns]:
        reward, terminated, truncated = step_once(environment, normalized_action(action))
        rewards.append(reward)
        if terminated or truncated:
            break
    return rewards, terminated


def episode_return(rewards: Sequence[float], terminated: bool) -> float:
    """The last reward clipped to [-1, 1] when the episode terminated, else 0.0; a non finite last reward is 0.0."""
    if not rewards or not terminated:
        return 0.0
    last = float(rewards[-1])
    if not math.isfinite(last):
        return 0.0
    return max(-1.0, min(1.0, last))
