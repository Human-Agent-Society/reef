"""Load and play a Gym style environment class; standard library only, shipped into every SPADE task.

This module ships verbatim as ``env_loader.py`` into a task's ``environment/`` (beside the observe and act
commands the agent runs inside the container) and into its ``tests/`` (beside the replay verifier), and the recipe
imports it on the host, so there is one loader, one action rule and one return rule. The rules are those
of SPADE's reference code (``spade/core/envs/synthetic_game_env.py`` and ``spade/core/utils/parsing.py``):
the class is the first top level class whose name ends in ``Env``, the code may import the standard
library only, the module namespace holds the standard library names an environment tends to use without
importing them, the agent's text reaches ``step`` as ``\\boxed{action}`` (the first box, boxed again after
extraction; the raw text when there is no box), an exception inside ``step`` ends the episode with -1.0,
a missing reward is 0.0, and the episode return is the last reward of a terminated episode clipped to
[-1, 1]. The action log is one JSON string per line, so an action that spans lines or is empty replays
as written.
"""

from __future__ import annotations

import ast
import collections
import copy
import functools
import heapq
import importlib.util
import inspect
import itertools
import json
import math
import operator
import random
import re
import statistics
import string
import sys
import typing
from collections.abc import Sequence

ENV_CLASS_SUFFIX = "Env"
BASE_CLASS_NAMES = ("Env",)
TOOL_USE_BASE_NAME = "ToolUseBaseEnv"
ERROR_REWARD = -1.0
MISSING = object()
BOXED_PATTERNS = (
    re.compile(r"\\boxed\{([^{}]+)\}"),
    re.compile(r"\\boxed\{\{([^{}]+)\}\}"),
    re.compile(r"\\boxed\{\{\{([^{}]+)\}\}\}"),
)
# The names the reference injects before running a generated class, minus numpy: the task image has none.
INJECTED_NAMES: dict[str, object] = {
    "random": random,
    "re": re,
    "math": math,
    "json": json,
    "itertools": itertools,
    "collections": collections,
    "functools": functools,
    "string": string,
    "copy": copy,
    "heapq": heapq,
    "operator": operator,
    "statistics": statistics,
    "deque": collections.deque,
    "Counter": collections.Counter,
    "defaultdict": collections.defaultdict,
    "OrderedDict": collections.OrderedDict,
    "namedtuple": collections.namedtuple,
    "combinations": itertools.combinations,
    "permutations": itertools.permutations,
    "product": itertools.product,
    "chain": itertools.chain,
    "accumulate": itertools.accumulate,
    "reduce": functools.reduce,
    "lru_cache": functools.lru_cache,
    "deepcopy": copy.deepcopy,
    "heappush": heapq.heappush,
    "heappop": heapq.heappop,
    "Any": typing.Any,
    "Optional": typing.Optional,
    "Union": typing.Union,
    "List": list,
    "Dict": dict,
    "Tuple": tuple,
    "Set": set,
    "FrozenSet": frozenset,
    "Callable": typing.Callable,
    "Iterable": typing.Iterable,
    "Sequence": typing.Sequence,
    "Mapping": typing.Mapping,
}


def environment_class_name(code: str) -> str:
    """The first top level class whose name ends in ``Env``, found without running the code."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"environment code is not valid Python: {exc}") from exc
    if any(isinstance(node, ast.Name) and node.id == TOOL_USE_BASE_NAME for node in ast.walk(tree)):
        raise ValueError(
            f"environment code uses {TOOL_USE_BASE_NAME}, which no task ships; tool use environments are not supported"
        )
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module is not None:
            modules = [node.module]
        else:
            modules = []
        for module in modules:
            if module.split(".")[0] not in sys.stdlib_module_names:
                raise ValueError(f"environment code imports {module!r}; only the standard library is available")
    for node in tree.body:
        if (
            isinstance(node, ast.ClassDef)
            and node.name.endswith(ENV_CLASS_SUFFIX)
            and node.name not in BASE_CLASS_NAMES
        ):
            return node.name
    raise ValueError("environment code defines no top level class ending in 'Env'")


def load_environment_class(environment_path: str) -> type:
    """Run the module at ``environment_path`` with the injected names and return its environment class."""
    with open(environment_path, encoding="utf-8") as handle:
        name = environment_class_name(handle.read())
    spec = importlib.util.spec_from_file_location("generated_env", environment_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load {environment_path}")
    module = importlib.util.module_from_spec(spec)
    vars(module).update(INJECTED_NAMES)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    value = vars(module)[name]
    if not isinstance(value, type):
        raise ValueError(f"{name} is not a class")
    return value


def extract_boxed_answer(text: str) -> str | None:
    """The first plain box as the reference reads it, else the last nested box with up to two brace layers stripped."""
    for pattern in BOXED_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    start = text.rfind("\\boxed")
    if start < 0:
        return None
    opening = text.find("{", start)
    if opening < 0:
        return None
    depth = 0
    for position in range(opening, len(text)):
        if text[position] == "{":
            depth += 1
        elif text[position] == "}":
            depth -= 1
            if depth == 0:
                content = text[opening + 1 : position].strip()
                for _ in range(2):
                    if content.startswith("{") and content.endswith("}"):
                        content = content[1:-1].strip()
                return content
    return None


def normalized_action(text: str) -> str:
    """What ``step`` receives: the extracted answer boxed again, else the raw text stripped."""
    answer = extract_boxed_answer(text)
    if not answer:
        return text.strip()
    return f"\\boxed{{{answer}}}"


def make_environment(environment_class: type, max_turns: int) -> object:
    """Instantiate the class as the reference does and cap its own turn limit at ``max_turns``."""
    try:
        environment = environment_class()
    except TypeError as exc:
        parameters = inspect.signature(environment_class).parameters
        is_max_turns_accepted = "max_turns" in parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
        )
        if not is_max_turns_accepted:
            raise exc
        environment = environment_class(max_turns=max_turns)
    # Generated code is third party: its limit may sit on a base class or a property, so it is read by name.
    designed = getattr(environment, "max_turns", MISSING)
    if isinstance(designed, int) and not isinstance(designed, bool) and designed > 0:
        environment.max_turns = min(designed, max_turns)
    elif designed is not MISSING:
        environment.max_turns = max_turns
    return environment


def step_result(environment: object, action: str) -> tuple[str, float, bool, bool, str | None]:
    """One ``step``: (observation, reward, terminated, truncated, error); a broken step is the error reward, terminated, named."""
    try:
        result = environment.step(action)  # type: ignore[attr-defined]
        if not isinstance(result, tuple) or len(result) != 5:
            raise TypeError("step did not return (observation, reward, terminated, truncated, info)")
        observation, reward, terminated, truncated, _ = result
        if reward is None:
            reward = 0.0
        reward = float(reward)
        if not math.isfinite(reward):
            reward = 0.0
        return str(observation), reward, bool(terminated), bool(truncated), None
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        return f"Error: {error}", ERROR_REWARD, True, False, error


def step_once(environment: object, action: str) -> tuple[str, float, bool, bool]:
    """One ``step`` as the verifier and the act command take it: (observation, reward, terminated, truncated)."""
    observation, reward, terminated, truncated, _ = step_result(environment, action)
    return observation, reward, terminated, truncated


def replay(environment: object, actions: Sequence[str], max_turns: int) -> tuple[list[float], bool]:
    """Replay ``actions`` (at most ``max_turns``) and return the rewards and whether the episode terminated."""
    rewards: list[float] = []
    terminated = False
    for action in list(actions)[:max_turns]:
        _, reward, terminated, truncated = step_once(environment, normalized_action(action))
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


def read_actions(actions_path: str) -> list[str]:
    """The action log at ``actions_path``: one JSON string per line; a line that is not one is taken as written."""
    try:
        with open(actions_path, encoding="utf-8", errors="replace") as handle:
            lines = [line.rstrip("\r\n") for line in handle if line.strip()]
    except OSError:
        return []
    actions: list[str] = []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            value = line
        if isinstance(value, str):
            actions.append(value)
        else:
            actions.append(line)
    return actions


def write_actions(actions_path: str, actions: Sequence[str]) -> None:
    """Write the action log the way ``read_actions`` reads it: one JSON string per line, ASCII only."""
    with open(actions_path, "w", encoding="utf-8") as handle:
        for action in actions:
            handle.write(json.dumps(action) + "\n")
