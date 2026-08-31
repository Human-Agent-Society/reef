"""Evaluator for the Frontier-CS 2.0 Generals.io bot arena task."""

from __future__ import annotations

import importlib.util
import ast
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jrandom
from generals import GeneralsEnv, get_observation
from generals.agents import Agent, ExpanderAgent, HunterAgent, RandomAgent
from generals.core.action import compute_valid_move_mask_obs

_DIRECTIONS = jnp.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=jnp.int32)

MAX_PATCH_BYTES = 500_000
MAX_CHANGED_FILES = 20
TASK_CONFIG_PATH = Path("/judge/task_config.json")
DEFAULT_CLEAN_SOURCE = Path("/opt/generals-agent-clean")
LOCAL_CLEAN_SOURCE = Path(__file__).parent / "harbor" / "app" / "generals_agent"

ALLOWED_FILES = {
    "bot.py",
    "strategy.py",
    "utils.py",
}
DENIED_TOKENS = (
    "import os",
    "from os",
    "import sys",
    "from sys",
    "subprocess",
    "import socket",
    "from socket",
    "socket",
    "import requests",
    "from requests",
    "requests",
    "import urllib",
    "from urllib",
    "urllib",
    "urllib3",
    "httpx",
    "aiohttp",
    "websocket",
    "socketio",
    "generals.remote",
    "http.client",
    "ftplib",
    "open(",
    "io.",
    "import pathlib",
    "from pathlib",
    "pathlib",
    "Path(",
    "read_text",
    "read_bytes",
    "write_text",
    "write_bytes",
    "os.environ",
    "os.getenv",
    "__import__",
    "importlib",
    "eval(",
    "exec(",
    "compile(",
)
DENIED_IMPORT_ROOTS = {
    "builtins",
    "ftplib",
    "http",
    "httpx",
    "importlib",
    "io",
    "os",
    "pathlib",
    "requests",
    "socket",
    "subprocess",
    "sys",
    "urllib",
    "urllib3",
    "websocket",
}
DENIED_IMPORT_PREFIXES = {
    "generals.remote",
}
DENIED_CALL_NAMES = {
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "dir",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
}
DENIED_ATTR_NAMES = {
    "environ",
    "getenv",
    "popen",
    "read_bytes",
    "read_text",
    "remove",
    "rename",
    "replace",
    "rmdir",
    "system",
    "unlink",
    "write_bytes",
    "write_text",
}
DENIED_DUNDER_ATTRS = {
    "__bases__",
    "__class__",
    "__code__",
    "__dict__",
    "__getattribute__",
    "__globals__",
    "__mro__",
    "__subclasses__",
}

DEFAULT_BASELINES = (
    "random_low_split",
    "expander",
    "strongest_frontier",
    "hunter",
    "fast_pathing",
    "flobot_fast",
)


class _EvaluationTimeout(Exception):
    pass


def _load_task_config() -> dict[str, Any]:
    try:
        payload = json.loads(TASK_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


TASK_CONFIG = _load_task_config()
EVALUATION_CONFIG = (
    TASK_CONFIG.get("evaluation", {})
    if isinstance(TASK_CONFIG.get("evaluation"), dict)
    else {}
)


def _config_int(name: str, default: int) -> int:
    try:
        return int(EVALUATION_CONFIG.get(name, default))
    except Exception:
        return default


def _config_float(name: str, default: float) -> float:
    try:
        return float(EVALUATION_CONFIG.get(name, default))
    except Exception:
        return default


def _config_tuple_int(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    raw = EVALUATION_CONFIG.get(name, default)
    if isinstance(raw, list):
        return tuple(int(x) for x in raw)
    if isinstance(raw, tuple):
        return tuple(int(x) for x in raw)
    return default


def _config_tuple_str(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = EVALUATION_CONFIG.get(name, default)
    if isinstance(raw, list):
        return tuple(str(x) for x in raw)
    if isinstance(raw, tuple):
        return tuple(str(x) for x in raw)
    return default


def _clean_source_dir() -> Path:
    if DEFAULT_CLEAN_SOURCE.exists():
        return DEFAULT_CLEAN_SOURCE
    return LOCAL_CLEAN_SOURCE


def _run(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )


def _changed_files(patch_path: Path) -> list[str]:
    result = _run(["git", "diff", "--name-only", "--no-index", "--", "/dev/null", str(patch_path)], cwd=Path("/tmp"))
    del result
    text = patch_path.read_text(encoding="utf-8", errors="replace")
    paths: list[str] = []
    for line in text.splitlines():
        if line.startswith("+++ b/"):
            path = line.removeprefix("+++ b/").strip()
            if path != "/dev/null" and path not in paths:
                paths.append(path)
    return paths


def _validate_patch(patch_path: Path) -> tuple[bool, str, list[str]]:
    try:
        data = patch_path.read_bytes()
    except Exception as exc:
        return False, f"could not read patch: {exc}", []
    if not data.strip():
        return True, "baseline skeleton", []
    if len(data) > MAX_PATCH_BYTES:
        return False, f"patch too large: {len(data)} bytes > {MAX_PATCH_BYTES}", []
    if b"\x00" in data:
        return False, "binary patches are not allowed", []

    text = data.decode("utf-8", errors="replace")
    if re.search(r"^GIT binary patch$", text, flags=re.MULTILINE):
        return False, "binary patches are not allowed", []

    paths = _changed_files(patch_path)
    if not paths:
        return False, "patch does not modify any tracked file", []
    if len(paths) > MAX_CHANGED_FILES:
        return False, f"too many changed files: {len(paths)} > {MAX_CHANGED_FILES}", paths
    for path in paths:
        normalized = Path(path)
        if normalized.is_absolute() or ".." in normalized.parts:
            return False, f"unsafe patch path: {path}", paths
        if path not in ALLOWED_FILES:
            return False, f"patch may only modify {sorted(ALLOWED_FILES)}; got {path}", paths

    lowered = text.lower()
    for token in DENIED_TOKENS:
        if token.lower() in lowered:
            return False, f"patch contains denied token: {token}", paths
    return True, "ok", paths


def _validate_candidate_source(work: Path) -> tuple[bool, str]:
    for filename in sorted(ALLOWED_FILES):
        path = work / filename
        if not path.exists():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            return False, f"{filename} has invalid Python syntax"
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".", 1)[0]
                    if root in DENIED_IMPORT_ROOTS or any(
                        alias.name == prefix or alias.name.startswith(prefix + ".")
                        for prefix in DENIED_IMPORT_PREFIXES
                    ):
                        return False, f"{filename} imports denied module: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                root = module.split(".", 1)[0]
                if root in DENIED_IMPORT_ROOTS or any(
                    module == prefix or module.startswith(prefix + ".")
                    for prefix in DENIED_IMPORT_PREFIXES
                ):
                    return False, f"{filename} imports denied module: {module}"
            elif isinstance(node, ast.Name):
                if node.id == "__builtins__" or (node.id.startswith("__") and node.id.endswith("__")):
                    return False, f"{filename} uses denied dynamic name: {node.id}"
            elif isinstance(node, ast.Attribute):
                if node.attr in DENIED_ATTR_NAMES or node.attr in DENIED_DUNDER_ATTRS:
                    return False, f"{filename} uses denied attribute: {node.attr}"
            elif isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in DENIED_CALL_NAMES:
                    return False, f"{filename} calls denied function: {func.id}"
                if isinstance(func, ast.Attribute) and func.attr in DENIED_ATTR_NAMES:
                    return False, f"{filename} calls denied method: {func.attr}"
    return True, "ok"


def _prepare_candidate(patch_path: Path) -> tuple[Path | None, str]:
    ok, message, _ = _validate_patch(patch_path)
    if not ok:
        return None, message

    clean_source = _clean_source_dir()
    if not clean_source.exists():
        return None, f"clean source not found: {clean_source}"

    tmp = Path(tempfile.mkdtemp(prefix="frontier-generals-"))
    work = tmp / "generals_agent"
    shutil.copytree(clean_source, work, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    _run(["git", "init", "-q"], cwd=work)
    _run(["git", "config", "user.email", "frontier-cs@example.invalid"], cwd=work)
    _run(["git", "config", "user.name", "Frontier-CS Judge"], cwd=work)
    _run(["git", "add", "."], cwd=work)
    _run(["git", "commit", "-q", "-m", "base"], cwd=work)

    if patch_path.read_text(encoding="utf-8", errors="replace").strip():
        check = _run(["git", "apply", "--check", str(patch_path)], cwd=work)
        if check.returncode != 0:
            return None, "patch failed to apply"
        apply = _run(["git", "apply", str(patch_path)], cwd=work)
        if apply.returncode != 0:
            return None, "patch failed to apply"
    ok, message = _validate_candidate_source(work)
    if not ok:
        return None, message
    return work, "ok"


def _load_candidate_factory(work: Path):
    sys.path.insert(0, str(work))
    try:
        spec = importlib.util.spec_from_file_location("frontier_candidate_bot", work / "bot.py")
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load bot.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls = getattr(module, "FrontierAgent")
        return cls
    finally:
        try:
            sys.path.remove(str(work))
        except ValueError:
            pass


def _shift_grid(values, fill_value, step: int, axis: int):
    shifted = jnp.roll(values, step, axis)
    edge = 0 if step == 1 else -1
    if axis == 0:
        return shifted.at[edge, :].set(fill_value)
    return shifted.at[:, edge].set(fill_value)


def _bfs_distance(passable, sources):
    h, w = passable.shape
    inf = jnp.int32(h * w + 7)

    def relax(_, dist):
        neighbors = jnp.minimum(
            jnp.minimum(_shift_grid(dist, inf, 1, 0), _shift_grid(dist, inf, -1, 0)),
            jnp.minimum(_shift_grid(dist, inf, 1, 1), _shift_grid(dist, inf, -1, 1)),
        )
        return jnp.where(
            sources,
            0,
            jnp.where(passable, jnp.minimum(dist, neighbors + 1), inf),
        )

    return jax.lax.fori_loop(0, h * w, relax, jnp.where(sources, 0, inf))


def _best_direction_toward(field, passable):
    inf = jnp.int32(field.size + 13)
    values = jnp.stack(
        [
            jnp.where(_shift_grid(passable, False, 1, 0), _shift_grid(field, inf, 1, 0), inf),
            jnp.where(_shift_grid(passable, False, -1, 0), _shift_grid(field, inf, -1, 0), inf),
            jnp.where(_shift_grid(passable, False, 1, 1), _shift_grid(field, inf, 1, 1), inf),
            jnp.where(_shift_grid(passable, False, -1, 1), _shift_grid(field, inf, -1, 1), inf),
        ],
        axis=0,
    )
    return jnp.argmin(values, axis=0).astype(jnp.int32), jnp.min(values, axis=0)


class _PathingBaselineAgent(Agent):
    """Judge-only pathing baseline with tuned hunting and exploration pressure."""

    def __init__(
        self,
        *,
        id: str = "Pathing",
        release_threshold: int = 6,
        scout_far: bool = True,
        block_neutral_cities: bool = True,
        convoy_weight: float = 100.0,
    ):
        super().__init__(id=id)
        self.release_threshold = int(release_threshold)
        self.scout_far = bool(scout_far)
        self.block_neutral_cities = bool(block_neutral_cities)
        self.convoy_weight = float(convoy_weight)

    @partial(jax.jit, static_argnums=0)
    def act(self, observation, key):
        del key
        army = observation.armies
        mine = observation.owned_cells
        h, w = army.shape
        reach = jnp.int32(h * w + 7)
        city_block = observation.cities & ~mine if self.block_neutral_cities else jnp.zeros_like(mine)
        passable = ~(observation.mountains | observation.structures_in_fog | city_block)
        movable = mine & (army > 1)
        mine_army = jnp.where(mine, army, 0)

        own_general = mine & observation.generals
        own_general_army = jnp.sum(jnp.where(own_general, army, 0))
        own_general_idx = jnp.argmax(own_general.reshape(-1).astype(jnp.int32))
        dist_from_general = _bfs_distance(passable, own_general)

        enemy_general = observation.opponent_cells & observation.generals
        enemy_land = observation.opponent_cells & ~observation.cities
        fog = observation.fog_cells & passable & (dist_from_general < reach)
        open_land = passable & ~mine & (dist_from_general < reach)

        def farthest(mask):
            return mask & (dist_from_general == jnp.max(jnp.where(mask, dist_from_general, -1)))

        def nearest(mask):
            return mask & (dist_from_general == jnp.min(jnp.where(mask, dist_from_general, reach)))

        fog_goal = farthest(fog) if self.scout_far else nearest(fog)
        open_goal = farthest(open_land) if self.scout_far else nearest(open_land)
        goal = jnp.where(
            jnp.any(enemy_general),
            enemy_general,
            jnp.where(jnp.any(enemy_land), enemy_land, jnp.where(jnp.any(fog), fog_goal, open_goal)),
        )

        dist_to_goal = _bfs_distance(passable, goal)
        direction, neighbor_dist = _best_direction_toward(dist_to_goal, passable)
        advances = neighbor_dist < dist_to_goal
        flat_direction = direction.reshape(-1)

        enemy_general_army = jnp.sum(jnp.where(enemy_general, army, 0))
        killing_move = (
            jnp.any(enemy_general)
            & movable
            & (dist_to_goal == 1)
            & advances
            & (army - 1 > enemy_general_army)
        )
        kill_idx = jnp.argmax(jnp.where(killing_move, mine_army, -1).reshape(-1))

        feed_from_general = (own_general_army >= self.release_threshold) & advances.reshape(-1)[own_general_idx]
        convoy_move = movable & ~own_general & advances
        convoy_score = mine_army.astype(jnp.float32) * self.convoy_weight - dist_to_goal.astype(jnp.float32)
        convoy_idx = jnp.argmax(jnp.where(convoy_move, convoy_score, -1.0).reshape(-1))

        do_kill = jnp.any(killing_move)
        do_feed = (~do_kill) & feed_from_general
        do_convoy = (~do_kill) & (~do_feed) & jnp.any(convoy_move)
        idx = jnp.where(do_kill, kill_idx, jnp.where(do_feed, own_general_idx, convoy_idx))
        should_pass = ~(do_kill | do_feed | do_convoy)
        return jnp.array(
            [should_pass, idx // w, idx % w, flat_direction[idx], do_feed],
            dtype=jnp.int32,
        )


class _FlobotStyleAgent(Agent):
    """Judge-only Flobot-inspired baseline: spread, infiltrate, then end-game push."""

    def __init__(self, *, id: str = "FlobotStyle", early_threshold: int = 12):
        super().__init__(id=id)
        self.early_threshold = int(early_threshold)

    @partial(jax.jit, static_argnums=0)
    def act(self, observation, key):
        del key
        army = observation.armies
        mine = observation.owned_cells
        h, w = army.shape
        reach = jnp.int32(h * w + 7)
        passable = ~(observation.mountains | observation.structures_in_fog)
        movable = mine & (army > 1)
        mine_army = jnp.where(mine, army, 0)

        own_general = mine & observation.generals
        own_general_idx = jnp.argmax(own_general.reshape(-1).astype(jnp.int32))
        own_general_army = jnp.sum(jnp.where(own_general, army, 0))
        dist_from_general = _bfs_distance(passable, own_general)

        enemy_general = observation.opponent_cells & observation.generals
        enemy_land = observation.opponent_cells & ~observation.cities
        visible_enemy = enemy_general | enemy_land
        neutral_city = observation.cities & ~mine & ~observation.opponent_cells
        fog = observation.fog_cells & passable & (dist_from_general < reach)
        open_land = passable & ~mine & (dist_from_general < reach)

        far_fog = fog & (dist_from_general == jnp.max(jnp.where(fog, dist_from_general, -1)))
        border_target = visible_enemy | (fog & (dist_from_general == jnp.min(jnp.where(fog, dist_from_general, reach))))
        economy_target = neutral_city & (dist_from_general == jnp.min(jnp.where(neutral_city, dist_from_general, reach)))
        spread_target = open_land & (dist_from_general == jnp.max(jnp.where(open_land, dist_from_general, -1)))
        goal = jnp.where(
            jnp.any(enemy_general),
            enemy_general,
            jnp.where(
                jnp.any(enemy_land),
                border_target,
                jnp.where(jnp.any(neutral_city) & (own_general_army >= self.early_threshold), economy_target, jnp.where(jnp.any(fog), far_fog, spread_target)),
            ),
        )

        dist_to_goal = _bfs_distance(passable, goal)
        direction, neighbor_dist = _best_direction_toward(dist_to_goal, passable)
        advances = neighbor_dist < dist_to_goal
        flat_direction = direction.reshape(-1)

        enemy_general_army = jnp.sum(jnp.where(enemy_general, army, 0))
        can_end = (
            jnp.any(enemy_general)
            & movable
            & advances
            & ((army - 1) > (enemy_general_army + dist_to_goal))
        )
        end_idx = jnp.argmax(jnp.where(can_end, mine_army - dist_to_goal, -1).reshape(-1))

        early_launch = (own_general_army >= self.early_threshold) & advances.reshape(-1)[own_general_idx]
        spread = movable & advances
        border = mine & (dist_to_goal <= 2)
        spread_score = mine_army.astype(jnp.float32) * jnp.where(border, 2.0, 1.0) - dist_to_goal.astype(jnp.float32)
        spread_idx = jnp.argmax(jnp.where(spread, spread_score, -1.0).reshape(-1))

        do_end = jnp.any(can_end)
        do_early = (~do_end) & early_launch
        do_spread = (~do_end) & (~do_early) & jnp.any(spread)
        idx = jnp.where(do_end, end_idx, jnp.where(do_early, own_general_idx, spread_idx))
        should_pass = ~(do_end | do_early | do_spread)
        return jnp.array(
            [should_pass, idx // w, idx % w, flat_direction[idx], do_early],
            dtype=jnp.int32,
        )


class _HiddenFrontierAgent(Agent):
    def __init__(self, id: str = "Frontier"):
        super().__init__(id=id)

    def act(self, observation, key):
        del key
        valid = compute_valid_move_mask_obs(observation)
        h, w = observation.armies.shape
        positions = jnp.argwhere(valid, size=h * w * 4, fill_value=-1)
        num_valid = jnp.sum(jnp.all(positions >= 0, axis=-1))

        def score_move(idx):
            move = positions[idx]
            ok = jnp.all(move >= 0)
            r, c, d = move[0], move[1], move[2]
            dr = jnp.array([-1, 1, 0, 0], dtype=jnp.int32)[d]
            dc = jnp.array([0, 0, -1, 1], dtype=jnp.int32)[d]
            nr = jnp.clip(r + dr, 0, h - 1)
            nc = jnp.clip(c + dc, 0, w - 1)
            dest_owned = observation.owned_cells[nr, nc]
            dest_opponent = observation.opponent_cells[nr, nc]
            dest_neutral = observation.neutral_cells[nr, nc]
            dest_general = observation.generals[nr, nc] & dest_opponent
            source_army = observation.armies[r, c]
            dest_army = observation.armies[nr, nc]
            can_capture = source_army > dest_army + 1
            value = source_army.astype(jnp.float32)
            value += jnp.where(dest_neutral, 20.0, 0.0)
            value += jnp.where(dest_opponent, 60.0, 0.0)
            value += jnp.where(dest_general, 10000.0, 0.0)
            value = jnp.where(dest_owned, value * 0.1, value)
            value = jnp.where(can_capture & ok, value, -1.0)
            return value

        scores = jax.vmap(score_move)(jnp.arange(h * w * 4))
        best = jnp.argmax(scores)
        move = positions[best]
        should_pass = (num_valid == 0) | (scores[best] < 0)
        return jnp.array([should_pass, move[0], move[1], move[2], 0], dtype=jnp.int32)


def _make_hidden_baseline(name: str, *, player: int):
    if name == "random_low_split":
        return RandomAgent(id=f"RandomLowSplit-{player}", idle_prob=0.03, split_prob=0.10)
    if name == "random_high_split":
        return RandomAgent(id=f"RandomHighSplit-{player}", idle_prob=0.08, split_prob=0.45)
    if name == "expander":
        return ExpanderAgent(id=f"Expander-{player}")
    if name in {"hunter", "hunter_pressure", "hunter_mirror"}:
        return HunterAgent(id=f"Hunter-{player}")
    if name == "fast_pathing":
        return _PathingBaselineAgent(id=f"FastPathing-{player}", release_threshold=6, scout_far=True)
    if name == "near_scout_pathing":
        return _PathingBaselineAgent(id=f"NearScoutPathing-{player}", release_threshold=8, scout_far=False)
    if name == "flobot_style":
        return _FlobotStyleAgent(id=f"FlobotStyle-{player}", early_threshold=12)
    if name == "flobot_fast":
        return _FlobotStyleAgent(id=f"FlobotFast-{player}", early_threshold=8)
    if name == "strongest_frontier":
        return _HiddenFrontierAgent(id=f"Frontier-{player}")
    raise ValueError(f"unknown baseline {name!r}")


def _play_hidden_game(
    candidate_factory,
    *,
    baseline_name: str,
    candidate_player: int,
    seed: int,
    grid_size: int,
    truncation: int,
    pool_size: int,
) -> tuple[bool, int]:
    env = GeneralsEnv(grid_dims=(grid_size, grid_size), truncation=truncation, pool_size=pool_size)
    key = jrandom.PRNGKey(seed)
    pool, state = env.reset(key)
    candidate = candidate_factory()
    baseline = _make_hidden_baseline(baseline_name, player=1 - candidate_player)

    agents = [None, None]
    agents[candidate_player] = candidate
    agents[1 - candidate_player] = baseline
    for agent in agents:
        reset = getattr(agent, "reset", None)
        if callable(reset):
            reset()

    terminated = truncated = False
    turns = 0
    while not (terminated or truncated):
        obs_0 = get_observation(state, 0)
        obs_1 = get_observation(state, 1)
        key, k0, k1 = jrandom.split(key, 3)
        actions = jnp.stack([agents[0].act(obs_0, k0), agents[1].act(obs_1, k1)])
        timestep, state = env.step(state, actions, pool)
        terminated = bool(timestep.terminated)
        truncated = bool(timestep.truncated)
        turns += 1

    return int(timestep.info.winner) == candidate_player, turns


def _evaluate_hidden_agent(
    candidate_factory,
    *,
    games_per_matchup: int,
    seed: int,
    grid_sizes: tuple[int, ...],
    truncation: int,
    baselines: tuple[str, ...],
    pool_size: int,
    speed_weight: float,
) -> dict[str, Any]:
    total = 0
    wins = 0
    speed_credit = 0.0
    by_baseline: dict[str, dict[str, float]] = {}

    for baseline in baselines:
        b_total = 0
        b_wins = 0
        for grid_size in grid_sizes:
            for game_idx in range(games_per_matchup):
                for candidate_player in (0, 1):
                    baseline_offset = sum((idx + 1) * ord(ch) for idx, ch in enumerate(baseline))
                    game_seed = seed + 100003 * game_idx + 1009 * grid_size + 17 * candidate_player
                    game_seed += 7919 * baseline_offset
                    candidate_won, turns = _play_hidden_game(
                        candidate_factory,
                        baseline_name=baseline,
                        candidate_player=candidate_player,
                        seed=game_seed,
                        grid_size=grid_size,
                        truncation=truncation,
                        pool_size=pool_size,
                    )
                    total += 1
                    b_total += 1
                    if candidate_won:
                        wins += 1
                        b_wins += 1
                        speed_credit += max(0.0, (truncation - turns) / truncation)
        by_baseline[baseline] = {
            "games": float(b_total),
            "wins": float(b_wins),
            "win_rate": float(b_wins / b_total if b_total else 0.0),
        }

    win_rate = wins / total if total else 0.0
    speed_tiebreak = speed_credit / total if total else 0.0
    mean_baseline_win_rate = (
        sum(item["win_rate"] for item in by_baseline.values()) / len(by_baseline)
        if by_baseline
        else 0.0
    )
    speed_weight = max(0.0, min(1.0, speed_weight))
    win_weight = 1.0 - speed_weight
    score = 100.0 * (win_weight * mean_baseline_win_rate + speed_weight * speed_tiebreak)
    return {
        "score": score,
        "win_rate": win_rate,
        "mean_baseline_win_rate": mean_baseline_win_rate,
        "speed_tiebreak": speed_tiebreak,
        "win_weight": win_weight,
        "speed_weight": speed_weight,
        "games": float(total),
        "wins": float(wins),
        "by_baseline": by_baseline,
    }


def evaluate(solution_path: str) -> tuple[float, float, str]:
    patch_path = Path(solution_path).resolve()
    work, prep_message = _prepare_candidate(patch_path)
    if work is None:
        message = json.dumps({"status": "invalid", "reason": prep_message}, sort_keys=True)
        return 0.0, 0.0, message

    try:
        candidate_factory = _load_candidate_factory(work)
        timeout_seconds = max(1, _config_int("max_eval_seconds", 240))

        def _raise_timeout(_signum, _frame):
            raise _EvaluationTimeout()

        previous_handler = signal.getsignal(signal.SIGALRM)
        previous_timer = signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
        try:
            signal.signal(signal.SIGALRM, _raise_timeout)
            metrics = _evaluate_hidden_agent(
                candidate_factory,
                games_per_matchup=max(1, _config_int("games_per_matchup", 1)),
                seed=_config_int("arena_seed", 20260608),
                grid_sizes=_config_tuple_int("grid_sizes", (10,)),
                truncation=_config_int("truncation", 180),
                baselines=_config_tuple_str("baselines", DEFAULT_BASELINES),
                pool_size=max(2, _config_int("pool_size", 2)),
                speed_weight=_config_float("speed_weight", 0.25),
            )
        finally:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)
            signal.signal(signal.SIGALRM, previous_handler)
    except _EvaluationTimeout:
        message = json.dumps(
            {
                "status": "timeout",
                "reason": "evaluation exceeded task time budget",
            },
            sort_keys=True,
        )
        return 0.0, 0.0, message
    except Exception as exc:
        message = json.dumps(
            {"status": "error", "reason": type(exc).__name__},
            sort_keys=True,
        )
        return 0.0, 0.0, message

    score = max(0.0, min(100.0, float(metrics["score"])))
    public_metrics = {
        "status": "scored",
        "score": score,
        "win_rate": round(float(metrics["win_rate"]), 4),
        "mean_baseline_win_rate": round(float(metrics["mean_baseline_win_rate"]), 4),
        "speed_tiebreak": round(float(metrics["speed_tiebreak"]), 4),
        "speed_weight": round(float(metrics["speed_weight"]), 4),
        "games": int(metrics["games"]),
        "wins": int(metrics["wins"]),
    }
    return score, score, json.dumps(public_metrics, sort_keys=True)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: evaluator.py /path/to/solution.patch", file=sys.stderr)
        return 2
    score, score_unbounded, message = evaluate(sys.argv[1])
    print(json.dumps({"score": score, "score_unbounded": score_unbounded, "message": message}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
