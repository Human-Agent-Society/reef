"""TTT-Discover PUCT search algorithm and state-reuse archive.

This module implements the environment/search side of Algorithm 1 and
Appendix A.2 of "Learning to Discover at Test Time". It calls a model function,
evaluates the response, and updates its local archive. Reef-specific behavior
is added by the concrete harness subclass in ``agent.py``.

The ``Scorer`` callable (``str -> ScoredSolution``) is the minimal interface
a task must provide so the PUCT search can drive it. Prompt construction,
state management, and serialization are handled generically by the harness.
The instruction shown to the model comes from the Harbor task's
``instruction.md``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

Scorer = Callable[[str], "ScoredSolution"]


@dataclass(frozen=True)
class ScoredSolution:
    """The result of scoring a solution."""

    solution: str
    reward: float
    value: float
    metrics: dict[str, Any] = field(default_factory=dict)
    output: str = ""


@dataclass
class Candidate:
    candidate_id: str
    solution: str
    reward: float
    value: float
    parent_id: str | None = None
    seed: bool = False
    visits: int = 0
    best_child_reward: float | None = None
    children: set[str] = field(default_factory=set)
    output: str = ""

    @property
    def q(self) -> float:
        return self.value if self.visits == 0 or self.best_child_reward is None else self.best_child_reward


@dataclass(frozen=True)
class RolloutResult:
    parent_id: str
    solution: str
    reward: float
    action: str
    error: str | None = None
    search_value: float | None = None
    agent_record_id: str | None = None


class PUCTArchive:
    """Rank-prior PUCT archive from paper Appendix A.2."""

    def __init__(self, *, exploration: float = 1.0, max_size: int = 1000, top_children: int = 2) -> None:
        if exploration < 0:
            raise ValueError("exploration must be non-negative")
        if max_size < 1 or top_children < 1:
            raise ValueError("max_size and top_children must be positive")
        self.exploration = exploration
        self.max_size = max_size
        self.top_children = top_children
        self.total_expansions = 0
        self._next_id = 0
        self._items: dict[str, Candidate] = {}

    @property
    def candidates(self) -> tuple[Candidate, ...]:
        return tuple(self._items.values())

    def add_seed(self, solution: str, reward: float, value: float | None = None, *, output: str = "") -> Candidate:
        candidate = self._new_candidate(
            solution,
            reward,
            reward if value is None else value,
            parent_id=None,
            seed=True,
            output=output,
        )
        self._items[candidate.candidate_id] = candidate
        return candidate

    def scores(self) -> dict[str, float]:
        if not self._items:
            return {}
        ranked = sorted(self._items.values(), key=lambda item: -item.value)
        count = len(ranked)
        prior_denominator = count * (count + 1) / 2
        non_seed_values = [item.value for item in ranked if not item.seed]
        scale_values = non_seed_values or [item.value for item in ranked]
        scale = max(max(scale_values) - min(scale_values), 1e-6)
        return {
            item.candidate_id: item.q
            + self.exploration
            * scale
            * ((count - rank) / prior_denominator)
            * math.sqrt(1 + self.total_expansions)
            / (1 + item.visits)
            for rank, item in enumerate(ranked)
        }

    def select(self, count: int = 1) -> tuple[Candidate, ...]:
        """Select parents, blocking each selected candidate's full lineage.

        Selection blocks the complete ancestor/descendant lineage. The caller
        must seed enough independent roots for the requested batch, matching
        the reference PUCT sampler.
        """
        if count < 1:
            raise ValueError("count must be positive")
        if not self._items:
            raise RuntimeError("cannot select from an empty archive")
        scores = self.scores()
        selected: list[Candidate] = []
        blocked: set[str] = set()
        while len(selected) < count:
            available = [item for item in self._items.values() if item.candidate_id not in blocked]
            if not available:
                break
            winner = max(available, key=lambda item: (scores[item.candidate_id], item.value))
            selected.append(winner)
            blocked.update(self._lineage(winner.candidate_id))
        return tuple(selected)

    def record_expansion(
        self,
        parent_id: str,
        children: Iterable[tuple[str, float, float, str] | tuple[str, float, float]],
        *,
        failed_rollouts: int = 0,
    ) -> tuple[Candidate, ...]:
        """Record every rollout, then retain the best two children per parent.

        Each child tuple is ``(solution, reward, value)`` or
        ``(solution, reward, value, output)``.
        """
        if failed_rollouts < 0:
            raise ValueError("failed_rollouts must be non-negative")
        parent = self._items[parent_id]
        normalized = [(item[0], float(item[1]), float(item[2]), item[3] if len(item) > 3 else "") for item in children]
        if normalized:
            best_reward = max(item[2] for item in normalized)
            parent.best_child_reward = (
                best_reward if parent.best_child_reward is None else max(parent.best_child_reward, best_reward)
            )

        existing_solutions: set[str] = {item.solution for item in self._items.values()}
        child_candidates: list[Candidate] = []
        for solution, reward, value, output in normalized:
            if solution in existing_solutions:
                continue
            child = self._new_candidate(solution, reward, value, parent_id=parent_id, output=output)
            self._items[child.candidate_id] = child
            parent.children.add(child.candidate_id)
            child_candidates.append(child)
            existing_solutions.add(solution)

        for _ in range(len(normalized) + failed_rollouts):
            current: Candidate | None = parent
            while current is not None:
                current.visits += 1
                current = self._items.get(current.parent_id) if current.parent_id else None
            self.total_expansions += 1

        self._prune_top_children()
        self._prune()
        return tuple(child for child in child_candidates if child.candidate_id in self._items)

    def best(self) -> Candidate:
        if not self._items:
            raise RuntimeError("archive is empty")
        return max(self._items.values(), key=lambda item: (item.value, item.candidate_id))

    def state_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible snapshot."""
        return {
            "exploration": self.exploration,
            "max_size": self.max_size,
            "top_children": self.top_children,
            "total_expansions": self.total_expansions,
            "next_id": self._next_id,
            "candidates": [
                {
                    "candidate_id": candidate.candidate_id,
                    "solution": candidate.solution,
                    "reward": candidate.reward,
                    "value": candidate.value,
                    "parent_id": candidate.parent_id,
                    "seed": candidate.seed,
                    "visits": candidate.visits,
                    "best_child_reward": candidate.best_child_reward,
                    "children": sorted(candidate.children),
                    "output": candidate.output,
                }
                for candidate in self._items.values()
            ],
        }

    def load_state_dict(self, snapshot: Mapping[str, Any]) -> None:
        """Restore a snapshot produced by :meth:`state_dict`."""
        try:
            exploration = float(snapshot["exploration"])
            max_size = int(snapshot["max_size"])
            top_children = int(snapshot["top_children"])
            total_expansions = int(snapshot["total_expansions"])
            next_id = int(snapshot["next_id"])
            rows = snapshot["candidates"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid PUCT archive snapshot metadata") from exc
        if (
            not math.isfinite(exploration)
            or exploration < 0
            or max_size < 1
            or top_children < 1
            or total_expansions < 0
            or next_id < 0
            or not isinstance(rows, Sequence)
            or isinstance(rows, (str, bytes))
        ):
            raise ValueError("invalid PUCT archive snapshot metadata")

        restored: dict[str, Candidate] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("invalid PUCT archive candidate snapshot")
            try:
                candidate_id = row["candidate_id"]
                solution = row["solution"]
                reward = float(row["reward"])
                value = float(row["value"])
                parent_id = row["parent_id"]
                seed = row["seed"]
                visits = int(row["visits"])
                best_child_reward = row["best_child_reward"]
                children = row["children"]
                output = row.get("output", "")
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("invalid PUCT archive candidate snapshot") from exc
            if (
                not isinstance(candidate_id, str)
                or not candidate_id
                or candidate_id in restored
                or not isinstance(solution, str)
                or (parent_id is not None and not isinstance(parent_id, str))
                or not isinstance(seed, bool)
                or visits < 0
                or not math.isfinite(reward)
                or not math.isfinite(value)
                or (
                    best_child_reward is not None
                    and (
                        not isinstance(best_child_reward, (int, float)) or not math.isfinite(float(best_child_reward))
                    )
                )
                or not isinstance(children, Sequence)
                or isinstance(children, (str, bytes))
                or any(not isinstance(child, str) for child in children)
            ):
                raise ValueError("invalid PUCT archive candidate snapshot")
            restored[candidate_id] = Candidate(
                candidate_id=candidate_id,
                solution=solution,
                reward=reward,
                value=value,
                parent_id=parent_id,
                seed=seed,
                visits=visits,
                best_child_reward=(None if best_child_reward is None else float(best_child_reward)),
                children=set(children),
                output=output,
            )

        known_ids = set(restored)
        if not restored:
            raise ValueError("PUCT archive snapshot contains no candidates")
        for candidate in restored.values():
            if candidate.parent_id is not None and candidate.parent_id not in known_ids:
                raise ValueError("PUCT archive snapshot references a missing parent")
            if not candidate.children.issubset(known_ids):
                raise ValueError("PUCT archive snapshot references missing children")
            if any(restored[child].parent_id != candidate.candidate_id for child in candidate.children):
                raise ValueError("PUCT archive snapshot has inconsistent parent/child links")

        self.exploration = exploration
        self.max_size = max_size
        self.top_children = top_children
        self.total_expansions = total_expansions
        self._next_id = next_id
        self._items = restored

    def _new_candidate(
        self,
        solution: str,
        reward: float,
        value: float,
        *,
        parent_id: str | None,
        seed: bool = False,
        output: str = "",
    ) -> Candidate:
        if not math.isfinite(float(reward)) or not math.isfinite(float(value)):
            raise ValueError("candidate reward and search value must be finite")
        candidate = Candidate(
            f"state-{self._next_id}",
            solution,
            float(reward),
            float(value),
            parent_id=parent_id,
            seed=seed,
            output=output,
        )
        self._next_id += 1
        return candidate

    def _lineage(self, candidate_id: str) -> set[str]:
        lineage: set[str] = set()
        current = self._items.get(candidate_id)
        while current is not None:
            lineage.add(current.candidate_id)
            current = self._items.get(current.parent_id) if current.parent_id else None
        stack = [candidate_id]
        while stack:
            item = self._items.get(stack.pop())
            if item is None:
                continue
            lineage.add(item.candidate_id)
            stack.extend(item.children)
        return lineage

    def _prune(self) -> None:
        seeds = [item for item in self._items.values() if item.seed]
        non_seeds = sorted(
            (item for item in self._items.values() if not item.seed),
            key=lambda item: (-item.value, item.candidate_id),
        )
        keep = {item.candidate_id for item in seeds}
        keep.update(item.candidate_id for item in non_seeds[: max(0, self.max_size - len(seeds))])
        for candidate_id in tuple(self._items):
            if candidate_id not in keep:
                del self._items[candidate_id]
        for item in self._items.values():
            item.children.intersection_update(keep)
            if item.parent_id not in keep:
                item.parent_id = None

    def _prune_top_children(self) -> None:
        children_by_parent: dict[str, list[Candidate]] = {}
        for item in self._items.values():
            if item.parent_id is not None:
                children_by_parent.setdefault(item.parent_id, []).append(item)
        remove: set[str] = set()
        for children in children_by_parent.values():
            children.sort(key=lambda item: item.value, reverse=True)
            remove.update(item.candidate_id for item in children[self.top_children :])
        for candidate_id in remove:
            del self._items[candidate_id]
        for item in self._items.values():
            item.children.difference_update(remove)


# ---------------------------------------------------------------------------
# Generic prompt construction
# ---------------------------------------------------------------------------

_CODEBLOCK_RE = re.compile(r"```python\n(?!```)(.*?)(?:\n```)?(?=\n```|$)", re.DOTALL)


def extract_solution(action: str) -> str:
    """Extract the last Python code block from a model response."""
    matches = list(_CODEBLOCK_RE.finditer(action))
    if not matches:
        return ""
    return "```python\n" + matches[-1].group(1).rstrip() + "\n```"


def build_prompt(instruction: str, parent: Candidate | None) -> Sequence[Mapping[str, Any]]:
    """Build the chat messages for a PUCT rollout.

    The prompt is the task instruction followed by context about the current
    selected parent and its score, so the model is asked to improve that
    branch of the frontier.
    """
    content = instruction
    if parent is not None:
        content += f"\n\n## Selected parent (reward={parent.reward:.6f})\n"
        content += f"{parent.solution}\n"
        if parent.output:
            output = parent.output.strip()
            if len(output) > 500:
                output = "\n\n\t\t ...(TRUNCATED)...\n" + output[-500:]
            content += f"\n--- Previous Program Output ---\n{output}\n--- End Output ---\n"
        content += "\nImprove on the above. Try different algorithmic ideas, adjust heuristics or hyperparameters."
    else:
        content += "\n\nWrite code to solve this problem."
    return ({"role": "user", "content": content},)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class _TTTDiscoverHarnessBase:
    """Shared search/evaluation mechanics for the plain and Reef harnesses."""

    def __init__(
        self,
        scorer: Scorer,
        instruction: str,
        *,
        model: str,
        groups_per_step: int = 8,
        rollouts_per_group: int = 64,
        exploration: float = 1.0,
        invalid_reward: float = 0.0,
        max_workers: int = 32,
        request_extra: Mapping[str, Any] | None = None,
        action_from_response: Callable[[Mapping[str, Any]], str] | None = None,
        request_builder: Callable[[str, Sequence[Mapping[str, Any]], Mapping[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        if groups_per_step < 1 or rollouts_per_group < 2:
            raise ValueError("groups_per_step must be positive and rollouts_per_group must be at least two")
        self.scorer = scorer
        self._instruction = instruction
        self.model = model
        self.groups_per_step = groups_per_step
        self.rollouts_per_group = rollouts_per_group
        self.invalid_reward = float(invalid_reward)
        self.max_workers = max_workers
        self.request_extra = dict(request_extra or {})
        self.action_from_response = action_from_response or openai_action
        self.request_builder = request_builder
        self.archive = PUCTArchive(exploration=exploration)
        for _ in range(groups_per_step):
            self.archive.add_seed("", 0.0, 0.0)

    def run_step(self, step: int) -> tuple[RolloutResult, ...]:
        parents = self.archive.select(self.groups_per_step)
        if len(parents) != self.groups_per_step:
            raise RuntimeError(
                f"PUCT could select only {len(parents)} independent lineages for {self.groups_per_step} groups"
            )
        tasks = [
            (
                parent,
                f"tttd-step-{step}-group-{group_index}",
                group_index,
                rollout_index,
            )
            for group_index, parent in enumerate(parents)
            for rollout_index in range(self.rollouts_per_group)
        ]
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(tasks))) as pool:
            futures = [
                pool.submit(
                    self._rollout,
                    parent,
                    comparison_set,
                    step,
                    group_index,
                    rollout_index,
                )
                for parent, comparison_set, group_index, rollout_index in tasks
            ]
            all_results = tuple(future.result() for future in futures)

        for group_index, parent in enumerate(parents):
            start = group_index * self.rollouts_per_group
            results = all_results[start : start + self.rollouts_per_group]
            valid = [
                (result.solution, result.reward, result.search_value)
                for result in results
                if result.solution and result.search_value is not None
            ]
            self.archive.record_expansion(
                parent.candidate_id,
                valid,
                failed_rollouts=len(results) - len(valid),
            )
        return all_results

    def run(self, steps: int = 50) -> Candidate:
        if steps < 1:
            raise ValueError("steps must be positive")
        for step in range(steps):
            self.run_step(step)
        return self.archive.best()

    def _rollout(
        self, parent: Candidate, comparison_set: str, step: int, group_index: int, _rollout_index: int
    ) -> RolloutResult:
        raise NotImplementedError

    def _request_payload(self, parent: Candidate) -> dict[str, Any]:
        real_parent = parent if parent.solution else None
        messages = build_prompt(self._instruction, real_parent)
        if self.request_builder is not None:
            return self.request_builder(self.model, messages, self.request_extra)
        return {
            "model": self.model,
            "messages": messages,
            "temperature": 1.0,
            **self.request_extra,
        }

    def _evaluate_response(self, parent: Candidate, response: Mapping[str, Any]) -> RolloutResult:
        action = ""
        error: str | None = None
        solution = ""
        reward = self.invalid_reward
        search_value: float | None = None
        try:
            action = self.action_from_response(response)
            solution = extract_solution(action)
            if not solution:
                raise ValueError("response does not contain a Python code block")
            scored = self.scorer(solution)
            reward = float(scored.reward)
            search_value = float(scored.value)
            if not math.isfinite(reward) or not math.isfinite(search_value):
                raise ValueError("evaluator returned a non-finite reward")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            solution = ""
            reward = self.invalid_reward
            search_value = None
        return RolloutResult(parent.candidate_id, solution, reward, action, error, search_value)


class TTTDiscoverHarness(_TTTDiscoverHarnessBase):
    """A normal harness that calls a model and keeps evaluation local."""

    def __init__(
        self,
        generate: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        scorer: Scorer,
        instruction: str,
        *,
        model: str,
        groups_per_step: int = 8,
        rollouts_per_group: int = 64,
        exploration: float = 1.0,
        invalid_reward: float = 0.0,
        max_workers: int = 32,
        request_extra: Mapping[str, Any] | None = None,
        action_from_response: Callable[[Mapping[str, Any]], str] | None = None,
        request_builder: Callable[[str, Sequence[Mapping[str, Any]], Mapping[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(
            scorer,
            instruction,
            model=model,
            groups_per_step=groups_per_step,
            rollouts_per_group=rollouts_per_group,
            exploration=exploration,
            invalid_reward=invalid_reward,
            max_workers=max_workers,
            request_extra=request_extra,
            action_from_response=action_from_response,
            request_builder=request_builder,
        )
        self.generate = generate

    def _rollout(
        self,
        parent: Candidate,
        _comparison_set: str,
        _step: int,
        _group_index: int,
        _rollout_index: int,
    ) -> RolloutResult:
        response = self.generate(self._request_payload(parent))
        return self._evaluate_response(parent, response)


def openai_action(response: Mapping[str, Any]) -> str:
    """Extract assistant text from an OpenAI chat-completions response."""
    choices = response.get("choices")
    if not isinstance(choices, Sequence) or not choices:
        raise ValueError("response has no choices")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise ValueError("response choice is not an object")
    message = choice.get("message")
    if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
        raise ValueError("response has no assistant content")
    return message["content"]


class TTTDChatRequestBuilder:
    """Describe TTTD sampling through Reef's OpenAI-compatible chat route.

    Prompt rendering and exact token/log-prob capture belong to the configured
    inference backend. The harness supplies chat semantics only: Reef's weight
    surface addresses the request to whatever adapter the deployment serves,
    so no caller can sample the frozen base by forgetting to name it.
    """

    def __init__(
        self,
        *,
        max_new_tokens: int = 26_000,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        enable_thinking: bool = True,
    ) -> None:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if top_k != -1 and top_k < 1:
            raise ValueError("top_k must be -1 or positive")
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._top_k = top_k
        self._enable_thinking = enable_thinking

    def __call__(
        self,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        request_extra: Mapping[str, Any],
    ) -> dict[str, Any]:
        if "lora_path" in request_extra:
            raise ValueError("lora_path belongs to Reef's weight surface, not to a harness request")
        payload: dict[str, Any] = {
            "model": model,
            "messages": [dict(message) for message in messages],
            "temperature": self._temperature,
            "top_p": self._top_p,
            "top_k": self._top_k,
            "max_completion_tokens": self._max_new_tokens,
            "chat_template_kwargs": {"enable_thinking": self._enable_thinking},
            "stream": False,
        }
        payload.update(dict(request_extra))
        return payload
