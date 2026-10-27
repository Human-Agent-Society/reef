"""The tools of one harness step, as the backend implements them.

``StepTools`` checks trees against the adapter, scores trees through the
step's evaluator and the history's score cache, and hands out the step's
budgeted model bindings and call meter.
"""

from __future__ import annotations

from collections.abc import Sequence

from reef.evaluation.harness import EpisodeRecord, HarnessEpisodeEvaluator, History
from reef.harness.adapters.descriptor import AdapterDescriptor
from reef.harness.episodes.model_binding import ModelBindings
from reef.harness.tree import HarnessTree, ValidationResult, validate_tree
from reef.train.harness_backend.calls import StepCalls, budgeted_bindings
from reef.train.harness_backend.contract import ModelCalls, Tools


class StepTools(Tools):
    """The tools of one step.

    Values: ``descriptor`` is the adapter the trees render for. ``evaluator``
    scores trees, ``None`` when the deployment configured no scorer.
    ``history`` is the scenario's history; a tree the method evaluates joins
    it as a ``method`` candidate whose parent is ``head``, the checksum of the
    head tree. ``step`` ≥ 1 is the step's number. ``step_calls`` is the
    step's call meter. ``max_episodes`` ≥ 0 bounds the episodes ``evaluate``
    starts in the step, 0 for no bound. ``episodes_run`` counts the episodes
    started so far, also those of a call whose scorer raised, so a method
    that retries after an error stays within the bound.
    Purpose: the backend makes one per step and passes it to ``Method.step``.
    """

    def __init__(
        self,
        *,
        descriptor: AdapterDescriptor,
        evaluator: HarnessEpisodeEvaluator | None,
        history: History,
        head: str,
        step: int,
        models: ModelBindings,
        step_calls: StepCalls,
        max_episodes: int = 0,
    ) -> None:
        self.descriptor = descriptor
        self.evaluator = evaluator
        self.history = history
        self.head = head
        self.step = step
        self.step_calls = step_calls
        self.budgeted = budgeted_bindings(models, step_calls)
        self.max_episodes = max_episodes
        self.episodes_run = 0

    @property
    def models(self) -> ModelBindings:
        return self.budgeted

    @property
    def calls(self) -> ModelCalls:
        return self.step_calls

    def check(self, tree: HarnessTree) -> ValidationResult:
        return validate_tree(tree, self.descriptor)

    def evaluate(self, tree: HarnessTree, tasks: Sequence[str], *, repeats: int = 1) -> tuple[EpisodeRecord, ...]:
        if self.evaluator is None:
            raise RuntimeError("the deployment configured no evolution.evaluation.scorer, so no tree can be scored")
        if isinstance(tasks, str):
            raise TypeError("tasks must be a sequence of tasks, not one string")
        validation = self.check(tree)
        if not validation.passed:
            raise ValueError(f"the tree fails validation: {validation.message()}")
        # The budget is checked before the tree joins the history, so a refused call adds nothing.
        needed = self.evaluator.episodes_to_run(tree, tasks, repeats=repeats)
        left = self.max_episodes - self.episodes_run
        if self.max_episodes and needed > left:
            raise RuntimeError(f"{needed} episodes must run, and the step's episode budget has {left} left")
        self.history.add_candidate(tree, step=self.step, origin="method", parent=self.head)
        self.step_calls.note("evaluate", f"scoring a tree on {len(tasks)} task(s), {repeats} repeat(s) each")
        started = self.history.next_episode
        try:
            return self.evaluator.episodes(tree, tasks, repeats=repeats, step=self.step)
        finally:
            # Each episode that started counts, also in a batch that raised: a retry spends the same budget.
            self.episodes_run += self.history.next_episode - started


__all__ = ["StepTools"]
