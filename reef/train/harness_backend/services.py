"""The services of one harness step, as the backend implements them.

``StepServices`` checks trees against the adapter, scores trees through the
step's evaluator and the history's score cache, runs the deployment's coding
agent in the step's workspace, and hands out the step's budgeted model
bindings and call meter.
"""

from __future__ import annotations

from collections.abc import Sequence

from reef.evaluation.harness import HarnessTaskEvaluator, History, RunRecord
from reef.harness.adapters.descriptor import AdapterDescriptor
from reef.harness.gateway import ModelCalls
from reef.harness.tasks.model_binding import ModelBindings
from reef.harness.tree import HarnessTree, ValidationResult, validate_tree
from reef.train.harness_backend.agent.run import AgentExecution, run_agent
from reef.train.harness_backend.calls import StepCalls, budgeted_bindings
from reef.train.harness_backend.contract import AgentOutcome, Services
from reef.train.harness_backend.workspace import Workspace


class StepServices(Services):
    """The services of one step.

    Values: ``workspace`` is the step's workspace. ``descriptor`` is the
    adapter the trees render for. ``evaluator`` scores trees, ``None`` when
    the deployment configured no scorer. ``agent`` is the deployment's
    coding agent, ``None`` when it configured none or the method does not
    run it. ``history`` is the scenario's history; a tree the method
    evaluates joins it as a ``method`` candidate of the workspace's step
    whose parent is ``head``, the checksum of the head tree. ``step_calls``
    is the step's call meter. ``max_runs`` ≥ 0 bounds the runs
    ``evaluate`` starts in the step, 0 for no bound. ``runs_run`` counts
    the runs started so far, also those of a call whose scorer raised,
    so a method that retries after an error stays within the bound.
    Purpose: the backend makes one per step and passes it to ``Method.step``.
    """

    def __init__(
        self,
        *,
        workspace: Workspace,
        descriptor: AdapterDescriptor,
        evaluator: HarnessTaskEvaluator | None,
        history: History,
        head: str,
        models: ModelBindings,
        step_calls: StepCalls,
        max_runs: int = 0,
        agent: AgentExecution | None = None,
    ) -> None:
        self.workspace = workspace
        self.descriptor = descriptor
        self.evaluator = evaluator
        self.history = history
        self.head = head
        self.step_calls = step_calls
        self.budgeted = budgeted_bindings(models, step_calls)
        self.max_runs = max_runs
        self.runs_run = 0
        self.agent = agent

    @property
    def models(self) -> ModelBindings:
        return self.budgeted

    @property
    def calls(self) -> ModelCalls:
        return self.step_calls

    def check(self, tree: HarnessTree) -> ValidationResult:
        return validate_tree(tree, self.descriptor)

    def evaluate(self, tree: HarnessTree, tasks: Sequence[str], *, repeats: int = 1) -> tuple[RunRecord, ...]:
        if self.evaluator is None:
            raise RuntimeError("the deployment configured no evolution.evaluation.scorer, so no tree can be scored")
        if isinstance(tasks, str):
            raise TypeError("tasks must be a sequence of tasks, not one string")
        validation = self.check(tree)
        if not validation.passed:
            raise ValueError(f"the tree fails validation: {validation.message()}")
        # The budget is checked before the tree joins the history, so a refused call adds nothing.
        needed = self.evaluator.runs_to_run(tree, tasks, repeats=repeats)
        left = self.max_runs - self.runs_run
        if self.max_runs and needed > left:
            raise RuntimeError(f"{needed} runs must run, and the step's run budget has {left} left")
        self.history.add_candidate(tree, step=self.workspace.step, origin="method", parent=self.head)
        self.step_calls.note("evaluate", f"scoring a tree on {len(tasks)} task(s), {repeats} repeat(s) each")
        started = self.history.next_run
        try:
            return self.evaluator.runs(tree, tasks, repeats=repeats, step=self.workspace.step)
        finally:
            # Each run that started counts, also in a batch that raised: a retry spends the same budget.
            self.runs_run += self.history.next_run - started

    def run_agent(self, prompt: str, *, tree: HarnessTree, model: str) -> AgentOutcome:
        if self.agent is None:
            raise RuntimeError("no coding agent runs for this method: it needs runs_agent and evolution.agent")
        return run_agent(self.agent, self.workspace, self, prompt, tree=tree, model=model)


__all__ = ["StepServices"]
