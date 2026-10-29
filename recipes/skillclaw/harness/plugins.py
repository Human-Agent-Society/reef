"""The plugin classes skillclaw.yaml names: ``evolution.method`` and ``evolution.evaluation.scorer``.

``NightMethod`` runs the night of ``skillclaw.py`` and then the probe
rollouts; ``ProbeScorer`` grades the probes with ``skillclaw.py``'s grader.
"""

from __future__ import annotations

from reef.core.job_data import TrajectorySample
from reef.evaluation.harness import RolloutScorer
from reef.harness.rollouts.run import RolloutResult
from reef.train.harness_backend import Method, StepContext, StepResult

from . import skillclaw


class NightMethod(Method):
    """The sealed night flow (:func:`skillclaw.propose`) as one step: the day's traffic in, one patch of the pool out.

    Values: the method holds no state, so every instance behaves the same.
    Purpose: skillclaw.yaml names it in ``evolution.method``; the harness
    backend calls ``step`` once per night.

    The night reads the ATIF trajectories of the step's traffic traces and
    writes its patch to the workspace. The probe rollouts then score the
    served pool and the night's pool on the evaluation tasks, and the step's
    notes keep both as ``probe_scores``: each rollout's value, or
    ``"failed"``. They decide nothing: skillclaw.yaml's ``policy: none``
    keeps every night that changes the pool, and the next day's traffic measures it. A night that changes
    nothing leaves the served pool, and the step skips.
    """

    def step(self, context: StepContext) -> StepResult:
        workspace, tools = context.workspace, context.tools
        samples = tuple(TrajectorySample(trace.trajectory()) for trace in workspace.traces())
        served = workspace.tree()
        patch = skillclaw.propose(served.files, samples, tools.models)
        if patch is None:
            return StepResult(f"the night over {len(samples)} trace(s) changed no skill")
        night = served.apply(patch)
        workspace.write_tree(night)
        summary = f"the night wrote {', '.join(sorted(patch.changes))}"
        if not tools.check(night).passed:
            # The backend skips the step and records the problems; a pool that fails its checks runs no probe.
            return StepResult(summary)
        tasks = workspace.tasks()
        probes = {
            side: [record.score.to_json() for record in tools.evaluate(tree, tasks)]
            for side, tree in (("served", served), ("night", night))
        }
        return StepResult(summary, {"probe_scores": probes})


class ProbeScorer(RolloutScorer):
    """Exact last-line grading of the probe rollouts (:func:`skillclaw.evaluate`)."""

    def score(self, task: str, result: RolloutResult) -> float:
        return skillclaw.evaluate(task, result)
