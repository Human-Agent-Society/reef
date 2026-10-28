"""Reefine's method: answer a person's request with a change to the harness, or learn one skill from failures.

``ReefineMethod.step`` reads its step's workspace. A step with a request
answers it: the coding agent on pi when the recipe gave the method an agent
(:mod:`recipes.beta.reefine.agent`), else the served model in text
(:mod:`recipes.beta.reefine.answer`). A step without a request learns one
skill from its traces, each a failure a client reported. The step leaves the
answer's tree in ``harness/`` and says what it did in plain text: why
nothing changed, the design with its How to use paragraph, and the review.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from recipes.beta.reefine.agent import ReefineAgent, answer_with_agent
from recipes.beta.reefine.answer import EXTENSION_ADAPTER, answer_request, learn_from_failures
from reef.harness.adapters import get_adapter
from reef.train.harness_backend import Method, StepContext, StepResult


class ReefineMethod(Method):
    """Reefine's step: a request answered with a change, or one skill learned from the step's failing traces.

    Values: ``agent`` runs the coding agent on a request step on pi;
    ``None`` answers every request with the served model's text.
    Purpose: ``ReefineRecipe`` names it as the default ``evolution.method``,
    built without an agent, and gives each scenario's backend one with the
    agent it binds there; the backend calls ``step`` once per step.
    """

    def __init__(self, agent: ReefineAgent | None = None) -> None:
        self.agent = agent

    def step(self, context: StepContext) -> StepResult:
        workspace, tools = context.workspace, context.tools
        head = workspace.tree()
        traces = workspace.traces()
        request = workspace.request()
        adapter = str(workspace.host()["adapter"])
        if request is None:
            layout = get_adapter(adapter).layout
            patch = None if not traces else learn_from_failures(head, traces, tools.models, layout)
            if patch is None:
                return StepResult()
            workspace.write_tree(head.apply(patch))
            plural = "" if len(traces) == 1 else "s"
            return StepResult(summary=f"Learned {', '.join(patch.changes)} from {len(traces)} failing trace{plural}.")
        # The agent answers requests on pi alone: its rules and its trials are pi's.
        if self.agent is not None and adapter == EXTENSION_ADAPTER:
            answer = answer_with_agent(workspace, tools, head, request, self.agent)
        else:
            answer = answer_request(head, request, traces, tools, adapter)
        workspace.write_tree(head.apply(answer.patch))
        return StepResult(summary=step_summary(answer.notes), notes=answer.notes, requires=answer.requires)


def step_summary(notes: Mapping[str, Any]) -> str:
    """A request step's notes as plain text for the step pages: why nothing changed, the design with its How to
    use paragraph, and the review; empty for notes that hold none of them."""
    parts: list[str] = []
    reason = notes.get("failure") or notes.get("declined")
    if reason:
        parts.append(f"No change: {reason}")
    if notes.get("design"):
        parts.append(f"Design:\n{notes['design']}")
    review = notes.get("review")
    if isinstance(review, Mapping):
        lines = [f"Review: {review['result']}"]
        for heading, key in (("Covered", "covered"), ("Uncovered", "uncovered"), ("Out of reach here", "limits")):
            points = review.get(key) or []
            if points:
                lines.append(f"{heading}:")
                lines.extend(f"- {point}" for point in points)
        parts.append("\n".join(lines))
    elif notes.get("review_failure"):
        parts.append(
            f"Review: did not run, so nothing checked that the change delivers the request: "
            f"{notes['review_failure']}"
        )
    return "\n\n".join(parts)


__all__ = ["ReefineMethod"]
