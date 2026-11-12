"""Reefine's method: answer a person's request with a change to the harness, or learn one skill from failures.

``ReefineMethod.run`` reads its job's workspace. A job with a request
answers it: the coding agent on pi when the job's services hold an agent
(``job.services.agent``, :mod:`recipes.beta.reefine.agent`), else the served
model in text (:mod:`recipes.beta.reefine.answer`). A job without a request learns one
skill from its traces, each a failure a client reported. The job leaves the
answer's tree in ``harness/`` and says what it did in plain text: why
nothing changed, the design with its How to use paragraph, and the review.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Self

from recipes.beta.reefine.agent import ReefineAgent, answer_with_agent
from recipes.beta.reefine.answer import EXTENSION_ADAPTER, answer_request, learn_from_failures
from recipes.beta.reefine.multimodal import MultimodalSettings
from reef.core.plugins import PluginConfigError
from reef.harness.adapters import get_adapter
from reef.train.harness_backend import Method, MethodJob

#: How long one trial of the agent may take when the method's ``trial_timeout_s`` says nothing.
DEFAULT_TRIAL_TIMEOUT_S = 300.0


class ReefineMethod(Method):
    """Reefine's job: a request answered with a change, or one skill learned from the job's failing traces.

    Values: ``trial_timeout_s`` > 0 bounds each trial of the coding agent.
    ``multimodal`` is the gateway for images, embeddings, speech and
    decisions that the agent's trials reach, ``None`` for none. Both come
    from the method's config (``evolution.method.config``): ``trial_timeout_s``
    and ``multimodal`` (``preset``, ``url``, ``api_key`` or ``api_key_env``).
    The method runs the deployment's coding agent (``runs_agent``) on a
    request job on pi, and answers every request with the served model's
    text when the deployment configures no agent (``runs_without_agent``).
    Purpose: a deployment names it in ``evolution.method``; ``ReefineRecipe``
    names it by default. The backend calls ``run`` once per job.
    """

    runs_agent = True
    runs_without_agent = True

    def __init__(
        self, trial_timeout_s: float = DEFAULT_TRIAL_TIMEOUT_S, multimodal: MultimodalSettings | None = None
    ) -> None:
        if isinstance(trial_timeout_s, bool) or not isinstance(trial_timeout_s, (int, float)) or trial_timeout_s <= 0:
            raise ValueError("trial_timeout_s must be a positive number")
        self.trial_timeout_s = float(trial_timeout_s)
        self.multimodal = multimodal

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Self:
        unknown = sorted(str(key) for key in config if key not in ("trial_timeout_s", "multimodal"))
        if unknown:
            raise PluginConfigError(f"ReefineMethod takes trial_timeout_s and multimodal, got {', '.join(unknown)}")
        try:
            # A key that the section names by variable (api_key_env) is read from the process environment.
            multimodal = MultimodalSettings.from_config(config.get("multimodal"), os.environ)
            return cls(config.get("trial_timeout_s", DEFAULT_TRIAL_TIMEOUT_S), multimodal)
        except ValueError as exc:
            raise PluginConfigError(f"ReefineMethod: {exc}") from exc

    def run(self, job: MethodJob) -> None:
        workspace, services = job.workspace, job.services
        head = workspace.tree()
        traces = workspace.traces()
        request = workspace.request()
        adapter = str(workspace.host()["adapter"])
        if request is None:
            layout = get_adapter(adapter).layout
            patch = None if not traces else learn_from_failures(head, traces, services.models, layout)
            if patch is None:
                return
            workspace.write_tree(head.apply(patch))
            plural = "" if len(traces) == 1 else "s"
            workspace.write_output(
                summary=f"Learned {', '.join(patch.changes)} from {len(traces)} failing trace{plural}."
            )
            return
        # The agent answers requests on pi alone: its rules and its trials are pi's.
        execution = services.agent
        if execution is not None and adapter == EXTENSION_ADAPTER:
            # The gateway's key is the configured one, else the served model's when that is the same gateway.
            served = services.models.served
            provider = None if self.multimodal is None else self.multimodal.provider(served.base_url, served.api_key)
            agent = ReefineAgent(execution, self.trial_timeout_s, provider)
            answer = answer_with_agent(workspace, services, head, request, agent)
        else:
            answer = answer_request(head, request, traces, services, adapter)
        workspace.write_tree(head.apply(answer.patch))
        workspace.write_output(summary=job_summary(answer.notes), notes=answer.notes, requires=answer.requires)


def job_summary(notes: Mapping[str, Any]) -> str:
    """A request job's notes as plain text for the job pages: why nothing changed, the design with its How to
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


__all__ = ["DEFAULT_TRIAL_TIMEOUT_S", "ReefineMethod"]
