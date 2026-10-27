"""The classes the serve files name: ``evolution.method`` and ``evolution.evaluation.scorer``.

``RulesMethod`` is the pi variant's method and ``NativeMethod`` the native
variant's. A step of either one shows the served model the step's failing
traces in one call and writes the change the model answers into the step's
workspace. The scorers grade an episode's final answer with the grader of
``evolution.py``. These classes import Reef, so they live apart from
``evolution.py``, which ``run.py`` imports with only reef-client installed.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from string import Template

from reef.core.prompts import untrusted_text
from reef.evaluation.harness import EpisodeScorer
from reef.harness.episodes.model_binding import ModelBindingError
from reef.harness.episodes.run import EpisodeResult
from reef.harness.tree import TreePatch
from reef.train.harness_backend import Method, StepContext, StepResult, Trace

from . import evolution, native_evolution

#: The prompt of ``RulesMethod``: ``$traces``, ``$request`` and ``$rules`` take the step's values.
RULES_PROMPT = Template(Path(__file__).with_name("rules_prompt.md").read_text(encoding="utf-8"))
#: The limits of the one model call of a step. A stalled endpoint holds the training thread for the whole timeout
#: before the step skips. Raise both for a local thinking model, which spends its reply budget on reasoning first.
MAX_TOKENS = 2048
TIMEOUT_S = 120.0
#: The text of one fenced block in a reply.
FENCED_BLOCK = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)


def failing(traces: Sequence[Trace]) -> list[Trace]:
    """The traces scored below 1.0, the score of an exact answer; a trace without a score is not one of them."""
    return [trace for trace in traces if trace.score is not None and trace.score < 1.0]


def failures_text(traces: Sequence[Trace]) -> str:
    """The traces as a prompt shows them: each task, the agent's reply, its score and the feedback, fenced as data.

    Traces are client text: the fence keeps their text from speaking as the prompt.
    """
    views = [
        {"task": trace.task, "reply": trace.reply, "score": trace.score, "feedback": trace.feedback}
        for trace in traces
    ]
    return untrusted_text(json.dumps(views, indent=2, ensure_ascii=False), "failing traces")


class RulesMethod(Method):
    """The pi variant's method: one model call reads the step's failing traces and rewrites ``rules.md``.

    Values: the method holds no state, so every instance behaves the same.
    Purpose: ``serve.yaml`` and ``deployment.yaml`` name it in
    ``evolution.method``; the harness backend calls ``step`` once per step.

    ``step`` shows the served model the failing traces, the person's request
    when the step answers one (``POST /reef/train``), and the current
    ``rules.md``. The rules the model answers become the new ``rules.md``.
    A step with neither a failing trace nor a request, or whose model call
    fails, leaves the tree unchanged: the step skips, and its summary says why.
    """

    def step(self, context: StepContext) -> StepResult:
        workspace, tools = context.workspace, context.tools
        traces = failing(workspace.traces())
        request = workspace.request()
        if not traces and request is None:
            return StepResult("no failing trace and no request to learn from")
        tree = workspace.tree()
        prompt = RULES_PROMPT.substitute(
            traces=failures_text(traces),
            request="(none)" if request is None else untrusted_text(request.text, "request"),
            rules=tree.files.get("rules.md", "(none yet)"),
        )
        try:
            reply = tools.models.served.chat(
                [{"role": "user", "content": prompt}], max_tokens=MAX_TOKENS, timeout_s=TIMEOUT_S
            )
        except ModelBindingError as exc:
            return StepResult(f"the model call failed: {exc}")
        # The new rules are the reply's last fenced block, or the whole reply when it fenced nothing.
        blocks = FENCED_BLOCK.findall(reply)
        rules = (blocks[-1] if blocks else reply).strip()
        if not rules:
            return StepResult("the model replied with no rules")
        workspace.write_tree(tree.apply(TreePatch({"rules.md": rules + "\n"})))
        summary = f"rewrote rules.md from {len(traces)} failing trace(s)"
        if request is not None:
            summary += " and the request"
        return StepResult(summary, {"failing_traces": len(traces)})


class NativeMethod(Method):
    """The native variant's method: one model call turns the step's failing traces into one change of the loop.

    Values: the method holds no state, so every instance behaves the same.
    Purpose: ``serve-native.yaml`` names it in ``evolution.method``; the
    harness backend calls ``step`` once per step.

    ``step`` shows the served model the failing traces and the tree's skill
    and native files. The change the model answers is a skill, a native tool,
    a native hook, a native agent or the main graph (``native_evolution``),
    and it is written into the tree. The method reads no request. A step
    without a failing trace, whose model call fails, or whose reply holds no
    usable change leaves the tree unchanged: the step skips, and its summary
    says why.
    """

    def step(self, context: StepContext) -> StepResult:
        workspace, tools = context.workspace, context.tools
        traces = failing(workspace.traces())
        if not traces:
            return StepResult("no failing trace to learn from")
        tree = workspace.tree()
        prompt = native_evolution.change_prompt(tree, failures_text(traces))
        try:
            reply = tools.models.served.chat(
                [{"role": "user", "content": prompt}], max_tokens=MAX_TOKENS, timeout_s=TIMEOUT_S
            )
        except ModelBindingError as exc:
            return StepResult(f"the model call failed: {exc}")
        patch = native_evolution.patch_of(tree, reply)
        if patch is None:
            return StepResult("the reply held no usable change")
        workspace.write_tree(tree.apply(patch))
        return StepResult(
            f"wrote {', '.join(sorted(patch.changes))} from {len(traces)} failing trace(s)",
            {"failing_traces": len(traces)},
        )


class AnswerScorer(EpisodeScorer):
    """``evolution.evaluate``: exact last-line answer grading of the episode's final assistant text."""

    def score(self, task: str, result: EpisodeResult) -> float:
        return evolution.evaluate(task, result)


class NativeAnswerScorer(EpisodeScorer):
    """``native_evolution.evaluate``: the same grading over the native-jsonl trajectory."""

    def score(self, task: str, result: EpisodeResult) -> float:
        return native_evolution.evaluate(task, result)
