"""Shared pieces of the Reefine tests: a canned model, one request, a head tree, request replies, and a step.

``Model`` stands in for a deployment's ``ModelBindings``, and ``Served`` is a
served model binding with canned replies. Both answer a request's plan call
with no step and every other call with the next canned reply. ``ModelServices``
are a step's services around a ``Model``. ``REQUEST`` is
one queued instruction and ``TREE`` a pi head tree with the starter skill
and a rules file. ``designed`` writes a request reply as the Reefine prompt
asks for it: the design, the files, and the requires items last.
``step_workspace`` writes one step's workspace over ``TREE``;
``reefine_step`` runs a whole step of a method through the backend's phases
and keeps its activity.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tests.support.harness_steps import harness_backend

from recipes.beta.reefine.answer import EXTENSION_ADAPTER, Answer, answer_request
from reef.core.job_data import TrajectorySample
from reef.core.training_request import TrainingRequest
from reef.evaluation.harness import NoEvaluationPolicy
from reef.harness.adapters import get_adapter
from reef.harness.rollouts.model_binding import ModelBinding, ModelBindings
from reef.harness.tree import HarnessTree, ValidationResult, validate_tree
from reef.train.harness_backend import AgentOutcome, Method, Services, Trace, Workspace
from reef.train.harness_backend.calls import StepCalls
from reef.train.types import JobData, TrainStepResult

#: What marks the plan call's prompt, the first of a request's calls.
PLAN_MARKER = '"needs_tool": true or false'


class Model:
    """A ModelBindings stand-in: ``served`` answers the canned replies in order, the last one again once they run
    out; a reply that is an exception is raised on its call, and ``failure`` is raised on every call. A request's
    plan call, which comes first, is answered with ``plan`` (no step by default) and consumes no canned reply."""

    def __init__(self, *replies: str | Exception, plan: str = "[]", failure: Exception | None = None) -> None:
        self.replies = list(replies)
        self.plan = plan
        self.failure = failure
        self.calls = 0
        self.answered = 0
        self.prompts: list[str] = []
        self.params_of: list[dict[str, object]] = []
        self.served = self

    @property
    def prompt(self) -> str | None:
        """The write prompt: the second call's on a request (the plan call comes first), the only one otherwise."""
        return self.prompts[1] if len(self.prompts) > 1 else (self.prompts[0] if self.prompts else None)

    def chat(self, messages: Sequence[Mapping[str, str]], **params: object) -> str | None:
        self.calls += 1
        prompt = messages[-1]["content"]
        self.prompts.append(prompt)
        self.params_of.append(dict(params))
        if self.failure is not None:
            raise self.failure
        if PLAN_MARKER in prompt:
            return self.plan
        self.answered += 1
        reply = self.replies[min(self.answered, len(self.replies)) - 1] if self.replies else None
        if isinstance(reply, Exception):
            raise reply
        return reply

    def last_response(self) -> None:
        return None


class ModelServices(Services):
    """The services of a text answer's step around a canned model.

    Values: ``bindings`` answer every model call; ``step_calls`` keeps the
    step's activity and sets no budget; ``descriptor`` is the adapter that
    ``check`` validates a tree for. No tree is scored.
    Purpose: ``answer`` and the answer loop's tests pass it to
    ``answer_request``.
    """

    def __init__(self, models: Model | ModelBindings, adapter: str = EXTENSION_ADAPTER) -> None:
        self.bindings = models
        self.step_calls = StepCalls(0, [])
        self.descriptor = get_adapter(adapter)

    @property
    def models(self) -> Model | ModelBindings:
        return self.bindings

    @property
    def calls(self) -> StepCalls:
        return self.step_calls

    def check(self, tree: HarnessTree) -> ValidationResult:
        return validate_tree(tree, self.descriptor)

    def evaluate(self, tree: HarnessTree, tasks: Sequence[str], *, repeats: int = 1) -> tuple:
        raise RuntimeError("a text answer scores no tree")

    def run_agent(self, prompt: str, *, tree: HarnessTree, model: str) -> AgentOutcome:
        raise RuntimeError("a text answer runs no agent")


class Served(ModelBinding):
    """A served model binding with canned replies, as ``Model`` answers them; ``prompts`` keeps every prompt."""

    def __init__(self, *replies: str) -> None:
        super().__init__(base_url="http://127.0.0.1:9", model="served-model")
        object.__setattr__(self, "replies", list(replies))
        object.__setattr__(self, "prompts", [])

    def chat(self, messages: Sequence[Mapping[str, Any]], *, timeout_s: float | None = None, **params: Any) -> str:
        self.prompts.append(messages[-1]["content"])
        if PLAN_MARKER in messages[-1]["content"]:
            return "[]"
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


#: One queued instruction, as a step's material holds it.
REQUEST = TrainingRequest(
    text="Add a skill that runs the tests before answering", session="session-1", release_id="release-1", id="ask-1"
)

#: The head tree a request step starts from: the starter skill and one rule.
TREE = HarnessTree(
    {
        "rules.md": "Be brief.\n",
        "skills/answer-style/SKILL.md": "---\nname: answer-style\ndescription: answer briefly\n---\n# answer-style\n",
    }
)

DESIGN = "The user wants the tests run before every answer. Trigger: every task; no state. Nothing to set up."

#: A complete review, so a case answers once; the retry after a short review has cases of its own.
REVIEW = {"result": "complete", "covered": ["the tests run first"], "uncovered": []}

#: A review that finds the answer short: it delivers, with one point uncovered.
SHORT = {"result": "partial", "delivers": True, "covered": ["the tests run first"], "uncovered": ["no retry"]}


def designed(*files: Mapping[str, str | None], design: str = DESIGN, requires: list | None = None) -> str:
    """A request reply as the prompt asks for it: the design, the files merged in order, the requires items last."""
    answer: dict[str, object] = {
        "design": design,
        "files": {path: text for part in files for path, text in part.items()},
    }
    if requires is not None:
        answer["requires"] = requires
    return json.dumps(answer)


def skill(name: str, text: str | None = None) -> dict[str, str]:
    """One skill file, its text a SKILL.md with frontmatter unless ``text`` is given."""
    body = f"---\nname: {name}\ndescription: {name} skill\n---\n# {name}\n\ntext" if text is None else text
    return {f"skills/{name}/SKILL.md": body}


def extension(name: str, code: str) -> dict[str, str]:
    """One pi extension file."""
    return {f"extensions/{name}.ts": code}


def answer(
    models: Model | ModelBindings,
    adapter: str = EXTENSION_ADAPTER,
    *,
    tree: HarnessTree = TREE,
    request: TrainingRequest = REQUEST,
    traces: tuple[Trace, ...] = (),
) -> Answer:
    """The text answer to ``request`` on ``adapter``, from the replies of ``models``."""
    return answer_request(tree, request, traces, ModelServices(models, adapter), adapter)


def failure_of(written: Answer) -> str:
    """The reason a request step recorded for producing nothing: an answer with the empty patch carries it."""
    assert written.patch.changes == {}
    return written.notes["failure"]


def step_workspace(
    root: Path,
    *,
    request: TrainingRequest | None = REQUEST,
    samples: Sequence[TrajectorySample] = (),
    adapter: str = EXTENSION_ADAPTER,
) -> Workspace:
    """Step 1's workspace under ``root`` over ``TREE``, as the backend writes it: the request and the traces."""
    return Workspace.create(
        root / "step",
        history_dir=root / "history",
        step=1,
        tree=TREE,
        descriptor=get_adapter(adapter),
        tasks=("[health] reply reef-ok",),
        repeats=0,
        task_directories=False,
        samples=samples,
        request=request,
    )


def reefine_step(
    root: Path, method: Method, data: JobData, models: ModelBindings
) -> tuple[TrainStepResult, tuple[Mapping[str, Any], ...]]:
    """One step of ``method`` through the backend's phases on the pi seed of ``tests.support.harness_steps``, its
    state under ``root``, ``models`` the deployment's, and the evaluation off: a changed tree publishes. Beside
    the result, the step's activity as the request page reads it while the candidate is evaluated; none for a
    skipped step."""
    backend, _ = harness_backend(root, method, evaluation={"policy": "none"})
    backend.models = models
    prepared = backend.prepare_step(data, backend.initial_state(), 0)
    if prepared.candidate is None:
        return TrainStepResult(prepared.state, prepared.metrics), ()
    # A step with a candidate keeps its progress, the request page's view of it, until it settles.
    activity = backend.step_progress.activity
    policy = NoEvaluationPolicy()
    decision = policy.decide(prepared.candidate, policy.evaluate(prepared.candidate))
    return backend.settle_step(prepared, decision), activity


__all__ = [
    "DESIGN",
    "PLAN_MARKER",
    "REQUEST",
    "REVIEW",
    "SHORT",
    "TREE",
    "Model",
    "ModelServices",
    "Served",
    "answer",
    "designed",
    "extension",
    "failure_of",
    "reefine_step",
    "skill",
    "step_workspace",
]
