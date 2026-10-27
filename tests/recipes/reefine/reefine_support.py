"""Shared pieces of the Reefine tests: a canned model, one request, a head tree, and request replies.

``Model`` stands in for a deployment's ``ModelBindings``. It answers a
request's plan call with ``plan`` and every other call with the next canned
reply. ``REQUEST`` is one queued instruction and ``TREE`` a pi head tree with
the starter skill and a rules file. ``designed`` writes a request reply as the
Reefine prompt asks for it: the design, the files, and the requires items
last.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from recipes.beta.reefine.proposer import EXTENSION_ADAPTER, answer_request
from reef.core.batches import TrajectoryItem
from reef.core.training_request import TrainingRequest
from reef.harness.episodes.model_binding import ModelBindings
from reef.harness.tree import HarnessTree
from reef.train.harness_backend.contract import TreeProposal

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

    def note(self, kind: str, text: str, *, failed: bool = False) -> None:
        """The step activity a ModelBinding keeps; the stand-in keeps none."""


#: One queued instruction, as the backend hands it to a proposer that reads requests.
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
    samples: tuple[TrajectoryItem, ...] = (),
) -> TreeProposal:
    """The text proposer's answer to ``request`` on ``adapter``, from the replies of ``models``."""
    return answer_request(tree, request, samples, models, adapter)


def failure_of(proposal: TreeProposal) -> str:
    """The reason a request step recorded for producing nothing: a proposal with the empty patch carries it."""
    assert proposal.patch.changes == {}
    return proposal.notes["failure"]


__all__ = [
    "DESIGN",
    "PLAN_MARKER",
    "REQUEST",
    "REVIEW",
    "SHORT",
    "TREE",
    "Model",
    "answer",
    "designed",
    "extension",
    "failure_of",
    "skill",
]
