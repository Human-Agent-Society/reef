"""Shared pieces of the harness method tests: canned models, one request, head trees, and a Meta-Harness setup.

Reefine: ``Model`` stands in for a deployment's ``ModelBindings``. It answers
a request's plan call with ``plan`` and every other call with the next canned
reply. ``REQUEST`` is one queued instruction and ``TREE`` a pi head tree with
the starter skill and a rules file. ``designed`` writes a request reply as the
Reefine prompt asks for it: the design, the files, and the requires items
last.

Meta-Harness: ``QueueChat`` is the proposer model, ``proposal_reply`` one
reply it writes, and ``meta_harness_sections`` the recipe section over the
fake pi binary of ``tests.support.harness_steps``, scored by ``score_marker``.
``report_once`` records one scored trace, enough to wake one step.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tests.support._strategies import resolve_episode_scorer
from tests.support._trajectories import recorded_trajectory

from reef.core import AgentRecord, RequestType
from reef.core.job_data import TrajectorySample
from reef.core.training_request import TrainingRequest
from reef.core.trajectories import recorded_payload
from reef.harness.episodes.model_binding import ModelBinding, ModelBindings
from reef.harness.episodes.run import EpisodeResult
from reef.harness.tree import HarnessTree
from reef.inference.http import InferenceProxyRuntime
from reef.recipe.reefine.proposer import EXTENSION_ADAPTER, answer_request
from reef.scenario.scenario import Scenario
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
    samples: tuple[TrajectorySample, ...] = (),
) -> TreeProposal:
    """The text proposer's answer to ``request`` on ``adapter``, from the replies of ``models``."""
    return answer_request(tree, request, samples, models, adapter)


def failure_of(proposal: TreeProposal) -> str:
    """The reason a request step recorded for producing nothing: a proposal with the empty patch carries it."""
    assert proposal.patch.changes == {}
    return proposal.notes["failure"]


# -- Meta-Harness ------------------------------------------------------------------------------------------------

#: What the scorer looks for in the rules an episode ran with.
MARKER = "META-HARNESS-WINNER"
#: The seed tree, and two improvements of it that carry the marker.
SEED = HarnessTree({"rules.md": "Answer carefully.\n"})
IMPROVED = HarnessTree({"rules.md": f"Answer carefully. {MARKER}\n"})
ALTERNATE = HarnessTree({"rules.md": f"Answer carefully. {MARKER} Check twice.\n"})
TASK = "Solve the supplied task."
#: One failing trace of the task, with its report's feedback.
SAMPLE = recorded_trajectory(
    "trace-1",
    {
        "messages": [{"role": "user", "content": TASK}],
        "response": {"choices": [{"message": {"role": "assistant", "content": "attempt"}}]},
    },
    0.0,
    feedback="the attempt failed",
)


class QueueChat(ModelBinding):
    """The proposer model: it answers the canned replies in order and keeps every prompt it was asked."""

    def __init__(self, *replies: str) -> None:
        super().__init__(base_url="http://127.0.0.1:9", model="proposer", api_key="dummy")
        object.__setattr__(self, "replies", list(replies))
        object.__setattr__(self, "prompts", [])

    def chat(self, messages, **params):
        self.prompts.append(messages[-1]["content"])
        return self.replies.pop(0)


def proposal_reply(parent: HarnessTree, tree: HarnessTree, *, hypothesis: str = "improve rules") -> str:
    """The proposer's reply: ``tree`` whole, branched from ``parent``."""
    return json.dumps(
        {
            "parent_id": parent.checksum(),
            "hypothesis": hypothesis,
            "changes": "revise the general instruction",
            "files": dict(tree.files),
        }
    )


def score_marker(task: str, result: EpisodeResult) -> float:
    """1.0 when the episode's rendered rules carry ``MARKER``."""
    last = result.trajectory[-1] if result.trajectory else {}
    return 1.0 if MARKER in str(last.get("rules", "")) else 0.0


def served_runtime(api: str = "openai") -> InferenceProxyRuntime:
    return InferenceProxyRuntime(model_path="served-model", base_url="http://127.0.0.1:9", api_key="dummy", api=api)


def meta_harness_sections(
    tmp_path: Path, *, scorer: Any = score_marker, adapter: str = "pi", **method: Any
) -> dict[str, Any]:
    """A Meta-Harness recipe section over the fake pi binary in ``tmp_path``, the seed written beside it."""
    seed = tmp_path / "seed"
    if not seed.exists():
        SEED.write(seed)
    return {
        "implementation": "recipes.meta_harness.recipe:MetaHarnessRecipe",
        "model": {"path": "served-model"},
        "evolution": {
            "adapter": adapter,
            "binary": str(tmp_path / "fake-pi"),
            "evaluate": resolve_episode_scorer(scorer),
            "tasks": [TASK],
            "seed": str(seed),
            "meta_harness": dict(method),
            "batch_size": 1,
        },
    }


def report_once(scenario: Scenario, name: str, suffix: str) -> None:
    """One recorded trace and its report: enough to wake one step."""
    scenario.records.append_result(
        AgentRecord.create(
            scenario=name,
            request_type=RequestType.INFERENCE,
            payload=recorded_payload(SAMPLE),
            agent_record_id=f"i{suffix}",
        )
    )
    scenario.records.append_result(
        AgentRecord.create(
            scenario=name,
            request_type=RequestType.REPORT,
            payload={"score": 0.0, "feedback": SAMPLE.metadata.get("feedback"), "references": [f"i{suffix}"]},
            agent_record_id=f"r{suffix}",
        )
    )


__all__ = [
    "ALTERNATE",
    "DESIGN",
    "IMPROVED",
    "MARKER",
    "PLAN_MARKER",
    "REQUEST",
    "REVIEW",
    "SAMPLE",
    "SEED",
    "SHORT",
    "TASK",
    "TREE",
    "Model",
    "QueueChat",
    "answer",
    "designed",
    "extension",
    "failure_of",
    "meta_harness_sections",
    "proposal_reply",
    "report_once",
    "score_marker",
    "served_runtime",
    "skill",
]
