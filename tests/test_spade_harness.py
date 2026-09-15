"""The Designer's harness evolution: the prompt as a tree, the rewrite from regret reports, one loop step."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from reef_service._trajectories import recorded_trajectory

from recipes.beta.spade.designer import DESIGNER_RULES_ENTRY, DESIGNER_SYSTEM_ENTRY, DesignerPrompt
from recipes.beta.spade.harness import (
    DESIGNER_SEED,
    DESIGNER_TASK,
    POLICY,
    REPLY_REFUSED,
    ReportedRegretSelection,
    SpadeDesignerHarnessRecipe,
    current_texts,
    never_scored,
    parse_prompt_reply,
    propose_prompt,
    sample_view,
)
from reef.core.batches import TrainingBatch
from reef.core.evaluation import UpdateCandidate
from reef.harness.adapters import get_adapter
from reef.harness.episodes.model_binding import ModelBinding, ModelBindings
from reef.harness.tree.mutations import Mutation
from reef.train.cordis_backend.backend import CordisBackend
from reef.train.cordis_backend.strategies import resolve_episode_scorer, resolve_proposer

MODEL = ModelBinding(base_url="http://localhost:8000", model="qwen3-8b", api_key="dummy")
NEW_RULES = "RULES:\n- The agent has at most {turn_limit} commands.\n- Hide state in files the build wrote."


class ChatStandIn:
    """A served binding whose chat returns a scripted reply and keeps the prompts it was asked."""

    base_url = MODEL.base_url
    model = MODEL.model
    api_key = MODEL.api_key
    api = MODEL.api
    timeout_s = MODEL.timeout_s

    def __init__(self, reply: str | None) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def chat(self, messages: Sequence[Mapping[str, object]], **params: object) -> str:
        self.prompts.append(str(messages[-1]["content"]))
        if self.reply is None:
            raise RuntimeError("the endpoint is down")
        return self.reply

    def compose_nodes(self, descriptor):
        return MODEL.compose_nodes(descriptor)


def rewrite_reply(rules: str = NEW_RULES, system: str | None = None) -> str:
    entries = [{"id": DESIGNER_RULES_ENTRY, "name": "skill", "config": {"name": DESIGNER_RULES_ENTRY, "text": rules}}]
    if system is not None:
        entries.append(
            {"id": DESIGNER_SYSTEM_ENTRY, "name": "skill", "config": {"name": DESIGNER_SYSTEM_ENTRY, "text": system}}
        )
    entries.append({"id": "reef-requests", "name": "skill", "config": {"name": "reef-requests", "text": "no"}})
    return "Here is the rewrite:\n```json\n" + json.dumps(entries) + "\n```"


def designer_payload(instruction: str | None) -> dict[str, object]:
    content = (
        "```json\n"
        + json.dumps(
            {
                "instruction": instruction,
                "environment": {"Dockerfile": "FROM ubuntu:24.04\nRUN apt-get install -y tmux\n"},
                "tests": {"test.sh": "#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n"},
                "solution": {"solve.sh": "#!/bin/sh\ntrue\n"},
                "hint": "look under /var",
            }
        )
        + "\n```"
        if instruction is not None
        else "I cannot write that."
    )
    return {
        "messages": [{"role": "user", "content": "Create ONE Harbor task"}],
        "response": {"choices": [{"message": {"role": "assistant", "content": content}}]},
    }


def proposals(*regrets: float) -> TrainingBatch:
    samples = []
    for index, regret in enumerate(regrets):
        instruction = (
            None if regret < 0 else f"Inspect the service under /var/log and fix its config, task {index}." * 3
        )
        feedback = {"task": f"harbor-00001-{index:03d}", "round": {"generation": 1, "mean_regret": 0.25}}
        samples.append(recorded_trajectory(f"designer-{index}", designer_payload(instruction), regret, feedback))
    return TrainingBatch("designer:harness_evolve:1", tuple(samples))


def nodes_of(prompt: DesignerPrompt) -> tuple[tuple[str, object], ...]:
    return tuple((str(entry["name"]), entry["config"]) for entry in prompt.entries())


def test_the_seed_is_the_designer_prompt_as_two_skill_entries() -> None:
    assert [entry["id"] for entry in DESIGNER_SEED] == [DESIGNER_SYSTEM_ENTRY, DESIGNER_RULES_ENTRY]
    assert current_texts(nodes_of(DesignerPrompt())) == {
        DESIGNER_SYSTEM_ENTRY: DesignerPrompt().system,
        DESIGNER_RULES_ENTRY: DesignerPrompt().rules,
    }
    assert DesignerPrompt(system="s", rules="r").with_entries(DESIGNER_SEED) == DesignerPrompt()


def test_a_sample_view_carries_the_regret_the_feedback_and_the_instruction_or_the_refusal() -> None:
    batch = proposals(0.5, -1.0)
    views = [sample_view(sample) for sample in batch.items]
    assert views[0]["regret"] == 0.5 and views[0]["feedback"]["round"]["mean_regret"] == 0.25
    assert str(views[0]["instruction"]).startswith("Inspect the service under /var/log")
    assert views[1] == {"regret": -1.0, "feedback": views[1]["feedback"], "instruction": REPLY_REFUSED}


def test_the_proposer_updates_the_texts_the_reply_changes_and_nothing_else() -> None:
    served = ChatStandIn(rewrite_reply(system=DesignerPrompt().system))
    mutations = propose_prompt(nodes_of(DesignerPrompt()), proposals(0.5, 0.0).items, ModelBindings(served=served))
    assert mutations == [
        Mutation(
            "update",
            DESIGNER_RULES_ENTRY,
            {"name": "skill", "config": {"name": DESIGNER_RULES_ENTRY, "text": NEW_RULES}},
        )
    ], "an unchanged system text and a reserved id are not proposed"
    prompt = served.prompts[0]
    assert "proposals" in prompt and "TWO NETWORK PHASES" in prompt and "Inspect the service" in prompt
    assert "{turn_limit}" in prompt


@pytest.mark.parametrize("reply", [None, "no json here", "[]", '[{"id": "designer-rules", "config": {"text": ""}}]'])
def test_a_failed_call_or_an_unusable_reply_skips_the_step(reply: str | None) -> None:
    served = ChatStandIn(reply)
    assert propose_prompt(nodes_of(DesignerPrompt()), proposals(0.5).items, ModelBindings(served=served)) is None
    assert propose_prompt(nodes_of(DesignerPrompt()), (), ModelBindings(served=served)) is None, "no samples, no call"
    assert parse_prompt_reply("text [1, 2] text") == {}


def test_the_selection_publishes_every_rewrite_without_an_episode() -> None:
    plugin = ReportedRegretSelection().build(candidate_backend=None)  # type: ignore[arg-type]
    candidate = UpdateCandidate("cand-1")
    evaluation = plugin.evaluate(candidate)
    decision = plugin.decide(candidate, evaluation)
    assert evaluation.metrics == {"episodes": 0, "candidate_id": "cand-1"}
    assert decision.selected and decision.policy == POLICY and decision.evaluation is evaluation
    with pytest.raises(RuntimeError, match="runs no gate episodes"):
        never_scored(DESIGNER_TASK, None)


def test_the_recipe_fills_the_loop_with_the_designer_defaults() -> None:
    settings = {"evolution": {"step_record_dir": "steps"}, "batch_size": 8}
    kwargs = SpadeDesignerHarnessRecipe._recipe_kwargs(settings, {})
    assert kwargs["tasks"] == (DESIGNER_TASK,) or kwargs["tasks"] == [DESIGNER_TASK]
    assert [entry["id"] for entry in kwargs["seed"]] == [DESIGNER_SYSTEM_ENTRY, DESIGNER_RULES_ENTRY]
    assert kwargs["adapter"] == "native" and isinstance(kwargs["candidate_plugin"], ReportedRegretSelection)


def test_one_loop_step_publishes_the_rewrite_and_the_next_proposal_reads_it(tmp_path: Path) -> None:
    seen_nodes: list[tuple[tuple[str, object], ...]] = []
    served = ChatStandIn(rewrite_reply())

    def recording_propose(nodes, samples, models):
        seen_nodes.append(tuple(nodes))
        return propose_prompt(nodes, samples, models)

    binary = tmp_path / "reef-native"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    backend = CordisBackend(
        descriptor=get_adapter("native"),
        propose=resolve_proposer(recording_propose),
        score_episode=resolve_episode_scorer(never_scored),
        tasks=(DESIGNER_TASK,),
        models=ModelBindings(served=served),
        binary=str(binary),
        seed=DESIGNER_SEED,
    )
    prepared = backend.prepare_step(proposals(0.5, 0.0), {}, 0)
    assert prepared.outcome != "skip" and prepared.candidate is not None
    plugin = ReportedRegretSelection().build(backend)
    decision = plugin.decide(prepared.candidate, plugin.evaluate(prepared.candidate))
    result = backend.settle_step(prepared, decision)
    assert result.metrics["selection"]["policy"] == POLICY and result.metrics["traces"] == 2
    second = backend.prepare_step(proposals(0.25), result.state, 1)
    assert second.outcome == "skip", "the same reply changes nothing the second time"
    assert current_texts(seen_nodes[-1])[DESIGNER_RULES_ENTRY] == NEW_RULES
