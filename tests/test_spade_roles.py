"""The two SPADE roles: what each names, the rewards they carry, and what a role refuses."""

from __future__ import annotations

from pathlib import Path

import pytest

from recipes.beta.spade.designer import DesignerPrompt, PlayRecord
from recipes.beta.spade.roles import (
    REFUSAL_SCORE,
    AgentReward,
    DesignerReward,
    HarborAgent,
    Regret,
    Role,
    RoleVersion,
    TaskMeasure,
    VerifierReward,
    regret,
    verifier_reward,
)
from reef.harness.client.tasks import DEFAULT_AGENT, TaskPlay


def play(reward: float | None) -> TaskPlay:
    return TaskPlay(Path("/tasks/t"), "t", "e", reward, {}, "", ("rec-1",), 0, (), None)


def measure(without: float, with_hint: float) -> TaskMeasure:
    record = PlayRecord(name="harbor-00001-000", return_without_hint=without, return_with_hint=with_hint)
    return TaskMeasure(
        "harbor-00001-000", None, Path("/tasks/harbor-00001-000"), "ab" * 32, (without,), (with_hint,), record
    )


def test_regret_is_the_measured_regret_negative_when_the_hint_hurt() -> None:
    assert isinstance(regret, Regret) and isinstance(regret, DesignerReward)
    assert regret.score(measure(0.25, 0.75)) == 0.5 and regret.score(measure(0.75, 0.25)) == -0.5
    assert REFUSAL_SCORE == -1.0 and measure(0.75, 0.25).outcome == "frontier"


def test_the_verifier_reward_is_what_the_task_player_read() -> None:
    assert isinstance(verifier_reward, VerifierReward) and isinstance(verifier_reward, AgentReward)
    assert verifier_reward.score(play(1.0)) == 1.0 and verifier_reward.score(play(None)) is None


def test_a_harbor_agent_binds_the_default_agent_under_its_name_unless_given_a_spec() -> None:
    assert HarborAgent().bound_spec() == dict(DEFAULT_AGENT)
    assert HarborAgent(name="codex").bound_spec() == {**DEFAULT_AGENT, "name": "codex"}
    spec = {"import_path": "my.agents:Agent", "kwargs": {"api_base": "{base_url}"}}
    assert HarborAgent(name="mine", spec=spec).bound_spec() == spec
    with pytest.raises(ValueError, match="needs a name"):
        HarborAgent(name=" ")
    with pytest.raises(ValueError, match="concurrency must be a positive integer"):
        HarborAgent(concurrency=0)


def test_the_designer_and_reasoning_agent_roles_carry_their_harness_and_reward() -> None:
    designer = Role.designer("http://127.0.0.1:8901", "designer", "strong", token="t")
    assert designer.is_designer and designer.harness == DesignerPrompt() and designer.reward is regret
    agent = Role.reasoning_agent("http://127.0.0.1:8900", "spade", "Qwen/Qwen3-8B")
    assert not agent.is_designer and agent.harness == HarborAgent() and agent.reward is verifier_reward
    prompt = DesignerPrompt(system="You write tasks.")
    assert Role.designer("http://127.0.0.1:8901", "designer", "strong", prompt=prompt).harness is prompt


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"scenario": ""}, "scenario must be non-empty text"),
        ({"model": " "}, "model must be non-empty text"),
        ({"harness": "terminus-2"}, "harness must be a DesignerPrompt or a HarborAgent"),
        ({"reward": regret}, "a Reasoning Agent role scores its episodes with an AgentReward"),
        ({"harness": DesignerPrompt()}, "a Designer role scores its proposals with a DesignerReward"),
        ({"token": 5}, "token must be text"),
    ],
)
def test_a_role_that_names_too_little_is_refused(fields: dict[str, object], message: str) -> None:
    values: dict[str, object] = {
        "reef_url": "http://127.0.0.1:8900",
        "scenario": "spade",
        "model": "m",
        "harness": HarborAgent(),
        "reward": verifier_reward,
    }
    values.update(fields)
    with pytest.raises(ValueError, match=message):
        Role(**values)  # type: ignore[arg-type]


def test_a_role_version_names_a_release_a_runtime_load_or_a_fixed_model() -> None:
    assert RoleVersion("release", "abc").kind == "release"
    with pytest.raises(ValueError, match="kind must be one of"):
        RoleVersion("commit", "abc")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="needs an id"):
        RoleVersion("runtime", "")
