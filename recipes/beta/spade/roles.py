"""The two SPADE roles: each names its Reef service, scenario, model, harness and reward; nothing else names a model.

The Environment Designer's harness is its prompt; the Reasoning Agent's harness is a Harbor agent. The
deployment behind a role's scenario decides how the role evolves, by weights or by harness; the role
only says where its records go and how they are scored. A task's measure, both arms' rewards, lives
here too: it is what the Designer's reward reads.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from recipes.beta.spade.designer import DesignerPrompt, PlayRecord
from reef.harness.client.tasks import DEFAULT_AGENT, TaskPlay

VERSION_KINDS = ("release", "runtime", "model")
#: A refused proposal scores below any measured task: with group centering, 0 would outrank a task whose hint hurt.
REFUSAL_SCORE = -1.0


@dataclass(frozen=True)
class TaskMeasure:
    """A written task after both arms played: the rewards, the regret and the band, and the plays themselves."""

    name: str
    skill: str | None
    task_path: Path
    digest: str
    plain_rewards: tuple[float, ...]
    hint_rewards: tuple[float, ...]
    record: PlayRecord
    plain_plays: tuple[TaskPlay, ...] = ()
    hint_plays: tuple[TaskPlay, ...] = ()

    @property
    def regret(self) -> float:
        return self.record.regret

    @property
    def outcome(self) -> str:
        return self.record.outcome


class DesignerReward(ABC):
    """How a measured task scores the Designer's proposal."""

    @abstractmethod
    def score(self, measure: TaskMeasure) -> float: ...


class AgentReward(ABC):
    """How one played episode scores the Reasoning Agent; None for an episode with no score."""

    @abstractmethod
    def score(self, play: TaskPlay) -> float | None: ...


class Regret(DesignerReward):
    """The hint based regret as measured, negative when the hint hurt."""

    def score(self, measure: TaskMeasure) -> float:
        return measure.regret


class VerifierReward(AgentReward):
    """What the verifier scored, as the task player read it."""

    def score(self, play: TaskPlay) -> float | None:
        return play.reward


regret = Regret()
verifier_reward = VerifierReward()


@dataclass(frozen=True)
class HarborAgent:
    """The Reasoning Agent's harness: a Harbor agent by name or by a full agent spec with placeholders."""

    name: str = "terminus-2"
    spec: Mapping[str, object] | None = None
    host: str | None = None
    concurrency: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("a Harbor agent needs a name")
        if self.spec is not None and not isinstance(self.spec, Mapping):
            raise ValueError("spec must be the agent's configuration object")
        if isinstance(self.concurrency, bool) or not isinstance(self.concurrency, int) or self.concurrency < 1:
            raise ValueError("concurrency must be a positive integer")

    def bound_spec(self) -> dict[str, object]:
        """The agent configuration the task player takes: the spec, or the default agent under this name."""
        if self.spec is not None:
            return dict(self.spec)
        return {**DEFAULT_AGENT, "name": self.name}


@dataclass(frozen=True)
class RoleVersion:
    """Which version of a role a score was measured against: a harness release, a weights load or a fixed model."""

    kind: Literal["release", "runtime", "model"]
    id: str

    def __post_init__(self) -> None:
        if self.kind not in VERSION_KINDS:
            raise ValueError(f"kind must be one of {VERSION_KINDS}")
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("a role version needs an id")


@dataclass(frozen=True)
class Role:
    """One role's binding: the Reef service and scenario its records go to, the served model, its harness and reward."""

    reef_url: str
    scenario: str
    model: str
    harness: DesignerPrompt | HarborAgent
    reward: DesignerReward | AgentReward
    token: str | None = None

    def __post_init__(self) -> None:
        for label, value in (("reef_url", self.reef_url), ("scenario", self.scenario), ("model", self.model)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label} must be non-empty text")
        if isinstance(self.harness, DesignerPrompt) and not isinstance(self.reward, DesignerReward):
            raise ValueError("a Designer role scores its proposals with a DesignerReward")
        if isinstance(self.harness, HarborAgent) and not isinstance(self.reward, AgentReward):
            raise ValueError("a Reasoning Agent role scores its episodes with an AgentReward")
        if not isinstance(self.harness, (DesignerPrompt, HarborAgent)):
            raise ValueError("harness must be a DesignerPrompt or a HarborAgent")
        if self.token is not None and not isinstance(self.token, str):
            raise ValueError("token must be text when set")

    @property
    def is_designer(self) -> bool:
        return isinstance(self.harness, DesignerPrompt)

    @classmethod
    def designer(
        cls,
        reef_url: str,
        scenario: str,
        model: str,
        *,
        prompt: DesignerPrompt | None = None,
        reward: DesignerReward = regret,
        token: str | None = None,
    ) -> Role:
        """The Environment Designer: its prompt as the harness, regret as the reward by default."""
        return cls(reef_url, scenario, model, prompt if prompt is not None else DesignerPrompt(), reward, token)

    @classmethod
    def reasoning_agent(
        cls,
        reef_url: str,
        scenario: str,
        model: str,
        *,
        agent: HarborAgent | None = None,
        reward: AgentReward = verifier_reward,
        token: str | None = None,
    ) -> Role:
        """The Reasoning Agent: a Harbor agent as the harness, the verifier reward by default."""
        return cls(reef_url, scenario, model, agent if agent is not None else HarborAgent(), reward, token)
