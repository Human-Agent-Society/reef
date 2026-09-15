"""A generated SPADE environment as a gym task: SPADE's naming, metadata and hint over ``reef.core.tasks.gym``.

The Environment Designer writes one Python class with the Gym interface (SPADE Sec. 3.1): a game, a
simulated tool use setting, anything a class can hold. ``environment_task`` puts it into the task form
every reef consumer reads: the class under ``tests/`` with the shared loader and the replay verifier, the
privileged hint under ``solution/hint.txt`` (Harbor never mounts ``solution/`` for the agent), and skill,
generation, step, index, difficulty and document in ``task.toml`` beside the turn limit and the seed. The
source record of every task is the Designer's generation record, so ``split_generation`` keeps the
environments of one generation call on one side of a split.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from reef.core.tasks import HarborTask, TaskSplit, gym_task, split_by_source
from reef.core.tasks.gym import DEFAULT_IMAGE, DEFAULT_MAX_TURNS

SKILL_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,40}$")


@dataclass(frozen=True)
class GeneratedEnvironment:
    """One environment as the Designer emitted it, with where it came from."""

    code: str
    skill: str
    generation: int
    index: int
    hint: str
    source_record_id: str
    step: int = 0
    difficulty: str | None = None
    document_id: str | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code.strip():
            raise ValueError("environment code must be non-empty text")
        if not isinstance(self.skill, str) or not SKILL_PATTERN.fullmatch(self.skill):
            raise ValueError(f"skill {self.skill!r} must match {SKILL_PATTERN.pattern}")
        for label, number in (
            ("generation", self.generation),
            ("index", self.index),
            ("step", self.step),
            ("seed", self.seed),
        ):
            if isinstance(number, bool) or not isinstance(number, int) or number < 0:
                raise ValueError(f"{label} must be a non-negative integer")
        if not isinstance(self.hint, str) or not self.hint.strip():
            raise ValueError("hint must be non-empty text")
        if not isinstance(self.source_record_id, str) or not self.source_record_id:
            raise ValueError("source_record_id must name the Designer's generation record")
        for label, text in (("difficulty", self.difficulty), ("document_id", self.document_id)):
            if text is not None and (not isinstance(text, str) or not text):
                raise ValueError(f"{label} must be a non-empty string when set")

    @property
    def name(self) -> str:
        """The task directory name: unique per generation and index, readable by skill."""
        return f"game-{self.generation:05d}-{self.index:03d}-{self.skill}"


def environment_task(
    environment: GeneratedEnvironment, *, max_turns: int = DEFAULT_MAX_TURNS, image: str = DEFAULT_IMAGE
) -> HarborTask:
    """The gym task that holds ``environment``: the class, the hint and SPADE's metadata."""
    metadata: dict[str, object] = {
        "skill": environment.skill,
        "generation": environment.generation,
        "step": environment.step,
        "index": environment.index,
    }
    if environment.difficulty is not None:
        metadata["difficulty"] = environment.difficulty
    if environment.document_id is not None:
        metadata["document"] = environment.document_id
    return gym_task(
        name=environment.name,
        code=environment.code,
        seed=environment.seed,
        max_turns=max_turns,
        metadata=metadata,
        solution={"hint.txt": environment.hint.strip() + "\n"},
        source_agent_record_ids=(environment.source_record_id,),
        image=image,
    )


def split_generation(tasks: Sequence[HarborTask], *, eval_fraction: float, seed: int) -> TaskSplit:
    """Split one generation's tasks so that every environment of one Designer call lands in one split."""
    names = [task.name for task in tasks]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f"tasks share a name: {', '.join(repeated)}")
    return split_by_source(
        {task.name: task.source_agent_record_ids for task in tasks}, eval_fraction=eval_fraction, seed=seed
    )
