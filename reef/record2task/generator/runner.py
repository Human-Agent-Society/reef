"""Run the episodes of one saved task: stage it, play one arm, and keep what each episode returned.

The task store holds the task; the executor holds a staged copy of it. Staging the same task again returns
the staged copy, so a play that a stop interrupted can stage and play again, and a generator service with an
empty tasks root can still play a saved task. Staged copies stay after a play: the generator service checks
new tasks for duplicates against them.

The agent's model calls go through Reef's scenario proxy, so each episode's inference records and its
report reach Reef as records, tagged with the play's labels. The runner plays episodes; how many plays a
task gets, and how they group into training batches, belong to its caller.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from reef.core.tasks import HarborTask
from reef.record2task.generator.client import Generator


@dataclass(frozen=True)
class Episode:
    """One played episode: its reward, the reports that reached Reef, and its error.

    An episode with an error and no reward could not run: the sandbox, the agent or the verifier failed
    before it scored. That is an infrastructure failure, not a failure of the solver.
    """

    episode_id: str
    reward: float | None
    report_record_ids: tuple[str, ...]
    error: str

    @property
    def is_reported(self) -> bool:
        return bool(self.report_record_ids)

    @property
    def is_infrastructure_failure(self) -> bool:
        return self.reward is None and bool(self.error)


class EpisodeRunner(ABC):
    """Stage a saved task where it runs, play episodes of it, and remove the staged copy."""

    @abstractmethod
    async def stage(self, task: HarborTask) -> Path:
        """Stage the task and return its path; staging the same task again returns the same path.

        Raises ``TaskNameConflict`` when a different task holds the name, and ``DuplicateTask`` when the same
        content is staged under another name.
        """

    @abstractmethod
    async def play(
        self,
        task_path: Path,
        *,
        scenario: str,
        arm: str,
        plays: int,
        extra_instruction_files: Sequence[str],
        labels: Mapping[str, str],
        model: str | None,
    ) -> tuple[Episode, ...]:
        """Play ``plays`` episodes of a staged task on one arm; each episode's records carry ``labels``."""

    @abstractmethod
    async def unstage(self, name: str) -> None:
        """Remove a staged copy that this runner created, unless another stage of the same task reused it."""


class GeneratorEpisodeRunner(EpisodeRunner):
    """Episodes through the generator service, whose task player runs each one in its sandbox."""

    def __init__(self, service: Generator) -> None:
        self.service = service
        #: How many stages of each task this runner made that are not unstaged.
        self.stage_counts: dict[str, int] = {}
        #: The tasks whose staged copy this runner created; a copy that was already there belongs to its writer.
        self.created: set[str] = set()

    async def stage(self, task: HarborTask) -> Path:
        written = await self.service.write_task(task)
        self.stage_counts[task.name] = self.stage_counts.get(task.name, 0) + 1
        if written.created:
            self.created.add(task.name)
        return written.path

    async def play(
        self,
        task_path: Path,
        *,
        scenario: str,
        arm: str,
        plays: int,
        extra_instruction_files: Sequence[str],
        labels: Mapping[str, str],
        model: str | None,
    ) -> tuple[Episode, ...]:
        played = await self.service.play(
            task_path,
            scenario=scenario,
            arm=arm,
            plays=plays,
            is_reporting=True,
            extra_instruction_files=extra_instruction_files,
            tags=labels,
            model=model,
        )
        return tuple(
            Episode(play.episode_id, play.reward, play.report_agent_record_ids, play.error) for play in played
        )

    async def unstage(self, name: str) -> None:
        remaining = self.stage_counts.get(name, 0) - 1
        if remaining > 0:
            self.stage_counts[name] = remaining
            return
        self.stage_counts.pop(name, None)
        if name in self.created:
            self.created.discard(name)
            await self.service.delete_task(name)
