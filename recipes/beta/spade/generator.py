"""The SPADE task generator: the Environment Designer's generations, with the hint based regret as its signal.

One ``generate`` call is one generation: ``count`` proposals across the configured ``skills``, each a
Designer call through the generator service (``reef.record2task.generator``), with the last generation's
results in the prompt as SPADE's experience section. A task with ``solution/hint.txt`` asks for a
``hint`` arm of ``hint_plays`` plays with that file appended to the instruction.

``validate`` writes the task under the service's tasks root, runs Harbor's oracle and nop agents on it,
and removes it again: the generated-task recipe writes the valid task once more to play it.

``record_results`` closes the generation. It reports each proposal's regret against the Designer's
receipt (0 for a refused one), writes the split manifest of the played tasks, and keeps the
generation's report under ``state_dir``. A restart reads the last report to continue with the next
generation and the last experience.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Self

from recipes.beta.spade.generation import (
    RECORDED_EXCERPT_CHARS,
    GenerationRecord,
    PlayRecord,
    ProposalRecord,
    TaskMeasure,
    experience_for,
    experience_text,
    load_experience,
    mean_reward,
    recorded_generations,
    report_path_for,
    skill_tag,
    write_generation_report,
)
from reef.core.plugins import PluginConfigError
from reef.core.tasks import (
    GeneratedTask,
    HarborTask,
    PlayArm,
    TaskGenerationRequest,
    TaskGenerator,
    TaskPlayResult,
    TaskValidationResult,
)
from reef.record2task.generator.client import (
    DuplicateTask,
    Generator,
    GeneratorError,
    HttpGenerator,
    ProposedTask,
    TaskNameConflict,
    WrittenTask,
)
from reef.record2task.generator.designer import DIFFICULTIES, DesignerRequest

logger = logging.getLogger(__name__)

DEFAULT_COUNT = 8
DEFAULT_HINT_PLAYS = 2
DEFAULT_EVAL_FRACTION = 0.25
HINT_FILE = "solution/hint.txt"
HINT_NAME = "hint.txt"
HINT_ARM = "hint"
CONFIG_KEYS = frozenset(
    {
        "generator_url",
        "state_dir",
        "count",
        "skills",
        "difficulty",
        "turn_limit",
        "grounding_path",
        "hint_plays",
        "eval_fraction",
        "seed",
        "designer_report",
        "served_model",
    }
)


@dataclass(frozen=True)
class PendingProposal:
    """A proposal of the open generation that became a task; its result has not come back yet."""

    index: int
    skill: str | None
    designer_record_id: str


class SpadeTaskGenerator(TaskGenerator):
    """The Designer's generations through the generator service, adapted by the regret of the last one."""

    def __init__(
        self,
        *,
        scenario: str,
        service: Generator,
        state_dir: Path,
        count: int = DEFAULT_COUNT,
        skills: tuple[str, ...] = (),
        difficulty: str = "medium",
        turn_limit: int = DesignerRequest.turn_limit,
        grounding: str | None = None,
        hint_plays: int = DEFAULT_HINT_PLAYS,
        eval_fraction: float = DEFAULT_EVAL_FRACTION,
        seed: int = 0,
        is_reporting_designer: bool = True,
        served_model: str | None = None,
    ) -> None:
        if count < 1 or hint_plays < 0:
            raise ValueError("count must be positive and hint_plays non-negative")
        if difficulty not in DIFFICULTIES:
            raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
        if turn_limit < 2:
            raise ValueError("turn_limit must be at least 2")
        if not 0 <= eval_fraction < 1:
            raise ValueError("eval_fraction must be in [0, 1)")
        self.scenario = scenario
        self.service = service
        self.state_dir = state_dir
        self.count = count
        self.skills = skills
        self.difficulty = difficulty
        self.turn_limit = turn_limit
        self.grounding = grounding
        self.hint_plays = hint_plays
        self.eval_fraction = eval_fraction
        self.seed = seed
        self.is_reporting_designer = is_reporting_designer
        self.served_model = served_model
        # What the last generation measured, for the next prompt; read back from the state directory.
        recorded = recorded_generations(state_dir)
        self.experience: tuple[PlayRecord, ...] = (
            load_experience(report_path_for(state_dir, recorded[-1])) if recorded else ()
        )
        self.generation = recorded[-1] + 1 if recorded else 0
        # The open generation: what generate made and record_results has not closed yet.
        self.refused: list[ProposalRecord] = []
        self.pending: dict[str, PendingProposal] = {}
        self.error = ""

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, scenario: str, environ: Mapping[str, str]) -> Self:
        unknown = sorted(str(key) for key in config if key not in CONFIG_KEYS)
        if unknown:
            raise PluginConfigError(f"SpadeTaskGenerator does not take {', '.join(unknown)}")
        generator_url = str(config.get("generator_url", "") or "")
        if not generator_url or "${" in generator_url:
            raise PluginConfigError(
                "SpadeTaskGenerator needs generator_url: ${endpoints.generator} when the deployment has a "
                "generator section"
            )
        state_dir = str(config.get("state_dir", "") or "")
        if not state_dir:
            raise PluginConfigError("SpadeTaskGenerator needs a state_dir for its generation reports")
        skills = config.get("skills", ())
        skill_names = skills.split(",") if isinstance(skills, str) else skills
        grounding_path = str(config.get("grounding_path", "") or "")
        served_model = config.get("served_model")
        try:
            return cls(
                scenario=scenario,
                service=HttpGenerator(generator_url),
                # Each scenario keeps its own experience.
                state_dir=Path(state_dir) / scenario,
                count=int(config.get("count", DEFAULT_COUNT)),
                skills=tuple(str(skill).strip() for skill in skill_names if str(skill).strip()),
                difficulty=str(config.get("difficulty", "medium")),
                turn_limit=int(config.get("turn_limit", DesignerRequest.turn_limit)),
                grounding=Path(grounding_path).read_text(encoding="utf-8") if grounding_path else None,
                hint_plays=int(config.get("hint_plays", DEFAULT_HINT_PLAYS)),
                eval_fraction=float(config.get("eval_fraction", DEFAULT_EVAL_FRACTION)),
                seed=int(config.get("seed", 0)),
                is_reporting_designer=bool(config.get("designer_report", True)),
                served_model=str(served_model) if served_model else None,
            )
        except (OSError, ValueError) as exc:
            raise PluginConfigError(f"invalid SpadeTaskGenerator config: {exc}") from exc

    async def generate(self, request: TaskGenerationRequest) -> tuple[GeneratedTask, ...]:
        """One generation: ``count`` Designer proposals; a failed service call ends it with what ran."""
        if any(record.scenario != self.scenario for record in request.source_records):
            raise ValueError("source records must belong to the generator's scenario")
        self.refused, self.pending, self.error = [], {}, ""
        tasks: list[GeneratedTask] = []
        try:
            for index in range(self.count):
                skill = self.skills[index % len(self.skills)] if self.skills else None
                proposed = await self.propose(request, skill=skill, index=index)
                if proposed.task is None:
                    self.refused.append(ProposalRecord(index, skill, proposed.record_id, None, proposed.refusal))
                    continue
                self.pending[proposed.task.name] = PendingProposal(index, skill, proposed.record_id)
                arms = (
                    (PlayArm(HINT_ARM, self.hint_plays, (HINT_FILE,)),)
                    if self.hint_plays and HINT_NAME in proposed.task.solution
                    else ()
                )
                tasks.append(GeneratedTask(proposed.task, arms))
        except GeneratorError as exc:
            self.error = str(exc)
            logger.error("SPADE generation %d stopped: %s", self.generation, exc)
        return tuple(tasks)

    async def propose(self, request: TaskGenerationRequest, *, skill: str | None, index: int) -> ProposedTask:
        """Ask the Designer for one task, with the experience section for its skill; keep its receipt."""
        designer_request = DesignerRequest(
            inputs=request,
            skill=skill,
            difficulty=self.difficulty,
            turn_limit=self.turn_limit,
            grounding=self.grounding,
            experience_text=experience_text(experience_for(self.experience, skill)),
        )
        tags = {"role": "designer", "generation": str(self.generation), **skill_tag(skill)}
        return await self.service.propose(
            designer_request,
            scenario=self.scenario,
            generation=self.generation,
            index=index,
            tags=tags,
            model=self.served_model,
        )

    async def validate(self, task: HarborTask) -> TaskValidationResult:
        """Harbor's oracle and nop agents on a written copy: the reference solution scores 1, doing nothing below 1."""
        try:
            written = await self.write(task)
        except (DuplicateTask, TaskNameConflict) as exc:
            return TaskValidationResult((f"refused: {exc}",))
        try:
            oracle = await self.service.check(written.path)
        except GeneratorError as exc:
            self.error = str(exc)
            raise
        await self.service.delete_task(task.name)
        return TaskValidationResult(() if oracle.is_solvable else (f"oracle check refused: {oracle.reason}",))

    async def write(self, task: HarborTask) -> WrittenTask:
        """Write a task; only a description-based generation's rerun may replace a conflicting name."""
        try:
            return await self.service.write_task(task)
        except TaskNameConflict:
            if "designer_record_id" in task.metadata:
                raise
            # A reload cancelled this generation's earlier attempt; what it wrote under the name gives way.
            await self.service.delete_task(task.name)
            return await self.service.write_task(task)

    async def record_results(self, results: tuple[TaskPlayResult, ...]) -> None:
        """Close the generation: the Designer's reports, the manifest, the report file and the next experience."""
        proposals: list[tuple[ProposalRecord, TaskMeasure | None]] = [(proposal, None) for proposal in self.refused]
        for result in results:
            pending = self.pending.pop(result.task.name, None)
            if pending is None:
                continue
            proposal = ProposalRecord(pending.index, pending.skill, pending.designer_record_id, None, result.error)
            if not result.is_played:
                proposals.append((proposal, None))
                continue
            plain = result.training_rewards
            hint = result.measurement_rewards.get(HINT_ARM, ())
            record = PlayRecord(
                name=result.task.name,
                skill=pending.skill,
                return_without_hint=mean_reward(plain),
                return_with_hint=mean_reward(hint) if hint else mean_reward(plain),
                instruction_excerpt=result.task.instruction[:RECORDED_EXCERPT_CHARS],
            )
            measure = TaskMeasure(
                name=result.task.name,
                skill=pending.skill,
                digest=result.task.digest,
                plain_rewards=plain,
                hint_rewards=hint,
                record=record,
            )
            proposals.append((replace(proposal, task_name=result.task.name), measure))
        proposals.sort(key=lambda entry: entry[0].index)
        reported = [await self.reported(proposal, measure) for proposal, measure in proposals]
        measures = tuple(measure for _, measure in proposals if measure is not None)
        manifest_path: Path | None = None
        if measures and not self.error:
            manifest_path = await self.service.write_manifest(
                generation=self.generation,
                names=[measure.name for measure in measures],
                eval_fraction=self.eval_fraction,
                seed=self.seed,
            )
        record = GenerationRecord(self.generation, tuple(reported), measures, manifest_path, self.error)
        report_path = write_generation_report(self.state_dir, record)
        logger.info(
            "SPADE generation %d on scenario %r: %d tasks of %d proposals under %s",
            self.generation,
            self.scenario,
            len(measures),
            len(reported),
            report_path,
        )
        self.experience = record.experience
        self.generation += 1
        self.refused, self.pending = [], {}

    async def reported(self, proposal: ProposalRecord, measure: TaskMeasure | None) -> ProposalRecord:
        """The Designer's report for one proposal: its regret as the score, 0 for a refused one."""
        if not self.is_reporting_designer:
            return proposal
        metadata: dict[str, object] = {"generation": self.generation, **skill_tag(proposal.skill)}
        if measure is None:
            score = 0.0
            metadata["refusal"] = proposal.refusal
        else:
            score = max(measure.regret, 0.0)
            metadata["task"] = {"name": measure.name, "digest": measure.digest}
            metadata["outcome"] = measure.outcome
            metadata["regret"] = measure.regret
            metadata["return_without_hint"] = measure.record.return_without_hint
            metadata["return_with_hint"] = measure.record.return_with_hint
        report_id = await self.service.report_proposal(
            proposal.designer_record_id, scenario=self.scenario, score=score, metadata=metadata
        )
        return replace(proposal, designer_report_id=report_id)
