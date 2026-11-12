"""The SPADE task generator: the Environment Designer's generations, with the hint based regret as its signal.

One ``generate`` call is one generation: ``count`` proposals across the configured ``skills``, each a
Designer call through the generator service (``reef.record2sample.generator``), with the last generation's
results in the prompt as SPADE's experience section.

``validate`` writes the task under the service's tasks root, runs Harbor's oracle and nop agents on it,
and removes it again: the generated-task recipe writes the valid task once more to run it.

The generator keeps its state in its ``GeneratorState``, which is in the scenario's task store, and keeps no
file of its own. ``generate`` keeps the open generation, its proposals and refusals, in the document
``open/<generation>``, named by the recipe's generation number as five digits. ``record_results`` closes it,
also after later generations started or after a restart. For each train task that the recipe ran and that has
``solution/hint.txt``, it plays the task ``hint_plays`` more times through the generator service, with that
file appended to the instruction. These plays report on the ``hint`` arm, so the recipe does not train on
them. It reports each proposal's score against the Designer's receipt: the mix of the task's plateau reward and
its floored regret (``designer_score``), 0 for a refused one. It keeps the generation report in the document
``report/<number>``, named by SPADE's own number of the generation, with the advantage of each Designer reply that
trains (``centered_advantages``). Eval tasks and tasks that could not run get no hint plays and no Designer report,
and do not go into the experience. A second call for a closed generation does nothing. A restart reads the last
report for the experience, and continues numbering after the reports and open generations.

With ``train_designer``, each Designer call asks the generator service to keep the reply's sampled tokens in its
inference record, and ``designer_advantages`` gives the recipe the advantages that the generation's report keeps.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Self

from pydantic import TypeAdapter

from recipes.beta.spade.generation import (
    DEFAULT_MAX_DESIGNER_SCALE,
    RECORDED_EXCERPT_CHARS,
    GenerationError,
    GenerationRecord,
    PlayRecord,
    ProposalRecord,
    ScoreSettings,
    TaskMeasure,
    centered_advantages,
    designer_score,
    experience_for,
    experience_text,
    generation_report,
    load_experience,
    mean_reward,
    skill_tag,
)
from reef.core.plugins import PluginConfigError
from reef.core.tasks import HarborTask
from reef.harness.harbor.rollouts import rewards_of
from reef.recipe.generated_tasks.generator import (
    FailedTask,
    GeneratedTaskResult,
    GeneratorState,
    HeldOutTask,
    RejectedTask,
    TaskGenerator,
    TaskValidationResult,
)
from reef.record2sample.generator.client import (
    DuplicateTask,
    Generator,
    GeneratorError,
    HttpGenerator,
    ProposedTask,
    TaskNameConflict,
    WrittenTask,
)
from reef.record2sample.generator.designer import DIFFICULTIES, DesignerRequest
from reef.record2sample.generator.inputs import TaskGenerationRequest

logger = logging.getLogger(__name__)

DEFAULT_COUNT = 8
DEFAULT_HINT_PLAYS = 2
HINT_FILE = "solution/hint.txt"
HINT_NAME = "hint.txt"
HINT_ARM = "hint"
# The state keys: an open generation by the recipe's generation number, and a report by SPADE's own number.
OPEN_PREFIX = "open/"
REPORT_PREFIX = "report/"
DEFAULT_SCORE_SETTINGS = ScoreSettings()
#: The config keys of the Designer's score, the fields of ``ScoreSettings``.
SCORE_KEYS = tuple(entry.name for entry in fields(ScoreSettings))
CONFIG_KEYS = frozenset(
    {
        "generator_url",
        "count",
        "skills",
        "difficulty",
        "turn_limit",
        "grounding_path",
        "hint_plays",
        "designer_report",
        "served_model",
        "plateau_low",
        "plateau_high",
        "plateau_ramp",
        "plateau_weight",
        "regret_weight",
        "regret_scale",
        "max_designer_scale",
        "train_designer",
    }
)


@dataclass(frozen=True)
class PendingProposal:
    """A proposal of an open generation that became a task; its result has not come back yet."""

    index: int
    skill: str | None
    designer_record_id: str


@dataclass(frozen=True)
class OpenGeneration:
    """A generation that ``generate`` made and ``record_results`` has not closed yet.

    Values: ``number`` is SPADE's number of the generation. ``refused`` holds the refused proposals, and
    ``pending`` the proposals that became tasks, by task name. ``error`` is the error that stopped the
    generation, or empty. ``reports`` maps the index of each proposal already reported to the Designer to its
    report ID; a delivery that stopped part way does not report those proposals again.
    Purpose: ``generate`` writes it to the state document ``open/<generation>``; ``record_results`` reads it to
    close the generation, also after later generations and after a restart.
    """

    number: int
    refused: tuple[ProposalRecord, ...]
    pending: dict[str, PendingProposal]
    error: str = ""
    #: The Designer report ID of each proposal already reported, by proposal index.
    reports: dict[int, str] = field(default_factory=dict)


OPEN_GENERATION = TypeAdapter(OpenGeneration)


def saved_designer_advantages(state: GeneratorState, recipe_generation: int) -> dict[str, float]:
    """The Designer advantages that the report of the recipe's generation ``recipe_generation`` keeps.

    The reports are the state documents ``report/<number>`` of ``state``. The result is empty when no report names
    that generation, such as a report written before reports named it. Raises ``GenerationError`` for a report that
    cannot be read.
    """
    # The report of a generation usually has the recipe's number; another report is a fallback.
    keys = state.keys(REPORT_PREFIX)
    ordered = sorted(keys, key=lambda key: key != f"{REPORT_PREFIX}{recipe_generation:05d}")
    for key in ordered:
        report = state.read(key)
        if report is None:
            continue
        try:
            document = json.loads(report)
        except ValueError as exc:
            raise GenerationError(f"the state document {key} is not a generation report: {exc}") from exc
        if not isinstance(document, dict) or document.get("recipe_generation") != recipe_generation:
            continue
        advantages = document.get("designer_advantages")
        if not isinstance(advantages, dict):
            raise GenerationError(f"the state document {key} holds no Designer advantages")
        return {str(record_id): float(advantage) for record_id, advantage in advantages.items()}
    return {}


class SpadeTaskGenerator(TaskGenerator):
    """The Designer's generations through the generator service, adapted by the regret of the last one.

    ``state`` holds the document ``open/<generation>`` of each open generation and the document
    ``report/<number>`` of each closed one, with each number as five digits. ``score_settings`` score each train
    task that ran (``designer_score``), and ``max_designer_scale`` caps the scale of the advantages of a
    generation's replies (``centered_advantages``). With ``is_training_designer``, each Designer call keeps its
    sample, and ``designer_advantages`` gives the recipe the advantages of a closed generation.
    """

    def __init__(
        self,
        *,
        scenario: str,
        service: Generator,
        state: GeneratorState,
        count: int = DEFAULT_COUNT,
        skills: tuple[str, ...] = (),
        difficulty: str = "medium",
        turn_limit: int = DesignerRequest.turn_limit,
        grounding: str | None = None,
        hint_plays: int = DEFAULT_HINT_PLAYS,
        is_reporting_designer: bool = True,
        served_model: str | None = None,
        score_settings: ScoreSettings = DEFAULT_SCORE_SETTINGS,
        max_designer_scale: float = DEFAULT_MAX_DESIGNER_SCALE,
        is_training_designer: bool = False,
    ) -> None:
        if count < 1 or hint_plays < 0:
            raise ValueError("count must be positive and hint_plays non-negative")
        if (
            isinstance(max_designer_scale, bool)
            or not isinstance(max_designer_scale, (int, float))
            or not math.isfinite(max_designer_scale)
            or max_designer_scale <= 0
        ):
            raise ValueError("max_designer_scale must be a positive finite number")
        if difficulty not in DIFFICULTIES:
            raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
        if turn_limit < 2:
            raise ValueError("turn_limit must be at least 2")
        self.scenario = scenario
        self.service = service
        self.state = state
        self.count = count
        self.skills = skills
        self.difficulty = difficulty
        self.turn_limit = turn_limit
        self.grounding = grounding
        self.hint_plays = hint_plays
        self.is_reporting_designer = is_reporting_designer
        self.served_model = served_model
        self.score_settings = score_settings
        self.max_designer_scale = float(max_designer_scale)
        self.is_training_designer = is_training_designer
        # What the last generation measured, for the next prompt; read back from the newest report.
        recorded = [int(key.removeprefix(REPORT_PREFIX)) for key in state.keys(REPORT_PREFIX)]
        self.experience_generation = max(recorded, default=-1)
        last_report = state.read(f"{REPORT_PREFIX}{self.experience_generation:05d}") if recorded else None
        self.experience: tuple[PlayRecord, ...] = () if last_report is None else load_experience(last_report)
        open_numbers = [open_generation.number for _, open_generation in self.open_generations()]
        self.generation = max((*recorded, *open_numbers), default=-1) + 1

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], *, scenario: str, environ: Mapping[str, str], state: GeneratorState
    ) -> Self:
        if "state_dir" in config:
            raise PluginConfigError(
                "SpadeTaskGenerator does not take state_dir: the generator keeps its state in the scenario's task "
                "store (RFC #1370). Remove the key; state under an old state_dir is not read."
            )
        unknown = sorted(str(key) for key in config if key not in CONFIG_KEYS)
        if unknown:
            raise PluginConfigError(f"SpadeTaskGenerator does not take {', '.join(unknown)}")
        generator_url = str(config.get("generator_url", "") or "")
        if not generator_url or "${" in generator_url:
            raise PluginConfigError(
                "SpadeTaskGenerator needs generator_url: ${endpoints.generator} when the deployment has a "
                "generator section"
            )
        skills = config.get("skills", ())
        skill_names = skills.split(",") if isinstance(skills, str) else skills
        grounding_path = str(config.get("grounding_path", "") or "")
        served_model = config.get("served_model")
        train_designer = config.get("train_designer", False)
        if not isinstance(train_designer, bool):
            raise PluginConfigError(
                f"SpadeTaskGenerator's train_designer must be true or false, got {train_designer!r}"
            )
        try:
            return cls(
                scenario=scenario,
                service=HttpGenerator(generator_url),
                state=state,
                count=int(config.get("count", DEFAULT_COUNT)),
                skills=tuple(str(skill).strip() for skill in skill_names if str(skill).strip()),
                difficulty=str(config.get("difficulty", "medium")),
                turn_limit=int(config.get("turn_limit", DesignerRequest.turn_limit)),
                grounding=Path(grounding_path).read_text(encoding="utf-8") if grounding_path else None,
                hint_plays=int(config.get("hint_plays", DEFAULT_HINT_PLAYS)),
                is_reporting_designer=bool(config.get("designer_report", True)),
                served_model=str(served_model) if served_model else None,
                # The settings refuse a bool or a value that is not a number, by its key.
                score_settings=ScoreSettings(**{key: config[key] for key in SCORE_KEYS if key in config}),
                max_designer_scale=config.get("max_designer_scale", DEFAULT_MAX_DESIGNER_SCALE),
                is_training_designer=train_designer,
            )
        except (OSError, ValueError) as exc:
            raise PluginConfigError(f"invalid SpadeTaskGenerator config: {exc}") from exc

    async def generate(self, request: TaskGenerationRequest) -> tuple[HarborTask, ...]:
        """One generation: ``count`` Designer proposals; a failed service call ends it with what ran."""
        if any(record.scenario != self.scenario for record in request.source_records):
            raise ValueError("source records must belong to the generator's scenario")
        refused: list[ProposalRecord] = []
        pending: dict[str, PendingProposal] = {}
        error = ""
        tasks: list[HarborTask] = []
        try:
            for index in range(self.count):
                skill = self.skills[index % len(self.skills)] if self.skills else None
                proposed = await self.propose(request, skill=skill, index=index)
                if proposed.task is None:
                    refused.append(ProposalRecord(index, skill, proposed.record_id, None, proposed.refusal))
                    continue
                pending[proposed.task.name] = PendingProposal(index, skill, proposed.record_id)
                tasks.append(proposed.task)
        except GeneratorError as exc:
            error = str(exc)
            logger.error("SPADE generation %d stopped: %s", self.generation, exc)
        self.save_open(request.generation, OpenGeneration(self.generation, tuple(refused), pending, error))
        self.generation += 1
        return tuple(tasks)

    def save_open(self, generation: int, open_generation: OpenGeneration) -> None:
        """Keep an open generation in the state document ``open/<generation>``, replacing the document."""
        self.state.write(f"{OPEN_PREFIX}{generation:05d}", OPEN_GENERATION.dump_json(open_generation).decode("utf-8"))

    def open_generations(self) -> tuple[tuple[int, OpenGeneration], ...]:
        """Each open generation with the recipe's generation number that names its document, in key order."""
        opened = []
        for key in self.state.keys(OPEN_PREFIX):
            document = self.state.read(key)
            # A document that another call removed since the key was read is a closed generation.
            if document is not None:
                opened.append((int(key.removeprefix(OPEN_PREFIX)), OPEN_GENERATION.validate_json(document)))
        return tuple(opened)

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
            is_recording_training=self.is_training_designer,
        )

    def save_error(self, task_name: str, error: str) -> None:
        """Name an error in the report of the open generation that holds the task."""
        for generation, open_generation in self.open_generations():
            if task_name in open_generation.pending:
                self.save_open(generation, replace(open_generation, error=error))
                return

    async def validate(self, task: HarborTask) -> TaskValidationResult:
        """Harbor's oracle and nop agents on a written copy: the reference solution scores 1, doing nothing below 1."""
        try:
            written = await self.write(task)
        except (DuplicateTask, TaskNameConflict) as exc:
            return TaskValidationResult((f"refused: {exc}",))
        try:
            oracle = await self.service.check(written.path)
        except GeneratorError as exc:
            self.save_error(task.name, str(exc))
            raise
        # A copy that was already staged belongs to its writer, which may be running it.
        if written.created:
            await self.service.delete_task(task.name)
        return TaskValidationResult(() if oracle.is_solvable else (f"oracle check refused: {oracle.reason}",))

    async def hint_rewards(self, task: HarborTask, *, generation: int) -> tuple[float, ...]:
        """Run a task that the recipe ran ``hint_plays`` times with its hint, and return the rewards of the plays.

        Staging the task again returns the copy that the recipe ran. The plays report on the ``hint`` arm,
        which the recipe does not train on.
        """
        rollouts = await self.service.run_rollouts(
            (await self.service.write_task(task)).path,
            scenario=self.scenario,
            arm=HINT_ARM,
            count=self.hint_plays,
            extra_instruction_files=(HINT_FILE,),
            tags={"generation": str(generation)},
            model=self.served_model,
        )
        return rewards_of(rollouts)

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

    async def record_results(self, results: tuple[GeneratedTaskResult, ...], *, generation: int) -> None:
        """Close an open generation: save its report and experience, and report Designer rewards for train tasks."""
        open_key = f"{OPEN_PREFIX}{generation:05d}"
        open_document = self.state.read(open_key)
        if open_document is None:
            logger.info("SPADE generation %d of scenario %r is already closed", generation, self.scenario)
            return
        open_generation = OPEN_GENERATION.validate_json(open_document)
        error = open_generation.error
        pending_proposals = dict(open_generation.pending)
        proposals: list[tuple[ProposalRecord, TaskMeasure | None]] = [
            (proposal, None) for proposal in open_generation.refused
        ]
        for result in results:
            pending = pending_proposals.pop(result.task.name, None)
            if pending is None:
                continue
            proposal = ProposalRecord(pending.index, pending.skill, pending.designer_record_id, None, "")
            if isinstance(result, HeldOutTask):
                proposals.append((replace(proposal, task_name=result.task.name, split="eval"), None))
                continue
            if isinstance(result, RejectedTask):
                proposals.append((replace(proposal, refusal=result.reason), None))
                continue
            if isinstance(result, FailedTask):
                failed = replace(proposal, refusal=result.error, infrastructure_failure=result.infrastructure_failure)
                proposals.append((failed, None))
                continue
            plain = result.rollout_rewards
            hint: tuple[float, ...] = ()
            if self.hint_plays and HINT_NAME in result.task.solution:
                try:
                    hint = await self.hint_rewards(result.task, generation=open_generation.number)
                except GeneratorError as exc:
                    error = error or f"the hint plays of {result.task.name} failed: {exc}"
                    logger.error("SPADE hint plays of %s failed: %s", result.task.name, exc)
            play_record = PlayRecord(
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
                record=play_record,
                score=designer_score(play_record.regret, play_record.return_without_hint, self.score_settings),
            )
            proposals.append((replace(proposal, task_name=result.task.name), measure))
        proposals.sort(key=lambda entry: entry[0].index)
        reported: list[ProposalRecord] = []
        for entry, entry_measure in proposals:
            # A delivery that stopped part way already reported some proposals; they are not reported again.
            report_id = open_generation.reports.get(entry.index)
            if report_id is not None:
                reported.append(replace(entry, designer_report_id=report_id))
                continue
            entry = await self.reported(entry, entry_measure, generation=open_generation.number)
            reported.append(entry)
            if entry.designer_report_id:
                open_generation = replace(
                    open_generation, reports={**open_generation.reports, entry.index: entry.designer_report_id}
                )
                self.save_open(generation, open_generation)
        measures = tuple(measure for _, measure in proposals if measure is not None)
        generation_record = GenerationRecord(
            open_generation.number,
            tuple(reported),
            measures,
            error,
            recipe_generation=generation,
            designer_advantages=centered_advantages(reported, measures, max_scale=self.max_designer_scale),
        )
        report_key = f"{REPORT_PREFIX}{open_generation.number:05d}"
        self.state.write(report_key, generation_report(generation_record))
        logger.info(
            "SPADE generation %d on scenario %r: %d tasks of %d proposals in the state document %s",
            open_generation.number,
            self.scenario,
            len(measures),
            len(reported),
            report_key,
        )
        # Generations can close out of order; the experience is the newest closed one's.
        if open_generation.number > self.experience_generation:
            self.experience = generation_record.experience
            self.experience_generation = open_generation.number
        self.state.remove(open_key)

    def designer_advantages(self, generation: int) -> Mapping[str, float]:
        """The advantages that the report of the recipe's ``generation`` keeps, by Designer record ID; with
        ``train_designer`` off, none. The report does not change once written, so a repeated call gives the same
        answer."""
        if not self.is_training_designer:
            return {}
        return saved_designer_advantages(self.state, generation)

    async def reported(
        self, proposal: ProposalRecord, measure: TaskMeasure | None, *, generation: int
    ) -> ProposalRecord:
        """The Designer's report for one proposal: the task's score (``designer_score``), 0 for a refused one."""
        if not self.is_reporting_designer or proposal.split == "eval" or proposal.infrastructure_failure:
            return proposal
        metadata: dict[str, object] = {"generation": generation, **skill_tag(proposal.skill)}
        if measure is None:
            score = 0.0
            metadata["refusal"] = proposal.refusal
        else:
            score = measure.score
            metadata["task"] = {"name": measure.name, "digest": measure.digest}
            metadata["outcome"] = measure.outcome
            metadata["regret"] = measure.regret
            metadata["return_without_hint"] = measure.record.return_without_hint
            metadata["return_with_hint"] = measure.record.return_with_hint
        report_id = await self.service.report_proposal(
            proposal.designer_record_id, scenario=self.scenario, score=score, metadata=metadata
        )
        return replace(proposal, designer_report_id=report_id)
