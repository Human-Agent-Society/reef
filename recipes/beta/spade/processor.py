"""The SPADE processor: the Designer's generations driven from here, the Reasoning Agent's episodes trained here.

Two contracts meet in this class. As a reported feedback processor it takes the task player's reports:
every plain episode names its task under ``metadata.task``, the episodes of one task form a group, a
group is complete at ``rollouts_per_task`` episodes, and a batch holds ``tasks_per_step`` complete groups,
in arrival order. As a task generation processor it writes the tasks those episodes are played on:
``generate`` asks the Designer for one task through the generator service (``reef.record2dataset``), with
the last generation's results in the prompt as SPADE's experience section, and ``validate`` runs Harbor's
oracle and nop agents on the written task.

One generation is one job on a private worker, off the trainer's thread: ``count`` proposals, each parsed,
written (a duplicate refused; a name an earlier attempt of the generation took before a reload cancelled
it is replaced), validated, played ``rollouts_per_task`` times as it is (the training data) and
``hint_plays`` times with the hint appended (measured only), and reported against the Designer's receipt
with its raw regret as the score (the refusal floor for a refused one), the generation's size in the
metadata and the last generation's summary in the feedback. The first generation starts when the processor first looks for a
batch; the next once ``batches_per_generation`` batches were acknowledged since the previous one started
(its episodes train while it runs), so the Designer always writes for the policy that trains now; a
generation that measured no task is followed at once. Every generation's report goes under ``state_dir``,
which is what a restart reads to carry on with the next number and the last experience.

The Designer's own reports and the hint arm's reports share the scenario with the training data; the
processor tells them apart (``metadata.role``, the episode's ``arm`` label) and releases them unassembled.

With ``report_plays_after_generation`` every play is held until the generation landed, so the Designer is
measured against one Reasoning Agent version. The plain plays then go out stamped with the round
(``metadata.round``, the generation's label, and ``metadata.round_plays``, how many plain plays report under
it); the processor keeps a round as one unit, ready at that many reports whatever ``tasks_per_step`` says,
and orders its batch by task so the objective's contiguous groups hold. The hint plays go out beside them.
"""

from __future__ import annotations

import logging
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from recipes.beta.spade.generation import (
    RECORDED_EXCERPT_CHARS,
    GenerationRecord,
    GenerationSummary,
    PlayRecord,
    ProposalRecord,
    TaskMeasure,
    experience_for,
    experience_text,
    generation_label,
    load_experience,
    load_generation_summary,
    mean_reward,
    recorded_generations,
    report_path_for,
    rewards_of,
    skill_tag,
    write_generation_report,
)
from reef.core import AgentRecord
from reef.core.tasks import HarborTask
from reef.harness.client.tasks import TaskPlay
from reef.record2dataset.client import (
    DuplicateTask,
    Generator,
    GeneratorError,
    HttpGenerator,
    TaskNameConflict,
    WrittenTask,
)
from reef.record2dataset.designer import DesignerRequest
from reef.train.processors.computed import Failed, JudgingWorker, SupportsReceipt
from reef.train.processors.reported import GroupDecision, ReportContext, ReportedFeedbackProcessor, SampleAssembly
from reef.train.processors.task_generation import TaskGenerationProcessor, TaskGenerationRequest, TaskValidationResult
from reef.train.types import ProcessorContext, TrainDataItem, TrainingBatch, TrajectoryItem

logger = logging.getLogger(__name__)

DEFAULT_TASKS_PER_STEP = 4
DEFAULT_ROLLOUTS_PER_TASK = 4
# Qwen3 with thinking off ends the generation prompt with an empty think block of four tokens.
DEFAULT_SCAFFOLD_TOLERANCE = 8
DEFAULT_COUNT = 8
DEFAULT_HINT_PLAYS = 2
DEFAULT_EVAL_FRACTION = 0.25
DEFAULT_BATCHES_PER_GENERATION = 1
HINT_FILE = "solution/hint.txt"
HINT_NAME = "hint.txt"
TRAINING_ARM = "plain"
HINT_ARM = "hint"
# A refused proposal scores below any measured task: with group centering, 0 would outrank a task whose hint hurt.
REFUSAL_SCORE = -1.0
ROUND_KEY = "round:"


def reported_task_name(report: AgentRecord) -> str | None:
    """The task a report names under ``metadata.task.name``, the way the task player sends it."""
    metadata = report.payload.get("metadata")
    task = metadata.get("task") if isinstance(metadata, Mapping) else None
    name = task.get("name") if isinstance(task, Mapping) else None
    return name if isinstance(name, str) and name else None


def reported_round(report: AgentRecord) -> tuple[str, int] | None:
    """The round a held play reports under and how many plain plays report with it, when the generation stamped them."""
    metadata = report.payload.get("metadata")
    if not isinstance(metadata, Mapping) or "round" not in metadata:
        return None
    label, plays = metadata.get("round"), metadata.get("round_plays")
    if not isinstance(label, str) or not label:
        raise ValueError("SPADE training requires metadata.round to be a label, text that is not empty")
    if isinstance(plays, bool) or not isinstance(plays, int) or plays < 1:
        raise ValueError("SPADE training requires metadata.round_plays, the round's plain plays, a positive integer")
    return label, plays


def held_plays(measures: Sequence[TaskMeasure]) -> tuple[tuple[str, tuple[TaskPlay, ...]], ...]:
    """A generation's held plays by arm, the plain arm first."""
    return (
        (TRAINING_ARM, tuple(play for measure in measures for play in measure.plain_plays)),
        (HINT_ARM, tuple(play for measure in measures for play in measure.hint_plays)),
    )


def reported_role(report: AgentRecord) -> str | None:
    """``metadata.role`` on a report: the Designer's reports say ``designer``."""
    metadata = report.payload.get("metadata")
    role = metadata.get("role") if isinstance(metadata, Mapping) else None
    return role if isinstance(role, str) else None


def reported_arm(report: AgentRecord) -> str | None:
    """The ``arm`` label of an episode, under ``metadata.episode.labels`` where the task player puts the labels."""
    metadata = report.payload.get("metadata")
    episode = metadata.get("episode") if isinstance(metadata, Mapping) else None
    labels = episode.get("labels") if isinstance(episode, Mapping) else None
    arm = labels.get("arm") if isinstance(labels, Mapping) else None
    return arm if isinstance(arm, str) else None


class UnassembledEpisode(ValueError):
    """An episode whose records do not chain into one sample: a mid episode failure or a forked history."""


class ProposalRefused(ValueError):
    """The Designer's reply was no task; the record its call left is kept for the report."""

    def __init__(self, reason: str, record_id: str) -> None:
        super().__init__(reason)
        self.record_id = record_id


@dataclass(frozen=True)
class GenerationJob(SupportsReceipt):
    """One generation to run; the receipt names it."""

    generation: int


@dataclass(frozen=True)
class GenerationOutcome(SupportsReceipt):
    """What the worker hands back: the generation's record, written under the state directory."""

    record: GenerationRecord
    report_path: Path


class SpadeProcessor(ReportedFeedbackProcessor, TaskGenerationProcessor):
    """Groups the Reasoning Agent's episodes by task and runs the Designer's generations that produce them."""

    output_schema = TrainingBatch
    exclusive_sources = True

    def __init__(
        self,
        context: ProcessorContext,
        *,
        generator: Generator | None = None,
        worker: JudgingWorker | None = None,
    ) -> None:
        config = dict(context.config)
        self.tasks_per_step = int(config.get("tasks_per_step", DEFAULT_TASKS_PER_STEP))
        self.rollouts_per_task = int(config.get("rollouts_per_task", DEFAULT_ROLLOUTS_PER_TASK))
        # Episodes given up per group key; they count toward the group's size.
        self.unassembled_episodes: dict[Hashable, int] = {}
        if self.tasks_per_step <= 0:
            raise ValueError("tasks_per_step must be positive")
        if self.rollouts_per_task < 2:
            raise ValueError("rollouts_per_task must be at least two: a group of one has no relative reward")
        # An agent's episode is many model calls that extend one conversation; the sample spans them.
        config.setdefault("accept_multi_turn_policy_samples", True)
        config.setdefault("scaffold_tolerance", DEFAULT_SCAFFOLD_TOLERANCE)
        self._assembly = SampleAssembly.from_config(context.with_config(config))
        # One unit of the batch is one complete task group.
        super().__init__(context.with_config({**config, "batch_size": self.tasks_per_step}))

        # --- the Designer's generations ---
        self.generations = int(config.get("generations", 0))
        self.batches_per_generation = int(config.get("batches_per_generation", DEFAULT_BATCHES_PER_GENERATION))
        self.count = int(config.get("count", DEFAULT_COUNT))
        self.hint_plays = int(config.get("hint_plays", DEFAULT_HINT_PLAYS))
        self.eval_fraction = float(config.get("eval_fraction", DEFAULT_EVAL_FRACTION))
        self.seed = int(config.get("seed", 0))
        self.description = str(config.get("description", "") or "")
        skills = config.get("skills", ())
        self.skills = tuple(skills.split(",") if isinstance(skills, str) else skills)
        self.skills = tuple(skill.strip() for skill in self.skills if skill.strip())
        self.difficulty = str(config.get("difficulty", "medium"))
        self.turn_limit = int(config.get("turn_limit", DesignerRequest.turn_limit))
        self.is_reporting_designer = bool(config.get("designer_report", True))
        self.report_plays_after_generation = bool(config.get("report_plays_after_generation", False))
        served_model = config.get("served_model")
        self.served_model = str(served_model) if served_model else None
        grounding_path = str(config.get("grounding_path", "") or "")
        self.grounding = Path(grounding_path).read_text(encoding="utf-8") if grounding_path else None
        state_dir = str(config.get("state_dir", "") or "")
        self.state_dir = Path(state_dir) if state_dir else None
        if self.generations < 0 or self.batches_per_generation < 1 or self.count < 1 or self.hint_plays < 0:
            raise ValueError(
                "generations must be non-negative, batches_per_generation and count positive, hint_plays non-negative"
            )
        if not 0 <= self.eval_fraction < 1:
            raise ValueError("eval_fraction must be in [0, 1)")
        generator_url = str(config.get("generator_url", "") or "")
        if generator is None and self.generations > 0:
            if not generator_url or "${" in generator_url:
                raise ValueError(
                    "generations need generator_url: the generator service's address, ${endpoints.generator} "
                    "when the deployment carries a generator section"
                )
            generator = HttpGenerator(generator_url)
        if self.generations > 0 and (not self.description.strip() or self.state_dir is None):
            raise ValueError("generations need a description (what the tasks are about) and a state_dir")
        self.generator = generator
        self._worker = worker
        if self._worker is None and self.generator is not None and self.generations > 0:
            self._worker = JudgingWorker(self.run_generation, concurrency=1)
        # What the last generation measured, for the next prompt and the next reports; read back from the state directory.
        recorded = recorded_generations(self.state_dir) if self.state_dir is not None else ()
        last_report = (
            report_path_for(self.state_dir, recorded[-1]) if recorded and self.state_dir is not None else None
        )
        self._experience: tuple[PlayRecord, ...] = load_experience(last_report) if last_report is not None else ()
        self.previous_summary: GenerationSummary | None = (
            load_generation_summary(last_report) if last_report is not None else None
        )
        self._next_generation = recorded[-1] + 1 if recorded else 0
        self._completed = len(recorded)
        self._in_flight: int | None = None
        self._generation = self._next_generation
        # Batches acknowledged since the last generation started; the next one is due at batches_per_generation.
        self._batches_since_generation = 0
        self._has_landed = False
        # A generation without a task makes no batch; waiting for one would stall the loop until a restart.
        self._landed_empty = False
        # Batches the backend dropped as stale trained nothing, so they do not pace the Designer.
        self._dropped: set[str] = set()
        self._last_error = ""

    # ------------------------------------------------------- the reported half

    def is_training_report(self, report: AgentRecord) -> bool:
        # The Designer's regret and the hint arm's rewards share the scenario; only the plain arm trains.
        if reported_role(report) == "designer":
            return False
        arm = reported_arm(report)
        return arm is None or arm == TRAINING_ARM

    def make_sample(self, context: ReportContext) -> TrajectoryItem:
        task_name = reported_task_name(context.report)
        if task_name is None:
            raise ValueError("SPADE training requires metadata.task.name on the report, as the task player sends it")
        try:
            sample = self._assembly.build(context, context.require_score())
        except ValueError as error:
            raise UnassembledEpisode(str(error)) from error
        stamped = reported_round(context.report)
        if stamped is not None:
            sample = sample.with_metadata(round=stamped[0], round_plays=stamped[1])
        return replace(sample, group_id=task_name)

    def unassembled(self, context: ReportContext, error: ValueError) -> bool:
        # An episode that cannot be assembled is given up and still counts toward its group, so the
        # group completes instead of waiting forever for a sample that will never come.
        if not isinstance(error, UnassembledEpisode):
            return False
        key, _ = self.grouping(context)
        self.unassembled_episodes[key] = self.unassembled_episodes.get(key, 0) + 1
        return True

    def grouping(self, context: ReportContext) -> tuple[Hashable | None, Hashable | None]:
        # A round's plays are one unit, so one step trains on all of them before the weights move.
        stamped = reported_round(context.report)
        if stamped is not None:
            return f"{ROUND_KEY}{stamped[0]}", None
        return reported_task_name(context.report), None

    def decide_group(self, key: Hashable, items: tuple[TrainDataItem, ...]) -> GroupDecision:
        arrived = len(items) + self.unassembled_episodes.get(key, 0)
        if isinstance(key, str) and key.startswith(ROUND_KEY):
            sizes = [int(item.metadata["round_plays"]) for item in items if isinstance(item, TrajectoryItem)]
            return GroupDecision.READY if sizes and arrived >= sizes[0] else GroupDecision.INCOMPLETE
        return GroupDecision.READY if arrived >= self.rollouts_per_task else GroupDecision.INCOMPLETE

    def make_batch(self, items: tuple[TrainDataItem, ...], batch_number: int) -> TrainingBatch:
        # The objective reads contiguous task groups, so a round's interleaved plays are ordered by task.
        ordered = tuple(sorted(items, key=lambda item: str(item.group_id)))
        return TrainingBatch(f"{self.scenario}:spade:{batch_number}", ordered)

    # ----------------------------------------------------- the generation half

    def ready(self) -> bool:
        self.catch_up()
        # A complete round is a batch on its own, whatever tasks_per_step says.
        if any(isinstance(key, str) and key.startswith(ROUND_KEY) for key in self.ready_group_keys()):
            return True
        return super().ready()

    def dropped(self, batch_id: str) -> None:
        self._dropped.add(batch_id)

    def acknowledge(self, batch_id: str) -> frozenset[str]:
        consumed = super().acknowledge(batch_id)
        if batch_id in self._dropped:
            self._dropped.discard(batch_id)
        else:
            self._batches_since_generation += 1
        return consumed

    def derivation_pending(self) -> bool:
        return self._in_flight is not None

    def close(self) -> None:
        if self._worker is not None:
            self._worker.close()

    def operational_metrics(self) -> Mapping[str, float | int]:
        return {
            **super().operational_metrics(),
            "generations_completed": self._completed,
            "generation_in_flight": 0 if self._in_flight is None else 1,
        }

    def status(self) -> Mapping[str, Any]:
        return {
            **super().status(),
            "unassembled_episodes": sum(self.unassembled_episodes.values()),
            "generation": {
                "in_flight": self._in_flight,
                "next": self._next_generation,
                "completed": self._completed,
                "of": self.generations,
                "last_error": self._last_error,
            },
        }

    def catch_up(self) -> None:
        """Absorb the generation the worker finished, then start the next one when it is due."""
        if self._worker is None:
            return
        for outcome in self._worker.poll():
            self._in_flight = None
            self._has_landed = True
            if isinstance(outcome, Failed):
                self._last_error = f"{outcome.receipt} did not finish; see the service log"
                logger.error("SPADE %s failed on scenario %r", outcome.receipt, self.scenario)
                continue
            if not isinstance(outcome, GenerationOutcome):
                raise TypeError("the generation worker must hand back a GenerationOutcome")
            self._completed += 1
            self._experience = outcome.record.experience
            self.previous_summary = outcome.record.summary
            self._last_error = outcome.record.error
            self._landed_empty = not outcome.record.measures
            logger.info(
                "SPADE generation %d on scenario %r: %d tasks of %d proposals under %s",
                outcome.record.generation,
                self.scenario,
                len(outcome.record.measures),
                len(outcome.record.proposals),
                outcome.report_path,
            )
        if self._in_flight is not None or self._next_generation >= self.generations:
            return
        is_due = (
            not self._has_landed or self._landed_empty or self._batches_since_generation >= self.batches_per_generation
        )
        if not is_due:
            return
        job = GenerationJob(receipt=generation_label(self._next_generation), generation=self._next_generation)
        if self._worker.submit(job):
            self._in_flight = self._next_generation
            self._next_generation += 1
            # The batches this generation's episodes make count toward the next one.
            self._batches_since_generation = 0
            self._landed_empty = False
        else:
            self._last_error = "the generation worker is closed or broken; no further generation runs"

    async def generate(
        self, request: TaskGenerationRequest, *, skill: str | None = None, index: int = 0
    ) -> HarborTask:
        """One Designer proposal through the generator service; a reply that is no task raises :class:`ProposalRefused`."""
        if self.generator is None:
            raise GeneratorError("this processor has no generator service to ask")
        designer_request = DesignerRequest(
            target=request.description,
            skill=skill,
            difficulty=self.difficulty,
            turn_limit=self.turn_limit,
            grounding=self.grounding,
            experience_text=experience_text(experience_for(self._experience, skill)),
        )
        tags = {"role": "designer", "generation": str(self._generation), **skill_tag(skill)}
        proposed = await self.generator.propose(
            designer_request,
            scenario=self.scenario,
            generation=self._generation,
            index=index,
            tags=tags,
            model=self.served_model,
        )
        if proposed.task is None:
            raise ProposalRefused(proposed.refusal, proposed.record_id)
        return proposed.task

    async def write(self, task: HarborTask) -> WrittenTask:
        """The task written under the generator's root; a name an earlier attempt of the generation took is replaced."""
        if self.generator is None:
            raise GeneratorError("this processor has no generator service to ask")
        try:
            return await self.generator.write_task(task)
        except TaskNameConflict:
            # A reload cancelled this generation's earlier attempt; what it wrote under the name gives way.
            await self.generator.delete_task(task.name)
            return await self.generator.write_task(task)

    async def validate(self, task_path: Path) -> TaskValidationResult:
        """Harbor's oracle and nop agents on the written task: the reference solution scores 1, doing nothing below 1."""
        if self.generator is None:
            raise GeneratorError("this processor has no generator service to ask")
        oracle = await self.generator.check(task_path)
        return TaskValidationResult(errors=() if oracle.is_solvable else (f"oracle check refused: {oracle.reason}",))

    async def run_generation(self, job: GenerationJob) -> GenerationOutcome:
        """The worker's job: ``count`` proposals, each written, checked, played and reported; then the manifest."""
        if self.state_dir is None:
            raise GeneratorError("a generation needs a state_dir for its report")
        self._generation = job.generation
        proposals: list[ProposalRecord] = []
        measures: list[TaskMeasure] = []
        manifest_path: Path | None = None
        error = ""
        try:
            for index in range(self.count):
                skill = self.skills[index % len(self.skills)] if self.skills else None
                proposal, measure = await self.proposed(job.generation, index, skill)
                proposals.append(proposal)
                if measure is not None:
                    measures.append(measure)
            if measures and self.generator is not None:
                manifest_path = await self.generator.write_manifest(
                    generation=job.generation,
                    names=[measure.name for measure in measures],
                    eval_fraction=self.eval_fraction,
                    seed=self.seed,
                )
        except GeneratorError as exc:
            # What ran is kept: the tasks are written and their episodes reported; the report says where it stopped.
            error = str(exc)
            logger.error("SPADE generation %d stopped: %s", job.generation, exc)
        held_plays_reported = None
        if self.report_plays_after_generation:
            # The held plays go out once the loop is over, so what it measured trains even when it stopped early.
            held_plays_reported = 0
            try:
                for arm, plays in held_plays(measures):
                    held_plays_reported += await self.report_held_plays(job.generation, arm, plays)
            except GeneratorError as exc:
                error = error or f"the held plays were not all reported: {exc}"
                logger.error("SPADE generation %d could not report its held plays: %s", job.generation, exc)
        record = GenerationRecord(
            job.generation, tuple(proposals), tuple(measures), manifest_path, error, held_plays_reported
        )
        return GenerationOutcome(
            receipt=job.receipt, record=record, report_path=write_generation_report(self.state_dir, record)
        )

    async def proposed(
        self, generation: int, index: int, skill: str | None
    ) -> tuple[ProposalRecord, TaskMeasure | None]:
        """One proposal from the Designer's call to its report: the task measured, or the refusal."""
        if self.generator is None:
            raise GeneratorError("this processor has no generator service to ask")
        request = TaskGenerationRequest((), self.description)
        try:
            task = await self.generate(request, skill=skill, index=index)
        except ProposalRefused as exc:
            return await self.reported(ProposalRecord(index, skill, exc.record_id, None, str(exc)), None)
        record_id = task.source_agent_record_ids[0]
        try:
            written = await self.write(task)
        except (DuplicateTask, TaskNameConflict) as exc:
            return await self.reported(ProposalRecord(index, skill, record_id, None, f"refused: {exc}"), None)
        validation = await self.validate(written.path)
        if not validation.is_valid:
            await self.generator.delete_task(task.name)
            return await self.reported(
                ProposalRecord(index, skill, record_id, None, "; ".join(validation.errors)), None
            )
        measure, refusal = await self.measured(task, written.path, written.digest, skill, generation)
        if measure is None:
            await self.generator.delete_task(task.name)
            return await self.reported(ProposalRecord(index, skill, record_id, None, refusal), None)
        return await self.reported(ProposalRecord(index, skill, record_id, task.name, ""), measure)

    async def measured(
        self, task: HarborTask, task_path: Path, digest: str, skill: str | None, generation: int
    ) -> tuple[TaskMeasure | None, str]:
        """Both arms played: the plain arm is the training data, the hint arm measures how much the hint helps.

        A task the Reasoning Agent could not play at all (every plain episode ended before the agent ran) is no
        measure of the Reasoning Agent; it comes back as None with the first episode's error. A held play (the
        generation reports after it landed) stays on the measure.
        """
        if self.generator is None:
            raise GeneratorError("this processor has no generator service to ask")
        tags = {"generation": str(generation), **skill_tag(skill)}
        is_held = self.report_plays_after_generation
        plain = await self.generator.play(
            task_path,
            scenario=self.scenario,
            arm=TRAINING_ARM,
            plays=self.rollouts_per_task,
            is_reporting=not is_held,
            extra_instruction_files=(),
            tags=tags,
            model=self.served_model,
        )
        if plain and all(play.reward is None and play.error for play in plain):
            return None, f"the Reasoning Agent could not play the task: {plain[0].error[:300]}"
        hint_plays = self.hint_plays if HINT_NAME in task.solution else 0
        hint = (
            await self.generator.play(
                task_path,
                scenario=self.scenario,
                arm=HINT_ARM,
                plays=hint_plays,
                is_reporting=not is_held,
                extra_instruction_files=(HINT_FILE,),
                tags=tags,
                model=self.served_model,
            )
            if hint_plays
            else ()
        )
        record = PlayRecord(
            name=task.name,
            skill=skill,
            return_without_hint=mean_reward(plain),
            return_with_hint=mean_reward(hint) if hint else mean_reward(plain),
            instruction_excerpt=task.instruction[:RECORDED_EXCERPT_CHARS],
        )
        measure = TaskMeasure(
            name=task.name,
            skill=skill,
            task_path=task_path,
            digest=digest,
            plain_rewards=rewards_of(plain),
            hint_rewards=rewards_of(hint),
            record=record,
            plain_plays=tuple(plain) if is_held else (),
            hint_plays=tuple(hint) if is_held else (),
        )
        return measure, ""

    async def report_held_plays(self, generation: int, arm: str, plays: Sequence[TaskPlay]) -> int:
        """One arm's held plays reported with their rewards, the plain arm stamped as the round; how many were reported."""
        if self.generator is None:
            raise GeneratorError("this processor has no generator service to ask")
        score_of = {play.episode_id: play.reward for play in plays if play.reward is not None and play.receipts}
        if not score_of:
            return 0
        metadata: dict[str, object] = {"arm": arm, "generation": generation}
        if arm == TRAINING_ARM:
            # The processor trains the round as one batch once this many plain reports arrived.
            metadata["round"] = generation_label(generation)
            metadata["round_plays"] = len(score_of)
        reported = await self.generator.report_plays(
            plays, scenario=self.scenario, score_of=score_of, metadata=metadata, model=self.served_model
        )
        return sum(1 for play in reported if play.is_reported)

    async def reported(
        self, proposal: ProposalRecord, measure: TaskMeasure | None
    ) -> tuple[ProposalRecord, TaskMeasure | None]:
        """The Designer's report for one proposal: its raw regret as the score, the refusal floor for a refused one.

        The metadata names the generation and its size, so a Designer processor groups a generation's reports;
        the feedback carries the task's measure and the generation beside the last one's summary.
        """
        if not self.is_reporting_designer or self.generator is None:
            return proposal, measure
        measured: dict[str, object] = (
            {"outcome": None, "regret": None, "return_without_hint": None, "return_with_hint": None}
            if measure is None
            else {
                "outcome": measure.outcome,
                "regret": measure.regret,
                "return_without_hint": measure.record.return_without_hint,
                "return_with_hint": measure.record.return_with_hint,
            }
        )
        metadata: dict[str, object] = {
            "generation": self._generation,
            "proposals": self.count,
            **skill_tag(proposal.skill),
        }
        if measure is None:
            score = REFUSAL_SCORE
            metadata["refusal"] = proposal.refusal
        else:
            score = measure.regret
            metadata["task"] = {"name": measure.name, "path": str(measure.task_path), "digest": measure.digest}
            metadata.update(measured)
        feedback: dict[str, object] = {
            "task": None if measure is None else measure.name,
            "refusal": proposal.refusal,
            **measured,
            "round": {
                "generation": self._generation,
                "previous": None if self.previous_summary is None else self.previous_summary.document(),
            },
        }
        report_id = await self.generator.report_proposal(
            proposal.designer_record_id, scenario=self.scenario, score=score, metadata=metadata, feedback=feedback
        )
        return replace(proposal, designer_report_id=report_id), measure
