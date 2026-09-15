"""One SPADE generation: the Designer proposes, the oracle check refuses, the tasks are written, both arms play, regret splits.

The Designer is the served model: every proposal is one chat call through Reef, so it is a record with
a receipt. Per proposal the reply is parsed, the task is written under the tasks root, Harbor's oracle
and nop agents check it (the reference solution must score 1, doing nothing below 1), and the Reasoning Agent
plays it: ``plays`` times as it is, reported to Reef as training data, and ``hint_plays`` times with the
hint appended to the instruction, measured only. Regret is the mean hint reward minus the mean plain
reward; the plain mean puts the task in a band (mastered, frontier, out of reach). The written tasks are
split by the Designer's record ids into train and eval, the manifest is written beside them, a JSON
report keeps every proposal and measure, and each proposal is reported to Reef against the Designer's
receipt with its regret as the score, which is the Designer's own training signal. The report's
experience feeds the next generation's prompts.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from reef_client.client import ReefClient, ReefClientError

from recipes.beta.spade.designer import (
    DEFAULT_TURN_LIMIT,
    DESIGNER_TIMEOUT_S,
    MAX_EXPERIENCE_RECORDS,
    DesignerPrompt,
    DesignerReplyError,
    DesignerRequest,
    PlayRecord,
    designer_messages,
    parse_harbor_reply,
)
from recipes.beta.spade.harbor import (
    GeneratedHarborTask,
    OracleResult,
    content_hash,
    harbor_task,
    oracle_check,
    split_generation,
)
from recipes.beta.spade.roles import (
    REFUSAL_SCORE,
    AgentReward,
    DesignerReward,
    HarborAgent,
    Role,
    RoleVersion,
    TaskMeasure,
    regret,
    verifier_reward,
)
from reef.core.tasks import (
    HarborTask,
    HarborTaskConflict,
    HarborTaskError,
    read_harbor_task,
    write_harbor_task,
    write_split_manifest,
)
from reef.harness.client.tasks import TaskPlay, TaskPlayer

REPORT_DIRECTORY = ".spade"
INSTRUCTION_EXCERPT_CHARS = 600
CHAT_PATH = "/v1/chat/completions"


class GenerationError(RuntimeError):
    """A generation could not run as asked."""


@dataclass(frozen=True)
class DesignerAnswer:
    """What the Designer said and the record its call left on Reef."""

    text: str
    record_id: str


class Designer(ABC):
    """Who answers a Designer prompt and takes a report about the proposal: the served model through Reef."""

    @abstractmethod
    def answer(self, messages: Sequence[Mapping[str, str]], *, tags: Mapping[str, str]) -> DesignerAnswer: ...

    @abstractmethod
    def report(
        self,
        record_id: str,
        *,
        score: float,
        metadata: Mapping[str, object],
        feedback: str | Mapping[str, object] = "SPADE Designer regret",
    ) -> str: ...


class ReefDesigner(Designer):
    """The served model behind a Reef scenario; each proposal is an inference record, each outcome a report."""

    def __init__(
        self,
        *,
        reef_url: str,
        scenario: str,
        model: str,
        token: str | None = None,
        request_options: Mapping[str, object] | None = None,
        timeout_s: float = DESIGNER_TIMEOUT_S,
    ) -> None:
        self.scenario = scenario
        self.model = model
        # Extra fields of the chat request, e.g. {"reasoning_effort": "none"} for a model that would think for
        # thousands of tokens before writing an environment and run past the service's inference deadline.
        self.request_options = dict(request_options or {})
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or timeout_s <= 0:
            raise GenerationError("timeout_s must be a positive number of seconds")
        # A large local model writes an environment in minutes; the service's own inference deadline must allow it too.
        self.client = ReefClient(reef_url, token=token, timeout_s=float(timeout_s))

    def answer(self, messages: Sequence[Mapping[str, str]], *, tags: Mapping[str, str]) -> DesignerAnswer:
        headers = {f"x-reef-tag-{name}": value for name, value in tags.items()}
        payload = {**self.request_options, "model": self.model, "messages": [dict(message) for message in messages]}
        try:
            body, record_id = self.client.inference_with_record(
                self.scenario, CHAT_PATH, payload, extra_headers=headers
            )
        except ReefClientError as exc:
            raise GenerationError(f"the Designer call was refused ({exc.status}): {exc.body[:300]}") from exc
        except OSError as exc:
            raise GenerationError(
                f"the Designer call did not complete within {self.client.timeout_s:g} s: {exc}"
            ) from exc
        choices = body.get("choices")
        message = choices[0].get("message") if isinstance(choices, list) and choices else None
        text = message.get("content") if isinstance(message, Mapping) else None
        return DesignerAnswer(text=text if isinstance(text, str) else "", record_id=record_id)

    def report(
        self,
        record_id: str,
        *,
        score: float,
        metadata: Mapping[str, object],
        feedback: str | Mapping[str, object] = "SPADE Designer regret",
    ) -> str:
        payload = {
            "score": score,
            "feedback": dict(feedback) if isinstance(feedback, Mapping) else feedback,
            "metadata": dict(metadata),
        }
        try:
            answer = self.client.report(self.scenario, payload, references=[record_id])
        except ReefClientError as exc:
            raise GenerationError(f"the Designer report was refused ({exc.status}): {exc.body[:300]}") from exc
        except OSError as exc:
            raise GenerationError(f"the Designer report did not reach Reef: {exc}") from exc
        return str(answer.get("agent_record_id", ""))


class Checks(ABC):
    """The check before a written task is played: Harbor's oracle and nop agents through the harbor command line."""

    @abstractmethod
    def oracle(self, task_path: Path) -> OracleResult: ...


class RealChecks(Checks):
    def __init__(self, *, harbor: str | None = None) -> None:
        self.harbor = harbor

    def oracle(self, task_path: Path) -> OracleResult:
        return oracle_check(task_path, harbor=self.harbor)


class ReasoningAgent(ABC):
    """Who plays a written task for one arm: the served model through a Harbor agent and the task player."""

    @abstractmethod
    def play(
        self,
        task_path: Path,
        *,
        arm: str,
        plays: int,
        is_reporting: bool,
        extra_instruction_paths: Sequence[Path],
        tags: Mapping[str, str],
    ) -> tuple[TaskPlay, ...]: ...

    @abstractmethod
    def report_plays(
        self, plays: Sequence[TaskPlay], *, reward: AgentReward, metadata: Mapping[str, object]
    ) -> tuple[TaskPlay, ...]:
        """Report played episodes after the fact, each with the score ``reward`` gives it; the plays with their report ids."""


class ReefReasoningAgent(ReasoningAgent):
    """A task player per arm: the plain arm is reported when the generation says so, the hint arm only measures."""

    def __init__(
        self,
        *,
        reef_url: str,
        scenario: str,
        model: str,
        work_dir: Path,
        token: str | None = None,
        agent: Mapping[str, object] | None = None,
        agent_host: str | None = None,
        concurrency: int = 2,
    ) -> None:
        self.reef_url = reef_url
        self.scenario = scenario
        self.model = model
        self.work_dir = Path(work_dir)
        self.token = token
        self.agent = agent
        self.agent_host = agent_host
        self.concurrency = concurrency

    def play(
        self,
        task_path: Path,
        *,
        arm: str,
        plays: int,
        is_reporting: bool,
        extra_instruction_paths: Sequence[Path],
        tags: Mapping[str, str],
    ) -> tuple[TaskPlay, ...]:
        if plays < 1:
            return ()
        player = self.player(labels={**tags, "arm": arm}, extra_instruction_paths=extra_instruction_paths)
        player.is_reporting = is_reporting
        return player.play_concurrently([task_path] * plays, concurrency=min(self.concurrency, plays))

    def report_plays(
        self, plays: Sequence[TaskPlay], *, reward: AgentReward, metadata: Mapping[str, object]
    ) -> tuple[TaskPlay, ...]:
        players: dict[tuple[tuple[str, str], ...], TaskPlayer] = {}
        reported = []
        for play in plays:
            score = reward.score(play)
            if score is None or not play.receipts:
                reported.append(play)
                continue
            key = tuple(sorted(play.labels.items()))
            if key not in players:
                players[key] = self.player(labels=play.labels, extra_instruction_paths=())
            report_ids = players[key].report_play(play, score=score, metadata=metadata)
            reported.append(replace(play, report_agent_record_ids=report_ids))
        return tuple(reported)

    def player(self, *, labels: Mapping[str, str], extra_instruction_paths: Sequence[Path]) -> TaskPlayer:
        return TaskPlayer(
            reef_url=self.reef_url,
            scenario=self.scenario,
            model=self.model,
            work_dir=self.work_dir,
            token=self.token,
            agent=self.agent,
            agent_host=self.agent_host,
            labels=labels,
            extra_instruction_paths=extra_instruction_paths,
        )


def reef_designer(role: Role) -> Designer:
    """The Designer a role names: the served model behind its scenario, with its prompt's request options and timeout."""
    if not isinstance(role.harness, DesignerPrompt):
        raise GenerationError("the Designer role's harness must be a DesignerPrompt")
    return ReefDesigner(
        reef_url=role.reef_url,
        scenario=role.scenario,
        model=role.model,
        token=role.token,
        request_options=role.harness.request_options,
        timeout_s=role.harness.timeout_s,
    )


def reef_reasoning_agent(role: Role, *, work_dir: Path) -> ReasoningAgent:
    """The Reasoning Agent a role names: a task player for its scenario and model with its Harbor agent."""
    if not isinstance(role.harness, HarborAgent):
        raise GenerationError("the Reasoning Agent role's harness must be a HarborAgent")
    return ReefReasoningAgent(
        reef_url=role.reef_url,
        scenario=role.scenario,
        model=role.model,
        work_dir=work_dir,
        token=role.token,
        agent=role.harness.bound_spec(),
        agent_host=role.harness.host,
        concurrency=role.harness.concurrency,
    )


@dataclass(frozen=True)
class GenerationRequest:
    """What one generation asks the Designer for and how the Reasoning Agent measures what it gets."""

    description: str
    skills: tuple[str, ...] = ()
    count: int = 8
    generation: int = 0
    difficulty: str = "medium"
    turn_limit: int = DEFAULT_TURN_LIMIT
    grounding: str | None = None
    experience: tuple[PlayRecord, ...] = ()
    plays: int = 4
    hint_plays: int = 2
    eval_fraction: float = 0.25
    seed: int = 0
    prompt: DesignerPrompt = field(default_factory=DesignerPrompt)
    designer_version: RoleVersion | None = None
    agent_version: RoleVersion | None = None
    previous: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.description, str) or not self.description.strip():
            raise GenerationError("description must be non-empty text")
        if not isinstance(self.prompt, DesignerPrompt):
            raise GenerationError("prompt must be a DesignerPrompt")
        for label, value in (("designer_version", self.designer_version), ("agent_version", self.agent_version)):
            if value is not None and not isinstance(value, RoleVersion):
                raise GenerationError(f"{label} must be a RoleVersion when set")
        if self.previous is not None and not isinstance(self.previous, Mapping):
            raise GenerationError("previous must be the last round's record, a mapping, when set")
        if not isinstance(self.skills, tuple) or not all(isinstance(skill, str) and skill for skill in self.skills):
            raise GenerationError(
                "skills must be a tuple of skill names; empty when the description alone is the target"
            )
        for label, value in (("count", self.count), ("generation", self.generation), ("seed", self.seed)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise GenerationError(f"{label} must be a non-negative integer")
        if self.count < 1:
            raise GenerationError("count must be at least 1")
        for label, value in (("plays", self.plays), ("hint_plays", self.hint_plays)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise GenerationError(f"{label} must be a non-negative integer")
        if self.plays < 1:
            raise GenerationError("plays must be at least 1: the plain arm is the training data")
        if not isinstance(self.eval_fraction, (int, float)) or not 0 <= self.eval_fraction < 1:
            raise GenerationError("eval_fraction must be in [0, 1)")


@dataclass(frozen=True)
class Proposal:
    """One Designer call and what became of it."""

    index: int
    skill: str | None
    designer_record_id: str
    task_name: str | None
    refusal: str
    designer_report_id: str = ""

    @property
    def is_written(self) -> bool:
        return self.task_name is not None and not self.refusal


@dataclass(frozen=True)
class GenerationResult:
    """What one generation produced: the tasks, their measures, the manifest and the report on disk."""

    generation: int
    tasks_root: Path
    manifest_path: Path | None
    report_path: Path
    proposals: tuple[Proposal, ...]
    measures: tuple[TaskMeasure, ...]

    @property
    def experience(self) -> tuple[PlayRecord, ...]:
        return tuple(measure.record for measure in self.measures)


def round_document(request: GenerationRequest, measures: Sequence[TaskMeasure], *, refused: int) -> dict[str, object]:
    """What a generation looked like as a whole, carried by every Designer report of it beside the last round's record."""
    return {
        "generation": request.generation,
        "mean_regret": statistics.fmean(measure.regret for measure in measures) if measures else None,
        "measured": len(measures),
        "refused": refused,
        "previous": dict(request.previous) if request.previous is not None else None,
    }


def version_document(version: RoleVersion | None) -> dict[str, str] | None:
    """A role version as it travels in report metadata and on disk: its kind and id, or None for an unknown one."""
    return None if version is None else {"kind": version.kind, "id": version.id}


def skill_tag(skill: str | None) -> dict[str, str]:
    """The skill as a tag or metadata entry, absent when the generation has no skill axis."""
    return {} if skill is None else {"skill": skill}


def is_unplayed(play: TaskPlay) -> bool:
    """An episode that never ran: no reward and an error (the agent could not start, the trial raised)."""
    return play.reward is None and bool(play.error)


def mean_reward(plays: Sequence[TaskPlay]) -> float:
    """The mean reward of the episodes of an arm that ran; a run that scored nothing counts as 0, like a loss."""
    ran = [play for play in plays if not is_unplayed(play)]
    if not ran:
        return 0.0
    return statistics.fmean(play.reward if play.reward is not None else 0.0 for play in ran)


def experience_for(
    records: Sequence[PlayRecord], skill: str | None, limit: int = MAX_EXPERIENCE_RECORDS
) -> tuple[PlayRecord, ...]:
    """The records a prompt gets: this skill's first when there is one, the frontier by regret, then the two bands."""
    rank = {"frontier": 0, "out_of_reach": 1, "mastered": 2}
    ordered = sorted(
        records,
        key=lambda record: (skill is not None and record.skill != skill, rank[record.outcome], -record.regret),
    )
    return tuple(ordered[:limit])


def load_experience(report_path: Path) -> tuple[PlayRecord, ...]:
    """The play records a generation report holds, for the next generation's prompts."""
    try:
        document = json.loads(Path(report_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GenerationError(f"{report_path} is not a generation report: {exc}") from exc
    records = document.get("experience") if isinstance(document, dict) else None
    if not isinstance(records, list):
        raise GenerationError(f"{report_path} holds no experience")
    try:
        return tuple(PlayRecord(**record) for record in records)
    except (TypeError, ValueError) as exc:
        raise GenerationError(f"{report_path} holds a record the Designer cannot take: {exc}") from exc


class Generation:
    """Runs one generation against a Designer, a ReasoningAgent and the checks, under a tasks root."""

    def __init__(
        self,
        *,
        designer: Designer,
        reasoning_agent: ReasoningAgent,
        checks: Checks,
        tasks_root: Path,
        is_reporting_designer: bool = True,
        is_reporting_agent: bool = True,
        designer_reward: DesignerReward = regret,
        agent_reward: AgentReward = verifier_reward,
    ) -> None:
        self.designer = designer
        self.reasoning_agent = reasoning_agent
        self.checks = checks
        self.tasks_root = Path(tasks_root)
        self.is_reporting_designer = is_reporting_designer
        # A rounds driver holds the plain arm's episodes and reports them once the Designer's side has stepped.
        self.is_reporting_agent = is_reporting_agent
        self.designer_reward = designer_reward
        self.agent_reward = agent_reward

    def run(self, request: GenerationRequest) -> GenerationResult:
        self.tasks_root.mkdir(parents=True, exist_ok=True)
        known_hashes = self.known_hashes()
        # Every Designer report waits for the end: an aborted generation reports nothing, and a step never mixes two.
        pending: list[tuple[Proposal, TaskMeasure | None]] = []
        measures: list[TaskMeasure] = []
        for index in range(request.count):
            skill = request.skills[index % len(request.skills)] if request.skills else None
            tags = {"role": "designer", "generation": str(request.generation), **skill_tag(skill)}
            designer_request = DesignerRequest(
                skill=skill,
                skill_description=request.description,
                difficulty=request.difficulty,
                turn_limit=request.turn_limit,
                grounding=request.grounding,
                experience=experience_for(request.experience, skill),
            )
            answer = self.designer.answer(designer_messages(designer_request, request.prompt), tags=tags)
            written, refusal = self.written_task(answer, skill, index, request, known_hashes)
            if written is None:
                pending.append((Proposal(index, skill, answer.record_id, None, refusal), None))
                continue
            measure, refusal = self.measured(written, skill, request)
            if measure is None:
                shutil.rmtree(self.tasks_root / written.name, ignore_errors=True)
                known_hashes.discard(content_hash(written))
                pending.append((Proposal(index, skill, answer.record_id, None, refusal), None))
                continue
            measures.append(measure)
            pending.append((Proposal(index, skill, answer.record_id, measure.name, ""), measure))
        round_record = round_document(request, measures, refused=sum(1 for _, measure in pending if measure is None))
        proposals = [self.reported(proposal, request, measure, round_record) for proposal, measure in pending]
        manifest_path = self.written_manifest(measures, request)
        report_path = self.written_report(request, proposals, measures, manifest_path)
        return GenerationResult(
            generation=request.generation,
            tasks_root=self.tasks_root,
            manifest_path=manifest_path,
            report_path=report_path,
            proposals=tuple(proposals),
            measures=tuple(measures),
        )

    def known_hashes(self) -> set[str]:
        """The content hashes of the tasks already under the root, so a generation never writes one twice."""
        hashes = set()
        for entry in sorted(self.tasks_root.iterdir()):
            if entry.is_dir() and (entry / "task.toml").is_file():
                try:
                    hashes.add(content_hash(read_harbor_task(entry)))
                except HarborTaskError:
                    continue
        return hashes

    def written_task(
        self,
        answer: DesignerAnswer,
        skill: str | None,
        index: int,
        request: GenerationRequest,
        known_hashes: set[str],
    ) -> tuple[HarborTask | None, str]:
        """Parse, write and check one proposal; the written task, or None and why it was refused."""
        try:
            reply = parse_harbor_reply(answer.text)
            generated = GeneratedHarborTask(
                reply=reply,
                skill=skill,
                generation=request.generation,
                index=index,
                source_record_id=answer.record_id,
                difficulty=request.difficulty,
            )
            task = harbor_task(generated)
        except (DesignerReplyError, ValueError) as exc:
            return None, f"reply refused: {exc}"
        digest = content_hash(task)
        if digest in known_hashes:
            return None, "duplicate of a task already under the root"
        try:
            task_path = write_harbor_task(task, self.tasks_root)
        except HarborTaskConflict as exc:
            return None, f"a different task holds the name {task.name}: {exc}"
        oracle = self.checks.oracle(task_path)
        if not oracle.is_solvable:
            shutil.rmtree(task_path, ignore_errors=True)
            return None, f"oracle check refused: {oracle.reason}"
        known_hashes.add(digest)
        return task, ""

    def measured(
        self, task: HarborTask, skill: str | None, request: GenerationRequest
    ) -> tuple[TaskMeasure | None, str]:
        """Both arms played: the plain arm is the training data, reported with the agent's reward; the hint arm only measures.

        A task the Reasoning Agent could not play at all (every plain episode ended before the agent ran) is no
        measure of the Reasoning Agent; it comes back as None with the first episode's error.
        """
        task_path = self.tasks_root / task.name
        tags = {"generation": str(request.generation), **skill_tag(skill)}
        plain = self.reasoning_agent.play(
            task_path, arm="plain", plays=request.plays, is_reporting=False, extra_instruction_paths=(), tags=tags
        )
        if plain and all(is_unplayed(play) for play in plain):
            return None, f"the Reasoning Agent could not play the task: {plain[0].error[:300]}"
        hint_path = task_path / "solution" / "hint.txt"
        hint = self.reasoning_agent.play(
            task_path,
            arm="hint",
            plays=request.hint_plays,
            is_reporting=False,
            extra_instruction_paths=(hint_path,),
            tags=tags,
        )
        if self.is_reporting_agent:
            # The hint arm is reported too, so its records are released; the agent's processor discards that arm.
            plain = self.reasoning_agent.report_plays(
                plain, reward=self.agent_reward, metadata=self.agent_report_metadata(request, "plain")
            )
            hint = self.reasoning_agent.report_plays(
                hint, reward=self.agent_reward, metadata=self.agent_report_metadata(request, "hint")
            )
        record = PlayRecord(
            name=task.name,
            skill=skill,
            return_without_hint=mean_reward(plain),
            return_with_hint=mean_reward(hint) if hint else mean_reward(plain),
            instruction_excerpt=task.instruction[:INSTRUCTION_EXCERPT_CHARS],
        )
        measure = TaskMeasure(
            name=task.name,
            skill=skill,
            task_path=task_path,
            digest=task.digest,
            plain_rewards=tuple(
                play.reward if play.reward is not None else 0.0 for play in plain if not is_unplayed(play)
            ),
            hint_rewards=tuple(
                play.reward if play.reward is not None else 0.0 for play in hint if not is_unplayed(play)
            ),
            record=record,
            plain_plays=tuple(plain),
            hint_plays=tuple(hint),
        )
        return measure, ""

    def agent_report_metadata(self, request: GenerationRequest, arm: str) -> dict[str, object]:
        """What every Reasoning Agent report carries beside the task: the arm, the generation and the Designer version."""
        return {
            "arm": arm,
            "generation": request.generation,
            "designer_version": version_document(request.designer_version),
        }

    def reported(
        self,
        proposal: Proposal,
        request: GenerationRequest,
        measure: TaskMeasure | None,
        round_record: Mapping[str, object],
    ) -> Proposal:
        """The Designer's report for one proposal: the Designer reward of its measure as the score, below any measure for a refusal."""
        if not self.is_reporting_designer:
            return proposal
        metadata: dict[str, object] = {
            "generation": request.generation,
            **skill_tag(proposal.skill),
            "proposals": request.count,
            "designer_version": version_document(request.designer_version),
            "opponent": {"role": "reasoning_agent", "version": version_document(request.agent_version)},
        }
        feedback: dict[str, object] = {
            "task": measure.name if measure is not None else None,
            "refusal": proposal.refusal,
            "round": dict(round_record),
        }
        if measure is None:
            score = REFUSAL_SCORE
            metadata["refusal"] = proposal.refusal
        else:
            score = self.designer_reward.score(measure)
            metadata["task"] = {"name": measure.name, "path": str(measure.task_path), "digest": measure.digest}
            metadata["outcome"] = measure.outcome
            metadata["regret"] = measure.regret
            metadata["return_without_hint"] = measure.record.return_without_hint
            metadata["return_with_hint"] = measure.record.return_with_hint
            feedback.update(
                outcome=measure.outcome,
                regret=measure.regret,
                return_without_hint=measure.record.return_without_hint,
                return_with_hint=measure.record.return_with_hint,
            )
        report_id = self.designer.report(
            proposal.designer_record_id, score=score, metadata=metadata, feedback=feedback
        )
        return Proposal(
            proposal.index,
            proposal.skill,
            proposal.designer_record_id,
            proposal.task_name,
            proposal.refusal,
            report_id,
        )

    def written_manifest(self, measures: Sequence[TaskMeasure], request: GenerationRequest) -> Path | None:
        if not measures:
            return None
        tasks = [read_harbor_task(measure.task_path) for measure in measures]
        split = split_generation(tasks, eval_fraction=request.eval_fraction, seed=request.seed)
        manifest_path = self.tasks_root / f"manifest-{request.generation:05d}.json"
        write_split_manifest(manifest_path, split)
        return manifest_path

    def written_report(
        self,
        request: GenerationRequest,
        proposals: Sequence[Proposal],
        measures: Sequence[TaskMeasure],
        manifest_path: Path | None,
    ) -> Path:
        directory = self.tasks_root / REPORT_DIRECTORY
        directory.mkdir(parents=True, exist_ok=True)
        report_path = directory / f"generation-{request.generation:05d}.json"
        request_fields = {
            name: value
            for name, value in asdict(request).items()
            if name not in ("experience", "grounding", "prompt", "designer_version", "agent_version", "previous")
        }
        document = {
            "generation": request.generation,
            "request": {
                **request_fields,
                "experience_records": len(request.experience),
                "grounding_chars": len(request.grounding or ""),
                "prompt_entries": [str(entry["id"]) for entry in request.prompt.entries()],
                "designer_version": version_document(request.designer_version),
                "agent_version": version_document(request.agent_version),
                "previous": dict(request.previous) if request.previous is not None else None,
            },
            "manifest": str(manifest_path) if manifest_path is not None else None,
            "proposals": [asdict(proposal) for proposal in proposals],
            "tasks": [
                {
                    "name": measure.name,
                    "skill": measure.skill,
                    "path": str(measure.task_path),
                    "digest": measure.digest,
                    "plain_rewards": list(measure.plain_rewards),
                    "hint_rewards": list(measure.hint_rewards),
                    "regret": measure.regret,
                    "outcome": measure.outcome,
                }
                for measure in measures
            ],
            "experience": [asdict(measure.record) for measure in measures],
        }
        report_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        return report_path


def add_generation_arguments(parser: argparse.ArgumentParser) -> None:
    """The flags one generation takes: the two roles, the request, and how the plays and checks run."""
    parser.add_argument("--reef-url", required=True)
    parser.add_argument(
        "--scenario", required=True, help="the scenario the Designer's and the Reasoning Agent's records belong to"
    )
    parser.add_argument("--model", required=True, help="the served model, Designer and Reasoning Agent alike")
    parser.add_argument("--token", default=os.environ.get("REEF_TOKEN") or None)
    parser.add_argument("--designer-reef-url", default=None, help="the Designer's Reef service; --reef-url by default")
    parser.add_argument("--designer-scenario", default=None, help="the Designer's scenario; --scenario by default")
    parser.add_argument("--designer-model", default=None, help="the Designer's served model; --model by default")
    parser.add_argument("--designer-token", default=None, help="the Designer's service token; --token by default")
    parser.add_argument("--tasks-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("work/spade"))
    parser.add_argument("--description", required=True, help="what the environments are about")
    parser.add_argument(
        "--skills", default="", help="comma separated skill names, cycled over the proposals; none by default"
    )
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--generation", type=int, default=0)
    parser.add_argument("--difficulty", default="medium")
    parser.add_argument("--turn-limit", type=int, default=DEFAULT_TURN_LIMIT)
    parser.add_argument("--grounding", type=Path, default=None, help="a text file the Designer grounds in")
    parser.add_argument("--experience", type=Path, default=None, help="the previous generation's report")
    parser.add_argument("--plays", type=int, default=4)
    parser.add_argument("--hint-plays", type=int, default=2)
    parser.add_argument("--eval-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--designer-timeout-s",
        type=float,
        default=DESIGNER_TIMEOUT_S,
        help="seconds one Designer call may take; the service's inference.timeout_s must allow it too",
    )
    parser.add_argument(
        "--designer-json",
        default=None,
        help='extra fields of the Designer\'s chat request as JSON, e.g. {"reasoning_effort": "none"}',
    )
    parser.add_argument("--agent-json", default=None, help="the Reasoning Agent's Harbor agent; terminus-2 by default")
    parser.add_argument("--agent-host", default=None)
    parser.add_argument("--concurrency", type=int, default=2, help="episodes in flight per arm")
    parser.add_argument("--harbor", default=None, help="the harbor command line for the oracle check")
    parser.add_argument(
        "--no-designer-report", action="store_true", help="do not report regret against the Designer's calls"
    )


def request_from_arguments(arguments: argparse.Namespace) -> GenerationRequest:
    """The generation request the flags describe; GenerationError, ValueError or OSError when they cannot be read."""
    return GenerationRequest(
        description=arguments.description,
        skills=tuple(part.strip() for part in arguments.skills.split(",") if part.strip()),
        count=arguments.count,
        generation=arguments.generation,
        difficulty=arguments.difficulty,
        turn_limit=arguments.turn_limit,
        grounding=arguments.grounding.read_text(encoding="utf-8") if arguments.grounding else None,
        experience=load_experience(arguments.experience) if arguments.experience else (),
        plays=arguments.plays,
        hint_plays=arguments.hint_plays,
        eval_fraction=arguments.eval_fraction,
        seed=arguments.seed,
    )


def roles_from_arguments(arguments: argparse.Namespace) -> tuple[Role, Role]:
    """The Designer and the Reasoning Agent the flags name; the Designer falls back to the shared service and model."""
    agent = json.loads(arguments.agent_json) if arguments.agent_json else None
    if agent is not None and not isinstance(agent, dict):
        raise GenerationError("--agent-json must hold an object")
    designer_options = json.loads(arguments.designer_json) if arguments.designer_json else {}
    if not isinstance(designer_options, dict):
        raise GenerationError("--designer-json must hold an object")
    designer = Role.designer(
        arguments.designer_reef_url or arguments.reef_url,
        arguments.designer_scenario or arguments.scenario,
        arguments.designer_model or arguments.model,
        prompt=DesignerPrompt(request_options=designer_options, timeout_s=arguments.designer_timeout_s),
        token=arguments.designer_token or arguments.token,
    )
    agent_name = str(agent.get("name") or agent.get("import_path") or "terminus-2") if agent else "terminus-2"
    reasoning_agent = Role.reasoning_agent(
        arguments.reef_url,
        arguments.scenario,
        arguments.model,
        agent=HarborAgent(name=agent_name, spec=agent, host=arguments.agent_host, concurrency=arguments.concurrency),
        token=arguments.token,
    )
    return designer, reasoning_agent


def main(
    argv: Sequence[str] | None = None,
    *,
    designer: Designer | None = None,
    reasoning_agent: ReasoningAgent | None = None,
    checks: Checks | None = None,
) -> int:
    """Run one generation from the command line and print one line per proposal."""
    parser = argparse.ArgumentParser(
        prog="python -m recipes.beta.spade.generation",
        description="One SPADE generation: propose, check, write, play both arms, split, report.",
    )
    add_generation_arguments(parser)
    arguments = parser.parse_args(argv)
    try:
        request = request_from_arguments(arguments)
        designer_role, agent_role = roles_from_arguments(arguments)
    except (GenerationError, ValueError, OSError) as exc:
        parser.error(str(exc))
    generation = Generation(
        designer=designer if designer is not None else reef_designer(designer_role),
        reasoning_agent=(
            reasoning_agent
            if reasoning_agent is not None
            else reef_reasoning_agent(agent_role, work_dir=arguments.work_dir)
        ),
        checks=checks if checks is not None else RealChecks(harbor=arguments.harbor),
        tasks_root=arguments.tasks_root,
        is_reporting_designer=not arguments.no_designer_report,
        designer_reward=designer_role.reward,
        agent_reward=agent_role.reward,
    )
    try:
        result = generation.run(request)
    except GenerationError as exc:
        parser.error(str(exc))
    measures = {measure.name: measure for measure in result.measures}
    for proposal in result.proposals:
        line: dict[str, object] = {"index": proposal.index, **skill_tag(proposal.skill)}
        if proposal.is_written and proposal.task_name in measures:
            measure = measures[proposal.task_name]
            line.update(
                task=measure.name,
                plain=measure.record.return_without_hint,
                hint=measure.record.return_with_hint,
                regret=measure.regret,
                outcome=measure.outcome,
            )
        else:
            line["refusal"] = proposal.refusal
        print(json.dumps(line), flush=True)
    print(
        json.dumps(
            {
                "manifest": str(result.manifest_path) if result.manifest_path else None,
                "report": str(result.report_path),
            }
        )
    )
    return 0 if result.measures else 1


if __name__ == "__main__":
    sys.exit(main())
