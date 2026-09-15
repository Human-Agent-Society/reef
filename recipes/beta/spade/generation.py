"""One SPADE generation: the Designer proposes, the checks refuse, the tasks are written, both arms play, regret splits.

The Designer is the served model: every proposal is one chat call through Reef, so it is a record with
a receipt. Per proposal the kind's reply is parsed, the kind's check runs (``gym``: the smoke test on the
host; ``harbor``: the oracle and nop agents through the harbor command line; ``openenv``: the docker
serve check), the task is written under the tasks root, and the solver plays it: ``plays`` times as it
is, reported to Reef as training data, and ``hint_plays`` times with the hint appended to the
instruction, measured only. Regret is the mean hint reward minus the mean plain reward; the plain mean
puts the task in a band (mastered, frontier, out of reach). The written tasks are split by the
Designer's record ids into train and eval, the manifest is written beside them, a JSON report keeps
every proposal and measure, and each proposal is reported to Reef against the Designer's receipt with
its regret as the score, which is the Designer's own training signal. The report's experience feeds
the next generation's prompts.
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
from dataclasses import asdict, dataclass, field
from pathlib import Path

from reef_client.client import ReefClient, ReefClientError

from recipes.beta.spade.designer import (
    KINDS,
    MAX_EXPERIENCE_RECORDS,
    DesignerReplyError,
    DesignerRequest,
    PlayRecord,
    SmokeResult,
    designer_messages,
    parse_gym_reply,
    parse_harbor_reply,
    parse_openenv_reply,
    smoke_test,
)
from recipes.beta.spade.harbor import GeneratedHarborTask, OracleResult, content_hash, harbor_task, oracle_check
from recipes.beta.spade.openenv import GeneratedOpenEnvTask, OpenEnvCheck, openenv_check, openenv_task
from recipes.beta.spade.tasks import DEFAULT_MAX_TURNS, GeneratedEnvironment, environment_task, split_generation
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
CODE_EXCERPT_CHARS = 600
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
    def report(self, record_id: str, *, score: float, metadata: Mapping[str, object]) -> str: ...


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
    ) -> None:
        self.scenario = scenario
        self.model = model
        # Extra fields of the chat request, e.g. {"reasoning_effort": "none"} for a model that would think for
        # thousands of tokens before writing an environment and run past the service's inference deadline.
        self.request_options = dict(request_options or {})
        self.client = ReefClient(reef_url, token=token, timeout_s=600.0)

    def answer(self, messages: Sequence[Mapping[str, str]], *, tags: Mapping[str, str]) -> DesignerAnswer:
        headers = {f"x-reef-tag-{name}": value for name, value in tags.items()}
        payload = {**self.request_options, "model": self.model, "messages": [dict(message) for message in messages]}
        try:
            body, record_id = self.client.inference_with_record(
                self.scenario, CHAT_PATH, payload, extra_headers=headers
            )
        except ReefClientError as exc:
            raise GenerationError(f"the Designer call was refused ({exc.status}): {exc.body[:300]}") from exc
        choices = body.get("choices")
        message = choices[0].get("message") if isinstance(choices, list) and choices else None
        text = message.get("content") if isinstance(message, Mapping) else None
        return DesignerAnswer(text=text if isinstance(text, str) else "", record_id=record_id)

    def report(self, record_id: str, *, score: float, metadata: Mapping[str, object]) -> str:
        payload = {"score": score, "feedback": "SPADE Designer regret", "metadata": dict(metadata)}
        try:
            answer = self.client.report(self.scenario, payload, references=[record_id])
        except ReefClientError as exc:
            raise GenerationError(f"the Designer report was refused ({exc.status}): {exc.body[:300]}") from exc
        return str(answer.get("agent_record_id", ""))


class Checks(ABC):
    """The kind's check before a proposal becomes a task; the real ones need the host, harbor and docker."""

    @abstractmethod
    def smoke(self, code: str, *, seed: int, max_turns: int) -> SmokeResult: ...

    @abstractmethod
    def oracle(self, task_path: Path) -> OracleResult: ...

    @abstractmethod
    def serving(self, task_path: Path, action_example: Mapping[str, object]) -> OpenEnvCheck: ...


class RealChecks(Checks):
    def __init__(self, *, harbor: str | None = None) -> None:
        self.harbor = harbor

    def smoke(self, code: str, *, seed: int, max_turns: int) -> SmokeResult:
        return smoke_test(code, seed=seed, max_turns=max_turns)

    def oracle(self, task_path: Path) -> OracleResult:
        return oracle_check(task_path, harbor=self.harbor)

    def serving(self, task_path: Path, action_example: Mapping[str, object]) -> OpenEnvCheck:
        return openenv_check(task_path, action_example=dict(action_example))


class Solver(ABC):
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


class ReefSolver(Solver):
    """A task player per arm: the plain arm reports its episodes, the hint arm only measures."""

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
        player = TaskPlayer(
            reef_url=self.reef_url,
            scenario=self.scenario,
            model=self.model,
            work_dir=self.work_dir,
            token=self.token,
            agent=self.agent,
            agent_host=self.agent_host,
            labels={**tags, "arm": arm},
            extra_instruction_paths=extra_instruction_paths,
            is_reporting=is_reporting,
        )
        return player.play_concurrently([task_path] * plays, concurrency=min(self.concurrency, plays))


@dataclass(frozen=True)
class GenerationRequest:
    """What one generation asks the Designer for and how the solver measures what it gets."""

    description: str
    skills: tuple[str, ...]
    kinds: tuple[str, ...] = KINDS
    count: int = 8
    generation: int = 0
    difficulty: str = "medium"
    turn_limit: int = DEFAULT_MAX_TURNS
    grounding: str | None = None
    experience: tuple[PlayRecord, ...] = ()
    plays: int = 4
    hint_plays: int = 2
    eval_fraction: float = 0.25
    seed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.description, str) or not self.description.strip():
            raise GenerationError("description must be non-empty text")
        if not self.skills or not all(isinstance(skill, str) and skill for skill in self.skills):
            raise GenerationError("skills must name at least one skill")
        if not self.kinds or any(kind not in KINDS for kind in self.kinds):
            raise GenerationError(f"kinds must be among {', '.join(KINDS)}")
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
    kind: str
    skill: str
    designer_record_id: str
    task_name: str | None
    refusal: str
    designer_report_id: str = ""

    @property
    def is_written(self) -> bool:
        return self.task_name is not None and not self.refusal


@dataclass(frozen=True)
class TaskMeasure:
    """A written task after both arms played: the rewards, the regret and the band."""

    name: str
    kind: str
    skill: str
    task_path: Path
    digest: str
    plain_rewards: tuple[float, ...]
    hint_rewards: tuple[float, ...]
    record: PlayRecord

    @property
    def regret(self) -> float:
        return self.record.regret

    @property
    def outcome(self) -> str:
        return self.record.outcome


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


@dataclass(frozen=True)
class WrittenTask:
    task: HarborTask
    code_excerpt: str
    action_example: Mapping[str, object] = field(default_factory=dict)


def mean_reward(plays: Sequence[TaskPlay]) -> float:
    """The mean reward of an arm; an unscored episode counts as 0, like a loss."""
    if not plays:
        return 0.0
    return statistics.fmean(play.reward if play.reward is not None else 0.0 for play in plays)


def experience_for(
    records: Sequence[PlayRecord], skill: str, limit: int = MAX_EXPERIENCE_RECORDS
) -> tuple[PlayRecord, ...]:
    """The records a prompt gets: this skill's first, the frontier by regret, then the two bands as examples."""
    rank = {"frontier": 0, "out_of_reach": 1, "mastered": 2}
    ordered = sorted(records, key=lambda record: (record.skill != skill, rank[record.outcome], -record.regret))
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
    """Runs one generation against a Designer, a Solver and the checks, under a tasks root."""

    def __init__(
        self,
        *,
        designer: Designer,
        solver: Solver,
        checks: Checks,
        tasks_root: Path,
        is_reporting_designer: bool = True,
    ) -> None:
        self.designer = designer
        self.solver = solver
        self.checks = checks
        self.tasks_root = Path(tasks_root)
        self.is_reporting_designer = is_reporting_designer

    def run(self, request: GenerationRequest) -> GenerationResult:
        self.tasks_root.mkdir(parents=True, exist_ok=True)
        known_hashes = self.known_hashes()
        proposals: list[Proposal] = []
        measures: list[TaskMeasure] = []
        for index in range(request.count):
            kind = request.kinds[index % len(request.kinds)]
            skill = request.skills[index % len(request.skills)]
            tags = {"role": "designer", "generation": str(request.generation), "kind": kind, "skill": skill}
            designer_request = DesignerRequest(
                kind=kind,
                skill=skill,
                skill_description=request.description,
                difficulty=request.difficulty,
                turn_limit=request.turn_limit,
                grounding=request.grounding,
                experience=experience_for(request.experience, skill),
            )
            answer = self.designer.answer(designer_messages(designer_request), tags=tags)
            written, refusal = self.written_task(kind, answer, skill, index, request, known_hashes)
            if written is None:
                proposal = Proposal(index, kind, skill, answer.record_id, None, refusal)
                proposals.append(self.reported(proposal, request, None))
                continue
            measure = self.measured(written, kind, skill, request)
            measures.append(measure)
            proposal = Proposal(index, kind, skill, answer.record_id, measure.name, "")
            proposals.append(self.reported(proposal, request, measure))
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
        kind: str,
        answer: DesignerAnswer,
        skill: str,
        index: int,
        request: GenerationRequest,
        known_hashes: set[str],
    ) -> tuple[WrittenTask | None, str]:
        """Parse, check and write one proposal; the written task, or None and why it was refused."""
        try:
            prepared = self.prepared_task(kind, answer, skill, index, request)
        except (DesignerReplyError, ValueError) as exc:
            return None, f"{kind} reply refused: {exc}"
        if prepared is None:
            return None, "refused"
        digest = content_hash(prepared.task)
        if digest in known_hashes:
            return None, "duplicate of a task already under the root"
        try:
            task_path = write_harbor_task(prepared.task, self.tasks_root)
        except HarborTaskConflict as exc:
            return None, f"a different task holds the name {prepared.task.name}: {exc}"
        if kind == "harbor":
            oracle = self.checks.oracle(task_path)
            if not oracle.is_solvable:
                shutil.rmtree(task_path, ignore_errors=True)
                return None, f"oracle check refused: {oracle.reason}"
        elif kind == "openenv":
            serving = self.checks.serving(task_path, prepared.action_example)
            if not serving.is_serving:
                shutil.rmtree(task_path, ignore_errors=True)
                return None, f"serve check refused: {serving.reason}"
        known_hashes.add(digest)
        return prepared, ""

    def prepared_task(
        self, kind: str, answer: DesignerAnswer, skill: str, index: int, request: GenerationRequest
    ) -> WrittenTask | None:
        """The kind's reply parsed and, for gym, smoke tested; None when the smoke test refused it."""
        seed = request.seed * 1000 + index
        if kind == "gym":
            reply = parse_gym_reply(answer.text)
            smoke = self.checks.smoke(reply.code, seed=seed, max_turns=request.turn_limit)
            if not smoke.is_runnable:
                raise ValueError(f"smoke test refused: {smoke.reason}")
            environment = GeneratedEnvironment(
                code=reply.code,
                skill=skill,
                generation=request.generation,
                index=index,
                hint=reply.hint,
                source_record_id=answer.record_id,
                difficulty=request.difficulty,
                seed=seed,
            )
            return WrittenTask(
                environment_task(environment, max_turns=request.turn_limit), reply.code[:CODE_EXCERPT_CHARS]
            )
        if kind == "harbor":
            harbor_reply = parse_harbor_reply(answer.text)
            generated = GeneratedHarborTask(
                reply=harbor_reply,
                skill=skill,
                generation=request.generation,
                index=index,
                source_record_id=answer.record_id,
                difficulty=request.difficulty,
            )
            return WrittenTask(harbor_task(generated), harbor_reply.instruction[:CODE_EXCERPT_CHARS])
        openenv_reply = parse_openenv_reply(answer.text)
        generated_openenv = GeneratedOpenEnvTask(
            reply=openenv_reply,
            skill=skill,
            generation=request.generation,
            index=index,
            source_record_id=answer.record_id,
            difficulty=request.difficulty,
            seed=seed,
        )
        return WrittenTask(
            openenv_task(generated_openenv, max_turns=request.turn_limit),
            openenv_reply.environment[:CODE_EXCERPT_CHARS],
            openenv_reply.action_example,
        )

    def measured(self, written: WrittenTask, kind: str, skill: str, request: GenerationRequest) -> TaskMeasure:
        """Both arms played: the plain arm reported as training data, the hint arm measured only."""
        task_path = self.tasks_root / written.task.name
        tags = {"generation": str(request.generation), "kind": kind, "skill": skill}
        plain = self.solver.play(
            task_path, arm="plain", plays=request.plays, is_reporting=True, extra_instruction_paths=(), tags=tags
        )
        hint_path = task_path / "solution" / "hint.txt"
        hint = self.solver.play(
            task_path,
            arm="hint",
            plays=request.hint_plays,
            is_reporting=False,
            extra_instruction_paths=(hint_path,),
            tags=tags,
        )
        record = PlayRecord(
            name=written.task.name,
            skill=skill,
            return_without_hint=mean_reward(plain),
            return_with_hint=mean_reward(hint) if hint else mean_reward(plain),
            code_excerpt=written.code_excerpt,
        )
        return TaskMeasure(
            name=written.task.name,
            kind=kind,
            skill=skill,
            task_path=task_path,
            digest=written.task.digest,
            plain_rewards=tuple(play.reward if play.reward is not None else 0.0 for play in plain),
            hint_rewards=tuple(play.reward if play.reward is not None else 0.0 for play in hint),
            record=record,
        )

    def reported(self, proposal: Proposal, request: GenerationRequest, measure: TaskMeasure | None) -> Proposal:
        """The Designer's report for one proposal: its regret as the score, 0 for a refused one."""
        if not self.is_reporting_designer:
            return proposal
        metadata: dict[str, object] = {
            "generation": request.generation,
            "kind": proposal.kind,
            "skill": proposal.skill,
        }
        if measure is None:
            score = 0.0
            metadata["refusal"] = proposal.refusal
        else:
            score = max(measure.regret, 0.0)
            metadata["task"] = {"name": measure.name, "path": str(measure.task_path), "digest": measure.digest}
            metadata["outcome"] = measure.outcome
            metadata["regret"] = measure.regret
            metadata["return_without_hint"] = measure.record.return_without_hint
            metadata["return_with_hint"] = measure.record.return_with_hint
        report_id = self.designer.report(proposal.designer_record_id, score=score, metadata=metadata)
        return Proposal(
            proposal.index,
            proposal.kind,
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
        document = {
            "generation": request.generation,
            "request": {
                **{name: value for name, value in asdict(request).items() if name not in ("experience", "grounding")},
                "experience_records": len(request.experience),
                "grounding_chars": len(request.grounding or ""),
            },
            "manifest": str(manifest_path) if manifest_path is not None else None,
            "proposals": [asdict(proposal) for proposal in proposals],
            "tasks": [
                {
                    "name": measure.name,
                    "kind": measure.kind,
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


def main(
    argv: Sequence[str] | None = None,
    *,
    designer: Designer | None = None,
    solver: Solver | None = None,
    checks: Checks | None = None,
) -> int:
    """Run one generation from the command line and print one line per proposal."""
    parser = argparse.ArgumentParser(
        prog="python -m recipes.beta.spade.generation",
        description="One SPADE generation: propose, check, write, play both arms, split, report.",
    )
    parser.add_argument("--reef-url", required=True)
    parser.add_argument(
        "--scenario", required=True, help="the scenario the Designer's and the solver's records belong to"
    )
    parser.add_argument("--model", required=True, help="the served model, Designer and solver alike")
    parser.add_argument("--token", default=os.environ.get("REEF_TOKEN") or None)
    parser.add_argument("--tasks-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("work/spade"))
    parser.add_argument("--description", required=True, help="what the environments are about")
    parser.add_argument("--skills", required=True, help="comma separated skill names, cycled over the proposals")
    parser.add_argument("--kinds", default=",".join(KINDS), help="comma separated kinds, cycled over the proposals")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--generation", type=int, default=0)
    parser.add_argument("--difficulty", default="medium")
    parser.add_argument("--turn-limit", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--grounding", type=Path, default=None, help="a text file the Designer grounds in")
    parser.add_argument("--experience", type=Path, default=None, help="the previous generation's report")
    parser.add_argument("--plays", type=int, default=4)
    parser.add_argument("--hint-plays", type=int, default=2)
    parser.add_argument("--eval-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--designer-json",
        default=None,
        help='extra fields of the Designer\'s chat request as JSON, e.g. {"reasoning_effort": "none"}',
    )
    parser.add_argument("--agent-json", default=None, help="the solver's Harbor agent; terminus-2 by default")
    parser.add_argument("--agent-host", default=None)
    parser.add_argument("--concurrency", type=int, default=2, help="episodes in flight per arm")
    parser.add_argument("--harbor", default=None, help="the harbor command line for the oracle check")
    parser.add_argument(
        "--no-designer-report", action="store_true", help="do not report regret against the Designer's calls"
    )
    arguments = parser.parse_args(argv)
    try:
        request = GenerationRequest(
            description=arguments.description,
            skills=tuple(part.strip() for part in arguments.skills.split(",") if part.strip()),
            kinds=tuple(part.strip() for part in arguments.kinds.split(",") if part.strip()),
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
        agent = json.loads(arguments.agent_json) if arguments.agent_json else None
        designer_options = json.loads(arguments.designer_json) if arguments.designer_json else {}
        if not isinstance(designer_options, dict):
            raise GenerationError("--designer-json must hold an object")
    except (GenerationError, ValueError, OSError) as exc:
        parser.error(str(exc))
    generation = Generation(
        designer=(
            designer
            if designer is not None
            else ReefDesigner(
                reef_url=arguments.reef_url,
                scenario=arguments.scenario,
                model=arguments.model,
                token=arguments.token,
                request_options=designer_options,
            )
        ),
        solver=(
            solver
            if solver is not None
            else ReefSolver(
                reef_url=arguments.reef_url,
                scenario=arguments.scenario,
                model=arguments.model,
                work_dir=arguments.work_dir,
                token=arguments.token,
                agent=agent,
                agent_host=arguments.agent_host,
                concurrency=arguments.concurrency,
            )
        ),
        checks=checks if checks is not None else RealChecks(harbor=arguments.harbor),
        tasks_root=arguments.tasks_root,
        is_reporting_designer=not arguments.no_designer_report,
    )
    try:
        result = generation.run(request)
    except GenerationError as exc:
        parser.error(str(exc))
    measures = {measure.name: measure for measure in result.measures}
    for proposal in result.proposals:
        line: dict[str, object] = {"index": proposal.index, "kind": proposal.kind, "skill": proposal.skill}
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
