"""The workspace of one harness step: the directories a method reads and edits.

Each step gets one directory, which is kept afterwards as the step record::

    harness/      the head tree, which the method edits into the candidate
    material/     what the step learns from, read-only:
                    traces/<n>/trace.json        one trace (``Trace``): a client's request, or a rollout
                    traces/<n>/trajectory.json   the ATIF trajectory of a client's request
                    request.json                 the instruction the step answers (instruction steps only)
                    evaluation.json              the step's evaluation tasks and repeats
                    host.json                    the adapter, the parts of its tree, and its config targets
    state/        the method's own files, committed with the step
    scratch/      temporary files, removed when the step ends
    output/       files the method keeps with the step record, and what it says about its step:
                    summary.md                   plain text for the step pages
                    notes.json                   a JSON object, the commit's method_notes
                    requires.json                what the change needs where the harness runs

The scenario's history (``reef.evaluation.harness.history``) is beside the
steps, at ``Workspace.history_dir``; a method reads it and never writes it.
Material is client data: a method treats its text as untrusted and wraps it
with ``reef.core.prompts.untrusted_text`` before the text enters a prompt.
"""

from __future__ import annotations

import json
import math
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from reef.core.job_data import TrajectorySample
from reef.core.requirements import parse_requires
from reef.core.tasks import task_instruction
from reef.core.training_request import TrainingRequest
from reef.core.trajectories import source_record_id
from reef.evaluation.harness import EpisodeRecord, History
from reef.evaluation.harness.tasks import client_of, prompt_of
from reef.harness.adapters.descriptor import AdapterDescriptor
from reef.harness.tree import HarnessTree

HARNESS_DIR = "harness"
MATERIAL_DIR = "material"
STATE_DIR = "state"
SCRATCH_DIR = "scratch"
OUTPUT_DIR = "output"
TRACES_DIR = "traces"
TRACE_FILE = "trace.json"
TRAJECTORY_FILE = "trajectory.json"
REQUEST_FILE = "request.json"
EVALUATION_FILE = "evaluation.json"
HOST_FILE = "host.json"
#: In ``output/``: what a method says about its step. Each file is optional.
SUMMARY_FILE = "summary.md"
NOTES_FILE = "notes.json"
REQUIRES_FILE = "requires.json"
#: Where a trace came from: a client's request (``traffic``), or an episode the step ran on a train task
#: (``rollout``).
TRACE_SOURCES = ("traffic", "rollout")
#: The most bytes ``state/`` may hold: the commit record stores every file of it.
MAX_STATE_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class Trace:
    """One trace of a step's material: a request a client sent, as served, or an episode the step rolled out.

    Values: ``source`` is one of ``TRACE_SOURCES``. ``directory`` holds
    ``trace.json``, and for traffic the request's ATIF ``trajectory.json``;
    a rollout keeps its trajectory files in the history's episode directory.
    ``task`` is the last user prompt of the request, ``None`` for a request
    without one, or what the episode's harness was asked: the prompt, or the
    instruction of the task directory. ``reply`` is the final
    assistant text, ``None`` when there is none, as for a traffic trace that
    ends on a tool call (one model call in the middle of a run). ``score`` is
    the client's reward or the scorer's score, a finite number, ``None`` when
    the client gave none or the episode could not run. ``feedback`` is the
    client's feedback text, ``None`` when the client gave none and for a
    rollout. ``failure`` is why the episode failed, ``None`` when it did not
    fail; traffic has none. ``record`` names the agent record of the request,
    or the history episode of the rollout. ``client`` is the client's tag,
    ``"untagged"`` without one and for a rollout.
    Purpose: ``Workspace.create`` writes one per traffic sample and one per
    rolled-out episode of the step; ``Workspace.traces`` reads them for a
    method.
    """

    directory: Path
    source: str
    task: str | None
    reply: str | None
    score: float | None
    feedback: str | None
    failure: str | None
    record: str
    client: str

    def __post_init__(self) -> None:
        if self.source not in TRACE_SOURCES:
            raise ValueError(f"Trace.source must be one of {', '.join(TRACE_SOURCES)}")
        if self.score is not None and (isinstance(self.score, bool) or not math.isfinite(self.score)):
            raise ValueError("Trace.score must be a finite number or None")
        if self.source == "traffic" and self.failure is not None:
            raise ValueError("Trace.failure says why an episode failed; a traffic trace has none")

    def to_json(self) -> dict[str, Any]:
        """``trace.json``: every field but ``directory``."""
        row = asdict(self)
        row.pop("directory")
        return row

    def trajectory(self) -> Mapping[str, Any] | None:
        """The request's ATIF document, read from ``trajectory.json``; ``None`` for a rollout, whose episode keeps
        its trajectory files in the history (``History.episode_dir(record)``)."""
        if self.source == "rollout":
            return None
        return json.loads((self.directory / TRAJECTORY_FILE).read_text(encoding="utf-8"))


def json_object(value: Any) -> dict[str, Any]:
    """``value`` as a JSON object: text keys, and no value that ``json.dumps`` refuses, NaN and infinity included.

    ``ValueError`` otherwise.
    """
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError("notes must be a JSON object, whose keys are text")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"notes must be a JSON object: {exc}") from exc
    return dict(value)


@dataclass(frozen=True)
class StepOutput:
    """What a method said about its step, read back from its ``output/``.

    Values: ``summary`` is the stripped text of ``summary.md``, empty without
    one. ``notes`` is the JSON object of ``notes.json``, and ``requires`` the
    ``reef.core.requirements`` items of ``requires.json``; each is empty
    without its file or when the file was refused. ``refused`` maps the name
    of each refused file to the reason.
    Purpose: ``Workspace.read_output`` makes it after the method's step; the
    backend keeps it with the step and the history.
    """

    summary: str = ""
    notes: Mapping[str, Any] = field(default_factory=dict)
    requires: tuple[Mapping[str, Any], ...] = ()
    refused: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "notes", MappingProxyType(json_object(self.notes)))
        object.__setattr__(self, "requires", tuple(parse_requires(list(self.requires))))
        object.__setattr__(self, "refused", MappingProxyType(dict(self.refused)))


@dataclass(frozen=True)
class Workspace:
    """The directories of one harness step.

    Values: ``root`` holds ``harness/``, ``material/``, ``state/``,
    ``scratch/`` and ``output/``; ``history_dir`` is the scenario's history
    directory, which may be empty; ``step`` ≥ 1 is the step's number.
    Purpose: the backend makes one per step with ``create`` and passes it to
    ``Method.step``; the method reads and edits the directories, directly or
    through the readers below.
    """

    root: Path
    history_dir: Path
    step: int

    def __post_init__(self) -> None:
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 1:
            raise ValueError("Workspace.step must be a positive integer")

    @property
    def harness(self) -> Path:
        return self.root / HARNESS_DIR

    @property
    def material(self) -> Path:
        return self.root / MATERIAL_DIR

    @property
    def state(self) -> Path:
        return self.root / STATE_DIR

    @property
    def scratch(self) -> Path:
        return self.root / SCRATCH_DIR

    @property
    def output(self) -> Path:
        return self.root / OUTPUT_DIR

    @classmethod
    def create(
        cls,
        root: Path,
        *,
        history_dir: Path,
        step: int,
        tree: HarnessTree,
        descriptor: AdapterDescriptor,
        tasks: Sequence[str],
        repeats: int,
        task_directories: bool,
        samples: Sequence[TrajectorySample] = (),
        rollouts: Sequence[EpisodeRecord] = (),
        request: TrainingRequest | None = None,
        state_files: Mapping[str, str] | None = None,
    ) -> Workspace:
        """Write a step's workspace under ``root``, which must be absent or empty.

        ``tree`` goes to ``harness/`` and ``state_files`` (relative path to
        text) to ``state/``. ``material/`` gets one traffic trace per sample,
        then one rollout trace per episode of ``rollouts``, the request, the
        evaluation's ``tasks`` and ``repeats`` (0 when no episode can run), and
        the host facts. ``task_directories`` says that the tasks are Harbor
        task directories rather than prompts.
        """
        workspace = cls(Path(root), Path(history_dir), step)
        for directory in (workspace.harness, workspace.material, workspace.state, workspace.scratch, workspace.output):
            directory.mkdir(parents=True)
        tree.write(workspace.harness)
        HarnessTree(dict(state_files or {})).write(workspace.state)
        material = workspace.material
        for index, sample in enumerate(samples):
            write_traffic_trace(material / TRACES_DIR / str(index), sample)
        for index, record in enumerate(rollouts, start=len(samples)):
            write_rollout_trace(material / TRACES_DIR / str(index), record, task_directories)
        if request is not None:
            write_json(material / REQUEST_FILE, {"id": request.id, **request.to_dict()})
        write_json(material / EVALUATION_FILE, {"tasks": list(tasks), "repeats": repeats})
        write_json(material / HOST_FILE, host_facts(descriptor, task_directories))
        return workspace

    def tree(self) -> HarnessTree:
        """The tree in ``harness/`` now; ``ValueError`` for a symbolic link or a file that is not UTF-8 text."""
        return HarnessTree.read(self.harness)

    def write_tree(self, tree: HarnessTree) -> None:
        """Replace everything in ``harness/`` with ``tree``."""
        shutil.rmtree(self.harness)
        self.harness.mkdir()
        tree.write(self.harness)

    def traces(self) -> tuple[Trace, ...]:
        """The material's traces, in trace order."""
        directory = self.material / TRACES_DIR
        if not directory.is_dir():
            return ()
        traces = []
        for trace_dir in sorted(directory.iterdir(), key=lambda path: int(path.name)):
            row = json.loads((trace_dir / TRACE_FILE).read_text(encoding="utf-8"))
            traces.append(Trace(directory=trace_dir, **row))
        return tuple(traces)

    def request(self) -> TrainingRequest | None:
        """The instruction this step answers, ``None`` for a step without one."""
        path = self.material / REQUEST_FILE
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return TrainingRequest.from_dict(payload, request_id=str(payload.get("id", "")))

    def tasks(self) -> tuple[str, ...]:
        """The tasks this step's evaluation runs, the seed tasks first."""
        return tuple(json.loads((self.material / EVALUATION_FILE).read_text(encoding="utf-8"))["tasks"])

    def evaluation_repeats(self) -> int:
        """The episodes the evaluation runs per task and tree; 0 when the deployment configured no scorer."""
        return int(json.loads((self.material / EVALUATION_FILE).read_text(encoding="utf-8"))["repeats"])

    def host(self) -> Mapping[str, Any]:
        """The host facts: ``adapter``, ``parts`` (each part of the adapter's tree, with its ``path``, ``format``,
        ``renders``, ``holds`` and ``review``), ``config_targets`` and ``tasks_are_directories``."""
        return json.loads((self.material / HOST_FILE).read_text(encoding="utf-8"))

    def history(self) -> History:
        """The scenario's history as it was when the step started reading it.

        Records of this step's number or a later one come from an attempt of this step that did not commit;
        a method that counts or compares what it tried skips them.
        """
        return History(self.history_dir)

    def write_output(
        self, summary: str = "", notes: Mapping[str, Any] | None = None, requires: Sequence[Mapping[str, Any]] = ()
    ) -> None:
        """Say what the step did: ``summary.md``, ``notes.json`` and ``requires.json`` in ``output/``.

        An empty argument writes no file. ``ValueError``, before any file is
        written, when ``notes`` is not a JSON object or an item of
        ``requires`` is not a requirement.
        """
        checked = StepOutput(summary=summary, notes=notes or {}, requires=tuple(requires))
        self.output.mkdir(parents=True, exist_ok=True)
        if summary.strip():
            (self.output / SUMMARY_FILE).write_text(summary.strip() + "\n", encoding="utf-8")
        if checked.notes:
            write_json(self.output / NOTES_FILE, dict(checked.notes))
        if checked.requires:
            write_json(self.output / REQUIRES_FILE, [dict(item) for item in checked.requires])

    def read_output(self) -> StepOutput:
        """What the method said about its step; a ``notes.json`` or ``requires.json`` that is not valid is refused."""
        summary = output_text(self.output / SUMMARY_FILE)
        notes: dict[str, Any] = {}
        requires: tuple[Mapping[str, Any], ...] = ()
        refused: dict[str, str] = {}
        if text := output_text(self.output / NOTES_FILE):
            try:
                notes = json_object(json.loads(text))
            except ValueError as exc:
                refused[NOTES_FILE] = str(exc)
        if text := output_text(self.output / REQUIRES_FILE):
            try:
                requires = tuple(parse_requires(json.loads(text)))
            except ValueError as exc:
                refused[REQUIRES_FILE] = str(exc)
        return StepOutput(summary, notes, requires, refused)

    def state_files(self) -> dict[str, str]:
        """Every file in ``state/`` as relative path to text.

        ``ValueError``, whose message starts ``state/ cannot be committed``, for a symbolic link, a file that is
        not UTF-8 text, or more than ``MAX_STATE_BYTES``.
        """
        try:
            files = dict(HarnessTree.read(self.state).files)
        except ValueError as exc:
            raise ValueError(f"state/ cannot be committed: {exc}") from exc
        size = sum(len(path.encode("utf-8")) + len(text.encode("utf-8")) for path, text in files.items())
        if size > MAX_STATE_BYTES:
            raise ValueError(
                f"state/ cannot be committed: it holds {size} bytes, and a step commits at most {MAX_STATE_BYTES}"
            )
        return files


def output_text(path: Path) -> str:
    """The stripped text of ``path``; empty when there is no such file."""
    return path.read_text(encoding="utf-8", errors="replace").strip() if path.is_file() else ""


def write_json(path: Path, value: Any) -> None:
    """``value`` as indented JSON at ``path``, creating its directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def write_traffic_trace(directory: Path, sample: TrajectorySample) -> None:
    """One client trace: ``trace.json`` and its ATIF ``trajectory.json``."""
    # A client may report structured feedback; the trace keeps it as JSON text.
    feedback = sample.metadata.get("feedback")
    if feedback is not None and not isinstance(feedback, str):
        feedback = json.dumps(feedback, ensure_ascii=False)
    reward = sample.metadata.get("reward")
    score = float(reward) if isinstance(reward, (int, float)) and not isinstance(reward, bool) else None
    trace = Trace(
        directory=directory,
        source="traffic",
        task=prompt_of(sample),
        reply=reply_of(sample),
        score=score,
        feedback=feedback,
        failure=None,
        record=source_record_id(sample),
        client=client_of(sample),
    )
    write_json(directory / TRACE_FILE, trace.to_json())
    write_json(directory / TRAJECTORY_FILE, dict(sample.trajectory))


def write_rollout_trace(directory: Path, record: EpisodeRecord, task_directories: bool) -> None:
    """One rolled-out episode: ``trace.json`` alone, since the episode keeps its trajectory files in the history.

    The trace's task is what the harness was asked: the prompt itself, or the
    instruction of the task directory when the tasks are ``task_directories``.
    """
    task = task_instruction(Path(record.task)) if task_directories else record.task
    trace = Trace(
        directory=directory,
        source="rollout",
        task=task,
        reply=record.reply,
        score=record.score,
        feedback=None,
        failure=None if record.failure is None else record.failure.cause,
        record=record.episode,
        client="untagged",
    )
    write_json(directory / TRACE_FILE, trace.to_json())


def reply_of(sample: TrajectorySample) -> str | None:
    """The final assistant text of a trajectory, read from its last agent step that is not copied context.

    ``None`` when there is no such step, or when that step calls a tool: then the trace ends in the middle of a
    run (one model call of several), and no reply exists yet.
    """
    for step in reversed(sample.trajectory["steps"]):
        if not isinstance(step, Mapping) or step.get("source") != "agent" or step.get("is_copied_context"):
            continue
        if step.get("tool_calls"):
            return None
        message = step.get("message")
        if isinstance(message, str):
            return message
        if isinstance(message, list):
            return "".join(str(part.get("text", "")) for part in message if isinstance(part, Mapping))
        return None
    return None


def host_facts(descriptor: AdapterDescriptor, task_directories: bool) -> dict[str, Any]:
    """What a method needs to know about the adapter: the parts of its tree, its config targets, and whether the
    tasks are Harbor task directories."""
    return {
        "adapter": descriptor.name,
        "parts": {name: part.to_json() for name, part in descriptor.layout.parts.items()},
        "config_targets": sorted(descriptor.config_targets),
        "tasks_are_directories": task_directories,
    }


__all__ = ["MAX_STATE_BYTES", "TRACE_SOURCES", "Trace", "Workspace", "host_facts", "reply_of"]
