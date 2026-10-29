"""The workspace of one harness step: the directories a method reads and edits.

Each step gets one directory, which is kept afterwards as the step record::

    harness/      the head tree, which the method edits into the candidate
    material/     what the step learns from, read-only:
                    traces/<n>/trace.json        one trace: its task, reply, score, feedback, record and client
                    traces/<n>/trajectory.json   the trace's ATIF trajectory
                    request.json                 the instruction the step answers (instruction steps only)
                    evaluation.json              the step's evaluation tasks and repeats
                    failures.json                the head tree's failure manifest (after its first evaluation)
                    host.json                    the adapter, the tree kinds it renders, and its config targets
    state/        the method's own files, committed with the step
    scratch/      temporary files, removed when the step ends
    output/       files the method keeps with the step record

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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from reef.core.job_data import TrajectorySample
from reef.core.training_request import TrainingRequest
from reef.core.trajectories import source_record_id
from reef.evaluation.harness import History
from reef.evaluation.harness.tasks import client_of, prompt_of
from reef.harness.adapters.descriptor import AdapterDescriptor
from reef.harness.tree import HarnessTree, TreeKind
from reef.harness.tree.layout import KIND_GUIDE
from reef.train.harness_backend.manifest import FailureManifest

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
FAILURES_FILE = "failures.json"
HOST_FILE = "host.json"
#: The most bytes ``state/`` may hold: the commit record stores every file of it.
MAX_STATE_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class Trace:
    """One trace of a step's material: one request a client sent, as served.

    Values: ``directory`` holds ``trace.json`` and the trace's ATIF
    ``trajectory.json``. ``task`` is the last user prompt of the request,
    ``None`` for a request without one. ``reply`` is the final assistant
    text, ``None`` when there is none, as for a trace that ends on a tool call
    (one model call in the middle of a run). ``score`` is the trace's reward,
    a finite number, and ``feedback`` its feedback text, each ``None`` when
    the client gave none. ``record`` names the agent record; ``client`` is
    the client's tag, ``"untagged"`` without one.
    Purpose: ``Workspace.create`` writes one per traffic sample of the step;
    ``Workspace.traces`` reads them for a method.
    """

    directory: Path
    task: str | None
    reply: str | None
    score: float | None
    feedback: str | None
    record: str
    client: str

    def __post_init__(self) -> None:
        if self.score is not None and (isinstance(self.score, bool) or not math.isfinite(self.score)):
            raise ValueError("Trace.score must be a finite number or None")

    def to_json(self) -> dict[str, Any]:
        """``trace.json``: every field but ``directory``."""
        row = asdict(self)
        row.pop("directory")
        return row

    def trajectory(self) -> Mapping[str, Any]:
        """The trace's ATIF document, read from ``trajectory.json``."""
        return json.loads((self.directory / TRAJECTORY_FILE).read_text(encoding="utf-8"))


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
        samples: Sequence[TrajectorySample] = (),
        request: TrainingRequest | None = None,
        manifest: FailureManifest | None = None,
        state_files: Mapping[str, str] | None = None,
    ) -> Workspace:
        """Write a step's workspace under ``root``, which must be absent or empty.

        ``tree`` goes to ``harness/`` and ``state_files`` (relative path to
        text) to ``state/``. ``material/`` gets one trace per sample, the
        request, the evaluation's ``tasks`` and ``repeats`` (0 when no rollout
        can run), the manifest and the host facts.
        """
        workspace = cls(Path(root), Path(history_dir), step)
        for directory in (workspace.harness, workspace.material, workspace.state, workspace.scratch, workspace.output):
            directory.mkdir(parents=True)
        tree.write(workspace.harness)
        HarnessTree(dict(state_files or {})).write(workspace.state)
        material = workspace.material
        for index, sample in enumerate(samples):
            write_traffic_trace(material / TRACES_DIR / str(index), sample)
        if request is not None:
            write_json(material / REQUEST_FILE, {"id": request.id, **request.to_dict()})
        write_json(material / EVALUATION_FILE, {"tasks": list(tasks), "repeats": repeats})
        if manifest is not None:
            write_json(material / FAILURES_FILE, manifest.to_state())
        write_json(material / HOST_FILE, host_facts(descriptor))
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
        """The rollouts the evaluation runs per task and tree; 0 when the deployment configured no scorer."""
        return int(json.loads((self.material / EVALUATION_FILE).read_text(encoding="utf-8"))["repeats"])

    def host(self) -> Mapping[str, Any]:
        """The host facts: ``adapter``, ``kinds`` (each kind the adapter renders, with its guide),
        ``config_targets`` and ``tasks_are_directories``."""
        return json.loads((self.material / HOST_FILE).read_text(encoding="utf-8"))

    def history(self) -> History:
        """The scenario's history as it was when the step started reading it.

        Records of this step's number or a later one come from an attempt of this step that did not commit;
        a method that counts or compares what it tried skips them.
        """
        return History(self.history_dir)

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
        task=prompt_of(sample),
        reply=reply_of(sample),
        score=score,
        feedback=feedback,
        record=source_record_id(sample),
        client=client_of(sample),
    )
    write_json(directory / TRACE_FILE, trace.to_json())
    write_json(directory / TRAJECTORY_FILE, dict(sample.trajectory))


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


def host_facts(descriptor: AdapterDescriptor) -> dict[str, Any]:
    """What a method needs to know about the adapter: the kinds it renders, with their guide, and its targets."""
    kinds = [
        kind
        for kind in TreeKind
        if (kind is TreeKind.CONFIG and descriptor.config_targets) or kind.value in descriptor.kind_paths
    ]
    return {
        "adapter": descriptor.name,
        "kinds": {kind.value: KIND_GUIDE[kind] for kind in kinds},
        "config_targets": sorted(descriptor.config_targets),
        "tasks_are_directories": descriptor.is_prompt_task_directory,
    }


__all__ = ["MAX_STATE_BYTES", "Trace", "Workspace", "host_facts", "reply_of"]
