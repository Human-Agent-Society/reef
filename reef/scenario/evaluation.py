"""Experimental, read-only comparison of retained harness releases (#355).

Run files are private evaluation exports, not scenario commits. The format and
Python interface are provisional pending upstream design review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import shutil
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from threading import Event

from reef.harness.adapters.descriptor import AdapterDescriptor
from reef.harness.episodes.executor import LocalExecutor
from reef.harness.episodes.model_binding import ModelBindingError, ModelBindings
from reef.harness.episodes.trajectory import TrajectoryError, reader_for
from reef.harness.tree.nodes import redact_secret_shaped
from reef.harness.tree.render import render_composition
from reef.scenario.scenario import Scenario
from reef.train.cordis_backend.backend import EpisodeEvaluationWorker
from reef.train.cordis_backend.strategies import EpisodeScorer


@dataclass(frozen=True)
class EvaluationTask:
    """A text prompt with a stable suite-local identity."""

    task_id: str
    prompt: str
    family: str = "default"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.task_id, str)
            or not self.task_id.strip()
            or not isinstance(self.prompt, str)
            or not self.prompt
        ):
            raise ValueError("task_id and prompt must be nonempty")
        if not isinstance(self.family, str) or not self.family.strip():
            raise ValueError("task family must be nonempty")


@dataclass(frozen=True)
class EvaluationConditions:
    """Operator-pinned identities, including versions not discoverable locally.

    environment_version must identify installed harness/dependency versions,
    fixtures and sampling settings. A seed is metadata only unless the harness
    actually applies it. Model aliases are not immutable model revisions.
    """

    suite_version: str
    scorer_version: str
    model_version: str
    environment_version: str
    repeats: int = 1
    episode_timeout_seconds: float = 60.0
    retain_episodes: bool = False

    def __post_init__(self) -> None:
        for value in (self.suite_version, self.scorer_version, self.model_version, self.environment_version):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("evaluation version identifiers must be nonempty strings")
        if type(self.retain_episodes) is not bool:
            raise ValueError("retain_episodes must be a boolean")
        if type(self.repeats) is not int or self.repeats < 1:
            raise ValueError("repeats must be a positive integer")
        if (
            isinstance(self.episode_timeout_seconds, bool)
            or not isinstance(self.episode_timeout_seconds, (int, float))
            or not math.isfinite(self.episode_timeout_seconds)
            or self.episode_timeout_seconds <= 0
        ):
            raise ValueError("episode timeout must be finite and positive")


@dataclass(frozen=True)
class EvaluationOutcome:
    ordinal: int
    release_id: str
    task_id: str
    repeat: int
    status: str
    score: float | None
    elapsed_seconds: float
    failure_stage: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    model_responses: int | None = None
    episode_checksum: str | None = None

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or self.ordinal < 0 or type(self.repeat) is not int or self.repeat < 0:
            raise ValueError("invalid evaluation ordinal or repeat")
        if not isinstance(self.release_id, str) or not isinstance(self.task_id, str):
            raise ValueError("invalid evaluation identity")
        if self.status not in {"scored", "execution_error", "invalid_score", "unrun"}:
            raise ValueError("invalid evaluation status")
        if self.status == "scored":
            if (
                isinstance(self.score, bool)
                or not isinstance(self.score, (int, float))
                or not math.isfinite(self.score)
            ):
                raise ValueError("scored outcomes require a finite score")
        elif self.score is not None:
            raise ValueError("unscored outcomes must not have a score")
        for value in (self.input_tokens, self.output_tokens, self.model_responses):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("usage counts must be nonnegative integers or unknown")
        if self.episode_checksum is not None and (
            not isinstance(self.episode_checksum, str)
            or len(self.episode_checksum) != 64
            or any(character not in "0123456789abcdef" for character in self.episode_checksum)
        ):
            raise ValueError("invalid episode checksum")
        if type(self.elapsed_seconds) not in (int, float) or not math.isfinite(self.elapsed_seconds):
            raise ValueError("elapsed time must be finite")
        if self.elapsed_seconds < 0 or (self.failure_stage is not None and not isinstance(self.failure_stage, str)):
            raise ValueError("invalid elapsed time or failure stage")


def json_digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    """Replace one complete record; an interrupted write never counts as a result."""
    descriptor, temporary = tempfile.mkstemp(prefix=".evaluation-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_outcome(path: Path, expected: EvaluationOutcome, fingerprint: str) -> EvaluationOutcome:
    """Validate both the result shape and its exact planned position before reuse."""
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or record.get("fingerprint") != fingerprint:
        raise ValueError(f"incompatible evaluation result: {path.name}")
    raw = record.get("result")
    if not isinstance(raw, dict):
        raise ValueError(f"invalid evaluation result: {path.name}")
    try:
        outcome = EvaluationOutcome(**raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid evaluation result: {path.name}") from exc
    if (outcome.ordinal, outcome.release_id, outcome.task_id, outcome.repeat) != (
        expected.ordinal,
        expected.release_id,
        expected.task_id,
        expected.repeat,
    ) or outcome.status == "unrun":
        raise ValueError(f"evaluation result does not match planned task: {path.name}")
    if outcome.episode_checksum is not None:
        episode_path = path.parent.parent / "episodes" / path.name
        if (
            not episode_path.is_file()
            or hashlib.sha256(episode_path.read_bytes()).hexdigest() != outcome.episode_checksum
        ):
            raise ValueError(f"retained episode is missing or changed: {path.name}")
    return outcome


def episode_record(directory: Path, trajectory_format: str) -> tuple[dict[str, object], dict[str, int | None]]:
    """Recover the worker's observation, including partial failed executions.

    Native usage is reported only when every response carries both counts.
    Missing counters and unsupported formats remain unknown, never free.
    """
    try:
        trajectory = reader_for(trajectory_format)(directory)
    except TrajectoryError:
        trajectory = ()
    episode_path = directory / "episode.json"
    details = json.loads(episode_path.read_text()) if episode_path.is_file() else None
    usage: dict[str, int | None] = dict.fromkeys(("input_tokens", "output_tokens", "model_responses"))
    if trajectory_format == "native-jsonl":
        responses = [
            event["data"] for event in trajectory if event.get("type") in {"assistant/message", "context/compacted"}
        ]
        usage["model_responses"] = len(responses)
        for field in ("input_tokens", "output_tokens"):
            counts = [response.get("usage", {}).get(field) for response in responses]
            if counts and all(type(count) is int and count >= 0 for count in counts):
                usage[field] = sum(counts)
    return {"trajectory": trajectory, "episode": details}, usage


class RetainedHarnessEvaluation:
    """Snapshot committed harness targets, then evaluate without invoking training.

    First implementation is deliberately serial. Cancellation stops admission;
    the current episode drains under its timeout. Scorers must return promptly
    or enforce their own I/O deadline. A model judge's internal calls are not
    bounded by the episode admission budget.
    """

    def __init__(
        self,
        scenario: Scenario,
        release_ids: Sequence[str],
        tasks: Sequence[EvaluationTask],
        conditions: EvaluationConditions,
        *,
        descriptor: AdapterDescriptor,
        scorer: EpisodeScorer,
        binary: str,
        models: ModelBindings | None = None,
    ) -> None:
        if len(release_ids) < 2 or len(set(release_ids)) != len(release_ids):
            raise ValueError("select at least two distinct committed releases; first is baseline")
        if not tasks or len({task.task_id for task in tasks}) != len(tasks):
            raise ValueError("tasks must have unique, nonempty IDs")
        if descriptor.is_prompt_task_directory:
            raise ValueError("directory tasks need fixture snapshotting; this version supports text prompts only")
        if scenario.surface.files is None:
            raise ValueError("evaluation currently supports harness file surfaces, not weight targets")
        if scorer.execution_requirements().gpus_per_worker or scorer.execution_requirements().cluster:
            raise ValueError("this serial evaluator requires a local CPU or API scorer")
        binary_path = shutil.which(binary)
        if binary_path is None:
            raise ValueError(f"harness binary is unavailable: {binary}")
        self.binary_path = Path(binary_path).absolute()
        self.binary_checksum = hashlib.sha256(self.binary_path.read_bytes()).hexdigest()
        self.tasks = tuple(tasks)
        self.release_ids = tuple(release_ids)
        self.conditions = conditions
        self.files: list[dict[str, str]] = []
        targets = []
        pending = {row["release_id"] for row in scenario.releases() if row.get("pending")}
        for release_id in release_ids:
            if release_id in pending:
                raise ValueError(f"release {release_id} is pending publication, not a served historical target")
            artifact = scenario.artifact_for_version(release_id)
            files = dict(scenario.surface.files.read_files(artifact) or {})
            selected_checksum = json_digest(files)
            if models is not None:
                # Execute the retained bytes. Re-rendering the whole composition
                # can change tool modules after a JSON round-trip, even when
                # dictionary key order is the only difference. Override only
                # the adapter's explicitly declared provider configuration files.
                binding_nodes = models.served.compose_nodes(descriptor)
                target_names = {str(config.get("target", "primary")) for _, config in binding_nodes}
                retained_config = []
                for target_name in sorted(target_names):
                    path = descriptor.config_targets[target_name].path
                    data = json.loads(files.get(path, "{}"))
                    if not isinstance(data, dict):
                        raise ValueError(f"retained model configuration {path} must be a JSON object")
                    retained_config.append(("config", {"target": target_name, "data": data}))
                bound = render_composition((*retained_config, *binding_nodes), descriptor)
                for target_name in target_names:
                    path = descriptor.config_targets[target_name].path
                    files[path] = bound[path]
            self.files.append(files)
            targets.append(
                {"release_id": release_id, "content_id": artifact.ref.content_id, "files_checksum": selected_checksum}
            )
        self.models = models
        self.worker = EpisodeEvaluationWorker(
            descriptor=descriptor,
            scorer=scorer,
            binary=str(self.binary_path),
            timeout=conditions.episode_timeout_seconds,
            executor=LocalExecutor(),
            forbid_residue=True,
            owner_lease=True,
        )
        # Only digests of descriptor/binding configuration enter the manifest;
        # credentials and rendered provider files stay out of exports.
        binding_identity = (
            []
            if models is None
            else [
                {
                    "name": name,
                    "url": models[name].base_url,
                    "model": models[name].model,
                    "api": models[name].api,
                    "timeout": models[name].timeout_s,
                }
                for name in models
            ]
        )
        source_root = Path(__file__).resolve().parents[1]
        source_checksums = {
            str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(source_root.rglob("*.py"))
        }
        self.specification = {
            "format": "reef-release-evaluation/2",
            "scenario": scenario.name,
            "targets": targets,
            "tasks": [asdict(task) for task in tasks],
            "conditions": asdict(conditions),
            "binary_checksum": self.binary_checksum,
            "adapter_checksum": json_digest(
                {
                    "name": descriptor.name,
                    "argv": descriptor.argv,
                    "env": dict(descriptor.env),
                    "trajectory": descriptor.trajectory_format,
                    "trajectory_path": descriptor.trajectory_path,
                    "cleanup": descriptor.cleanup_whitelist,
                    "writable": descriptor.writable_paths,
                }
            ),
            "bindings_checksum": json_digest(binding_identity),
            "implementation_checksum": json_digest(source_checksums),
            "scorer_type": f"{type(scorer).__module__}.{type(scorer).__qualname__}",
            "python_version": platform.python_version(),
            "platform": platform.system(),
        }
        self.fingerprint = json_digest(self.specification)

    def run(
        self, output: Path, *, cancel: Event | None = None, max_new_episodes: int | None = None
    ) -> tuple[EvaluationOutcome, ...]:
        """Resume an identical run. A finite per-invocation budget permits incremental evaluation.

        Completed errors are retained too; use a new directory for retries so a
        favorable retry cannot silently replace a previously observed failure.
        """
        if os.name != "posix":
            raise ValueError("evaluation run locking currently requires POSIX")
        import fcntl

        if max_new_episodes is not None and (type(max_new_episodes) is not int or max_new_episodes < 0):
            raise ValueError("episode budget must be a nonnegative integer")
        if hashlib.sha256(self.binary_path.read_bytes()).hexdigest() != self.binary_checksum:
            raise ValueError("harness binary changed since target snapshot")
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (output / ".lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("another evaluation owns this run directory") from exc
            manifest_path = output / "manifest.json"
            if manifest_path.exists():
                if json.loads(manifest_path.read_text(encoding="utf-8")) != self.specification:
                    raise ValueError("evaluation conditions changed; use a new output directory")
            else:
                if any(path.name != ".lock" for path in output.iterdir()):
                    raise ValueError("new evaluation requires an empty output directory")
                atomic_json(manifest_path, self.specification)
            records = output / "results"
            records.mkdir(exist_ok=True, mode=0o700)
            outcomes: list[EvaluationOutcome] = []
            episodes = output / "episodes"
            if self.conditions.retain_episodes:
                episodes.mkdir(exist_ok=True, mode=0o700)
            executed = 0
            # Interleave releases per task/repetition, instead of running each
            # release's entire suite under a different period of provider load.
            for task in self.tasks:
                for repeat in range(self.conditions.repeats):
                    for target_index, release_id in enumerate(self.release_ids):
                        expected = EvaluationOutcome(
                            len(outcomes), release_id, task.task_id, repeat, "unrun", None, 0.0
                        )
                        result_path = records / f"{expected.ordinal:08d}.json"
                        if result_path.exists():
                            outcome = read_outcome(result_path, expected, self.fingerprint)
                        elif (cancel is not None and cancel.is_set()) or (
                            max_new_episodes is not None and executed >= max_new_episodes
                        ):
                            outcome = expected
                        else:
                            stage: str | None
                            started = time.monotonic()
                            # The existing worker owns execution and scoring. A private temporary
                            # record also captures failed executions; no second execution path.
                            with tempfile.TemporaryDirectory(prefix=".episode-", dir=output) as temporary:
                                keep_dir = Path(temporary) / "record"
                                try:
                                    scored = self.worker.run(
                                        self.files[target_index], task.prompt, keep_dir=keep_dir, models=self.models
                                    )
                                except ModelBindingError:
                                    status, score, stage = "execution_error", None, "scorer_model"
                                except TimeoutError:
                                    status, score, stage = "execution_error", None, "scorer_timeout"
                                except (ValueError, TypeError, ArithmeticError):
                                    status, score, stage = "invalid_score", None, "scorer"
                                else:
                                    stage = None if scored.failure is None else scored.failure.stage
                                    if scored.score is None or scored.failure is not None:
                                        status, score = "execution_error", None
                                    else:
                                        status, score = "scored", scored.score
                                record, usage = episode_record(keep_dir, self.worker.descriptor.trajectory_format)
                                checksum = None
                                if self.conditions.retain_episodes:
                                    # Keep normalized observations, not rendered provider bindings.
                                    # Explicit binding credentials are redacted in addition to the
                                    # repository's credential-shape filter. Private task data remains.
                                    serialized = json.dumps(record, ensure_ascii=False, allow_nan=False)
                                    if self.models is not None:
                                        for binding in self.models.values():
                                            if binding.api_key:
                                                serialized = serialized.replace(
                                                    json.dumps(binding.api_key, ensure_ascii=False)[1:-1], "[REDACTED]"
                                                )
                                    episode_path = episodes / result_path.name
                                    atomic_json(episode_path, json.loads(redact_secret_shaped(serialized)))
                                    checksum = hashlib.sha256(episode_path.read_bytes()).hexdigest()
                            outcome = EvaluationOutcome(
                                expected.ordinal,
                                release_id,
                                task.task_id,
                                repeat,
                                status,
                                score,
                                time.monotonic() - started,
                                stage,
                                usage["input_tokens"],
                                usage["output_tokens"],
                                usage["model_responses"],
                                checksum,
                            )
                            atomic_json(result_path, {"fingerprint": self.fingerprint, "result": asdict(outcome)})
                            executed += 1
                        outcomes.append(outcome)
            atomic_json(
                output / "run.json",
                {
                    "fingerprint": self.fingerprint,
                    "specification": self.specification,
                    "results": [asdict(outcome) for outcome in outcomes],
                    "comparisons": [
                        asdict(row) for row in release_comparisons(outcomes, self.release_ids, self.tasks)
                    ],
                },
            )
            (output / "report.md").write_text(
                comparison_markdown(outcomes, self.release_ids, self.tasks), encoding="utf-8"
            )
            return tuple(outcomes)


@dataclass(frozen=True)
class PairedSummary:
    paired_tasks: int
    paired_runs: int
    planned_tasks: int
    mean_delta: float | None
    task_bootstrap_95_percent_interval: tuple[float, float] | None
    improved_tasks: int
    regressed_tasks: int
    unchanged_tasks: int


@dataclass(frozen=True)
class FamilyComparison:
    release_id: str
    family: str
    vs_baseline: PairedSummary
    vs_previous: PairedSummary


def paired_summary(
    outcomes: Sequence[EvaluationOutcome], release_id: str, reference_id: str, task_ids: Sequence[str]
) -> PairedSummary:
    """Equal-weight task differences; repetitions remain in their task cluster.

    Percentile bootstrap over paired tasks (2,000 draws, fixed analysis seed).
    This describes sampling uncertainty, not provider drift, benchmark leakage
    or multiple-comparison correction. Single-task intervals are undefined.
    """
    indexed = {(row.release_id, row.task_id, row.repeat): row for row in outcomes}
    differences = []
    matched_runs = 0
    for task_id in task_ids:
        reference = [row for row in outcomes if row.release_id == reference_id and row.task_id == task_id]
        paired = []
        for row in reference:
            target = indexed.get((release_id, task_id, row.repeat))
            if (
                row.status == "scored"
                and target is not None
                and target.status == "scored"
                and row.score is not None
                and target.score is not None
            ):
                paired.append(target.score - row.score)
        if paired:
            differences.append(fmean(paired))
            matched_runs += len(paired)
    interval = None
    if len(differences) > 1:
        generator = random.Random(0)
        samples = sorted(fmean(generator.choices(differences, k=len(differences))) for _ in range(2000))
        interval = (samples[49], samples[1949])
    return PairedSummary(
        len(differences),
        matched_runs,
        len(task_ids),
        fmean(differences) if differences else None,
        interval,
        sum(delta > 0 for delta in differences),
        sum(delta < 0 for delta in differences),
        sum(delta == 0 for delta in differences),
    )


def release_comparisons(
    outcomes: Sequence[EvaluationOutcome], release_ids: Sequence[str], tasks: Sequence[EvaluationTask]
) -> list[FamilyComparison]:
    """Baseline/previous comparisons per family, without changing selection policy."""
    families = sorted({task.family for task in tasks})
    comparisons = []
    for index, release_id in enumerate(release_ids):
        for family in families:
            task_ids = [task.task_id for task in tasks if task.family == family]
            comparisons.append(
                FamilyComparison(
                    release_id,
                    family,
                    paired_summary(outcomes, release_id, release_ids[0], task_ids),
                    paired_summary(outcomes, release_id, release_ids[max(0, index - 1)], task_ids),
                )
            )
    return comparisons


def comparison_markdown(
    outcomes: Sequence[EvaluationOutcome], release_ids: Sequence[str], tasks: Sequence[EvaluationTask] = ()
) -> str:
    """Compare matched valid tasks only; retain coverage beside every score."""
    lines = [
        "# Retained harness release comparison",
        "",
        "Higher scores are better. Deltas use only matched scored runs.",
        "Errors and unrun tasks are not zero scores. Wall time is diagnostic, not serving throughput.",
        "",
        "| Release | Scored / planned | Errors | Unrun | Paired with baseline | Mean paired delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    baseline = {(o.task_id, o.repeat): o for o in outcomes if o.release_id == release_ids[0]}
    for release in release_ids:
        rows = [o for o in outcomes if o.release_id == release]
        deltas = []
        for row in rows:
            before = baseline.get((row.task_id, row.repeat))
            if before is not None and before.score is not None and row.score is not None:
                deltas.append(row.score - before.score)
        mean_delta_label = "N/A" if not deltas else f"{sum(deltas) / len(deltas):+.4f}"
        label = release.replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {label} | {sum(o.status == 'scored' for o in rows)} / {len(rows)} | "
            f"{sum(o.status in {'execution_error', 'invalid_score'} for o in rows)} | "
            f"{sum(o.status == 'unrun' for o in rows)} | {len(deltas)} | {mean_delta_label} |"
        )
    lines.extend(
        [
            "",
            "## Per-task outcomes",
            "",
            "| Task | Repeat | Release | Status | Score | Delta vs baseline | Delta vs previous |",
            "| --- | ---: | --- | --- | ---: | ---: | ---: |",
        ]
    )
    indexed = {(row.release_id, row.task_id, row.repeat): row for row in outcomes}
    for row in outcomes:
        index = release_ids.index(row.release_id)
        before = indexed.get((release_ids[0], row.task_id, row.repeat))
        previous = indexed.get((release_ids[max(0, index - 1)], row.task_id, row.repeat))
        deltas_text = []
        for reference in (before, previous):
            if reference is None or reference.score is None or row.score is None:
                deltas_text.append("N/A")
            else:
                deltas_text.append(f"{row.score - reference.score:+.4f}")
        task_label = row.task_id.replace("|", "\\|").replace("\n", " ")
        release_label = row.release_id.replace("|", "\\|").replace("\n", " ")
        score_label = "N/A" if row.score is None else f"{row.score:.4f}"
        lines.append(
            f"| {task_label} | {row.repeat} | {release_label} | {row.status} | "
            f"{score_label} | {deltas_text[0]} | {deltas_text[1]} |"
        )
    if tasks:
        lines.extend(
            [
                "",
                "## Paired task-family comparisons",
                "",
                "Repetitions are averaged within each task; tasks receive equal weight. Intervals bootstrap paired tasks,",
                "not individual repetitions. They are descriptive, unadjusted for multiple comparisons, and do not account",
                "for provider drift or pretraining contamination. Missing pairs remain excluded; inspect coverage.",
                "",
                "| Release | Family | Paired tasks / planned | Delta vs baseline | 95% interval | Delta vs previous | Regressed tasks vs previous |",
                "| --- | --- | ---: | ---: | --- | ---: | ---: |",
            ]
        )
        for comparison in release_comparisons(outcomes, release_ids, tasks):
            baseline_stats = comparison.vs_baseline
            previous_stats = comparison.vs_previous
            delta = baseline_stats.mean_delta
            previous_delta = previous_stats.mean_delta
            interval = baseline_stats.task_bootstrap_95_percent_interval
            delta_text = "N/A" if delta is None else f"{delta:+.4f}"
            previous_text = "N/A" if previous_delta is None else f"{previous_delta:+.4f}"
            interval_text = "N/A" if interval is None else f"[{interval[0]:+.4f}, {interval[1]:+.4f}]"
            label = comparison.family.replace("|", "\\|").replace("\n", " ")
            release_label = comparison.release_id.replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {release_label} | {label} | {baseline_stats.paired_tasks} / {baseline_stats.planned_tasks} | "
                f"{delta_text} | {interval_text} | {previous_text} | {previous_stats.regressed_tasks} |"
            )
    lines.extend(
        [
            "",
            "## Reported harness usage",
            "",
            "Counts cover observed native responses only, excluding model judges and missing/failed provider responses.",
            "Unknown usage is not zero cost. Currency cost is not inferred.",
            "",
            "| Release | Episodes with both token counts / executed | Reported input tokens | Reported output tokens |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for release_id in release_ids:
        rows = [row for row in outcomes if row.release_id == release_id and row.status != "unrun"]
        known = [row for row in rows if row.input_tokens is not None and row.output_tokens is not None]
        input_total = str(sum(row.input_tokens or 0 for row in known)) if known else "N/A"
        output_total = str(sum(row.output_tokens or 0 for row in known)) if known else "N/A"
        lines.append(f"| {release_id} | {len(known)} / {len(rows)} | {input_total} | {output_total} |")
    lines.extend(["", "Per-task results and conditions: `run.json`. No statistical significance is inferred.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect an exported retained-release evaluation")
    parser.add_argument("run", type=Path, help="run.json from RetainedHarnessEvaluation.run")
    args = parser.parse_args()
    raw = json.loads(args.run.read_text(encoding="utf-8"))
    outcomes = [EvaluationOutcome(**result) for result in raw["results"]]
    print(
        comparison_markdown(
            outcomes,
            [target["release_id"] for target in raw["specification"]["targets"]],
            [EvaluationTask(**task) for task in raw["specification"]["tasks"]],
        )
    )


if __name__ == "__main__":
    main()
