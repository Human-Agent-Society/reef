"""The JSON forms the generator service and its client exchange: tasks, oracle results and played episodes."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

from reef.core.artifact_ref import decode_artifact_ref
from reef.core.records_types import AgentRecord, RequestType
from reef.core.tasks import HarborTask, HarborTaskError
from reef.harness.client.tasks import TaskPlay
from reef.record2dataset.designer import DesignerRequest
from reef.record2dataset.harbor import OracleResult
from reef.record2dataset.inputs import record_document


class WireError(ValueError):
    """A document that does not hold what the message needs."""


def checked_object(document: object, label: str) -> dict[str, object]:
    if not isinstance(document, Mapping):
        raise WireError(f"{label} must be a JSON object")
    return {str(key): value for key, value in document.items()}


def checked_string(document: Mapping[str, object], key: str, *, label: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value:
        raise WireError(f"{label} needs a non-empty string {key!r}")
    return value


def checked_string_list(document: Mapping[str, object], key: str, *, label: str) -> list[str]:
    value = document.get(key, [])
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise WireError(f"{label}: {key!r} must be a list of strings")
    return [str(item) for item in value]


def checked_text_files(document: Mapping[str, object], key: str, *, label: str) -> dict[str, str]:
    value = document.get(key, {})
    if not isinstance(value, Mapping) or any(
        not isinstance(name, str) or not isinstance(text, str) for name, text in value.items()
    ):
        raise WireError(f"{label}: {key!r} must map file names to text")
    return {str(name): text for name, text in value.items()}


def checked_tables(document: Mapping[str, object], key: str, *, label: str) -> dict[str, dict[str, object]]:
    value = document.get(key, {})
    if not isinstance(value, Mapping) or any(not isinstance(table, Mapping) for table in value.values()):
        raise WireError(f"{label}: {key!r} must map table names to objects")
    return {str(name): {str(field): item for field, item in table.items()} for name, table in value.items()}


def designer_request_document(request: DesignerRequest) -> dict[str, object]:
    """Serialize designer inputs without exposing caller-local filesystem paths."""
    if request.inputs.assets:
        raise WireError("local assets must be read by HttpGenerator before serialization")
    return {
        "target": request.inputs.description,
        "skill": request.skill,
        "difficulty": request.difficulty,
        "turn_limit": request.turn_limit,
        "grounding": request.grounding,
        "experience_text": request.experience_text,
        "source_records": [record_document(record) for record in request.inputs.source_records],
        "asset_files": dict(request.inputs.asset_files),
    }


def source_records_from_document(fields: Mapping[str, object], *, scenario: str) -> tuple[AgentRecord, ...]:
    """Read complete source records and reject sources from another scenario."""
    sources = fields.get("source_records", [])
    if not isinstance(sources, list):
        raise WireError("source_records must be a list of record objects")
    records: list[AgentRecord] = []
    for source in sources:
        record = checked_object(source, "a source record")
        if checked_string(record, "scenario", label="a source record") != scenario:
            raise WireError("source records must belong to the proposal's scenario")
        created_at = record.get("created_at")
        if isinstance(created_at, bool) or not isinstance(created_at, (int, float)) or not math.isfinite(created_at):
            raise WireError("a source record's created_at must be a finite number")
        artifact = record.get("artifact_ref")
        records.append(
            AgentRecord(
                agent_record_id=checked_string(record, "agent_record_id", label="a source record"),
                scenario=scenario,
                request_type=RequestType(checked_string(record, "request_type", label="a source record")),
                payload=checked_object(record.get("payload"), "a source record's payload"),
                created_at=float(created_at),
                references=tuple(checked_string_list(record, "references", label="a source record")),
                artifact_ref=(
                    decode_artifact_ref(checked_object(artifact, "artifact_ref")) if artifact is not None else None
                ),
            )
        )
    return tuple(records)


def task_document(task: HarborTask) -> dict[str, object]:
    """A task as JSON: every field of :class:`HarborTask`, the digest beside them."""
    return {
        "name": task.name,
        "instruction": task.instruction,
        "tests": dict(task.tests),
        "environment": dict(task.environment),
        "solution": dict(task.solution),
        "config": {table: dict(values) for table, values in task.config.items()},
        "metadata": dict(task.metadata),
        "source_agent_record_ids": list(task.source_agent_record_ids),
        "digest": task.digest,
    }


def task_from_document(document: object) -> HarborTask:
    """The task a document describes, validated the way :class:`HarborTask` validates itself."""
    fields = checked_object(document, "a task")
    metadata = fields.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise WireError("a task's metadata must be an object")
    sources = fields.get("source_agent_record_ids", [])
    if not isinstance(sources, list) or any(not isinstance(item, str) for item in sources):
        raise WireError("a task's source_agent_record_ids must be a list of strings")
    try:
        return HarborTask(
            name=checked_string(fields, "name", label="a task"),
            instruction=checked_string(fields, "instruction", label="a task"),
            tests=checked_text_files(fields, "tests", label="a task"),
            environment=checked_text_files(fields, "environment", label="a task"),
            solution=checked_text_files(fields, "solution", label="a task"),
            config=checked_tables(fields, "config", label="a task"),
            metadata={str(key): value for key, value in metadata.items()},
            source_agent_record_ids=tuple(sources),
        )
    except HarborTaskError as exc:
        raise WireError(f"not a Harbor task: {exc}") from exc


def oracle_document(result: OracleResult) -> dict[str, object]:
    return {
        "is_solvable": result.is_solvable,
        "reason": result.reason,
        "oracle_reward": result.oracle_reward,
        "nop_reward": result.nop_reward,
    }


def oracle_from_document(document: object) -> OracleResult:
    fields = checked_object(document, "an oracle result")
    is_solvable = fields.get("is_solvable")
    reason = fields.get("reason", "")
    if not isinstance(is_solvable, bool) or not isinstance(reason, str):
        raise WireError("an oracle result needs a boolean is_solvable and a text reason")
    return OracleResult(
        is_solvable=is_solvable,
        reason=reason,
        oracle_reward=checked_optional_number(fields.get("oracle_reward"), "oracle_reward"),
        nop_reward=checked_optional_number(fields.get("nop_reward"), "nop_reward"),
    )


def checked_optional_number(value: object, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WireError(f"{label} must be a number or null")
    return float(value)


def checked_reward_map(value: Mapping[object, object], label: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, reward in value.items():
        if isinstance(reward, bool) or not isinstance(reward, (int, float)):
            raise WireError(f"{label} must map names to numbers")
        result[str(name)] = float(reward)
    return result


def play_document(play: TaskPlay) -> dict[str, object]:
    """One played episode as JSON: what the verifier said and what reached Reef."""
    return {
        "task_path": str(play.task_path),
        "name": play.name,
        "episode_id": play.episode_id,
        "reward": play.reward,
        "rewards": dict(play.rewards),
        "error": play.error,
        "receipts": list(play.receipts),
        "failed_calls": play.failed_calls,
        "report_agent_record_ids": list(play.report_agent_record_ids),
        "trial_uri": play.trial_uri,
    }


def play_from_document(document: object) -> TaskPlay:
    fields = checked_object(document, "a played episode")
    rewards = fields.get("rewards", {})
    if not isinstance(rewards, Mapping):
        raise WireError("a played episode's rewards must be an object")
    receipts = checked_string_list(fields, "receipts", label="a played episode")
    reports = checked_string_list(fields, "report_agent_record_ids", label="a played episode")
    failed_calls = fields.get("failed_calls", 0)
    if isinstance(failed_calls, bool) or not isinstance(failed_calls, int):
        raise WireError("a played episode's failed_calls must be an integer")
    trial_uri = fields.get("trial_uri")
    error = fields.get("error", "")
    if (trial_uri is not None and not isinstance(trial_uri, str)) or not isinstance(error, str):
        raise WireError("a played episode's trial_uri and error must be text")
    return TaskPlay(
        task_path=Path(checked_string(fields, "task_path", label="a played episode")),
        name=checked_string(fields, "name", label="a played episode"),
        episode_id=checked_string(fields, "episode_id", label="a played episode"),
        reward=checked_optional_number(fields.get("reward"), "reward"),
        rewards=checked_reward_map(rewards, "a played episode's rewards"),
        error=error,
        receipts=tuple(receipts),
        failed_calls=failed_calls,
        report_agent_record_ids=tuple(reports),
        trial_uri=trial_uri,
    )
