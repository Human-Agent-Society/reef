"""Terminal episode reports and atomic local run records; never report evaluation."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import cast
from uuid import NAMESPACE_URL, uuid5

type JsonValue = bool | int | float | str | list[JsonValue] | dict[str, JsonValue] | None
type JsonObject = dict[str, JsonValue]


def read_object(path: Path) -> JsonObject:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return cast(JsonObject, value)


def write_object(path: Path, value: JsonObject) -> None:
    """Replace a record atomically, flushing it before updating the cursor."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def stable_id(run_id: str, phase: str, category: str, task_id: str, attempt: int, kind: str) -> str:
    """UUID5 over unambiguous run/task/attempt coordinates."""
    identity = json.dumps([run_id, phase, category, task_id, attempt, kind], separators=(",", ":"))
    return str(uuid5(NAMESPACE_URL, identity))


def terminal_report(method: str, record: JsonObject, teacher_context: str, step: int) -> JsonObject:
    if record["phase"] != "train":
        raise ValueError("evaluation episodes must never produce training reports")
    references = record["references"]
    if not isinstance(references, list) or not references or any(not isinstance(value, str) for value in references):
        raise ValueError("a terminal report requires every ordered episode receipt")
    if len(set(references)) != len(references):
        raise ValueError("episode receipts must not repeat")
    score = record["score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        raise ValueError("a verifier score must be finite")
    metadata: JsonObject = {"teacher_context": teacher_context}
    if method == "sdpo":
        metadata.update(step=step, group=0, rollout=record["attempt"])
    elif method == "opd":
        if teacher_context:
            raise ValueError("OPD teacher context must be empty")
    elif method != "sdft":
        raise ValueError("unknown distillation method")
    return {
        "agent_record_id": record["report_id"],
        "references": references,
        "score": float(score),
        "metadata": metadata,
    }
