"""Recorded traffic to session records, task digests, and skill groups.

parse_recorded maps a task's last recorded request and its response onto the
official capture's turn schema; the record store sees exactly what their
proxy saw, so the turns come from the model visible message list, not from a
client side chat log. build_aggregate_session folds one task run into the
digest their experiment feeds the evolver: one synthetic turn with the task
prompt and a benchmark summary. group_sessions buckets digests per
referenced skill, with their no skill bucket for the rest.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_READ_TOOLS = {"read", "file_read", "read_file", "readfile"}
EMPTY_GROUP = "__no_skill__"
_RESULT_CONTENT_MAX = 800


def _text(message: dict[str, Any]) -> str:
    parts = message.get("content") or []
    if isinstance(parts, list):
        return " ".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in parts).strip()
    return str(parts or "")


def _norm_call_id(raw: Any) -> str:
    return str(raw or "").replace("-", "").replace("_", "")


def _has_error(content: str) -> bool:
    # Their effective classifier is the content sniff alone; the is_error
    # key their capture reads is never present on their wire.
    return "Traceback" in content[:500] or "Error" in content[:500] or "error" in content[:200]


def _read_skills(tool_calls: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Their extractor: read type tools whose path argument ends in SKILL.md."""
    names: list[str] = []
    for call in tool_calls:
        function = call.get("function") or {}
        if str(function.get("name") or "").strip().lower() not in _READ_TOOLS:
            continue
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except Exception:
            arguments = {}
        if not isinstance(arguments, dict):
            continue
        raw_path = str(arguments.get("path") or arguments.get("file") or "").strip()
        if not raw_path or not raw_path.endswith("SKILL.md"):
            continue
        name = Path(raw_path).parent.name.strip()
        if name and name not in names:
            names.append(name)
    return [{"skill_id": "", "skill_name": name} for name in names]


def _normalize_calls(message: dict[str, Any], call_id_to_name: dict[str, str]) -> list[dict[str, Any]]:
    """An assistant message's tool calls in the official capture shape."""
    calls: list[dict[str, Any]] = []
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function") or {}
        call_id = _norm_call_id(call.get("id"))
        name = str(function.get("name") or "unknown")
        call_id_to_name[call_id] = name
        arguments = function.get("arguments")
        calls.append(
            {
                "id": call_id,
                "function": {
                    "name": name,
                    "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments or {}),
                },
            }
        )
    return calls


def _assistant_turn(message: dict[str, Any], prompt: str, turn_num: int, call_id_to_name: dict[str, str]) -> dict:
    tool_calls = _normalize_calls(message, call_id_to_name)
    # Their capture joins text parts; a tool call only turn records the
    # calls themselves as the response text.
    response = _text(message).strip()
    if not response and tool_calls:
        response = json.dumps(tool_calls, ensure_ascii=False)
    return {
        "turn_num": turn_num,
        "prompt_text": prompt,
        "response_text": response,
        "tool_calls": tool_calls,
        "tool_results": [],
        "tool_observations": [],
        "tool_errors": [],
        "read_skills": _read_skills(tool_calls),
        "injected_skills": [],
        "prm_score": None,
    }


def parse_recorded(payload: dict[str, Any], *, task_id: str = "", session_id: str = "") -> dict[str, Any]:
    """One session in the official turn schema, from a recorded request.

    The recorded payload holds the request's full message list plus the
    response the provider returned, so the last request of a session carries
    every earlier turn and the response supplies the final one.
    """
    turns: list[dict[str, Any]] = []
    call_id_to_name: dict[str, str] = {}
    prompt = ""
    messages = list(payload.get("messages") or [])
    response = payload.get("response") or {}
    if isinstance(response, dict):
        # Buffered responses carry choices[0].message; streamed ones carry the
        # message reassembled by reef.service.streaming.stream_record.
        choices = response.get("choices") or []
        final = choices[0].get("message") if choices and isinstance(choices[0], dict) else response.get("message")
        if isinstance(final, dict):
            messages.append(final)
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "user" and not prompt:
            prompt = _text(message)
        elif role == "assistant":
            turns.append(_assistant_turn(message, prompt, len(turns) + 1, call_id_to_name))
        elif role == "tool" and turns:
            content = _text(message)
            result = {
                "tool_name": call_id_to_name.get(_norm_call_id(message.get("tool_call_id")), "unknown"),
                "tool_call_id": _norm_call_id(message.get("tool_call_id")),
                "content": content[:_RESULT_CONTENT_MAX],
                "has_error": _has_error(content),
            }
            turn = turns[-1]
            turn["tool_results"].append(result)
            turn["tool_observations"].append(dict(result))
            if result["has_error"]:
                turn["tool_errors"].append(dict(result))
    response_text = "\n".join(
        str(turn.get("response_text") or "").strip() for turn in turns if str(turn.get("response_text") or "").strip()
    ).strip()
    return {
        "session_id": session_id or task_id,
        "task_id": task_id,
        "num_turns": len(turns),
        "turns": turns,
        "response_text": response_text,
    }


def annotate_score(session: dict[str, Any], score: float) -> None:
    """Their post grade annotation: benchmark score onto the session."""
    turns = session.get("turns") or []
    if turns:
        turns[-1]["prm_score"] = score
    session["benchmark"] = {"overall_score": score}


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip()).strip("-").lower()
    return slug or "task"


def _notes(record: dict[str, Any]) -> str:
    """Their executor's notes field: error, else response, else the scores."""
    return str(
        record.get("error")
        or record.get("response_text")
        or json.dumps(record.get("breakdown") or {}, ensure_ascii=False)
    )[:400]


def _aggregate_response(task_id: str, record: dict[str, Any]) -> str:
    """Their single rollout benchmark digest, the synthetic turn's response."""
    score = record.get("score")
    score_text = f"{float(score):.3f}" if isinstance(score, (int, float)) else "n/a"
    lines = [
        f"Aggregated benchmark summary for task `{task_id}` (1 rollout(s)).",
        "",
        f"Rollout 0 (device {record.get('device_id', '?')}): success={bool(record.get('success'))} score={score_text}",
    ]
    used = [name for name in record.get("used_skills") or [] if str(name).strip()]
    if used:
        lines.append(f"  Used skills: {', '.join(used)}")
    notes = _notes(record).strip()
    if notes:
        lines.append(f"  Notes: {notes[:800]}")
    breakdown = record.get("breakdown") or {}
    if isinstance(breakdown, dict) and breakdown:
        parts = ", ".join(f"{key}={value}" for key, value in breakdown.items() if key != "mode")
        if parts:
            lines.append(f"  Score breakdown: {parts}")
    lines.append("")
    return "\n".join(lines).strip()


def build_aggregate_session(
    *, task_id: str, prompt: str, record: dict[str, Any], source: dict[str, Any], round_index: int
) -> dict[str, Any]:
    """One aggregated session for one task, their one rollout branch."""
    read_names = read_skill_names(source)
    score = record.get("score")
    scored = isinstance(score, (int, float))
    success = bool(record.get("success"))
    return {
        "session_id": f"agg-r{round_index}-{slugify(task_id)}",
        "task_id": task_id,
        "num_turns": 1,
        "phase": "aggregate",
        "round": round_index,
        "turns": [
            {
                "turn_num": 1,
                "prompt_text": prompt,
                "response_text": _aggregate_response(task_id, record),
                "tool_calls": [],
                "tool_results": [],
                "tool_observations": [],
                "tool_errors": [],
                "read_skills": [{"skill_id": "", "skill_name": name} for name in read_names],
                "injected_skills": [],
                "prm_score": float(score) if scored else None,
            }
        ],
        "aggregate": {
            "task_id": task_id,
            "device_count": 1,
            "rollout_count": 1,
            "scores": [float(score)] if scored else [],
            "mean_score": float(score) if scored else None,
            "score_std": 0.0,
            "success_count": int(success),
            "fail_count": int(not success),
            "stability": "all_success" if success else "all_fail",
        },
    }


def group_sessions(sessions: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Skill name (or the empty group) to sessions, first appearance order."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for session in sessions:
        names = sorted(session.get("_skills_referenced") or [])
        for name in names or [EMPTY_GROUP]:
            groups.setdefault(name, []).append(session)
    return groups


def read_skill_names(source: dict[str, Any]) -> list[str]:
    return sorted(
        {
            str(item.get("skill_name") or "").strip()
            for turn in source.get("turns") or []
            for item in turn.get("read_skills") or []
            if isinstance(item, dict) and str(item.get("skill_name") or "").strip()
        }
    )
