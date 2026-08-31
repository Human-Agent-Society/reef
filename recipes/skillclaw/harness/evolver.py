"""The official evolve server's LLM stages, prompts ported byte for byte.

Ported from benchmarks/skill_claw/evolver.py at commit 0519eefb; the one
adaptation is the endpoint binding: ``chat_client`` wraps the ``ModelBinding``
reef hands ``propose`` (the deployment's upstream), so the evolver is the
model under test and this module never names an endpoint or holds a key.

summarize (temperature 0.2) condenses each session; judge (0.1) scores
sessions whose grader failed; decide (0.4) picks improve_skill,
optimize_description, create_skill, or skip per skill group; merge (0.3)
resolves a same name collision. The chat client retries six times with
backoff, as theirs does, and an unparseable reply degrades to skip.
"""

from __future__ import annotations

import json
import random
import re
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from .prompts import CREATE_SYSTEM, EVOLVE_SYSTEM, JUDGE_SYSTEM, MERGE_SYSTEM, SUMMARIZE_SYSTEM

MAX_TOKENS = 8192
SUMMARY_MAX_TOKENS = 100000
SUMMARY_TEMPERATURE = 0.2
EVOLVE_TEMPERATURE = 0.4
MERGE_TEMPERATURE = 0.3
JUDGE_TEMPERATURE = 0.1
JUDGE_MAX_TOKENS = 1200
MAX_SESSIONS_PER_GROUP = 30
MAX_RETRIES = 6

_PROMPT_MAX = 400
_RESPONSE_MAX = 400
_TOOL_ARG_MAX = 400
_TOOL_RESULT_MAX = 400
_TOOL_ERR_MAX = 300
_MAX_TOOLS_PER_STEP = 8

# Their summarizer payload budgets, tighter than the trajectory's.
_PAYLOAD_PROMPT_MAX = 500
_PAYLOAD_RESPONSE_MAX = 400
_PAYLOAD_TOOL_MAX = 240
_PAYLOAD_CALLS_PER_TURN = 6
_PAYLOAD_OBSERVATIONS_PER_TURN = 4
_ARTIFACT_MAX = 1200

_JUDGE_WEIGHTS = {
    "task_completion": 0.55,
    "response_quality": 0.30,
    "efficiency": 0.05,
    "tool_usage": 0.10,
}


def _normalize_temperature(model: str, requested: float) -> float:
    if str(model or "").strip().lower() in {"kimi-k2.5", "ccr/kimi-k2.5"}:
        return 1
    return requested


class ModelLike(Protocol):
    """The slice of reef's ``ModelBinding`` the evolver uses."""

    model: str

    def chat(self, messages: Sequence[Mapping[str, Any]], *, timeout_s: float | None = None, **params: Any) -> str: ...


class ModelsLike(Protocol):
    served: ModelLike


#: Their chat signature, which every stage below takes as ``llm``.
ChatFn = Callable[[str, str, float, int], str]


def chat_client(model: ModelLike) -> ChatFn:
    """Their chat client's loop over ``model``: six attempts, exponential
    backoff capped at 30s, the temperature and stream-only fallbacks."""

    def chat(system: str, user: str, temperature: float, max_tokens: int = MAX_TOKENS) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        params: dict[str, Any] = {
            "max_tokens": max_tokens,
            "temperature": _normalize_temperature(model.model, temperature),
        }
        for attempt in range(MAX_RETRIES):
            try:
                return model.chat(messages, timeout_s=600, **params)
            except Exception as exc:  # noqa: PERF203 - their retry loop semantics
                status = getattr(exc, "status", None)
                detail = str(getattr(exc, "detail", "") or "")
                if status == 400 and "'temperature' is not supported" in detail:
                    params.pop("temperature", None)
                    continue
                if status == 400 and "Stream must be set to true" in detail:
                    return model.chat(messages, timeout_s=120, stream=True, **params)
                if attempt < MAX_RETRIES - 1:
                    time.sleep(min(2**attempt + random.uniform(0, 1), 30))
                    continue
                raise
        raise RuntimeError("unreachable: the loop above returns a reply or re-raises")

    return chat


def _clip_line(text: Any, limit: int) -> str:
    flat = str(text or "").strip().replace("\n", " ")
    return flat if len(flat) <= limit else flat[:limit] + "..."


def _format_tool_calls(turn: dict[str, Any]) -> list[str]:
    """Their tool call lines: each call with its matched result or error."""
    result_by_id: dict[str, dict[str, Any]] = {}
    for result in turn.get("tool_results") or []:
        if isinstance(result, dict) and result.get("tool_call_id"):
            result_by_id[result["tool_call_id"]] = result
    for observation in turn.get("tool_observations") or []:
        if isinstance(observation, dict) and observation.get("tool_call_id"):
            result_by_id.setdefault(observation["tool_call_id"], observation)
    error_by_tool: dict[str, list[str]] = {}
    for error in turn.get("tool_errors") or []:
        if isinstance(error, dict):
            error_by_tool.setdefault(str(error.get("tool_name") or ""), []).append(
                _clip_line(error.get("content", ""), _TOOL_ERR_MAX)
            )
    calls = turn.get("tool_calls") or []
    lines: list[str] = []
    for call in calls[:_MAX_TOOLS_PER_STEP]:
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = str(function.get("name") or "unknown")
        arguments = _clip_line(function.get("arguments", ""), _TOOL_ARG_MAX)
        outcome = ""
        result = result_by_id.get(str(call.get("id") or ""))
        if result:
            content = _clip_line(result.get("content", ""), _TOOL_RESULT_MAX)
            ok = f" -> ok {content}" if content else " -> ok"
            outcome = f" -> ERR {content}" if result.get("has_error") else ok
        if not outcome and name in error_by_tool:
            outcome = f" -> ERR {error_by_tool[name][0]}"
        lines.append(f"    {name}({arguments}){outcome}")
    called = {
        str((call.get("function") or {}).get("name") or "")
        for call in calls[:_MAX_TOOLS_PER_STEP]
        if isinstance(call, dict)
    }
    leftovers = [
        f"    WARN {tool}: {error}" for tool, errors in error_by_tool.items() if tool not in called for error in errors
    ]
    lines.extend(leftovers[:3])
    if len(calls) > _MAX_TOOLS_PER_STEP:
        lines.append(f"    ... +{len(calls) - _MAX_TOOLS_PER_STEP} more tool calls")
    return lines


def _format_step(turn: dict[str, Any], number: int, *, show_prompt: bool) -> str:
    skills = [
        item.get("skill_name", "").strip() if isinstance(item, dict) else str(item or "").strip()
        for item in turn.get("read_skills") or []
    ]
    skills = [name for name in skills if name]
    header_parts = [f"[Step {number}]"]
    if turn.get("prm_score") is not None:
        header_parts.append(f"PRM={turn['prm_score']}")
    if skills:
        header_parts.append(f"read_skills={skills}")
    lines = [" | ".join(header_parts)]
    prompt = _clip_line(turn.get("prompt_text", ""), _PROMPT_MAX)
    if show_prompt and prompt:
        lines.append(f"  User: {prompt}")
    tool_lines = _format_tool_calls(turn)
    if tool_lines:
        lines.append("  Tools:")
        lines.extend(tool_lines)
    response = _clip_line(turn.get("response_text", ""), _RESPONSE_MAX)
    if response:
        lines.append(f"  Agent: {response}")
    return "\n".join(lines)


def build_trajectory(session: dict[str, Any]) -> str:
    """Their programmatic trajectory: one step per turn, prompt shown once."""
    turns = session.get("turns") or []
    if not turns:
        return "(empty session)"
    return "\n".join(_format_step(turn, number, show_prompt=number == 1) for number, turn in enumerate(turns, 1))


def attach_metadata(session: dict[str, Any]) -> None:
    """Their enrichment: skills referenced, PRM stats, tool error flag."""
    skills: set[str] = set()
    prm_scores: list[float] = []
    has_tool_errors = False
    for turn in session.get("turns") or []:
        for item in turn.get("read_skills") or []:
            name = item.get("skill_name", "").strip() if isinstance(item, dict) else str(item or "").strip()
            if name:
                skills.add(name)
        for item in turn.get("injected_skills") or []:
            if str(item or "").strip():
                skills.add(str(item).strip())
        if turn.get("prm_score") is not None:
            prm_scores.append(float(turn["prm_score"]))
        if turn.get("tool_errors"):
            has_tool_errors = True
    session["_skills_referenced"] = sorted(skills)
    session["_prm_scores"] = prm_scores
    session["_avg_prm"] = round(sum(prm_scores) / len(prm_scores), 3) if prm_scores else None
    session["_has_tool_errors"] = has_tool_errors
    session["_trajectory"] = build_trajectory(session)


def _payload(session: dict[str, Any]) -> dict[str, Any]:
    """Their summarizer payload: compact structured turns plus the aggregate."""
    turns = session.get("turns") or []
    first_prompt = (turns[0].get("prompt_text") or "")[:_PAYLOAD_PROMPT_MAX] if turns else ""
    interactions: list[dict[str, Any]] = []
    for index, turn in enumerate(turns):
        raw_prompt = (turn.get("prompt_text") or "")[:_PAYLOAD_PROMPT_MAX]
        interaction: dict[str, Any] = {
            "prompt": raw_prompt if index == 0 else ("(same)" if raw_prompt == first_prompt else raw_prompt),
            "response": (turn.get("response_text") or "")[:_PAYLOAD_RESPONSE_MAX],
            "prm_score": turn.get("prm_score"),
        }
        read_skills = turn.get("read_skills") or []
        if read_skills:
            interaction["read_skills"] = [
                item.get("skill_name", "") if isinstance(item, dict) else str(item or "") for item in read_skills
            ]
        injected = [str(item or "").strip() for item in turn.get("injected_skills") or [] if str(item or "").strip()]
        if injected:
            interaction["injected_skills"] = injected
        calls = [
            {
                "id": str(call.get("id") or ""),
                "name": str((call.get("function") or {}).get("name") or "unknown"),
                "arguments": _clip_line((call.get("function") or {}).get("arguments") or "", _PAYLOAD_TOOL_MAX),
            }
            for call in (turn.get("tool_calls") or [])[:_PAYLOAD_CALLS_PER_TURN]
            if isinstance(call, dict)
        ]
        if calls:
            interaction["tool_calls"] = calls
        for key, budget in (
            ("tool_results", _PAYLOAD_CALLS_PER_TURN),
            ("tool_observations", _PAYLOAD_OBSERVATIONS_PER_TURN),
        ):
            items = [
                {
                    "tool_name": str(item.get("tool_name") or ""),
                    "tool_call_id": str(item.get("tool_call_id") or ""),
                    "content": _clip_line(item.get("content", ""), _PAYLOAD_TOOL_MAX),
                    "has_error": bool(item.get("has_error")),
                }
                for item in (turn.get(key) or [])[:budget]
                if isinstance(item, dict)
            ]
            if items:
                interaction[key] = items
        if turn.get("tool_errors"):
            interaction["tool_errors"] = turn["tool_errors"]
        interactions.append(interaction)
    payload: dict[str, Any] = {
        "session_id": session.get("session_id", ""),
        "total_interactions": len(turns),
        "interactions": interactions,
    }
    aggregate = session.get("aggregate")
    if aggregate:
        payload["aggregate"] = {
            key: aggregate.get(key)
            for key in ("rollout_count", "scores", "mean_score", "success_count", "fail_count", "stability")
        }
    return payload


def summarize(session: dict[str, Any], llm: ChatFn) -> str:
    """Their per session causal summary; an LLM failure leaves it empty."""
    try:
        return llm(
            SUMMARIZE_SYSTEM,
            json.dumps(_payload(session), ensure_ascii=False),
            SUMMARY_TEMPERATURE,
            SUMMARY_MAX_TOKENS,
        )
    except Exception:
        return ""


def judge_needed(session: dict[str, Any]) -> bool:
    """Their skip test inverted: only sessions with no reliable score."""
    if not session.get("turns"):
        return False
    judge_scores = session.get("_judge_scores")
    if isinstance(judge_scores, dict) and isinstance(judge_scores.get("overall_score"), (int, float)):
        return False
    benchmark = session.get("benchmark")
    if isinstance(benchmark, dict) and isinstance(benchmark.get("overall_score"), (int, float)):
        return False
    aggregate = session.get("aggregate")
    return not (isinstance(aggregate, dict) and isinstance(aggregate.get("mean_score"), (int, float)))


def _judge_output_artifacts(session: dict[str, Any], max_artifacts: int = 4) -> list[dict[str, str]]:
    """Their extractor: write tool call payloads, path plus content."""
    artifacts: list[dict[str, str]] = []
    for turn in session.get("turns") or []:
        for call in turn.get("tool_calls") or []:
            function = call.get("function") if isinstance(call, dict) else None
            if not isinstance(function, dict) or str(function.get("name") or "").strip() != "write":
                continue
            try:
                arguments = json.loads(function.get("arguments") or "")
            except (json.JSONDecodeError, TypeError):
                continue
            path = str(arguments.get("path") or "").strip()
            content = arguments.get("content")
            if not path or content is None:
                continue
            artifacts.append({"path": path, "content": _clip_block(content, _ARTIFACT_MAX)})
            if len(artifacts) >= max_artifacts:
                return artifacts
    return artifacts


def _judge_source_artifacts(session: dict[str, Any], max_artifacts: int = 6) -> list[dict[str, str]]:
    """Their extractor: successful read results outside /root/, deduped."""
    artifacts: list[dict[str, str]] = []
    seen: set[str] = set()
    for turn in session.get("turns") or []:
        args_by_id: dict[str, dict[str, Any]] = {}
        for call in turn.get("tool_calls") or []:
            function = call.get("function") if isinstance(call, dict) else None
            if not isinstance(function, dict) or str(function.get("name") or "").strip() != "read":
                continue
            try:
                arguments = json.loads(function.get("arguments") or "")
            except (json.JSONDecodeError, TypeError):
                continue
            call_id = str(call.get("id") or "").replace("_", "")
            if call_id:
                args_by_id[call_id] = arguments
        for result in turn.get("tool_results") or []:
            if not isinstance(result, dict) or str(result.get("tool_name") or "").strip() != "read":
                continue
            if result.get("has_error"):
                continue
            arguments = args_by_id.get(str(result.get("tool_call_id") or "").replace("_", ""))
            if not isinstance(arguments, dict):
                continue
            path = str(arguments.get("path") or "").strip()
            if not path or path in seen or path.startswith("/root/"):
                continue
            content = str(result.get("content") or "").strip()
            if not content or content == "(see attached image)":
                continue
            artifacts.append({"path": path, "content": _clip_block(content, _ARTIFACT_MAX)})
            seen.add(path)
            if len(artifacts) >= max_artifacts:
                return artifacts
    return artifacts


def _clip_block(value: Any, max_chars: int) -> str:
    text = str(value or "").strip()
    return text if len(text) <= max_chars else text[:max_chars] + "..."


def judge(session: dict[str, Any], llm: ChatFn) -> dict[str, Any] | None:
    """Their session judge: backfill an overall score for unscored sessions."""
    payload = {
        "session_id": session.get("session_id"),
        "num_turns": session.get("num_turns"),
        "skills_referenced": sorted(session.get("_skills_referenced") or []),
        "has_tool_errors": bool(session.get("_has_tool_errors")),
        "prior_prm_scores": list(session.get("_prm_scores") or []),
        "avg_prm_before_judge": session.get("_avg_prm"),
        "source_artifacts": _judge_source_artifacts(session),
        "output_artifacts": _judge_output_artifacts(session),
        "trajectory": session.get("_trajectory") or "",
        "summary": session.get("_summary") or "",
    }
    try:
        raw = llm(JUDGE_SYSTEM, json.dumps(payload, ensure_ascii=False), JUDGE_TEMPERATURE, JUDGE_MAX_TOKENS)
    except Exception:
        return None
    scores = _parse_judge(raw)
    if not scores:
        return None
    turns = session.get("turns") or []
    if turns:
        turns[-1]["prm_score"] = scores["overall_score"]
    session["_judge_scores"] = scores
    session["_prm_scores"] = [scores["overall_score"]]
    session["_avg_prm"] = scores["overall_score"]
    return scores


def _parse_judge(raw: str) -> dict[str, Any] | None:
    clean = re.sub(r"```(?:json)?\s*", "", str(raw or "").strip()).strip().rstrip("`")
    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    dimensions: dict[str, float] = {}
    for key in _JUDGE_WEIGHTS:
        value = payload.get(key)
        # Their number test excludes booleans; a boolean voids the reply.
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None
        dimensions[key] = round(min(1.0, max(0.0, float(value))), 3)
    overall = round(sum(dimensions[key] * weight for key, weight in _JUDGE_WEIGHTS.items()), 3)
    return {**dimensions, "overall_score": overall, "rationale": str(payload.get("rationale") or "")}


def _evidence(sessions: list[dict[str, Any]]) -> str:
    """Their session evidence: header, trajectory, analysis per session."""
    blocks = []
    for session in sessions[:MAX_SESSIONS_PER_GROUP]:
        header = f"### Session {session.get('session_id', '?')}"
        if session.get("_avg_prm") is not None:
            header += f", avg PRM: {session['_avg_prm']}"
        aggregate = session.get("aggregate") or {}
        if aggregate:
            parts = []
            if aggregate.get("rollout_count"):
                parts.append(f"{aggregate['rollout_count']} rollouts")
            if aggregate.get("mean_score") is not None:
                parts.append(f"mean ORM={aggregate['mean_score']:.3f}")
            if aggregate.get("success_count") or aggregate.get("fail_count"):
                parts.append(f"success={aggregate.get('success_count', 0)} fail={aggregate.get('fail_count', 0)}")
            if aggregate.get("stability"):
                parts.append(f"stability={aggregate['stability']}")
            if parts:
                header += f", {', '.join(parts)}"
        if session.get("_has_tool_errors"):
            header += ", has tool errors"
        if session.get("_skills_referenced"):
            header += f", skills: {sorted(session['_skills_referenced'])}"
        parts = [header]
        if session.get("_trajectory"):
            parts.append(f"**Trajectory**:\n{session['_trajectory']}")
        if session.get("_summary"):
            parts.append(f"**Analysis**:\n{session['_summary']}")
        if not session.get("_trajectory") and not session.get("_summary"):
            parts.append("(no data)")
        blocks.append("\n\n".join(parts))
    if len(sessions) > MAX_SESSIONS_PER_GROUP:
        blocks.append(f"\n... and {len(sessions) - MAX_SESSIONS_PER_GROUP} more sessions")
    return "\n\n---\n\n".join(blocks)


def decide(
    *,
    skill_name: str,
    current: dict[str, Any] | None,
    sessions: list[dict[str, Any]],
    existing_names: list[str],
    llm: ChatFn,
) -> dict[str, Any]:
    """One combined decision and execution call for one skill group."""
    system = EVOLVE_SYSTEM.replace("{skill_name}", skill_name)
    skill_block = ""
    if current:
        skill_block = (
            f"## Current skill\n\n"
            f"Name: {current.get('name', '')}\n"
            f"Description: {current.get('description', '')}\n"
            f"Category: {current.get('category', 'general')}\n\n"
            f"Content:\n```\n{current.get('content', '')}\n```\n\n"
        )
    user = (
        f"{skill_block}"
        f"## Session evidence ({len(sessions)} sessions)\n\n"
        f"{_evidence(sessions)}\n\n"
        f"## Existing skill names in the library\n\n"
        f"{', '.join(existing_names) or '(none)'}\n"
    )
    return _parse(llm(system, user, EVOLVE_TEMPERATURE), skill_name)


def create(*, sessions: list[dict[str, Any]], existing_names: list[str], llm: ChatFn) -> dict[str, Any]:
    """Their create call for the bucket of sessions referencing no skill."""
    user = (
        f"## Session evidence ({len(sessions)} sessions)\n\n"
        f"{_evidence(sessions)}\n\n"
        f"## Existing skill names in the library\n\n"
        f"{', '.join(existing_names) or '(none)'}\n"
    )
    return _parse(llm(CREATE_SYSTEM, user, EVOLVE_TEMPERATURE), "")


def merge(existing: dict[str, Any], incoming: dict[str, Any], llm: ChatFn) -> dict[str, Any] | None:
    """Their conflict merge: two versions of one name into a superior one."""
    user = (
        f"## Version A (currently in shared storage, v{existing.get('_version', '?')})\n\n"
        f"Name: {existing.get('name', '')}\n"
        f"Description: {existing.get('description', '')}\n"
        f"Category: {existing.get('category', 'general')}\n\n"
        f"Content:\n```\n{existing.get('content', '')}\n```\n\n"
        f"---\n\n"
        f"## Version B (newly evolved)\n\n"
        f"Name: {incoming.get('name', '')}\n"
        f"Description: {incoming.get('description', '')}\n"
        f"Category: {incoming.get('category', 'general')}\n\n"
        f"Content:\n```\n{incoming.get('content', '')}\n```"
    )
    raw = llm(MERGE_SYSTEM, user, MERGE_TEMPERATURE)
    clean = re.sub(r"```(?:json)?\s*", "", str(raw or "").strip()).strip().rstrip("`")
    try:
        parsed = json.loads(clean)
        if isinstance(parsed, dict) and parsed.get("name"):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    start, end = clean.find("{"), clean.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(clean[start : end + 1])
            if isinstance(parsed, dict) and parsed.get("name"):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def _parse(reply: str, skill_name: str) -> dict[str, Any]:
    """Their result parse with the same fallbacks; unparseable means skip."""
    clean = re.sub(r"```(?:json)?\s*", "", reply.strip()).strip().rstrip("`")
    try:
        result = json.loads(clean)
    except json.JSONDecodeError:
        start, end = clean.find("{"), clean.rfind("}")
        if start == -1 or end <= start:
            return {"action": "skip", "rationale": "no JSON object in reply"}
        try:
            result = json.loads(clean[start : end + 1])
        except json.JSONDecodeError:
            return {"action": "skip", "rationale": "invalid JSON reply"}
    if not isinstance(result, dict):
        return {"action": "skip", "rationale": "reply is not an object"}
    # Their parse: only skip stops here; other actions flow through, and a
    # missing name is backfilled, never overwritten.
    action = result.get("action", "skip")
    if action == "skip":
        return {"action": "skip", "rationale": result.get("rationale", "")}
    skill = result.get("skill")
    if not isinstance(skill, dict):
        return {"action": "skip", "rationale": "action without skill data"}
    if action == "create_skill":
        if not skill.get("name"):
            return {"action": "skip", "rationale": "create without a name"}
        if skill["name"] == skill_name:
            action = "improve_skill"
    elif skill_name and not skill.get("name"):
        skill["name"] = skill_name
    return {"action": action, "rationale": result.get("rationale", ""), "skill": skill}
