"""ATIF construction and projections shared by record processors and consumers.

No I/O or model calls live here. Captured requests and responses are retained
verbatim in ``extra.reef.records``; standard steps make the conversation portable.
Reef's training extension stores exact rollout tensors, never inferred token IDs.
"""

from __future__ import annotations

import json
import math
import mimetypes
from collections.abc import Mapping, Sequence
from typing import Any

from reef.core.batches import TrajectoryItem
from reef.core.records_types import AgentRecord


def source_record_id(item: TrajectoryItem) -> str:
    """The primary recorded source, or the identity of an imported ATIF document."""
    return str(
        item.metadata.get("source_agent_record_id")
        or item.trajectory.get("trajectory_id")
        or item.trajectory.get("session_id")
        or ""
    )


def trajectory_reward(item: TrajectoryItem) -> float:
    """Read the finite reward required by a policy objective."""
    reward = item.metadata.get("reward")
    if isinstance(reward, bool) or not isinstance(reward, (float, int)) or not math.isfinite(reward):
        raise ValueError("training requires a finite reward in ATIF extra.reef.reward")
    return float(reward)


def recorded_payloads(item: TrajectoryItem) -> tuple[Mapping[str, Any], ...]:
    """Read every original provider exchange, in reference order.

    Imported ATIF without captured records is projected into one exchange for
    harness methods. The original ATIF remains on the item, including fields
    that a provider-message projection cannot express.
    """
    records = item.metadata.get("records")
    if records is None:
        return (_project_exchange(item.trajectory),)
    if not isinstance(records, list) or not records:
        raise ValueError("ATIF extra.reef.records must be a non-empty list")
    payloads = []
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(record.get("payload"), Mapping):
            raise ValueError("every ATIF captured record must contain a payload object")
        payloads.append(record["payload"])
    return tuple(payloads)


def recorded_payload(item: TrajectoryItem) -> Mapping[str, Any]:
    """The last captured exchange, retaining the original provider request body."""
    return recorded_payloads(item)[-1]


def make_trajectory(
    records: Sequence[AgentRecord],
    reward: float | None = None,
    feedback: str | Mapping[str, Any] | None = None,
) -> TrajectoryItem:
    """Encode ordered inference records as ATIF, retaining all raw provider fields."""
    if not records:
        raise ValueError("an ATIF trajectory requires at least one inference record")
    steps: list[dict[str, Any]] = []
    history: list[Mapping[str, Any]] = []
    for record in records:
        request, responses = exchange_messages(record.payload)
        shared = 0
        for previous, current in zip(history, request, strict=False):
            if previous != current:
                break
            shared += 1
        for message in request[shared:]:
            _append_message(steps, message, copied=True)
        for message in responses:
            _append_message(steps, message, copied=False)
        history = [*request, *responses]
    if not steps:
        steps.append({"step_id": 1, "source": "agent", "message": "", "extra": {"reef": {"text_available": False}}})
    primary = records[-1].agent_record_id
    agent: dict[str, Any] = {"name": "recorded-agent", "version": "unknown"}
    model = records[-1].payload.get("model")
    if isinstance(model, str):
        agent["model_name"] = model
    return TrajectoryItem(
        {
            "schema_version": "ATIF-v1.7",
            "session_id": records[0].agent_record_id,
            "trajectory_id": primary,
            "agent": agent,
            "steps": steps,
            "extra": {
                "reef": {
                    "source_agent_record_id": primary,
                    "reward": reward,
                    "feedback": feedback,
                    "records": [
                        {
                            "agent_record_id": record.agent_record_id,
                            "created_at": record.created_at,
                            "payload": dict(record.payload),
                        }
                        for record in records
                    ],
                }
            },
        },
        source_agent_record_ids=tuple(record.agent_record_id for record in records),
    )


def exchange_messages(payload: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    response = payload.get("response", {})
    response = response if isinstance(response, Mapping) else {"content": response}
    training = response.get("training", {})
    training = training if isinstance(training, Mapping) else {}
    request = training.get(
        "request_messages", payload.get("messages", payload.get("input", payload.get("prompt", [])))
    )
    if isinstance(request, str):
        request = [{"role": "user", "content": request}]
    request = [entry for entry in request if isinstance(entry, Mapping)] if isinstance(request, list) else []
    system = payload.get("system", payload.get("instructions"))
    if system and not any(message.get("role") == "system" for message in request):
        request.insert(0, {"role": "system", "content": system})
    message = training.get("response_message")
    if isinstance(message, Mapping):
        return request, [message]
    choices = response.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        message = choices[0].get("message")
        if isinstance(message, Mapping):
            return request, [message]
        return request, [{"role": "assistant", "content": choices[0].get("text", "")}]
    output = response.get("output")
    if isinstance(output, list):
        return request, [entry for entry in output if isinstance(entry, Mapping)]
    message = response.get("message")
    if isinstance(message, Mapping):
        return request, [message]
    return request, [
        {
            "role": "assistant",
            "content": response.get("content", response.get("output_text", "")),
            **({"text_available": False} if not response else {}),
        }
    ]


def _content(value: Any) -> str | list[dict[str, Any]]:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if not isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    parts: list[dict[str, Any]] = []
    for part in value:
        if not isinstance(part, Mapping):
            parts.append({"type": "text", "text": str(part)})
            continue
        kind = part.get("type")
        if kind in ("text", "input_text", "output_text"):
            parts.append({"type": "text", "text": str(part.get("text", ""))})
        elif kind in ("image", "image_url", "input_image"):
            source = part.get("source", {})
            image_url = part.get("image_url", {})
            path = image_url.get("url") if isinstance(image_url, Mapping) else image_url
            if isinstance(source, Mapping):
                path = path or source.get("path") or source.get("url")
                if source.get("type") == "base64":
                    path = f"data:{source.get('media_type')};base64,{source.get('data')}"
            if not isinstance(path, str):
                parts.append({"type": "text", "text": json.dumps(dict(part), ensure_ascii=False)})
                continue
            media_type = (source.get("media_type") if isinstance(source, Mapping) else None) or (
                path[5:].split(";", 1)[0]
                if path.startswith("data:")
                else mimetypes.guess_type(path.split("?", 1)[0])[0]
            )
            if media_type not in ("image/jpeg", "image/png", "image/gif", "image/webp"):
                # Keep unknown media verbatim instead of guessing its MIME type.
                parts.append({"type": "text", "text": json.dumps(dict(part), ensure_ascii=False)})
            else:
                parts.append({"type": "image", "source": {"media_type": media_type, "path": path}})
        elif kind not in ("tool_use", "tool_result", "thinking", "redacted_thinking"):
            parts.append({"type": "text", "text": json.dumps(dict(part), ensure_ascii=False)})
    if all(part["type"] == "text" for part in parts):
        return "".join(part["text"] for part in parts)
    return parts


def _append_message(steps: list[dict[str, Any]], message: Mapping[str, Any], *, copied: bool) -> None:
    role = message.get("role", "assistant")
    kind = message.get("type")
    content = message.get("content", "")
    blocks = content if isinstance(content, list) else []
    results = [part for part in blocks if isinstance(part, Mapping) and part.get("type") == "tool_result"]
    if role == "tool" or kind == "function_call_output":
        results.append(
            {
                "tool_use_id": message.get("tool_call_id", message.get("call_id")),
                "content": message.get("output", content),
            }
        )
    for result in results:
        call_id = result.get("tool_use_id")
        target = next(
            (
                step
                for step in reversed(steps)
                if any(call.get("tool_call_id") == call_id for call in step.get("tool_calls", []))
            ),
            None,
        )
        if target is not None:
            target.setdefault("observation", {"results": []})["results"].append(
                {
                    "source_call_id": call_id,
                    "content": _content(result.get("content")),
                    "extra": {"provider_result": dict(result)},
                }
            )
        else:
            steps.append(
                {
                    "step_id": len(steps) + 1,
                    "source": "user",
                    "message": _content(result.get("content")),
                    "extra": {"provider_message": dict(message)},
                }
            )
    if role == "tool" or kind == "function_call_output":
        return
    remaining = [part for part in blocks if not isinstance(part, Mapping) or part.get("type") != "tool_result"]
    if results and not remaining:
        return
    source = "agent" if role == "assistant" else "system" if role in ("system", "developer") else "user"
    step: dict[str, Any] = {
        "step_id": len(steps) + 1,
        "source": source,
        "message": _content(remaining if blocks else content),
        "extra": {"provider_message": dict(message)},
    }
    if copied and source == "agent":
        step["is_copied_context"] = True
    calls = list(message.get("tool_calls") or [])
    calls.extend(
        {"id": part.get("id"), "function": {"name": part.get("name"), "arguments": part.get("input", {})}}
        for part in blocks
        if isinstance(part, Mapping) and part.get("type") == "tool_use"
    )
    if kind == "function_call":
        calls.append({"id": message.get("call_id"), "function": message})
    if calls and source == "agent":
        step["tool_calls"] = [_tool_call(call, len(steps) + 1, index) for index, call in enumerate(calls)]
    reasoning = message.get("reasoning_content") or "".join(
        str(part.get("thinking", ""))
        for part in blocks
        if isinstance(part, Mapping) and part.get("type") == "thinking"
    )
    if reasoning and source == "agent":
        step["reasoning_content"] = reasoning
    steps.append(step)


def _tool_call(call: Mapping[str, Any], step_id: int, index: int) -> dict[str, Any]:
    function = call.get("function", call)
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"raw_arguments": arguments}
    if not isinstance(arguments, Mapping):
        arguments = {"raw_arguments": arguments}
    return {
        "tool_call_id": str(call.get("id") or f"call-{step_id}-{index}"),
        "function_name": str(function.get("name") or "unknown"),
        "arguments": dict(arguments),
    }


def _provider_content(content: Any) -> Any:
    """Project ATIF multimodal parts into OpenAI message content."""
    if not isinstance(content, list):
        return content
    return [
        (
            {"type": "image_url", "image_url": {"url": part["source"]["path"]}}
            if part.get("type") == "image"
            else dict(part)
        )
        for part in content
    ]


def _project_exchange(trajectory: Mapping[str, Any]) -> Mapping[str, Any]:
    messages: list[dict[str, Any]] = []
    for step in trajectory["steps"]:
        source = step["source"]
        message: dict[str, Any] = {
            "role": "assistant" if source == "agent" else source,
            "content": _provider_content(step["message"]),
        }
        if step.get("reasoning_content"):
            message["reasoning_content"] = step["reasoning_content"]
        if step.get("tool_calls"):
            message["tool_calls"] = [
                {
                    "id": call["tool_call_id"],
                    "type": "function",
                    "function": {"name": call["function_name"], "arguments": json.dumps(call["arguments"])},
                }
                for call in step["tool_calls"]
            ]
        messages.append(message)
        messages.extend(
            {
                **(
                    {"role": "tool", "tool_call_id": observation["source_call_id"]}
                    if observation.get("source_call_id") is not None
                    else {"role": "user"}
                ),
                "content": _provider_content(observation.get("content", "")),
            }
            for observation in (step.get("observation") or {}).get("results", [])
        )
    final = messages[-1] if messages and messages[-1]["role"] == "assistant" else None
    return {
        "messages": messages[:-1] if final is not None else messages,
        "response": {"choices": [{"message": final}]} if final is not None else {},
    }
