"""ATIF construction and projections shared by record processors and consumers.

No I/O or model calls live here. Captured requests and responses are retained
verbatim in ``extra.reef.records``; standard steps make the conversation portable.
A training component renders the steps into the tokens that it trains (``reef.core.chat_rendering``).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from reef.core.atif_payloads import (
    OPENAI_CHAT,
    atif_agent,
    canonical_json,
    folded_steps,
    message_entry,
    payload_provider,
)
from reef.core.job_data import TrajectorySample
from reef.core.records_types import AgentRecord
from reef.core.trajectory_index import comparable_message


def source_record_id(item: TrajectorySample) -> str:
    """The primary recorded source, or the identity of an imported ATIF document."""
    return str(
        item.metadata.get("source_agent_record_id")
        or item.trajectory.get("trajectory_id")
        or item.trajectory.get("session_id")
        or ""
    )


def trajectory_reward(item: TrajectorySample) -> float:
    """Read the finite reward required by a policy objective."""
    reward = item.metadata.get("reward")
    if isinstance(reward, bool) or not isinstance(reward, (float, int)) or not math.isfinite(reward):
        raise ValueError("training requires a finite reward in ATIF extra.reef.reward")
    return float(reward)


def recorded_payloads(item: TrajectorySample) -> tuple[Mapping[str, Any], ...]:
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


def recorded_payload(item: TrajectorySample) -> Mapping[str, Any]:
    """The last captured exchange, retaining the original provider request body."""
    return recorded_payloads(item)[-1]


def make_trajectory(
    records: Sequence[AgentRecord],
    reward: float | None = None,
    feedback: str | Mapping[str, Any] | None = None,
) -> TrajectorySample:
    """Encode ordered inference records as ATIF, retaining all raw provider fields."""
    if not records:
        raise ValueError("an ATIF trajectory requires at least one inference record")
    entries: list[list[dict[str, Any]]] = []
    history: list[Mapping[str, Any]] = []
    for record in records:
        provider = payload_provider(record.payload) or OPENAI_CHAT
        request, responses = exchange_messages(record.payload)
        shared = 0
        for previous, current in zip(history, request, strict=False):
            if canonical_json(comparable_message(previous)) != canonical_json(comparable_message(current)):
                break
            shared += 1
        entries.extend(message_entry(message, provider, copied=True) for message in request[shared:])
        entries.extend(message_entry(message, provider, copied=False) for message in responses)
        history = [*request, *responses]
    steps = folded_steps(entries)
    if not steps:
        steps.append({"step_id": 1, "source": "agent", "message": "", "extra": {"reef": {"text_available": False}}})
    primary = records[-1].agent_record_id
    agent: dict[str, Any] = {"name": "recorded-agent", "version": "unknown"}
    model = records[-1].payload.get("model")
    if isinstance(model, str):
        agent["model_name"] = model
    last = records[-1].payload
    if "tools" in last:
        agent.update(atif_agent(last["tools"], payload_provider(last) or OPENAI_CHAT))
    return TrajectorySample(
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


def _provider_content(content: Any) -> Any:
    """Project ATIF multimodal parts into OpenAI message content.

    ``ValueError``: a part is not an object, a text part has a ``text`` that is not a string, or an image part has
    no ``source.path``.
    """
    if not isinstance(content, list):
        return content
    parts: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, Mapping):
            raise ValueError("ATIF content parts must be objects")
        if part.get("type") == "text" and not isinstance(part.get("text", ""), str):
            raise ValueError("an ATIF text part needs a string text")
        if part.get("type") != "image":
            parts.append(dict(part))
            continue
        source = part.get("source")
        if not isinstance(source, Mapping) or "path" not in source:
            raise ValueError("an ATIF image part needs source.path")
        parts.append({"type": "image_url", "image_url": {"url": source["path"]}})
    return parts


def atif_messages(trajectory: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The provider chat messages of an ATIF document's ``steps``, in step order.

    An ``agent`` step becomes an ``assistant`` message with its reasoning and tool calls; a step of another source
    keeps the source as its role. Each result of a step's observation follows the step as a ``tool`` message, or as
    a ``user`` message when it names no tool call. ``ValueError``: ``steps`` is not a list of objects, each with a
    string ``source``; or a step's ``tool_calls`` is not a list of objects with ``tool_call_id``, ``function_name``
    and object ``arguments``; or its ``observation`` is not empty and not an object, or ``observation.results`` is not
    a list of objects; or a message or result content list holds a part that is not an object, a text part whose
    ``text`` is not a string, or an image part without ``source.path``.
    """
    steps = trajectory.get("steps")
    if not isinstance(steps, list) or any(
        not isinstance(step, Mapping) or not isinstance(step.get("source"), str) for step in steps
    ):
        raise ValueError("ATIF steps must be a list of step objects, each with a string source")
    for step in steps:
        calls = step.get("tool_calls") or []
        if not isinstance(calls, list) or any(
            not isinstance(call, Mapping)
            or not isinstance(call.get("tool_call_id"), str)
            or not isinstance(call.get("function_name"), str)
            or not isinstance(call.get("arguments"), Mapping)
            for call in calls
        ):
            raise ValueError("ATIF tool_calls must be objects with tool_call_id, function_name and object arguments")
        observation = step.get("observation") or {}
        if not isinstance(observation, Mapping):
            raise ValueError("ATIF observation must be an object")
        results = observation.get("results", [])
        if not isinstance(results, list) or any(not isinstance(result, Mapping) for result in results):
            raise ValueError("ATIF observation results must be a list of objects")
    messages: list[dict[str, Any]] = []
    for step in steps:
        source = step["source"]
        message: dict[str, Any] = {
            "role": "assistant" if source == "agent" else source,
            "content": _provider_content(step.get("message", "")),
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
    return messages


def _project_exchange(trajectory: Mapping[str, Any]) -> Mapping[str, Any]:
    messages = atif_messages(trajectory)
    final = messages[-1] if messages and messages[-1]["role"] == "assistant" else None
    return {
        "messages": messages[:-1] if final is not None else messages,
        "response": {"choices": [{"message": final}]} if final is not None else {},
    }
