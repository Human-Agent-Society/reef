"""ATIF construction and projections shared by record processors and consumers.

No I/O or model calls live here. The standard steps hold the conversation. ``extra.reef.records`` holds a summary of
each captured call (``call_summary``): its ID, its time, its artifact and its settings, without the conversation.
A training component renders the steps into the tokens that it trains (``reef.core.chat_rendering``).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from reef.core.atif_payloads import (
    OPENAI_CHAT,
    atif_agent,
    canonical_json,
    folded_steps,
    message_entry,
    payload_provider,
    response_metrics,
)
from reef.core.job_data import TrajectorySample
from reef.core.records_types import AgentRecord
from reef.core.trajectory_index import atif_form


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


#: The fields of a provider payload that hold a call's conversation, its tools or its answer. The steps and
#: ``agent.tools`` of a trajectory hold them, so a call's summary (``call_summary``) leaves them out.
CONVERSATION_FIELDS = frozenset({"messages", "input", "prompt", "system", "instructions", "tools", "response"})


def call_summary(record: AgentRecord) -> dict[str, Any]:
    """What a trajectory keeps of ``record`` in ``extra.reef.records``: ``{agent_record_id, created_at, artifact_id,
    payload}``. ``payload`` is the record's provider payload without ``CONVERSATION_FIELDS``: the call's settings, such
    as ``model``, ``chat_template_kwargs`` and ``metadata``. Each request repeats the conversation so far, so a
    trajectory that kept whole payloads would grow with the square of its calls."""
    return {
        "agent_record_id": record.agent_record_id,
        "created_at": record.created_at,
        "artifact_id": record.artifact_id,
        "payload": {key: value for key, value in record.payload.items() if key not in CONVERSATION_FIELDS},
    }


def recorded_payloads(item: TrajectorySample) -> tuple[Mapping[str, Any], ...]:
    """The stored payload of each call that ``item`` recorded, oldest first: the call's settings (``call_summary``),
    the whole provider payload in a trajectory that kept it, or ``{}`` for a summary without a payload. ATIF without
    records gives one exchange, which its steps project. ``ValueError`` when ``extra.reef.records`` is not a non-empty
    list of objects."""
    records = item.metadata.get("records")
    if records is None:
        return (_project_exchange(item.trajectory),)
    if not isinstance(records, list) or not records or not all(isinstance(record, Mapping) for record in records):
        raise ValueError("ATIF extra.reef.records must be a non-empty list of objects")
    payloads = []
    for record in records:
        payload = record.get("payload", {})
        if not isinstance(payload, Mapping):
            raise ValueError("the payload of an ATIF captured record must be an object")
        payloads.append(payload)
    return tuple(payloads)


def recorded_payload(item: TrajectorySample) -> Mapping[str, Any]:
    """The last call as served: its stored payload (``recorded_payloads``) when it holds the conversation, else that
    payload over the exchange that the steps project, so the conversation comes from the steps and the settings from
    the call."""
    last = recorded_payloads(item)[-1]
    if any(key in last for key in ("messages", "input", "prompt")):
        return last
    return {**_project_exchange(item.trajectory), **last}


def _string_fields(value: Mapping[str, Any], *, nested: str | None = None) -> bool:
    """Whether a plain dictionary has only string keys and string or null fields, except for one named field."""
    return type(value) is dict and all(
        type(key) is str and (key == nested or type(field) is str or field is None) for key, field in value.items()
    )


def _simple_history_message(message: Mapping[str, Any]) -> bool:
    """Recognize flat text messages and Chat Completions tool calls with JSON arguments in a string."""
    if not _string_fields(message, nested="tool_calls"):
        return False
    calls = message.get("tool_calls")
    if calls is None:
        return True
    return type(calls) is list and all(
        _string_fields(call, nested="function") and _string_fields(call.get("function")) for call in calls
    )


def _same_history_message(previous: Mapping[str, Any], current: Mapping[str, Any], provider: str) -> bool:
    # Python treats several numeric types as equal. Only use direct equality for the string/null wire shape.
    if _simple_history_message(previous) and _simple_history_message(current) and previous == current:
        return True
    return canonical_json(atif_form(previous, provider)) == canonical_json(atif_form(current, provider))


@dataclass
class TrajectoryBuilder:
    """An ATIF trajectory that grows by one inference record at a time, oldest first.

    Values: ``include_records`` says whether the trajectory keeps the summary of each record (``call_summary``) in
    ``extra.reef.records``. ``entries`` holds the message entries so far: each new request message once
    (``copied``) and each response (not copied). ``history`` is the request and the responses of the last record
    added, the only messages that a new record is compared with. ``kept`` holds the summaries of the records when
    ``include_records`` is true and is empty otherwise. ``record_ids`` holds the ID of each record added, in order;
    ``first`` and ``last`` are the first and the last record added, both None exactly when none was added.

    Purpose: ``make_trajectory`` adds a list of records, and a reader of a stored rollout adds the records of each
    of its conversations as it reads them, so that it holds one record's body at a time (``add``). ``sample`` gives
    the trajectory.
    """

    include_records: bool = True
    entries: list[list[dict[str, Any]]] = field(default_factory=list)
    history: list[Mapping[str, Any]] = field(default_factory=list)
    kept: list[dict[str, Any]] = field(default_factory=list)
    record_ids: list[str] = field(default_factory=list)
    first: AgentRecord | None = None
    last: AgentRecord | None = None

    def add(self, record: AgentRecord) -> None:
        """Add the messages of ``record`` that the last record did not have, and its responses. The last agent step
        of the responses carries the ATIF ``metrics`` of the record's response (``response_metrics``): its token
        counts, and its token IDs and log probabilities when the response has them."""
        provider = payload_provider(record.payload) or OPENAI_CHAT
        request, responses = exchange_messages(record.payload)
        shared = 0
        for previous, current in zip(self.history, request, strict=False):
            if not _same_history_message(previous, current, provider):
                break
            shared += 1
        self.entries.extend(message_entry(message, provider, copied=True) for message in request[shared:])
        reply = [message_entry(message, provider, copied=False) for message in responses]
        response = record.payload.get("response")
        metrics = response_metrics(response) if isinstance(response, Mapping) else None
        if metrics and reply and reply[-1] and reply[-1][-1].get("source") == "agent":
            reply[-1][-1] = {**reply[-1][-1], "metrics": metrics}
        self.entries.extend(reply)
        self.history = [*request, *responses]
        self.record_ids.append(record.agent_record_id)
        if self.include_records:
            self.kept.append(call_summary(record))
        if self.first is None:
            self.first = record
        self.last = record

    def sample(self, reward: float | None = None, feedback: str | Mapping[str, Any] | None = None) -> TrajectorySample:
        """The ATIF trajectory of the records added, with ``reward`` and ``feedback``. ``ValueError`` when no record
        was added."""
        first, last = self.first, self.last
        if first is None or last is None:
            raise ValueError("an ATIF trajectory requires at least one inference record")
        steps = folded_steps(self.entries)
        if not steps:
            steps.append(
                {"step_id": 1, "source": "agent", "message": "", "extra": {"reef": {"text_available": False}}}
            )
        primary = last.agent_record_id
        agent: dict[str, Any] = {"name": "recorded-agent", "version": "unknown"}
        model = last.payload.get("model")
        if isinstance(model, str):
            agent["model_name"] = model
        if "tools" in last.payload:
            agent.update(atif_agent(last.payload["tools"], payload_provider(last.payload) or OPENAI_CHAT))
        reef: dict[str, Any] = {"source_agent_record_id": primary, "reward": reward, "feedback": feedback}
        if self.include_records:
            reef["records"] = list(self.kept)
        return TrajectorySample(
            {
                "schema_version": "ATIF-v1.7",
                "session_id": first.agent_record_id,
                "trajectory_id": primary,
                "agent": agent,
                "steps": steps,
                "extra": {"reef": reef},
            },
            source_agent_record_ids=tuple(self.record_ids),
        )


def make_trajectory(
    records: Iterable[AgentRecord],
    reward: float | None = None,
    feedback: str | Mapping[str, Any] | None = None,
    *,
    include_records: bool = True,
) -> TrajectorySample:
    """Encode ordered inference records as ATIF (``TrajectoryBuilder``), with the summary of each record in
    ``extra.reef.records`` (``call_summary``).

    It reads ``records`` once, in order, and keeps only the messages of the previous record, so an iterator that
    reads each record when asked holds one record at a time. With ``include_records`` false, ``extra.reef`` has no
    ``records``. Either way the trajectory holds no provider body, so its size is about that of its steps."""
    builder = TrajectoryBuilder(include_records)
    for record in records:
        builder.add(record)
    return builder.sample(reward, feedback)


def has_image_content(payload: Mapping[str, object]) -> bool:
    """Whether provider conversation fields or ATIF steps contain an image part.

    Only conversation fields are searched; metadata, tool definitions and tool arguments are ignored.
    """
    pending = [
        payload.get(key) for key in ("messages", "input", "prompt", "system", "instructions", "response", "steps")
    ]
    while pending:
        value = pending.pop()
        if isinstance(value, list):
            pending.extend(value)
        elif isinstance(value, Mapping):
            if value.get("type") in (
                "image",
                "image_url",
                "input_image",
                "computer_screenshot",
                "image_generation_call",
            ):
                return True
            pending.extend(
                value.get(key)
                for key in ("content", "output", "source", "message", "choices", "observation", "results")
            )
    return False


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
