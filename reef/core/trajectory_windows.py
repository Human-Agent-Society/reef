"""Verified request windows for supervised training of captured conversations.

The stored trajectory stays unchanged. SFT consumers can split its captured
calls where the next request no longer extends the preceding conversation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from reef.core.atif_payloads import (
    ANTHROPIC,
    OPENAI_RESPONSES,
    atif_agent,
    call_shape,
    canonical_json,
    folded_steps,
    message_entry,
    reply_messages,
    response_metrics,
)
from reef.core.job_data import TrajectorySample
from reef.core.trajectories import CONVERSATION_FIELDS, exchange_messages
from reef.core.trajectory_index import atif_form


def check_captured_messages(messages: Sequence[Mapping[str, object]], provider: str) -> None:
    """Refuse message content that the text ATIF projection cannot represent.

    Tool arguments stay structured arguments. Only conversation content is
    traversed, including nested Anthropic tool results.
    """
    contents: list[object] = []
    for message in messages:
        kind = message.get("type")
        if provider == OPENAI_RESPONSES and kind in ("function_call", "function_call_output", "reasoning"):
            if kind == "function_call_output":
                contents.append(message.get("output"))
            elif kind == "reasoning":
                if message.get("encrypted_content") or message.get("content"):
                    raise ValueError("captured context windows cannot represent opaque reasoning")
                summary = message.get("summary", [])
                if not isinstance(summary, list) or any(
                    not isinstance(part, Mapping)
                    or part.get("type") != "summary_text"
                    or not isinstance(part.get("text"), str)
                    for part in summary
                ):
                    raise ValueError("captured context windows need text reasoning summaries")
            continue
        if message.get("role") not in ("system", "developer", "user", "assistant", "tool") or (
            provider == OPENAI_RESPONSES and kind not in (None, "message")
        ):
            raise ValueError("captured context windows contain an unsupported provider message")
        contents.append(message.get("content"))
        calls = message.get("tool_calls")
        if calls is None:
            calls = []
        if not isinstance(calls, list) or any(
            not isinstance(call, Mapping) or call.get("type") != "function" for call in calls
        ):
            raise ValueError("captured context windows require function tool calls")
    while contents:
        content = contents.pop()
        if content is None or isinstance(content, str):
            continue
        if not isinstance(content, list):
            raise ValueError("captured context windows require text content or supported content blocks")
        for block in content:
            if not isinstance(block, Mapping):
                raise ValueError("captured context windows require content block objects")
            kind = block.get("type")
            if kind in ("text", "input_text", "output_text") and isinstance(block.get("text"), str):
                continue
            if provider == ANTHROPIC:
                if kind == "tool_result":
                    contents.append(block.get("content"))
                    continue
                if kind == "tool_use" and isinstance(block.get("input"), Mapping):
                    continue
                if kind == "thinking" and isinstance(block.get("thinking"), str):
                    continue
            raise ValueError(f"captured context windows cannot represent content block {kind!r}")


def supervised_steps(steps: object) -> list[dict[str, object]]:
    """The fields that determine the conversation and its supervised turns.

    Raise ``ValueError`` unless ``steps`` is a list of ATIF step objects with
    boolean copied-context flags. Display metadata and step numbers do not
    affect this comparison.
    """
    if not isinstance(steps, list) or any(not isinstance(step, Mapping) for step in steps):
        raise ValueError("captured context windows require ATIF step objects")
    projected: list[dict[str, object]] = []
    for step in steps:
        copied = step.get("is_copied_context", False)
        if not isinstance(copied, bool):
            raise ValueError("ATIF is_copied_context must be a boolean")
        projected.append(
            {
                "source": step.get("source"),
                "message": step.get("message", ""),
                "reasoning_content": step.get("reasoning_content", ""),
                "tool_calls": step.get("tool_calls", []),
                "observation": step.get("observation", {}),
                "is_copied_context": copied,
            }
        )
    return projected


def captured_context_windows(sample: TrajectorySample) -> tuple[TrajectorySample, ...]:
    """Split captured calls into ordered, independently supervised contexts.

    A sample without multiple complete captured records, with only call
    summaries, or with training tokens returns unchanged. Mixed summaries
    and complete calls are refused. Otherwise its steps and copied-context flags must
    match its captured calls. Each window begins with one complete request;
    its request's agent turns are copied context, and its captured replies
    train exactly once. A following call joins only when provider, tools,
    model and template arguments agree, its request extends the complete
    preceding request and reply, and its added context has no agent turn.

    Each result preserves the input's metadata, grouping and source IDs.
    Its records and tools describe that window, and its trajectory and primary
    source IDs name the window's last call. No input value changes.
    ``ValueError`` means a captured call is incomplete, depends on
    server-held context, or disagrees with the trajectory's supervised steps.
    """
    records = sample.metadata.get("records")
    if sample.training.get("tokens") or records is None:
        return (sample,)
    if not isinstance(records, list):
        raise ValueError("captured context records must be a list")
    if len(records) < 2:
        return (sample,)
    full_payloads = []
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(record.get("payload", {}), Mapping):
            raise ValueError("captured records must be objects with payload objects")
        full_payloads.append(any(name in record.get("payload", {}) for name in CONVERSATION_FIELDS))
    if not any(full_payloads):
        return (sample,)
    if not all(full_payloads):
        raise ValueError("captured context windows cannot mix complete calls with call summaries")

    all_entries: list[list[dict[str, object]]] = []
    windows: list[list[list[dict[str, object]]]] = []
    window_records: list[list[Mapping[str, object]]] = []
    window_payloads: list[Mapping[str, object]] = []
    providers: list[str] = []
    history: Sequence[Mapping[str, object]] = ()
    raw_history: Sequence[object] = ()
    previous_settings: str | None = None
    record_ids: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(record.get("payload"), Mapping):
            raise ValueError("every captured record must contain a payload object")
        record_id = record.get("agent_record_id")
        if not isinstance(record_id, str) or not record_id or record_id in record_ids:
            raise ValueError("captured records must have distinct non-empty record IDs")
        record_ids.add(record_id)
        payload = record["payload"]
        if any(payload.get(key) not in (None, "") for key in ("previous_response_id", "conversation")):
            raise ValueError("captured context windows cannot recover server-held conversation state")
        shape = call_shape(payload)
        if shape is None:
            raise ValueError("captured context windows need a known provider")
        provider = shape.provider
        message_field = shape.message_field
        if any(not isinstance(message, Mapping) for message in payload[message_field]):
            raise ValueError("captured context windows require request message objects")
        if payload.get("system", payload.get("instructions")) and any(
            message.get("role") == "system" for message in payload[message_field]
        ):
            raise ValueError("captured context windows cannot combine top-level and message system instructions")
        replies = reply_messages(payload.get("response"))
        if not payload[message_field] or not replies or any(not isinstance(reply, Mapping) for reply in replies):
            raise ValueError("captured context windows need a complete request and response for every call")
        training = payload["response"].get("training", {})
        if isinstance(training, Mapping) and any(
            name in training for name in ("request_messages", "response_message")
        ):
            raise ValueError("captured context windows cannot use training overrides of provider messages")
        tools = payload.get("tools", [])
        if tools is None:
            tools = []
        if not isinstance(tools, list) or (atif_agent(tools, provider).get("extra") or {}).get("reef", {}).get(
            "other_tools"
        ):
            raise ValueError("captured context windows require function tool definitions")
        # A client may omit null response fields when replaying history. All
        # other provider content remains part of the prefix comparison.
        raw_request = [
            {name: value for name, value in message.items() if value is not None} for message in payload[message_field]
        ]
        if shape.system_field is not None:
            raw_request.insert(0, payload[shape.system_field])
        try:
            settings = canonical_json(
                [shape.as_json(), payload.get("tools"), payload.get("model"), payload.get("chat_template_kwargs")]
            )
            canonical_json([raw_request, replies])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"captured context windows require JSON calls: {exc}") from exc
        request, response = exchange_messages(payload)
        check_captured_messages([*request, *response], provider)
        # Match the existing stored projection before changing any context.
        shared = 0
        for previous, current in zip(history, request, strict=False):
            if canonical_json(atif_form(previous, provider)) != canonical_json(atif_form(current, provider)):
                break
            shared += 1
        all_entries.extend(message_entry(message, provider, copied=True) for message in request[shared:])
        response_entries = [message_entry(message, provider, copied=False) for message in response]
        metrics = response_metrics(payload["response"])
        if metrics and response_entries and response_entries[-1] and response_entries[-1][-1].get("source") == "agent":
            response_entries[-1][-1] = {**response_entries[-1][-1], "metrics": metrics}
        all_entries.extend(response_entries)

        following = request[len(history) :]
        continuation = (
            previous_settings == settings
            and len(raw_request) >= len(raw_history)
            and canonical_json(raw_request[: len(raw_history)]) == canonical_json(raw_history)
            and not any(
                element.get("source") == "agent"
                for message in following
                for element in message_entry(message, provider, copied=True)
            )
        )
        if continuation:
            windows[-1].extend(message_entry(message, provider, copied=True) for message in request[len(history) :])
            window_records[-1].append(record)
            window_payloads[-1] = payload
        else:
            windows.append([message_entry(message, provider, copied=True) for message in request])
            window_records.append([record])
            window_payloads.append(payload)
            providers.append(provider)
        windows[-1].extend(response_entries)
        history = [*request, *response]
        raw_history = [
            *raw_request,
            *({name: value for name, value in reply.items() if value is not None} for reply in replies),
        ]
        previous_settings = settings

    if canonical_json(supervised_steps(sample.trajectory["steps"])) != canonical_json(
        supervised_steps(folded_steps(all_entries))
    ):
        raise ValueError("captured calls do not match the trajectory's steps and supervised targets")
    recorded_tools = atif_agent(window_payloads[-1].get("tools", []), providers[-1]).get("tool_definitions", [])
    if canonical_json(sample.trajectory["agent"].get("tool_definitions", [])) != canonical_json(recorded_tools):
        raise ValueError("captured calls do not match the trajectory's tool definitions")
    if len(windows) == 1:
        return (sample,)

    result: list[TrajectorySample] = []
    for entries, captured, payload, provider in zip(windows, window_records, window_payloads, providers, strict=True):
        agent = dict(sample.trajectory["agent"])
        fragment = atif_agent(payload["tools"], provider) if "tools" in payload else {}
        agent.pop("tool_definitions", None)
        if "tool_definitions" in fragment:
            agent["tool_definitions"] = fragment["tool_definitions"]
        agent_extra = dict(agent.get("extra") or {})
        reef_extra = dict(agent_extra.get("reef") or {})
        reef_extra.pop("other_tools", None)
        reef_extra.pop("patch", None)
        reef_extra.update((fragment.get("extra") or {}).get("reef") or {})
        if reef_extra:
            agent_extra["reef"] = reef_extra
        else:
            agent_extra.pop("reef", None)
        if agent_extra:
            agent["extra"] = agent_extra
        else:
            agent.pop("extra", None)
        if isinstance(payload.get("model"), str):
            agent["model_name"] = payload["model"]
        else:
            agent.pop("model_name", None)
        window = TrajectorySample(
            {
                **sample.trajectory,
                "trajectory_id": captured[-1]["agent_record_id"],
                "agent": agent,
                "steps": folded_steps(entries),
            },
            group_id=sample.group_id,
            source_agent_record_ids=sample.source_agent_record_ids,
        )
        result.append(window.with_metadata(records=captured, source_agent_record_id=captured[-1]["agent_record_id"]))
    return tuple(result)
