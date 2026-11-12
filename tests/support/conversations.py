"""Inferences of one agent conversation, whose requests resend the earlier turns (RFC #1054)."""

from __future__ import annotations

from dataclasses import replace

from reef.core.records_types import AgentRecord, RequestType


def turn(index: int, filler: str = "") -> dict[str, str]:
    """Turn ``index`` of the conversation: users have the even turns and the assistant the odd ones."""
    return {"role": ("user", "assistant")[index % 2], "content": f"turn {index} 海{filler}"}


def conversation(
    record_id: str,
    turns: int,
    *,
    scenario: str = "math",
    field: str = "messages",
    created_at: float = 123.0,
    filler: str = "",
    tools: tuple[str, ...] = ("read_file", "bash"),
) -> AgentRecord:
    """The inference whose request holds a system message and turns ``0 .. turns - 1`` in ``field``, and whose reply
    is turn ``turns``.

    Turn ``t`` has the same content in each call, so a call continues the call with one turn less. ``filler`` is
    appended to each message, to make the messages larger. ``tools`` names the function tools of the request.
    """
    messages = [{"role": "system", "content": "be brief" + filler}, *(turn(index, filler) for index in range(turns))]
    return AgentRecord.create(
        agent_record_id=record_id,
        scenario=scenario,
        request_type=RequestType.INFERENCE,
        payload={
            "model": "m",
            field: messages,
            # A response request names its function tools without the ``function`` wrapper.
            "tools": [
                (
                    {"type": "function", "name": name, "parameters": {}}
                    if field == "input"
                    else {"type": "function", "function": {"name": name, "parameters": {}}}
                )
                for name in tools
            ],
            "temperature": 0.5,
            "response": {"choices": [{"message": {**turn(turns, filler), "refusal": None}}]},
        },
        created_at=created_at,
    )


def routed_call(record: AgentRecord, purpose: str, request_id: str, reply: int | None = None) -> AgentRecord:
    """``record`` as a routed call stores it: with routing metadata of ``purpose`` (``serve`` or ``shadow``) and
    ``request_id``, and with reply turn ``reply`` instead of its own when given. A shadow call names the primary
    request ``primary-request``."""
    routing = {"purpose": purpose, "request_id": request_id}
    if purpose == "shadow":
        routing["primary_request_id"] = "primary-request"
    payload = {**record.payload, "metadata": {"caller": "kept", "routing": routing}}
    if reply is not None:
        payload["response"] = {"choices": [{"message": {**turn(reply), "refusal": None}}]}
    return replace(record, payload=payload)
