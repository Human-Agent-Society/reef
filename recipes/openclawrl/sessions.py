"""Session reconstruction from harness tags or trace matching.

A stable, conversation-unique ``x-reef-tag-session`` value identifies a
conversation directly and binds its turns in arrival order. Without one, a
session is inferred from how each main-turn request's message list relates to
the ones seen before it. :class:`SessionIndex` is the whole mechanism; a
:class:`Binding` is its one product — "a previous turn just received its next
state".
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from recipes.openclawrl.prm import flatten_content


def message_key(message: Mapping[str, Any]) -> tuple[Any, ...]:
    """One message's matching identity: role, text, (name, arguments) calls.

    Deliberately ignores tool-call ids and content formatting a client may
    rewrite between turns; non-text content parts are invisible (matching is
    text-only).
    """
    calls = tuple(
        (
            str((call.get("function") or {}).get("name", "")),
            str((call.get("function") or {}).get("arguments", "")),
        )
        for call in message.get("tool_calls") or []
        if isinstance(call, Mapping)
    )
    return (str(message.get("role", "")), flatten_content(message.get("content")), calls)


def request_key(messages: Sequence[Any]) -> tuple[tuple[Any, ...], ...]:
    return tuple(message_key(message) for message in messages if isinstance(message, Mapping))


@dataclass(frozen=True)
class Binding:
    """A previous turn just received its next state."""

    receipt: str
    next_state_text: str
    next_state_role: str


@dataclass(frozen=True)
class Observation:
    """What one observed main turn did to the session index."""

    duplicate: bool = False
    binding: Binding | None = None


@dataclass
class _Session:
    last_receipt: str
    last_request_key: tuple[tuple[Any, ...], ...]
    last_activity: float


def _advance(
    session: _Session,
    receipt: str,
    key: tuple[tuple[Any, ...], ...],
    messages: Sequence[Any],
    now: float,
) -> Observation:
    """Move the session onto its newest turn and bind the displaced one to
    whatever arrived next.

    Every main turn is judged, whatever the next state's role. A tool result
    is as much a next state as a user reply: upstream fires its judge on
    ``messages[-1]`` unconditionally, and the role is an INPUT to the judge
    rather than a filter — both prompts branch on it, and a tool error is a
    ``\boxed{-1}``. Under the dynamic-history paradigm one LLM call is one
    training datum, so a turn that makes N model calls yields N samples;
    filtering to user replies here discards the tool-mechanics half of the
    method and, measured on real traffic, ~50 out of every 51 judgments.
    """
    trailing = messages[-1] if messages and isinstance(messages[-1], Mapping) else {}
    previous = session.last_receipt
    session.last_receipt = receipt
    session.last_request_key = key
    session.last_activity = now
    return Observation(
        binding=Binding(
            receipt=previous,
            next_state_text=flatten_content(trailing.get("content")),
            next_state_role=str(trailing.get("role", "user")),
        )
    )


class SessionIndex:
    """Reconstruct sessions from main-turn requests, in append order.

    A request whose canonical message list strictly extends an open
    session's last request — the served reply in the next slot (verified by
    role), then at least one further message — is that session's next turn,
    and its trailing message is the previous turn's next state. A request
    *equal* to a session's last request is a client retry: the first receipt
    keeps the slot. Only the newest turn of a session is ever unbound, so
    the index holds one (receipt, request key) pair per session; an idle
    session's final turn has no next state and never trains — ``expire``
    drops it after the TTL.

    A harness that stamps ``x-reef-tag-session`` skips all of that: the tag
    names the conversation outright, so turns bind in arrival order within it
    and nothing depends on the client resending its transcript. That matters
    for agents that keep history locally — Hermes restarts each turn from
    ``[system, user]``, so no cross-turn request ever extends the previous
    one and the header-free path binds only within a turn, never across the
    user reply that carries the method's whole signal.

    Header-free matching has documented limits: ties between canonically
    identical transcripts go to the most recently active session; a client
    that rewrites its own history (compaction, sliding windows) breaks the
    chain and the pre-rewrite turn flushes terminal; an extension carrying
    only the assistant echo (continue-style prefill) binds nothing — a turn
    must never be judged against its own reply.
    """

    def __init__(self, ttl_s: float) -> None:
        if ttl_s <= 0:
            raise ValueError("session_ttl_s must be positive")
        self._ttl_s = ttl_s
        self._sessions: dict[str, _Session] = {}

    def observe(
        self,
        receipt: str,
        messages: Sequence[Any],
        now: float,
        session_tag: str | None = None,
    ) -> Observation:
        key = request_key(messages)
        if session_tag:
            return self._observe_tagged(session_tag, receipt, key, messages, now)
        matched: _Session | None = None
        for session in sorted(self._sessions.values(), key=lambda s: s.last_activity, reverse=True):
            if key == session.last_request_key:
                return Observation(duplicate=True)
            prefix = len(session.last_request_key)
            if (
                matched is None
                and len(key) > prefix + 1
                and key[:prefix] == session.last_request_key
                and key[prefix][0] == "assistant"
            ):
                matched = session  # keep scanning: retry equality outranks a prefix match
        if matched is None:
            self._sessions[receipt] = _Session(last_receipt=receipt, last_request_key=key, last_activity=now)
            return Observation()
        return _advance(matched, receipt, key, messages, now)

    def _observe_tagged(
        self,
        session_tag: str,
        receipt: str,
        key: tuple[tuple[Any, ...], ...],
        messages: Sequence[Any],
        now: float,
    ) -> Observation:
        """Bind in arrival order inside a harness-declared conversation.

        The tag is the session, so a turn needs no textual relationship to
        the one before it — only to have arrived after it. Retries still
        collapse on the request key, and the newest turn stays unbound until
        the next one names its next state.
        """
        session = self._sessions.get(session_tag)
        if session is None:
            self._sessions[session_tag] = _Session(last_receipt=receipt, last_request_key=key, last_activity=now)
            return Observation()
        if key == session.last_request_key:
            return Observation(duplicate=True)
        return _advance(session, receipt, key, messages, now)

    def expire(self, now: float) -> tuple[str, ...]:
        """Flush idle sessions; return their final (never-bound) receipts."""
        expired = [k for k, session in self._sessions.items() if now - session.last_activity > self._ttl_s]
        return tuple(self._sessions.pop(k).last_receipt for k in expired)
