"""Session reconstruction from harness tags or trace matching.

A stable, conversation-unique ``x-reef-tag-session`` value identifies a
conversation directly and binds its turns in arrival order. Without one, a
session is inferred from how each main-turn request's message list relates to
the ones seen before it. :class:`SessionIndex` is the whole mechanism; a
:class:`Binding` is its one product — "a previous turn just received its next
state".

:class:`Stats` is how that mechanism reports on itself. Correlation fails
silently by construction — an unbound turn is simply never judged — so the
counters are the only way the outside can tell "no session ever bound" from
"nothing has arrived yet".
"""

from __future__ import annotations

import heapq
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from recipes.openclawrl.prm import flatten_content

#: Ceiling on live sessions. The fallback opens one session per unmatched
#: request, so without a ceiling a client it never matches grows the table
#: with the request rate for a whole TTL, and every scan pays for it.
DEFAULT_MAX_SESSIONS = 4096


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


@dataclass(frozen=True)
class Stats:
    """Cumulative correlation counters, for diagnosing a run from outside.

    ``dropped_unbound`` is the one that matters: a session that dies without
    ever binding a turn produced nothing trainable. A few are normal (a
    conversation whose last turn never got a next state); many, against few
    ``bound``, means correlation is not working — the shape a harness that
    stamps no tag and keeps its history locally produces on every turn.
    """

    tagged: int = 0
    untagged: int = 0
    bound: int = 0
    dropped_unbound: int = 0
    evicted: int = 0


@dataclass
class _Session:
    last_receipt: str
    last_request_key: tuple[tuple[Any, ...], ...]
    last_activity: float
    bound: bool = False  # ever handed a turn its next state


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

    Everything here runs on the trainer thread, which must not block
    (``reef/train/processors/base.py``), once per record: the untagged path
    walks the live sessions and the tagged path is a single lookup, and
    ``max_sessions`` is what bounds the first of those.
    """

    def __init__(self, ttl_s: float, *, max_sessions: int = DEFAULT_MAX_SESSIONS) -> None:
        if ttl_s <= 0:
            raise ValueError("session_ttl_s must be positive")
        if max_sessions <= 0:
            raise ValueError("max_sessions must be positive")
        self._ttl_s = ttl_s
        self._max_sessions = max_sessions
        self._sessions: dict[str, _Session] = {}
        self._counts = {"tagged": 0, "untagged": 0, "bound": 0, "dropped_unbound": 0, "evicted": 0}

    @property
    def stats(self) -> Stats:
        """Correlation counters since this index was built."""
        return Stats(**self._counts)

    def observe(
        self,
        receipt: str,
        messages: Sequence[Any],
        now: float,
        session_tag: str | None = None,
    ) -> Observation:
        key = request_key(messages)
        if session_tag:
            self._counts["tagged"] += 1
            return self._observe_tagged(session_tag, receipt, key, messages, now)
        self._counts["untagged"] += 1
        # One pass, no sort. This runs per record on a thread that must not
        # block, and ordering the whole table just to read its most recent
        # entry cost more than the comparisons it ordered. The outcome is
        # unchanged: equality returns, so a retry still outranks every prefix
        # match, and among prefix matches the most recently active session
        # still wins — strict ``>`` keeps the first of equally active
        # candidates, exactly as the stable sort did.
        matched: _Session | None = None
        for session in self._sessions.values():
            if key == session.last_request_key:
                return Observation(duplicate=True)
            prefix = len(session.last_request_key)
            # Cheap tests first: the O(T) prefix compare only runs for a
            # candidate long enough to be a successor and recent enough to win.
            if (
                len(key) > prefix + 1
                and (matched is None or session.last_activity > matched.last_activity)
                and key[:prefix] == session.last_request_key
                and key[prefix][0] == "assistant"
            ):
                matched = session
        if matched is None:
            self._open(receipt, receipt, key, now)
            return Observation()
        return self._advance(matched, receipt, key, messages, now)

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
            self._open(session_tag, receipt, key, now)
            return Observation()
        if key == session.last_request_key:
            return Observation(duplicate=True)
        return self._advance(session, receipt, key, messages, now)

    def _open(self, slot: str, receipt: str, key: tuple[tuple[Any, ...], ...], now: float) -> None:
        """Start a session under ``slot`` — its tag, or its own receipt."""
        self._sessions[slot] = _Session(last_receipt=receipt, last_request_key=key, last_activity=now)

    def _advance(
        self,
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
        session.bound = True
        self._counts["bound"] += 1
        return Observation(
            binding=Binding(
                receipt=previous,
                next_state_text=flatten_content(trailing.get("content")),
                next_state_role=str(trailing.get("role", "user")),
            )
        )

    def expire(self, now: float) -> tuple[str, ...]:
        """Flush idle and overflowing sessions; return their final (never-bound) receipts.

        Overflow evicts the least recently active sessions. Their pending
        turns retire the same way an idle session's does — the ceiling costs
        the tail of a busy table, and buys a bounded scan on the trainer
        thread instead of one that grows with the request rate.
        """
        dead = [slot for slot, session in self._sessions.items() if now - session.last_activity > self._ttl_s]
        overflow = len(self._sessions) - len(dead) - self._max_sessions
        if overflow > 0:
            expiring = set(dead)
            dead += heapq.nsmallest(
                overflow,
                (slot for slot in self._sessions if slot not in expiring),
                key=lambda slot: self._sessions[slot].last_activity,
            )
            self._counts["evicted"] += overflow
        receipts = []
        for slot in dead:
            session = self._sessions.pop(slot)
            # A session that never bound anything produced nothing trainable.
            # Counted apart from the ordinary final turn because it is the
            # signature of correlation not working at all.
            if not session.bound:
                self._counts["dropped_unbound"] += 1
            receipts.append(session.last_receipt)
        return tuple(receipts)
