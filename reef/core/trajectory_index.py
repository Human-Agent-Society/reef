"""The rule that links the model calls of one scenario into trajectories (RFC #1054).

An agent sends its whole conversation again in each model call, with the same tools. The request of a call therefore
starts with the request of the calls before it, followed by their replies. Each call links to the earlier call of the
same trajectory whose request is the longest prefix of its own; the links form a tree, and the path from the root to
a call is the trajectory that the call ends. A record store keeps this tree in SQL; ``TrajectoryIndex`` keeps it in
memory for a record-to-sample stage. Both use ``call_key`` and ``best_parent``, so they link the same calls.

Let ``P`` be a payload with a message list (``payload_provider``), and ``canonical(v)`` the JSON of ``v`` with sorted
keys and compact separators. The request values of ``P`` are its ``system`` or ``instructions`` value, when it has
one, and then each message of its message list. ``trajectory_sha256(P)`` is the SHA-256 of ``"trajectory:" +
canonical(shape)``, where the shape holds the provider, the field names and ``P["tools"]``: two calls have one value
if and only if they have the same API format and the same tools. ``prefix_sha256(P)[i]`` chains the first ``i + 1``
request values onto it, so two calls of one trajectory have a common prefix of ``i + 1`` request values if and only
if their ``i``-th prefix hashes are equal.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from reef.core.atif_payloads import OPENAI_CHAT, OPENAI_RESPONSES, canonical_json, payload_provider, reply_messages
from reef.core.records_types import AgentRecord, RequestType


def sha256_text(text: str) -> str:
    """The hexadecimal SHA-256 of the UTF-8 bytes of ``text``."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def comparable_message(message: Any) -> Any:
    """A message as a later request resends it: a client drops the fields that a server sets to null."""
    if isinstance(message, Mapping):
        return {key: value for key, value in message.items() if value is not None}
    return message


def reply_sha256(messages: Sequence[Any]) -> str:
    """The hash that compares reply messages with the messages of a later request (``comparable_message``)."""
    return sha256_text(canonical_json([comparable_message(message) for message in messages]))


@dataclass(frozen=True)
class CallKey:
    """The hashes that place one call ``P`` in its trajectory, as the module docstring defines them.

    Values: ``request`` holds the request values of ``P``, and ``prefix_sha256`` one chained hash per value, the
    first chained onto ``trajectory_sha256``. ``reply_count`` and ``reply_sha256`` describe the reply messages of
    ``P`` (``reply_messages``): their number and ``reply_sha256``, or ``0`` and ``None`` without a reply. The
    constructor checks the hashes and the lengths.

    Purpose: ``call_key`` makes it; a record store and ``TrajectoryIndex`` find a call's parent with it
    (``best_parent``).
    """

    trajectory_sha256: str
    request: tuple[Any, ...]
    prefix_sha256: tuple[str, ...]
    reply_count: int
    reply_sha256: str | None

    def __post_init__(self) -> None:
        if len(self.request) != len(self.prefix_sha256):
            raise ValueError("each request value has one prefix hash")
        if (self.reply_sha256 is None) != (self.reply_count == 0):
            raise ValueError("a reply has a hash and at least one message, or neither")
        previous = self.trajectory_sha256
        for value, prefix in zip(self.request, self.prefix_sha256, strict=True):
            previous = sha256_text(previous + sha256_text(canonical_json(value)))
            if previous != prefix:
                raise ValueError("a prefix hash is not the chained SHA-256 of the request values before it")

    @property
    def entry_count(self) -> int:
        """The number of request values."""
        return len(self.request)

    @property
    def list_sha256(self) -> str:
        """The hash of the shape, the tools and all request values: the last prefix hash, or the trajectory hash
        for a request without values."""
        return self.prefix_sha256[-1] if self.prefix_sha256 else self.trajectory_sha256

    def continues(self, entry_count: int, reply_count: int, reply: str | None) -> bool:
        """Whether this request resends, right after its first ``entry_count`` values, a reply of ``reply_count``
        messages whose hash is ``reply``."""
        following = self.request[entry_count : entry_count + reply_count]
        return reply is not None and len(following) == reply_count and reply_sha256(following) == reply


def call_key(payload: Mapping[str, Any]) -> CallKey | None:
    """The key of ``payload``, or ``None`` when the payload has no message list.

    Raise ``TypeError`` when the payload holds a value that is not JSON.
    """
    provider = payload_provider(payload)
    if provider is None:
        return None
    message_field = "input" if provider == OPENAI_RESPONSES else "messages"
    system_field = next(
        (name for name in ("system", "instructions") if name in payload and provider != OPENAI_CHAT), None
    )
    request = list(payload[message_field])
    if system_field is not None:
        request.insert(0, payload[system_field])
    shape: dict[str, Any] = {"provider": provider, "message_field": message_field, "system_field": system_field}
    if "tools" in payload:
        shape["tools"] = payload["tools"]
    try:
        trajectory = sha256_text("trajectory:" + canonical_json(shape))
        prefixes: list[str] = []
        previous = trajectory
        for value in request:
            previous = sha256_text(previous + sha256_text(canonical_json(value)))
            prefixes.append(previous)
    except (TypeError, ValueError) as exc:
        raise TypeError("a record must contain JSON-serializable values") from exc
    replies = reply_messages(payload.get("response")) or []
    return CallKey(
        trajectory, tuple(request), tuple(prefixes), len(replies), reply_sha256(replies) if replies else None
    )


@dataclass(frozen=True)
class ParentCandidate:
    """A stored call whose request is a prefix of a new call's request.

    Values: ``sequence`` orders the calls as they arrived; ``entry_count``, ``reply_count`` and ``reply_sha256``
    are those of the stored call's key.
    Purpose: a record store or ``TrajectoryIndex`` reads the calls whose ``list_sha256`` is one of a new key's
    prefix hashes, and ``best_parent`` chooses among them.
    """

    sequence: int
    entry_count: int
    reply_count: int
    reply_sha256: str | None


def best_parent(key: CallKey, candidates: Iterable[ParentCandidate]) -> ParentCandidate | None:
    """The parent of the call of ``key`` among ``candidates``: the longest prefix; among one length, a call whose
    reply the request resends; then the newest. ``None`` when there is no candidate."""
    return max(
        candidates,
        key=lambda candidate: (
            candidate.entry_count,
            key.continues(candidate.entry_count, candidate.reply_count, candidate.reply_sha256),
            candidate.sequence,
        ),
        default=None,
    )


def trajectory_calls(path: Sequence[tuple[str, int]]) -> tuple[str, ...]:
    """The calls of the trajectory that ends at the last call of ``path``, oldest first.

    ``path`` holds ``(record ID, entry count)`` for each call from the root to the last call. A call whose entry
    count equals the next call's has the same request: the next call retries it and takes its place.
    """
    calls = [path[-1][0]] if path else []
    for (record_id, count), (_, next_count) in zip(reversed(path[:-1]), reversed(path[1:]), strict=True):
        if count < next_count:
            calls.append(record_id)
    return tuple(reversed(calls))


@dataclass
class IndexedCall:
    """One call of a ``TrajectoryIndex``: its arrival order, parent, entry count, list hash and reply, whether a later
    call continued it (a child with more entries), and whether a later call retried it (a child with as many)."""

    sequence: int
    parent: str | None
    entry_count: int
    list_sha256: str
    reply_count: int
    reply_sha256: str | None
    continued: bool = False
    retried: bool = False


@dataclass
class TrajectoryIndex:
    """The inference records of one scenario linked into trajectories in memory, by the rule of this module.

    Values: ``calls`` maps the ID of each added inference record with a message list to its ``IndexedCall``;
    ``by_list`` maps a ``CallKey.list_sha256`` to the IDs of the calls with that hash; ``keyless`` holds the IDs of
    added inference records without a message list, each its own trajectory. A call's ``continued`` is true once a
    later call with more entries names it as parent, and ``retried`` once a later call with as many does. A call stays after ``forget`` drops its record, as a record
    store keeps a node that later calls need; ``forget`` only drops what it knows nothing more about.

    Purpose: a record-to-sample stage adds each inference record that it reads (``add``) and asks for the trajectory that a
    record ends (``trajectory``), and whether it ends a trajectory now (``is_end``). A replay of the retained
    records rebuilds it.
    """

    calls: dict[str, IndexedCall] = field(default_factory=dict)
    by_list: dict[str, list[str]] = field(default_factory=dict)
    keyless: set[str] = field(default_factory=set)
    next_sequence: int = 0

    def add(self, record: AgentRecord) -> None:
        """Link an inference record into its trajectory; a record added before, or that is not an inference, is
        ignored."""
        record_id = record.agent_record_id
        if record.request_type is not RequestType.INFERENCE or record_id in self.calls or record_id in self.keyless:
            return
        key = call_key(record.payload)
        if key is None:
            self.keyless.add(record_id)
            return
        candidates = [
            (candidate_id, ParentCandidate(call.sequence, call.entry_count, call.reply_count, call.reply_sha256))
            for prefix in key.prefix_sha256
            for candidate_id in self.by_list.get(prefix, ())
            for call in (self.calls[candidate_id],)
        ]
        chosen = best_parent(key, (candidate for _, candidate in candidates))
        parent = next((candidate_id for candidate_id, candidate in candidates if candidate is chosen), None)
        if parent is not None:
            if self.calls[parent].entry_count < key.entry_count:
                self.calls[parent].continued = True
            else:
                self.calls[parent].retried = True
        self.calls[record_id] = IndexedCall(
            self.next_sequence, parent, key.entry_count, key.list_sha256, key.reply_count, key.reply_sha256
        )
        self.next_sequence += 1
        self.by_list.setdefault(key.list_sha256, []).append(record_id)

    def path(self, record_id: str) -> tuple[str, ...]:
        """The IDs of the calls from the root to ``record_id``, retried calls included; ``(record_id,)`` for a record
        without a message list, and ``()`` for a record never added."""
        if record_id in self.keyless:
            return (record_id,)
        path: list[str] = []
        current: str | None = record_id
        while current is not None and current in self.calls:
            path.append(current)
            current = self.calls[current].parent
        return tuple(reversed(path))

    def trajectory(self, record_id: str) -> tuple[str, ...]:
        """The IDs of the calls of the trajectory that ends at ``record_id``, oldest first: its ``path`` without the
        calls that a retry replaced (``trajectory_calls``)."""
        if record_id in self.keyless:
            return (record_id,)
        return trajectory_calls([(call, self.calls[call].entry_count) for call in self.path(record_id)])

    def continued(self, record_id: str) -> bool:
        """Whether a later call with more entries continues the call ``record_id``."""
        call = self.calls.get(record_id)
        return call is not None and call.continued

    def is_end(self, record_id: str) -> bool:
        """Whether the added record ``record_id`` ends a trajectory now: no later call continued or retried it. A
        record without a message list always does."""
        if record_id in self.keyless:
            return True
        call = self.calls.get(record_id)
        return call is not None and not call.continued and not call.retried

    def forget(self, record_ids: Iterable[str]) -> None:
        """Drop released records that no other call names as parent. Others stay as links of later calls."""
        parents = {call.parent for call in self.calls.values()}
        for record_id in record_ids:
            self.keyless.discard(record_id)
            if record_id in self.calls and record_id not in parents:
                self.by_list[self.calls.pop(record_id).list_sha256].remove(record_id)


__all__ = [
    "CallKey",
    "ParentCandidate",
    "TrajectoryIndex",
    "best_parent",
    "call_key",
    "comparable_message",
    "reply_sha256",
    "sha256_text",
    "trajectory_calls",
]
