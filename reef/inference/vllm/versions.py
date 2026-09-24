"""Which weight version produced each sampled token of a vLLM request.

The connector in :mod:`reef.inference.vllm.connector` feeds this tracker from
inside vLLM's engine-core process; the chat client reads its output from the
response. Neither side needs the other's imports, and this module imports no
vLLM, so the bookkeeping is unit-tested on CPU.
"""

from __future__ import annotations

#: The ``kv_transfer_params`` entry that carries one runtime load ID per sampled token.
TOKEN_RUNTIME_LOAD_IDS_KEY = "reef_token_runtime_load_ids"
#: Stamp for a token whose producing version was never observed; the chat client rejects it.
UNKNOWN_RUNTIME_LOAD_ID = "unknown"


class TokenVersionTracker:
    """Record, per in-flight request, the output index from which each weight version applies.

    ``note`` is called once per scheduled request per step, before the forward
    pass, with the index of the next token that step will produce. ``finish``
    turns the recorded events into one version per sampled token. Reading the
    index from the live request rather than assuming zero for a new request is
    what keeps a request resumed after preemption correct: its earlier tokens
    keep their versions.
    """

    def __init__(self, version: str = "default") -> None:
        self.version = version
        self._events: dict[str, list[tuple[int, str]]] = {}

    def set_version(self, version: str) -> None:
        self.version = version

    def note(self, request_id: str, output_index: int) -> None:
        """Record that tokens from ``output_index`` on are produced by the current version."""
        events = self._events.setdefault(request_id, [])
        if events and events[-1][1] == self.version:
            return
        if events and events[-1][0] == output_index:
            # Same index seen again under a newer version (a version bump during
            # prefill chunks): the version that produces the token wins.
            events[-1] = (output_index, self.version)
            return
        events.append((output_index, self.version))

    def finish(self, request_id: str, output_count: int) -> list[str]:
        """Return one version per sampled token and drop the request's events."""
        events = self._events.pop(request_id, [])
        stamps: list[str] = []
        for index, (start, version) in enumerate(events):
            end = events[index + 1][0] if index + 1 < len(events) else output_count
            if start > len(stamps):
                # Tokens produced by steps this tracker never saw scheduled:
                # never expected, so mark them instead of guessing and let the
                # client refuse the response.
                stamps.extend([UNKNOWN_RUNTIME_LOAD_ID] * (start - len(stamps)))
            if end > len(stamps):
                stamps.extend([version] * (end - len(stamps)))
        if len(stamps) < output_count:
            stamps.extend([UNKNOWN_RUNTIME_LOAD_ID] * (output_count - len(stamps)))
        return stamps[:output_count]

    def forget(self, request_id: str) -> None:
        self._events.pop(request_id, None)


__all__ = ["TOKEN_RUNTIME_LOAD_IDS_KEY", "UNKNOWN_RUNTIME_LOAD_ID", "TokenVersionTracker"]
