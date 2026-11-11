"""Receiver-side admission for production requests on a mutable engine stack."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any, Literal

from aiohttp import web

from reef.inference.interfaces import parse_full_weight_name
from reef.inference.provider_protocol import SSE_DONE, SSEFrameDecoder, frame_data


def generation_outcome(body: bytes, content_type: str, engine: str) -> Literal["finished", "refused", "ambiguous"]:
    """Require a native finish reason before releasing a healthy pin, including after an HTTP disconnect.

    Admission and the drain before each change of weights bind the request to its weights; the body only shows
    whether the generation stopped. The receiver keeps the bytes unchanged. A truncated or malformed body leaves
    execution uncertain; a pre-generation refusal does not.
    """
    try:
        values = []
        if "text/event-stream" in content_type:
            decoder = SSEFrameDecoder()
            frames = decoder.feed(body)
            final, remainder = decoder.finalize()
            if remainder.strip():
                return "ambiguous"
            done = False
            for frame in (*frames, *final):
                data = frame_data(frame)
                if not data:
                    continue
                if done:
                    return "ambiguous"
                if data == SSE_DONE:
                    done = True
                else:
                    values.append(json.loads(data))
        else:
            values.append(json.loads(body))
        if (
            len(values) == 1
            and isinstance(values[0], dict)
            and values[0].get("error") is not None
            and not any(key in values[0] for key in ("text", "choices", "output_ids", "meta_info"))
        ):
            return "refused"
        if not values or not all(isinstance(value, dict) and not value.get("error") for value in values):
            return "ambiguous"
        last = values[-1]
        if engine == "sglang":
            meta = last.get("meta_info")
            return "finished" if isinstance(meta, dict) and meta.get("finish_reason") is not None else "ambiguous"
        choices = last.get("choices")
        finished = (
            isinstance(choices, list)
            and bool(choices)
            and all(isinstance(choice, dict) and choice.get("finish_reason") is not None for choice in choices)
        )
        return "finished" if finished else "ambiguous"
    except (ValueError, UnicodeError, TypeError, AttributeError):
        return "ambiguous"


class RequestPins:
    """Pin admitted requests until their upstream transport ends, before any engine mutation.

    Values: ``enabled`` is true for engines that serve published weights, whose owner always requests pinning.
    It is false before startup and for training rollouts or endpoint-only use, which claim no artifact binding.
    For pinned engines, ``layout`` is "full" or "adapter" and
    ``drain_timeout_s`` is a positive, finite drain bound in seconds; pinned startup checks both fields.
    The drain bound also limits waiting for the control lock when an owner sequence is supplied.
    ``full_load_id`` is None or an
    acknowledged full-weight load ID; ``adapters`` is the set of acknowledged immutable adapter names.
    ``base_load_id`` is None or, on an adapter stack, the load ID that the owner gave the base model before any load
    (``initialize-version``). No adapter load changes the base model, so the ID names it until a reset.
    ``loaded`` records whether any load succeeded since reset, even when every adapter was later removed.
    ``uncertain`` latches an ambiguous operation until engine retirement and recovery. Admission requires
    ``accepting`` and not ``uncertain``; a concurrent quarantine can invalidate a resume's admission flag.
    ``active`` holds this event loop's forwarding tasks; ``idle`` is set exactly when it is empty.

    All methods run on the component's event loop. Admission checks and registration contain no await, so they
    are atomic against closing admission. The server's control lock serializes drain, mutation and resume.
    A disconnected client is drained upstream; an ambiguous upstream closes admission and requires recovery.
    """

    def __init__(self) -> None:
        self.enabled = False
        self.layout = "full"
        self.drain_timeout_s = 300.0
        self.accepting = False
        self.uncertain = False
        self.full_load_id: str | None = None
        self.adapters: set[str] = set()
        self.base_load_id: str | None = None
        self.loaded = False
        self.active: set[asyncio.Task[Any]] = set()
        self.idle = asyncio.Event()
        self.idle.set()

    def reset(self) -> None:
        """Forget bindings after confirmed engine retirement; admission stays closed."""
        self.accepting = False
        self.uncertain = False
        self.full_load_id = None
        self.adapters.clear()
        self.base_load_id = None
        self.loaded = False

    def quarantine(self) -> None:
        """Close admission after an ambiguous operation; a load or resume cannot clear this latch."""
        self.accepting = False
        self.uncertain = True
        self.full_load_id = None
        self.adapters.clear()
        self.base_load_id = None

    def admit(self, name: str, payload: Mapping[str, Any]) -> asyncio.Task[Any]:
        """Check and count a request without yielding; refuse stale, unbound or mismatched model selections."""
        valid = self.accepting and not self.uncertain
        valid = valid and not any(key in payload for key in ("lora_id", "lora_name"))
        try:
            full = parse_full_weight_name(name)
        except (ValueError, UnicodeError):
            full = None
        if self.layout == "full" or full is not None:
            # An adapter stack serves its base model under a full-weight name of ``base_load_id``.
            bound_load_id = self.full_load_id if self.layout == "full" else self.base_load_id
            valid = valid and full is not None and full[1] == bound_load_id and payload.get("lora_path") is None
            if full is not None and payload.get("model") is not None:
                valid = valid and payload["model"] == full[0]
        else:
            valid = valid and name in self.adapters and payload.get("lora_path") == name
        if not valid:
            raise web.HTTPMisdirectedRequest(text="the selected inference binding is unavailable")
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("request admission needs an asyncio task")
        self.active.add(task)
        self.idle.clear()
        return task

    def release(self, task: asyncio.Task[Any]) -> None:
        """Release a relay after completion or quarantine, never as proof that a detached engine stopped."""
        self.active.discard(task)
        if not self.active:
            self.idle.set()

    async def interrupt(self) -> None:
        """Cancel client relays before engine retirement; cancellation alone does not restore a binding."""
        self.accepting = False
        tasks = tuple(self.active)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def drain(self) -> None:
        """Close admission and wait before pausing the engine, or quarantine on an unbounded generation."""
        self.accepting = False
        if self.uncertain:
            await self.interrupt()
            return
        try:
            await asyncio.wait_for(self.idle.wait(), self.drain_timeout_s)
        except TimeoutError as exc:
            self.quarantine()
            await self.interrupt()
            raise web.HTTPServiceUnavailable(text="inference drain timed out; engine recovery is required") from exc
