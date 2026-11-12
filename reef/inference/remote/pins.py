"""Receiver-side admission for production requests on a mutable engine stack."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any

from aiohttp import web

from reef.core.weight_layout import WeightLayout
from reef.inference.interfaces import parse_full_weight_name
from reef.inference.provider_protocol import SSE_DONE, SSEFrameDecoder, frame_data


def generation_stopped(body: bytes, content_type: str, engine: str) -> bool:
    """Whether the complete ``body`` of a successful upstream response proves that its generation stopped.

    ``body`` is the answer of the "sglang" or "vllm" ``engine`` to one admitted native generation: one JSON object,
    or its server-sent events when ``content_type`` names an event stream. True for a refusal before generation,
    which is a single object with an ``error`` and no generation field, and for a body without errors whose last
    object has a native finish reason. False for every other body, a truncated or malformed one included: the
    engine may then still generate. Admission and the drain before each change of weights bind the request to its
    weights; the body only shows whether the generation stopped. The receiver keeps the bytes unchanged in every
    case.
    """
    if "text/event-stream" in content_type:
        decoder = SSEFrameDecoder()
        frames = decoder.feed(body)
        final, remainder = decoder.finalize()
        events = [data for frame in (*frames, *final) if (data := frame_data(frame))]
        # The terminal marker ends a stream: an unfinished event or data after the marker is not a native answer.
        if remainder.strip() or SSE_DONE in events[:-1]:
            return False
        documents: list[str | bytes] = [data for data in events if data != SSE_DONE]
    else:
        documents = [body]
    try:
        values = [json.loads(document) for document in documents]
    except ValueError:
        return False
    if not values or not all(isinstance(value, dict) for value in values):
        return False
    if (
        len(values) == 1
        and values[0].get("error") is not None
        and not any(key in values[0] for key in ("text", "choices", "output_ids", "meta_info"))
    ):
        return True
    if any(value.get("error") for value in values):
        return False
    last = values[-1]
    if engine == "sglang":
        meta = last.get("meta_info")
        return isinstance(meta, dict) and meta.get("finish_reason") is not None
    choices = last.get("choices")
    return (
        isinstance(choices, list)
        and bool(choices)
        and all(isinstance(choice, dict) and choice.get("finish_reason") is not None for choice in choices)
    )


class RequestPins:
    """Pin admitted requests until their upstream transport ends, before any engine mutation.

    Values: ``enabled`` is true for engines that serve published weights, whose owner always requests pinning.
    It is false before startup and for training rollouts or endpoint-only use, which claim no artifact binding.
    ``layout`` is "full" or "adapter" and ``drain_timeout_s`` is a positive, finite drain bound in seconds. A
    pinned startup checks and sets both; they keep their defaults otherwise. The drain bound also limits waiting
    for the control lock when an owner sequence is supplied. ``full_load_id`` is None or an acknowledged
    full-weight load ID; ``adapters`` is the set of acknowledged immutable adapter names.
    ``base_load_id`` is None or, on an adapter stack, the load ID that the owner gave the base model before any load
    (``initialize-version``). No adapter load changes the base model, so the ID names it until a reset or a
    quarantine.
    ``loaded`` records whether any load succeeded since reset, even when every adapter was later removed.
    ``uncertain`` latches an ambiguous operation until engine retirement and recovery. Admission requires
    ``accepting`` and not ``uncertain``; a concurrent quarantine can invalidate a resume's admission flag.
    ``active`` holds this event loop's forwarding tasks; ``idle`` is set exactly when it is empty.

    Purpose: one ``RemoteInferenceServer`` owns the value. Its control routes record the configuration and each
    binding that the engines acknowledged, and its ``forward`` admits and releases each pinned request.

    All methods run on the component's event loop. Admission checks and registration contain no await, so they
    are atomic against closing admission. The server's control lock serializes drain, mutation and resume.
    A disconnected client is drained upstream; an ambiguous upstream closes admission and requires recovery.
    """

    def __init__(self) -> None:
        self.enabled = False
        self.layout: WeightLayout = "full"
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
        """Check and count a request without yielding; refuse stale, unbound or mismatched model selections.

        ``name`` is the served-model name that the request selects and ``payload`` its native generation body.
        A full-weight name (``full_weight_name``) selects whole model weights: a full-weight stack admits it for
        its acknowledged load (``full_load_id``), and an adapter stack for its base model (``base_load_id``). In
        both cases the body's ``model`` is absent or equal to the name's model path, and the body has no
        ``lora_path``. An adapter stack also admits an acknowledged adapter name that is also the body's
        ``lora_path``. Neither stack admits another adapter selector. Any other request gets HTTP 421, which
        changes no state.
        """
        try:
            selected = parse_full_weight_name(name)
        except ValueError:
            selected = None
        if selected is not None:
            model_path, load_id = selected
            bound_load_id = self.full_load_id if self.layout == "full" else self.base_load_id
            # A load ID is a string, so it never equals a binding that is None.
            bound = (
                load_id == bound_load_id
                and payload.get("lora_path") is None
                and payload.get("model") in (None, model_path)
            )
        elif self.layout == "full":
            bound = False
        else:
            bound = name in self.adapters and payload.get("lora_path") == name
        if (
            not bound
            or not self.accepting
            or self.uncertain
            or any(key in payload for key in ("lora_id", "lora_name"))
        ):
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
