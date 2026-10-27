"""The model-call budget, record and live activity of one harness step.

Every model call a method makes in a step goes through ``StepCalls``: the
bindings it receives in ``Tools.models`` are ``BudgetedBinding`` wrappers,
and a gateway that serves an agent process spends from the same object. The
calls land in the step's ``calls.json``, and the activity lines feed the
request page while the step runs.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any

from reef.harness.episodes.model_binding import ModelBinding, ModelBindings
from reef.harness.tree.checks import redact_secret_shaped
from reef.train.harness_backend.contract import ModelCalls
from reef.train.harness_backend.records import bounded


#: The most activity lines a step keeps; the oldest go first.
MAX_ACTIVITY = 300


class StepCalls(ModelCalls):
    """One step's model-call budget, record and live activity, shared by the budgeted bindings and an agent's gateway.

    Values: ``cap`` ≥ 0 is the step's call budget, 0 for none, and ``spent``
    ≤ ``cap`` when ``cap`` > 0 counts the calls so far. ``entries`` is the
    list the step writes to ``calls.json``, one mapping per call in call
    order, each redacted and clipped as ``bounded`` makes it. ``lines`` holds
    the latest ``MAX_ACTIVITY`` activity lines, each ``{at, kind, text}``
    with ``failed`` on a line that went wrong, its text redacted.
    Purpose: the backend makes one per step and hands it out as
    ``Tools.calls``; the budgeted bindings and an agent's gateway spend and
    record through it, and the request page reads :meth:`activity` from
    another thread.
    """

    def __init__(self, cap: int, record: list[dict[str, Any]]) -> None:
        self.cap = cap
        self.spent = 0
        self.entries = record
        self.lines: deque[dict[str, Any]] = deque(maxlen=MAX_ACTIVITY)
        self.lock = threading.Lock()

    def spend(self) -> None:
        # A method may call models from several threads; the count and the check go together.
        with self.lock:
            if self.cap and self.spent >= self.cap:
                raise RuntimeError(f"model call budget of {self.cap} per evolve step exhausted")
            self.spent += 1

    def record(self, entry: Mapping[str, Any]) -> None:
        # Every caller's entry, a gateway's too, is redacted and clipped before the record keeps it.
        self.entries.append(bounded(entry))

    def note(self, kind: str, text: str, *, failed: bool = False) -> None:
        # A line can quote an error or a reply, which the request page shows: it is redacted as the record is.
        shown = redact_secret_shaped(" ".join(text.split()))[:300]
        line: dict[str, Any] = {"at": time.time(), "kind": kind, "text": shown}
        if failed:
            line["failed"] = True
        with self.lock:
            self.lines.append(line)

    def activity(self) -> tuple[dict[str, Any], ...]:
        with self.lock:
            return tuple(dict(line) for line in self.lines)


class BudgetedBinding(ModelBinding):
    """A ModelBinding that delegates ``chat`` and ``complete`` to a wrapped binding under the step's call budget.

    Values: ``inner`` is the binding the calls go to, and ``step_calls`` the
    step's ``StepCalls``; the endpoint fields repeat ``inner``'s. Every
    binding of one step shares one ``StepCalls``, so they spend one budget.
    Purpose: ``budgeted_bindings`` wraps every binding a method receives.
    Each call spends from the budget first, then lands in the step record:
    one entry with the model, the request, the reply or the error, the
    provider response when available, the seconds it took and the ``usage``
    (input and output tokens), every text cut at the record cap. A subclass,
    so the method still receives ModelBinding values; it forwards to the
    wrapped binding, so a method's own binding behavior is preserved.
    """

    inner: ModelBinding
    step_calls: StepCalls

    def __init__(self, inner: ModelBinding, calls: StepCalls) -> None:
        super().__init__(
            base_url=inner.base_url,
            model=inner.model,
            api_key=inner.api_key,
            api=inner.api,
            timeout_s=inner.timeout_s,
            metadata=inner.metadata,
        )
        object.__setattr__(self, "inner", inner)
        object.__setattr__(self, "step_calls", calls)

    def chat(self, messages: Sequence[Mapping[str, Any]], *, timeout_s: float | None = None, **params: Any) -> str:
        self.step_calls.spend()
        self.step_calls.note("model", f"asking {self.model}")
        kwargs: dict[str, Any] = dict(params)
        if timeout_s is not None:
            kwargs["timeout_s"] = timeout_s
        # The record bounds the entry, so it holds the call as made.
        entry: dict[str, Any] = {"model": self.model, "messages": messages, "params": kwargs}
        # ``chat`` returns text only; keep the provider response too, including any reasoning it exposed.
        if isinstance(self.inner, ModelBinding):
            object.__setattr__(self.inner, "_last_usage", None)
            object.__setattr__(self.inner, "_last_response", None)
        started = time.monotonic()
        try:
            reply = self.inner.chat(messages, **kwargs)
        except BaseException as exc:
            # The failed call is the step's decision too: the record keeps it before the error propagates.
            entry["error"] = f"{type(exc).__name__}: {exc}"
            raise
        else:
            entry["reply"] = reply
            return reply
        finally:
            # A provider may spend its budget on reasoning and return no final text; retain that response on error too.
            response = self.inner.last_response() if isinstance(self.inner, ModelBinding) else None
            if response is not None:
                entry["response"] = response
            entry["seconds"] = round(time.monotonic() - started, 3)
            usage = self.inner.last_usage() if isinstance(self.inner, ModelBinding) else None
            if usage is not None:
                entry["usage"] = usage
            self.step_calls.record(entry)
            self.note_answer(entry)

    def last_response(self) -> dict[str, Any] | None:
        """The provider response of the latest call, as the wrapped binding kept it."""
        return self.inner.last_response() if isinstance(self.inner, ModelBinding) else None

    def note_answer(self, entry: Mapping[str, Any]) -> None:
        """One activity line for a finished call: how long it took and its tokens, or its error."""
        seconds = entry.get("seconds", 0)
        if "error" in entry:
            self.step_calls.note("model", f"{self.model} failed after {seconds:g} s: {entry['error']}", failed=True)
            return
        usage = entry.get("usage")
        tokens = (
            f", {int(usage.get('input_tokens', 0) or 0):,} → {int(usage.get('output_tokens', 0) or 0):,} tokens"
            if isinstance(usage, Mapping)
            else ""
        )
        self.step_calls.note("model", f"{self.model} answered in {seconds:g} s{tokens}")

    def complete(self, body: Mapping[str, Any], *, timeout_s: float | None = None) -> dict[str, Any]:
        """A method's raw request goes through the same budget and record as ``chat``: ``body`` in, ``response`` out."""
        self.step_calls.spend()
        self.step_calls.note("model", f"asking {self.model}")
        kwargs: dict[str, Any] = {} if timeout_s is None else {"timeout_s": timeout_s}
        entry: dict[str, Any] = {"model": self.model, "body": body, "params": kwargs}
        started = time.monotonic()
        try:
            response = self.inner.complete(body, **kwargs)
        except BaseException as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["seconds"] = round(time.monotonic() - started, 3)
            self.step_calls.record(entry)
            self.note_answer(entry)
            raise
        entry["response"] = response
        entry["seconds"] = round(time.monotonic() - started, 3)
        # The wrapped binding's complete records the usage of the response it returned.
        usage = self.inner.last_usage()
        if usage is not None:
            entry["usage"] = usage
        self.step_calls.record(entry)
        self.note_answer(entry)
        return response


def call_tokens(record: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    """Input and output tokens summed over the record's calls that reported usage."""
    usages = [entry["usage"] for entry in record if isinstance(entry.get("usage"), Mapping)]
    return (
        sum(int(usage.get("input_tokens", 0) or 0) for usage in usages),
        sum(int(usage.get("output_tokens", 0) or 0) for usage in usages),
    )


def budgeted_bindings(models: ModelBindings, calls: StepCalls) -> ModelBindings:
    """The method's view: every ``chat`` shares the step's budget and lands in its record.

    The bindings are wrapped whatever the cap, so the record sees every call;
    with a cap of 0 nothing is refused. ``calls`` is per prepare_step call,
    so a cap bounds one step's model bill, never the campaign's.
    """

    def wrap(binding: ModelBinding) -> ModelBinding:
        return BudgetedBinding(binding, calls)

    return ModelBindings(
        served=wrap(models.served),
        named={name: wrap(models[name]) for name in models if name != "served"},
    )


__all__ = ["BudgetedBinding", "StepCalls", "budgeted_bindings", "call_tokens"]
