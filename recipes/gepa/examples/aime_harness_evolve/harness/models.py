"""Token accounting, the persistent spend cap, and tracked model clients."""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

from gepa.lm import LM as GEPALM

from reef.harness.model_binding import ModelBinding, ModelBindingError

from .files import read_json, write_json


class TokenUsage(TypedDict):
    requests: int
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_tokens: int


def empty_usage() -> TokenUsage:
    return {"requests": 0, "input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}


@dataclass(frozen=True)
class ModelPrice:
    """Standard-processing USD rates observed for one pinned model; copied into every report."""

    input_per_million: float
    cached_input_per_million: float
    output_per_million: float
    source: str
    observed_at: str = "2026-08-30"

    def estimate(self, usage: Mapping[str, int]) -> float:
        cached = int(usage.get("cached_input_tokens", 0))
        uncached = max(0, int(usage.get("input_tokens", 0)) - cached)
        output = int(usage.get("output_tokens", 0))
        return (
            uncached * self.input_per_million
            + cached * self.cached_input_per_million
            + output * self.output_per_million
        ) / 1_000_000


TASK_MODEL_PRICE = ModelPrice(0.40, 0.10, 1.60, "https://developers.openai.com/api/docs/models/gpt-4.1-mini")
REFLECTION_MODEL_PRICE = ModelPrice(1.25, 0.125, 10.00, "https://developers.openai.com/api/docs/models/gpt-5")


class SpendCapReached(SystemExit):
    """Stop the run before a scorer can turn budget exhaustion into a zero score."""


class SpendCap:
    """Persist completed-call cost and refuse to start a call once it reaches the cap.

    Calls already in flight can overshoot, and one Pi episode may hold several
    requests, so the account-side budget remains the hard ceiling.
    """

    def __init__(self, path: Path, max_usd: float) -> None:
        self.path = Path(path)
        self.max_usd = float(max_usd)
        self._lock = threading.Lock()
        state = read_json(self.path) if self.path.is_file() else {}
        self.observed_usd = float(state.get("observed_cost_usd", 0.0))
        self.completed_calls = int(state.get("completed_calls", 0))
        self._write()

    def before_call(self) -> None:
        with self._lock:
            if self.observed_usd >= self.max_usd:
                raise SpendCapReached(
                    f"observed cost ${self.observed_usd:.4f} reached the ${self.max_usd:.4f} cap; "
                    "no new model call was started"
                )

    def record_call(self, cost_usd: float) -> None:
        with self._lock:
            self.observed_usd += float(cost_usd)
            self.completed_calls += 1
            self._write()

    def _write(self) -> None:
        write_json(
            self.path,
            {
                "completed_calls": self.completed_calls,
                "max_observed_cost_usd": self.max_usd,
                "observed_cost_usd": self.observed_usd,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )


class UsageLedger:
    """Thread-safe token totals, persisted after every call so a resumed run keeps them."""

    def __init__(self, price: ModelPrice, path: Path | None = None) -> None:
        self.price = price
        self.path = path
        self._lock = threading.Lock()
        self._usage = empty_usage()
        if path is not None and path.is_file():
            self._usage.update(read_json(path)["usage"])
        self._persist()

    def add(self, usage: Mapping[str, int]) -> None:
        with self._lock:
            for key in self._usage:
                self._usage[key] += int(usage.get(key, 0))
            self._persist()

    def add_openai_response(self, response: Mapping[str, Any]) -> TokenUsage:
        raw = response.get("usage") or {}
        usage: TokenUsage = {
            "requests": 1,
            "input_tokens": _count(raw.get("prompt_tokens")),
            "cached_input_tokens": _count((raw.get("prompt_tokens_details") or {}).get("cached_tokens")),
            "output_tokens": _count(raw.get("completion_tokens")),
            "reasoning_tokens": _count((raw.get("completion_tokens_details") or {}).get("reasoning_tokens")),
        }
        self.add(usage)
        return usage

    def snapshot(self) -> TokenUsage:
        with self._lock:
            return dict(self._usage)  # type: ignore[return-value]

    def _persist(self) -> None:
        if self.path is not None:
            write_json(self.path, {"pricing": asdict(self.price), "usage": self._usage})


class TrackedChatModel:
    """GEPA's reflection callable, backed by Reef's model binding with usage and spend tracking."""

    def __init__(
        self,
        binding: ModelBinding,
        *,
        price: ModelPrice,
        spend_cap: SpendCap | None = None,
        usage_path: Path | None = None,
    ) -> None:
        self.binding = binding
        self.usage = UsageLedger(price, usage_path)
        self.spend_cap = spend_cap

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        if self.spend_cap is not None:
            self.spend_cap.before_call()
        response = self.binding.complete({"messages": messages})
        usage = self.usage.add_openai_response(response)
        if self.spend_cap is not None:
            self.spend_cap.record_call(self.usage.price.estimate(usage))
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelBindingError(f"model endpoint returned no completion: {response!r}"[:600]) from exc
        if not isinstance(content, str):
            raise ModelBindingError("model endpoint returned non-text content")
        return content


class TrackedGEPALM(GEPALM):
    """Upstream GEPA's LiteLLM transport, unchanged, with the same usage ledger and spend cap."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str,
        base_url: str,
        max_completion_tokens: int | None = None,
        price: ModelPrice,
        spend_cap: SpendCap | None = None,
        usage_path: Path | None = None,
    ) -> None:
        super().__init__(
            f"openai/{model}",
            api_key=api_key,
            api_base=f"{base_url.rstrip('/')}/v1",
            max_completion_tokens=max_completion_tokens,
        )
        self.usage = UsageLedger(price, usage_path)
        self.spend_cap = spend_cap
        self._lock = threading.Lock()

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        self._before(1)
        with self._lock:
            before = (self.total_tokens_in, self.total_tokens_out)
            result = super().__call__(prompt)
            usage = self._record(before, requests=1)
        self._after(usage, 1)
        return result

    def batch_complete(
        self, messages_list: list[list[dict[str, Any]]], max_workers: int = 10, **kwargs: Any
    ) -> list[str]:
        self._before(len(messages_list))
        with self._lock:
            before = (self.total_tokens_in, self.total_tokens_out)
            results = super().batch_complete(messages_list, max_workers=max_workers, **kwargs)
            usage = self._record(before, requests=len(results))
        self._after(usage, len(results))
        return results

    def _before(self, calls: int) -> None:
        if self.spend_cap is not None:
            for _ in range(calls):
                self.spend_cap.before_call()

    def _after(self, usage: TokenUsage, calls: int) -> None:
        if self.spend_cap is not None and calls:
            per_call = self.usage.price.estimate(usage) / calls
            for _ in range(calls):
                self.spend_cap.record_call(per_call)

    def _record(self, before: tuple[int, int], *, requests: int) -> TokenUsage:
        # Upstream keeps running token totals; the difference is this call's usage.
        usage = empty_usage()
        usage["requests"] = requests
        usage["input_tokens"] = max(0, self.total_tokens_in - before[0])
        usage["output_tokens"] = max(0, self.total_tokens_out - before[1])
        self.usage.add(usage)
        return usage


def trajectory_usage(trajectory: Sequence[Mapping[str, Any]]) -> TokenUsage:
    """Sum Pi's per-message usage. Pi reports uncached input and cache reads
    separately; their sum is stored as input so ModelPrice applies the cached
    discount correctly."""
    total = empty_usage()
    for event in trajectory:
        wrapped = event.get("message")
        message = wrapped if isinstance(wrapped, Mapping) else event
        usage = message.get("usage") if message.get("role") == "assistant" else None
        if not isinstance(usage, Mapping):
            continue
        cached = _count(usage.get("cacheRead"))
        total["requests"] += 1
        total["input_tokens"] += _count(usage.get("input")) + cached
        total["cached_input_tokens"] += cached
        total["output_tokens"] += _count(usage.get("output"))
        total["reasoning_tokens"] += _count(usage.get("reasoning"))
    return total


def _count(value: Any) -> int:
    return int(value) if isinstance(value, int | float) and value >= 0 else 0
