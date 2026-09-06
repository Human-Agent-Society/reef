"""Read-only model tools over a frozen, committed evaluation history."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping
from typing import Any

from reef.harness.model_binding import ModelBinding, ModelBindingError

from .proposer_usage import response_cost
from .runtime_source import read_source, search_source, source_manifest


class ProposerBudgetReached(RuntimeError):
    pass


class ProposerUsageUnknown(RuntimeError):
    pass


class HistoryBinding:
    """Let the proposer inspect complete trajectories without filling its prompt.

    Tools address committed record IDs and page through their steps. They have
    no filesystem or network access, and cannot read another scenario's data.
    """

    def __init__(
        self,
        binding: ModelBinding,
        records: Mapping[str, Any],
        *,
        effort: str = "xhigh",
        pricing=None,
        remaining_cost_usd=None,
        executable=False,
        sources=None,
        timeout_s=None,
    ) -> None:
        if binding.api not in ("openai", "responses"):
            raise ValueError("the Terminal-Bench history tools require an OpenAI Chat or Responses binding")
        self.binding = binding
        self.records = json.loads(json.dumps(records))
        self.effort = effort
        self.audit = []
        self.pricing = pricing
        self.remaining_cost_usd = remaining_cost_usd
        self.cost_usd = 0.0
        self.unknown_usage = False
        self.executable = executable
        self.sources = dict(sources or {})
        if timeout_s is not None and (not math.isfinite(timeout_s) or timeout_s <= 0):
            raise ValueError("proposer timeout must be finite and positive")
        self.timeout_s = timeout_s

    def read(self, record_id: str, start: int = 0, limit: int = 4) -> dict[str, Any]:
        if record_id not in self.records:
            raise ValueError("unknown committed record_id")
        if isinstance(start, bool) or not isinstance(start, int) or start < 0:
            raise ValueError("start must be a non-negative integer")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 8:
            raise ValueError("limit must be between 1 and 8")
        record = self.records[record_id]
        steps = record.get("trajectory", [])
        return {
            **{k: v for k, v in record.items() if k != "trajectory"},
            "steps": steps[start : start + limit],
            "start": start,
            "total_steps": len(steps),
        }

    def chat(self, messages, **params):
        deadline = time.monotonic() + self.timeout_s if self.timeout_s is not None else None
        index = [
            {
                "record_id": key,
                **{
                    k: row.get(k)
                    for k in (
                        "candidate_id",
                        "side",
                        "task",
                        "repeat",
                        "phase",
                        "reward",
                        "benchmark_score",
                        "error",
                        "measurement_role",
                        "baseline_included",
                    )
                },
                "steps": len(row.get("trajectory", [])),
            }
            for key, row in self.records.items()
        ]
        conversation = [dict(message) for message in messages]
        if self.executable:
            conversation.append(
                {
                    "role": "user",
                    "content": "The harness is a single self-contained Python code_extension module. Define class Agent "
                    "as a subclass of harbor.agents.terminus_2.Terminus2 (Harbor 0.20.0). It runs in an E2B "
                    "runner and controls another E2B task sandbox. Preserve the fixed model, task instruction, "
                    "verifier, timeouts and usage accounting. Do not inspect hidden tests or write reward files. "
                    "Use a valid Python identifier for the module name. Return the full composition JSON.",
                }
            )
        conversation.append(
            {
                "role": "user",
                "content": "Inspect the committed evaluation evidence before proposing a change. "
                "Use read_evaluation to diagnose failed commands, model behavior and verifier feedback. "
                "All prior candidates' recorded trajectories are available in pages. "
                "The final response must be the requested composition JSON.\n" + json.dumps(index),
            }
        )
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_evaluation",
                    "description": "Read a page of a committed evaluation's trajectory.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "record_id": {"type": "string"},
                            "start": {"type": "integer", "minimum": 0},
                            "limit": {"type": "integer", "minimum": 1, "maximum": 8},
                        },
                        "required": ["record_id"],
                        "additionalProperties": False,
                    },
                },
            }
        ]
        if self.sources:
            conversation.append(
                {
                    "role": "user",
                    "content": "These frozen Harbor source files are available through read_runtime_source and "
                    "search_runtime_source. Inspect the methods you override before proposing code.\n"
                    + json.dumps(
                        {
                            path: {"sha256": digest, "lines": len(self.sources[path].splitlines())}
                            for path, digest in source_manifest(self.sources).items()
                        }
                    ),
                }
            )
            for name, description, properties, required in (
                (
                    "read_runtime_source",
                    "Read lines of a frozen Harbor source file.",
                    {
                        "path": {"type": "string"},
                        "start": {"type": "integer", "minimum": 0},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 240},
                    },
                    ["path"],
                ),
                (
                    "search_runtime_source",
                    "Search the frozen Harbor source for a literal string.",
                    {"query": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 100}},
                    ["query"],
                ),
            ):
                tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": description,
                            "parameters": {
                                "type": "object",
                                "properties": properties,
                                "required": required,
                                "additionalProperties": False,
                            },
                        },
                    }
                )
        for _ in range(32):
            if self.remaining_cost_usd is not None and self.cost_usd >= self.remaining_cost_usd:
                raise ProposerBudgetReached("observed proposer spend reached its committed allowance")
            request_options = {}
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("proposer exhausted its configured time allowance")
                request_options["timeout_s"] = min(getattr(self.binding, "timeout_s", 600), remaining)
            responses = self.binding.api == "responses"
            if responses:
                body = {
                    "input": conversation,
                    "tools": [{"type": "function", **tool["function"], "strict": False} for tool in tools],
                    "reasoning": {"effort": self.effort},
                    "store": False,
                    "include": ["reasoning.encrypted_content"],
                    **params,
                }
                if self.pricing:
                    body.update(service_tier="default", max_output_tokens=16384)
            else:
                body = {"messages": conversation, "tools": tools, "reasoning_effort": self.effort, **params}
                if self.pricing:
                    body.update(service_tier="default", max_completion_tokens=16384)
            request_hash = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
            self.unknown_usage = True
            try:
                response = self.binding.complete(body, **request_options)
            except ModelBindingError as exc:
                # A specific provider validation rejection cannot have run
                # generation. Keep transport/server errors unknown; never
                # infer zero cost from an absent usage object alone.
                try:
                    error = json.loads(exc.detail).get("error", {})
                except (ValueError, TypeError):
                    error = {}
                rejected = (
                    exc.status == 400
                    and error.get("type") == "invalid_request_error"
                    and error.get("param") == "reasoning_effort"
                    and (
                        error.get("code") == "unsupported_value"
                        or "Function tools with reasoning_effort are not supported" in error.get("message", "")
                    )
                )
                self.audit.append(
                    {
                        "request_sha256": request_hash,
                        "http_status": exc.status,
                        "provider_error": {key: error.get(key) for key in ("type", "code", "param")},
                        "rejected_before_generation": rejected,
                        "cost_usd": 0.0 if rejected else None,
                    }
                )
                self.unknown_usage = not rejected
                raise
            except Exception:
                self.unknown_usage = True
                raise
            cost = response_cost(response, self.pricing) if self.pricing else None
            self.audit.append(
                {
                    "request_sha256": request_hash,
                    "response_id": response.get("id"),
                    "response": response.get("output") if responses else response.get("choices"),
                    "usage": response.get("usage"),
                    "cost_usd": cost,
                }
            )
            if self.pricing:
                if cost is None:
                    self.unknown_usage = True
                    raise ProposerUsageUnknown("proposer returned incomplete or unsupported usage")
                self.cost_usd += cost
            self.unknown_usage = False
            if responses:
                if response.get("status", "completed") != "completed":
                    raise ValueError("proposer response did not complete")
                output = response.get("output", [])
                calls = [
                    {"id": item["call_id"], "function": {"name": item["name"], "arguments": item["arguments"]}}
                    for item in output
                    if item.get("type") == "function_call"
                ]
                content = "".join(
                    part["text"]
                    for item in output
                    if item.get("type") == "message"
                    for part in item.get("content", [])
                    if part.get("type") == "output_text"
                )
            else:
                message = response["choices"][0]["message"]
                calls = message.get("tool_calls") or []
                content = message.get("content")
            if not calls:
                if not isinstance(content, str) or not content:
                    raise ValueError("proposer returned no composition text")
                return content
            # Preserve reasoning items and function-call identities for
            # stateless tool continuation, as required by Responses.
            if responses:
                conversation.extend(output)
            else:
                conversation.append(message)
            for call in calls:
                try:
                    function = call["function"]
                    arguments = json.loads(function["arguments"])
                    if function["name"] == "read_evaluation":
                        result = self.read(**arguments)
                    elif self.sources and function["name"] == "read_runtime_source":
                        result = read_source(self.sources, **arguments)
                    elif self.sources and function["name"] == "search_runtime_source":
                        result = search_source(self.sources, **arguments)
                    else:
                        raise ValueError("unknown history tool")
                except (ValueError, TypeError, KeyError) as exc:
                    result = {"error": str(exc)}
                conversation.append(
                    {"type": "function_call_output", "call_id": call["id"], "output": json.dumps(result)}
                    if responses
                    else {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)}
                )
        raise RuntimeError("proposer exhausted its 32 history-tool turns without returning a composition")
