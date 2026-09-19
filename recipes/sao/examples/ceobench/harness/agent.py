"""CEO-Bench's bash agent, played from the host.

This is the benchmark's agent loop (``saas_bench.agents.bash_agent.agent``
at commit d2b7b32e, its OpenAI chat-completions path) kept step for step, so
the model sees what the benchmark's own runner shows it:

- The conversation starts empty. The system prompt (the benchmark's template
  with the workspace's ``MEMORY.md`` appended) enters when the first week
  advances, and the conversation is rebuilt from it at every later week.
- One tool call per turn: the first call of a response is executed and the
  others are answered ``[Skipped - only one tool per turn ...]``; a response
  without a tool call, or with arguments that are not JSON, gets the
  benchmark's feedback text and is regenerated.
- Retryable API errors (5xx, 429, connection and timeout errors) back off and
  retry without limit; other API and request-validation errors are fed back
  to the model as a user turn and the turn is regenerated. Internal
  programming errors propagate to Harbor.

Neither the prompt nor the tool definitions are copied here: the harness
reads both from the pinned checkout in the task image at the start of the
episode (``harness.harbor_agent``), so they are the benchmark's byte for
byte. The tools themselves are ``harness.tools``; this module only decides
what to call.
"""

from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import openai
from openai.types.chat import ChatCompletion

from .values import JsonValue

#: The benchmark detects an advanced day in bash output by this header. The
#: engine's CLI prints ``=== Week N Dashboard (Day D) ===``, which never
#: matches, so in practice the day advances through the engine's status
#: (``act``'s ``info["day"]``); the check is kept as the benchmark has it.
DASHBOARD_DAY_RE = re.compile(r"=== Day (\d+) Dashboard ===")
#: ``MEMORY.md`` is appended to the system prompt up to this many characters.
MEMORY_MAX_CHARS = 40_000
DEFAULT_MAX_COMPLETION_TOKENS = 16384
TEMPERATURE = 1.0
#: Retry policy of the benchmark's loop: exponential backoff for retryable
#: errors, a short linear wait after feeding a non-retryable one back.
RETRY_BACKOFF_CAP_S, RETRY_BACKOFF_BASE_S, RETRY_BACKOFF_STEPS = 120, 10, 3
FEEDBACK_WAIT_CAP_S, FEEDBACK_WAIT_STEP_S = 60, 5
RETRYABLE_ERROR_CODES = ("429", "500", "502", "503", "504", "529")

NO_TOOL_FEEDBACK = (
    "You must call a tool to proceed. If you have nothing else to do this week, "
    "call `./novamind-operation next-week <cash_1wk> <cash_4wk> <cash_12wk>` via bash to advance."
)


def invalid_json_feedback(name: str, error: str, preview: str) -> str:
    return (
        f"Your previous response contained invalid JSON in the `{name}` tool_call arguments.\n"
        f"JSON decode error: {error}\n"
        f"Arguments started with: {preview}...\n\n"
        f'Valid JSON escape sequences are limited to: \\" \\\\ \\/ \\b \\f \\n \\r \\t \\uXXXX. '
        f"Shell-style escapes like \\$ or \\! are NOT valid JSON. "
        f"Please re-emit the tool call with valid JSON."
    )


def non_retryable_feedback(error: BaseException) -> str:
    return (
        f"The previous API request failed with a non-retryable error:\n"
        f"{type(error).__name__}: {error}\n\n"
        f"Please re-emit your response. If the error mentions input validation, "
        f"check your tool_call arguments are valid JSON. "
        f"If the error mentions context length, produce a shorter response."
    )


def skipped_tool_feedback(name: str) -> str:
    return f"[Skipped - only one tool per turn. Call {name} again if needed.]"


@dataclass
class Message:
    """A message in the conversation, in the chat-completions shape."""

    role: str
    content: str
    tool_calls: list[dict[str, JsonValue]] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    def as_payload(self) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {"role": self.role, "content": self.content or ""}
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        if self.name:
            payload["name"] = self.name
        if self.tool_calls:
            payload["tool_calls"] = self.tool_calls
        return payload


@dataclass
class Action:
    """The tool call the agent chose for this turn."""

    tool: str
    arguments: dict[str, JsonValue] = field(default_factory=dict)


class Workspace(ABC):
    """The agent's workspace as the loop needs it: the notes file it keeps between weeks."""

    @abstractmethod
    def memory(self) -> str | None:
        """The current contents of ``MEMORY.md``, or ``None`` when the agent has not written one."""


class BashAgent:
    """The benchmark's bash agent: its conversation, its turn loop, and its feedback rules."""

    def __init__(
        self,
        client: openai.OpenAI,
        model: str,
        system_prompt: str,
        tool_descriptions: list[dict],
        workspace: Workspace,
        *,
        reasoning_effort: str | None = "none",
        max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
        logger: logging.Logger | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.tool_descriptions = list(tool_descriptions)
        self.workspace = workspace
        self.reasoning_effort = reasoning_effort
        self.max_completion_tokens = int(max_completion_tokens)
        self.logger = logger or logging.getLogger(__name__)
        self.conversation: list[Message] = []
        self.current_day = 0
        self.turns_today = 0
        self.total_turns = 0
        self.pending_tool_calls: list[dict[str, str]] = []
        self.day_advanced = False
        self.consecutive_errors = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cached_tokens = 0
        self.total_reasoning_tokens = 0
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.last_cached_tokens = 0
        self.last_reasoning_tokens = 0

    def system_prompt_with_memory(self) -> str:
        """The system prompt with ``MEMORY.md`` appended, as the benchmark injects it."""
        prompt = self.system_prompt
        memory = (self.workspace.memory() or "").strip()
        if not memory:
            return prompt
        if len(memory) > MEMORY_MAX_CHARS:
            memory = memory[:MEMORY_MAX_CHARS] + (
                "\n\n--- MEMORY.md TRUNCATED ---\n"
                f"Showing first {MEMORY_MAX_CHARS:,} of {len(memory):,} characters. "
                "Use the read_file tool to see the full contents if needed."
            )
        return prompt + (
            "\n\n## Your MEMORY.md (auto-loaded)\n\n"
            "The following is the contents of your MEMORY.md file. "
            "This is automatically loaded into your context at the start of every day.\n\n"
            f"{memory}"
        )

    def check_day_advanced(self, bash_output: str) -> bool:
        match = DASHBOARD_DAY_RE.search(bash_output)
        if match and int(match.group(1)) > self.current_day:
            self.day_advanced = True
            return True
        return False

    def clear_day_advanced(self) -> None:
        self.day_advanced = False

    def act(self, observation: str, info: dict) -> Action:
        """Take in the last tool's output (or a dashboard) and choose the next tool call."""
        day = int(info.get("day", 0) or 0)
        if day > self.current_day:
            self.conversation = [Message(role="system", content=self.system_prompt_with_memory())]
            self.pending_tool_calls = []
            self.current_day = day
            self.turns_today = 0
        if self.pending_tool_calls:
            for call in self.pending_tool_calls:
                self.conversation.append(
                    Message(role="tool", content=observation, tool_call_id=call["id"], name=call["name"])
                )
            self.pending_tool_calls = []
        else:
            self.conversation.append(Message(role="user", content=observation))
        action = self.call_llm()
        self.turns_today += 1
        return action

    def request(self) -> dict[str, JsonValue]:
        """The chat-completions request for the conversation as it stands."""
        payload: dict[str, JsonValue] = {
            "model": self.model,
            "messages": [message.as_payload() for message in self.conversation],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]},
                }
                for t in self.tool_descriptions
            ],
            "tool_choice": "auto",
            "max_completion_tokens": self.max_completion_tokens,
            "temperature": TEMPERATURE,
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        return payload

    def record_usage(self, response: ChatCompletion) -> None:
        usage = response.usage
        prompt_details = usage.prompt_tokens_details if usage else None
        completion_details = usage.completion_tokens_details if usage else None
        self.last_input_tokens = int((usage.prompt_tokens or 0) if usage else 0)
        self.last_output_tokens = int((usage.completion_tokens or 0) if usage else 0)
        self.last_cached_tokens = int((prompt_details.cached_tokens or 0) if prompt_details else 0)
        self.last_reasoning_tokens = int((completion_details.reasoning_tokens or 0) if completion_details else 0)
        self.total_input_tokens += self.last_input_tokens
        self.total_output_tokens += self.last_output_tokens
        self.total_cached_tokens += self.last_cached_tokens
        self.total_reasoning_tokens += self.last_reasoning_tokens

    def call_llm(self) -> Action:
        """Call the model until it returns one executable tool call."""
        while True:
            try:
                response = self.client.chat.completions.create(**self.request())
            except (openai.APIError, ValueError) as error:
                retryable = isinstance(error, openai.APIStatusError) and (
                    error.status_code >= 500 or error.status_code == 429
                )
                retryable = retryable or isinstance(error, (openai.APIConnectionError, openai.APITimeoutError))
                retryable = retryable or any(code in str(error) for code in RETRYABLE_ERROR_CODES)
                self.consecutive_errors += 1
                if retryable:
                    wait = min(
                        RETRY_BACKOFF_CAP_S,
                        RETRY_BACKOFF_BASE_S * 2 ** min(self.consecutive_errors - 1, RETRY_BACKOFF_STEPS),
                    )
                    self.logger.warning("model call failed (%s); retrying in %ds", error, wait)
                    time.sleep(wait)
                    continue
                wait = min(FEEDBACK_WAIT_CAP_S, FEEDBACK_WAIT_STEP_S * self.consecutive_errors)
                self.logger.warning("model call failed (%s); feeding the error back in %ds", error, wait)
                self.conversation.append(Message(role="user", content=non_retryable_feedback(error)))
                time.sleep(wait)
                continue
            self.total_turns += 1
            self.consecutive_errors = 0
            self.record_usage(response)
            assistant = response.choices[0].message
            calls = list(assistant.tool_calls or [])
            invalid = None
            for call in calls:
                if call.function.arguments:
                    try:
                        json.loads(call.function.arguments)
                    except json.JSONDecodeError as error:
                        invalid = (call.function.name, str(error), call.function.arguments[:300])
                        break
            if invalid is not None:
                self.logger.info("invalid JSON in tool call %s; feeding the error back", invalid[0])
                self.conversation.append(Message(role="user", content=invalid_json_feedback(*invalid)))
                continue
            tool_calls_data = None
            if calls:
                tool_calls_data = []
                for call in calls:
                    entry: dict[str, JsonValue] = {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.function.name, "arguments": call.function.arguments},
                    }
                    extra_content = (call.model_extra or {}).get("extra_content")
                    if extra_content:
                        entry["extra_content"] = extra_content
                    tool_calls_data.append(entry)
            self.conversation.append(
                Message(role="assistant", content=assistant.content or "", tool_calls=tool_calls_data)
            )
            if not calls:
                self.logger.info("no tool call in the response; feeding the reminder back")
                self.conversation.append(Message(role="user", content=NO_TOOL_FEEDBACK))
                continue
            first = calls[0]
            arguments = json.loads(first.function.arguments) if first.function.arguments else {}
            for extra in calls[1:]:
                self.conversation.append(
                    Message(
                        role="tool",
                        content=skipped_tool_feedback(extra.function.name),
                        tool_call_id=extra.id,
                        name=extra.function.name,
                    )
                )
            self.pending_tool_calls = [{"id": first.id, "name": first.function.name}]
            return Action(tool=first.function.name, arguments=arguments)
