"""Recorded, append-only student episodes with an isolated stateful code tool."""

from __future__ import annotations

import asyncio
import copy
import json
import math
import re
import sys
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import quote

from reef_client import ReefClient, ReefClientError

from .answer_contract import AnswerContractError, validate_final_module

JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject = dict[str, JsonValue]
Phase = Literal["train", "baseline", "frozen-repeat", "independent"]


class EpisodeFault(RuntimeError):
    """A recorded episode cannot satisfy the receipt, version or tool contract."""


@dataclass(frozen=True)
class EpisodeSettings:
    episode_id: str
    model_name: str
    expected_release: str
    phase: Phase = "train"
    expected_runtime_load_id: str | None = None
    max_turns: int = 8
    max_response_tokens: int = 2048
    max_episode_tokens: int = 8192
    tool_timeout_seconds: int = 30
    max_tool_output_chars: int = 8000
    temperature: float = 0.7
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.episode_id or not self.model_name or not self.expected_release:
            raise ValueError("episode, model and expected release IDs are required")
        if self.phase not in ("train", "baseline", "frozen-repeat", "independent"):
            raise ValueError("unknown AgentCL phase")
        for budget in (
            self.max_turns,
            self.max_response_tokens,
            self.max_episode_tokens,
            self.tool_timeout_seconds,
            self.max_tool_output_chars,
        ):
            if budget <= 0:
                raise ValueError("episode budgets must be positive")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")


class RecordedModel(ABC):
    @abstractmethod
    def complete(self, request: JsonObject, release_id: str) -> tuple[JsonObject, Mapping[str, str]]:
        """Return an unchanged public response and exact Reef response headers."""

    @abstractmethod
    def read_record(self, receipt: str) -> JsonObject:
        """Read the canonical authenticated inference record, including native tensors."""


class ReefRecordedModel(RecordedModel):
    def __init__(self, client: ReefClient, scenario: str, record_timeout_seconds: float = 10) -> None:
        self.client = client
        self.scenario = scenario
        self.record_timeout_seconds = record_timeout_seconds

    def complete(self, request: JsonObject, release_id: str) -> tuple[JsonObject, Mapping[str, str]]:
        # The release request header binds a scenario's initial base, not its current policy.
        return self.client.post("/v1/chat/completions", self.scenario, request)

    def read_record(self, receipt: str) -> JsonObject:
        path = f"/reef/scenarios/{quote(self.scenario, safe='')}/records/{quote(receipt, safe='')}"
        deadline = time.monotonic() + self.record_timeout_seconds
        while True:
            try:
                return self.client.get(path)
            except ReefClientError as error:
                if error.status != 404 or time.monotonic() >= deadline:
                    raise EpisodeFault("canonical inference record is unavailable") from error
                time.sleep(0.05)


class EpisodeSandbox(ABC):
    @abstractmethod
    async def start(self) -> None:
        """Initialize a fresh isolated interpreter before the first action."""

    @abstractmethod
    async def execute(self, code: str, timeout_seconds: int, output_limit: int) -> JsonObject:
        """Execute code in the same sandbox namespace throughout this episode."""

    @abstractmethod
    async def submit(self, code: str) -> None:
        """Write the full Python answer artifact without evaluating it on the host."""

    @abstractmethod
    async def close(self) -> None:
        """Stop owned interpreter processes; Harbor retires the whole environment."""


def native_training(response: JsonObject, required: bool) -> JsonObject:
    """Validate native tensors without reconstructing any sampled tokens."""
    training = response.get("training")
    if not isinstance(training, dict):
        if required:
            raise EpisodeFault("training response is missing native policy tensors")
        return {}
    if not required:
        return training
    tokens = training.get("tokens")
    mask = training.get("loss_mask")
    log_probs = training.get("rollout_log_probs")
    runtime_load_id = training.get("runtime_load_id")
    if not isinstance(runtime_load_id, str) or not runtime_load_id:
        raise EpisodeFault("training response is missing runtime_load_id")
    if not isinstance(tokens, list) or not all(
        isinstance(token, int) and not isinstance(token, bool) for token in tokens
    ):
        raise EpisodeFault("training response has invalid native tokens")
    if not isinstance(mask, list) or not mask or any(value != 1 for value in mask) or len(tokens) <= len(mask):
        raise EpisodeFault("every assistant-generated token must have a native loss mask of one")
    if (
        not isinstance(log_probs, list)
        or len(log_probs) != len(mask)
        or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in log_probs)
    ):
        raise EpisodeFault("training response is missing complete finite native log probabilities")
    return training


def trajectory_record(settings: EpisodeSettings, turns: list[JsonObject], messages: list[JsonObject]) -> JsonObject:
    """Represent every observed message and native token detail as ATIF v1.7."""
    steps: list[JsonValue] = []
    turn_index = 0
    for position, message in enumerate(messages, start=1):
        role = message["role"]
        step: JsonObject = {
            "step_id": position,
            "source": "agent" if role == "assistant" else role,
            "message": message["content"],
        }
        if role == "assistant":
            turn = turns[turn_index]
            turn_index += 1
            response = turn["response"]
            usage = response["usage"]
            metrics: JsonObject = {
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
            }
            record_payload = turn.get("record", {}).get("payload", {})
            training = record_payload.get("response", {}).get("training", {})
            tokens = training.get("tokens", [])
            mask = training.get("loss_mask", [])
            if tokens and mask:
                metrics.update(
                    {
                        "prompt_token_ids": tokens[: -len(mask)],
                        "completion_token_ids": tokens[-len(mask) :],
                        "logprobs": training.get("rollout_log_probs", []),
                    }
                )
            step.update(
                {
                    "llm_call_count": 1,
                    "model_name": settings.model_name,
                    "metrics": metrics,
                    "extra": {
                        "receipt": turn["receipt"],
                        "release_id": turn["release_id"],
                        "runtime_load_id": turn["runtime_load_id"],
                    },
                }
            )
        steps.append(step)
    return {
        "schema_version": "ATIF-v1.7",
        "session_id": settings.episode_id,
        "trajectory_id": settings.episode_id,
        "agent": {"name": "reef-agentcl", "version": "1", "model_name": settings.model_name},
        "steps": steps,
    }


class EpisodeRunner:
    """One episode, one append-only transcript and no training-report submission."""

    def __init__(
        self,
        model: RecordedModel,
        sandbox: EpisodeSandbox,
        settings: EpisodeSettings,
        logs_dir: Path,
        episode_output: Path | None = None,
    ) -> None:
        self.model = model
        self.sandbox = sandbox
        self.settings = settings
        self.logs_dir = logs_dir
        self.episode_output = episode_output
        self.messages: list[JsonObject] = []
        self.turns: list[JsonObject] = []
        self.outcome = "fault"
        self.fault: str | None = None
        self.runtime_load_id: str | None = None
        self.started = 0.0

    def snapshot(self) -> JsonObject:
        settings = self.settings
        return {
            "episode_id": settings.episode_id,
            "phase": settings.phase,
            "outcome": self.outcome,
            "fault": self.fault,
            "release_id": settings.expected_release,
            "runtime_load_id": self.runtime_load_id,
            "references": [turn["receipt"] for turn in self.turns],
            "turns": self.turns,
            "messages": self.messages,
            "prompt_tokens": sum(turn["response"]["usage"]["prompt_tokens"] for turn in self.turns),
            "completion_tokens": sum(turn["response"]["usage"]["completion_tokens"] for turn in self.turns),
            "elapsed_seconds": time.monotonic() - self.started,
            "trajectory": trajectory_record(settings, self.turns, self.messages),
        }

    def persist(self) -> JsonObject:
        snapshot = self.snapshot()
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        destinations = [self.logs_dir / "episode.json"]
        if self.episode_output is not None:
            destinations.append(self.episode_output)
        for destination in destinations:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            temporary.write_text(json.dumps(snapshot, ensure_ascii=False) + "\n", encoding="utf-8")
            temporary.replace(destination)
        (self.logs_dir / "turns.jsonl").write_text(
            "".join(json.dumps(turn, ensure_ascii=False) + "\n" for turn in self.turns), encoding="utf-8"
        )
        (self.logs_dir / "trajectory.json").write_text(
            json.dumps(snapshot["trajectory"], ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return snapshot

    async def run(self, instruction: str) -> JsonObject:
        self.started = time.monotonic()
        self.messages = [{"role": "user", "content": instruction}]
        previous_tokens: list[JsonValue] = []
        submission_reminder = (
            "To submit your completed solution, start your response with the literal uppercase line FINAL, "
            "with no heading or explanation before it.\nUse this exact format:\n"
            "FINAL\n```python\n<your complete Python module>\n```\n"
            "Replace the placeholder with your code. Do not write 'Final solution:'.\n"
            "A code block without uppercase FINAL runs another development action."
        )
        try:
            await self.sandbox.start()
            for turn_number in range(self.settings.max_turns):
                request: JsonObject = {
                    "model": self.settings.model_name,
                    "messages": copy.deepcopy(self.messages),
                    "max_tokens": self.settings.max_response_tokens,
                    "temperature": self.settings.temperature,
                    "seed": self.settings.seed + turn_number,
                    "stream": False,
                }
                response, headers = await asyncio.to_thread(
                    self.model.complete, request, self.settings.expected_release
                )
                receipt = headers.get("x-reef-agent-record-id")
                if not receipt or receipt in [turn["receipt"] for turn in self.turns]:
                    raise EpisodeFault("missing or duplicate inference receipt")
                header_release = headers.get("x-reef-release-id")
                if header_release is not None and header_release != self.settings.expected_release:
                    raise EpisodeFault("episode inference response header reports an unexpected release")
                choices = response.get("choices")
                if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
                    raise EpisodeFault("inference must return exactly one assistant choice")
                choice = choices[0]
                assistant = choice.get("message")
                if (
                    not isinstance(assistant, dict)
                    or assistant.get("role") != "assistant"
                    or not isinstance(assistant.get("content"), str)
                    or assistant.get("tool_calls")
                ):
                    raise EpisodeFault("ordinary code agent requires a text-only assistant message")
                usage = response.get("usage")
                if not isinstance(usage, dict) or any(
                    not isinstance(usage.get(key), int) or usage[key] < 0
                    for key in ("prompt_tokens", "completion_tokens")
                ):
                    raise EpisodeFault("inference is missing exact token usage")
                self.turns.append(
                    {
                        "turn": turn_number,
                        "receipt": receipt,
                        "release_id": None,
                        "runtime_load_id": None,
                        "request": request,
                        "response": copy.deepcopy(response),
                        "response_headers": dict(headers),
                    }
                )
                self.messages.append(copy.deepcopy(assistant))
                self.persist()
                record = await asyncio.to_thread(self.model.read_record, receipt)
                self.turns[-1]["record"] = record
                if record.get("agent_record_id") != receipt or record.get("request_type") != "inference":
                    raise EpisodeFault("canonical record does not identify the expected inference receipt")
                artifact_ref = record.get("artifact_ref")
                if (
                    not isinstance(artifact_ref, dict)
                    or artifact_ref.get("release_id") != self.settings.expected_release
                ):
                    raise EpisodeFault("canonical inference record used an unexpected or unknown release")
                self.turns[-1]["release_id"] = artifact_ref["release_id"]
                recorded_payload = record.get("payload")
                if not isinstance(recorded_payload, dict) or any(
                    recorded_payload.get(key) != value for key, value in request.items()
                ):
                    raise EpisodeFault("canonical inference request differs from the exact submitted request")
                recorded_response = recorded_payload.get("response")
                if not isinstance(recorded_response, dict):
                    raise EpisodeFault("canonical inference record is missing its response")
                public_recorded_response = {
                    key: value for key, value in recorded_response.items() if key != "training"
                }
                if public_recorded_response != response:
                    raise EpisodeFault("canonical inference response differs from the captured public response")
                training = native_training(recorded_response, self.settings.phase == "train")
                runtime_load_id = recorded_payload.get("runtime_load_id") or training.get("runtime_load_id")
                if training.get("runtime_load_id") and runtime_load_id != training["runtime_load_id"]:
                    raise EpisodeFault("canonical runtime_load_id disagrees with native policy tensors")
                if runtime_load_id:
                    expected_runtime = self.settings.expected_runtime_load_id or self.runtime_load_id
                    if expected_runtime is not None and expected_runtime != runtime_load_id:
                        raise EpisodeFault("mixed or unexpected runtime_load_id within episode")
                    self.runtime_load_id = runtime_load_id
                    self.turns[-1]["runtime_load_id"] = runtime_load_id
                if self.settings.phase == "train":
                    tokens, mask = training["tokens"], training["loss_mask"]
                    prompt = tokens[: -len(mask)]
                    if previous_tokens and prompt[: len(previous_tokens)] != previous_tokens:
                        raise EpisodeFault(
                            "native prompt tokens forked or drifted from the full prior assistant history"
                        )
                    previous_tokens = tokens
                self.persist()
                if (
                    choice.get("finish_reason") == "length"
                    or usage["prompt_tokens"] + usage["completion_tokens"] > self.settings.max_episode_tokens
                ):
                    self.outcome = "truncated"
                    self.fault = "student response or episode token window was exhausted"
                    break
                text = assistant["content"]
                blocks = re.findall(r"```python\s*\n(.*?)```", text, flags=re.DOTALL)
                if len(blocks) != 1:
                    self.messages.append(
                        {"role": "user", "content": "Return exactly one fenced python block.\n" + submission_reminder}
                    )
                    continue
                if text.lstrip().startswith("FINAL"):
                    try:
                        validate_final_module(blocks[0])
                    except AnswerContractError as error:
                        self.messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "Final module rejected: " + str(error) + "\n"
                                    "Remove all executable examples and the `if __name__ == '__main__':` guard. "
                                    "Submit only allowed imports and top-level function definitions.\n"
                                    + submission_reminder
                                ),
                            }
                        )
                        continue
                    await self.sandbox.submit(blocks[0])
                    self.outcome = "completed"
                    break
                observation = await self.sandbox.execute(
                    blocks[0], self.settings.tool_timeout_seconds, self.settings.max_tool_output_chars
                )
                self.messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Code execution result:\n"
                            + json.dumps(observation, ensure_ascii=False)
                            + "\n"
                            + submission_reminder
                        ),
                    }
                )
                if observation.get("status") == "timeout":
                    self.outcome = "truncated"
                    self.fault = "student code tool timed out"
                    break
            else:
                self.outcome = "truncated"
                self.fault = "maximum student turns reached without a final submission"
        except (EpisodeFault, ReefClientError, OSError, ValueError) as error:
            self.outcome = "fault"
            self.fault = str(error)
            raise
        finally:
            primary_error = sys.exception()
            if isinstance(primary_error, asyncio.CancelledError):
                self.outcome = "fault"
                self.fault = str(primary_error) or "student episode cancelled"
            try:
                await self.sandbox.close()
            except (OSError, RuntimeError, ValueError, asyncio.CancelledError) as error:
                if primary_error is None:
                    self.outcome = "fault"
                    self.fault = str(error) or "isolated Python kernel cleanup cancelled"
                    raise
                primary_error.add_note(f"isolated Python kernel cleanup also failed: {error}")
            finally:
                self.persist()
        return self.snapshot()
