"""ServeLearnBench's agent loop over the Reef protocol client.

The benchmark's own environment, tools, budgets, and verifier run unchanged;
only the model transport goes through Reef, and every call's inference
receipt is kept for the episode's report.
"""

import uuid
from types import SimpleNamespace

import httpx
from openai import APIStatusError
from openai.types import CompletionUsage
from reef_client import ReefClient, ReefClientError
from servelearnbench import agent, llm
from servelearnbench.engine.types import FINISH_ACTION_NAME, Action, RewardResult, Task


def run_episode(
    model: llm.LLM, bundle: dict, task: Task, *, max_steps: int, extra_system: str = ""
) -> tuple[RewardResult, list[dict], dict, dict]:
    """Run the official environment and agent loop with an explicit turn limit.

    The pinned benchmark's run_episode does not expose run_turns' max_steps.
    This entry point preserves its setup and scoring while passing that limit.
    """
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    llm.usage_reset()
    meta = bundle["meta"]
    system = meta["system_prompt"] + ("\n\n" + extra_system if extra_system else "")
    messages = [{"role": "system", "content": system}, {"role": "user", "content": task.instruction}]
    env = agent.new_env(bundle)
    env.reset(task)
    specs = agent.tool_specs(env.tools_info, meta["domain"])

    def _execute(name: str, arguments: dict) -> str:
        if name not in env.tools_map:
            raise agent.UnknownTool(agent.MSG_UNKNOWN.format(name=name))
        return env.step(Action(name=name, kwargs=arguments)).observation

    trace = agent.run_turns(
        model,
        messages,
        specs,
        _execute,
        on_finish=lambda arguments: env.step(Action(name=FINISH_ACTION_NAME, kwargs=arguments)),
        max_steps=max_steps,
    )
    return env.compute_reward(), trace, llm.agent_usage(), llm.judge_usage()


class ReefLLM(llm.LLM):
    """An official LLM interface backed by one Reef scenario.

    Each instance has a nonempty scenario, one ModelConfig, and ordered,
    distinct receipts for the current episode. begin_episode resets receipts.
    complete appends one receipt per successful, nonstreaming model call.
    The official agent owns conversation history, tools, and termination.
    """

    def __init__(self, client: ReefClient, scenario: str, cfg: llm.ModelConfig, seed: int) -> None:
        if not scenario:
            raise ValueError("scenario must be nonempty")
        self.reef = client
        self.scenario = scenario
        self.cfg = cfg
        self.seed = seed
        self.receipts: list[str] = []
        self.episode_id = uuid.uuid4().hex

    def begin_episode(self) -> None:
        self.receipts.clear()
        self.episode_id = uuid.uuid4().hex

    def complete(
        self,
        messages: list,
        *,
        tools: list | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        judge: bool = False,
    ) -> SimpleNamespace:
        if judge:
            raise ValueError("Retail uses the official rule-based verifier, not an LLM judge")
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": False,
            "temperature": 1.0 if temperature is None else temperature,
            "top_p": 1.0 if top_p is None else top_p,
            "seed": self.seed,
            **self.cfg.extra_body,
            "user": self.episode_id,
        }
        if self.cfg.max_tokens is not None:
            payload["max_tokens"] = self.cfg.max_tokens
        if self.cfg.reasoning_effort:
            payload["reasoning_effort"] = self.cfg.reasoning_effort
        if tools:
            payload.update(tools=tools, parallel_tool_calls=False)
        try:
            response, receipt = self.reef.inference_with_record(self.scenario, "/v1/chat/completions", payload)
        except ReefClientError as error:
            # Preserve the official agent's distinction between context-limit
            # HTTP errors (scored endings) and infrastructure failures (abort).
            response = httpx.Response(error.status, request=httpx.Request("POST", self.reef.service_url))
            raise APIStatusError(error.body, response=response, body=error.body) from error
        if receipt in self.receipts:
            raise RuntimeError(f"Duplicate inference receipt: {receipt}")
        self.receipts.append(receipt)
        # This private accounting hook is the pinned benchmark's only usage
        # writer. Its agent uses the account to enforce the episode token cap.
        usage = response.get("usage")
        if usage is None:
            raise RuntimeError("Inference response has no usage; cannot enforce the official token budget")
        llm._record_usage(CompletionUsage.model_validate(usage), judge=False)
        choice = response["choices"][0]
        message = choice["message"]
        # Reef's tool parser can label a length-limited native completion as
        # tool_calls. The official loop must still reject a truncated call.
        native_finish = choice.get("meta_info", {}).get("finish_reason", {})
        finish_reason = choice["finish_reason"]
        if isinstance(native_finish, dict) and native_finish.get("type") == "length":
            finish_reason = "length"
        calls = [
            {"id": call["id"], "name": call["function"]["name"], "arguments": call["function"]["arguments"]}
            for call in message.get("tool_calls") or []
        ]
        return SimpleNamespace(
            content=message.get("content") or "",
            reasoning=message.get("reasoning_content") or message.get("reasoning"),
            tool_calls=calls,
            finish_reason=finish_reason,
        )
