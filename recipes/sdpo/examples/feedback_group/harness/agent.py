"""Sample and grade one complete question group through Reef and Harbor."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shlex
from pathlib import Path

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from reef_client import ReefClient

MODEL = "reef"
ROLLOUTS = 8


class HarborAgent(BaseAgent):
    """One Harbor trial collects eight on-policy attempts and grader feedback."""

    @staticmethod
    def name() -> str:
        return "reef-sdpo-feedback-group"

    def version(self) -> str | None:
        return None

    async def setup(self, environment: BaseEnvironment) -> None:
        return None

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        service = os.environ["REEF_SERVICE_URL"]
        scenario = os.environ["REEF_SCENARIO"]
        client = ReefClient(service, token=os.environ["REEF_TOKEN"], timeout_s=1800)
        attempts = []
        receipts = []
        input_tokens = output_tokens = 0
        for index in range(ROLLOUTS):
            response, receipt = await asyncio.to_thread(
                client.inference_with_record,
                scenario,
                "/v1/chat/completions",
                {
                    "model": MODEL,
                    "messages": [{"role": "user", "content": instruction}],
                    "chat_template_kwargs": {"enable_thinking": False},
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "max_tokens": int(os.environ.get("SDPO_MAX_TOKENS", "256")),
                },
            )
            answer = response["choices"][0]["message"]["content"]
            encoded = base64.b64encode(answer.encode("utf-8")).decode("ascii")
            grade = await environment.exec(f"python3 /opt/grade.py {shlex.quote(encoded)}")
            if grade.return_code != 0:
                raise RuntimeError(f"grader failed for attempt {index}: {grade.stderr}")
            verdict = json.loads(grade.stdout)
            # Reef's token-native chat handler exposes the producing version
            # on the choice; the private training block is removed from the
            # client response before inference_with_record returns it.
            choice_meta = response["choices"][0].get("meta_info") or {}
            version = choice_meta.get("runtime_load_id")
            if not version:
                raise RuntimeError("Reef inference did not return a choice runtime version for SDPO grouping")
            attempts.append(
                {
                    "question_id": "feedback-group-arithmetic",
                    "inference_id": receipt,
                    "artifact_version": str(version),
                    "response": answer,
                    "score": float(verdict["score"]),
                    "feedback": verdict["feedback"],
                }
            )
            receipts.append(receipt)
            input_tokens += int(response["usage"]["prompt_tokens"])
            output_tokens += int(response["usage"]["completion_tokens"])
        versions = {item["artifact_version"] for item in attempts}
        if len(versions) != 1:
            raise RuntimeError("the policy changed while this question group was being sampled")
        path = Path(os.environ["SDPO_ATTEMPTS_PATH"])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(attempts, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)
        context.metadata = {**(context.metadata or {}), "reef": {"agent_record_ids": receipts}}
        context.n_input_tokens = input_tokens
        context.n_output_tokens = output_tokens
