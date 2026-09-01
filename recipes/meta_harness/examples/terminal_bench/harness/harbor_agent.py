"""Harbor BaseAgent that evaluates a rendered Reef composition through Pi."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from reef.harness.adapters import get_adapter
from reef.harness.model_binding import ModelBinding
from reef.harness.render import render_composition

from .composition import CompositionCandidate
from .config import PI_VERSION
from .pi_bridge import HarborExecBridge, PiEpisodeRunner


class HarborAgent(BaseAgent):
    """Keep Harbor's verifier/container fixed while Pi supplies the harness loop."""

    def __init__(
        self,
        *args: Any,
        composition_path: str,
        target_base_url: str,
        target_api_key_env: str,
        pi_binary: str,
        pi_timeout_s: float = 1800.0,
        pi_runner: PiEpisodeRunner | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._composition_path = Path(composition_path).resolve()
        self._target_base_url = target_base_url
        self._target_api_key_env = target_api_key_env
        self._pi_binary = Path(pi_binary).resolve()
        self._pi_timeout_s = float(pi_timeout_s)
        self._pi_runner = pi_runner

    @staticmethod
    def name() -> str:
        return "reef-meta-harness-pi"

    def version(self) -> str | None:
        return PI_VERSION

    async def setup(self, environment: BaseEnvironment) -> None:
        """Pi runs on the trusted host; Harbor owns the task container."""

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        api_key = os.environ.get(self._target_api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"target credential environment {self._target_api_key_env!r} is empty")
        candidate = CompositionCandidate.from_path(self._composition_path)
        descriptor = get_adapter("pi")
        binding = ModelBinding(
            base_url=self._target_base_url,
            model=self.model_name or "",
            api_key=api_key,
            api="openai",
            timeout_s=self._pi_timeout_s,
        )
        files = render_composition((*candidate.nodes, *binding.compose_nodes(descriptor)), descriptor)
        runner = self._pi_runner or PiEpisodeRunner(
            descriptor,
            binary=self._pi_binary,
            timeout_s=self._pi_timeout_s,
        )
        loop = asyncio.get_running_loop()
        with HarborExecBridge(environment.exec, loop) as bridge:
            result = await asyncio.to_thread(
                runner.run,
                files,
                instruction,
                bridge_url=bridge.url,
                bridge_token=bridge.token,
            )

        self.logs_dir.mkdir(parents=True, exist_ok=True)
        trajectory_path = self.logs_dir / "pi-trajectory.json"
        trajectory_path.write_text(
            json.dumps(list(result.trajectory), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        usage = result.usage
        cached = int(usage.get("cached_input_tokens", 0))
        context.n_input_tokens = int(usage.get("input_tokens", 0)) + cached
        context.n_cache_tokens = cached
        context.n_output_tokens = int(usage.get("output_tokens", 0))
        context.cost_usd = result.estimated_cost_usd
        context.metadata = {
            **(context.metadata or {}),
            "reef_meta_harness": {
                "pi_version": PI_VERSION,
                "pi_exit_code": result.exit_code,
                "pi_stderr": result.stderr.replace(api_key, "[REDACTED]")[-4000:],
                "provider_reported_cost_usd": result.provider_reported_cost_usd,
                "trajectory_path": str(trajectory_path),
            },
        }
