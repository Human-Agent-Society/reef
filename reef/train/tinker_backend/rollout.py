"""Tinker's answers to a task job's rollouts: each model call sampled on Tinker at the job's start weights.

The rollouts themselves run in ``reef.train.task_rollouts``; ``SamplerChat`` answers their calls on the
sampler of the job's start, so the rollouts sample the weights that the job trains.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reef.runtime.interfaces import UpstreamStatusError
from reef.tinker.sampling import TinkerSampler, sample_chat
from reef.train.task_rollouts import CHAT_COMPLETIONS_PATH, RolloutChat


class SamplerChat(RolloutChat):
    """Answer OpenAI text chat calls on one Tinker sampler, for the rollouts of one job.

    ``sampler_path`` is the job's start sampler, or None for the base model. Each answer carries the training
    capture with ``runtime_load_id``. Another path gets 404, a sampler refusal its own status, and a sampler
    failure (a malformed sample, a timeout) 502 with the reason.
    """

    def __init__(self, sampler: TinkerSampler, sampler_path: str | None, *, model: str, runtime_load_id: str) -> None:
        self.sampler = sampler
        self.sampler_path = sampler_path
        self.model = model
        self.runtime_load_id = runtime_load_id

    def answer(self, path: str, payload: Any) -> tuple[int, Mapping[str, Any]]:
        if path != CHAT_COMPLETIONS_PATH:
            return 404, {"error": {"message": f"no route {path}"}}
        try:
            return 200, sample_chat(
                self.sampler, self.sampler_path, payload, model=self.model, runtime_load_id=self.runtime_load_id
            )
        except UpstreamStatusError as exc:
            return exc.status, {"error": {"message": str(exc)}}
        except (ValueError, TimeoutError) as exc:
            return 502, {"error": {"message": f"the Tinker sampler failed: {exc}"}}
