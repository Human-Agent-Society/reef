"""Reef harness subclass for TTT-Discover.

``ReefTTTDiscoverHarness`` extends the generic PUCT harness to route
inference through a Reef scenario and report each rollout's reward against
its exact inference receipt.

The Harbor ``BaseAgent`` subclass lives in ``harbor_agent.py`` so this module
stays importable without the external ``harbor`` package (e.g. in tests).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from urllib.parse import quote

from reef_client import ReefClient

from .search import Candidate, RolloutResult, Scorer, _TTTDiscoverHarnessBase


class ReefTTTDiscoverHarness(_TTTDiscoverHarnessBase):
    """TTTD with scenario inference and linked reward reports."""

    def __init__(
        self,
        client: ReefClient,
        scorer: Scorer,
        instruction: str,
        *,
        scenario: str,
        model: str,
        artifact_id: str | None = None,
        inference_path: str = "/v1/chat/completions",
        groups_per_step: int = 8,
        rollouts_per_group: int = 64,
        exploration: float = 1.0,
        invalid_reward: float = 0.0,
        max_workers: int = 32,
        request_builder: Callable[[str, Sequence[Mapping[str, Any]], Mapping[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(
            scorer,
            instruction,
            model=model,
            groups_per_step=groups_per_step,
            rollouts_per_group=rollouts_per_group,
            exploration=exploration,
            invalid_reward=invalid_reward,
            max_workers=max_workers,
            request_builder=request_builder,
        )
        self.client = client
        self.scenario = scenario
        if artifact_id is not None:
            if not isinstance(artifact_id, str) or not artifact_id:
                raise ValueError("artifact_id must be a non-empty string")
            if not inference_path.startswith("/v1/"):
                raise ValueError("an artifact inference_path must start with /v1/")
            self.inference_path = (
                f"/reef/scenarios/{quote(scenario, safe='')}/artifacts/{quote(artifact_id, safe='')}{inference_path}"
            )
        else:
            self.inference_path = inference_path

    def _rollout(
        self,
        parent: Candidate,
        comparison_set: str,
        step: int,
        group_index: int,
        rollout_index: int,
    ) -> RolloutResult:
        response, agent_record_id = self.client.inference_with_record(
            self.scenario,
            self.inference_path,
            self._request_payload(parent),
        )

        result = dataclasses.replace(
            self._evaluate_response(parent, response),
            agent_record_id=agent_record_id,
        )

        self.client.report(
            self.scenario,
            {
                "score": result.reward,
                "feedback": result.error,
                "references": [agent_record_id],
                "metadata": {
                    "comparison_set": comparison_set,
                    "algorithm": "ttt-discover",
                    "step": step,
                    "group": group_index,
                    "rollout": rollout_index,
                    "groups_per_step": self.groups_per_step,
                    "rollouts_per_group": self.rollouts_per_group,
                    "parent_id": parent.candidate_id,
                    "search_value": result.search_value,
                },
            },
        )
        return result
