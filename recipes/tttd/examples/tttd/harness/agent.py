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
        release_id: str | None = None,
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
        self.release_id = release_id
        self.inference_path = inference_path

    def _rollout(
        self,
        parent: Candidate,
        comparison_set: str,
        step: int,
        group_index: int,
        rollout_index: int,
    ) -> RolloutResult:
        release_headers = {} if self.release_id is None else {"x-reef-release-id": self.release_id}
        response, agent_record_id = self.client.inference_with_record(
            self.scenario,
            self.inference_path,
            self._request_payload(parent),
            extra_headers=release_headers,
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
            extra_headers=release_headers,
        )
        return result
