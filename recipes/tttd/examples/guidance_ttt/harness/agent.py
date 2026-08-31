"""Reef harness for summary-only Guidance-TTT.

The trainable Qwen policy emits only guidance. An external model turns that
guidance into executable code, and the verifier reward is linked back to the
exact guidance receipt recorded by Reef.
"""

from __future__ import annotations

import hashlib
import math
import shutil
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from examples.reef_client import ReefClient
from examples.tttd.harness import TTTDChatRequestBuilder, openai_action

from .execution import ExecutionClient
from .library import GuidanceLibrary
from .prompts import (
    Prompt,
    build_execution_prompt,
    build_guidance_prompt,
    extract_strict_guidance,
    extract_terminal_tag_or_none,
)
from .state import LibraryEntry, LibraryNode, LLMRequest, VerificationResult
from .tasks import POLYOMINO_TASK, TaskSpec


@dataclass(frozen=True)
class GuidanceRolloutResult:
    step: int
    group: int
    rollout: int
    parent_node_id: str
    agent_data_id: str
    guidance: str | None
    guidance_format_ok: bool
    execution_text: str
    verification: VerificationResult
    entry_id: str
    child_node_id: str | None

    @property
    def reward(self) -> float:
        return float(self.verification.reward)


@dataclass(frozen=True)
class _GroupContext:
    group: int
    group_uid: str
    parent: LibraryNode
    parent_entry: LibraryEntry
    guidance_prompt: Prompt


def prepare_library(
    *,
    seed_path: str | Path,
    run_path: str | Path,
    groups_per_step: int,
    rollouts_per_group: int,
    puct_c: float = 1.0,
    max_buffer_size: int = 1_000,
    topk_children: int = 2,
    task: TaskSpec = POLYOMINO_TASK,
) -> GuidanceLibrary:
    """Create or validate a Discover-compatible run archive from one seed."""
    seed_path = Path(seed_path)
    run_path = Path(run_path)
    if groups_per_step < 1 or rollouts_per_group < 2:
        raise ValueError("groups_per_step must be positive and rollouts_per_group must be at least two")
    if not run_path.exists():
        if not seed_path.is_file():
            raise FileNotFoundError(f"Guidance-TTT seed library is missing: {seed_path}")
        run_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(seed_path, run_path)
        library = GuidanceLibrary(run_path)
        library.ensure_pristine_root_count(groups_per_step)
        library.configure_pristine_archive(
            rollout_n=rollouts_per_group,
            puct_c=puct_c,
            puct_q_mode="best_child",
            max_buffer_size=max_buffer_size,
            topk_children=topk_children,
            discover_compat=True,
            groups_per_batch=groups_per_step,
            score_direction=task.score_direction,
        )
    else:
        library = GuidanceLibrary(run_path)
    _assert_task_identity(library, task)
    library.assert_runtime_config(
        rollout_n=rollouts_per_group,
        puct_c=puct_c,
        puct_q_mode="best_child",
        max_buffer_size=max_buffer_size,
        topk_children=topk_children,
        discover_compat=True,
        groups_per_batch=groups_per_step,
        score_direction=task.score_direction,
    )
    return library


def _assert_task_identity(library: GuidanceLibrary, task: TaskSpec) -> None:
    snapshot = library.snapshot()
    problem_ids = {
        str(item.get("problem_id"))
        for collection in (snapshot.get("nodes", {}), snapshot.get("entries", {}))
        for item in collection.values()
        if item.get("problem_id") is not None
    }
    if problem_ids != {task.task_id}:
        raise ValueError(
            f"Guidance library task mismatch: archive={sorted(problem_ids)!r}, requested={task.task_id!r}"
        )


class ReefGuidanceTTTHarness:
    """One policy call and, when well formed, one external execution per rollout."""

    def __init__(
        self,
        client: ReefClient,
        execution_client: ExecutionClient,
        library: GuidanceLibrary,
        *,
        scenario: str,
        model: str,
        task: TaskSpec = POLYOMINO_TASK,
        recipe: str = "tttd",
        artifact_version: str | None = None,
        inference_path: str = "/v1/chat/completions",
        groups_per_step: int = 8,
        rollouts_per_group: int = 16,
        guidance_max_tokens: int = 8_192,
        guidance_temperature: float = 1.0,
        guidance_top_p: float = 1.0,
        guidance_top_k: int = -1,
        sampling_seed: int | None = None,
        max_workers: int | None = None,
        verifier_timeout_s: int = 340,
        verifier_concurrency: int | None = None,
        verifier_config: Mapping[str, Any] | None = None,
        request_extra: Mapping[str, Any] | None = None,
        report_training: bool = True,
    ) -> None:
        if groups_per_step < 1 or rollouts_per_group < 2:
            raise ValueError("groups_per_step must be positive and rollouts_per_group must be at least two")
        if guidance_max_tokens < 1 or verifier_timeout_s < 1:
            raise ValueError("guidance_max_tokens and verifier_timeout_s must be positive")
        if sampling_seed is not None and int(sampling_seed) < 0:
            raise ValueError("sampling_seed must be non-negative")
        if verifier_concurrency is not None and verifier_concurrency < 1:
            raise ValueError("verifier_concurrency must be positive")
        if library.groups_per_batch != groups_per_step or library.rollout_n != rollouts_per_group:
            raise ValueError("library cardinalities do not match the harness")
        self.client = client
        self.execution_client = execution_client
        self.library = library
        self.scenario = scenario
        self.model = model
        self.task = task
        self.recipe = recipe
        self.artifact_version = artifact_version
        self.inference_path = inference_path
        self.groups_per_step = groups_per_step
        self.rollouts_per_group = rollouts_per_group
        self.guidance_max_tokens = guidance_max_tokens
        self.guidance_temperature = float(guidance_temperature)
        self.guidance_top_p = float(guidance_top_p)
        self.guidance_top_k = int(guidance_top_k)
        self.sampling_seed = None if sampling_seed is None else int(sampling_seed)
        self.max_workers = max_workers or min(groups_per_step * rollouts_per_group, 32)
        self.verifier_timeout_s = verifier_timeout_s
        self.verifier_concurrency = verifier_concurrency or self.max_workers
        self.verifier_config = dict(verifier_config or {})
        self.request_extra = dict(request_extra or {})
        self.report_training = bool(report_training)
        self._request_builder = TTTDChatRequestBuilder(
            max_new_tokens=guidance_max_tokens,
            temperature=self.guidance_temperature,
            top_p=self.guidance_top_p,
            top_k=self.guidance_top_k,
            enable_thinking=True,
            lora_path="reef_lora",
        )
        self._execution_slots = threading.BoundedSemaphore(execution_client.backend.concurrency)
        self._verifier_slots = threading.BoundedSemaphore(self.verifier_concurrency)

    def run_step(self, step: int) -> tuple[GuidanceRolloutResult, ...]:
        if step < 0:
            raise ValueError("step must be non-negative")
        archive_timestep = step + 1
        groups = tuple(self._prepare_group(group, archive_timestep) for group in range(self.groups_per_step))
        tasks = [(context, rollout) for context in groups for rollout in range(self.rollouts_per_group)]
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(tasks))) as pool:
            futures = [
                pool.submit(self._rollout, context, step, archive_timestep, rollout) for context, rollout in tasks
            ]
            results = tuple(future.result() for future in futures)
        if len(results) != self.groups_per_step * self.rollouts_per_group:
            raise RuntimeError("Guidance-TTT step returned the wrong rollout count")
        self.library.assert_runtime_config(
            rollout_n=self.rollouts_per_group,
            puct_c=self.library.puct_c,
            puct_q_mode="best_child",
            max_buffer_size=self.library.max_buffer_size,
            topk_children=self.library.topk_children,
            discover_compat=True,
            groups_per_batch=self.groups_per_step,
            score_direction=self.task.score_direction,
        )
        return results

    def _prepare_group(self, group: int, archive_timestep: int) -> _GroupContext:
        group_uid = f"{archive_timestep}:{group}"
        parent = self.library.acquire_group(
            group_uid,
            visible_timestep_exclusive=archive_timestep,
            require_solution=True,
        )
        context = self.library.context_for_node(parent, visible_timestep_exclusive=archive_timestep)
        parent_entry = context["selected_entry"]
        if parent_entry is None or not parent_entry.solution.strip():
            raise RuntimeError(f"selected node {parent.id!r} has no visible executable parent entry")
        prompt = build_guidance_prompt(
            problem_prompt=self.task.problem_prompt,
            selected_node=parent,
            selected_entry=parent_entry,
            objective_text=self.task.guidance_objective(parent.raw_score),
            mechanism_constraint=self.task.guidance_mechanism_constraint,
            raw_score_label=self.task.raw_score_label,
        )
        return _GroupContext(group, group_uid, parent, parent_entry, prompt)

    def _rollout(
        self,
        context: _GroupContext,
        step: int,
        archive_timestep: int,
        rollout: int,
    ) -> GuidanceRolloutResult:
        request_extra = dict(self.request_extra)
        policy_sampling_seed = None
        if self.sampling_seed is not None:
            policy_sampling_seed = (
                self.sampling_seed
                + step * self.groups_per_step * self.rollouts_per_group
                + context.group * self.rollouts_per_group
                + rollout
            )
            request_extra["seed"] = policy_sampling_seed
        payload = self._request_builder(
            self.model,
            [
                {"role": "system", "content": context.guidance_prompt.system},
                {"role": "user", "content": context.guidance_prompt.user},
            ],
            request_extra,
        )
        response, agent_data_id = self.client.inference_with_record(
            self.scenario,
            self.inference_path,
            payload,
            recipe=self.recipe,
            artifact_version=self.artifact_version,
        )
        guidance_text = openai_action(response)
        guidance, guidance_error = extract_strict_guidance(guidance_text)
        execution_text = ""
        execution_reasoning = ""
        execution_metadata: dict[str, Any] = {}
        execution_usage: dict[str, Any] = {}
        if guidance is None:
            verification = VerificationResult(
                reward=0.0,
                raw_score=None,
                valid=False,
                status="guidance_format_error",
                message=guidance_error or "invalid guidance format",
                artifacts={},
            )
        else:
            execution_prompt = build_execution_prompt(
                problem_prompt=self.task.problem_prompt,
                selected_entry=context.parent_entry,
                guidance=guidance,
                solution_language=self.task.solution_language,
                solution_contract=self.task.execution_solution_contract,
                score_direction=self.task.score_direction,
                raw_score_label=self.task.raw_score_label,
            )
            try:
                with self._execution_slots:
                    execution_response = self.execution_client.complete(
                        LLMRequest(
                            system=execution_prompt.system,
                            user=execution_prompt.user,
                            model=self.execution_client.backend.model,
                            temperature=self.execution_client.backend.temperature,
                            max_tokens=self.execution_client.backend.max_tokens,
                            metadata={"purpose": "execution"},
                        )
                    )
                execution_text = execution_response.text
                execution_reasoning = execution_response.reasoning
                execution_metadata = {
                    **execution_response.metadata,
                    "finish_reason": execution_response.finish_reason,
                }
                execution_usage = execution_response.usage
                with self._verifier_slots:
                    verification = self.task.verify_execution_text(
                        execution_text,
                        timeout_s=self.verifier_timeout_s,
                        config=self.verifier_config,
                    )
            except Exception as exc:  # External execution is an environment outcome.
                verification = VerificationResult.execution_error(f"{type(exc).__name__}: {exc}")

        solution = self.task.solution_extractor(execution_text) or ""
        model_summary = extract_terminal_tag_or_none(execution_text, "summary")
        summary = model_summary or "The execution model did not emit a usable canonical summary."
        entry = LibraryEntry(
            id=str(uuid4()),
            parent_id=context.parent.id,
            problem_id=context.parent.problem_id,
            timestep=archive_timestep,
            guidance=guidance or "",
            execution_thinking=execution_reasoning,
            solution=solution,
            verifier_reward=float(verification.reward),
            verifier_raw_score=verification.raw_score,
            verifier_status=verification.status,
            verifier_message=verification.message,
            summary=summary,
            reusable_idea=summary,
            failure_mode=None if verification.valid else verification.status,
            metadata={
                "group_uid": context.group_uid,
                "selected_node_id": context.parent.id,
                "selected_parent_entry_id": context.parent_entry.id,
                "selected_parent_solution_sha256": hashlib.sha256(
                    context.parent_entry.solution.strip().encode()
                ).hexdigest(),
                "guidance_format_ok": guidance is not None,
                "guidance_generation_attempts": 1,
                "raw_guidance_text": guidance_text,
                "guidance_prompt": {
                    "system": context.guidance_prompt.system,
                    "user": context.guidance_prompt.user,
                },
                "execution_text": execution_text,
                "raw_model_summary": model_summary,
                "prompt_mode": "summary_only",
                "summary_semantics": "canonical_full_candidate",
                "execution_backend": self.execution_client.backend.safe_dict(),
                "execution_response_metadata": execution_metadata,
                "execution_response_usage": execution_usage,
                "verification_artifacts": verification.artifacts,
                "training_reported": self.report_training,
                "policy_sampling_seed": policy_sampling_seed,
            },
        )
        child = self.library.submit_child(context.group_uid, entry)
        comparison_set = f"tttd-step-{step}-group-{context.group}"
        score = float(verification.reward)
        if not math.isfinite(score):
            raise RuntimeError(f"verifier returned a non-finite reward: {score!r}")
        if self.report_training:
            self.client.report(
                self.scenario,
                {
                    "score": score,
                    "feedback": f"{verification.status}: {verification.message}",
                    "references": [agent_data_id],
                    "metadata": {
                        "comparison_set": comparison_set,
                        "algorithm": "ttt-discover",
                        "task": self.task.task_id,
                        "step": step,
                        "group": context.group,
                        "rollout": rollout,
                        "groups_per_step": self.groups_per_step,
                        "rollouts_per_group": self.rollouts_per_group,
                        "parent_id": context.parent.id,
                        "search_value": verification.raw_score,
                        "guidance_ttt": True,
                        "prompt_mode": "summary_only",
                        "archive_timestep": archive_timestep,
                        "library_entry_id": entry.id,
                        "policy_sampling_seed": policy_sampling_seed,
                    },
                },
                recipe=self.recipe,
                artifact_version=self.artifact_version,
            )
        return GuidanceRolloutResult(
            step=step,
            group=context.group,
            rollout=rollout,
            parent_node_id=context.parent.id,
            agent_data_id=agent_data_id,
            guidance=guidance,
            guidance_format_ok=guidance is not None,
            execution_text=execution_text,
            verification=verification,
            entry_id=entry.id,
            child_node_id=None if child is None else child.id,
        )


def openai_response_text(response: Mapping[str, Any]) -> str:
    """Compatibility wrapper kept public for small external harness tests."""
    choices = response.get("choices")
    if not isinstance(choices, Sequence) or not choices:
        raise ValueError("response has no choices")
    return openai_action(response)
