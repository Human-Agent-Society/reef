"""Upstream GEPA adapter that evaluates Reef text nodes through Pi episodes."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, TypedDict

from gepa.adapters.default_adapter.default_adapter import ContainsAnswerEvaluator
from gepa.core.adapter import EvaluationBatch

from reef.harness import (
    AdapterDescriptor,
    EpisodeError,
    EpisodeResult,
    TrajectoryError,
    render_composition,
    run_episode,
)
from reef.harness.model_binding import ModelBinding

from .models import TASK_MODEL_PRICE, SpendCap, UsageLedger, trajectory_usage

QUICKSTART_EXTENSION = "pi-agent/extensions/reef-quickstart-system.ts"


class AIMEExample(TypedDict, total=False):
    """The subset of GEPA's AIME example schema that Reef consumes."""

    input: str
    answer: str
    additional_context: dict[str, str]


@dataclass(frozen=True)
class TextComponent:
    """One named GEPA text parameter mapped to one fixed Reef text node."""

    key: str
    kind: str
    name: str | None = None

    def __post_init__(self) -> None:
        if (self.kind == "rules") != (self.name is None):
            raise ValueError("rules nodes are unnamed; skill and agent_command nodes need a name")

    def node(self, text: str) -> tuple[str, dict[str, str]]:
        return self.kind, {"text": text} if self.name is None else {"name": self.name, "text": text}

    @property
    def role(self) -> str:
        return "global rules loaded for every episode" if self.name is None else f"{self.kind} node {self.name!r}"


RULES_ONLY = (TextComponent("rules", "rules"),)
MULTI_NODE = (TextComponent("rules", "rules"), TextComponent("skill", "skill", "aime-solver"))


class ReefAdapter:
    """Expose GEPA text components as one complete rendered Reef harness.

    The topology is fixed when the adapter is built. The task model binding
    is rendered into every throwaway episode but never becomes part of the
    GEPA candidate, so a published candidate carries no endpoint or credential.
    """

    propose_new_texts = None

    def __init__(
        self,
        *,
        descriptor: AdapterDescriptor,
        task_model: ModelBinding,
        components: Sequence[TextComponent] = RULES_ONLY,
        binary: str | None = None,
        timeout_s: float = 600.0,
        max_workers: int = 1,
        episode_runner: Callable[..., EpisodeResult] = run_episode,
        spend_cap: SpendCap | None = None,
        usage_path: Path | None = None,
        task_max_tokens: int | None = None,
    ) -> None:
        if not components or len({component.key for component in components}) != len(components):
            raise ValueError("components need unique keys")
        self.descriptor = descriptor
        self.components = tuple(components)
        self.binary = binary
        self.timeout_s = timeout_s
        self.max_workers = max_workers
        self.episode_runner = episode_runner
        self.spend_cap = spend_cap
        self.usage = UsageLedger(TASK_MODEL_PRICE, usage_path)
        binding_nodes = list(task_model.compose_nodes(descriptor))
        if task_max_tokens is not None:
            models = {"providers": {"reef": {"models": [{"id": task_model.model, "maxTokens": task_max_tokens}]}}}
            binding_nodes.append(("config", {"target": "models", "data": models}))
        self._binding_nodes = tuple(binding_nodes)

    def render_candidate(self, candidate: Mapping[str, str]) -> dict[str, str]:
        """The provider-free tree that Reef publishes."""
        return render_composition(self._nodes(candidate), self.descriptor)

    def render_episode_candidate(self, candidate: Mapping[str, str]) -> dict[str, str]:
        """The same tree plus the transient model binding, for one episode."""
        return render_composition((*self._binding_nodes, *self._nodes(candidate)), self.descriptor)

    def evaluate(
        self,
        batch: list[AIMEExample],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[dict[str, Any], dict[str, Any]]:
        files = self.render_episode_candidate(candidate)
        evaluate = partial(self._evaluate_example, files, capture_traces=capture_traces)
        if self.max_workers == 1 or len(batch) <= 1:
            evaluated = [evaluate(example) for example in batch]
        else:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(batch))) as executor:
                evaluated = list(executor.map(evaluate, batch))
        return EvaluationBatch(
            outputs=[output for output, _, _ in evaluated],
            scores=[score for _, score, _ in evaluated],
            trajectories=[trajectory for _, _, trajectory in evaluated] if capture_traces else None,
            objective_scores=[{"score": score} for _, score, _ in evaluated],
            num_metric_calls=len(batch),
        )

    def _evaluate_example(
        self,
        files: Mapping[str, str],
        example: AIMEExample,
        capture_traces: bool,
    ) -> tuple[dict[str, Any], float, dict[str, Any] | None]:
        task, expected = example["input"], example["answer"]
        if self.spend_cap is not None:
            self.spend_cap.before_call()
        try:
            result = self.episode_runner(self.descriptor, files, task, binary=self.binary, timeout=self.timeout_s)
        except (EpisodeError, TrajectoryError) as exc:
            result = EpisodeResult(exit_code=-1, stdout="", stderr=str(exc), trajectory=(), residue=())
        response = final_assistant_text(result.trajectory) or result.stdout
        clean = result.exit_code == 0 and not result.residue
        output = {
            "assistant_response": response,
            "exit_code": result.exit_code,
            "stderr": result.stderr,
            "residue": list(result.residue),
            "usage": trajectory_usage(result.trajectory),
        }
        if self.spend_cap is not None:
            self.spend_cap.record_call(TASK_MODEL_PRICE.estimate(output["usage"]))
        self.usage.add(output["usage"])
        trajectory = None
        if capture_traces:
            trajectory = {
                **output,
                "input": task,
                "expected_answer": expected,
                "feedback": _feedback(example, response, result),
                "events": [dict(event) for event in result.trajectory],
            }
        return output, score_aime_answer(expected, response) if clean else 0.0, trajectory

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[dict[str, Any], dict[str, Any]],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        roles = {component.key: component.role for component in self.components}
        unknown = set(components_to_update) - set(roles)
        if unknown or not components_to_update:
            raise ValueError(f"reflection components must be a non-empty subset of {sorted(roles)}, not {unknown}")
        if eval_batch.trajectories is None:
            raise ValueError("captured trajectories are required for reflection")
        return {
            key: [
                {
                    "Inputs": trajectory["input"],
                    "Generated Outputs": trajectory["assistant_response"],
                    "Feedback": trajectory["feedback"],
                    "Component role": roles[key],
                    "Harness trajectory": trajectory["events"],
                }
                for trajectory in eval_batch.trajectories
            ]
            for key in components_to_update
        }

    def _nodes(self, candidate: Mapping[str, str]) -> tuple[tuple[str, Mapping[str, Any]], ...]:
        expected = {component.key for component in self.components}
        if set(candidate) != expected:
            raise ValueError(f"candidate components must be exactly {sorted(expected)}")
        return tuple(component.node(candidate[component.key]) for component in self.components)


def score_aime_answer(expected: str, response: str) -> float:
    """The pinned quickstart's exact expected-string containment score."""
    return 1.0 if expected in response else 0.0


def run_quickstart_episode(
    descriptor: AdapterDescriptor,
    files: Mapping[str, str],
    prompt: str,
    *,
    binary: str | None = None,
    timeout: float = 600.0,
) -> EpisodeResult:
    """Run a rules candidate through Pi with the upstream quickstart's request envelope.

    The candidate is still rendered as Reef's rules node; for the rules-only
    conformance arm Pi receives that text as its exact system prompt with its
    coding-agent prompt, tools, skills, and context discovery disabled. An
    extension loaded only for this arm replaces Pi's final system prompt and
    flattens the single text user message to the upstream string shape.
    """
    if descriptor.name != "pi":
        raise ValueError("the quickstart episode runner requires the Pi adapter")
    system_prompt = files.get(descriptor.node_paths["rules"], "").removesuffix("\n")
    if not system_prompt.strip():
        raise ValueError("rendered Pi rules text is missing")
    runtime_files = {
        **files,
        QUICKSTART_EXTENSION: (
            "export default function reefQuickstartSystem(pi) {\n"
            '  pi.on("before_agent_start", async () => ({\n'
            f"    systemPrompt: {json.dumps(system_prompt)},\n"
            "  }));\n"
            '  pi.on("before_provider_request", async ({ payload }) => {\n'
            '    if (!payload || typeof payload !== "object" || !Array.isArray(payload.messages)) return payload;\n'
            "    const messages = payload.messages.map((message) => {\n"
            "      const content = message?.content;\n"
            '      if (Array.isArray(content) && content.length === 1 && content[0]?.type === "text") {\n'
            "        return { ...message, content: content[0].text };\n"
            "      }\n"
            "      return message;\n"
            "    });\n"
            "    return { ...payload, messages };\n"
            "  });\n"
            "}\n"
        ),
    }
    argv = (
        *("--mode", "json", "--print", "--system-prompt", system_prompt),
        *("--no-tools", "--no-skills", "--no-prompt-templates", "--no-themes", "--no-context-files"),
        *("--thinking", "off", "{prompt}"),
    )
    return run_episode(replace(descriptor, argv=argv), runtime_files, prompt, binary=binary, timeout=timeout)


def final_assistant_text(trajectory: Sequence[Mapping[str, Any]]) -> str | None:
    """The final assistant text from Pi's wrapped or flat events."""
    for event in reversed(trajectory):
        wrapped = event.get("message")
        message = wrapped if isinstance(wrapped, Mapping) else event
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = [part["text"] for part in content if isinstance(part, Mapping) and part.get("type") == "text"]
            if texts:
                return "\n".join(texts)
    return None


def _feedback(example: AIMEExample, response: str, result: EpisodeResult) -> str:
    expected = example["answer"]
    if result.exit_code != 0:
        return f"The harness exited with code {result.exit_code} ({result.stderr[-300:]!r}). Expected {expected!r}."
    if result.residue:
        return f"The harness left unexpected residue {list(result.residue)!r}. Expected {expected!r}."
    upstream_example = {**example, "additional_context": example.get("additional_context") or {}}
    return ContainsAnswerEvaluator()(upstream_example, response).feedback
