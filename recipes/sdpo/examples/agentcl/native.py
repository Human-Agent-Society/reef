"""Opt-in local capture of native SDPO inputs; the objective and batch stay unchanged."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

from recipes.sdpo.processor import SDPOProcessor
from recipes.sdpo.recipe import SDPORecipe
from reef.recipe.base import WeightTrainingSpec
from reef.recipe.config_fields import config_field
from reef.train.types import ProcessorContext, TrainDataItem, TrainingBatch, trajectories

from .report import JsonObject, JsonValue, read_object, write_object


def token_checksum(tokens: Sequence[int]) -> str:
    return hashlib.sha256(json.dumps(list(tokens), separators=(",", ":")).encode()).hexdigest()


def diagnostic_json(value: object) -> JsonValue:
    """Copy native JSON while removing only known credential fields."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        result: JsonObject = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("captured native metadata requires string keys")
            if key.lower().replace("-", "_") in {
                "authorization",
                "proxy_authorization",
                "api_key",
                "apikey",
                "password",
                "credential",
                "credentials",
                "access_token",
                "refresh_token",
                "secret",
                "token",
            }:
                continue
            result[key] = diagnostic_json(item)
        return result
    if isinstance(value, (tuple, list)):
        return [diagnostic_json(item) for item in value]
    raise ValueError("captured native metadata must be JSON-compatible")


class CapturedSDPOProcessor(SDPOProcessor):
    """Save the actual assembled teacher/student tensors before native training."""

    def __init__(self, context: ProcessorContext) -> None:
        output = context.config.get("native_sample_dir", "")
        if not isinstance(output, str):
            raise ValueError("native_sample_dir must be a path string")
        self.sample_dir = Path(output).resolve() if output else None
        self.teacher_inputs: dict[str, JsonObject] = {}
        self.sample_teacher_keys: dict[str, str] = {}
        if self.sample_dir is not None and context.config.get("accept_multi_turn_policy_samples") is not True:
            raise ValueError("native sample capture requires strict whole-episode distillation")
        super().__init__(context)

    def teacher_tokens(
        self,
        messages: Sequence[JsonValue],
        tools: Sequence[JsonValue] | None,
        response_ids: Sequence[int],
        *,
        max_prompt_tokens: int = 0,
        enable_thinking: bool | None = None,
    ) -> list[int]:
        tokens = super().teacher_tokens(
            messages, tools, response_ids, max_prompt_tokens=max_prompt_tokens, enable_thinking=enable_thinking
        )
        if self.sample_dir is not None:
            prefix = tokens[: -len(response_ids)]
            decoded = self.tokenizer.decode(prefix, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            if not isinstance(decoded, str):
                raise ValueError("tokenizer must decode the captured prompt into one string")
            self.teacher_inputs[token_checksum(tokens)] = {
                "messages": cast(list[JsonValue], list(messages)),
                "tools": cast(list[JsonValue], list(tools)) if tools else None,
                "enable_thinking": enable_thinking,
                "prompt_token_count": len(prefix),
                "prompt_text_decoded_from_captured_tokens": decoded,
            }
        return tokens

    def make_batch(self, items: tuple[TrainDataItem, ...], batch_number: int) -> TrainingBatch:
        batch = super().make_batch(items, batch_number)
        if self.sample_dir is None:
            return batch
        self.sample_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.sample_dir.chmod(0o700)
        for sample in trajectories(batch):
            report_id = sample.metadata["report_agent_record_id"]
            if not isinstance(report_id, str) or not report_id or Path(report_id).name != report_id:
                raise ValueError("native sample capture requires a safe terminal report ID")
            training = sample.training
            teacher_ids = list(training["teacher_tokens"])
            teacher_input = self.teacher_inputs[token_checksum(teacher_ids)]
            source_ids = [record["agent_record_id"] for record in sample.metadata["records"]]
            native_batch_sources = list(sample.source_agent_record_ids)
            if native_batch_sources != [*source_ids, report_id]:
                raise ValueError("native sample source order differs from captured receipt order")
            records = cast(list[JsonObject], sample.metadata["records"])
            turns: list[JsonValue] = []
            for record in records:
                payload = cast(JsonObject, record["payload"])
                response = cast(JsonObject, payload.get("response", {}))
                native = cast(JsonObject, response.get("training", payload))
                turns.append(
                    {
                        "receipt": record["agent_record_id"],
                        "tokens": native["tokens"],
                        "loss_mask": native["loss_mask"],
                        "rollout_log_probs": native["rollout_log_probs"],
                        "runtime_load_id": native.get("runtime_load_id"),
                    }
                )
            value: JsonObject = {
                "schema_version": 1,
                "capture_stage": "native processor make_batch before optimizer execution; not a commit claim",
                "batch_id": batch.batch_id,
                "group_id": sample.group_id,
                "trajectory": diagnostic_json(dict(sample.trajectory)),
                "report_id": report_id,
                "references": cast(list[JsonValue], list(sample.metadata["references"])),
                "source_agent_record_ids": cast(list[JsonValue], native_batch_sources),
                "tokens": list(training["tokens"]),
                "loss_mask": list(training["loss_mask"]),
                "rollout_log_probs": list(training["rollout_log_probs"]),
                "teacher_tokens": teacher_ids,
                "distill_sample_weight": float(training.get("distill_sample_weight", 1.0)),
                "runtime_load_id": training["runtime_load_id"],
                "turns": turns,
                "teacher_input": teacher_input,
                "teacher_token_sha256": token_checksum(teacher_ids),
                "student_token_sha256": token_checksum(training["tokens"]),
            }
            value = cast(JsonObject, diagnostic_json(value))
            path = self.sample_dir / f"{report_id}.json"
            if path.exists():
                if read_object(path) != value:
                    raise ValueError("a repeated report produced different captured native training inputs")
            else:
                write_object(path, value)
                path.chmod(0o600)
        self.teacher_inputs.clear()
        return batch


@dataclass(frozen=True, kw_only=True)
class CapturedSDPORecipe(SDPORecipe):
    """Native SDPO with an optional private, local sample directory."""

    native_sample_dir: str = config_field("")

    @classmethod
    def training_spec(cls) -> WeightTrainingSpec:
        return replace(super().training_spec(), processor=CapturedSDPOProcessor)
