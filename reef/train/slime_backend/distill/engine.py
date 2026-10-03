"""Score recorded response tokens with an independent, frozen SGLang teacher.

The bridge fills the existing distillation columns before partitioning or
starting a training step. No generation, retokenization or actor weight swap
is involved. This module deliberately has no torch or serving-runtime imports.
"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.request
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any

from reef.train.slime_backend.distill.algorithm import DistillSettings

ENGINE_COLUMNS = (
    "distill_teacher_topk_ids",
    "distill_teacher_topk_log_probs",
    "distill_teacher_sampled_log_probs",
)


def validate_teacher_tokenizer(student_path: str, teacher_path: str) -> None:
    """Reject different token mappings before accepting a training batch."""
    from transformers import AutoTokenizer

    student = AutoTokenizer.from_pretrained(student_path, trust_remote_code=False)
    teacher = AutoTokenizer.from_pretrained(teacher_path, trust_remote_code=False)
    if student.get_vocab() != teacher.get_vocab():
        raise ValueError("independent distillation teacher must use the student's token-to-ID vocabulary")
    # The teacher reads IDs, never the chat template. Different instruction
    # templates are expected, but the underlying token bytes must agree.
    student_backend = getattr(student, "backend_tokenizer", None)
    teacher_backend = getattr(teacher, "backend_tokenizer", None)
    if student_backend is None or teacher_backend is None:
        raise ValueError("independent teacher validation requires fast tokenizers")
    student_decoder = student_backend.decoder
    teacher_decoder = teacher_backend.decoder
    if student_decoder is None or teacher_decoder is None:
        raise ValueError("independent teacher validation requires token decoders")
    if student_decoder.__getstate__() != teacher_decoder.__getstate__():
        raise ValueError("independent distillation teacher must use the student's token decoder")


def _entry(value: Any) -> tuple[float, int]:
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError("teacher logprob entry must contain a probability and token ID")
    probability, token = value[:2]
    if (
        not isinstance(probability, Real)
        or isinstance(probability, bool)
        or not math.isfinite(probability)
        or float(probability) > 1e-5
        or not isinstance(token, Integral)
        or isinstance(token, bool)
        or token < 0
    ):
        raise ValueError("teacher returned an invalid log probability or token ID")
    return float(probability), int(token)


def parse_teacher_scores(result: Any, tokens: Sequence[int], response_length: int, top_k: int) -> dict[str, Any]:
    """Validate SGLang's prefill alignment, including its leading unscored token.

    We request from the last prompt token. SGLang returns that token with
    probability None, then exactly R scored response tokens. Requiring the
    IDs and row counts prevents silently shifting or truncating supervision.
    """
    if not isinstance(result, Mapping) or not isinstance(result.get("meta_info"), Mapping):
        raise ValueError("teacher response is missing meta_info")
    meta = result["meta_info"]
    if meta.get("completion_tokens") != 0 or result.get("output_ids", []):
        raise ValueError("teacher scoring must not generate tokens")
    sampled = meta.get("input_token_logprobs")
    top_rows = meta.get("input_top_logprobs")
    if not isinstance(sampled, list) or not isinstance(top_rows, list):
        raise ValueError("teacher response is missing input probability rows")
    if len(sampled) != response_length + 1 or len(top_rows) != response_length + 1:
        raise ValueError("teacher probability row count does not match the recorded response")
    first = sampled[0]
    if (
        not isinstance(first, list)
        or len(first) < 2
        or first[0] is not None
        or first[1] != tokens[-response_length - 1]
        or top_rows[0] is not None
    ):
        raise ValueError("teacher response has an invalid leading prompt-token row")
    selected_ids, selected_probs, sampled_probs = [], [], []
    for token, sample, entries in zip(tokens[-response_length:], sampled[1:], top_rows[1:], strict=True):
        probability, actual_token = _entry(sample)
        if actual_token != token:
            raise ValueError("teacher probability token IDs do not match the recorded response")
        if not isinstance(entries, list) or len(entries) != top_k:
            raise ValueError("teacher top-K row has the wrong number of entries")
        parsed = [_entry(entry) for entry in entries]
        ids = [entry[1] for entry in parsed]
        if len(set(ids)) != top_k:
            raise ValueError("teacher top-K row contains duplicate token IDs")
        selected_ids.append(ids)
        selected_probs.append([entry[0] for entry in parsed])
        sampled_probs.append(probability)
    return dict(zip(ENGINE_COLUMNS, (selected_ids, selected_probs, sampled_probs), strict=True))


class EngineTeacher:
    """A synchronous, bounded scoring client owned by one training bridge."""

    def __init__(self, settings: DistillSettings) -> None:
        self.settings = settings

    def score(self, tokens: list[int], response_length: int) -> dict[str, Any]:
        """Score exact IDs with a prefill-only request; HTTP errors fail the batch."""
        if not 0 < response_length < len(tokens):
            raise ValueError("teacher scoring needs a nonempty prompt and response")
        body = {
            "input_ids": tokens,
            "sampling_params": {"max_new_tokens": 0, "temperature": 1.0},
            "return_logprob": True,
            "logprob_start_len": len(tokens) - response_length - 1,
            "top_logprobs_num": self.settings.top_k,
        }
        request = urllib.request.Request(
            self.settings.teacher_url.rstrip("/") + "/generate",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=self.settings.teacher_timeout) as response:
            result = json.load(response)
        return parse_teacher_scores(result, tokens, response_length, self.settings.top_k)

    def validate_model(self) -> None:
        """Refuse an endpoint serving a different model than the validated tokenizer."""
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(
            self.settings.teacher_url.rstrip("/") + "/get_model_info", timeout=self.settings.teacher_timeout
        ) as response:
            info = json.load(response)
        actual = info.get("model_path") if isinstance(info, Mapping) else None
        if not isinstance(actual, str) or os.path.realpath(actual) != os.path.realpath(
            self.settings.teacher_model_path
        ):
            raise ValueError("teacher endpoint model_path does not match teacher_model_path")

    def prepare(self, rollout_data: dict[str, Any]) -> dict[str, float]:
        """Install all teacher columns atomically before a training job starts."""
        started = time.monotonic()
        self.validate_model()
        columns: dict[str, list[Any]] = {key: [] for key in ENGINE_COLUMNS}
        for tokens, length in zip(rollout_data["teacher_tokens"], rollout_data["response_lengths"], strict=True):
            row = self.score(tokens, length)
            for key in ENGINE_COLUMNS:
                columns[key].append(row[key])
        rollout_data.update(columns)
        return {"perf/distill_teacher_time": time.monotonic() - started}
