"""The teacher engine: a frozen SGLang server that scores the recorded response tokens.

This is the second of a distillation family's two teacher paths. ``teacher``
runs the teacher on the actor's weights inside the training step;
this module asks a separate engine and runs on the bridge, which fills the
distillation columns before partitioning or starting the step. No
generation, retokenization or actor weight swap is involved, and the module
deliberately has no torch or serving-runtime imports.
"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.request
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from numbers import Integral, Real
from typing import Any

#: The columns the teacher engine fills on the bridge: the top-K representation. The exact
#: full-vocabulary column stays with the checkpoint teacher, which scores on the actor.
ENGINE_TEACHER_BATCH_KEYS = (
    "distill_teacher_topk_ids",
    "distill_teacher_topk_log_probs",
    "distill_teacher_sampled_log_probs",
)
#: Scoring requests in flight at once; the engine batches them, and the columns are installed
#: only once every row is in.
SCORING_CONCURRENCY = 16


def validate_teacher_tokenizer(student_path: str, teacher_path: str) -> None:
    """Reject different token mappings before accepting a training batch."""
    from transformers import AutoTokenizer

    student = AutoTokenizer.from_pretrained(student_path, trust_remote_code=False)
    teacher = AutoTokenizer.from_pretrained(teacher_path, trust_remote_code=False)
    if student.get_vocab() != teacher.get_vocab():
        raise ValueError("a distillation teacher engine must use the student's token-to-ID vocabulary")
    # The teacher reads ids, never the chat template. Different instruction
    # templates are expected, but the token bytes behind the ids must agree.
    if not student.is_fast or not teacher.is_fast:
        raise ValueError("teacher engine validation requires fast tokenizers")
    student_decoder = json.loads(student.backend_tokenizer.to_str()).get("decoder")
    teacher_decoder = json.loads(teacher.backend_tokenizer.to_str()).get("decoder")
    if student_decoder is None or teacher_decoder is None:
        raise ValueError("teacher engine validation requires token decoders")
    if student_decoder != teacher_decoder:
        raise ValueError("a distillation teacher engine must use the student's token decoder")


def log_prob_entry(value: Any) -> tuple[float, int]:
    """One SGLang log-prob entry, ``[log_prob, token_id, text]``, as a finite log-prob and a token id."""
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError("teacher log-prob entry must contain a log-prob and token ID")
    log_prob, token = value[:2]
    if (
        not isinstance(log_prob, Real)
        or isinstance(log_prob, bool)
        or not math.isfinite(log_prob)
        or float(log_prob) > 1e-5
        or not isinstance(token, Integral)
        or isinstance(token, bool)
        or token < 0
    ):
        raise ValueError("teacher returned an invalid log-prob or token ID")
    return float(log_prob), int(token)


def teacher_vocab_size(model_path: str) -> int:
    """The teacher's vocabulary size from its HF config, ``text_config`` for multimodal models.

    This is the size SGLang serves. Checkpoints pad it past the tokenizer's last id,
    and the student samples from the padded rows too.
    """
    with open(os.path.join(model_path, "config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    text_config = config.get("text_config", config) if isinstance(config, Mapping) else None
    vocab_size = text_config.get("vocab_size") if isinstance(text_config, Mapping) else None
    if not isinstance(vocab_size, int) or isinstance(vocab_size, bool) or vocab_size <= 0:
        raise ValueError("teacher config.json must declare a positive vocab_size")
    return vocab_size


def parse_teacher_scores(
    result: Any, tokens: Sequence[int], response_length: int, top_k: int, vocab_size: int
) -> dict[str, Any]:
    """Validate SGLang's prefill alignment, including its leading unscored token.

    We request from the last prompt token. SGLang returns that token with
    log-prob None, then exactly R scored response tokens. Requiring the
    IDs and row counts prevents silently shifting or truncating supervision.
    The one id SGLang does not echo is the last row of the vocabulary: its
    multimodal placeholder clip reports that row as 0 while the log-prob stays
    the row's own, so a recorded ``vocab_size - 1`` accepts a returned 0.
    """
    if not isinstance(result, Mapping) or not isinstance(result.get("meta_info"), Mapping):
        raise ValueError("teacher response is missing meta_info")
    meta = result["meta_info"]
    if meta.get("completion_tokens") != 0 or result.get("output_ids", []):
        raise ValueError("teacher scoring must not generate tokens")
    sampled = meta.get("input_token_logprobs")
    top_rows = meta.get("input_top_logprobs")
    if not isinstance(sampled, list) or not isinstance(top_rows, list):
        raise ValueError("teacher response is missing input log-prob rows")
    if len(sampled) != response_length + 1 or len(top_rows) != response_length + 1:
        raise ValueError("teacher log-prob row count does not match the recorded response")
    first = sampled[0]
    if (
        not isinstance(first, list)
        or len(first) < 2
        or first[0] is not None
        or first[1] != tokens[-response_length - 1]
        or top_rows[0] is not None
    ):
        raise ValueError("teacher response has an invalid leading prompt-token row")
    selected_ids, selected_log_probs, sampled_log_probs = [], [], []
    for token, sample, entries in zip(tokens[-response_length:], sampled[1:], top_rows[1:], strict=True):
        log_prob, actual_token = log_prob_entry(sample)
        if actual_token != token and not (actual_token == 0 and token == vocab_size - 1):
            raise ValueError("teacher log-prob token IDs do not match the recorded response")
        if not isinstance(entries, list) or len(entries) != top_k:
            raise ValueError("teacher top-K row has the wrong number of entries")
        parsed = [log_prob_entry(entry) for entry in entries]
        ids = [entry[1] for entry in parsed]
        if len(set(ids)) != top_k:
            raise ValueError("teacher top-K row contains duplicate token IDs")
        selected_ids.append(ids)
        selected_log_probs.append([entry[0] for entry in parsed])
        sampled_log_probs.append(log_prob)
    return dict(zip(ENGINE_TEACHER_BATCH_KEYS, (selected_ids, selected_log_probs, sampled_log_probs), strict=True))


class TeacherEngine:
    """A synchronous, bounded client of the teacher engine; the bound distillation family holds one for the run."""

    def __init__(self, *, url: str, model_path: str, timeout_seconds: float, top_k: int) -> None:
        self.url = url.rstrip("/")
        self.model_path = model_path
        self.timeout_seconds = timeout_seconds
        self.top_k = top_k
        self.vocab_size = teacher_vocab_size(model_path)

    def score(self, tokens: list[int], response_length: int) -> dict[str, Any]:
        """Score exact IDs with a prefill-only request; HTTP errors fail the batch."""
        if not 0 < response_length < len(tokens):
            raise ValueError("teacher scoring needs a nonempty prompt and response")
        body = {
            "input_ids": tokens,
            "sampling_params": {"max_new_tokens": 0, "temperature": 1.0},
            "return_logprob": True,
            "logprob_start_len": len(tokens) - response_length - 1,
            "top_logprobs_num": self.top_k,
        }
        request = urllib.request.Request(
            self.url + "/generate",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=self.timeout_seconds) as response:
            result = json.load(response)
        return parse_teacher_scores(result, tokens, response_length, self.top_k, self.vocab_size)

    def validate_model(self) -> None:
        """Refuse an endpoint serving a different model than the validated tokenizer."""
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(self.url + "/model_info", timeout=self.timeout_seconds) as response:
            info = json.load(response)
        actual = info.get("model_path") if isinstance(info, Mapping) else None
        if not isinstance(actual, str) or os.path.realpath(actual) != os.path.realpath(self.model_path):
            raise ValueError("teacher endpoint model_path does not match teacher_model_path")

    def prepare(self, rollout_data: dict[str, Any]) -> dict[str, float]:
        """Install all teacher columns atomically before a training job starts."""
        started = time.monotonic()
        self.validate_model()
        samples = list(zip(rollout_data["teacher_tokens"], rollout_data["response_lengths"], strict=True))
        with ThreadPoolExecutor(max_workers=SCORING_CONCURRENCY) as pool:
            rows = list(pool.map(lambda sample: self.score(*sample), samples))
        rollout_data.update({key: [row[key] for row in rows] for key in ENGINE_TEACHER_BATCH_KEYS})
        return {"perf/distill_teacher_time": time.monotonic() - started}
