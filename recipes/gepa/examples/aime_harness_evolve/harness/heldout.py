"""Per-example checkpoints for the sealed held-out evaluations."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from gepa.core.adapter import EvaluationBatch

from .adapter import AIMEExample
from .files import fingerprint, read_json, write_json


class BatchAdapter(Protocol):
    def evaluate(
        self, batch: list[AIMEExample], candidate: dict[str, str], capture_traces: bool = False
    ) -> EvaluationBatch[Any, Any]: ...


class CheckpointedEvaluator:
    """Evaluate a held-out batch with one durable result file per example.

    A resumed run skips finished examples; a checkpoint written for a different
    candidate or example is refused rather than reused. Pending examples go to
    the adapter ``batch_size`` at a time, so its own worker pool applies.
    """

    def __init__(
        self,
        adapter: BatchAdapter,
        root: Path,
        *,
        batch_size: int = 1,
        failure_score: float | None = None,
    ) -> None:
        self.adapter = adapter
        self.root = Path(root)
        self.batch_size = batch_size
        self.failure_score = failure_score

    def evaluate(self, label: str, batch: Sequence[AIMEExample], candidate: Mapping[str, str]) -> EvaluationBatch:
        candidate = dict(candidate)
        candidate_sha256 = fingerprint(candidate)
        payloads: list[dict[str, Any] | None] = [None] * len(batch)
        pending = []
        for index, example in enumerate(batch):
            path = self.root / label / f"example-{index:04d}.json"
            if path.is_file():
                payload = read_json(path)
                identity = {
                    "index": index,
                    "example_sha256": fingerprint(example),
                    "candidate_sha256": candidate_sha256,
                }
                if any(payload.get(key) != value for key, value in identity.items()):
                    raise RuntimeError(f"held-out checkpoint identity changed across resume: {path}")
                payloads[index] = payload
            else:
                pending.append((index, example, path))
        for offset in range(0, len(pending), self.batch_size):
            chunk = pending[offset : offset + self.batch_size]
            for (index, example, path), (score, output) in zip(chunk, self._evaluate(chunk, candidate), strict=True):
                payload = {
                    "index": index,
                    "example_sha256": fingerprint(example),
                    "candidate_sha256": candidate_sha256,
                    "score": score,
                    "output": output,
                }
                write_json(path, payload)
                payloads[index] = payload
        complete = [payload for payload in payloads if payload is not None]
        return EvaluationBatch(
            outputs=[payload["output"] for payload in complete],
            scores=[float(payload["score"]) for payload in complete],
            trajectories=None,
            num_metric_calls=len(batch),
        )

    def _evaluate(
        self, chunk: list[tuple[int, AIMEExample, Path]], candidate: dict[str, str]
    ) -> list[tuple[float, Any]]:
        try:
            evaluated = self.adapter.evaluate([example for _, example, _ in chunk], candidate, capture_traces=False)
            if len(evaluated.scores) != len(chunk):
                raise RuntimeError("held-out adapter must return exactly one result per example")
            return [
                (float(score), json.loads(json.dumps(output, default=str)))
                for output, score in zip(evaluated.outputs, evaluated.scores, strict=True)
            ]
        except Exception as exc:
            # The spend cap raises SystemExit, which passes through untouched.
            if self.failure_score is None:
                raise
            if len(chunk) > 1:
                return [result for item in chunk for result in self._evaluate([item], candidate)]
            return [(self.failure_score, {"error": type(exc).__name__})]
