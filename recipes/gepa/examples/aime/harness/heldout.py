"""Per-example checkpoints for the two sealed test passes.

A test pass is 150 Pi episodes against one composition, so it is the part of
a run most likely to be interrupted and the part that must never be paid for
twice. Each example's score lands in its own file keyed by the example and
the composition it was measured on; a resumed pass skips what is finished and
refuses a checkpoint written for a different composition rather than folding
it into the score.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


def fingerprint(value: Any) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class CheckpointedEvaluator:
    """Score a held-out batch with one durable result file per example."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def scores(
        self,
        label: str,
        tasks: Sequence[str],
        composition: Mapping[str, str],
        run: Callable[[int, str], tuple[float, dict[str, Any]]],
    ) -> list[float]:
        """The per-example scores for ``tasks`` under ``composition``.

        ``run`` is called only for examples with no checkpoint; it returns the
        score and whatever record of the episode belongs beside it.
        """
        composition_sha256 = fingerprint(dict(composition))
        scores: list[float] = []
        for index, task in enumerate(tasks):
            path = self.root / label / f"example-{index:04d}.json"
            identity = {
                "index": index,
                "task_sha256": fingerprint(task),
                "composition_sha256": composition_sha256,
            }
            if path.is_file():
                payload = json.loads(path.read_text(encoding="utf-8"))
                if any(payload.get(key) != value for key, value in identity.items()):
                    raise RuntimeError(f"held-out checkpoint identity changed across resume: {path}")
            else:
                score, output = run(index, task)
                payload = {**identity, "score": float(score), "output": output}
                _write_json(path, payload)
            scores.append(float(payload["score"]))
        return scores


def _write_json(path: Path, value: Any) -> None:
    """Write through a sibling temporary file so an interrupted pass never leaves a torn checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)
