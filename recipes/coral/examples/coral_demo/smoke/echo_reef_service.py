#!/usr/bin/env python3
"""The Reef service for the no-GPU smoke run: real service, canned model.

Everything above the inference backend is the production code path —
``reef.service.app.create_app`` with the default dispatcher, token auth,
receipts, reports, dedup. Only the model is fake: instead of SGLang serving
a trained policy, the backend cycles through canned ``merge_sorted``
implementations of different quality, so graded attempts show score
contrast without a GPU in sight.
"""

from __future__ import annotations

import argparse
import itertools

from aiohttp import web

from reef.dispatcher import build_default_dispatcher
from reef.runtime.inference import InferenceBackend
from reef.service.app import create_app
from reef.storage.factory import SQLiteScenarioStoreFactory

CORRECT = """\
def merge_sorted(a, b):
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        if b[j] < a[i]:
            out.append(b[j])
            j += 1
        else:
            out.append(a[i])
            i += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out
"""

UNSTABLE = """\
def merge_sorted(a, b):
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        if b[j] <= a[i]:
            out.append(b[j])
            j += 1
        else:
            out.append(a[i])
            i += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out
"""


class CannedSolutionBackend(InferenceBackend):
    """Cycle canned solutions; the wiring around this stays real."""

    def __init__(self) -> None:
        self._answers = itertools.cycle([CORRECT, UNSTABLE])

    async def inference(self, artifact, path, payload):
        del artifact, path
        content = next(self._answers)
        prompt_tokens = sum(len(str(m)) for m in payload.get("messages", [])) // 4
        return {
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": len(content) // 4,
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8900)
    parser.add_argument("--token", default="reef-local")
    args = parser.parse_args()
    app = create_app(
        build_default_dispatcher(scenario_store_factory=SQLiteScenarioStoreFactory()),
        tokens=args.token,
        inference_backend=CannedSolutionBackend(),
        close_dispatcher=True,
    )
    web.run_app(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
