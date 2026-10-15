"""MLX's in-process handler reports the resident engine version."""

from __future__ import annotations

import asyncio
from pathlib import Path

from tests.support.mlx_engine_stubs import FakeMLXEngine

from reef.artifact.artifact import Artifact
from reef.train.mlx_backend.serving import MLXServingRuntime


def test_handler_reports_the_resident_version_in_the_completion_receipt(tmp_path: Path) -> None:
    runtime = MLXServingRuntime(FakeMLXEngine())

    response = asyncio.run(
        runtime.inference_handler.inference(
            Artifact.local(Path(tmp_path)), "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}
        )
    )

    assert response["choices"][0]["meta_info"]["runtime_load_id"] == runtime.serving_runtime_load_id()
