"""MLX's in-process handler reports the resident engine version."""

from __future__ import annotations

import asyncio
from pathlib import Path

from tests.support.mlx_engine_stubs import FakeAdapterSnapshot, FakeMLXEngine

from reef.artifact.artifact import Artifact
from reef.runtime.interfaces import ModelCandidate
from reef.train.mlx_backend.serving import MLXServingRuntime


def test_handler_reports_the_resident_version_in_the_completion_receipt(tmp_path: Path) -> None:
    runtime = MLXServingRuntime(FakeMLXEngine())

    response = asyncio.run(
        runtime.inference_handler.inference(
            Artifact.local(Path(tmp_path)), "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}
        )
    )

    assert response["choices"][0]["meta_info"]["runtime_load_id"] == runtime.serving_runtime_load_id()


def test_handler_reports_the_resident_candidate_while_publication_is_pending(tmp_path: Path) -> None:
    runtime = MLXServingRuntime(FakeMLXEngine())
    runtime.mark_published()
    prior_runtime_load_id = runtime.current_runtime_load_id()
    candidate = ModelCandidate(
        candidate_id="candidate-1",
        training_job_id="training-1",
        checkpoint_path=str(tmp_path / "candidate"),
        current_runtime_load_id=prior_runtime_load_id,
    )
    runtime.stage_candidate(candidate.candidate_id, FakeAdapterSnapshot({"layer.lora_a": 1.0, "layer.lora_b": 1.0}))
    activated = runtime.activate_candidate(candidate)

    response = asyncio.run(
        runtime.inference_handler.inference(
            Artifact.local(Path(tmp_path)), "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}
        )
    )

    assert runtime.current_runtime_load_id() == prior_runtime_load_id
    assert response["choices"][0]["meta_info"]["runtime_load_id"] == activated.runtime_load_id
    assert response["training"]["runtime_load_id"] == activated.runtime_load_id
