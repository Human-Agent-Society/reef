"""Whole-episode OPD assembly, capture and masked loss using CPU fixtures."""

import copy
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from reef_service.runtime_stubs import StubTrainingRuntime, runtime_bindings

from recipes.opd import OPDProcessor, OPDRecipe
from recipes.opd.examples.agentcl import qualification
from recipes.opd.examples.agentcl.native import CapturedOPDProcessor
from reef.core import AgentRecord, RequestType
from reef.core.reports import TeacherContextReport
from reef.train.types import ProcessorContext, TrainingBatch

from .test_distill_processor import episode_inferences


def terminal_record(references: tuple[str, ...], score: float = 1.0) -> AgentRecord:
    return AgentRecord.create(
        scenario="science",
        request_type=RequestType.REPORT,
        agent_record_id="episode-report",
        references=references,
        payload=TeacherContextReport(teacher_context="", score=score).to_dict(references=references),
    )


def episode_processor(monkeypatch: pytest.MonkeyPatch, **settings: object) -> OPDProcessor:
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace())
    return OPDProcessor(
        ProcessorContext(
            "science", {"batch_size": 1, "accept_multi_turn_policy_samples": True, **settings}, TeacherContextReport
        )
    )


@pytest.mark.parametrize("score", [0.0, 1.0])
def test_whole_episode_keeps_exact_student_tokens_and_zero_context_loss(monkeypatch, score):
    processor = episode_processor(monkeypatch)
    records = episode_inferences()
    references = tuple(record.agent_record_id for record in records)
    for record in records:
        processor.ingest(record)
    processor.ingest(terminal_record(references, score))
    batch = processor.build_batch()
    assert isinstance(batch, TrainingBatch)
    (sample,) = batch.items
    assert sample.training["tokens"] == sample.training["teacher_tokens"] == [5, 6, 7, 1, 2, 3, 20, 21, 4, 5, 22, 6]
    assert sample.training["loss_mask"] == [1, 1, 1, 0, 0, 1, 1, 0, 1]
    assert sample.training["rollout_log_probs"] == [-0.1, -0.2, -0.3, 0.0, 0.0, -0.4, -0.5, 0.0, -0.6]
    assert sample.source_agent_record_ids == (*references, "episode-report")
    assert processor.tokenizer is None
    assert processor.releasable_record_ids() == frozenset()
    assert processor.acknowledge(batch.batch_id) == frozenset((*references, "episode-report"))


def test_episode_overflow_retains_every_source(monkeypatch):
    processor = episode_processor(monkeypatch, max_teacher_tokens=11)
    records = episode_inferences()
    for record in records:
        processor.ingest(record)
    with pytest.raises(ValueError, match="episode teacher sequence exceeds"):
        processor.ingest(terminal_record(tuple(record.agent_record_id for record in records)))
    assert processor.operational_metrics()["teacher_overflow_reports"] == 0
    assert not processor.ready()
    assert not processor.releasable_record_ids()


@pytest.mark.parametrize("fault", ["missing-turn", "mixed-release", "changed-response-token", "context-drift"])
def test_invalid_episode_cannot_fill_opd_batch(monkeypatch, fault):
    processor = episode_processor(monkeypatch)
    records = list(episode_inferences())
    references = tuple(record.agent_record_id for record in records)
    if fault == "missing-turn":
        references = (references[0], references[2])
    elif fault == "mixed-release":
        records[1] = replace(records[1], artifact_ref=replace(records[1].artifact_ref, release_id="different"))
    else:
        payload = copy.deepcopy(records[2].payload)
        tokens = payload["response"]["training"]["tokens"]
        tokens[4 if fault == "changed-response-token" else 6] = 999
        records[2] = replace(records[2], payload=payload)
    for record in records:
        processor.ingest(record)
    with pytest.raises(ValueError):
        processor.ingest(terminal_record(references))
    assert not processor.ready()
    assert not processor.releasable_record_ids()


def test_native_capture_is_exact_private_and_repeatable(monkeypatch, tmp_path: Path):
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace())
    directory = tmp_path / "teacher-records"
    processor = CapturedOPDProcessor(
        ProcessorContext(
            "science",
            {"batch_size": 1, "accept_multi_turn_policy_samples": True, "native_sample_dir": str(directory)},
            TeacherContextReport,
        )
    )
    records = episode_inferences()
    for record in records:
        processor.ingest(record)
    references = tuple(record.agent_record_id for record in records)
    processor.ingest(terminal_record(references))
    batch = processor.build_batch()
    path = directory / "episode-report.json"
    capture = json.loads(path.read_text())
    episode = {
        "report_id": "episode-report",
        "references": list(references),
        "release_id": "slime-v3",
        "turns": [
            {
                "receipt": record.agent_record_id,
                "record": {"payload": record.payload, "artifact_ref": {"release_id": "slime-v3"}},
            }
            for record in records
        ],
    }
    checked = qualification.check_native_sample(capture, episode)
    assert checked["assistant_token_count"] == 6
    assert checked["masked_context_token_count"] == 3
    assert checked["teacher_student_sequence_identity"] is True
    assert capture["teacher_token_sha256"] == capture["student_token_sha256"]
    assert capture["teacher_input"]["prompt_text_decoded_from_captured_tokens"] is None
    assert path.stat().st_mode & 0o777 == 0o600
    assert directory.stat().st_mode & 0o777 == 0o700
    before = path.read_bytes()
    processor.release_batch(batch.batch_id)
    assert processor.build_batch().items == batch.items
    assert path.read_bytes() == before
    capture["teacher_tokens"][0] = 999
    with pytest.raises(ValueError, match="exact recorded student"):
        qualification.check_native_sample(capture, episode)


def test_sampled_opd_loss_trains_assistant_positions_and_masks_tool_context(monkeypatch):
    torch = pytest.importorskip("torch")
    from argparse import Namespace

    from recipes.opd.slime import OpdAlgorithm, OpdSettings
    from reef.train.slime_backend.distill.objective import distill_loss

    from .test_distill_score_centering import cpu_batch_adapters

    cpu_batch_adapters.__wrapped__(monkeypatch)
    generator = torch.Generator().manual_seed(52)
    student = torch.randn(9, 11, generator=generator).requires_grad_()
    teacher = torch.randn(9, 11, generator=generator).log_softmax(-1)
    sampled = torch.tensor([1, 2, 3, 7, 8, 4, 5, 9, 6])
    mask = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    top_ids = teacher.argmax(-1, keepdim=True)
    batch = {
        "total_lengths": [12],
        "response_lengths": [9],
        "unconcat_tokens": [torch.cat([torch.tensor([0, 0, 0]), sampled])],
        "loss_masks": [mask],
        "distill_sample_weights": [1.0],
        "distill_teacher_topk_ids": [top_ids],
        "distill_teacher_topk_log_probs": [teacher.gather(-1, top_ids)],
        "distill_teacher_sampled_log_probs": [teacher.gather(-1, sampled[:, None])[:, 0]],
    }
    args = Namespace(score_centering=False, calculate_per_token_loss=False, log_probs_chunk_size=2)
    OpdAlgorithm().apply_driver_options(args, OpdSettings(teacher_checkpoint="/teacher"))

    def _reduce(values):
        return (values * mask).sum() / mask.sum()

    loss, _ = distill_loss(args, batch, student, _reduce)
    loss.backward()
    probabilities = student.detach().softmax(-1)
    sampled_log_probs = student.detach().log_softmax(-1).gather(-1, sampled[:, None])[:, 0]
    gap = sampled_log_probs - teacher.gather(-1, sampled[:, None])[:, 0]
    expected = (gap * mask / mask.sum())[:, None] * (torch.nn.functional.one_hot(sampled, 11) - probabilities)
    torch.testing.assert_close(student.grad, expected, atol=2e-6, rtol=2e-5)
    assert torch.equal(student.grad[mask == 0], torch.zeros_like(student.grad[mask == 0]))
    assert (student.grad[mask == 1].abs().sum(-1) > 0).all()
    assert torch.isfinite(loss) and torch.isfinite(student.grad).all()


def test_opd_recipe_rejects_non_boolean_episode_setting():
    with pytest.raises(ValueError, match="must be a boolean"):
        OPDRecipe(accept_multi_turn_policy_samples=1, **runtime_bindings(StubTrainingRuntime()))
