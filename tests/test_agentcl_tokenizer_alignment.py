"""Optional CPU-only Qwen tokenizer checks; native responses are synthetic, not model inference.

Set AGENTCL_TOKENIZER_PATH to a local Qwen2.5-7B-Instruct tokenizer directory.
No model weights, downloads, SGLang runtime, or GPU are required.
"""

from __future__ import annotations

import os
from itertools import pairwise
from pathlib import Path

import pytest

from recipes.sdft.processor import SDFTProcessor
from recipes.sdpo.processor import SDPOProcessor
from recipes.sdpo.report import SDPOReport
from reef.artifact import Artifact, LiveWeightArtifactRef
from reef.core import AgentRecord, RequestType
from reef.core.reports import TeacherContextReport
from reef.inference.sglang.chat import SGLangInferenceHandler
from reef.train.types import ProcessorContext, trajectories

transformers = pytest.importorskip("transformers")

QUESTION = "Use Python to calculate the sum of 2 and 3, then verify it."
ASSISTANT_RESPONSES = (
    "```python\nvalue = 2 + 3\nprint(value)\n```",
    "```python\nprint(value == 5)\n```",
    "```python\ndef answer():\n    return 5\n```",
)
OBSERVATIONS = ("Code execution observation:\n5\n", "Code execution observation:\nTrue\n")


@pytest.fixture
def local_tokenizer(monkeypatch: pytest.MonkeyPatch) -> transformers.PreTrainedTokenizerBase:
    tokenizer_path = os.environ.get("AGENTCL_TOKENIZER_PATH")
    if not tokenizer_path:
        pytest.skip("set AGENTCL_TOKENIZER_PATH for the optional local Qwen tokenizer checks")
    directory = Path(tokenizer_path)
    if not directory.is_dir() or not (directory / "tokenizer_config.json").is_file():
        pytest.fail("AGENTCL_TOKENIZER_PATH must contain the local Qwen tokenizer assets")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    tokenizer = transformers.AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    assert tokenizer.eos_token_id == 151645
    assert tokenizer.encode("<|endoftext|>", add_special_tokens=False) == [151643]
    assert tokenizer.encode("\n", add_special_tokens=False) == [198]
    return tokenizer


def native_episode(
    tokenizer: transformers.PreTrainedTokenizerBase,
    *,
    prefix: str,
    terminal_text: str,
    content_suffix: str = "",
    trim_visible_newline: bool = False,
) -> tuple[AgentRecord, ...]:
    """Run the real facade renderer and parser over fabricated SGLang log-probability pairs."""
    handler = SGLangInferenceHandler(
        "http://unused.invalid", model_path=str(tokenizer.name_or_path), tokenizer=tokenizer, force_reasoning=False
    )
    artifact = Artifact(
        LiveWeightArtifactRef(
            content_id="synthetic-agentcl",
            release_id="live:synthetic:1",
            parent_release_id="synthetic-base",
            runtime_load_id="synthetic:1",
        ),
        None,
    )
    messages = [{"role": "user", "content": QUESTION}]
    records = []
    for turn_index, assistant in enumerate(ASSISTANT_RESPONSES):
        content = assistant + content_suffix
        request = {
            "model": "synthetic-qwen",
            "messages": list(messages),
            "max_tokens": 2048,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        call = handler._chat_call("/v1/chat/completions", request)
        expected_prompt = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=False, enable_thinking=False
        )
        assert call.prompt_ids == expected_prompt
        assert call.prompt_ids[-3:] == tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        sampled_ids = tokenizer.encode(content + terminal_text, add_special_tokens=False)
        log_probs = [-0.1 - token_index / 1000 for token_index in range(len(sampled_ids))]
        visible_content = content.rstrip("\n") if trim_visible_newline else content
        native = {
            "text": visible_content,
            "meta_info": {
                "finish_reason": {"type": "stop" if terminal_text else "length"},
                "completion_tokens": len(sampled_ids),
                "output_token_logprobs": [list(pair) for pair in zip(log_probs, sampled_ids, strict=True)],
                "_reef_token_runtime_load_ids": ["synthetic:1"] * len(sampled_ids),
            },
        }
        captured = handler.client.parse(artifact, native, capture_topk=0)
        response = handler._response_from_capture(call, captured, None)
        assert captured.output_ids == sampled_ids
        assert response["training"]["tokens"] == [*expected_prompt, *sampled_ids]
        assert response["training"]["loss_mask"] == [1] * len(sampled_ids)
        records.append(
            AgentRecord.create(
                scenario="tokenizer-preflight",
                request_type=RequestType.INFERENCE,
                payload={**request, "response": response},
                agent_record_id=f"{prefix}-turn-{turn_index}",
                artifact_ref=artifact.ref,
            )
        )
        messages.append(response["choices"][0]["message"])
        if turn_index < len(OBSERVATIONS):
            messages.append({"role": "user", "content": OBSERVATIONS[turn_index]})
    return tuple(records)


@pytest.mark.unit
@pytest.mark.parametrize("method", ["sdft", "sdpo"])
@pytest.mark.parametrize("terminal_text", ["", "<|im_end|>", "<|im_end|>\n"])
@pytest.mark.parametrize("content_suffix", ["", "\n"])
def test_exact_qwen_episode_retains_all_sampled_ids_and_teacher_suffix(
    local_tokenizer: transformers.PreTrainedTokenizerBase,
    method: str,
    terminal_text: str,
    content_suffix: str,
) -> None:
    config = {
        "tokenizer_path": str(local_tokenizer.name_or_path),
        "accept_multi_turn_policy_samples": True,
        "max_teacher_tokens": 16384,
        "batch_size": 1,
    }
    if method == "sdft":
        processor = SDFTProcessor(ProcessorContext("tokenizer-preflight", config, TeacherContextReport))
        attempt_count = 1
    else:
        config.update(
            groups_per_step=1,
            rollouts_per_group=2,
            max_teacher_prompt_tokens=8192,
            include_environment_feedback=True,
            enable_thinking=False,
        )
        processor = SDPOProcessor(ProcessorContext("tokenizer-preflight", config, SDPOReport))
        attempt_count = 2
    episodes = []
    for attempt in range(attempt_count):
        records = native_episode(
            local_tokenizer,
            prefix=f"attempt-{attempt}",
            terminal_text=terminal_text,
            content_suffix=content_suffix,
        )
        episodes.append(records)
        for previous, current in pairwise(records):
            previous_tokens = previous.payload["response"]["training"]["tokens"]
            current_training = current.payload["response"]["training"]
            next_prompt = current_training["tokens"][: -len(current_training["loss_mask"])]
            assert next_prompt[: len(previous_tokens)] == previous_tokens
        for record in records:
            processor.ingest(record)
        references = tuple(record.agent_record_id for record in records)
        teacher_context = "Synthetic demonstration and feedback: print(2 + 3) produces 5."
        if method == "sdft":
            payload = TeacherContextReport(teacher_context=teacher_context, score=0.0).to_dict(references=references)
        else:
            payload = SDPOReport(teacher_context=teacher_context, score=0.0, step=0, group=0, rollout=attempt).to_dict(
                references=references
            )
        processor.ingest(
            AgentRecord.create(
                scenario="tokenizer-preflight",
                request_type=RequestType.REPORT,
                payload=payload,
                agent_record_id=f"attempt-{attempt}-report",
                references=references,
            )
        )
    samples = trajectories(processor.build_batch())
    assert len(samples) == attempt_count
    for sample, records in zip(samples, episodes, strict=True):
        training = sample.training
        mask = training["loss_mask"]
        suffix = training["tokens"][-len(mask) :]
        selected_ids = [token for token, selected in zip(suffix, mask, strict=True) if selected]
        expected_ids = []
        expected_log_probs = []
        for record in records:
            native = record.payload["response"]["training"]
            expected_ids.extend(native["tokens"][-len(native["loss_mask"]) :])
            expected_log_probs.extend(native["rollout_log_probs"])
        assert selected_ids == expected_ids
        assert sum(mask) == len(expected_ids)
        assert 0 in mask
        assert training["teacher_tokens"][-len(mask) :] == suffix
        assert [
            value for value, selected in zip(training["rollout_log_probs"], mask, strict=True) if selected
        ] == expected_log_probs
        assert all(
            value == 0.0 for value, selected in zip(training["rollout_log_probs"], mask, strict=True) if not selected
        )
        assert training["turn_count"] == 3
        assert training["runtime_load_id"] == "synthetic:1"
        assert len(sample.metadata["records"]) == 3


@pytest.mark.unit
@pytest.mark.parametrize(
    ("terminal_text", "content_suffix", "trim_visible_newline"),
    [("<|endoftext|>", "", False), ("<|im_end|>\n\n", "", False), ("<|im_end|>", "\n", True)],
    ids=["alternate-eos", "extra-hidden-newline", "trimmed-visible-newline"],
)
def test_strict_qwen_episode_rejects_terminal_or_text_drift(
    local_tokenizer: transformers.PreTrainedTokenizerBase,
    terminal_text: str,
    content_suffix: str,
    trim_visible_newline: bool,
) -> None:
    processor = SDFTProcessor(
        ProcessorContext(
            "tokenizer-preflight",
            {"tokenizer_path": str(local_tokenizer.name_or_path), "accept_multi_turn_policy_samples": True},
            TeacherContextReport,
        )
    )
    records = native_episode(
        local_tokenizer,
        prefix="drift",
        terminal_text=terminal_text,
        content_suffix=content_suffix,
        trim_visible_newline=trim_visible_newline,
    )
    for record in records:
        processor.ingest(record)
    references = tuple(record.agent_record_id for record in records)
    report = AgentRecord.create(
        scenario="tokenizer-preflight",
        request_type=RequestType.REPORT,
        payload=TeacherContextReport(teacher_context="Synthetic demonstration.").to_dict(references=references),
        agent_record_id="drift-report",
        references=references,
    )
    with pytest.raises(ValueError, match="cannot assemble the recorded multi-turn trajectory"):
        processor.ingest(report)
