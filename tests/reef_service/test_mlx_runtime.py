"""Contract tests for the MLX runtime that need no Apple hardware.

Everything MLX-specific lives behind the engine, so the runtime's contract
with Reef — preparation, candidate export, activation, rejection, rollback,
and the refusal paths — is testable against a fake engine on any machine.
The parts that genuinely need Metal (real generation, a real optimizer step,
memory ceilings) belong to the gated Apple Silicon qualification instead.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from reef.core.evaluation import EvaluationResult, SelectionDecision
from reef.runtime.base import RuntimeContractError
from reef.runtime.candidates import ModelCandidate
from reef.runtime.registry import RuntimeRegistry
from reef.train.algos.base import StepPreparer, register_step_preparer
from reef.train.algos.signals import StepScheduling, StepSignal
from reef.train.mlx_backend.runtime import MLXRuntime
from reef.train.types import PolicyBatch, PolicySample


class FakeEngineConfig:
    model_path = "fake/model"
    lora_layers = 2
    lora_rank = 4
    learning_rate = 1e-5


class FakeEngine:
    """Stand in for MLXEngine with plain Python state.

    ``moved`` decides whether a training step actually changes weights, which
    is what the refusal test needs to drive.
    """

    def __init__(self, *, moved: bool = True) -> None:
        self.config = FakeEngineConfig()
        self.weights = {"layer.lora_a": 0.0, "layer.lora_b": 0.0}
        self.moved = moved
        self.publications = 0
        self.saved: list[Path] = []
        self.loaded: list[Path] = []
        self.trained_rows: list[Any] = []
        self.distillation_rows: list[Any] | None = None

    def adapter_snapshot(self) -> dict[str, float]:
        return dict(self.weights)

    def apply_adapter(self, snapshot) -> None:
        self.weights = dict(snapshot)

    def adapter_delta(self, before, after) -> tuple[float, int]:
        changed = sum(1 for name, value in after.items() if before.get(name) != value)
        return (float(changed), changed)

    def train_step(self, rows) -> dict[str, Any]:
        self.trained_rows = list(rows)
        if self.moved:
            self.weights = {name: value + 1.0 for name, value in self.weights.items()}
        return {"loss": -1.0, "rows": len(rows)}

    def openclawrl_step(self, rows, **settings) -> dict[str, Any]:
        self.distillation_rows = list(rows)
        if self.moved:
            self.weights = {name: value + 1.0 for name, value in self.weights.items()}
        return {"loss": -1.0, "rows": len(rows), **settings}

    def base_log_probs(self, rows) -> list[list[float]]:
        return [[-1.0] * len(row.loss_mask) for row in rows]

    def next_runtime_load_id(self) -> str:
        self.publications += 1
        return f"fake-{self.publications}"

    def save_adapter(self, destination: Path, *, origin_extra=None) -> Path:
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "adapters.safetensors").write_bytes(b"weights")
        (destination / "reef_origin.json").write_text(json.dumps(dict(origin_extra or {})))
        self.saved.append(destination)
        return destination

    def load_adapter(self, source: Path) -> None:
        self.loaded.append(Path(source))


def sample(reward: float) -> PolicySample:
    return PolicySample(
        source_agent_record_id=f"rec-{reward}",
        tokens=(1, 2, 3, 4),
        loss_mask=(1, 1),
        rollout_log_probs=(-0.5, -0.25),
        reward=reward,
    )


@register_step_preparer
class _TwoAdvantagePreparer(StepPreparer):
    name = "mlx-test-preparer"

    def __call__(self, batch, state):
        return StepSignal(
            "train",
            "fake-family",
            {"steps": int(state.get("steps", 0)) + 1},
            {"prepared": True},
            (1.0, -1.0),
            StepScheduling(unit="sample", batch_size="actual"),
        )


@register_step_preparer
class _TttdFamilyPreparer(StepPreparer):
    name = "mlx-test-tttd"

    def __call__(self, batch, state):
        return StepSignal(
            "train",
            "tttd",
            {"steps": int(state.get("steps", 0)) + 1},
            {"prepared": True},
            (1.0, -1.0),
            StepScheduling(unit="sample", batch_size="actual"),
        )


@register_step_preparer
class _MultiEpochPreparer(StepPreparer):
    name = "mlx-test-multi-epoch"

    def __call__(self, batch, state):
        return StepSignal(
            "train",
            "tttd",
            {},
            {},
            (1.0, -1.0),
            StepScheduling(unit="sample", batch_size="actual", epochs=3),
        )


def build_runtime(tmp_path: Path, *, moved: bool = True, kl_coef: float = 0.0) -> MLXRuntime:
    return MLXRuntime(FakeEngine(moved=moved), checkpoint_dir=str(tmp_path / "ckpt"), kl_coef=kl_coef)


def batch() -> PolicyBatch:
    return PolicyBatch("batch-1", (sample(1.0), sample(0.0)))


@pytest.mark.unit
def test_boot_names_the_weights_that_answer_the_first_request(tmp_path: Path) -> None:
    # A durable training record must name the weights that produced it, and
    # the first rollout is served before anything has been published.
    runtime = build_runtime(tmp_path)
    assert runtime.serving_runtime_load_id() == "fake-1"


@pytest.mark.unit
def test_preparation_runs_the_recipe_preparer_in_process(tmp_path: Path) -> None:
    runtime = build_runtime(tmp_path)
    prepared = runtime.prepare_training_step(batch(), "mlx-test-tttd", {"steps": 4}, 7)

    assert prepared.action == "train"
    assert prepared.next_algorithm_state == {"steps": 5}
    assert prepared.payload["advantages"] == (1.0, -1.0)
    assert prepared.payload["rollout_id"] == 7
    assert len(prepared.payload["samples"]) == 2


@pytest.mark.unit
def test_unsupported_scheduling_fails_before_training(tmp_path: Path) -> None:
    # A schedule this runtime cannot honour must be refused, not ignored:
    # silently training one epoch when three were asked for changes the
    # objective without saying so.
    runtime = build_runtime(tmp_path)
    with pytest.raises(RuntimeContractError, match="epochs=3"):
        runtime.prepare_training_step(batch(), "mlx-test-multi-epoch", {}, 0)


@pytest.mark.unit
def test_a_candidate_exports_without_changing_serving(tmp_path: Path) -> None:
    runtime = build_runtime(tmp_path)
    engine = runtime.engine
    before = engine.adapter_snapshot()
    prepared = runtime.prepare_training_step(batch(), "mlx-test-tttd", {}, 3)

    candidate = runtime.train_candidate(prepared.payload)

    assert isinstance(candidate, ModelCandidate)
    assert Path(candidate.checkpoint_path).is_dir()
    assert candidate.training_metrics["adapter_tensors_changed"] == 2
    # Serving still holds the pre-step weights: the trained parameters exist
    # only in the export until Reef selects them.
    assert engine.adapter_snapshot() == before
    assert runtime.serving_runtime_load_id() == "fake-1"


@pytest.mark.unit
def test_activation_moves_serving_to_the_selected_candidate(tmp_path: Path) -> None:
    runtime = build_runtime(tmp_path)
    engine = runtime.engine
    prepared = runtime.prepare_training_step(batch(), "mlx-test-tttd", {}, 0)
    candidate = runtime.train_candidate(prepared.payload)

    activated = runtime.activate_candidate(candidate)

    assert activated.candidate_id == candidate.candidate_id
    assert activated.runtime_load_id == "fake-2"
    assert runtime.serving_runtime_load_id() == "fake-2"
    assert engine.adapter_snapshot() == {"layer.lora_a": 1.0, "layer.lora_b": 1.0}


@pytest.mark.unit
def test_a_rejected_candidate_never_reaches_serving(tmp_path: Path) -> None:
    runtime = build_runtime(tmp_path)
    engine = runtime.engine
    before = engine.adapter_snapshot()
    prepared = runtime.prepare_training_step(batch(), "mlx-test-tttd", {}, 0)
    candidate = runtime.train_candidate(prepared.payload)

    runtime.reject_candidate(
        candidate,
        SelectionDecision("reject", "test", "1", "not better", EvaluationResult("test", "1", {})),
    )

    assert engine.adapter_snapshot() == before
    assert runtime.serving_runtime_load_id() == "fake-1"
    # A rejected candidate is gone: activating it afterwards must fail loudly
    # rather than resurrect weights Reef declined.
    with pytest.raises(RuntimeContractError, match="no pending mlx candidate"):
        runtime.activate_candidate(candidate)


@pytest.mark.unit
def test_a_step_that_moves_nothing_is_refused(tmp_path: Path) -> None:
    # The failure mode this guards against: a trainer reports success, writes
    # an adapter identical to the base, and Reef publishes it as a trained
    # update. The step must fail instead.
    runtime = build_runtime(tmp_path, moved=False)
    prepared = runtime.prepare_training_step(batch(), "mlx-test-tttd", {}, 0)

    with pytest.raises(RuntimeContractError, match="left every adapter tensor unchanged"):
        runtime.train_candidate(prepared.payload)

    assert runtime.engine.saved == []


@pytest.mark.unit
def test_the_frozen_base_kl_term_shifts_advantages_per_token(tmp_path: Path) -> None:
    runtime = build_runtime(tmp_path, kl_coef=0.5)
    prepared = runtime.prepare_training_step(batch(), "mlx-test-tttd", {}, 0)

    runtime.train_candidate(prepared.payload)

    rows = runtime.engine.trained_rows
    assert len(rows) == 2
    # rollout log-probs are (-0.5, -0.25) and the fake base returns -1.0, so
    # the per-token differences are (0.5, 0.75) with mean 0.625; every
    # advantage moves by kl_coef * (mean - difference).
    assert rows[0].advantages == pytest.approx((1.0 + 0.5 * 0.125, 1.0 + 0.5 * -0.125))
    assert rows[1].advantages == pytest.approx((-1.0 + 0.5 * 0.125, -1.0 + 0.5 * -0.125))


@pytest.mark.unit
def test_rollback_loads_a_published_adapter(tmp_path: Path) -> None:
    from reef.artifact.artifact import Artifact

    adapter = tmp_path / "published"
    adapter.mkdir()
    (adapter / "adapters.safetensors").write_bytes(b"weights")
    runtime = build_runtime(tmp_path)

    restored = runtime.restore_checkpoint(Artifact.local(adapter))

    assert restored == "fake-2"
    assert runtime.engine.loaded == [adapter]


@pytest.mark.unit
def test_the_factory_requires_a_checkpoint_directory() -> None:
    with pytest.raises(RuntimeContractError, match="checkpoint_dir"):
        RuntimeRegistry().build({"type": "mlx"}, model_path="fake/model")


@pytest.mark.unit
def test_the_factory_refuses_stale_sample_training() -> None:
    # Nothing in this runtime corrects for a batch produced by older weights,
    # so admitting one would train on a ratio it cannot compute.
    with pytest.raises(RuntimeContractError, match="exact-version"):
        RuntimeRegistry().build(
            {"type": "mlx", "checkpoint_dir": "/tmp/reef-mlx-test", "max_staleness": 4},
            model_path="fake/model",
        )


@pytest.mark.unit
def test_the_factory_refuses_template_kwargs_of_the_wrong_shape() -> None:
    with pytest.raises(RuntimeContractError, match="chat_template_kwargs"):
        RuntimeRegistry().build(
            {
                "type": "mlx",
                "checkpoint_dir": "/tmp/reef-mlx-test",
                "chat_template_kwargs": "enable_thinking=false",
            },
            model_path="fake/model",
        )


@pytest.mark.unit
def test_an_unsupported_loss_family_is_refused(tmp_path: Path) -> None:
    # The runtime implements one objective; training a recipe's data under a
    # different loss than it asked for would be a silent substitution.
    runtime = build_runtime(tmp_path)
    with pytest.raises(RuntimeContractError, match="loss family 'fake-family'"):
        runtime.prepare_training_step(batch(), "mlx-test-preparer", {}, 0)


@pytest.mark.unit
def test_inference_stays_closed_between_activation_and_the_durable_commit(tmp_path: Path) -> None:
    # Reopening at activation would let a request freeze the old artifact head
    # and then be answered by the new weights — a runtime-load mismatch.
    runtime = build_runtime(tmp_path)
    prepared = runtime.prepare_training_step(batch(), "mlx-test-tttd", {}, 0)
    candidate = runtime.train_candidate(prepared.payload)

    runtime.activate_candidate(candidate)

    assert runtime.inference_admission_status["open"] is False
    # The engine holds the new weights, but Reef has not published them yet.
    assert runtime.serving_runtime_load_id() == "fake-2"
    assert runtime.current_runtime_load_id() == "fake-1"

    runtime.reconcile_training_job(0, committed_training_job_id=candidate.training_job_id)

    assert runtime.inference_admission_status["open"] is True
    assert runtime.current_runtime_load_id() == "fake-2"


@pytest.mark.unit
def test_a_failed_step_restores_the_weights_that_were_serving(tmp_path: Path) -> None:
    runtime = build_runtime(tmp_path)
    engine = runtime.engine
    before = engine.adapter_snapshot()
    prepared = runtime.prepare_training_step(batch(), "mlx-test-tttd", {}, 0)

    def explode(rows):
        # Mutate first, then fail: the shape of a step that dies after the
        # optimizer has already touched the parameters.
        engine.weights = {name: value + 99.0 for name, value in engine.weights.items()}
        raise RuntimeError("metal exploded")

    engine.train_step = explode
    with pytest.raises(RuntimeError, match="metal exploded"):
        runtime.train_candidate(prepared.payload)

    assert engine.adapter_snapshot() == before
    assert runtime.inference_admission_status["open"] is True


@pytest.mark.unit
def test_the_frozen_base_pass_runs_with_inference_closed(tmp_path: Path) -> None:
    # Zeroing lora_b turns the live model into the bare base. A request
    # admitted during that window would be answered by the wrong weights.
    runtime = build_runtime(tmp_path, kl_coef=0.5)
    engine = runtime.engine
    observed: list[bool] = []

    def watching_base_log_probs(rows):
        observed.append(runtime.inference_admission_status["open"])
        return [[-1.0] * len(row.loss_mask) for row in rows]

    engine.base_log_probs = watching_base_log_probs
    prepared = runtime.prepare_training_step(batch(), "mlx-test-tttd", {}, 0)
    runtime.train_candidate(prepared.payload)

    assert observed == [False]


class _FakeRollout:
    """The engine's Rollout shape, without loading a model."""

    def __init__(self, *, topk=True, text="an answer", finish_reason="stop"):
        self.prompt_tokens = (11, 12, 13)
        self.output_tokens = (21, 22)
        self.rollout_log_probs = (-0.5, -0.25)
        self.text = text
        self.finish_reason = finish_reason
        self.topk_indices = ((21, 99), (22, 98)) if topk else ()
        self.topk_log_probs = ((-0.5, -3.0), (-0.25, -4.0)) if topk else ()


def _parse_xml_call(text: str, tools) -> dict:
    """The shape of mlx-lm's ``qwen3_coder.parse_tool_call``: one call body to name and arguments."""
    del tools
    _, _, body = text.partition("<function=")
    name, _, params = body.partition(">")
    arguments = {}
    for chunk in params.split("<parameter=")[1:]:
        key, _, value = chunk.partition(">")
        arguments[key] = value.split("</parameter>")[0].strip()
    if not name.strip():
        raise ValueError("no function")
    return {"name": name.strip(), "arguments": arguments}


class _FakeServingTokenizer:
    """What the backend reads off mlx-lm's tokenizer: the prompt tail, and the parser it matched."""

    def __init__(self, *, prompt_tail="<|im_start|>assistant\n", tool_calling=True, tool_parser=_parse_xml_call):
        self.prompt_tail = prompt_tail
        self.has_tool_calling = tool_calling
        self.tool_call_start = "<tool_call>"
        self.tool_call_end = "</tool_call>"
        self.tool_parser = tool_parser

    def decode(self, tokens):
        return self.prompt_tail


class _FakeEngineForServing:
    def __init__(self, rollout, tokenizer=None, pieces=None):
        self._rollout = rollout
        self.config = FakeEngineConfig()
        self.tokenizer = tokenizer or _FakeServingTokenizer()
        self.publications = 0
        self.template_kwargs = "unset"
        self.tools = "unset"
        # What a streamed generation hands out piece by piece; the text
        # split in two unless a test wants specific boundaries.
        text = rollout.text
        self.pieces = list(pieces) if pieces is not None else [text[: len(text) // 2], text[len(text) // 2 :]]
        self.streamed: list[str] = []
        self.cancelled_after: int | None = None
        # How many single-sequence completions the serving backend asked for.
        self.generate_calls = 0
        # Tickets for the engine's ContinuousBatcher surface, kept so the fake
        # still mirrors the real engine even though serving no longer batches.
        self._batch: list[object] = []
        self.pump_calls = 0

    def next_runtime_load_id(self) -> str:
        self.publications += 1
        return f"fake-{self.publications}"

    def render_prompt(self, messages, *, tools=None, template_kwargs=None):
        self.template_kwargs = template_kwargs
        self.tools = tools
        return [11, 12, 13]

    def generate(self, prompt_tokens, *, max_tokens=None, temperature=None):
        self.generate_calls += 1
        return self._rollout

    def submit_rollout(self, prompt_tokens, *, max_tokens=None, temperature=None):
        ticket = object()
        self._batch.append(ticket)
        return ticket

    def rollouts_pending(self):
        return bool(self._batch)

    def pump_rollouts(self):
        self.pump_calls += 1
        finished = [(ticket, self._rollout) for ticket in self._batch]
        self._batch = []
        return finished

    def generate_stream(self, prompt_tokens, *, listener, max_tokens=None, temperature=None):
        for index, piece in enumerate(self.pieces):
            if listener.cancelled():
                self.cancelled_after = index
                self._rollout.finish_reason = "cancelled"
                return self._rollout
            self.streamed.append(piece)
            listener.emit(piece)
        return self._rollout


def _serve(payload, *, topk=True, text="an answer", finish_reason="stop", tokenizer=None):
    import asyncio

    from reef.artifact.artifact import Artifact
    from reef.train.mlx_backend.inference import MLXInferenceBackend

    runtime = MLXRuntime(
        _FakeEngineForServing(_FakeRollout(topk=topk, text=text, finish_reason=finish_reason), tokenizer),
        checkpoint_dir="/tmp/reef-mlx-serving-test",
    )
    backend = MLXInferenceBackend(runtime)
    return asyncio.run(backend.inference(Artifact.local(Path("/tmp")), "/v1/chat/completions", payload))


@pytest.mark.unit
def test_a_served_response_carries_the_tensors_that_make_it_trainable() -> None:
    response = _serve({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8})
    training = response["training"]

    # Full sequence, mask over the response only: what policy_row_violation
    # checks before a record can become a sample.
    assert training["tokens"] == [11, 12, 13, 21, 22]
    assert training["loss_mask"] == [1, 1]
    assert training["rollout_log_probs"] == [-0.5, -0.25]
    assert training["runtime_load_id"] == response["choices"][0]["meta_info"]["runtime_load_id"]


@pytest.mark.unit
def test_concurrent_completions_serialize_as_single_sequences_and_all_resolve() -> None:
    """Buffered requests are sampled one sequence at a time, not batched.

    mlx-lm cannot continuously batch the hybrid cache, so the serving backend
    drives the engine as a single sequence per request, serialized under the
    engine lock. Every concurrent request must still resolve with its trainable
    record intact, and the engine sees exactly one ``generate`` per request.
    """
    import asyncio

    from reef.artifact.artifact import Artifact
    from reef.train.mlx_backend.inference import MLXInferenceBackend

    engine = _FakeEngineForServing(_FakeRollout())
    backend = MLXInferenceBackend(MLXRuntime(engine, checkpoint_dir="/tmp/reef-mlx-serial-test"))

    async def wave(n: int) -> list[dict]:
        calls = [
            backend.inference(
                Artifact.local(Path("/tmp")),
                "/v1/chat/completions",
                {"messages": [{"role": "user", "content": f"q{i}"}]},
            )
            for i in range(n)
        ]
        return await asyncio.gather(*calls)

    async def run() -> None:
        first = await wave(4)
        assert len(first) == 4
        # Every concurrent request resolved with the trainable record intact.
        assert all(r["training"]["tokens"] == [11, 12, 13, 21, 22] for r in first)
        # One single-sequence completion per request, no batch pump.
        assert engine.generate_calls == 4
        assert engine.pump_calls == 0

        # A second wave resolves the same way; each request is its own sequence.
        second = await wave(3)
        assert len(second) == 3
        assert all(r["training"]["response_length"] == 2 for r in second)
        assert engine.generate_calls == 7

    asyncio.run(run())


@pytest.mark.unit
def test_captured_candidates_reach_the_wire_when_the_engine_records_them() -> None:
    # A distillation objective trains on the candidate set the policy
    # considered; nothing downstream can rebuild it after generation.
    training = _serve({"messages": [{"role": "user", "content": "hi"}]})["training"]

    assert training["topk_indices"] == [[21, 99], [22, 98]]
    assert training["topk_log_probs"] == [[-0.5, -3.0], [-0.25, -4.0]]


@pytest.mark.unit
def test_no_candidate_channel_when_capture_is_off() -> None:
    # An on-policy objective needs none, and an absent key is what
    # make_policy_sample reads as "not captured".
    training = _serve({"messages": [{"role": "user", "content": "hi"}]}, topk=False)["training"]

    assert "topk_indices" not in training
    assert "topk_log_probs" not in training


@pytest.mark.unit
def test_a_request_steers_the_chat_template() -> None:
    """A reasoning model's ``<think>`` block is response tokens, so whether the
    template opens one has to be a per-request decision."""
    import asyncio

    from reef.artifact.artifact import Artifact
    from reef.train.mlx_backend.inference import MLXInferenceBackend

    engine = _FakeEngineForServing(_FakeRollout())
    backend = MLXInferenceBackend(MLXRuntime(engine, checkpoint_dir="/tmp/reef-mlx-template-test"))
    asyncio.run(
        backend.inference(
            Artifact.local(Path("/tmp")),
            "/v1/chat/completions",
            {
                "messages": [{"role": "user", "content": "hi"}],
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
    )
    assert engine.template_kwargs == {"enable_thinking": False}


@pytest.mark.unit
def test_openai_string_tool_call_arguments_render_as_a_mapping() -> None:
    """An OpenAI client serialises a tool call's ``arguments`` as a JSON string;
    the Qwen3 template needs a mapping to iterate. The engine parses the string
    before applying the template so a real OpenAI agent (Hermes) round-trips."""
    # Imported from ``messages`` (not ``engine``) so it runs without the
    # ``mlx`` extra installed, the way the rest of the adapter imports.
    from reef.train.mlx_backend.messages import prepare_messages as _prepare_messages

    messages = [
        {"role": "user", "content": "solve it"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "homework/0.txt"}'},
                }
            ],
        },
    ]
    prepared = _prepare_messages(messages)
    assert prepared[1]["tool_calls"][0]["function"]["arguments"] == {"path": "homework/0.txt"}

    # A dict is left alone; a non-JSON string and a message with no tool_calls
    # are untouched rather than dropped.
    already = [{"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": {"k": 1}}}]}]
    assert _prepare_messages(already)[0]["tool_calls"][0]["function"]["arguments"] == {"k": 1}
    malformed = [{"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": "not json"}}]}]
    assert _prepare_messages(malformed)[0]["tool_calls"][0]["function"]["arguments"] == "not json"
    assert _prepare_messages([{"role": "user", "content": "hi"}]) == [{"role": "user", "content": "hi"}]


@pytest.mark.unit
def test_a_request_without_template_kwargs_leaves_the_deployment_default() -> None:
    engine = _FakeEngineForServing(_FakeRollout())
    import asyncio

    from reef.artifact.artifact import Artifact
    from reef.train.mlx_backend.inference import MLXInferenceBackend

    backend = MLXInferenceBackend(MLXRuntime(engine, checkpoint_dir="/tmp/reef-mlx-template-test"))
    asyncio.run(
        backend.inference(
            Artifact.local(Path("/tmp")),
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "hi"}]},
        )
    )
    assert engine.template_kwargs is None


@pytest.mark.unit
def test_a_declared_toolset_reaches_the_chat_template() -> None:
    """The template renders the schemas *and* the one call syntax it parses
    back. A request whose tools are dropped leaves the model guessing both."""
    import asyncio

    from reef.artifact.artifact import Artifact
    from reef.train.mlx_backend.inference import MLXInferenceBackend

    tools = [{"type": "function", "function": {"name": "read_file", "parameters": {}}}]
    engine = _FakeEngineForServing(_FakeRollout())
    backend = MLXInferenceBackend(MLXRuntime(engine, checkpoint_dir="/tmp/reef-mlx-tools-test"))
    asyncio.run(
        backend.inference(
            Artifact.local(Path("/tmp")),
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "hi"}], "tools": tools},
        )
    )
    assert engine.tools == tools


@pytest.mark.unit
def test_a_request_without_tools_declares_none() -> None:
    engine = _FakeEngineForServing(_FakeRollout())
    import asyncio

    from reef.artifact.artifact import Artifact
    from reef.train.mlx_backend.inference import MLXInferenceBackend

    backend = MLXInferenceBackend(MLXRuntime(engine, checkpoint_dir="/tmp/reef-mlx-tools-test"))
    asyncio.run(
        backend.inference(
            Artifact.local(Path("/tmp")),
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "hi"}]},
        )
    )
    assert engine.tools is None


@pytest.mark.unit
def test_a_malformed_tools_field_is_refused() -> None:
    from reef.runtime.inference import UpstreamStatusError

    with pytest.raises(UpstreamStatusError, match="tools"):
        _serve({"messages": [{"role": "user", "content": "hi"}], "tools": {"name": "read_file"}})


@pytest.mark.unit
def test_a_malformed_template_kwargs_field_is_refused() -> None:
    from reef.runtime.inference import UpstreamStatusError

    with pytest.raises(UpstreamStatusError, match="chat_template_kwargs"):
        _serve({"messages": [{"role": "user", "content": "hi"}], "chat_template_kwargs": "no-think"})


@pytest.mark.unit
def test_the_buffered_path_refuses_a_stream_request_rather_than_faking_one() -> None:
    from reef.runtime.inference import UpstreamStatusError

    with pytest.raises(UpstreamStatusError, match="inference_stream"):
        _serve({"messages": [{"role": "user", "content": "hi"}], "stream": True})


@pytest.mark.unit
def test_an_empty_completion_is_refused_so_a_grid_cannot_stall() -> None:
    from reef.runtime.inference import UpstreamStatusError

    rollout = _FakeRollout()
    rollout.output_tokens = ()
    rollout.rollout_log_probs = ()
    rollout.topk_indices = ()
    rollout.topk_log_probs = ()

    import asyncio

    from reef.artifact.artifact import Artifact
    from reef.train.mlx_backend.inference import MLXInferenceBackend

    runtime = MLXRuntime(_FakeEngineForServing(rollout), checkpoint_dir="/tmp/reef-mlx-serving-test")
    backend = MLXInferenceBackend(runtime)
    with pytest.raises(UpstreamStatusError, match="no response tokens"):
        asyncio.run(backend.inference(Artifact.local(Path("/tmp")), "/v1/chat/completions", {"messages": [{}]}))


@register_step_preparer
class _OpenClawRLPreparer(StepPreparer):
    name = "mlx-test-openclawrl"

    def __call__(self, batch, state):
        return StepSignal(
            "train",
            "openclawrl",
            {},
            {},
            tuple(sample.reward for sample in batch.samples),
            # The real preparer leaves scheduling at its default, which names
            # the backend's own batch size.
            StepScheduling(),
        )


@register_step_preparer
class _SubBatchedPreparer(StepPreparer):
    name = "mlx-test-subbatched"

    def __call__(self, batch, state):
        return StepSignal("train", "tttd", {}, {}, (1.0, -1.0), StepScheduling(unit="sample", batch_size=4))


def _distillation_sample(*, topk=True, teacher=True) -> PolicySample:
    extras = {}
    if teacher:
        extras["teacher_cands"] = ({"hint": "Be terse.", "teacher_tokens": [7, 8, 1, 2]},)
    return PolicySample(
        source_agent_record_id="turn-1",
        tokens=(5, 6, 1, 2),
        loss_mask=(1, 1),
        rollout_log_probs=(-0.5, -0.25),
        reward=1.0,
        topk_indices=((1, 3), (2, 4)) if topk else (),
        topk_log_probs=((-0.5, -2.0), (-0.25, -3.0)) if topk else (),
        extras=extras,
    )


@pytest.mark.unit
def test_a_configured_batch_size_is_accepted_but_sub_batching_is_not(tmp_path: Path) -> None:
    # "configured" names a backend batch size that a single process does not
    # have, so the reserved batch is the step either way. An explicit integer
    # really does mean several steps, which this runtime cannot honour.
    runtime = build_runtime(tmp_path)
    batch = PolicyBatch("b", (_distillation_sample(),))

    prepared = runtime.prepare_training_step(batch, "mlx-test-openclawrl", {}, 0)
    assert prepared.action == "train"

    with pytest.raises(RuntimeContractError, match="batch_size=4"):
        runtime.prepare_training_step(batch, "mlx-test-subbatched", {}, 0)


@pytest.mark.unit
def test_the_distillation_objective_refuses_a_batch_with_no_captured_candidates(tmp_path: Path) -> None:
    # Training a distillation objective on rollouts that recorded no candidate
    # set would silently optimise nothing; say which setting is missing.
    runtime = build_runtime(tmp_path)
    batch = PolicyBatch("b", (_distillation_sample(topk=False),))
    prepared = runtime.prepare_training_step(batch, "mlx-test-openclawrl", {}, 0)

    with pytest.raises(RuntimeContractError, match="capture_topk"):
        runtime.train_candidate(prepared.payload)


@pytest.mark.unit
def test_the_distillation_objective_refuses_a_batch_with_no_teacher(tmp_path: Path) -> None:
    runtime = build_runtime(tmp_path)
    batch = PolicyBatch("b", (_distillation_sample(teacher=False),))
    prepared = runtime.prepare_training_step(batch, "mlx-test-openclawrl", {}, 0)

    with pytest.raises(RuntimeContractError, match="teacher_cands"):
        runtime.train_candidate(prepared.payload)


@pytest.mark.unit
def test_the_openclawrl_family_reaches_the_distillation_step(tmp_path: Path) -> None:
    runtime = build_runtime(tmp_path)
    engine = runtime.engine
    batch = PolicyBatch("b", (_distillation_sample(),))
    prepared = runtime.prepare_training_step(batch, "mlx-test-openclawrl", {}, 0)

    candidate = runtime.train_candidate(prepared.payload)

    # The distillation step ran, not the policy-gradient one.
    assert engine.distillation_rows is not None
    assert engine.trained_rows == []
    row = engine.distillation_rows[0]
    assert row.reward == 1.0
    assert row.candidates[0].hint == "Be terse."
    assert candidate.training_metrics["w_opd"] == 1.0


# ------------------------------------------------------- reading the reply back

READ_FILE = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        },
    }
]
THINKING_TAIL = "<|im_start|>assistant\n<think>\n"
NO_THINKING_TAIL = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
XML_CALL = "<tool_call>\n<function=read_file>\n<parameter=path>\nREADME.md\n</parameter>\n</function>\n</tool_call>"


@pytest.mark.unit
def test_reasoning_rides_reasoning_content_and_the_tensors_stay_whole() -> None:
    """The split is presentation: the trained tokens are still the whole sample."""
    response = _serve(
        {"messages": [{"role": "user", "content": "hi"}]},
        text="pondering</think>\nThe answer is 36.",
        tokenizer=_FakeServingTokenizer(prompt_tail=THINKING_TAIL),
    )
    message = response["choices"][0]["message"]

    assert message == {"role": "assistant", "content": "The answer is 36.", "reasoning_content": "pondering"}
    assert response["training"]["tokens"] == [11, 12, 13, 21, 22]
    assert response["training"]["loss_mask"] == [1, 1]


@pytest.mark.unit
def test_truncated_reasoning_never_comes_back_as_the_reply() -> None:
    """A pre-opened ``<think>`` with no closing tag hit the cap mid-thought: no content, or a judge
    scores chain-of-thought as the agent's answer."""
    response = _serve(
        {"messages": [{"role": "user", "content": "hi"}]},
        text="Okay, first I should read the file",
        finish_reason="length",
        tokenizer=_FakeServingTokenizer(prompt_tail=THINKING_TAIL),
    )
    message = response["choices"][0]["message"]

    assert message["content"] == ""
    assert message["reasoning_content"] == "Okay, first I should read the file"
    assert response["choices"][0]["finish_reason"] == "length"


@pytest.mark.unit
def test_a_request_that_disabled_thinking_is_read_as_plain_text() -> None:
    """Same model, same template: ``enable_thinking: false`` closes the block in the prompt, so
    an unclosed sample is an answer, not truncated reasoning. Decided per request."""
    response = _serve(
        {"messages": [{"role": "user", "content": "hi"}], "chat_template_kwargs": {"enable_thinking": False}},
        text="The answer is 36.",
        tokenizer=_FakeServingTokenizer(prompt_tail=NO_THINKING_TAIL),
    )

    assert response["choices"][0]["message"] == {"role": "assistant", "content": "The answer is 36."}


@pytest.mark.unit
def test_a_call_in_the_templates_syntax_reaches_the_wire_as_tool_calls() -> None:
    """The template asked for calls in its syntax; the reply carries them as ``tool_calls`` so the
    harness takes its tool branch, and feeds them back structurally instead of echoing markup."""
    response = _serve(
        {"messages": [{"role": "user", "content": "hi"}], "tools": READ_FILE}, text=f"I'll read it.\n{XML_CALL}"
    )
    choice = response["choices"][0]

    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == "I'll read it."
    [call] = choice["message"]["tool_calls"]
    assert call["function"] == {"name": "read_file", "arguments": '{"path": "README.md"}'}
    assert call["id"].startswith("call_")
    assert response["training"]["tokens"] == [11, 12, 13, 21, 22]


@pytest.mark.unit
def test_every_call_in_a_reply_is_parsed_in_order() -> None:
    second = XML_CALL.replace("README.md", "NOTES.md")
    response = _serve(
        {"messages": [{"role": "user", "content": "hi"}], "tools": READ_FILE}, text=f"{XML_CALL}\n{second}"
    )
    calls = response["choices"][0]["message"]["tool_calls"]

    assert [json.loads(call["function"]["arguments"])["path"] for call in calls] == ["README.md", "NOTES.md"]
    assert response["choices"][0]["message"]["content"] is None


@pytest.mark.unit
def test_call_markup_without_a_declared_toolset_stays_text() -> None:
    """No tools in the prompt means no call syntax was stated; the markup is what the model said."""
    response = _serve({"messages": [{"role": "user", "content": "hi"}]}, text=XML_CALL)

    assert response["choices"][0]["message"] == {"role": "assistant", "content": XML_CALL}
    assert response["choices"][0]["finish_reason"] == "stop"


@pytest.mark.unit
def test_tool_choice_none_declares_tools_for_context_only() -> None:
    response = _serve(
        {"messages": [{"role": "user", "content": "hi"}], "tools": READ_FILE, "tool_choice": "none"}, text=XML_CALL
    )

    assert "tool_calls" not in response["choices"][0]["message"]


@pytest.mark.unit
def test_a_template_mlx_lm_has_no_parser_for_leaves_the_reply_as_text() -> None:
    response = _serve(
        {"messages": [{"role": "user", "content": "hi"}], "tools": READ_FILE},
        text=XML_CALL,
        tokenizer=_FakeServingTokenizer(tool_calling=False),
    )

    assert response["choices"][0]["message"]["content"] == XML_CALL


@pytest.mark.unit
def test_a_call_that_never_closes_fails_the_request() -> None:
    """Half a call that hit the token cap is not a reply; the agent retries a rollout."""
    from reef.runtime.inference import UpstreamStatusError

    with pytest.raises(UpstreamStatusError) as caught:
        _serve(
            {"messages": [{"role": "user", "content": "hi"}], "tools": READ_FILE},
            text="<tool_call>\n<function=read_file>\n<parameter=path>\nREAD",
            finish_reason="length",
        )
    assert caught.value.status == 502
    assert "never closed" in str(caught.value)


@pytest.mark.unit
def test_a_call_opened_inside_thinking_ends_the_thinking() -> None:
    """A thinking model that starts acting has stopped thinking (Ollama's Qwen3 parser agrees):
    the call is a call, not the tail of truncated reasoning."""
    response = _serve(
        {"messages": [{"role": "user", "content": "hi"}], "tools": READ_FILE},
        text=f"I should look first.\n{XML_CALL}",
        tokenizer=_FakeServingTokenizer(prompt_tail=THINKING_TAIL),
    )
    message = response["choices"][0]["message"]

    assert message["reasoning_content"] == "I should look first."
    assert message["tool_calls"][0]["function"]["name"] == "read_file"


@pytest.mark.unit
def test_the_real_qwen3_coder_parser_coerces_arguments_by_schema() -> None:
    """mlx-lm's parser for the Qwen3.8 template, driven exactly as the backend drives it: the XML
    the model writes becomes typed arguments, with the tail of each value trimmed."""
    qwen3_coder = pytest.importorskip("mlx_lm.tool_parsers.qwen3_coder")
    from types import SimpleNamespace

    from reef.train.mlx_backend.inference import MLXToolCallParser

    tokenizer = SimpleNamespace(
        has_tool_calling=True,
        tool_call_start=qwen3_coder.tool_call_start,
        tool_call_end=qwen3_coder.tool_call_end,
        tool_parser=qwen3_coder.parse_tool_call,
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "set_uv_threshold",
                "parameters": {"type": "object", "properties": {"index": {"type": "integer"}}},
            },
        }
    ]
    parser = MLXToolCallParser.for_tokenizer(tokenizer, tools)
    text = "I'll set it.\n<tool_call>\n<function=set_uv_threshold>\n<parameter=index>\n5\n</parameter>\n</function>\n</tool_call>"

    remaining, [call] = parser.parse_non_stream(text)

    assert remaining == "I'll set it."
    assert call.name == "set_uv_threshold"
    assert call.parameters == {"index": 5}


# ------------------------------------------------------------------- streaming


def _serve_stream(payload, *, text="an answer", finish_reason="stop", tokenizer=None, pieces=None, take=None):
    """Stream one completion; the SSE frames, the stream object, and the engine.

    ``take`` stops consuming after that many frames and closes the stream,
    the way the service does when a client goes away."""
    import asyncio

    from reef.artifact.artifact import Artifact
    from reef.train.mlx_backend.inference import MLXInferenceBackend

    engine = _FakeEngineForServing(_FakeRollout(text=text, finish_reason=finish_reason), tokenizer, pieces=pieces)
    backend = MLXInferenceBackend(MLXRuntime(engine, checkpoint_dir="/tmp/reef-mlx-stream-test"))

    async def run():
        stream = await backend.inference_stream(
            Artifact.local(Path("/tmp")), "/v1/chat/completions", {**payload, "stream": True}
        )
        frames = []
        async for frame in stream.chunks:
            frames.append(frame)
            if take is not None and len(frames) >= take:
                break
        await stream.close()
        return frames, stream, backend

    frames, stream, backend = asyncio.run(run())
    return frames, stream, engine, backend


def _events(frames):
    """Every JSON chunk on the wire, in order; ``"[DONE]"`` for the terminator."""
    events = []
    for frame in frames:
        for line in frame.decode().split("\n"):
            if line.startswith("data: "):
                data = line[len("data: ") :]
                events.append(data if data == "[DONE]" else json.loads(data))
    return events


def _deltas(events, key):
    return "".join(
        event["choices"][0]["delta"].get(key, "") for event in events if isinstance(event, dict) and "choices" in event
    )


@pytest.mark.unit
def test_a_stream_shows_the_text_as_it_is_sampled_and_records_the_whole_sample() -> None:
    frames, stream, engine, _ = _serve_stream({"messages": [{"role": "user", "content": "hi"}]}, text="an answer")
    events = _events(frames)

    assert events[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert events[0]["object"] == "chat.completion.chunk"
    assert _deltas(events, "content") == "an answer"
    assert engine.streamed == ["an a", "nswer"]
    terminal = events[-2]["choices"][0]
    assert terminal["finish_reason"] == "stop" and terminal["meta_info"]["runtime_load_id"] == "fake-1"
    assert events[-1] == "[DONE]"
    # The record is what the buffered path would have built, tensors and all.
    assert stream.record_response_pending is True
    assert stream.record_response["choices"][0]["message"] == {"role": "assistant", "content": "an answer"}
    assert stream.record_response["training"]["tokens"] == [11, 12, 13, 21, 22]
    assert stream.record_response["id"] == events[0]["id"]


@pytest.mark.unit
def test_a_stream_splits_reasoning_from_content_as_it_goes() -> None:
    frames, stream, _, _ = _serve_stream(
        {"messages": [{"role": "user", "content": "hi"}]},
        text="pondering</think>\nThe answer is 36.",
        pieces=["ponder", "ing</thi", "nk>\nThe answer", " is 36."],
        tokenizer=_FakeServingTokenizer(prompt_tail=THINKING_TAIL),
    )
    events = _events(frames)

    assert _deltas(events, "reasoning_content") == "pondering"
    assert _deltas(events, "content") == "The answer is 36."
    assert stream.record_response["choices"][0]["message"]["reasoning_content"] == "pondering"


@pytest.mark.unit
def test_a_streamed_tool_call_is_held_and_arrives_structurally() -> None:
    """The wire never shows raw call markup: text before the marker streams, the call is
    parsed whole at the end and sent as one ``tool_calls`` delta before the terminal."""
    text = f"I'll read it.\n{XML_CALL}"
    frames, stream, _, _ = _serve_stream(
        {"messages": [{"role": "user", "content": "hi"}], "tools": READ_FILE},
        text=text,
        pieces=[
            "I'll read",
            " it.\n<tool_",
            "call>\n<function=read_file>\n",
            "<parameter=path>\nREADME.md\n</parameter>\n</function>\n</tool_call>",
        ],
    )
    events = _events(frames)
    wire = b"".join(frames).decode()

    assert "<tool_call>" not in wire and "<function=" not in wire
    assert _deltas(events, "content") == "I'll read it."
    [call_event] = [
        e for e in events if isinstance(e, dict) and "tool_calls" in e.get("choices", [{}])[0].get("delta", {})
    ]
    [call] = call_event["choices"][0]["delta"]["tool_calls"]
    assert call["index"] == 0 and call["function"] == {"name": "read_file", "arguments": '{"path": "README.md"}'}
    assert call["id"] == stream.record_response["choices"][0]["message"]["tool_calls"][0]["id"]
    assert events[-2]["choices"][0]["finish_reason"] == "tool_calls"


@pytest.mark.unit
def test_closing_a_stream_early_stops_the_engine_and_frees_the_backend() -> None:
    import asyncio

    from reef.artifact.artifact import Artifact

    _, stream, engine, backend = _serve_stream(
        {"messages": [{"role": "user", "content": "hi"}]},
        text="a b c d",
        pieces=["a ", "b ", "c ", "d"],
        take=2,
    )

    assert stream.record_response is None  # nothing complete, nothing to record
    assert len(engine.streamed) <= 4
    # The lock was released after the generation stopped: the next request runs.
    again = asyncio.run(
        backend.inference(
            Artifact.local(Path("/tmp")), "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}
        )
    )
    assert again["choices"][0]["finish_reason"] in ("stop", "cancelled")


@pytest.mark.unit
def test_a_stream_of_no_response_tokens_fails_before_the_terminal() -> None:
    import asyncio

    from reef.artifact.artifact import Artifact
    from reef.runtime.inference import UpstreamStatusError
    from reef.train.mlx_backend.inference import MLXInferenceBackend

    rollout = _FakeRollout(text="")
    rollout.output_tokens = ()
    rollout.rollout_log_probs = ()
    rollout.topk_indices = ()
    rollout.topk_log_probs = ()
    engine = _FakeEngineForServing(rollout, pieces=[])
    backend = MLXInferenceBackend(MLXRuntime(engine, checkpoint_dir="/tmp/reef-mlx-stream-test"))

    async def run():
        stream = await backend.inference_stream(
            Artifact.local(Path("/tmp")),
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
        frames = []
        with pytest.raises(UpstreamStatusError, match="no response tokens"):
            async for frame in stream.chunks:
                frames.append(frame)
        return frames, stream

    frames, stream = asyncio.run(run())
    assert b"[DONE]" not in b"".join(frames)
    assert stream.record_response is None
