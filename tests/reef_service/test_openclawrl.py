"""OpenClaw-RL processor (in-processor judging), backend preparation, and recipe."""

from __future__ import annotations

import json
import time

import pytest
from reef_service.runtime_stubs import StubTrainingRuntime

from recipes.openclawrl import OpenClawRLProcessor, OpenClawRLRecipe
from recipes.openclawrl.sessions import SessionIndex
from recipes.openclawrl.turns import TurnJudgment
from reef.core import AgentRecord, RequestType
from reef.surface import Surface, WeightInferenceHooks, WeightLoader
from reef.train.processors.computed import JudgingWorker
from reef.train.slime_backend.backend import SlimeTrainingBackend
from reef.train.slime_backend.reef_adapters.preparation import prepare_slime_step
from reef.train.types import PolicyBatch, PolicySample, ProcessorContext


class FakeWorker:
    """Duck-typed JudgingWorker: captures jobs, replays scripted judgments."""

    def __init__(self) -> None:
        self.jobs = []
        self.closed = False
        self._verdicts = []

    def submit(self, job) -> bool:
        self.jobs.append(job)
        return True

    def poll(self):
        judgments, self._verdicts = self._verdicts, []
        return judgments

    def close(self) -> None:
        self.closed = True

    def push(self, judgment) -> None:
        self._verdicts.append(judgment)


def _turn(
    agent_record_id: str,
    messages,
    *,
    tools=({"type": "function", "function": {"name": "write"}},),
    tokens=(1, 2, 3, 4),
    loss_mask=(1, 1),
    logprobs=(-0.1, -0.2),
    runtime_load_id="v1",
    topk_rows=None,
):
    training = {
        "tokens": list(tokens),
        "loss_mask": list(loss_mask),
        "rollout_log_probs": list(logprobs),
        # The paper objective requires the generation-time top-K capture on
        # every response token.
        "topk_indices": [[1, 2] for _ in loss_mask] if topk_rows is None else topk_rows,
        "topk_log_probs": [[-0.1, -2.0] for _ in loss_mask],
        "runtime_load_id": runtime_load_id,
        "response_length": len(loss_mask),
    }
    return AgentRecord.create(
        scenario="s",
        request_type=RequestType.INFERENCE,
        payload={
            "messages": list(messages),
            "tools": list(tools),
            "response": {
                "choices": [{"message": {"role": "assistant", "content": "a"}}],
                "training": training,
            },
        },
        agent_record_id=agent_record_id,
    )


def _successor(prior_messages, next_state: str):
    return [*prior_messages, {"role": "assistant", "content": "a"}, {"role": "user", "content": next_state}]


def _processor(batch_size: int = 1, *, worker=None, ttl_s: float = 900.0, clients=None):
    context = ProcessorContext("s", {"batch_size": batch_size, "session_ttl_s": ttl_s})
    return OpenClawRLProcessor(context, worker=worker, clients=clients)


Q1 = [{"role": "user", "content": "q1"}]
Q2 = [{"role": "user", "content": "q2"}]
#: The un-enhanced anchor for ``_turn``'s native ids — every trained sample
#: carries at least one candidate, so scoring judgments ship this.
ANCHOR = ({"hint": "", "teacher_tokens": [1, 2, 3, 4]},)


@pytest.mark.unit
def test_next_state_binds_by_trace_matching_and_verdicts_batch() -> None:
    worker = FakeWorker()
    processor = _processor(batch_size=1, worker=worker)
    processor.ingest(_turn("t1", Q1))
    assert not worker.jobs  # no successor yet
    processor.ingest(_turn("t2", _successor(Q1, "Great, that works!")))

    assert [job.receipt for job in worker.jobs] == ["t1"]
    assert worker.jobs[0].next_state_text == "Great, that works!"
    assert worker.jobs[0].next_state_role == "user"
    assert not processor.ready()

    worker.push(TurnJudgment("t1", score=1.0, teacher_cands=ANCHOR))
    assert processor.ready()
    batch = processor.build_batch()
    assert isinstance(batch, PolicyBatch)
    assert [sample.reward for sample in batch.samples] == [1.0]
    assert batch.samples[0].source_agent_record_id == "t1"
    processor.acknowledge(batch.batch_id)
    assert not processor.ready()
    decision = processor.retention_decision()
    assert "t1" in decision.releasable_agent_record_ids
    assert "t2" in decision.protected_agent_record_ids  # still the session's open turn


@pytest.mark.unit
def test_interleaved_sessions_with_distinct_prefixes_stay_separate() -> None:
    worker = FakeWorker()
    processor = _processor(worker=worker)
    processor.ingest(_turn("a1", Q1))
    processor.ingest(_turn("b1", Q2))
    processor.ingest(_turn("b2", _successor(Q2, "that crashes")))
    processor.ingest(_turn("a2", _successor(Q1, "works great")))

    bound = {job.receipt: job.next_state_text for job in worker.jobs}
    assert bound == {"b1": "that crashes", "a1": "works great"}


@pytest.mark.unit
def test_side_calls_and_retried_turns_are_terminal() -> None:
    worker = FakeWorker()
    processor = _processor(worker=worker)
    side = AgentRecord.create(
        scenario="s",
        request_type=RequestType.INFERENCE,
        payload={
            "messages": [{"role": "user", "content": "summarize the session"}],
            "response": {"choices": [{"message": {"role": "assistant", "content": "title"}}]},
        },
        agent_record_id="side",
    )
    processor.ingest(side)  # no tools: auxiliary call
    processor.ingest(_turn("t1", Q1))
    processor.ingest(_turn("t1-retry", Q1))  # identical request: retried logical turn

    releasable = processor.retention_decision().releasable_agent_record_ids
    assert {"side", "t1-retry"} <= set(releasable)
    assert "t1" in processor.retention_decision().protected_agent_record_ids
    assert not worker.jobs


@pytest.mark.unit
def test_session_ttl_flushes_the_final_turn_terminal() -> None:
    worker = FakeWorker()
    processor = _processor(worker=worker, ttl_s=1e-6)
    processor.ingest(_turn("t1", Q1))
    time.sleep(0.001)
    assert not processor.ready()  # catches up: the TTL pass runs
    decision = processor.retention_decision()
    assert "t1" in decision.releasable_agent_record_ids
    assert not processor.derivation_pending()


@pytest.mark.unit
def test_untrainable_verdicts_and_bad_tensors_are_terminal() -> None:
    worker = FakeWorker()
    processor = _processor(worker=worker)
    processor.ingest(_turn("declined", Q1))
    processor.ingest(_turn("cont1", _successor(Q1, "meh")))
    worker.push(TurnJudgment("declined"))  # judge declined: no RL, no directive
    assert not processor.ready()
    assert "declined" in processor.retention_decision().releasable_agent_record_ids

    # ragged top-K capture: one row missing → the select loss cannot run
    processor.ingest(_turn("ragged", Q2, topk_rows=[[1, 2]]))
    processor.ingest(_turn("cont2", _successor(Q2, "ok")))
    worker.push(TurnJudgment("ragged", score=-1.0, teacher_cands=ANCHOR))
    assert not processor.ready()
    assert "ragged" in processor.retention_decision().releasable_agent_record_ids


@pytest.mark.unit
def test_stale_runtime_load_ids_drop_from_pending_candidates() -> None:
    worker = FakeWorker()
    processor = _processor(batch_size=2, worker=worker)
    processor.ingest(_turn("old", Q1, runtime_load_id="v1"))
    processor.ingest(_turn("cont1", _successor(Q1, "ok")))
    processor.ingest(_turn("new", Q2, runtime_load_id="v2"))
    processor.ingest(_turn("cont2", _successor(Q2, "ok")))
    worker.push(TurnJudgment("old", score=1.0, teacher_cands=ANCHOR))
    worker.push(TurnJudgment("new", score=1.0, teacher_cands=ANCHOR))
    assert not processor.ready()  # the stale candidate was dropped, 1 < batch_size
    assert "old" in processor.retention_decision().releasable_agent_record_ids
    assert "new" in processor.retention_decision().protected_agent_record_ids


@pytest.mark.unit
def test_correlate_only_mode_never_trains_and_releases_everything() -> None:
    processor = _processor(worker=None)  # no prm_url, no injected worker
    processor.ingest(_turn("t1", Q1))
    processor.ingest(_turn("t2", _successor(Q1, "ok")))
    assert not processor.ready()
    assert "t1" in processor.retention_decision().releasable_agent_record_ids


@pytest.mark.unit
def test_derivation_pending_and_close_propagation() -> None:
    worker = FakeWorker()
    processor = _processor(worker=worker)
    assert not processor.derivation_pending()
    processor.ingest(_turn("t1", Q1))
    assert processor.derivation_pending()  # tracked turn awaits its TTL or successor
    processor.ingest(_turn("t2", _successor(Q1, "ok")))
    assert processor.derivation_pending()  # t1 in flight, t2 tracked
    processor.close()
    assert worker.closed


@pytest.mark.unit
def test_session_index_matching_ignores_tool_call_ids() -> None:
    index = SessionIndex(ttl_s=900.0)
    first = [{"role": "user", "content": "q"}]
    index.observe("t1", first, now=0.0)
    successor = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_REWRITTEN", "function": {"name": "write", "arguments": "{}"}}],
        },
        {"role": "tool", "content": "ok"},
    ]
    observation = index.observe("t2", successor, now=1.0)
    # The rewritten call id did not break the match, and the tool result is
    # t1's next state — the judge decides what it means, the index does not.
    assert observation.binding is not None
    assert observation.binding.receipt == "t1"
    assert observation.binding.next_state_role == "tool"


@pytest.mark.unit
def test_scoring_worker_threads_the_combine_dispatch() -> None:
    """End to end through a real worker thread with scripted judge/teacher."""

    class ScriptedJudge:
        async def evaluate(self, response_text, next_state, role):
            return 1.0

        async def evaluate_hint_votes(self, response_text, next_state, role):
            return []  # no hints accepted → RL-only → un-enhanced anchor

    class ScriptedTeacher:
        def __init__(self) -> None:
            self.calls = []

        async def candidate_tokens(self, enhanced_messages, *, tools=None, native_tokens, response_token_count):
            self.calls.append(enhanced_messages)
            return [101] + [int(v) for v in native_tokens[-response_token_count:]]

    teacher = ScriptedTeacher()
    # clients (not a scripted worker): the REAL worker thread runs the
    # processor's own judge against the scripted PRM clients.
    processor = _processor(clients=(ScriptedJudge(), teacher, None))
    processor.ingest(_turn("t1", Q1))
    processor.ingest(_turn("t2", _successor(Q1, "Great, that works!")))

    deadline = time.monotonic() + 10.0
    while not processor.ready() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert processor.ready(), "worker judgment never landed"
    batch = processor.build_batch()
    sample = batch.samples[0]
    assert sample.reward == 1.0
    # The RL-only anchor is the native ids verbatim, built without the
    # teacher (its construction cannot fail), for the Megatron teacher pass.
    assert teacher.calls == []
    assert sample.extras["teacher_cands"] == ({"hint": "", "teacher_tokens": [1, 2, 3, 4]},)
    processor.close()


@pytest.mark.unit
def test_backend_passes_raw_rewards_through_without_normalization() -> None:
    samples = tuple(
        PolicySample(
            str(index),
            (1, 2),
            (1,),
            (-0.1,),
            reward,
            topk_indices=((1, 2),),
            topk_log_probs=((-0.1, -0.9),),
        )
        for index, reward in enumerate((1.0, -1.0, 0.0, 1.0))
    )
    batch = PolicyBatch("s:openclawrl:1", samples)
    result = prepare_slime_step(batch, "openclawrl", {})
    assert result.payload is not None
    assert result.payload["advantages"] == [1.0, -1.0, 0.0, 1.0]
    assert result.payload["loss"] == "openclawrl"
    assert result.next_algorithm_state == {"steps": 1}


@pytest.mark.unit
def test_recipe_uses_weight_surface_and_builds_a_trainer() -> None:
    recipe = OpenClawRLRecipe(StubTrainingRuntime(), batch_size=4)
    surface = recipe.build_surface("s")
    assert type(surface) is Surface
    assert isinstance(surface.loader, WeightLoader)
    assert isinstance(surface.inference, WeightInferenceHooks)
    trainer = recipe.build("s", __import__("reef.records", fromlist=["RecordStore"]).RecordStore())
    assert isinstance(trainer.training_backend, SlimeTrainingBackend)
    assert trainer.training_backend.step_preparer == "openclawrl"
    trainer.close()


@pytest.mark.unit
def test_recipe_reads_config() -> None:
    configured = OpenClawRLRecipe.from_environment(
        {},
        config={"data": {"batch_size": 2, "prm_timeout_s": 3600.0}},
        runtime=StubTrainingRuntime(),
    )
    assert configured.batch_size == 2
    assert configured.prm_timeout_s == 3600.0


@pytest.mark.unit
def test_recipe_requires_the_tokenizer_next_to_prm_url() -> None:
    with pytest.raises(ValueError, match="prm_tokenizer_path"):
        OpenClawRLRecipe(StubTrainingRuntime(), prm_url="http://prm:23001")


@pytest.mark.unit
def test_stale_drop_uses_record_arrival_order_not_verdict_order() -> None:
    """A slow judgment for an old-version turn must not purge fresh candidates."""
    worker = FakeWorker()
    processor = _processor(batch_size=2, worker=worker)
    processor.ingest(_turn("old", Q1, runtime_load_id="v1"))
    processor.ingest(_turn("cont1", _successor(Q1, "ok")))
    processor.ingest(_turn("new", Q2, runtime_load_id="v2"))
    processor.ingest(_turn("cont2", _successor(Q2, "ok")))
    # Judgments land in the OPPOSITE order of generation: v2 first, v1 last.
    worker.push(TurnJudgment("new", score=1.0, teacher_cands=ANCHOR))
    worker.push(TurnJudgment("old", score=1.0, teacher_cands=ANCHOR))
    assert not processor.ready()
    assert "old" in processor.retention_decision().releasable_agent_record_ids
    assert "new" in processor.retention_decision().protected_agent_record_ids


@pytest.mark.unit
def test_echo_only_extension_never_binds_a_turn_to_its_own_reply() -> None:
    worker = FakeWorker()
    processor = _processor(worker=worker)
    processor.ingest(_turn("t1", Q1))
    # Continue/prefill shape: the successor carries ONLY the assistant echo.
    processor.ingest(_turn("t2", [*Q1, {"role": "assistant", "content": "the answer is"}]))
    assert not worker.jobs


@pytest.mark.unit
def test_sequential_sessions_with_identical_openers_stay_separate() -> None:
    worker = FakeWorker()
    processor = _processor(worker=worker)
    processor.ingest(_turn("a1", Q1))
    processor.ingest(_turn("a2", _successor(Q1, "session A feedback")))
    # Session B re-opens with the canonically identical first request. A has
    # progressed (its last request is longer), so this is a new session, not
    # a retry — and B's successor binds B's opener, never A's tail.
    processor.ingest(_turn("b1", Q1))
    processor.ingest(_turn("b2", _successor(Q1, "session B feedback")))
    bound = {job.receipt: job.next_state_text for job in worker.jobs}
    assert bound == {"a1": "session A feedback", "b1": "session B feedback"}


@pytest.mark.unit
def test_reports_are_terminal_on_sight() -> None:
    processor = _processor(worker=FakeWorker())
    processor.ingest(
        AgentRecord.create(scenario="s", request_type=RequestType.REPORT, payload={}, agent_record_id="r1")
    )
    releasable = processor.retention_decision().releasable_agent_record_ids
    assert "r1" in set(releasable)


@pytest.mark.unit
def test_rejected_submission_retires_the_turn() -> None:
    class RefusingWorker(FakeWorker):
        def submit(self, job) -> bool:
            return False

    processor = _processor(worker=RefusingWorker())
    processor.ingest(_turn("t1", Q1))
    processor.ingest(_turn("t2", _successor(Q1, "ok")))
    assert "t1" in processor.retention_decision().releasable_agent_record_ids


@pytest.mark.unit
def test_broken_worker_rejects_submissions() -> None:
    worker = JudgingWorker(object())  # the judge never runs: broken first
    worker._broken = True
    from recipes.openclawrl.turns import TurnJob

    job = TurnJob(
        receipt="t1",
        request_messages=(),
        request_tools=None,
        response_message={},
        next_state_text="",
        next_state_role="user",
    )
    assert worker.submit(job) is False


def test_session_tag_binds_across_a_client_that_restarts_its_transcript() -> None:
    """A tagged conversation binds turns the header-free path cannot see.

    Hermes keeps history locally: every turn starts a fresh
    ``[system, user]`` request, so no request extends the one before it and
    trace matching binds nothing across the student reply — the one message
    that carries this method's reward. The tag names the conversation, so
    binding follows arrival order instead.
    """

    def request(student: str, *, replied: bool = False) -> list[dict]:
        messages = [{"role": "system", "content": "hermes"}, {"role": "user", "content": student}]
        if replied:
            messages += [{"role": "assistant", "content": ""}, {"role": "tool", "content": "file contents"}]
        return messages

    calls = [
        ("c1", request("solve homework/0.txt")),
        ("c2", request("solve homework/0.txt", replied=True)),
        ("c3", request("too AI-written, redo it naturally")),
    ]

    # Header-free: only the intra-turn tool step extends its predecessor, so
    # that is the one binding trace matching can see. The student reply is
    # invisible to it — which is the whole reason the tag exists.
    untagged = SessionIndex(900.0)
    seen = [binding for i, (rid, ms) in enumerate(calls) if (binding := untagged.observe(rid, ms, float(i)).binding)]
    assert [(b.receipt, b.next_state_role) for b in seen] == [("c1", "tool")]

    tagged = SessionIndex(900.0)
    bound = [
        binding
        for i, (rid, ms) in enumerate(calls)
        if (binding := tagged.observe(rid, ms, float(i), session_tag="hw-0").binding)
    ]
    # Every main turn is judged against whatever arrived next: c1 by its own
    # tool result, c2 by the student's reaction.
    assert [(b.receipt, b.next_state_role) for b in bound] == [("c1", "tool"), ("c2", "user")]
    assert bound[1].next_state_text.startswith("too AI-written")


@pytest.mark.unit
def test_the_prm_record_file_reports_the_judged_population(tmp_path) -> None:
    """A batch mean cannot tell "the judges disagree" from "they never ran"."""
    record = tmp_path / "prm-records.jsonl"
    worker = FakeWorker()
    context = ProcessorContext("s", {"batch_size": 1, "prm_record_file": str(record)})
    processor = OpenClawRLProcessor(context, worker=worker)
    processor.ingest(_turn("t1", Q1))
    processor.ingest(_turn("t2", _successor(Q1, "that works")))
    worker.push(TurnJudgment("t1", score=1.0, teacher_cands=ANCHOR))
    assert processor.ready()
    processor.build_batch()

    line = json.loads(record.read_text().splitlines()[0])
    assert line["batch"] == 1
    assert line["samples"] == 1
    assert line["rewards"] == {"+1": 1}


@pytest.mark.unit
def test_every_main_turn_is_judged_whatever_its_next_state_is() -> None:
    """call → tool → call → user reply: two judgments, one per model call.

    Upstream fires its judge on ``messages[-1]`` with no role test, and both
    judge prompts branch on the role — a tool error is a ``\\boxed{-1}``. One
    LLM call is one training datum, so a turn that iterates the tool loop N
    times contributes N samples, not one.
    """
    worker = FakeWorker()
    processor = _processor(worker=worker)
    tool_step = [*Q1, {"role": "assistant", "content": ""}, {"role": "tool", "content": "file contents"}]
    processor.ingest(_turn("t1", Q1))
    processor.ingest(_turn("t2", tool_step))
    assert [(job.receipt, job.next_state_role) for job in worker.jobs] == [("t1", "tool")]
    processor.ingest(_turn("t3", _successor(tool_step, "too AI-written, redo it")))
    assert [(job.receipt, job.next_state_role) for job in worker.jobs] == [("t1", "tool"), ("t2", "user")]
    assert worker.jobs[1].next_state_text == "too AI-written, redo it"


def test_session_tag_treats_a_repeated_request_as_a_retry() -> None:
    index = SessionIndex(900.0)
    messages = [{"role": "user", "content": "same"}]
    assert index.observe("first", messages, 0.0, session_tag="hw-1").binding is None
    retry = index.observe("second", messages, 1.0, session_tag="hw-1")
    assert retry.duplicate is True and retry.binding is None


def test_truncated_reasoning_never_comes_back_as_the_reply() -> None:
    """A thinking model that runs out of tokens mid-thought yields no content.

    Qwen3-*-Thinking templates pre-open ``<think>``, so the sample carries
    only the closing tag — and a sample without one is reasoning that hit the
    cap, not an answer. Handing it back as content is what makes a judge score
    chain-of-thought as the agent's reply.
    """
    from reef.train.slime_backend.reef_adapters.sglang.chat import SGLangChatTrainingInferenceBackend

    truncated = "Okay, the user wants me to solve this. First I should read the file"
    leaked, _ = SGLangChatTrainingInferenceBackend._assistant_message(truncated, None)
    assert leaked["content"] == truncated  # unguarded default is unchanged

    caught, _ = SGLangChatTrainingInferenceBackend._assistant_message(truncated, None, force_reasoning=True)
    assert caught["content"] == ""
    assert caught["reasoning_content"] == truncated

    closed = "<think>pondering</think>The answer is 36."
    parsed, _ = SGLangChatTrainingInferenceBackend._assistant_message(closed, None, force_reasoning=True)
    assert parsed["content"] == "The answer is 36."
    assert parsed["reasoning_content"] == "pondering"


def test_retired_openclawrl_fields_are_accepted_and_ignored(caplog) -> None:
    from reef_service.runtime_stubs import StubTrainingRuntime

    from recipes.openclawrl import OpenClawRLRecipe

    with caplog.at_level("WARNING"):
        recipe = OpenClawRLRecipe(StubTrainingRuntime(), prm_teacher_timeout_s=5.0)
    assert recipe.prm_teacher_timeout_s == 5.0
    assert "prm_teacher_timeout_s is deprecated and ignored" in caplog.text
