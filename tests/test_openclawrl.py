"""OpenClaw-RL method tests: hint judging, dispatch, combine feed."""

from __future__ import annotations

import asyncio
import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest

from recipes.openclawrl.prm import (
    append_hint_to_messages,
    build_hint_judge_messages,
    collect_hint_candidates,
    parse_hint_judgment,
)
from recipes.openclawrl.processor import OpenClawRLProcessor
from recipes.openclawrl.turns import TurnJob
from reef.train.slime_backend.reef_adapters.preparation import prepare_slime_step
from reef.train.types import PolicyBatch, PolicySample

pytestmark = pytest.mark.unit


@pytest.fixture
def topk_objective(monkeypatch):
    """Load the Megatron worker objective with only its device glue stubbed.

    The tensor kernel and runtime PPO helpers remain real.  Megatron is
    unavailable on the CPU CI host, so the fixture supplies the two worker
    seams the objective reads: the TP group and the runtime's response slicing.
    """
    torch = pytest.importorskip("torch")
    core = ModuleType("megatron.core")
    core.mpu = SimpleNamespace(get_tensor_model_parallel_group=lambda: None)
    megatron = ModuleType("megatron")
    megatron.core = core
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", core)

    loss_module = ModuleType("slime.backends.megatron_utils.loss")

    def get_log_probs_and_entropy(logits, **kwargs):
        log_probs = logits.log_softmax(dim=-1)[:, 0]
        output = {"log_probs": [log_probs]}
        if kwargs.get("with_entropy"):
            probabilities = logits.softmax(dim=-1)
            output["entropy"] = [-(probabilities * logits.log_softmax(dim=-1)).sum(dim=-1)]
        return torch.empty(0), output

    def get_responses(logits, **_kwargs):
        yield logits, torch.empty(0, dtype=torch.long)

    loss_module.get_log_probs_and_entropy = get_log_probs_and_entropy
    loss_module.get_responses = get_responses
    monkeypatch.setitem(sys.modules, "slime.backends.megatron_utils.loss", loss_module)

    module_name = "recipes.openclawrl.slime.objective"
    previous = sys.modules.pop(module_name, None)
    try:
        yield importlib.import_module(module_name)
    finally:
        sys.modules.pop(module_name, None)
        if previous is not None:
            sys.modules[module_name] = previous


def _anchor(tokens: tuple) -> tuple:
    """The un-enhanced anchor candidate for ``tokens`` (upstream ships one
    per RL-only sample; every trained sample carries at least one)."""
    return ({"hint": "", "teacher_tokens": [int(v) for v in tokens]},)


def _sample(reward: float, cands: tuple | None = None, n: int = 3) -> PolicySample:
    tokens = tuple(range(10, 10 + n + 2))
    if cands is None:
        cands = _anchor(tokens)
    return PolicySample(
        source_agent_record_id="src",
        tokens=tokens,
        loss_mask=(1,) * n,
        rollout_log_probs=(-1.0,) * n,
        reward=reward,
        runtime_load_id="v:1",
        topk_indices=tuple((1, 2, 3, 4) for _ in range(n)),
        topk_log_probs=tuple((-0.5, -1.0, -1.5, -2.0) for _ in range(n)),
        extras={"teacher_cands": cands},
    )


def test_topk_preserves_external_advantages_through_slime_hook(topk_objective) -> None:
    from reef.train.slime_backend.algorithm import resolve_objective_paths

    args = SimpleNamespace(loss_family="openclawrl")
    resolve_objective_paths(args)

    assert args.custom_advantage_function_path.endswith("openclawrl.slime.objective.openclawrl_advantages")
    advantages = [object()]
    rollout_data = {"advantages": advantages}
    topk_objective.openclawrl_advantages(args, rollout_data)
    assert rollout_data["advantages"] is advantages
    assert rollout_data["returns"] is advantages


class TestHintJudging:
    def test_parse_positive_with_hint(self):
        score, hint = parse_hint_judgment(
            "Thinking...\\boxed{1}\n[HINT_START]Use plain prose, no bullet lists.[HINT_END]"
        )
        assert score == 1
        assert hint == "Use plain prose, no bullet lists."

    def test_parse_negative_has_no_hint(self):
        score, hint = parse_hint_judgment("nah \\boxed{-1}")
        assert score == -1
        assert hint == ""

    def test_parse_garbage_is_none(self):
        assert parse_hint_judgment("no decision here") == (None, "")

    def test_collect_candidates_keeps_substantive_accepted_hints_shortest_first(self):
        votes = [
            {"score": 1, "hint": "short tip"},  # <= 10 chars: not substantive
            {"score": 1, "hint": "a much longer, information-dense hint"},
            {"score": -1, "hint": "ignored because negative"},
            {"score": 1, "hint": "medium-size hint text"},
            {"score": 1, "hint": "medium-size hint text"},  # duplicate
        ]
        assert collect_hint_candidates(votes) == [
            "medium-size hint text",
            "a much longer, information-dense hint",
        ]

    def test_collect_candidates_caps_at_max_cand(self):
        votes = [{"score": 1, "hint": f"hint number {n} with substance"} for n in range(5)]
        assert len(collect_hint_candidates(votes, max_cand=2)) == 2

    def test_collect_candidates_empty_when_no_positive(self):
        assert collect_hint_candidates([{"score": -1, "hint": "x" * 40}]) == []

    def test_append_hint_targets_last_user_message(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"},
        ]
        enhanced = append_hint_to_messages(messages, "keep all steps")
        assert enhanced[3]["content"].endswith("[user's hint / instruction]\nkeep all steps")
        assert enhanced[1]["content"] == "first"
        assert messages[3]["content"] == "second"  # original untouched

    def test_hint_judge_messages_carry_roles(self):
        messages = build_hint_judge_messages("resp", "next", "tool")
        assert "[role: tool]" in messages[1]["content"]


class TestTopkPreparer:
    def test_signals_ride_topk_channels(self):
        cand = {"hint": "h", "teacher_tokens": [7, 8, 12, 13, 14]}
        batch = PolicyBatch(
            "s:openclawrl:0",
            (
                _sample(1.0, cands=(cand,)),
                _sample(-1.0),  # RL only: the anchor candidate
            ),
        )
        step = prepare_slime_step(batch, "openclawrl", {})
        assert step.payload["loss"] == "openclawrl"
        assert step.payload["advantages"] == [1.0, -1.0]
        # The family's wire row: policy 5-tuple + the three top-K channels.
        rows = step.payload["samples"]
        assert rows[0][5] == [[1, 2, 3, 4]] * 3
        assert rows[0][6] == [[-0.5, -1.0, -1.5, -2.0]] * 3
        assert rows[0][7][0]["hint"] == "h"
        # The RL-only row carries the anchor: every sample ships >= 1 candidate.
        assert rows[1][7] == [{"hint": "", "teacher_tokens": [10, 11, 12, 13, 14]}]

    def test_missing_topk_rejected(self):
        from recipes.openclawrl.slime.utils.data_builder import build_openclawrl_rollout_data
        from reef.train.slime_backend.loss_families import resolve_loss_family

        spec = resolve_loss_family("openclawrl")
        # A sample whose top-K capture is absent cannot train on the paper
        # objective; the builder refuses it at the wire boundary.
        payload = {
            "samples": [["a", [1, 2, 3, 4, 5], [1, 1, 1], [-1.0, -1.5, -2.0], 1.0, [], [], []]],
            "rollout_ids": [0],
            "loss": "openclawrl",
            "advantages": [1.0],
        }
        with pytest.raises(ValueError, match="top-K"):
            build_openclawrl_rollout_data(payload, payload["samples"], spec)

    def test_topk_payload_builder_ships_candidate_sequences(self):
        from recipes.openclawrl.slime.utils.data_builder import build_openclawrl_rollout_data
        from reef.train.slime_backend.loss_families import resolve_loss_family

        spec = resolve_loss_family("openclawrl")

        def payload(cands):
            return {
                "samples": [
                    [
                        "a",
                        [1, 2, 3, 4, 5],
                        [1, 1, 1],
                        [-1.0, -1.5, -2.0],
                        1.0,
                        [[1, 2, 3, 4]] * 3,
                        [[-0.5, -1.0, -1.5, -2.0]] * 3,
                        cands,
                    ]
                ],
                "rollout_ids": [0],
                "loss": "openclawrl",
                "advantages": [1.0],
            }

        good = payload([{"hint": "h", "teacher_tokens": [9, 9, 3, 4, 5]}])
        data = build_openclawrl_rollout_data(good, good["samples"], spec)
        # The wire carries only the sequences; the Megatron teacher pass
        # computes the prm_teacher_*_cand tensors trainer-side.
        assert data["teacher_tokens_cand"] == [[[9, 9, 3, 4, 5]]]
        assert "prm_teacher_topk_log_probs_cand" not in data

        empty = payload([])
        with pytest.raises(ValueError, match="no teacher candidate"):
            build_openclawrl_rollout_data(empty, empty["samples"], spec)


@pytest.mark.parametrize("hint_selection", ["shortest", "token_optimal", "sequence_optimal"])
@pytest.mark.parametrize("subset_mode", ["student", "overlap", "teacher"])
def test_topk_objective_executes_every_selection_and_subset_mode(
    monkeypatch,
    topk_objective,
    hint_selection,
    subset_mode,
) -> None:
    """Exercise all nine public modes through the real differentiable kernel."""
    torch = pytest.importorskip("torch")
    del monkeypatch  # the knobs ride args now, not the environment

    logits = torch.tensor(
        [[1.2, 0.3, -0.4], [0.1, 0.8, -0.2]],
        dtype=torch.float32,
        requires_grad=True,
    )
    student_indices = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    student_log_probs = torch.tensor([[-0.4, -1.4], [-0.3, -1.6]])
    teacher_log_probs = torch.stack(
        [
            student_log_probs + torch.tensor([[0.2, -0.1], [0.1, -0.2]]),
            student_log_probs + torch.tensor([[-0.2, 0.3], [0.4, -0.1]]),
        ]
    )
    teacher_native_indices = torch.tensor(
        [
            [[0, 2], [0, 2]],
            [[0, 1], [1, 2]],
        ],
        dtype=torch.long,
    )
    teacher_overlap_indices = torch.tensor(
        [
            [[0, 2], [0, 2]],
            [[0, 1], [1, 2]],
        ],
        dtype=torch.long,
    )

    batch = {
        "unconcat_tokens": [torch.tensor([7, 8, 9])],
        "total_lengths": [3],
        "response_lengths": [2],
        "log_probs": [torch.tensor([-0.6, -0.7])],
        "rollout_log_probs": [torch.tensor([-0.6, -0.7])],
        "advantages": [torch.tensor([0.5, -0.25])],
        "topk_indices": [student_indices],
        "topk_log_probs": [student_log_probs],
        "step_wise_step_token_spans": [[[0, 1], [1, 2]]],
    }
    if subset_mode == "teacher":
        batch.update(
            prm_teacher_topk_log_probs=[teacher_log_probs[1]],
            prm_teacher_topk_indices=[student_indices.clone()],
        )
    else:
        batch.update(
            prm_teacher_topk_log_probs_cand=[teacher_log_probs],
            prm_teacher_topk_indices_cand=[
                (
                    torch.stack([student_indices, student_indices])
                    if subset_mode == "student"
                    else teacher_overlap_indices
                )
            ],
            prm_teacher_native_topk_indices_cand=[teacher_native_indices],
        )

    args = SimpleNamespace(
        eps_clip=0.2,
        eps_clip_high=0.3,
        entropy_coef=0.0,
        use_kl_loss=False,
        kl_loss_coef=0.0,
        kl_loss_type="low_var_kl",
        openclawrl_hint_selection=hint_selection,
        openclawrl_subset_mode=subset_mode,
        openclawrl_w_rl=1.0,
        openclawrl_w_opd=1.0,
        openclawrl_adv_diff_clip=1.0,
    )
    loss, metrics = topk_objective.openclawrl_loss(
        args,
        batch,
        logits,
        torch.mean,
    )

    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in metrics.values())
    assert (
        metrics["hint_selection_mode_id"].item()
        == {
            "shortest": 0,
            "token_optimal": 1,
            "sequence_optimal": 2,
        }[hint_selection]
    )
    assert (
        metrics["distill_subset_mode_id"].item()
        == {
            "student": 0,
            "overlap": 1,
            "teacher": 2,
        }[subset_mode]
    )
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


class _FakeJudge:
    def __init__(self, eval_score: float, hint: str | None):
        self._score = eval_score
        self._hint = hint

    async def evaluate(self, *_args):
        return self._score

    async def evaluate_hint_votes(self, *_args):
        if self._hint is None:
            return [{"score": -1, "hint": ""}]
        return [{"score": 1, "hint": self._hint}]


class _FakeTeacher:
    def __init__(self, result):
        self._fail = result is None
        self.calls: list = []

    async def candidate_tokens(self, enhanced_messages, *, tools=None, native_tokens, response_token_count):
        self.calls.append(enhanced_messages)
        if self._fail:
            return None
        return [101, 102] + [int(v) for v in native_tokens[-response_token_count:]]


def _judge_one_turn(judge, teacher):
    from reef.train.types import ProcessorContext

    processor = OpenClawRLProcessor(ProcessorContext("s", {"batch_size": 1}), clients=(judge, teacher, None))
    job = TurnJob(
        receipt="r1",
        request_messages=({"role": "user", "content": "do my homework"},),
        request_tools=None,
        response_message={"role": "assistant", "content": "the answer is 5"},
        next_state_text="too robotic, redo it",
        next_state_role="user",
        topk_indices=[[1, 2, 3, 4]],
        native_tokens=[7, 8, 9],
        response_token_count=1,
    )
    try:
        return asyncio.run(processor.judge(job))
    finally:
        processor.close()


class TestHybridDispatch:
    def test_combined_sample(self):
        teacher = _FakeTeacher([[-0.1], [-0.2]])
        judgment = _judge_one_turn(_FakeJudge(-1.0, "write casually, keep every step"), teacher)
        assert judgment.score == -1.0
        # The candidate is a TOKEN SEQUENCE ending in the native response ids;
        # the Megatron teacher pass scores it trainer-side.
        assert judgment.teacher_cands[0]["teacher_tokens"] == [101, 102, 9]
        assert judgment.teacher_cands[0]["hint"] == "write casually, keep every step"
        assert "[user's hint / instruction]" in teacher.calls[0][-1]["content"]

    def test_directive_only_sample_scores_zero(self):
        judgment = _judge_one_turn(_FakeJudge(0.0, "add the full derivation"), _FakeTeacher([[-0.3]]))
        assert judgment.score == 0.0
        assert judgment.teacher_cands[0]["hint"] == "add the full derivation"

    def test_rl_only_sample_ships_unenhanced_anchor(self):
        teacher = _FakeTeacher([[-0.3]])
        judgment = _judge_one_turn(_FakeJudge(1.0, None), teacher)
        assert judgment.score == 1.0
        # Upstream RL-only convention: one un-enhanced candidate — the native
        # ids verbatim, built without the teacher (its construction cannot fail).
        assert judgment.teacher_cands[0]["hint"] == ""
        assert judgment.teacher_cands[0]["teacher_tokens"] == [7, 8, 9]
        assert teacher.calls == []

    def test_neither_signal_drops_turn(self):
        judgment = _judge_one_turn(_FakeJudge(0.0, None), _FakeTeacher(None))
        assert judgment.score is None

    def test_teacher_failure_degrades_to_rl_only(self):
        judgment = _judge_one_turn(_FakeJudge(1.0, "a substantive hint here"), _FakeTeacher(None))
        assert judgment.score == 1.0
        # Hint templating failed, so the turn falls back to the anchor —
        # whose construction is id concatenation and cannot fail.
        assert judgment.teacher_cands == ({"hint": "", "teacher_tokens": [7, 8, 9]},)


@pytest.mark.unit
def test_processor_attaches_topk_and_candidates() -> None:
    from recipes.openclawrl import OpenClawRLProcessor
    from recipes.openclawrl.turns import TurnJudgment
    from reef.core.records_types import AgentRecord, RequestType
    from reef.train.types import ProcessorContext

    class FakeWorker:
        def __init__(self) -> None:
            self.jobs = []
            self._verdicts = []

        def submit(self, job) -> bool:
            self.jobs.append(job)
            return True

        def poll(self):
            judgments, self._verdicts = self._verdicts, []
            return judgments

        def close(self) -> None:
            pass

        def push(self, judgment) -> None:
            self._verdicts.append(judgment)

    def turn(index: int, messages: list) -> AgentRecord:
        return AgentRecord.create(
            scenario="s",
            request_type=RequestType.INFERENCE,
            payload={
                "messages": messages,
                "tools": [{"type": "function", "function": {"name": "noop"}}],
                "response": {
                    "choices": [{"message": {"role": "assistant", "content": "a"}}],
                    "training": {
                        "tokens": [1, 2, 3, 4],
                        "loss_mask": [0, 1, 1],
                        "rollout_log_probs": [0.0, -0.3, -0.2],
                        "topk_indices": [[1, 2], [3, 4], [5, 6]],
                        "topk_log_probs": [[-0.1, -0.2], [-0.3, -0.4], [-0.5, -0.6]],
                        "runtime_load_id": "v1",
                    },
                },
            },
            agent_record_id=f"i{index}",
        )

    worker = FakeWorker()
    processor = OpenClawRLProcessor(ProcessorContext("s", {"batch_size": 2}), worker=worker)
    # Two sessions with distinct prefixes; each first turn gets a successor.
    for index, opening in enumerate(["q-alpha", "q-beta"]):
        first = [{"role": "user", "content": opening}]
        processor.ingest(turn(index, first))
        processor.ingest(
            turn(
                index + 10,
                [*first, {"role": "assistant", "content": "a"}, {"role": "user", "content": "next"}],
            )
        )
    # A hint candidate re-tokenizes only the PROMPT; the native response ids
    # (tokens[-3:] here) ride verbatim, so validation checks that tail.
    cand = {"hint": "h", "teacher_tokens": [99, 2, 3, 4]}
    worker.push(TurnJudgment("i0", score=1.0, teacher_cands=(cand,)))
    # The RL-only turn ships the un-enhanced anchor: the native ids verbatim.
    worker.push(TurnJudgment("i1", score=1.0, teacher_cands=({"hint": "", "teacher_tokens": [1, 2, 3, 4]},)))

    batch = processor.build_batch()
    by_source = {sample.source_agent_record_id: sample for sample in batch.samples}
    sample = by_source["i0"]
    assert sample.topk_indices == ((1, 2), (3, 4), (5, 6))
    assert sample.extras["teacher_cands"] == ({"hint": "h", "teacher_tokens": [99, 2, 3, 4]},)
    assert by_source["i1"].extras["teacher_cands"] == ({"hint": "", "teacher_tokens": [1, 2, 3, 4]},)


@pytest.mark.unit
def test_candidate_validation_checks_the_native_tail_not_widths() -> None:
    # Alignment is exact by construction — a candidate ends with the native
    # response ids verbatim — so validation checks that construction and
    # nothing about capture widths (the Megatron teacher gathers at whatever
    # width S^q has).
    from recipes.openclawrl.turns import validate_teacher_cands
    from reef.train.types import PolicySample

    capture = 8
    sample = PolicySample(
        source_agent_record_id="t1",
        tokens=(1, 2, 3),
        loss_mask=(1, 1),
        rollout_log_probs=(-0.1, -0.2),
        reward=1.0,
        topk_indices=tuple(tuple(range(capture)) for _ in range(2)),
        topk_log_probs=tuple(tuple(-0.1 * i for i in range(capture)) for _ in range(2)),
    )
    good = {"hint": "h", "teacher_tokens": [9, 9, 2, 3]}
    assert validate_teacher_cands((good,), sample) == ({"hint": "h", "teacher_tokens": [9, 9, 2, 3]},)
    wrong_tail = {"hint": "h", "teacher_tokens": [9, 9, 3, 2]}
    assert validate_teacher_cands((wrong_tail,), sample) is None
    no_prompt = {"hint": "h", "teacher_tokens": [2, 3]}
    assert validate_teacher_cands((no_prompt,), sample) is None
