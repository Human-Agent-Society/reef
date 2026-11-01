"""OpenClaw-RL method tests: hint judging, dispatch, combine feed."""

from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from tests.support._trajectories import policy_trajectory, recorded_trajectory
from tests.support.chat_templates import CharTemplate

from recipes.openclawrl.judging import OpenClawRLJudgingSettings
from recipes.openclawrl.prm import (
    append_hint_to_messages,
    build_hint_judge_messages,
    collect_hint_candidates,
    parse_hint_judgment,
)
from recipes.openclawrl.processor import OpenClawRLJobData, OpenClawRLJobSettings, OpenClawRLSamples
from recipes.openclawrl.turns import TurnJob
from reef.core.chat_rendering import rendered_job_data
from reef.core.trajectories import source_record_id
from reef.runtime.interfaces import TrainingMethod
from reef.train.algos.schedule import StepScheduling
from reef.train.pipeline import TrainingPipeline
from reef.train.slime_backend.reef_adapters.preparation import prepare_slime_job
from reef.train.types import JobData, TrajectorySample

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
    """The rendered un-enhanced anchor candidate for ``tokens``: its teacher reads the student's own sequence
    (upstream ships one per RL-only sample; every trained sample carries at least one)."""
    return ({"hint": "", "teacher_tokens": [int(v) for v in tokens]},)


def _sample(reward: float, cands: tuple | None = None, n: int = 3) -> TrajectorySample:
    """A sample as the training component renders it: tokens, loss mask, and rendered teacher candidates."""
    tokens = tuple(range(10, 10 + n + 2))
    if cands is None:
        cands = _anchor(tokens)
    return policy_trajectory(
        source_agent_record_id="src",
        tokens=tokens,
        loss_mask=(1,) * n,
        rollout_log_probs=(-1.0,) * n,
        reward=reward,
    ).with_training(teacher_candidates=[dict(cand) for cand in cands])


class _FakeBackuper:
    """slime's TensorBackuper surface, over dicts of CPU tensors."""

    def __init__(self, live):
        self.live = live
        self.backups: dict[str, dict] = {}

    @property
    def backup_tags(self):
        return list(self.backups)

    def get(self, tag):
        return self.backups[tag]

    def backup(self, tag):
        self.backups[tag] = {name: tensor.clone() for name, tensor in self.live.items()}

    def restore(self, tag):
        for name, tensor in self.backups[tag].items():
            self.live[name].copy_(tensor)


class _FakeActor:
    """The two slime actor calls the init hook makes, over a fake model."""

    def __init__(self, load, hf_checkpoint, base, trained):
        torch = pytest.importorskip("torch")
        self.args = SimpleNamespace(load=load, hf_checkpoint=hf_checkpoint)
        self._base = base
        self.live = {"w": torch.tensor(trained)}
        self.weights_backuper = _FakeBackuper(self.live)
        self.weights_backuper.backup("actor")  # what init leaves behind
        self.calls: list = []

    def load_other_checkpoint(self, tag, path):
        torch = pytest.importorskip("torch")
        self.calls.append(("load", tag, path))
        self.live["w"].copy_(torch.tensor(self._base))
        self.weights_backuper.backup(tag)

    def _switch_model(self, tag):
        self.calls.append(("switch", tag))
        self.weights_backuper.restore(tag)


@pytest.fixture
def megatron_checkpoint_probe(monkeypatch):
    """The one slime checkpoint helper the init hook imports, without Megatron."""
    checkpoint = ModuleType("slime.backends.megatron_utils.checkpoint")
    checkpoint._is_megatron_checkpoint = lambda path: (Path(path) / "latest_checkpointed_iteration.txt").is_file()
    monkeypatch.setitem(sys.modules, "slime.backends.megatron_utils.checkpoint", checkpoint)


def test_actor_init_backs_up_the_fresh_load_as_the_teacher(
    tmp_path, topk_objective, megatron_checkpoint_probe
) -> None:
    torch = pytest.importorskip("torch")
    actor = _FakeActor(str(tmp_path / "megatron"), str(tmp_path / "hf"), base=[1.0, 2.0], trained=[1.0, 2.0])

    topk_objective.openclawrl_actor_init(actor)

    assert actor.calls == []  # a fresh bridge load IS the base: no reload
    assert torch.equal(actor.weights_backuper.get("openclaw_teacher")["w"], torch.tensor([1.0, 2.0]))


def test_actor_init_reloads_the_base_teacher_on_resume(tmp_path, topk_objective, megatron_checkpoint_probe) -> None:
    torch = pytest.importorskip("torch")
    load = tmp_path / "megatron"
    load.mkdir()
    (load / "latest_checkpointed_iteration.txt").write_text("1")
    actor = _FakeActor(str(load), str(tmp_path / "hf"), base=[1.0, 2.0], trained=[1.5, 2.5])

    topk_objective.openclawrl_actor_init(actor)

    # The base HF weights become the teacher; the trained weights stay the actor.
    assert actor.calls == [("load", "openclaw_teacher", str(tmp_path / "hf")), ("switch", "actor")]
    assert torch.equal(actor.weights_backuper.get("openclaw_teacher")["w"], torch.tensor([1.0, 2.0]))
    assert torch.equal(actor.live["w"], torch.tensor([1.5, 2.5]))


def test_actor_init_on_resume_needs_the_base_checkpoint(tmp_path, topk_objective, megatron_checkpoint_probe) -> None:
    load = tmp_path / "megatron"
    load.mkdir()
    (load / "latest_checkpointed_iteration.txt").write_text("1")
    actor = _FakeActor(str(load), None, base=[1.0], trained=[2.0])

    with pytest.raises(RuntimeError, match="--hf-checkpoint"):
        topk_objective.openclawrl_actor_init(actor)
    assert actor.calls == []


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


class TestOpenclawrlWireRow:
    def test_the_row_carries_the_rendered_teacher_candidates(self):
        cand = {"hint": "h", "teacher_tokens": [7, 8, 12, 13, 14]}
        batch = JobData(
            "s:openclawrl:0",
            (
                _sample(1.0, cands=(cand,)),
                _sample(-1.0),
            ),
        )
        step = prepare_slime_job(batch, TrainingMethod("openclawrl"), {}, StepScheduling(), step_size=2)
        assert step.payload["loss"] == "openclawrl"
        assert step.payload["advantages"] == [1.0, -1.0]
        # The family's wire row: policy 5-tuple + the teacher candidates; the actor computes the student's top-K.
        rows = step.payload["samples"]
        assert [len(row) for row in rows] == [6, 6]
        assert rows[0][5] == [cand]
        # The RL-only row carries the anchor: every sample ships >= 1 candidate.
        assert rows[1][5] == [{"hint": "", "teacher_tokens": [10, 11, 12, 13, 14]}]

    def test_a_row_of_another_shape_is_refused(self):
        from recipes.openclawrl.slime import build_openclawrl_rollout_data
        from reef.train.slime_backend.loss_families import resolve_loss_family

        spec = resolve_loss_family("openclawrl")
        # The retired 8-column row that carried the sampler's top-K: the builder refuses it at the wire boundary.
        payload = {
            "samples": [["a", [1, 2, 3, 4, 5], [1, 1, 1], [-1.0, -1.5, -2.0], 1.0, [], [], []]],
            "rollout_ids": [0],
            "loss": "openclawrl",
            "advantages": [1.0],
            "external_step_sizes": [1],
        }
        with pytest.raises(ValueError, match=r"teacher_cands\]"):
            build_openclawrl_rollout_data(payload, payload["samples"], spec)

    def test_the_payload_builder_ships_candidate_sequences(self):
        from recipes.openclawrl.slime import build_openclawrl_rollout_data
        from reef.train.slime_backend.loss_families import resolve_loss_family

        spec = resolve_loss_family("openclawrl")

        def payload(cands):
            return {
                "samples": [["a", [1, 2, 3, 4, 5], [1, 1, 1], [-1.0, -1.5, -2.0], 1.0, cands]],
                "rollout_ids": [0],
                "loss": "openclawrl",
                "advantages": [1.0],
                "external_step_sizes": [1],
            }

        good = payload([{"hint": "h", "teacher_tokens": [9, 9, 3, 4, 5]}])
        data = build_openclawrl_rollout_data(good, good["samples"], spec)
        # The wire carries only the sequences; the actor computes the student's top-K and the Megatron teacher
        # pass computes the prm_teacher_*_cand tensors trainer-side.
        assert data["teacher_tokens_cand"] == [[[9, 9, 3, 4, 5]]]
        assert "topk_indices" not in data
        assert "prm_teacher_topk_log_probs_cand" not in data

        empty = payload([])
        with pytest.raises(ValueError, match="no teacher candidate"):
            build_openclawrl_rollout_data(empty, empty["samples"], spec)

        short = payload([{"hint": "h", "teacher_tokens": [3, 4, 5]}])
        with pytest.raises(ValueError, match="longer than its 3-token response"):
            build_openclawrl_rollout_data(short, short["samples"], spec)


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


REQUEST = ({"role": "user", "content": "do my homework"},)


def _judge_one_turn(judge):
    from reef.pipeline_context import PipelineContext

    context = PipelineContext("s")

    stage = OpenClawRLSamples(OpenClawRLJudgingSettings(), clients=(judge, None))
    processor = TrainingPipeline(
        context,
        record2sample=stage,
        sample2job=OpenClawRLJobData(context, OpenClawRLJobSettings(), stage),
    )
    job = TurnJob(
        receipt="r1",
        request_messages=REQUEST,
        request_tools=None,
        response_message={"role": "assistant", "content": "the answer is 5"},
        next_state_text="too robotic, redo it",
        next_state_role="user",
    )
    try:
        return asyncio.run(stage.openclawrl_judging.judge(job))
    finally:
        processor.close()


class TestHybridDispatch:
    def test_combined_sample(self):
        judgment = _judge_one_turn(_FakeJudge(-1.0, "write casually, keep every step"))
        assert judgment.score == -1.0
        # The candidate is the hint-enhanced REQUEST; the training component renders it into a token sequence.
        hint = "write casually, keep every step"
        assert judgment.teacher_cands == ({"hint": hint, "messages": append_hint_to_messages(list(REQUEST), hint)},)
        assert "[user's hint / instruction]" in judgment.teacher_cands[0]["messages"][-1]["content"]

    def test_directive_only_sample_scores_zero(self):
        judgment = _judge_one_turn(_FakeJudge(0.0, "add the full derivation"))
        assert judgment.score == 0.0
        assert judgment.teacher_cands[0]["hint"] == "add the full derivation"

    def test_rl_only_sample_ships_unenhanced_anchor(self):
        judgment = _judge_one_turn(_FakeJudge(1.0, None))
        assert judgment.score == 1.0
        # Upstream RL-only convention: one un-enhanced candidate, whose teacher reads the student's own sequence.
        assert judgment.teacher_cands == ({"hint": "", "messages": None},)

    def test_neither_signal_drops_turn(self):
        judgment = _judge_one_turn(_FakeJudge(0.0, None))
        assert judgment.score is None


@pytest.mark.unit
def test_processor_attaches_teacher_candidates_that_the_component_renders() -> None:
    from recipes.openclawrl import OpenClawRLJobData
    from recipes.openclawrl.turns import TurnJudgment
    from reef.core.records_types import AgentRecord, RequestType
    from reef.pipeline_context import PipelineContext

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

    tools = [{"type": "function", "function": {"name": "noop"}}]

    def turn(index: int, messages: list) -> AgentRecord:
        return AgentRecord.create(
            scenario="s",
            request_type=RequestType.INFERENCE,
            payload={
                "messages": messages,
                "tools": tools,
                "artifact_id": "v1",
                "response": {"choices": [{"message": {"role": "assistant", "content": "a"}}]},
            },
            agent_record_id=f"i{index}",
        )

    worker = FakeWorker()
    context = PipelineContext("s")
    stage = OpenClawRLSamples(OpenClawRLJudgingSettings(), worker=worker)
    processor = TrainingPipeline(
        context,
        record2sample=stage,
        sample2job=OpenClawRLJobData(context, OpenClawRLJobSettings(), stage),
    )
    # Two sessions with distinct prefixes; each first turn gets a successor.
    for index, opening in enumerate(["q-alpha", "q-beta"]):
        first = [{"role": "user", "content": opening}]
        processor.push(turn(index, first))
        processor.push(
            turn(
                index + 10,
                [*first, {"role": "assistant", "content": "a"}, {"role": "user", "content": "next"}],
            )
        )
    enhanced = [{"role": "user", "content": "q-alpha hint"}]
    worker.push(TurnJudgment("i0", score=1.0, teacher_cands=({"hint": "h", "messages": enhanced},)))
    # The RL-only turn ships the un-enhanced anchor.
    worker.push(TurnJudgment("i1", score=1.0, teacher_cands=({"hint": "", "messages": None},)))

    processor.produce()
    batch = processor.build_job_data(2)
    by_source = {source_record_id(sample): sample for sample in batch.samples}
    request = {"tools": tools, "max_prompt_tokens": 0, "max_tokens": 0}
    assert by_source["i0"].metadata["teacher_candidates"] == [{"hint": "h", "messages": enhanced, **request}]
    assert by_source["i1"].metadata["teacher_candidates"] == [{"hint": "", "messages": None, **request}]
    assert not by_source["i0"].training.get("tokens"), "a sample from a record carries no tensors"

    # The training component renders each candidate: the enhanced prompt, then the student's reply tokens verbatim.
    template = CharTemplate()
    rendered, left_out = rendered_job_data(batch, template)
    assert left_out == ()
    by_source = {source_record_id(sample): sample.training for sample in rendered.samples}
    student = by_source["i0"]
    reply = student["tokens"][-len(student["loss_mask"]) :]
    prompt = template.render(enhanced, tools, prompt=True)
    assert student["teacher_candidates"] == [{"hint": "h", "teacher_tokens": [*prompt, *reply]}]
    assert by_source["i1"]["teacher_candidates"] == [{"hint": "", "teacher_tokens": by_source["i1"]["tokens"]}]


@pytest.mark.unit
def test_a_hint_candidate_whose_prompt_cannot_render_drops() -> None:
    class HintRefusingTemplate(CharTemplate):
        def render(self, messages, tools, *, prompt):
            if any("unrenderable" in message["content"] for message in messages):
                raise ValueError("the template cannot render this prompt")
            return super().render(messages, tools, prompt=prompt)

    payload = {
        "messages": [{"role": "user", "content": "q"}],
        "response": {"choices": [{"message": {"role": "assistant", "content": "a"}}]},
    }
    anchor = {"hint": "", "messages": None}
    broken = {"hint": "h", "messages": [{"role": "user", "content": "unrenderable"}]}
    fine = {"hint": "g", "messages": [{"role": "user", "content": "q g"}]}
    samples = (
        recorded_trajectory("s1", payload, 1.0).with_metadata(teacher_candidates=[broken, fine]),
        recorded_trajectory("s2", payload, 1.0).with_metadata(teacher_candidates=[broken]),
        recorded_trajectory("s3", payload, 1.0).with_metadata(teacher_candidates=[anchor]),
    )

    rendered, left_out = rendered_job_data(JobData("d", samples), HintRefusingTemplate())

    # The broken candidate drops; a sample whose candidates all drop is left out.
    assert left_out == (1,)
    assert [[cand["hint"] for cand in sample.training["teacher_candidates"]] for sample in rendered.samples] == [
        ["g"],
        [""],
    ]
