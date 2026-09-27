"""Optional CPU comparison against functions from the pinned author's checkout.

Set SDPO_REFERENCE_ROOT to lasgroup/SDPO at REFERENCE_COMMIT. Only the
three named numerical functions are compiled, so verl's GPU imports are
not needed. Their bodies are executed unchanged, not copied into Reef.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from recipes.sdpo.slime import SdpoSettings
from recipes.sdpo.slime.objective import sdpo_loss
from reef.train.slime_backend.loss_families import resolve_loss_family

REFERENCE_COMMIT = "7c457fc1b1f636ae794eb0362ba37d4743b06fbc"


@pytest.fixture(scope="module")
def reference():
    configured = os.environ.get("SDPO_REFERENCE_ROOT")
    if not configured:
        pytest.skip("set SDPO_REFERENCE_ROOT to the pinned author checkout")
    root = Path(configured)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    assert revision == REFERENCE_COMMIT
    namespace = {"torch": torch, "F": torch.nn.functional}
    for relative, names in (
        ("verl/utils/torch_functional.py", {"masked_sum"}),
        ("verl/trainer/ppo/core_algos.py", {"agg_loss", "compute_self_distillation_loss"}),
    ):
        path = root / relative
        # Read the committed blobs so local edits cannot silently change the oracle.
        source = subprocess.check_output(["git", "show", f"{REFERENCE_COMMIT}:{relative}"], cwd=root, text=True)
        tree = ast.parse(source, filename=str(path))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        assert {node.name for node in functions} == names
        module = ast.Module(body=[*ast.parse("from __future__ import annotations").body, *functions], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
        if "masked_sum" in names:
            namespace["verl_F"] = SimpleNamespace(masked_sum=namespace["masked_sum"])
    return namespace["compute_self_distillation_loss"]


@pytest.mark.unit
@pytest.mark.parametrize(("divergence", "alpha"), [("forward", 0.0), ("reverse", 1.0), ("jsd", 0.3)])
@pytest.mark.parametrize("top_k", [1, 4, 11])
def test_worker_loss_and_gradient_match_author_with_inactive_rows_and_token_is(
    monkeypatch: pytest.MonkeyPatch, reference, divergence: str, alpha: float, top_k: int
) -> None:
    lengths = [1, 3, 2]
    active = [1.0, 0.0, 1.0]
    generator = torch.Generator().manual_seed(47)
    values = torch.randn(6, 11, generator=generator)
    student = values.clone().requires_grad_(True)
    oracle_student = values.clone().requires_grad_(True)
    teacher = torch.log_softmax(torch.randn(6, 11, generator=generator), dim=-1)
    old = torch.tensor([-1.0, -2.0, -3.0, -4.0, -5.0, -10.0])
    ids = values.topk(top_k, dim=-1).indices
    teacher_at = teacher.gather(-1, ids)
    log_probs = student.log_softmax(-1)[:, 0]
    modules = {
        "megatron.core": SimpleNamespace(
            mpu=SimpleNamespace(
                get_context_parallel_world_size=lambda: 1, get_tensor_model_parallel_group=lambda: None
            )
        ),
        "slime.backends.megatron_utils.cp_utils": SimpleNamespace(get_sum_of_sample_mean=None),
        "slime.backends.megatron_utils.loss": SimpleNamespace(
            get_responses=lambda *args, **kwargs: [(row, None) for row in student.split(lengths)],
            get_log_probs_and_entropy=lambda *args, **kwargs: (None, {"log_probs": list(log_probs.split(lengths))}),
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    args = Namespace(calculate_per_token_loss=False)
    resolve_loss_family("sdpo").apply_driver_options(
        args, SdpoSettings(top_k=top_k, divergence=divergence, jsd_beta=0.3, importance_sampling_cap=2.0)
    )
    batch = {
        "total_lengths": [length + 1 for length in lengths],
        "response_lengths": lengths,
        "unconcat_tokens": [torch.zeros(length + 1, dtype=torch.long) for length in lengths],
        "loss_masks": [torch.ones(length) for length in lengths],
        "distill_sample_weights": active,
        "rollout_log_probs": list(old.split(lengths)),
        "distill_student_topk_ids": list(ids.split(lengths)),
        "distill_teacher_student_topk_log_probs": list(teacher_at.split(lengths)),
    }

    def reduce_samples(tokens):
        return torch.stack([part.mean() for part in tokens.split(lengths)]).sum()

    actual, metrics = sdpo_loss(args, batch, student, reduce_samples)
    actual = actual / len(lengths)
    oracle_log = oracle_student.log_softmax(-1)
    config = SimpleNamespace(
        full_logit_distillation=True, distillation_topk=top_k, distillation_add_tail=True, alpha=alpha, is_clip=2.0
    )
    # Match the author's resolved micro_batch_size=1 and accumulation over
    # every response, including the inactive middle response.
    expected_parts = []
    offset = 0
    for length, weight in zip(lengths, active, strict=True):
        section = slice(offset, offset + length)
        expected, _ = reference(
            student_log_probs=oracle_log[section, 0][None],
            teacher_log_probs=teacher[section, 0][None],
            response_mask=torch.ones(1, length),
            self_distillation_config=config,
            old_log_probs=old[section][None],
            student_topk_log_probs=oracle_log[section].gather(-1, ids[section])[None],
            teacher_topk_log_probs=teacher_at[section][None],
            self_distillation_mask=torch.tensor([weight]),
        )
        expected_parts.append(expected)
        offset += length
    expected = torch.stack(expected_parts).mean()
    actual.backward()
    expected.backward()
    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-5)
    assert torch.allclose(student.grad, oracle_student.grad, atol=2e-6, rtol=2e-5)
    assert not student.grad[1:4].any()
    assert metrics["distill_sample_weight"].item() == 2.0
