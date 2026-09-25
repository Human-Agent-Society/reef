"""Compare Reef's SDPO loss and gradient with the pinned author's function.

Run from a Reef training environment. The two author functions are compiled
from a verified local checkout, so this check needs neither verl workers nor
Ray; it executes the actual reference equations, without vendoring them.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import torch
import torch.nn.functional as functional

from reef.train.slime_backend.distill.objective import restricted_divergence, token_importance_weights

REFERENCE_COMMIT = "7c457fc1b1f636ae794eb0362ba37d4743b06fbc"


def reference_function(checkout: Path) -> tuple[Callable[..., Any], str]:
    revision = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if revision != REFERENCE_COMMIT:
        raise ValueError(f"reference checkout must be {REFERENCE_COMMIT}, got {revision}")
    relative = "verl/trainer/ppo/core_algos.py"
    subprocess.run(["git", "-C", str(checkout), "diff", "--exit-code", "HEAD", "--", relative], check=True)
    source = (checkout / relative).read_text()
    tree = ast.parse(source)
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in ("agg_loss", "compute_self_distillation_loss")
    ]
    if len(functions) != 2:
        raise ValueError("the pinned reference loss functions are missing")
    namespace = {
        "torch": torch,
        "F": functional,
        "Any": Any,
        "Optional": Optional,
        "verl_F": SimpleNamespace(masked_sum=lambda values, mask: (values * mask).sum()),
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(checkout / relative), "exec"), namespace)
    return namespace["compute_self_distillation_loss"], hashlib.sha256(source.encode()).hexdigest()


def compare(checkout: Path, device: str) -> dict[str, Any]:
    reference, source_hash = reference_function(checkout)
    generator = torch.Generator(device=device).manual_seed(2026)
    results = []
    for dtype in (torch.float32, torch.float64):
        for divergence, alpha in (("forward", 0.0), ("jsd", 0.5), ("reverse", 1.0)):
            for k in (20, 100):
                for active in ([1, 0, 1, 0], [0, 0, 0, 0]):
                    raw = torch.randn(4, 5, 137, generator=generator, device=device, dtype=dtype)
                    teacher = torch.randn(raw.shape, generator=generator, device=device, dtype=dtype).log_softmax(-1)
                    ids = raw.topk(k, dim=-1).indices
                    response_mask = raw.new_tensor(
                        [[1, 1, 1, 1, 1], [1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [1, 0, 0, 0, 0]]
                    )
                    mask = response_mask * raw.new_tensor(active).unsqueeze(1)
                    behavior = raw.log_softmax(-1)[..., 0] - 0.3
                    ours = raw.clone().requires_grad_()
                    student = ours.log_softmax(-1)
                    at_ids = student.gather(-1, ids)
                    teacher_at = teacher.gather(-1, ids)
                    terms = restricted_divergence(at_ids, teacher_at, divergence=divergence, distribution="tail")
                    weighted = terms * token_importance_weights(student[..., 0], behavior, 2.0)
                    actual = ((weighted * mask).sum(-1) / response_mask.sum(-1).clamp_min(1)).mean()
                    actual_gradient = torch.autograd.grad(actual, ours)[0]
                    theirs = raw.clone().requires_grad_()
                    theirs_log_probs = theirs.log_softmax(-1)
                    micro_losses = []
                    for row in range(raw.shape[0]):
                        # user.yaml fixes one rollout per microbatch; dp_actor
                        # averages these losses (including inactive ones).
                        micro_loss, _ = reference(
                            student_log_probs=theirs_log_probs[row : row + 1, ..., 0],
                            teacher_log_probs=teacher[row : row + 1, ..., 0],
                            response_mask=response_mask[row : row + 1],
                            self_distillation_config=SimpleNamespace(
                                full_logit_distillation=True,
                                distillation_topk=k,
                                distillation_add_tail=True,
                                alpha=alpha,
                                is_clip=2.0,
                            ),
                            old_log_probs=behavior[row : row + 1],
                            student_topk_log_probs=theirs_log_probs[row : row + 1].gather(-1, ids[row : row + 1]),
                            teacher_topk_log_probs=teacher_at[row : row + 1],
                            self_distillation_mask=raw.new_tensor(active[row : row + 1]),
                        )
                        micro_losses.append(micro_loss)
                    expected = torch.stack(micro_losses).mean()
                    expected_gradient = torch.autograd.grad(expected, theirs)[0]
                    tolerance = 3e-6 if dtype == torch.float32 else 1e-10
                    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
                    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=tolerance, atol=tolerance)
                    results.append(
                        {
                            "dtype": str(dtype),
                            "divergence": divergence,
                            "k": k,
                            "active": active,
                            "loss_abs_error": float((actual - expected).detach().abs()),
                            "gradient_max_abs_error": float((actual_gradient - expected_gradient).abs().max()),
                        }
                    )
    return {
        "reference_commit": REFERENCE_COMMIT,
        "source_sha256": source_hash,
        "device": device,
        "torch": torch.__version__,
        "cases": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.reference, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Passed {len(result['cases'])} pinned-reference value/gradient cases on {args.device}")


if __name__ == "__main__":
    main()
