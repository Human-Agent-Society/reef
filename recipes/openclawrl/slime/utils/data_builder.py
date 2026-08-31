"""Wire-row shaping and payload conversion for the top-K select loss.

The three extra channels ride the family's wire row as columns 5-7 after the
shared policy 5-tuple (``shape_sample_row``); the builder unpacks them back
into Slime's per-sample rollout keys:

* ``topk_indices`` / ``topk_log_probs`` — the student's generation-time
  top-K vocab ids and log-probs per response token (``ell_old`` on S^q),
  captured by the serving backend at sampling time.
* ``teacher_cands`` (carried in ``sample.extras``) — per candidate (accepted hints, shortest-first, up to
  the recipe's ``prm_max_hint_candidates``; or the un-enhanced anchor
  upstream ships for every RL-only turn), the candidate TOKEN SEQUENCE
  (upstream's ``teacher_tokens_candidates``: hint-enhanced prompt ids plus
  the native response ids verbatim). The train actor's Megatron teacher
  pass computes the ``prm_teacher_*_cand`` tensors from these before the
  loss runs; nothing numeric about the teacher rides the wire.

Every sample carries at least one candidate — the processor retires
candidate-less turns, matching upstream, so the builder treats an empty
candidate list as a contract violation rather than something to repair.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from reef.train.slime_backend.algorithm import SlimeAlgorithm

_ROW_SHAPE = "[source_id, tokens, loss_mask, rollout_log_probs, reward, topk_indices, topk_log_probs, teacher_cands]"


def openclawrl_sample_row(sample: Any) -> list[Any]:
    """Shape one Reef sample into this family's 8-column wire row."""
    return [
        sample.source_agent_record_id,
        list(sample.tokens),
        list(sample.loss_mask),
        list(sample.rollout_log_probs),
        sample.reward,
        [list(row) for row in sample.topk_indices],
        [list(row) for row in sample.topk_log_probs],
        [dict(cand) for cand in sample.extras.get("teacher_cands", ())],
    ]


def build_openclawrl_rollout_data(
    payload: Mapping[str, Any],
    samples: Sequence,
    spec: SlimeAlgorithm,
) -> dict:
    from reef.train.slime_backend.data_builder import build_policy_rollout_data

    base_rows: list[list[Any]] = []
    student_idx_rows: list[Any] = []
    student_lp_rows: list[Any] = []
    cand_rows: list[Any] = []
    for index, row in enumerate(samples):
        if not isinstance(row, Sequence) or isinstance(row, str | bytes) or len(row) != 8:
            raise ValueError(f"sample {index} must be {_ROW_SHAPE}")
        base_rows.append(list(row[:5]))
        student_idx_rows.append(row[5])
        student_lp_rows.append(row[6])
        cand_rows.append(row[7])

    data = build_policy_rollout_data({**payload, "samples": base_rows}, base_rows, spec)
    response_lengths = data["response_lengths"]

    topk_indices: list[list[list[int]]] = []
    topk_log_probs: list[list[list[float]]] = []
    teacher_tokens_cand: list[list[list[int]]] = []

    for index, (s_idx, s_lp, cands, response_length) in enumerate(
        zip(student_idx_rows, student_lp_rows, cand_rows, response_lengths, strict=True)
    ):
        if len(s_idx) != response_length or len(s_lp) != response_length:
            raise ValueError(f"sample {index} top-K rows must cover the {response_length}-token response")
        k = len(s_idx[0]) if s_idx else 0
        if any(len(row) != k for row in s_idx) or any(len(row) != k for row in s_lp):
            raise ValueError(f"sample {index} top-K rows must be rectangular")
        topk_indices.append([[int(v) for v in row] for row in s_idx])
        topk_log_probs.append([[float(v) for v in row] for row in s_lp])

        if not cands:
            raise ValueError(f"sample {index} carries no teacher candidate; the processor must retire such turns")
        sample_tokens: list[list[int]] = []
        for cand in cands:
            tokens = cand.get("teacher_tokens")
            if not tokens or len(tokens) <= response_length:
                raise ValueError(
                    f"sample {index} candidate must be a token sequence longer than its {response_length}-token response"
                )
            sample_tokens.append([int(v) for v in tokens])
        teacher_tokens_cand.append(sample_tokens)

    data["topk_indices"] = topk_indices
    data["topk_log_probs"] = topk_log_probs
    # The numeric prm_teacher_*_cand tensors are filled trainer-side by the
    # Megatron teacher pass (recipes/openclawrl/slime/teacher.py);
    # the wire carries only the candidate sequences it forwards.
    data["teacher_tokens_cand"] = teacher_tokens_cand
    return data
