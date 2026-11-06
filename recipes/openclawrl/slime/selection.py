"""Select the teacher top-K tensors that OpenClaw-RL's loss consumes.

The overlap, argmax, and gather operations follow upstream
hint_opd_select_loss.py. Candidate selection does not compute loss or assemble metrics.
"""

import torch


def overlap_count_per_token(
    student_idx: torch.Tensor,  # [response_length, K_q]
    teacher_idx: torch.Tensor,  # [K, R, K_p]
) -> torch.Tensor:
    """Return overlap counts [K, R] for student [R, Kq] and teacher [K, R, Kp] indices."""
    eq = student_idx.unsqueeze(0).unsqueeze(-1) == teacher_idx.unsqueeze(-2)
    return eq.any(dim=-1).sum(dim=-1).to(torch.long)  # [candidate_count, response_length]


def select_candidate_per_token(
    overlap_kr: torch.Tensor | None,  # [candidate_count, response_length] or None for shortest
    *,
    hint_selection: str,
    step_token_spans: list[list[int]] | None,  # per-sample list of [t0, t1]
    response_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute ``k*(t) ∈ [0, candidate_count)`` per response token.

    * ``shortest``         : ``k* = 0`` for every token (the rollout module
      orders candidates shortest-first, so candidate 0 is the shortest hint
      that survived the min-token / dedup filter). ``overlap_kr`` may be
      ``None`` since no overlap signal is required.
    * ``token_optimal``    : argmax over candidates per token.
    * ``sequence_optimal`` : argmax over candidates per PRM step
      (broadcast over tokens in the step). When ``step_token_spans`` is
      missing/empty -- the case for the hint_opt_exp single-CoT pipeline
      where the "sequence" IS the whole response -- this collapses to
      per-sample argmax (one ``k*`` for every token in the response).
    """
    if hint_selection == "shortest":
        return torch.zeros(response_length, dtype=torch.long, device=device)
    if overlap_kr is None:
        raise ValueError(f"overlap_kr is required for hint_selection={hint_selection!r}")
    candidate_count, overlap_response_length = overlap_kr.shape
    if overlap_response_length != response_length:
        raise ValueError(f"overlap_kr R mismatch: {overlap_response_length} vs {response_length}")
    if candidate_count == 1:
        return torch.zeros(response_length, dtype=torch.long, device=device)
    if hint_selection == "token_optimal":
        return overlap_kr.argmax(dim=0)  # [response_length]
    if hint_selection == "sequence_optimal":
        if not step_token_spans:
            k_star_scalar = int(overlap_kr.sum(dim=-1).argmax().item())
            return torch.full((response_length,), k_star_scalar, dtype=torch.long, device=device)
        selected_candidates = torch.zeros(response_length, dtype=torch.long, device=device)
        for span in step_token_spans:
            t0, t1 = int(span[0]), int(span[1])
            t0 = max(0, min(t0, response_length))
            t1 = max(t0, min(t1, response_length))
            if t1 == t0:
                continue
            seg_score = overlap_kr[:, t0:t1].sum(dim=-1)  # [candidate_count]
            selected_candidates[t0:t1] = int(seg_score.argmax().item())
        return selected_candidates
    raise ValueError(
        f"Unknown --openclawrl-hint-selection: {hint_selection!r}. "
        "Expected 'shortest' / 'token_optimal' / 'sequence_optimal'."
    )


def gather_candidates(
    cand_tensor: torch.Tensor,  # [candidate_count, response_length, *]
    k_star_per_token: torch.Tensor,  # [response_length]
) -> torch.Tensor:
    """Slice ``cand_tensor[k*(t), t, ...]`` per token. Returns ``[response_length, *]``."""
    response_length = cand_tensor.size(1)
    trailing = cand_tensor.shape[2:]
    expand_shape = (1, response_length, *trailing)
    view_shape = (1, response_length, *(1 for _ in trailing))
    gather_idx = k_star_per_token.view(view_shape).expand(expand_shape)
    return torch.gather(cand_tensor, dim=0, index=gather_idx).squeeze(0)


def select_teacher_topk(
    student_indices: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    teacher_indices: torch.Tensor,
    *,
    subset_mode: str,
    hint_selection: str,
    native_teacher_indices: torch.Tensor | None,
    step_token_spans: list[list[int]] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return selected indices, log probabilities, overlaps, and candidate IDs.

    The loss validates the subset and hint modes before calling this function.
    Student indices have shape [R, Kq]. Teacher tensors have shape [K, R, Kp]
    for student/overlap subsets, or [R, Kp] after actor-side teacher selection.
    Teacher log probabilities and subset indices share the student indices'
    device. Their dtypes are float32 and long, respectively.
    Native teacher indices may be on another device. Student optimal selection
    uses those indices [K, R, Kp] for its overlap signal, moving them when needed.

    Shortest selects candidate zero. Token optimal maximizes overlap per token.
    Sequence optimal maximizes summed overlap per step span, or per response
    without spans. Ties choose the first candidate. Returned overlap and
    candidate tensors have shape [response_length]; both are zero for the teacher subset,
    whose selection already ran in the actor. Shortest also reports zero overlap.
    """
    response_length = student_indices.size(0)
    device = student_indices.device
    if subset_mode != "teacher":
        # Multi-candidate path (subset in {student, overlap}):
        # tensors are [K, R, K_p].
        if teacher_indices.dim() != 3 or teacher_indices.size(1) != response_length:
            raise ValueError(
                "teacher topk_cand shape mismatch: "
                f"t_idx_cand={tuple(teacher_indices.shape)} vs R={response_length}; expected [K, R, K_p]"
            )
        if teacher_log_probs.shape != teacher_indices.shape:
            raise ValueError(
                "teacher logp/idx cand shape mismatch: "
                f"lp={tuple(teacher_log_probs.shape)} idx={tuple(teacher_indices.shape)}"
            )

        # Selection signal (only computed when needed).
        if hint_selection == "shortest":
            candidate_overlap = None
        else:
            if subset_mode == "student":
                # Cand log-probs are at S^q (constant across k);
                # use each candidate's NATIVE top-K for the
                # overlap signal instead.
                if native_teacher_indices is None:
                    raise RuntimeError("student subset selection has no native teacher indices")
                selection_indices = native_teacher_indices.to(device=device, dtype=torch.long)
                if selection_indices.shape[1] != response_length:
                    raise ValueError(
                        "native_topk shape mismatch: "
                        f"native={tuple(selection_indices.shape)} vs R={response_length}; expected [K, R, K_p]"
                    )
            else:  # overlap
                selection_indices = teacher_indices
            candidate_overlap = overlap_count_per_token(student_indices, selection_indices)

        selected_candidates = select_candidate_per_token(
            candidate_overlap,
            hint_selection=hint_selection,
            step_token_spans=step_token_spans,
            response_length=response_length,
            device=device,
        )

        # Slice cand tensors at k*(t).
        selected_log_probs = gather_candidates(teacher_log_probs, selected_candidates)
        if subset_mode == "student":
            # Force kernel's S^p to S^q (cand indices are already
            # constant across k under student subset; this keeps
            # the contract explicit).
            selected_indices = student_indices
        else:  # overlap
            selected_indices = gather_candidates(teacher_indices, selected_candidates)

        if candidate_overlap is None:
            # ``shortest``: report selected-overlap as 0 so the
            # wandb panel is well-defined and visibly distinct
            # from the optimal modes (where it ≈ K_q for student).
            selected_overlap = torch.zeros(response_length, device=device, dtype=torch.float32)
        else:
            response_indices = torch.arange(response_length, device=candidate_overlap.device)
            selected_overlap = candidate_overlap[selected_candidates, response_indices].float()
    else:
        # Legacy single-cand path (subset == teacher): the actor
        # has already done k*-selection AND re-gathered
        # student-old log-probs at the chosen S^p. We consume
        # the collapsed [R, K_p] tensors directly.
        if teacher_indices.dim() != 2 or teacher_indices.size(0) != response_length:
            raise ValueError(
                f"teacher topk shape mismatch: t_idx={tuple(teacher_indices.shape)} vs R={response_length}; expected [R, K_p]"
            )
        if teacher_log_probs.shape != teacher_indices.shape:
            raise ValueError("teacher log-prob and index shapes must match")
        selected_indices = teacher_indices
        selected_log_probs = teacher_log_probs
        selected_overlap = torch.zeros(response_length, device=device, dtype=torch.float32)
        selected_candidates = torch.zeros(response_length, device=device, dtype=torch.long)
    return selected_indices, selected_log_probs, selected_overlap, selected_candidates
