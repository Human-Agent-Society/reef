"""Check captured native episode tensors without executing models or reconstructing prompts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import cast

if __package__:
    from .report import JsonObject, read_object
else:
    from report import JsonObject, read_object


def check_native_sample(capture: JsonObject, episode: JsonObject, require_multi_turn: bool = False) -> JsonObject:
    """Compare any nonempty episode input to every canonical sampled turn."""
    references = cast(list[str], episode["references"])
    turns = cast(list[JsonObject], capture["turns"])
    if not turns:
        raise ValueError("native sample requires at least one sampled turn")
    if capture["report_id"] != episode["report_id"] or capture["references"] != references:
        raise ValueError("captured training input does not match the terminal report and receipt order")
    if [turn["receipt"] for turn in turns] != references or len(set(references)) != len(references):
        raise ValueError("native sample does not contain every ordered unique receipt")
    if require_multi_turn and len(turns) < 2:
        raise ValueError("native sample diagnostic requires a real multi-turn episode")
    tokens = cast(list[int], capture["tokens"])
    mask = cast(list[int], capture["loss_mask"])
    log_probs = cast(list[float], capture["rollout_log_probs"])
    teacher = cast(list[int], capture["teacher_tokens"])
    if not 0 < len(mask) < len(tokens) or len(mask) != len(log_probs):
        raise ValueError("native sample has invalid token/mask/log-probability lengths")
    if any(value not in (0, 1) or isinstance(value, bool) for value in mask):
        raise ValueError("native sample has an invalid loss mask")
    if any(not math.isfinite(value) for value in log_probs):
        raise ValueError("native sample has non-finite rollout log probabilities")
    if teacher != tokens:
        raise ValueError("OPD teacher must score the exact recorded student token sequence")
    native_tokens: list[int] = []
    native_mask: list[int] = []
    selected_ids: list[int] = []
    selected_log_probs: list[float] = []
    initial_prompt_length = 0
    versions: set[str] = set()
    for index, turn in enumerate(turns):
        turn_tokens = cast(list[int], turn["tokens"])
        turn_mask = cast(list[int], turn["loss_mask"])
        turn_probs = cast(list[float], turn["rollout_log_probs"])
        if not 0 < len(turn_mask) < len(turn_tokens) or any(value != 1 for value in turn_mask):
            raise ValueError("every sampled assistant token must remain selected")
        if len(turn_probs) != len(turn_mask) or any(not math.isfinite(value) for value in turn_probs):
            raise ValueError("each turn requires complete finite rollout log probabilities")
        version = turn["runtime_load_id"]
        if not isinstance(version, str) or not version:
            raise ValueError("each turn must have a canonical runtime load ID")
        versions.add(version)
        prompt = turn_tokens[: -len(turn_mask)]
        if index == 0:
            initial_prompt_length = len(prompt)
            native_tokens = list(prompt)
            native_mask = [0] * len(prompt)
        elif prompt[: len(native_tokens)] != native_tokens:
            raise ValueError("native histories drifted or omitted a sampled assistant turn")
        else:
            appended_context = prompt[len(native_tokens) :]
            native_tokens.extend(appended_context)
            native_mask.extend([0] * len(appended_context))
        response = turn_tokens[-len(turn_mask) :]
        native_tokens.extend(response)
        native_mask.extend(turn_mask)
        selected_ids.extend(response)
        selected_log_probs.extend(turn_probs)
    if len(versions) != 1 or capture["runtime_load_id"] not in versions:
        raise ValueError("native episode contains mixed runtime load IDs")
    canonical_turns = cast(list[JsonObject], episode["turns"])
    if native_tokens != tokens or native_mask[initial_prompt_length:] != mask:
        raise ValueError("actual native sample changed tokens or trained tool/context positions")
    actual_ids = [token for token, selected in zip(tokens[-len(mask) :], mask, strict=True) if selected]
    actual_probs = [value for value, selected in zip(log_probs, mask, strict=True) if selected]
    if actual_ids != selected_ids or actual_probs != selected_log_probs:
        raise ValueError("native sample did not retain all assistant IDs and exact log probabilities")
    if len(canonical_turns) != len(turns):
        raise ValueError("captured turns differ from the canonical episode")
    for source, captured in zip(canonical_turns, turns, strict=True):
        record = cast(JsonObject, source["record"])
        payload = cast(JsonObject, record["payload"])
        native = cast(JsonObject, cast(JsonObject, payload["response"])["training"])
        if any(
            native[key] != captured[key] for key in ("tokens", "loss_mask", "rollout_log_probs", "runtime_load_id")
        ):
            raise ValueError("capture differs from the canonical authenticated inference record")
        if cast(JsonObject, record["artifact_ref"])["release_id"] != episode["release_id"]:
            raise ValueError("canonical turn release differs from the sampled release")
    weight = capture["distill_sample_weight"]
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
        raise ValueError("native sample weight must be finite and non-negative")
    return {
        "report_id": capture["report_id"],
        "turn_count": len(turns),
        "assistant_token_count": len(selected_ids),
        "masked_context_token_count": mask.count(0),
        "teacher_suffix_identity": True,
        "teacher_student_sequence_identity": True,
        "all_assistant_tokens_selected": True,
        "tool_context_zero_loss": True,
        "runtime_load_id": capture["runtime_load_id"],
        "active": weight > 0,
        "capture_stage": capture["capture_stage"],
    }


def qualify_inputs(run_root: Path) -> JsonObject:
    """Check every training input and require active multi-turn coverage across the campaign."""
    episodes = [read_object(path) for path in sorted((run_root / "episodes").glob("*.json"))]
    training = [episode for episode in episodes if episode["phase"] == "train"]
    if not training:
        raise ValueError("native qualification requires recorded training episodes")
    checks = [
        check_native_sample(read_object(run_root / "teacher-records" / f"{episode['report_id']}.json"), episode)
        for episode in training
    ]
    if not any(check["active"] for check in checks):
        raise ValueError("native qualification has no effective distillation signal")
    if not any(check["active"] and cast(int, check["turn_count"]) >= 2 for check in checks):
        raise ValueError("native qualification requires at least one active multi-turn training episode")
    return {"native_input_checks_passed": True, "episodes": cast(list, checks), "optimizer_execution_verified": False}
