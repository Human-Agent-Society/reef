"""ATIF fixtures shared by backend and recipe contract tests."""

from dataclasses import asdict

from reef.core import AgentRecord, RequestType
from reef.core.trajectories import make_trajectory
from reef.train.types import TrajectoryItem


def policy_trajectory(
    source_agent_record_id,
    tokens,
    loss_mask,
    rollout_log_probs,
    reward,
    runtime_load_id=None,
    action_mask=(),
    rollout_created_at=None,
    turn_count=1,
    topk_indices=(),
    topk_log_probs=(),
    extras=None,
    runtime_load_spans=(),
) -> TrajectoryItem:
    item = recorded_trajectory(source_agent_record_id, {}, reward)
    return item.with_training(
        tokens=list(tokens),
        loss_mask=list(loss_mask),
        rollout_log_probs=list(rollout_log_probs),
        runtime_load_id=runtime_load_id,
        action_mask=list(action_mask),
        rollout_created_at=rollout_created_at,
        turn_count=turn_count,
        topk_indices=[list(row) for row in topk_indices],
        topk_log_probs=[list(row) for row in topk_log_probs],
        extras=extras or {},
        runtime_load_spans=[asdict(span) for span in runtime_load_spans],
    )


def recorded_trajectory(source_agent_record_id, payload, score, feedback=None, trajectory=()) -> TrajectoryItem:
    payloads = trajectory or (payload,)
    records = tuple(
        AgentRecord.create(
            scenario="test",
            request_type=RequestType.INFERENCE,
            payload=body,
            agent_record_id=(
                f"{source_agent_record_id}:{index}" if index < len(payloads) - 1 else source_agent_record_id
            ),
            created_at=0.0,
        )
        for index, body in enumerate(payloads)
    )
    return make_trajectory(records, score, feedback)
