"""Fail-fast checks before Reef starts the Slime training stack.

Everything here runs before any placement group or GPU worker exists, so a
configuration or storage problem stops the driver with a clear error instead
of a half-started cluster.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from reef.runtime.recovery import COMMITTED_MARKER_KEY, marker_committed, marker_rollouts, read_marker, write_marker
from reef.train.slime_backend.algorithm import SlimeAlgorithm
from reef.train.slime_backend.reef_adapters.arguments import SlimeArguments
from reef.train.slime_backend.reef_adapters.training_job.storage import CheckpointStorage, RetentionConfig

MEGATRON_INIT_PATH = "reef.train.slime_backend.reef_adapters.worker_hooks.initialize_megatron_objective"
CRITIC_ARGS_HOOK_PATH = "reef.train.slime_backend.reef_adapters.worker_hooks.configure_critic_objective"
REEF_ROLLOUT_DATA_KEYS = (
    "producing_runtime_load_spans",
    "producing_runtime_load_ids",
)


def validate_bridge_args(args: SlimeArguments, spec: SlimeAlgorithm | None) -> None:
    """Reject slime driver arguments the bridge cannot run with."""
    num_rollout = args.num_rollout
    if not isinstance(num_rollout, int) or isinstance(num_rollout, bool) or num_rollout <= 0:
        raise ValueError("the Reef bridge requires a positive --num-rollout")
    save_hf = args.save_hf
    if not isinstance(save_hf, str) or "{rollout_id}" not in save_hf:
        raise ValueError("the Reef bridge requires --save-hf with a {rollout_id} path template")
    validate_advantage_computation(args, spec)
    if args.debug_train_only:
        raise ValueError("the Reef bridge requires a live inference router; remove --debug-train-only")
    if args.debug_rollout_only:
        raise ValueError("the Reef bridge has no internal rollout loop; remove --debug-rollout-only")
    rollout_num_gpus = args.rollout_num_gpus
    if not args.rollout_external and (
        not isinstance(rollout_num_gpus, int) or isinstance(rollout_num_gpus, bool) or rollout_num_gpus <= 0
    ):
        raise ValueError("the Reef bridge requires a positive --rollout-num-gpus for its local inference router")
    colocate = bool(args.colocate)
    if colocate and (not args.offload_train or not args.offload_rollout):
        raise ValueError("the Reef bridge requires --offload-train and --offload-rollout with --colocate")
    if args.offload_rollout and not colocate:
        raise ValueError("the Reef bridge does not support --offload-rollout because Reef needs serving to stay live")
    if args.keep_lora_base_resident:
        # Only a frozen base may stay resident. Full-weight training rewrites
        # the served weights, which is exactly what releasing them is for, and
        # a non-colocated engine never releases anything to begin with.
        # Spelled out rather than calling megatron_lora_enabled, which would
        # put torch on this module's import path. It is the same predicate:
        # prepare_bridge derives its own `lora` from that helper, and the helper
        # is this comparison, so the two cannot disagree.
        if args.megatron_lora_rank <= 0:
            raise ValueError("--keep-lora-base-resident requires LoRA training; set --megatron-lora-rank")
        if not colocate:
            raise ValueError(
                "--keep-lora-base-resident applies to colocated training; without --colocate nothing is released"
            )
    save = args.save
    if not isinstance(save, str) or not save.strip():
        raise ValueError("the Reef bridge requires --save for Megatron recovery checkpoints")


def configure_megatron_runtime(args: SlimeArguments) -> None:
    """Install Reef's worker initialization through Slime's public hook."""
    if args.loss_family is None:
        return
    current = args.custom_megatron_init_path
    if current and current != MEGATRON_INIT_PATH:
        args.reef_chained_megatron_init_path = current
    args.custom_megatron_init_path = MEGATRON_INIT_PATH
    critic_hook = args.custom_critic_args_hook_path
    if critic_hook and critic_hook != CRITIC_ARGS_HOOK_PATH:
        args.reef_chained_critic_args_hook_path = critic_hook
    args.custom_critic_args_hook_path = CRITIC_ARGS_HOOK_PATH


def configure_rollout_runtime(args: SlimeArguments) -> None:
    """Declare Reef's per-sample columns through Slime's payload hook."""
    configured = tuple(args.custom_rollout_data_keys or ())
    args.custom_rollout_data_keys = tuple(dict.fromkeys((*configured, *REEF_ROLLOUT_DATA_KEYS)))


def validate_advantage_computation(args: SlimeArguments, spec: SlimeAlgorithm | None) -> None:
    """The bridge supplies training signals externally, with spec-declared exceptions.

    A loss family that keeps Slime's advantage pass declares
    ``allows_slime_advantage_computation``.  Without a resolved family
    (``prepare_bridge`` called directly), any registered family that allows
    it is accepted.
    """
    if not args.compute_advantages_and_returns:
        return
    if spec is not None:
        if spec.allows_slime_advantage_computation:
            return
    else:
        from reef.train.slime_backend.loss_families import LOSS_FAMILIES

        if any(LOSS_FAMILIES.resolve(name).allows_slime_advantage_computation for name in LOSS_FAMILIES.names):
            return
    raise ValueError(
        "the Reef bridge supplies training signals externally; pass "
        "--disable-compute-advantages-and-returns to the slime driver"
    )


def prepare_checkpoint_storage(args: SlimeArguments, retention: RetentionConfig) -> CheckpointStorage:
    """Build the checkpoint store, refuse ambiguous or blocked state, pin paths.

    Rewrites ``args.save_hf`` / ``args.save`` (and ``args.critic_save`` when a
    critic trains) to the storage's resolved absolute paths so the workers and
    the bridge actor agree on locations. A critic run without an explicit
    ``--critic-save`` gets ``<save>-critic`` so the critic's weights and
    optimizer survive restarts instead of cold-starting the value head.
    """
    critic_root = (args.critic_save or f"{args.save}-critic") if args.use_critic else None
    storage = CheckpointStorage(
        retention,
        hf_template=args.save_hf,
        megatron_root=args.save,
        critic_root=critic_root,
        source_hf=args.hf_checkpoint,
        source_megatron=args.load,
        lora=bool(args.megatron_lora_rank),
        critic_save_interval=args.critic_save_interval,
    )
    marker = read_marker(storage.marker_path)
    if marker is not None and marker["status"] == "RUNNING":
        settled = committed_marker_to_restore(storage, marker, args.megatron_lora_rank)
        if settled is None:
            raise RuntimeError(running_job_message(storage, marker, args.megatron_lora_rank))
        print(
            f"[reef] training job {marker['job_id']} was stopped before it saved rollout {marker['rollout_id']}; "
            f"restoring the committed marker of rollout {settled['rollout_id']} so its batch trains again. "
            f"Replaced marker: {json.dumps(dict(marker), sort_keys=True)}",
            file=sys.stderr,
            flush=True,
        )
        write_marker(storage.marker_path, settled)
        marker = read_marker(storage.marker_path)
    if marker is not None and marker["status"] in {"REJECTING", "REJECTED"}:
        # The newest training checkpoint still contains the declined candidate;
        # it cannot reconstruct the incumbent engines or committed adapters.
        raise RuntimeError(
            f"training job {marker['job_id']} is {marker['status']}; "
            "restore the committed checkpoint before restarting inference"
        )
    storage_plan = storage.validate_capacity(active_rollouts=marker_rollouts(marker))
    if storage_plan["blocked"]:
        reasons = "; ".join(storage_plan["reasons"])
        raise RuntimeError(f"checkpoint storage preflight blocked bridge startup: {reasons}")
    args.save_hf, args.save = storage.hf_template, str(storage.megatron_root)
    if storage.critic_root is not None:
        args.critic_save = str(storage.critic_root)
    return storage


def tracker_iteration_state(storage: CheckpointStorage) -> tuple[Path, int | None, str]:
    """Megatron's tracker under ``--save``, the iteration it names, and how to describe it."""
    tracker = storage.megatron_root / "latest_checkpointed_iteration.txt"
    try:
        tracker_text: str | None = tracker.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        tracker_text = None
    except (OSError, ValueError) as error:
        tracker_text = f"<{error}>"
    if tracker_text is None:
        return tracker, None, "is missing"
    if tracker_text.isdecimal():
        return tracker, int(tracker_text), f"names iteration {int(tracker_text)}"
    return tracker, None, f"is unreadable: {tracker_text!r}"


def rollout_paths(storage: CheckpointStorage, rollout_id: int) -> list[str]:
    """The checkpoint assets and record for ``rollout_id`` that exist on disk."""
    record_path = storage.records_root / f"{rollout_id:020d}.json"
    return [
        str(path) for path in (*storage.asset_paths(rollout_id), record_path) if path.exists() or path.is_symlink()
    ]


def committed_marker_to_restore(
    storage: CheckpointStorage, marker: Mapping[str, Any], megatron_lora_rank: int
) -> dict[str, Any] | None:
    """The settled marker a ``RUNNING`` job replaced, when restoring it is safe.

    This is the documented full-weight state: the tracker names an earlier
    rollout ``P``, the job left nothing for its own rollout, and the newest
    checkpoint record is ``P``'s. The marker the job replaced must be kept in
    the ``RUNNING`` marker, settle ``P``'s job, and carry Reef's commit
    acknowledgement, which a rejected job never gets. Any other state stays
    ambiguous.
    """
    committed = marker.get(COMMITTED_MARKER_KEY)
    if megatron_lora_rank > 0 or not isinstance(committed, Mapping) or not marker_committed(committed):
        return None
    _, tracker_iteration, _ = tracker_iteration_state(storage)
    previous_rollout = committed.get("rollout_id")
    if tracker_iteration is None or tracker_iteration != previous_rollout or tracker_iteration >= marker["rollout_id"]:
        return None
    if rollout_paths(storage, marker["rollout_id"]):
        return None
    parent_runtime_load_id = marker.get("parent_runtime_load_id")
    runtime_load_id = committed.get("runtime_load_id")
    if not isinstance(runtime_load_id, str) or not runtime_load_id:
        return None
    if parent_runtime_load_id is not None and parent_runtime_load_id != runtime_load_id:
        return None
    checkpoint_path = committed.get("checkpoint_path")
    if (
        not isinstance(checkpoint_path, str)
        or Path(checkpoint_path).is_symlink()
        or not Path(checkpoint_path).is_dir()
    ):
        return None
    record_paths = sorted(storage.records_root.glob("*.json"))
    if not record_paths or record_paths[-1] != storage.records_root / f"{tracker_iteration:020d}.json":
        return None
    try:
        record = json.loads(record_paths[-1].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (
        not isinstance(record, Mapping)
        or record.get("status") != "COMPLETE"
        or record.get("job_id") != committed.get("job_id")
    ):
        return None
    return dict(committed)


def running_job_message(storage: CheckpointStorage, marker: Mapping[str, Any], megatron_lora_rank: int) -> str:
    """List what a job stopped while ``RUNNING`` left on disk, and where its recovery is documented.

    These files cannot always show whether the job's optimizer step reached the
    checkpoint. They also cannot show whether the job before it was committed.
    So the message states facts only. The troubleshooting guide gives a manual
    recovery for one state only. In that state, the run is full-weight, its
    tracker names an earlier rollout, and the job left no files for its own
    rollout. Every other state needs the checkpoints restored from a copy, and
    the guide says which copy.
    """
    rollout_id = marker["rollout_id"]
    tracker, tracker_iteration, tracker_state = tracker_iteration_state(storage)
    paths = rollout_paths(storage, rollout_id)
    per_scenario_lora = megatron_lora_rank > 0
    lora_state = f"yes (--megatron-lora-rank {megatron_lora_rank})" if per_scenario_lora else "no"
    troubleshooting_entry = "'A restart fails with ambiguous training job' in the troubleshooting guide"
    restore_from_copy = f"restore {storage.root} from a copy; the entry says which copy"
    if not per_scenario_lora and tracker_iteration is not None and tracker_iteration < rollout_id and not paths:
        recovery = (
            f"{troubleshooting_entry} gives a manual recovery for this state, which applies only if all of its "
            f"conditions hold, including that the job for rollout {tracker_iteration} was committed; otherwise "
            f"{restore_from_copy}"
        )
    else:
        recovery = f"{troubleshooting_entry} gives no manual recovery for this state; {restore_from_copy}"
    return (
        f"ambiguous training job {marker['job_id']}: {storage.marker_path} is RUNNING for rollout {rollout_id}; "
        f"Megatron's tracker {tracker} {tracker_state}; paths for rollout {rollout_id}: "
        f"{', '.join(paths) or 'none'}; per-scenario LoRA: {lora_state}; {recovery}"
    )
