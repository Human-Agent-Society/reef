"""Names shared across the processes of the Reef-to-backend bridge contract.

The Reef service and a training backend run as separate processes that find
each other only by name, so these values must agree on both sides:

- ``DEFAULT_ACTOR_NAME`` / ``DEFAULT_NAMESPACE`` locate the named Ray bridge
  actor (``connect_ray_runtime`` on the Reef side, ``start_bridge`` and the
  driver healthcheck on the backend side).
- ``LATEST_JOB_MARKER_FILENAME`` is the durable training-job marker the
  backend writes next to its checkpoints and must recognize across restarts.
"""

DEFAULT_ACTOR_NAME = "reef-train-bridge"
DEFAULT_NAMESPACE = "reef"
LATEST_JOB_MARKER_FILENAME = ".reef-latest-job.json"
#: Per-scenario publication ledger of a LoRA training bridge; sits beside the marker.
SCENARIO_LEDGER_FILENAME = "reef_scenarios.json"
#: Rank-local adapter-slot snapshots of a LoRA training group; sits beside the Megatron checkpoint.
ADAPTER_SLOTS_DIRNAME = "reef_adapter_slots"

__all__ = [
    "ADAPTER_SLOTS_DIRNAME",
    "DEFAULT_ACTOR_NAME",
    "DEFAULT_NAMESPACE",
    "LATEST_JOB_MARKER_FILENAME",
    "SCENARIO_LEDGER_FILENAME",
]
