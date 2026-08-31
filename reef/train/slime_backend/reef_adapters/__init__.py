"""Slime-side adapters exposed to Reef.

One training step, end to end: the Reef service reserves a batch and drives
the named bridge actor (``TrainBridgeActor``) through its existing job API.

1. ``prepare_training_step(batch, step_preparer, algorithm_state)`` is pure: it
   runs the loss family's step preparer and returns a
   ``PreparedTrainingStep`` whose complete wire payload and next algorithm
   state Reef persists in its scenario commit log. Nothing on the Slime side
   changes.
2. ``execute_training_job`` trains through a durable checkpoint. The payload's
   content hash is the job identity; re-sending it after a crash replays the
   marker rather than training twice. ``update_serving_weights`` publishes that
   checkpoint but keeps generation paused until Reef acknowledges its commit.

The job marker lets a restart on either side resume the transaction exactly
where it stopped.

Keep the Ray bridge and the inference backends lazy so token-capture adapters
remain importable without loading the complete Slime GPU stack.
"""

from importlib import import_module
from typing import Any

from reef.train.slime_backend.reef_adapters.training_job.storage import RetentionConfig

_LAZY_EXPORTS = {
    "DEFAULT_BRIDGE_ACTOR_NAME": "bridge",
    "DEFAULT_NAMESPACE": "bridge",
    "TrainBridgeActor": "bridge",
    "TrainBridgeActorImpl": "bridge",
    "start_bridge": "bridge",
    "SGLangChatTrainingInferenceBackend": "sglang.chat",
}

__all__ = [
    "DEFAULT_BRIDGE_ACTOR_NAME",
    "DEFAULT_NAMESPACE",
    "RetentionConfig",
    "SGLangChatTrainingInferenceBackend",
    "TrainBridgeActor",
    "TrainBridgeActorImpl",
    "start_bridge",
]


def __getattr__(name: str) -> Any:
    module = _LAZY_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value
