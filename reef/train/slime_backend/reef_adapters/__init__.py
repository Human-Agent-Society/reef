"""Native Slime training backend and checkpoint storage.

Reef's runtime coordinator owns the training job API, admission, durable marker
transitions and the serving commit gate. Slime adapters supply sample packing,
optimizer execution, checkpoint I/O and tensor transport. Inference adapters
live in ``reef.inference`` and are assembled independently by Reef.
"""

from reef.train.slime_backend.reef_adapters.training_job.storage import RetentionConfig

__all__ = ["RetentionConfig"]
