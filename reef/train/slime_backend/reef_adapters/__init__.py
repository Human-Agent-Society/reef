"""Adapters between the Slime training component and Slime's workers.

``reef.train.slime_backend.component`` runs each training job and saves its
checkpoints. These adapters supply Slime's arguments, sample packing,
optimizer execution and checkpoint storage. Inference adapters live in
``reef.inference`` and are assembled independently by Reef.
"""

from reef.train.slime_backend.reef_adapters.training_job.storage import RetentionConfig

__all__ = ["RetentionConfig"]
