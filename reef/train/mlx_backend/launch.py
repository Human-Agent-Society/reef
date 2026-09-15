"""Selecting the MLX backend from a deployment.

`training.backend: mlx` resolves here. The base class does the work: it refuses
the Ray, executor and inference-engine settings that an in-process integration
has no use for, and parses this runtime's own options through its factory. What
is left to declare is which runtime kind the deployment builds.
"""

from __future__ import annotations

from reef.train.deployment import InProcessTrainingDeployment


class MLXDeployment(InProcessTrainingDeployment):
    """Serve and train one LoRA adapter in the service process, on Apple Silicon."""

    #: Resolved through the runtime registry; the factory imports MLX only when
    #: it actually builds the runtime, so selecting another backend costs a
    #: deployment nothing.
    runtime_type = "mlx"
    #: The engine loads the base model itself, from a local path or the Hub.
    requires_local_model = False
