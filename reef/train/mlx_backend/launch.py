"""The ``training.backend: mlx`` deployment.

The base class refuses Ray, executor and inference-engine settings, and
parses ``training.options`` with the ``mlx`` runtime factory's settings type.
"""

from __future__ import annotations

from reef.train.deployment import InProcessTrainingDeployment


class MLXDeployment(InProcessTrainingDeployment):
    """Serve and train one LoRA adapter in the service process, on Apple Silicon."""

    #: The runtime registry resolves this kind; the factory imports MLX only when it builds the runtime.
    runtime_type = "mlx"
    #: The engine loads the base model itself, from a local path or the Hub.
    requires_local_model = False
