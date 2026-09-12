"""Apple Silicon training and serving on MLX, behind Reef's runtime contract.

The package is importable without MLX installed: nothing here imports ``mlx``
at module scope, and :class:`~reef.train.mlx_backend.runtime.MLXRuntimeFactory`
pulls the engine in only when a deployment actually selects the ``mlx``
runtime kind. That is what lets the wheel ship this adapter while keeping
``mlx`` an optional extra.

Selected from a deployment config with::

    reef:
      runtime_type: mlx
      runtime_config:
        checkpoint_dir: work/checkpoints

See ``reef/train/mlx_backend/runtime.py`` for the topology this models and
what it deliberately does not do.
"""

from reef.train.mlx_backend.runtime import MLXRuntime, MLXRuntimeFactory

__all__ = ["MLXRuntime", "MLXRuntimeFactory"]
