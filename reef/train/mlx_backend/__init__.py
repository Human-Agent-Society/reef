"""Apple Silicon serving and training on MLX, behind Reef's runtime contract.

A deployment selects the backend with ``training.backend: mlx`` and sets
its options under ``training.options``; see
``recipes/openclawrl/examples/openclawrl/serve-mlx.yaml``::

    inference:
      model-path: mlx-community/Qwen3.5-9B-MLX-4bit
    training:
      backend: mlx
      options:
        checkpoint_dir: work/mlx/checkpoints

Only ``engine``, ``training``, ``losses`` and ``gated_delta`` import MLX.
``config`` declares the options, ``contracts`` the engine interface,
``runtime`` and ``serving`` the two runtime halves, and ``inference`` the
chat handler; these import without the optional ``mlx`` extra.
"""

from reef.train.mlx_backend.runtime import MLXRuntime, MLXRuntimeFactory

__all__ = ["MLXRuntime", "MLXRuntimeFactory"]
