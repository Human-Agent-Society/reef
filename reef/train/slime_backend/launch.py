"""Lightweight process defaults for the supported Slime training environment."""

from collections.abc import Mapping


def driver_environment(environ: Mapping[str, str]) -> dict[str, str]:
    """Set the CUDA/NCCL defaults used by the shipped Slime deployments."""
    return {
        "CUDA_DEVICE_MAX_CONNECTIONS": environ.get("CUDA_DEVICE_MAX_CONNECTIONS", "1"),
        "NCCL_NVLS_ENABLE": environ.get("NCCL_NVLS_ENABLE", "0"),
    }
