"""Runtime adapters for external model services."""

from reef.runtime.adapters.executor_runtime import ExecutorModelRuntime
from reef.runtime.adapters.inference_proxy import InferenceProxyRuntime
from reef.runtime.adapters.ray_runtime import (
    RayRuntime,
    RayRuntimeError,
    RayTrainGroupHandle,
    RemoteRayTrainGroupHandle,
    connect_ray_runtime,
)

__all__ = [
    "ExecutorModelRuntime",
    "InferenceProxyRuntime",
    "RayRuntime",
    "RayRuntimeError",
    "RayTrainGroupHandle",
    "RemoteRayTrainGroupHandle",
    "connect_ray_runtime",
]
