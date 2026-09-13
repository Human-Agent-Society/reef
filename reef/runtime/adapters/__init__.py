"""Runtime adapters for external model services."""

from reef.runtime.adapters.executor_inference import ExecutorInferenceRuntime
from reef.runtime.adapters.executor_runtime import ExecutorTrainingRuntime
from reef.runtime.adapters.inference_proxy import InferenceProxyRuntime
from reef.runtime.adapters.ray_runtime import (
    RayRuntimeError,
    RayTrainGroupHandle,
    RemoteRayTrainGroupHandle,
    connect_ray_runtime,
)

__all__ = [
    "ExecutorInferenceRuntime",
    "ExecutorTrainingRuntime",
    "InferenceProxyRuntime",
    "RayRuntimeError",
    "RayTrainGroupHandle",
    "RemoteRayTrainGroupHandle",
    "connect_ray_runtime",
]
