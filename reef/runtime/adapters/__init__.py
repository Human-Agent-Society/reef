"""Runtime adapters for external model services.

Each adapter module stays importable without its execution dependencies:
``ray_runtime`` does not import Ray at module scope. The heavy import happens
inside the factory, when a deployment actually selects that kind.
"""

from reef.runtime.adapters.executor_runtime import ExecutorTrainingRuntime
from reef.runtime.adapters.inference_proxy import InferenceProxyRuntime
from reef.runtime.adapters.ray_runtime import (
    RayRuntime,
    RayRuntimeError,
    RayTrainGroupHandle,
    RemoteRayTrainGroupHandle,
    connect_ray_runtime,
)

__all__ = [
    "ExecutorTrainingRuntime",
    "InferenceProxyRuntime",
    "RayRuntime",
    "RayRuntimeError",
    "RayTrainGroupHandle",
    "RemoteRayTrainGroupHandle",
    "connect_ray_runtime",
]
