"""Runtime contracts and adapters for external model services.

Every request needs an ``InferenceRuntime`` — it is what ``service/`` calls
to answer with the provider's API — which is why the package belongs to
serving even though ``train/`` also drives it for recipes that train.
``TrainingRuntime`` extends it with the durable candidate lifecycle:
candidate training exports a checkpoint without changing serving, and Reef
evaluates it before choosing activation or rejection.

Boundaries this package holds:

- No concrete training backend is imported here. Backends implement
  ``TrainingGroupHandle`` and ``ExecutorTrainingRuntime`` drives only the
  handle; ``reef.train.slime`` never appears at module scope.
- Malformed results and missing capabilities surface as contract errors
  (``RuntimeContractError``, ``TrainingRuntimeError``), never as silent fallbacks.
  The default ``restore_checkpoint`` refuses rather than moving the artifact
  head under an engine that kept newer weights.
- Surfaces see runtimes only structurally, through ``ServingRuntime`` and
  ``WeightRuntime`` in ``surface/base.py``; nothing in ``surface/`` imports
  this package.

Adding a runtime kind: subclass ``RuntimeFactory``, set its ``kind``, and
decorate the class with ``@register_runtime_kind`` in a module imported at
boot. Or set the config ``type`` to a dotted ``package.module:factory_name``
reference, which ``runtime_factory_for`` imports on resolution.
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
from reef.runtime.base import InferenceRuntime, PreparedTrainingStep, TrainingJobResult, TrainingRuntime
from reef.runtime.candidates import ActivatedModel, ModelCandidate
from reef.runtime.executor import Executor, ExecutorConfig, ExecutorFuture, WorkerSpec
from reef.runtime.proxy import resolve_proxy_runtime
from reef.runtime.registry import (
    RuntimeConfigError,
    RuntimeFactory,
    RuntimeRegistry,
    register_runtime_kind,
    runtime_factory_for,
    runtime_kinds,
)
from reef.runtime.training_group import ExecutorTrainGroupHandle, TrainingGroupHandle, TrainingRuntimeError

__all__ = [
    "ActivatedModel",
    "Executor",
    "ExecutorConfig",
    "ExecutorFuture",
    "ExecutorTrainGroupHandle",
    "ExecutorTrainingRuntime",
    "InferenceProxyRuntime",
    "InferenceRuntime",
    "ModelCandidate",
    "PreparedTrainingStep",
    "RayRuntime",
    "RayRuntimeError",
    "RayTrainGroupHandle",
    "RemoteRayTrainGroupHandle",
    "RuntimeConfigError",
    "RuntimeFactory",
    "RuntimeRegistry",
    "TrainingGroupHandle",
    "TrainingJobResult",
    "TrainingRuntime",
    "TrainingRuntimeError",
    "WorkerSpec",
    "connect_ray_runtime",
    "register_runtime_kind",
    "resolve_proxy_runtime",
    "runtime_factory_for",
    "runtime_kinds",
]
