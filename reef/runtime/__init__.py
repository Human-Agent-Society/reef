"""Runtime contracts and adapters for external model services.

``InferenceRuntime`` owns request execution and admission. ``TrainingRuntime``
prepares training jobs and exports checkpoints without requiring inference.
Neither inherits the other. The existing training backend coordinates their
candidate lifecycle: training exports a checkpoint, Reef selects activation or
rejection, and serving resumes only after durable publication.

``model_config.ModelConfig`` is the concrete, in-memory model selection shared
by a scenario and its recipe. It has no file paths or persistence behavior.

``deployment`` defines the minimal resource, inference-connection and training
component contracts used by Reef's model driver. ``training_job`` owns durable
job identity/replay, training/checkpoint ordering and commit-gated inference
resumption; concrete backends provide model operations and weight transport.
``inference_control`` coordinates engine pause/recovery and transport reconnect;
``health_monitor`` drains engine probes before lifecycle changes;
``inference_memory`` pairs acknowledged engine memory release/resume operations;
``weight_update`` supplies the transport lock's failure/phase semantics.

``sglang`` owns native SGLang launch, capture, engine control and inference
lifecycle without depending on Slime or Megatron.

Training batches and candidate evaluation contracts come from ``reef.core``.
This package never imports ``reef.train``.

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

from reef.runtime.adapters.executor_inference import ExecutorInferenceRuntime
from reef.runtime.adapters.executor_runtime import ExecutorTrainingRuntime
from reef.runtime.adapters.inference_proxy import InferenceProxyRuntime
from reef.runtime.adapters.ray_runtime import (
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
    "ExecutorInferenceRuntime",
    "ExecutorTrainGroupHandle",
    "ExecutorTrainingRuntime",
    "InferenceProxyRuntime",
    "InferenceRuntime",
    "ModelCandidate",
    "PreparedTrainingStep",
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
