Worker executors
================

Reef separates service orchestration and model semantics from worker execution. Training and inference runtimes own
checkpoint production and serving respectively; the existing training backend
coordinates activation with durable commit acknowledgement.
An ``Executor`` owns worker launch, ordered control RPC, health and shutdown.
The interface follows vLLM's executor pattern: configuration selects a concrete
class, while callers use the same methods for each backend.

.. code:: mermaid

   flowchart TD
       D[reef serve / dependency graph] --> SE[Service Executor]
       SE --> LW[Local ProcessWorker]
       SE --> RW[Ray ProcessWorker]
       SE --> CW[Custom executor]
       LW --> SV[Inference / training driver / Reef]
       RW --> SV
       T[ExecutorTrainingRuntime] --> H[TrainingGroupHandle]
       R[ExecutorInferenceRuntime] --> H
       H --> C[Coordinator Executor]
       C --> B[Training coordinator / Slime bridge]
       B --> G[SlimeTrainGroup: train, checkpoint, publish]
       G --> E[Worker Executor]
       E --> S[SlimeRayExecutor]
       E --> P[Custom Slime-compatible Executor]
       S --> W[Megatron workers]
       B --> BP[Local batch processor: tensorization / DP partitioning]
       B --> I[Inference control actor]
       W --> I
       I --> RE[Rollout Executor]
       RE --> SR[SGLangExecutor: SGLang engines / routers / update lock]
       RE --> CR[Custom rollout executor]

The coordinator executor targets one worker for each training-job RPC. The
training executor dispatches rank operations to the model workers. This keeps
checkpoint/publication transactions from being accidentally broadcast and
executed more than once.

Built-in executors
------------------

``uni``
   One worker in the caller's process. RPC may use one local thread. More
   than one worker is rejected. A GPU scorer uses the caller's CUDA visibility;
   no global visibility mask or cluster resource reservation is changed.
   For services, one ``ProcessWorker`` starts the actual service subprocess.

``mp``
   Independent worker ranks launched with Python ``spawn``. Each rank has
   ordered RPC, constructor readiness, health checks and bounded shutdown.
   Worker classes, scorers and arguments must be spawn-pickleable (normally
   module-level classes/functions), not closures or objects holding locks.
   No Ray installation is needed. Declared GPU scorer requirements produce
   disjoint per-rank CUDA masks before worker imports/construction. This does
   not implement model-parallel collectives. Timeouts stop waiting, not the
   work, and never retry RPC.

``ray``
   Launches Ray actors from ordered ``WorkerSpec`` objects. Worker options
   include Ray resources, placement strategies and runtime environments.
   New actors disable automatic task retries. ``RayExecutor.from_workers``
   attaches existing handles; it borrows them unless ``owned=True`` is explicit.
   Borrowed shutdown closes the executor without terminating its actors.

``auto`` (default)
   Resolves to ``uni`` for one worker or ``mp`` for multiple local workers.
   Existing multi-worker Ray placement context or explicit cluster/Ray actor
   options select ``ray``. The former thread-pool ``local`` backend is removed,
   not an alias: migrate it to ``auto`` or ``uni`` with one worker.

A Python class or ``package.module:ExecutorSubclass`` / ``package.module.Class``
selects a custom executor. CPU-only execution does not import Ray, Slime,
Torch or SGLang. Only declared local GPU requirements probe CUDA capacity.

.. code:: python

   from reef.runtime.executor import Executor, ExecutorConfig, WorkerSpec, resolve

   class Counter:
       def __init__(self, rank):
           self.rank = rank
           self.value = 0

       def increment(self, amount):
           self.value += amount
           return self.rank, self.value

   if __name__ == "__main__":  # required for spawn
       executor = Executor.create(ExecutorConfig(
           workers=tuple(WorkerSpec(Counter, args=(rank,)) for rank in range(2)),
       ))  # auto -> mp
       try:
           pending = executor.collective_rpc("increment", args=(3,), non_block=True)
           results = resolve(pending, timeout=10)  # [(0, 3), (1, 3)]
       finally:
           executor.shutdown()

RPC and lifetime contracts
-------------------------

``collective_rpc`` dispatches to every worker before waiting, returning results
in worker/rank order. ``rpc(rank, ...)`` addresses one rank. ``non_block=True``
returns an ``ExecutorFuture``; ``resolve`` waits on futures and nested lists or
tuples under one shared deadline. Ordinary results remain ordinary values.
Backend object references never cross this control-RPC interface as futures.

A timeout bounds the wait. It does not cancel GPU work or repeat a mutating
operation. Training-job recovery must consult the durable checkpoint/commit
state before deciding what can be resumed. ``shutdown`` terminates only owned
workers; a failed constructor cleans up workers already acquired by that
executor. Backend-specific launchers are responsible for tracking every
allocation, including partially initialized workers.

Executors expose a terminal ``failure`` (``ExecutorFailure``) and
``register_failure_listener(listener)``. An observer implements
``on_executor_failure(failure)``; it is called once, including when registered
after failure. Listeners must be nonblocking and must not call shutdown from
the callback. Observer exceptions are logged without hiding worker failure.
Normal shutdown and ordinary worker-method/scorer errors are not terminal
infrastructure failures; wait timeouts do not imply worker death.

``mp`` monitors process sentinels, including idle ranks. Owned ``ray`` groups
monitor actor readiness in the background; a queued probe on a busy actor is
not considered a failure. For borrowed/serialized Ray groups, registering a
listener starts process-local monitoring; listeners are not serialized with
actor handles. Local workers share the caller's process and have no separate
process-death detector. Custom executors report terminal failure via ``_fail``.
On detected failure, outstanding waits fail promptly with
``ExecutorFailedError``, new submissions are rejected, and owned worker peers
are retired. Borrowed actors remain owned by their original launcher. There
is no automatic recreation, request replay, or distributed-group recovery.

Implement ``_init_executor``, ``rpc``, ``collective_rpc``, ``check_health`` and
``shutdown`` in an ``Executor`` subclass. Backend-specific launch configuration
belongs in ``ExecutorConfig.options``. GPU tensors, optimizer state, NCCL
collectives and KV transfers remain the model backend's responsibility.
``reinitialize_distributed`` is an optional capability and raises
``NotImplementedError`` by default; changing executors does not imply live
topology resizing or checkpoint resharding.

Training runtime configuration
------------------------------

Existing ``type: ray_training`` configurations keep discovering a named Ray
bridge in the selected namespace and return separate ``ExecutorTrainingRuntime``
and ``ExecutorInferenceRuntime`` instances. ``RayTrainGroupHandle`` remains a
compatibility alias for ``TrainingGroupHandle``.

For a custom coordinator, ``executor_training`` creates its executor from
configuration. The selected worker implements ``TrainingGroupHandle``'s RPC
methods, including durable health and checkpoint results:

.. code:: yaml

   type: executor_training
   inference_url: http://serving:30000
   train_timeout_s: 14400
   coordinator_rank: 0
   executor:
     backend: ray
     workers:
       - worker_cls: my_backend.coordinator:TrainingCoordinator
         kwargs:
           checkpoint_root: /checkpoints
         options:
           num_cpus: 1

Pass this mapping to ``RuntimeRegistry.build`` (or a recipe's runtime binding).
Python integrations may instead provide an ``ExecutorConfig`` or an already
constructed executor under ``executor``. A factory-created executor is cleaned
up if runtime initialization fails. An injected executor remains under its
caller's control on initialization failure. Service-created runtimes close once
after the dispatcher has stopped all scenario workers. A ``Dispatcher`` always
closes its recipe's runtime, including when constructed directly from Python.
To reuse external workers across dispatchers, create a fresh runtime and a
non-owning executor for each dispatcher; closing one leaves the workers alive.
Python callers that do not use a dispatcher call ``runtime.shutdown()`` when
their runtime is no longer in use.

Slime integration
-----------------

``SlimeTrainGroup`` owns role-specific training, checkpoint arguments, LoRA
activation and weight publication. Its default ``SlimeRayExecutor`` uses the
pinned Slime allocator for GPU placement, rank-zero rendezvous, memory-saver
environment and tensor-transport options. RPC goes through the shared
``RayExecutor``. Group release keeps the shared actor/critic/rollout placement
reservation intact.

The Slime driver accepts:

.. code:: bash

   --reef-executor-backend ray
   # Or a custom worker executor that understands Slime's launch contract:
   --reef-executor-backend my_backend.executors:SlimeExecutor

The custom executor receives ``args``, node/GPU counts, ``pg``, per-actor GPU
allocation, ``role``, reference/teacher flags and ``actor_cls`` in
``ExecutorConfig.options``. Its constructor launches workers; ``SlimeTrainGroup``
then calls their collective ``init`` and connects the inference control actor.
Workers return their training parallel configuration from ``set_rollout_manager``
instead of sending batch configuration to inference. The group validates the
returned layouts and passes them to the coordinator's batch processor. It must
preserve rank order and support Slime's worker methods and rollout payloads.
Critic output is passed to the actor worker at the same rank, including empty
outputs on non-final pipeline stages.

The inference control actor owns the serving ``Executor`` selected by ``execution.rollout`` or
``--reef-rollout-executor-backend``. The default ``ray`` selection maps to
``reef.runtime.sglang.executor:SGLangExecutor``. Its independent SGLang backend
owns native engine launch, routers, health monitors, update locks, memory
operations and recovery. It never imports Slime or a concrete training backend.
Custom inference executors receive ``config`` (``SGLangConfig``) and ``pg``
(borrowed Ray reservations), replacing the old Slime ``args`` payload.
The training coordinator runs external-batch packing and DP scheduling locally
through ``TrainingBatchProcessor``; there is no batch-manager Ray actor or
inference RPC relay. NIXL tensor transport is enabled on the training coordinator
when selected. This removes one Ray actor and its CPU reservation.

Slime's training coordinator, inference control actor and weight transport use Ray.
Alternative rollout executors must support the existing Slime weight-transport
contract (including engine/lock handles); selecting a backend does not rewrite
that data plane. There is no built-in torchrun, Slurm or Kubernetes backend,
and no online TP/PP resizing. ``uni/mp`` are rejected for Slime GPU worker roles;
it remains valid for standalone inference and recipe-owned external processes.

``ReefRayTrainGroup`` remains an import alias for ``SlimeTrainGroup``. Its
``async_*`` methods now return executor futures (or an already available value)
and should be consumed with ``resolve``; code using ``ray.get`` directly on
these return values must migrate. The existing Slime algorithm ``resolve``
callback already handles this conversion.

Harness evolution: component-driven selection
---------------------------------------------

Harness evolution evaluates external model endpoints; that does not require
a local GPU. ``EpisodeScorer.execution_requirements()`` declares local worker
needs using ``ExecutionRequirements``. Its default is CPU-only. The recipe
supplies worker count/resources from ``execution.evolution``; the selector
chooses ``uni`` for one local worker and ``mp`` for multiple local workers.
An existing multi-worker Ray placement context or declared cluster/actor
options selects ``ray``. Installed GPUs or ``RAY_ADDRESS`` alone do not select
Ray. Like vLLM's local-topology default, insufficient local CUDA capacity is
an error, not an implicit switch to Ray: select ``ray`` explicitly for a cluster.

.. code:: yaml

   execution:
     evolution:
       workers: 8
   evolution:
     # Existing adapter/propose/evaluate/tasks/models/seed remain unchanged.
     executor: sandbox      # independent isolation policy; requires bubblewrap

``execution.evolution`` accepts a backend, named profile, or inline object.
Its backend defaults to ``auto``; omitted resources preserve component defaults
(normally one CPU and zero GPUs). ``workers`` is the fixed worker-group size,
not an episode count or a separate business-level concurrency limit. For
``uni/mp``, CPU requests describe capacity but do not reserve cores or
enforce a CPU quota. Ray maps them to ``num_cpus``/``num_gpus`` per actor;
these are scheduling reservations, not hard CPU limits.

Legacy ``evolution.episode_workers`` and ``worker_resources.num_gpus`` remain
deprecated aliases. Conflicting old/new resource values are rejected.
``evolution.worker_executor`` still overrides a backend-only role selector,
but cannot be combined with role-level workers/resources; migrate it into
``execution.evolution.backend``. Legacy Python constructor arguments remain
accepted; prefer ``worker_executor=ExecutorSettings(workers=..., resources=WorkerResources(...))``.
``evolution.executor`` still means ``local`` or ``sandbox`` episode isolation;
it is NOT renamed to ``uni``/``mp``. Sandbox preflight runs on the execution
node too. A custom episode executor must preserve its own isolation and
owner-loss cleanup. Built-in local episodes in remote/process workers use a
POSIX owner lease so loss of a worker also kills the episode process group;
commands must not detach. Worker loss can leave temporary episode directories.

To retain an in-process callback, choose ``uni`` with exactly one worker.
The former multi-worker shared-memory backend is no longer available.
With ``mp`` the scorer is copied per
rank: mutable scorer state is not shared or merged back into the recipe.
The parent retains proposal, composition, selection and publication ownership;
only episode execution/scoring crosses the worker boundary. Candidate/current
results retain pairing order. Each backend/scenario lazily starts one fixed
pool of ``execution.evolution.workers`` ranks on its first nonempty evaluation and reuses
it across evaluations. Workers and lazy scorer/model initialization persist;
each episode still gets a fresh harness subprocess and temporary directory.
Upstream step records and per-episode stage paths are retained. Uni/MP
workers write records on the same host; Ray/custom workers return a compressed
record with each successful RPC so the driver owns the durable path, without
requiring shared storage. Archives are extracted with Python's ``data`` filter
and cannot replace an existing record directory. A failed RPC or lost remote
worker may leave no returned trajectory; this is not a durable artifact-transfer
protocol, and large records add RPC memory/transfer overhead.
RPC is ordered within each rank. A scorer error drains the submitted batch
before the next evaluation, while worker death terminally fails the pool.
Failed pools are not automatically rebuilt and evaluations are never replayed.

Trainer/scenario close (including scenario replacement) closes the pool.
Direct Python callers must call ``backend.close()`` in ``finally``; garbage
collection is only a fallback. Ray GPU reservations remain held until close,
even between evaluations. This is worker reuse, not elastic resizing or
scale-to-zero; no YAML change is needed to enable it.
Python launchers must use the usual ``if __name__ == "__main__":`` guard.

A GPU scorer can override ``execution_requirements()`` to return
``ExecutionRequirements(gpus_per_worker=1)``; ``auto`` selects ``uni/mp`` when
local CUDA capacity is sufficient. Initialize the scorer's GPU model lazily
inside the worker, not in its constructor in the recipe process. For a
plain evaluator function without that interface, declare its needs explicitly:

.. code:: yaml

   execution:
     evolution:
       workers: 2
       resources:
         cpus_per_worker: 2
         gpus_per_worker: 1  # per worker: two local workers need two visible GPUs

This describes worker/scorer GPUs, not the external inference server's GPUs.
Local GPU allocations require whole GPUs. MP assigns disjoint CUDA masks
within its worker group; uni leaves the caller's visibility unchanged. These
are not exclusive reservations against other programs on the host. Choose
``backend: ray`` for cluster scheduling or fractional GPU reservations.
Sandbox device access is unchanged; declaring a GPU for the scorer does not
expose it inside a sandboxed harness. CPU workers don't reserve GPUs, but the
selector is not a hardware-access security boundary. Explicit settings cannot
reduce a scorer's declared GPU needs; ``uni/mp`` cannot fulfill cluster
reservations. Ray nodes need the same modules, binaries, models and
reachable model endpoints. Arbitrary Python/commands are not inspected to
guess GPU needs. Slime training/rollout still require specialized Ray launchers.

Whole-stack deployment configuration
------------------------------------

Version 2 has no public process list: Reef assembles native inference/training
and HTTP processes using the same Executor factory as the model workers.
Method services are deployed independently; recipes consume their endpoints. The explicit ``services`` and
``execution.services`` examples below apply only to unversioned legacy stacks. The orchestrator only handles dependencies, readiness, endpoint
publication, log tailing, failure detection and reverse-order shutdown. It
does not contain local process or Ray placement operations.

Omitted selectors (or YAML ``null``) use ``auto``. Existing YAML without
resource reservations or a Ray placement-group context continues to launch
services locally and Slime workers through Ray. No ``execution.services``
setting is needed for those defaults. Local-controller training stacks declare
the training/rollout roles so ``reef serve`` can prepare their shared runtime
before launching the driver (see below).

Selection happens before the Executor factory imports the chosen class.
Service startup and the Slime driver log the selected backend and reason.
Explicit backend/profile selections take precedence over automatic decisions.
For ``auto``, the policy is:

* Services with ``cuda`` or declared ``env.CUDA_VISIBLE_DEVICES`` use ``uni``.
  Combining that visibility pin with resource/worker options is an error.
* Services with nonempty ``resources`` or executor ``options`` use ``ray``.
* Otherwise, services already running inside a Ray placement group use ``ray``;
  remaining services use ``uni``. This selects the backend, not a new placement
  strategy; Ray's normal placement/capture rules and explicit worker options apply.
* Slime training and rollout use their specialized Ray executors, even with one
  GPU. Reef has no built-in ``mp``/``uni`` Slime GPU launcher yet.

Installing Ray, setting ``RAY_ADDRESS``, or initializing Ray outside a placement
group does not by itself change a service to Ray. The selector does not import
or initialize Ray to probe it, count GPUs, change TP/PP, or retry a failed backend
using another backend. Resource requests must describe the intended scheduling.
Standalone SGLang commands still control their own model parallelism. ``auto``
is also the default ``ExecutorConfig.backend`` for the low-level worker factory
and coordinator runtime. A service controller is not a model rank: Slime's
specialized Ray requirement and service resource reservations remain
component-specific, rather than pretending to implement vLLM's TP/PP launcher.

Each selector accepts a built-in name, import path, inline ``backend/options``
object, or a named profile under ``executors``. ``services[].executor``
overrides ``execution.services`` for that service. Explicit Slime command-line
flags override the corresponding role selection; they also accept profile
names. Profiles for training and rollout carry domain-launcher options, not
necessarily Ray actor options. The reserved Slime launch arguments (``args``,
``pg``, rank/GPU layout, role) come from the model configuration and cannot be
overridden by profile options.

OpenClawRL's PRM and user model are outside this lifecycle. Their deployment
commands, readiness checks and resource allocation live in the example's Docker
Compose file. Reef receives only recipe client parameters such as
``recipe.config.prm-url``; no auxiliary process definitions are added to its stack.

Ray executors share a process-wide runtime, initialized only when needed.
Without an external address, Reef starts a local cluster using the deployment's
visible GPUs. Set ``RAY_ADDRESS`` (or the deployment's ``reef.ray_address``) to
connect to an external cluster; connection errors never fall back to local.
An already initialized Ray connection is borrowed and left intact. Otherwise,
the last owner disconnects; only a local cluster started by Reef is stopped.
Attached/serialized executor handles borrow their owner's runtime.

An explicitly declared ``execution.training`` or ``execution.rollout`` role
that resolves to ``ray`` also requests this runtime, even when all service
controllers run locally. It does not reserve GPUs for the Slime driver;
Slime's model topology still determines its GPU placement. With an external
address and only local controllers, Reef passes the address through and the
drivers connect after their readiness dependencies are satisfied. Stacks
without Ray services or declared Ray training/rollout roles do not start Ray.

Slime inference ownership
~~~~~~~~~~~~~~~~~~~~~~~~~

``reef.service.training_driver`` owns model deployment lifecycle through
``ModelDeployment``. The selected training definition returns a
``ModelDeploymentPlan`` containing unstarted resources, inference and training
components. The owner checks their declared connection protocols before
allocation, then starts resources, inference and training in order. It checks
inference readiness before attachment and again after training initialization;
the readiness file is published only when both components are ready.

The Slime integration supplies ``SlimeDeploymentResources`` and
``SlimeTrainingService``. Inference is supplied by the independent
``reef.runtime.sglang.service.SGLangInferenceService``. Argument parsing and
preflight live in ``reef.train.slime_backend.driver``. Its ``inference_config``
adapter translates training requirements into a serializable ``SGLangConfig``;
no Slime namespace or Megatron object is passed to inference. The service layer resolves
the recipe and passes its declared loss family into that integration, keeping
recipe discovery out of the training package.

For managed full-weight and LoRA training, including colocated deployments,
resources connect one Ray client job
and use Slime's existing placement helper once for coordinated model allocation.
Inference borrows those reservations and owns ``SGLangControl`` as a
separate Ray control actor. It reserves one CPU and zero model GPUs; model
placement groups account for engine GPUs separately. Training receives the
existing inference connection and reservations and calls ``start_bridge`` with
both. The coordinator retains tensorization, DP partitions and micro-batch
scheduling in a local processor. Native training workers receive the inference
actor handle directly. Training never creates or closes inference or reservations.

The owner closes training, inference and resources in reverse dependency order,
including components whose startup failed partway through. Cleanup failures do
not skip later components, and the original startup error remains authoritative.
Shared reservations are released once. Disconnecting the owned Ray client job
leaves an external cluster running; an already initialized client session is
rejected without disconnecting it.

The minimal shared ``InferenceConnection`` contains a borrowed executor and a
versioned control-protocol identifier. Engine handles and Ray placement
representations stay inside the concrete inference and training adapters. Weight transfer remains directly
between training workers and engines. This protocol identifies the supported
attachment vocabulary; it is not a general engine capability negotiation API.

All Slime entrypoints use the same resource, inference and training ownership,
including external-engine mode. External engines remain borrowed and are not
stopped by Reef. ``reef.service.slime_driver`` retains legacy argument-file and
healthcheck syntax but uses this same lifecycle. Direct ``start_bridge`` calls
require supplied inference and placement groups; the old self-allocating
combined path has been removed.

Managed configurations use ``inference.num-gpus``,
``inference.tensor-parallel-size`` and ``inference.options`` in all modes.
Checkpoint production and backend-specific startup recovery remain in the
Slime adapters. Managed process recovery is described below. Independent
replacement that preserves surviving components and validation of additional
backend combinations remain in
`RFC #425 <https://github.com/Human-Agent-Society/reef/issues/425>`__.

Inference recovery and reconnect
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``reef.runtime.inference_control.InferenceControl`` owns pause intent and
recovery/reconnect ordering through three backend contracts: ``InferenceEngines``
for engine operations, ``WeightUpdateConnection`` for transport-lock inspection
and replacement, and ``InferenceMonitor`` for background recovery. The Slime
serving worker supplies these adapters; engine handles, GPU topology and Ray
fan-out remain private to it. The existing training-side RPC vocabulary and
six-field engine/lock attachment tuple are unchanged.

Pause intent is recorded before the engine barrier, so a partial pause failure
still fences replacement engines on retry. Recovery stops monitoring, checks the
update lock, replaces an uncertain managed lock and marks worker reconnect as
required. It then recovers dead engines and reapplies any pending generation
pause. The reconnect flag is cleared only by the existing acknowledgement after
training workers reconnect. Healthy-engine and initial-connect behavior remains
in the concrete engine adapter.

A paused publication keeps monitoring paused after recovery. Recovery failures
also retain pause intent; neither the recovery path nor the legacy monitor-resume
RPC can restart background recovery before generation is allowed to resume.
The training publication coordinator retains the durable commit gate. An
external deployment with an uncertain lock requires operator restart, and the
shared controller never terminates borrowed engines.

``reef.runtime.weight_update.WeightUpdateLock`` owns lock poisoning and transport
phase-result bookkeeping without Slime or Ray imports. The legacy
``ReefRolloutLock`` entrypoint wraps it as a serial Ray actor. Existing weight
updaters use the same lock methods and continue transferring directly between
workers and engines. There is no tensor relay through the shared controller.

This extraction does not make the Slime launch helper or attachment tuple a
universal inference API. Backend-neutral engine launch and real GPU combinations
remain separate work. A failed managed controller uses deployment reconstruction
as described below.

Standalone serving republication
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``TrainingPublication.republish(runtime_load_id)`` restores unchanged weights to
replaced inference engines through the same commit gate. The Slime bridge's
internal ``republish_serving`` delegates to it. Reef reasserts the pause barrier,
recovers engines and update connections, and requests a complete transfer under
the original runtime load ID. A cached bridge pause never substitutes for the
controller barrier. Known missing engine slots are skipped while pausing the
surviving engines; recovery reapplies the pause to replacements.

``READY_TO_COMMIT`` stays paused and awaits the scenario's acknowledgement.
``HEAD_COMMITTED`` completes its durable acknowledgement before resumption;
``COMPLETE`` and deployments without a training marker resume only after the
transferred identity is verified. Resumption releases both generation and
health monitoring. Failed recovery, transfer, identity verification or resume
aborts serving; retry retains the original runtime load ID.

``RUNNING``, ``CHECKPOINT``, ``UPDATING_WEIGHTS``, ``REJECTING`` and ``REJECTED``
markers reject standalone republication before model operations. The trainer
may contain a candidate rather than the incumbent, so those states must use
their existing training-job recovery path. Republication never runs an optimizer
step, increments training counters or creates a new publication version.
Slime retains tensor transfer and restoration of other scenarios' adapters.
This is in-process engine recovery. Controller-process failure uses the managed
deployment recovery path below; GPU validation remains separate work.

Training-coordinator restart attachment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The attachment protocol retains the wire identifier ``slime-sglang-control-v2``
for compatibility; its inference implementation is now independent of Slime. Its additional
``prepare_training_connection`` RPC is called by ``start_bridge`` before creating
training workers when inference and reservations are supplied by the deployment
owner. It records pause intent, drains monitoring, recovers engines/connections
and requires worker attachment even when every engine and the update lock are
healthy. The existing attachment tuple carries that requirement through its
new-engine count; acknowledgement clears it only after workers have connected.

The owner must stop the previous training workers before attaching their
replacement. This handshake does not elect a leader or authorize two trainers
to write to one inference deployment. It reuses the supplied executor and
reservations; startup failure leaves borrowed resources with their owner.
Custom Slime serving executors must implement the additional RPC.

``TrainingPublication.recovery(marker)`` fences the entire backend startup
restoration, including committed and marker-free startup. The bridge restores
checkpoint identity and adapters inside that scope, with updater-controlled
generation resumption disabled. ``finish_recovery`` preserves the commit gate:
uncommitted candidates stay paused; committed identities resume generation and
monitoring only after verification. Version discovery, checkpoint seeding and
other reconstruction failures abort recovery, including failures before tensor
transfer starts. Persisted marker formats do not change.

Real Ray CPU tests replace the training coordinator process while preserving
inference, restore checkpoint values and version identity, retain pending commit
barriers and keep serving paused after a damaged-checkpoint restart. These tests
use CPU backend fixtures; they do not validate GPU checkpoint loading.

Colocated memory and LoRA recovery
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Colocated inference and training share the GPU reservations produced by one
placement call; separating their control actors does not duplicate GPUs.
Before training workers initialize, the inference owner fences generation,
performs optional weight checks and releases inference memory. The borrowed
training coordinator does not repeat this preparation.

Cold startup releases weights, KV cache and CUDA graphs even when
``training.options.keep-lora-base-resident`` is enabled. Once training has
initialized and offloaded, recovery restores the frozen inference base before
registering each scenario's committed adapter. It restores the pending
scenario's publication last. A ``READY_TO_COMMIT`` checkpoint stays paused
until Reef acknowledges the commit; recovery does not repeat training.

Later colocated steps pause inference and release its memory before training.
With ``keep-lora-base-resident``, only KV cache and CUDA graphs are released;
otherwise weights are released too. Publication restores weights as needed,
transfers the update and restores KV/graphs. Commit acknowledgement permits
generation to resume. Keeping the base resident still requires enough physical
memory for that base and the active training workload.

``reef.runtime.inference_memory`` tracks acknowledged release/resume operations
per engine. Repeated operations skip regions already in the requested state.
A failed operation leaves memory state uncertain and prevents reuse until that
engine is replaced. SGLang supplies the concrete memory API adapter. This also
handles engines whose native recovery already resumed weights but left KV and
graphs released.

CPU tests cover startup order, scenario adapter restoration, memory transitions
and commit gating. Actual CUDA allocation, LoRA IPC/NCCL transfers and combined
mode throughput still require validation in the supported GPU environment.

Managed deployment recovery
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The managed driver automatically supervises separate inference and training
plans for full-weight, LoRA, colocated and LoRA-plus-colocated deployments.
A plan supplies a nonblocking ``DeploymentHealth.poll()`` and a
``ModelPlanSource`` rebuilds components from the original resolved configuration.
Slime probes retain one outstanding health RPC per component. An operation that
queues behind a long weight transfer is not treated as a dead actor. Training
executor failures also notify the owner, including idle worker loss.

On component failure, Reef removes readiness, closes training and inference,
confirms process cleanup and releases the old allocation. It reruns checkpoint
preflight, allocates a fresh plan and reconstructs both components. Readiness
returns only after checkpoint restoration, publication recovery and component
health checks. The policy allows three restarts within five minutes, with
interruptible one-, two- and four-second delays. Cleanup failure, invalid
recovery state or replacement startup failure stops the driver and leaves
readiness absent. An ambiguous ``RUNNING`` optimizer step is never replayed.
Slime also refuses cold restart from ``REJECTING``/``REJECTED``: the latest
training checkpoint contains a declined candidate, not the committed serving
weights. Restore the committed checkpoint before restarting those deployments.

This is a cold rebuild of the model deployment. It recreates the inference
controller, routers, engines and training workers; it does not attach a new
controller to surviving engine handles. The external Ray cluster and HTTP
service remain running. External-engine and explicit legacy entrypoints use
the same component ownership but do not enable automatic cold-rebuild supervision.

Owned Ray jobs install a POSIX process lease before model workers initialize.
A watchdog retires a worker's process group after owner loss, including native
children, and records completion. Cleanup workers on the deployment's nodes
confirm completion before a replacement can reserve GPUs. Workers starting
after cleanup begins are rejected. Node loss or an unconfirmed guard stops
automatic recovery. Model children must remain in their parent's process group;
custom launchers that detach processes must provide equivalent owned cleanup.
The driver does not stop the cluster or clean unrelated jobs.

``connect_ray_runtime`` now discovers the current named coordinator before each
operation. Discovery and read-only liveness calls tolerate replacement within
the configured timeout. Submitted training operations are never automatically
replayed: durable reconciliation retains that decision. A request verifies
serving health before admission, while the existing publication gate retains
control of reopening admissions. Pending commits stay blocked across recovery.

When the endpoint is discovered from coordinator health, the existing inference
backend follows a replacement URL through ``InferenceBackend.reconnect``. HTTP
and native SGLang chat backends preserve headers, capture configuration and
provider payloads. Explicit gateway URLs remain fixed. Custom backends must
implement endpoint replacement to support discovery across address changes.
Already submitted requests and streams can fail during a crash and are not
replayed by this mechanism.

The real-Ray CPU recovery tests kill coordinator and controller processes while
retaining the same HTTP runtime, verify native child retirement and recover the
same checkpoint identity. CUDA/NCCL transport, GPU memory release and real
checkpoint performance still require the supported GPU environment.

Engine health monitoring
~~~~~~~~~~~~~~~~~~~~~~~~

``reef.runtime.health_monitor.EngineHealthMonitor`` owns probe scheduling.
``pause()`` disables new checks and waits for the active probe or retirement to
finish before the owner can replace, offload or terminate engines. A late probe
failure after pause begins is discarded. A drain timeout leaves checks disabled,
raises to the owner and prevents the engine operation; shutdown retains the
monitor so draining can be retried. An internal monitoring failure is reported
by deployment health checks and prevents monitoring from resuming.

The Slime adapter supplies ``EngineHealthChecks`` snapshots with one target per
logical engine, including every node that must retire together. Both health and
graceful-shutdown RPC waits are bounded. Retirement uses captured actor handles
and clears a slot only if it still contains the captured actor. A stale probe
cannot kill or erase a replacement engine. The adapter attempts every node even
when a kill fails, then reports the failure.

Existing ``training.options.rollout-health-check-interval``,
``rollout-health-check-timeout`` and ``rollout-health-check-first-wait`` timings
remain in use when ``training.options.use-fault-tolerance`` is enabled. The
shared monitor imports no Ray, Slime or model framework. Other backends supply
bounded probe/retirement targets and serialize lifecycle operations in their
owning controller.

Training-step coordination
~~~~~~~~~~~~~~~~~~~~~~~~~~

``reef.runtime.training_job.execution.TrainingExecution`` owns job identity,
retry classification and the order of training and checkpoint recording. The
Slime bridge invokes it under the same operation lock used by publication and
shutdown. Execution and ``TrainingPublication`` share ``TrainingJobState`` for
health reporting; recovery decisions always use the durable marker.

``TrainingJobBackend.prepare`` performs admission, scoring and data packing
before yielding a ``PreparedTrainingJob``. Its context holds the checkpoint
reservation through the final marker write. It may return a stale or
storage-blocked result without starting a job. Slime retains scenario/staleness
admission, teacher scoring, tensorization and DP packing in this adapter.

Reef records ``RUNNING`` before calling ``train``, then invokes
``save_checkpoint`` and verifies the checkpoint directory. Training metrics and
method telemetry are recorded together with ``CHECKPOINT`` in one durable write,
so a crash cannot leave a replayable checkpoint without its training metrics.
Slime's prepared job performs colocated offload and scenario activation, runs
the optimizer, saves the actor/critic pair and updates its retention/scenario
metadata. Neither prepared operation publishes inference weights.

Preparation failure is retryable without a new job identity. Once ``RUNNING``
is recorded, a training/save failure is ambiguous and requires operator
recovery; automatic retry must not repeat a possible optimizer step. A
checkpointed or completed job replays without preparing or training again.
Resource cleanup failure after ``CHECKPOINT`` also replays the recorded result.
Job hashing, scenario/global checkpoint indexes and persisted marker fields
remain compatible with existing deployments.

Commit-gated weight publication
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``reef.runtime.training_job.publication.TrainingPublication`` owns the shared
checkpoint-publication transaction. The Slime bridge delegates publication,
candidate rejection, commit acknowledgement and startup commit gating to it.
``WeightPublisher`` supplies concrete engine operations: pause, recover,
transfer and verify weights, resume, restore incumbent resources and abort.
Tensors continue to travel directly from training workers to inference engines.

Publication crosses the pause barrier before persisting ``UPDATING_WEIGHTS``.
A failed pause leaves ``CHECKPOINT`` retryable. A partial transfer or failed
publication record leaves ``UPDATING_WEIGHTS`` and prevents serving; retry
recovers engines and forces a full transfer. Success records ``READY_TO_COMMIT``
while requests remain paused. Reef must durably commit its head before calling
``acknowledge_training_commit``. The coordinator persists ``HEAD_COMMITTED``
before resuming and then records ``COMPLETE``. A failed resume can retry from
``HEAD_COMMITTED`` without training or transferring weights again.

Startup uses the same commit gate after restoring checkpoint tensors. Previously
published jobs must retain their runtime load ID, and an uncommitted candidate
stays paused after recovery. Rejection persists ``REJECTING`` before restoring
incumbent resources and records ``REJECTED`` only on success. Plain LoRA capacity
refusal preserves unrelated engines; failed eviction follows engine recovery.

Marker and durable JSON helpers now live in ``reef.runtime.training_job``.
The on-disk filename, checkpoint-derived location, fields and transitions are
unchanged. Slime retains checkpoint layout, optimizer execution, scenario/LoRA
restoration and tensor transport. Callers serialize the shared publication
coordinator with training and shutdown; it does not allocate a new actor or own
an independent process lifecycle.

The service stack keeps the runtime alive through service shutdown, publishes
the actual address as ``reef.ray_address`` in runtime snapshots, and supplies
``RAY_ADDRESS`` to subsequent services, including the Slime driver. There is
no need for a ``ray-head`` service or a fixed port in a managed local stack.
Existing explicitly managed heads remain supported: connect to their address
and keep the head on ``executor: uni`` with the appropriate dependencies.
Reef does not create cloud machines or install software on Ray nodes. The command,
working directory, Python environment, models and recipe modules must exist
on the selected nodes; use shared storage/images or Ray runtime environments.
For a Ray-executed Slime driver, reserve coordinator CPUs, not its model GPUs:
Slime's placement group reserves those separately, otherwise the reservations
can deadlock. Standalone SGLang ``num_gpus`` must cover its complete local TP
group. This launcher does not synthesize multi-node SGLang CLI arguments.

``resources`` contains worker options (for Ray: ``num_gpus``, ``num_cpus``,
``resources``, scheduling options, etc.). Ray owns CUDA visibility; combining
Ray execution with ``cuda`` or a ``CUDA_VISIBLE_DEVICES`` override is rejected.
Local execution continues to use ``cuda`` and does not reserve cluster GPUs.
Keep existing Slime training/rollout GPU-layout flags; they describe model
parallelism, not service-worker placement.

The readiness command runs on the execution node with the service's environment
and working directory, under a bounded timeout. ``{host}`` in ``endpoint`` is
the Ray node address, or ``127.0.0.1`` for local execution. A local service that
must be reachable from remote consumers must declare a routable
``advertise_host``. Eligible services are prepared together: executors acquire
placement and publish ``${endpoints.SERVICE_NAME}`` before their service
processes start. Each receives a private node-local config snapshot via
``REEF_CONFIG`` containing all endpoints published so far, including those of
independent peers prepared in the same batch. References to a producer gated
behind dependencies still require that producer to be prepared first.
Ordinary ``127.0.0.1`` literals elsewhere are not automatically rewritten.

Process launch and readiness waits run concurrently for independent services.
``depends_on`` remains a readiness prerequisite: a dependent starts as soon as
all of its declared dependencies are ready, without waiting for unrelated
services. Readiness deadlines are per service, measured from its start RPC,
not from the beginning of the deployment. The stack is declared ready only
when every service is ready. Services that already passed readiness are still
checked for process exits while their peers initialize. Explicit Ray-head
services can remain dependencies of Ray-executed services; their dependents'
placement is not attempted before the head is ready.

Logs are kept on the worker and tailed into ``run_dir/SERVICE.log``. Worker
metadata (host, PIDs and worker log directory) is written to
``run_dir/SERVICE.worker.json``; a remote PID must never be signalled locally.
Each service has a separate readiness marker/config directory. Resolved config
snapshots are private files and may contain credentials; protect the run
directory as deployment state. Ray process workers use a POSIX owner-lease
guard so loss of the owning actor terminates their process group. Commands
must remain foreground processes and must not detach into another session.

Failure policy is fail-stop, not automatic restart/replay. Startup failures
cancel peer readiness waits and join launch tasks before closing already-created
executors, preventing a late launch from escaping cleanup. Shutdown attempts every dependent before
its dependencies, even if an RPC fails. Borrowed external rollout engines and
shared Slime placement groups are not deleted by an individual rollout/training
executor. The Slime driver asks the bridge to release owned workers before
retiring its actor.

Custom service executors
~~~~~~~~~~~~~~~~~~~~~~~

A service executor receives one ``WorkerSpec`` for a ``ProcessWorker`` (the
Ray backend uses ``RayProcessWorker``). Its constructor arguments are the config
snapshot, a one-element service list, run directory, readiness timeout and
config path. A backend may execute this worker on another node or implement
the equivalent RPC protocol: ``describe``, ``prepare(config)``, ``start``,
``probe(name, timeout)``, ``status``, ``read_log(name, offset)``,
``request_stop(force=False)``, ``tree_alive`` and ``shutdown(grace=...)``.
``describe`` returns ``host``, ``pids`` and ``run_dir``. ``status`` maps service
names to ``None`` while running or an exit code; a missing service is not alive.

Custom rollout executors expose one control rank. Its RPC vocabulary is
``inference_url``, ``get_runtime_load_ids``, ``get_updatable_engines_and_lock``,
``pause_generation_for_update``, ``continue_generation_after_update``,
``offload``, ``onload(tags)``, ``onload_weights``, ``onload_kv``,
``terminate_updatable_engines``, ``prepare_training_connection``,
``recover_updatable_engines``,
``clear_updatable_num_new_engines``, ``health_monitoring_pause``,
``health_monitoring_resume`` and ``check_weights(action)``. Executor shutdown
owns serving resource teardown; the manager does not manipulate backend
server groups or actor handles directly.

Verification
~~~~~~~~~~~~

CPU tests cover legacy configs, custom service/rollout executors, resource
selection, dependency ordering, endpoint propagation, rollback, process-tree
cleanup, bounded probes and owner-lease loss. To run the real Ray/HTTP tests:

.. code:: bash

   REEF_TEST_RAY=1 python -m pytest tests/reef_service/test_service_executor_ray.py

These are control-plane tests, not a GPU SGLang/Megatron validation or a
multi-machine networking benchmark.

Independent SGLang backend
--------------------------

``reef/runtime/sglang/`` owns the chat/capture backend, SGLang plugin, native
engine process, router, Ray engine groups and inference lifecycle. Native
``ServerArgs`` and ``launch_server`` come directly from SGLang; Reef no longer
calls Slime's ``start_rollout_servers`` or inherits its ``SGLangEngine``.
The runtime-load-ID value type lives in ``reef.runtime.runtime_load_id``.

Install ``reef-infra[sglang]`` for Python-side inference dependencies and install
native SGLang in the selected GPU environment. This extra does not install
Slime or Megatron. The Slime training extra includes the inference dependencies
for existing training deployments. Pure inference still uses the direct native
``sglang.launch_server`` command binding in the deployment launcher.

Slime retains checkpoint I/O, optimizer steps, batch partitioning and tensor
transport. Its native argument parser and legacy group-file parser run only at
the configuration boundary. Checkpoint pull RPCs receive explicit source and
local directories from the training updater; inference does not read training
checkpoint arguments. Existing configs and plugin name ``reef`` are preserved;
custom import paths pointing into the former Slime SGLang subtree must move to
``reef.runtime.sglang``. Maintained examples use the new paths.

CPU tests cover native launch bindings, multi-node rendezvous, shared placement,
external engine validation/ownership, capture, LoRA requirements and recovery.
Actual GPU execution and backend combinations still require acceptance testing.
