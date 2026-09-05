Worker executors
================

Reef separates service orchestration and model semantics from worker execution. The runtime owns
candidate checkpoints, serving admission, activation and commit acknowledgement.
An ``Executor`` owns worker launch, ordered control RPC, health and shutdown.
The interface follows vLLM's executor pattern: configuration selects a concrete
class, while callers use the same methods for each backend.

.. code:: mermaid

   flowchart TD
       D[reef serve / dependency graph] --> SE[Service Executor]
       SE --> LW[Local ProcessWorker]
       SE --> RW[Ray ProcessWorker]
       SE --> CW[Custom executor]
       LW --> SV[SGLang / PRM / user LLM / Slime driver / Reef]
       RW --> SV
       R[ExecutorTrainingRuntime] --> H[TrainingGroupHandle]
       H --> C[Coordinator Executor]
       C --> B[Training coordinator / Slime bridge]
       B --> G[SlimeTrainGroup: train, checkpoint, publish]
       G --> E[Worker Executor]
       E --> S[SlimeRayExecutor]
       E --> P[Custom Slime-compatible Executor]
       S --> W[Megatron workers]
       B --> RM[ReefRolloutManager: batches / DP partitioning]
       RM --> RE[Rollout Executor]
       RE --> SR[SlimeRayRolloutExecutor: SGLang engines / routers / update lock]
       RE --> CR[Custom rollout executor]

The coordinator executor targets one worker for each training-job RPC. The
training executor dispatches rank operations to the model workers. This keeps
checkpoint/publication transactions from being accidentally broadcast and
executed more than once.

Built-in executors
------------------

``uni``
   One ordinary CPU worker in the caller's process. RPC may use one local
   thread; this is not a GPU model launcher. More than one worker is rejected.

``mp``
   Independent CPU worker ranks launched with Python ``spawn``. Each rank has
   ordered RPC, constructor readiness, health checks and bounded shutdown.
   Worker classes, scorers and arguments must be spawn-pickleable (normally
   module-level classes/functions), not closures or objects holding locks.
   No Ray installation is needed, and this does not implement distributed
   GPU model ranks. Timeouts stop waiting, not the work, and never retry RPC.

``ray``
   Launches Ray actors from ordered ``WorkerSpec`` objects. Worker options
   include Ray resources, placement strategies and runtime environments.
   New actors disable automatic task retries. ``RayExecutor.from_workers``
   attaches existing handles; it borrows them unless ``owned=True`` is explicit.
   Borrowed shutdown closes the executor without terminating its actors.

``local``
   Runs ordinary CPU workers in one process with concurrent RPC threads. It
   supports contract tests and in-process integrations. For deployment, the
   worker is a ``ProcessWorker`` that starts the actual service as a separate
   process; that service can itself run SGLang on GPUs. LocalExecutor itself
   does not implement multi-process model ranks or GPU resource reservations.

A Python class or ``package.module:ExecutorSubclass`` / ``package.module.Class``
selects a custom executor. Importing the interface or using the local executor
does not import Ray, Slime, Torch or SGLang.

.. code:: python

   from reef.runtime.executor import Executor, ExecutorConfig, WorkerSpec, resolve

   class Counter:
       def __init__(self, rank):
           self.rank = rank
           self.value = 0

       def increment(self, amount):
           self.value += amount
           return self.rank, self.value

   executor = Executor.create(ExecutorConfig(
       backend="local",
       workers=tuple(WorkerSpec(Counter, args=(rank,)) for rank in range(2)),
   ))
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
bridge in the selected namespace. ``RayRuntime`` and ``RayTrainGroupHandle``
remain compatibility aliases for ``ExecutorTrainingRuntime`` and
``TrainingGroupHandle``.

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
after the dispatcher has stopped all scenario workers. A directly constructed
``Dispatcher`` borrows its runtime unless ``owns_runtime=True`` is explicit.
Other Python callers call ``runtime.shutdown()`` when their runtime is no
longer in use.

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
then calls their collective ``init`` and connects the rollout manager. It must
preserve rank order and support Slime's worker methods and rollout payloads.
Critic output is passed to the actor worker at the same rank, including empty
outputs on non-final pipeline stages.

The rollout manager no longer launches SGLang directly. Its serving operations
go through a separate ``Executor`` selected by ``execution.rollout`` or
``--reef-rollout-executor-backend``. The default ``ray`` selection maps to
``SlimeRayRolloutExecutor``: Slime-specific launch, placement, routers, health
monitors, update locks, offload/onload and recovery live in its backend worker.
The manager retains external-batch packing and DP scheduling.

Slime's bridge/manager actors and payload/weight transport still use Ray.
Alternative rollout executors must support the existing Slime weight-transport
contract (including engine/lock handles); selecting a backend does not rewrite
that data plane. There is no built-in torchrun, Slurm or Kubernetes backend,
and no online TP/PP resizing. ``local`` is rejected for Slime GPU worker roles;
it remains valid for launching standalone SGLang/PRM service processes.

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
supplies the worker count from ``evolution.episode_workers``; the selector
chooses ``uni`` for one CPU worker, ``mp`` for multiple local CPU workers, and
``ray`` for GPU/cluster requests. Installed GPUs or ``RAY_ADDRESS`` alone do
not select Ray or reserve GPUs for this role.

.. code:: yaml

   evolution:
     # Existing adapter/propose/evaluate/tasks/models/seed remain unchanged.
     episode_workers: 8
     worker_executor: auto   # optional; selects mp for ordinary CPU scorers
     executor: sandbox      # independent isolation policy; requires bubblewrap

``evolution.worker_executor`` overrides ``execution.evolution`` (default
``auto``) and accepts the same backend/profile/inline selection as other roles.
``evolution.executor`` still means ``local`` or ``sandbox`` episode isolation;
it is NOT renamed to ``uni``/``mp``. Sandbox preflight runs on the execution
node too. A custom episode executor must preserve its own isolation and
owner-loss cleanup. Built-in local episodes in remote/process workers use a
POSIX owner lease so loss of a worker also kills the episode process group;
commands must not detach. Worker loss can leave temporary episode directories.

To retain shared-memory callbacks, explicitly choose ``worker_executor: local``
(the previous thread-based behavior). With ``mp`` the scorer is copied per
rank: mutable scorer state is not shared or merged back into the recipe.
The parent retains proposal, composition, selection and publication ownership;
only episode execution/scoring crosses the worker boundary. Candidate/current
results retain pairing order. Each backend/scenario lazily starts one fixed
pool of ``episode_workers`` ranks on its first nonempty evaluation and reuses
it across evaluations. Workers and lazy scorer/model initialization persist;
each episode still gets a fresh harness subprocess and temporary directory.
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
``ExecutionRequirements(gpus_per_worker=1)``; ``auto`` then selects Ray without
an executor setting in YAML. Initialize the scorer's GPU model lazily inside
the allocated worker, not in its constructor in the recipe process. For a
plain evaluator function without that interface, declare its needs explicitly:

.. code:: yaml

   evolution:
     episode_workers: 2
     worker_resources:
       num_gpus: 1          # per worker: two workers reserve two GPUs

This reserves worker/scorer GPUs, not the external inference server's GPUs.
Sandbox device access is unchanged; declaring a GPU for the scorer does not
expose it inside a sandboxed harness. CPU workers don't reserve GPUs, but the
selector is not a hardware-access security boundary. Explicit settings cannot
reduce a scorer's declared GPU needs; ``uni/mp/local`` cannot fulfill GPU or
cluster reservations. Ray nodes need the same modules, binaries, models and
reachable model endpoints. Arbitrary Python/commands are not inspected to
guess GPU needs. Slime training/rollout still require specialized Ray launchers.

Whole-stack deployment configuration
------------------------------------

All services in ``services`` use the same Executor factory as the model
workers. The orchestrator only handles dependencies, readiness, endpoint
publication, log tailing, failure detection and reverse-order shutdown. It
does not contain local process or Ray placement operations.

Omitted selectors (or YAML ``null``) use ``auto``. Existing YAML without
resource reservations or a Ray placement-group context continues to launch
services locally and Slime workers through Ray:

.. code:: yaml

   execution:
     services: auto
     training: auto
     rollout: auto

Selection happens before the Executor factory imports the chosen class.
Service startup and the Slime driver log the selected backend and reason.
Explicit backend/profile selections take precedence over automatic decisions.
For ``auto``, the policy is:

* Services with ``cuda`` or declared ``env.CUDA_VISIBLE_DEVICES`` use ``local``.
  Combining that visibility pin with resource/worker options is an error.
* Services with nonempty ``resources`` or executor ``options`` use ``ray``.
* Otherwise, services already running inside a Ray placement group use ``ray``;
  remaining services use ``local``. This selects the backend, not a new placement
  strategy; Ray's normal placement/capture rules and explicit worker options apply.
* Slime training and rollout use their specialized Ray executors, even with one
  GPU. Reef has no built-in ``mp``/``uni`` Slime GPU launcher yet.

Installing Ray, setting ``RAY_ADDRESS``, or initializing Ray outside a placement
group does not by itself change a service to Ray. The selector does not import
or initialize Ray to probe it, count GPUs, change TP/PP, or retry a failed backend
using another backend. Resource requests must describe the intended scheduling.
Standalone SGLang/PRM commands still control their own model parallelism. ``auto``
is a role-aware configuration policy, not an ``ExecutorConfig.backend`` accepted
by the low-level worker factory or the coordinator runtime.

Each selector accepts a built-in name, import path, inline ``backend/options``
object, or a named profile under ``executors``. ``services[].executor``
overrides ``execution.services`` for that service. Explicit Slime command-line
flags override the corresponding role selection; they also accept profile
names. Profiles for training and rollout carry domain-launcher options, not
necessarily Ray actor options. The reserved Slime launch arguments (``args``,
``pg``, rank/GPU layout, role) come from the model configuration and cannot be
overridden by profile options.

For example, to move a standalone PRM service onto a Ray GPU worker:

.. code:: yaml

   execution:
     services: local
     training: ray
     rollout: ray

   executors:
     gpu_service:
       backend: ray
       options:
         num_cpus: 1

   reef:
     # Other existing recipe/model settings stay here.
     prm_url: ${endpoints.prm-sglang}

   services:
     - name: prm-sglang
       executor: gpu_service
       resources:
         num_gpus: 1
       command: >-
         python -m sglang.launch_server --model-path=/models/prm
         --host 0.0.0.0 --port 23001 --tp 1
       endpoint: http://{host}:23001
       ready: curl -sf http://127.0.0.1:23001/health
       ready_timeout: 600

     - name: reef
       command: [python, -m, reef.service]
       depends_on: [prm-sglang]
       ready: curl -sf http://127.0.0.1:8900/healthz

Connect the orchestrator to an existing Ray cluster using ``RAY_ADDRESS``.
A locally declared ``ray-head`` can still bootstrap it; keep that service on
``executor: local`` and declare it as a dependency of Ray services. Reef does
not create cloud machines or install software on Ray nodes. The command,
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
``advertise_host``. After startup, the endpoint is available as
``${endpoints.SERVICE_NAME}``. Consumers must depend on its producer, and receive
a node-local config snapshot with resolved endpoints via ``REEF_CONFIG``.
Ordinary ``127.0.0.1`` literals elsewhere are not automatically rewritten.

Logs are kept on the worker and tailed into ``run_dir/SERVICE.log``. Worker
metadata (host, PIDs and worker log directory) is written to
``run_dir/SERVICE.worker.json``; a remote PID must never be signalled locally.
Each service has a separate readiness marker/config directory. Resolved config
snapshots are private files and may contain credentials; protect the run
directory as deployment state. Ray process workers use a POSIX owner-lease
guard so loss of the owning actor terminates their process group. Commands
must remain foreground processes and must not detach into another session.

Failure policy is fail-stop, not automatic restart/replay. Startup failures
close already-created executors, and shutdown attempts every dependent before
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
``terminate_updatable_engines``, ``recover_updatable_engines``,
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
