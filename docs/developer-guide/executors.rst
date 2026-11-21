Executors
=========

An executor (``reef.execution.executor.Executor``) launches workers and carries their ordered control RPC, health checks and shutdown. Reef uses the same executor factory for model workers, harness evaluation workers and service processes. Configuration selects a concrete executor; callers use the same methods.

:For: developers who select, configure or write an executor, and operators who place a stack's processes
:Before you start: `Configuration <../reference/configuration.rst>`__ for the ``execution`` and ``executors`` keys
:You will have: the built-in executors, the RPC and failure contracts, and how a whole stack is launched

Related pages: `Training components <training-components.rst>`__, `Serving runtimes <serving-runtimes.rst>`__, and `Sandbox environments <sandboxes.rst>`__ for rollout isolation.

Built-in executors
------------------

.. list-table::
   :header-rows: 1
   :widths: 14 86

   * - Name
     - Behavior
   * - ``uni``
     - One worker in the caller's process; the executor rejects more than one. RPC may use one local thread. A GPU scorer uses the caller's CUDA visibility: ``uni`` changes no visibility mask and reserves no cluster resource. For services, one ``ProcessWorker`` starts the service subprocess.
   * - ``mp``
     - Independent worker ranks launched with Python ``spawn``. Each rank has ordered RPC, constructor readiness, health checks and bounded shutdown. Worker classes, scorers and arguments must be spawn-pickleable: module-level classes and functions, not closures or objects that hold locks. Declared GPU scorer requirements give disjoint per-rank CUDA masks before the worker imports. It needs no Ray and implements no model-parallel collectives.
   * - ``ray``
     - Ray actors from ordered ``WorkerSpec`` objects. Worker options include Ray resources, placement strategies and runtime environments. New actors disable automatic task retries. ``RayExecutor.from_workers`` attaches existing handles; it borrows them unless ``owned=True``, and a borrowed shutdown closes the executor without terminating its actors.
   * - ``auto`` (default)
     - ``uni`` for one worker, ``mp`` for several local workers. An existing multi-worker Ray placement context, or explicit cluster or Ray actor options, selects ``ray``.

A Python class or the reference ``package.module:ExecutorSubclass`` selects a custom executor. The dot form ``package.module.Class`` is not accepted. CPU-only execution imports no Ray, Slime, Torch or SGLang. Only declared local GPU requirements probe CUDA capacity.

.. code:: python

   from reef.execution.executor import Executor, ExecutorLaunch, WorkerSpec, resolve

   class Counter:
       def __init__(self, rank):
           self.rank = rank
           self.value = 0

       def increment(self, amount):
           self.value += amount
           return self.rank, self.value

   if __name__ == "__main__":  # required for spawn
       executor = Executor.create(ExecutorLaunch(
           workers=tuple(WorkerSpec(Counter, args=(rank,)) for rank in range(2)),
       ))  # auto -> mp
       try:
           pending = executor.collective_rpc("increment", args=(3,), non_block=True)
           results = resolve(pending, timeout=10)  # [(0, 3), (1, 3)]
       finally:
           executor.shutdown()

RPC and lifetime contracts
--------------------------

**Calls.**

- ``collective_rpc`` dispatches to every worker before it waits, and returns results in rank order. ``rpc(rank, ...)`` addresses one rank.
- ``non_block=True`` returns an ``ExecutorFuture``. ``resolve`` waits on futures and on nested lists or tuples of them, under one shared deadline. Ordinary results remain ordinary values. Backend object references never cross this interface as futures.
- A timeout bounds the wait. It does not cancel GPU work, repeat a mutating operation or retry an RPC. Training-job recovery must read the durable checkpoint and commit state before it decides what to resume.

**Shutdown.** ``shutdown`` terminates only owned workers. A failed constructor cleans up the workers that the executor already acquired. A backend-specific launcher must track every allocation, including partially initialized workers.

**Failure.** An executor exposes a terminal ``failure`` (``ExecutorFailure``) and ``register_failure_listener(listener)``:

- An observer implements ``on_executor_failure(failure)``. The executor calls it once, also when the observer registers after the failure. A listener must not block and must not call shutdown. The executor logs an observer's exception and still reports the worker failure.
- When the executor detects a failure, outstanding waits fail promptly with ``ExecutorFailedError``. The executor then rejects new submissions and retires its owned workers. Borrowed actors stay with their original launcher.
- Normal shutdown, an ordinary worker-method or scorer error, and a wait timeout are not terminal failures.
- Reef does not recreate workers, replay requests or recover a distributed group.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Executor
     - How it detects a dead worker
   * - ``uni``
     - No detector: the worker shares the caller's process.
   * - ``mp``
     - Process sentinels, including idle ranks.
   * - ``ray``, owned
     - Background readiness probes. A queued probe on a busy actor is not a failure.
   * - ``ray``, borrowed or serialized
     - Registering a listener starts process-local monitoring. Listeners are not serialized with actor handles.
   * - custom
     - The subclass reports it with ``_fail``.

**Write an executor.** Implement ``_init_executor``, ``rpc``, ``collective_rpc``, ``check_health`` and ``shutdown`` in an ``Executor`` subclass. A transport that submits one call and gets a future back can subclass ``SubmittingExecutor`` and implement only ``_submit`` and ``_submit_all``; the bundled executors and ``DelegatingExecutor`` are built that way. Put backend-specific launch configuration in ``ExecutorLaunch.options``. GPU tensors, optimizer state, NCCL collectives and KV transfers remain the model backend's responsibility. Changing executors does not resize a live topology or reshard checkpoints.

Custom service and inference executors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Service executors.** A service executor receives one ``WorkerSpec`` for a ``ProcessWorker`` (the Ray backend uses ``RayProcessWorker``). Its constructor arguments are the config snapshot, a one-element service list, the run directory, the readiness timeout and the config path. A backend may run this worker on another node, or implement the equivalent RPC protocol:

- ``describe`` returns ``host``, ``pids`` and ``run_dir``.
- ``prepare(config)``, ``start``, ``probe(name, timeout)``, ``read_log(name, offset)``, ``request_stop(force=False)``, ``tree_alive`` and ``shutdown(grace=...)``.
- ``status`` maps service names to ``None`` while running, or to an exit code. A missing service is not alive.

**Inference executors.** A custom inference executor exposes one control rank. Its RPC vocabulary is ``inference_url``, ``get_runtime_load_ids``, ``updatable_engines``, ``load_adapter_from_disk``, ``load_weights_from_disk``, ``pause_generation_for_update``, ``continue_generation_after_update``, ``terminate_updatable_engines``, ``recover_updatable_engines`` and ``check_health``. Executor shutdown owns the teardown of serving resources. The manager does not touch backend server groups or actor handles.

Slime worker executors
~~~~~~~~~~~~~~~~~~~~~~

``SlimeTrainGroup`` launches the Slime workers through its default ``SlimeRayExecutor``. That executor uses Slime's allocator for GPU placement, rank-zero rendezvous and the memory-saver environment. RPC goes through the shared ``RayExecutor``.

``--reef-executor-backend`` in ``training.options`` selects another worker executor. Reef refuses ``uni`` and ``mp`` for Slime GPU worker roles. The executor must preserve rank order and Slime's worker methods. It receives these values in ``ExecutorLaunch.options``:

- ``args``, the parsed Slime arguments
- the node and GPU counts, and the placement group ``pg``
- the GPU allocation of each actor
- ``role``, and the reference and teacher flags
- ``actor_cls``, the worker class

Harness evolution: worker executor selection
--------------------------------------------

Harness evolution evaluates external model endpoints, so it needs no local GPU. Set the worker group under ``execution.evolution``:

.. code:: yaml

   execution:
     evolution:
       workers: 8
   evolution:
     # The other evolution keys (adapter, method, evaluation, models, seed) stay as they are.
     executor: sandbox      # independent isolation policy; requires bubblewrap

``execution.evolution`` accepts a backend, a named profile, or an inline object. In Python, pass ``worker_executor=ExecutorSettings(workers=..., resources=WorkerResources(...))``.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Key
     - Meaning
   * - ``backend``
     - Defaults to ``auto``: ``uni`` for one local worker, ``mp`` for several. An existing multi-worker Ray placement context, or declared cluster or actor options, selects ``ray``. Installed GPUs or ``RAY_ADDRESS`` alone do not. Insufficient local CUDA capacity is an error, not a switch to Ray: select ``ray`` explicitly for a cluster.
   * - ``workers``
     - The fixed size of the worker group. It is not a rollout count or a separate concurrency limit.
   * - ``resources``
     - Omitted resources keep the scorer's declared requirements: by default one CPU and zero GPUs per worker (``TaskScorer.execution_requirements()``, an ``ExecutionRequirements``). For ``uni`` and ``mp``, CPU requests describe capacity but reserve no cores. Ray maps them to ``num_cpus`` and ``num_gpus`` per actor, which are scheduling reservations, not hard limits.

``evolution.executor`` is a separate setting: it selects ``local`` or ``sandbox`` rollout isolation (`Sandbox environments <sandboxes.rst>`__). Sandbox preflight runs on the execution node too. A custom rollout executor must preserve its own isolation and owner-loss cleanup.

**The worker pool.** Each scenario starts one fixed pool of ``execution.evolution.workers`` ranks on its first nonempty evaluation and reuses it across evaluations.

- Workers and lazy scorer or model initialization persist. Each rollout still gets a fresh harness subprocess and temporary directory.
- The parent keeps the method's step, composition, selection and publication. Only rollout execution and scoring cross the worker boundary. Candidate and current results keep their pairing order.
- With ``mp``, each rank gets its own copy of the scorer. Mutable scorer state is not shared or merged back. To keep an in-process callback, choose ``uni`` with exactly one worker.
- RPC is ordered within each rank. A scorer error drains the submitted batch before the next evaluation. Worker death terminally fails the pool: Reef does not rebuild it and never replays evaluations.
- Closing the trainer or the scenario closes the pool. Direct Python callers must call ``backend.close()`` in ``finally``. Ray GPU reservations stay held until close, also between evaluations. This is worker reuse, not elastic resizing.
- Python launchers must use the ``if __name__ == "__main__":`` guard.

**Rollout processes and records.**

- Built-in local rollouts in remote or process workers use a POSIX owner lease, so the loss of a worker also kills the rollout process group. Commands must not detach. Worker loss can leave temporary rollout directories.
- ``uni`` and ``mp`` workers write step records on the same host. Ray and custom workers return a compressed record with each successful RPC, so the parent owns the durable path without shared storage. Reef extracts archives with Python's ``data`` filter, and an archive cannot replace an existing record directory.
- A failed RPC or a lost remote worker may leave no returned trajectory. Large records add RPC memory and transfer overhead.

**GPU scorers.** A scorer overrides ``execution_requirements()`` to return ``ExecutionRequirements(gpus_per_worker=1)``. Initialize its GPU model lazily inside the worker, not in its constructor. For a plain evaluator function, declare the needs in the config:

.. code:: yaml

   execution:
     evolution:
       workers: 2
       resources:
         cpus_per_worker: 2
         gpus_per_worker: 1  # per worker: two local workers need two visible GPUs

- These are the scorer's GPUs, not the inference server's.
- Local GPU allocations need whole GPUs. ``mp`` assigns disjoint CUDA masks within its group, and ``uni`` leaves the caller's visibility unchanged. Neither is an exclusive reservation against other programs on the host.
- Choose ``backend: ray`` for cluster scheduling or fractional GPUs. Ray nodes need the same modules, binaries, models and reachable model endpoints.
- Explicit settings cannot reduce a scorer's declared GPU needs. Reef does not inspect code or commands to guess them.
- Declaring a GPU for the scorer does not expose it inside a sandboxed harness. The selector is not a security boundary for hardware access.

Whole-stack deployment configuration
------------------------------------

A configuration file has no process list. Reef assembles the inference, training and HTTP processes with the same executor factory as the model workers. The orchestrator handles dependencies, readiness, endpoint publication, log tailing, failure detection and reverse-order shutdown. It contains no local process or Ray placement code.

Method services such as a PRM run on their own, and recipes consume their endpoints (for example ``recipe.config.prm-url``). The one auxiliary child that Reef assembles is the generator service of a ``generator`` section, under the ``generator`` executor role.

**Selecting an executor.** Each selector accepts a built-in name, a ``package.module:ClassName`` reference, an inline ``backend``/``options`` object, or a named profile under ``executors``. An explicit selection takes precedence. An omitted selector, or YAML ``null``, uses ``auto``:

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Service
     - ``auto`` selects
   * - Has ``cuda`` or a declared ``env.CUDA_VISIBLE_DEVICES``
     - ``uni``. Combining this pin with resource or worker options is an error.
   * - Has nonempty ``resources`` or executor ``options``
     - ``ray``
   * - Already runs inside a Ray placement group
     - ``ray``
   * - Any other service
     - ``uni``
   * - Slime training
     - Its specialized Ray executor, even with one GPU.

Selection happens before the factory imports the chosen class, and service startup logs the selected backend and the reason. The selector imports and initializes no Ray, counts no GPUs, and never retries a failed backend with another one. Installing Ray, setting ``RAY_ADDRESS``, or initializing Ray outside a placement group does not change a service to Ray.

Explicit Slime command-line flags override the corresponding role selection, and accept profile names. The reserved Slime launch arguments (``args``, ``pg``, the rank and GPU layout, the role) come from the model configuration; profile options cannot override them.

**Resources.** ``resources`` holds worker options: for Ray, ``num_gpus``, ``num_cpus``, ``resources`` and scheduling options. Ray owns CUDA visibility, so Reef rejects Ray execution together with ``cuda`` or a ``CUDA_VISIBLE_DEVICES`` override. Local execution uses ``cuda`` and reserves no cluster GPUs. Slime's GPU-layout flags describe model parallelism, not service placement. A standalone SGLang's ``num_gpus`` must cover its complete local TP group; Reef synthesizes no multi-node SGLang arguments.

**The Ray runtime.** Ray executors share one process-wide runtime, initialized only when needed:

- Without an external address, Reef starts a local cluster on the deployment's visible GPUs. A managed local stack needs no ``ray-head`` service or fixed port.
- ``RAY_ADDRESS``, or the deployment's ``reef.ray_address``, connects to an external cluster. A connection error never falls back to local.
- Reef borrows a Ray connection that is already initialized and leaves it intact. Otherwise the last owner disconnects. Reef stops only a local cluster that it started.
- A declared ``execution.training`` or ``execution.rollout`` role that resolves to ``ray`` also requests the runtime, even when all service controllers run locally. It reserves no GPUs. A stack without Ray services or declared Ray roles starts no Ray.
- A training stack that runs Slime starts Ray before the service that hosts the training component.
- The stack keeps the runtime alive through service shutdown. It publishes the address as ``reef.ray_address`` in runtime snapshots and as ``RAY_ADDRESS`` to later services.
- An explicitly managed head remains supported: connect to its address and keep the head on ``executor: uni``. Reef does not place a head's dependents before the head is ready.

Reef creates no machines and installs no software on Ray nodes. The command, working directory, Python environment, models and recipe modules must exist on the selected nodes.

**Endpoints and readiness.**

- The readiness command runs on the execution node with the service's environment and working directory, under a bounded timeout.
- ``{host}`` in ``endpoint`` is the Ray node address, or ``127.0.0.1`` for local execution. A local service that remote consumers must reach declares a routable ``advertise_host``. Reef does not rewrite other ``127.0.0.1`` literals.
- Reef prepares eligible services together: executors acquire placement and publish ``${endpoints.SERVICE_NAME}`` before their processes start. Each service receives a private node-local config snapshot through ``REEF_CONFIG``, with every endpoint published so far. A reference to a producer behind dependencies still needs that producer prepared first.
- Independent services launch and wait for readiness concurrently. A dependent starts as soon as all of its ``depends_on`` services are ready. Each service has its own readiness deadline, measured from its start RPC. The stack is ready when every service is ready. Reef keeps checking ready services for process exits while their peers start.

**Logs and run directory.** Logs stay on the worker and are tailed into ``run_dir/SERVICE.log``. Worker metadata (host, PIDs and worker log directory) goes to ``run_dir/SERVICE.worker.json``; never signal a remote PID locally. Each service has its own config directory. Resolved config snapshots are private files and may contain credentials, so protect the run directory as deployment state.

**Failure and shutdown.** The policy is fail-stop, with no automatic restart or replay:

- A startup failure cancels the peers' readiness waits and joins the launch tasks before Reef closes the executors that it created. A late launch therefore cannot escape cleanup.
- Shutdown attempts every dependent before its dependencies, even if an RPC fails. An executor does not delete shared Slime placement groups.
- Ray process workers use a POSIX owner-lease guard, so the loss of the owning actor terminates their process group. Commands must stay foreground processes and must not detach into another session.
- The HTTP service closes its training component and inference runtime after the dispatcher stops every scenario worker.

Verification
------------

CPU tests cover custom service and inference executors, resource selection, dependency ordering, endpoint propagation, rollback, process-tree cleanup, bounded probes and owner-lease loss. To run the real Ray and HTTP tests:

.. code:: bash

   REEF_TEST_RAY=1 python -m pytest tests/reef_service/deploy/test_service_executor_ray.py

These are control-plane tests, not a GPU SGLang or Megatron validation, and not a multi-machine networking benchmark.
