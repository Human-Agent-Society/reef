Executors
=========

An ``Executor`` launches workers and provides ordered RPC, health checks
and shutdown. Use it for worker placement; use
`sandbox environments <sandboxes.rst>`__ for rollout isolation.

Built-in executors
------------------

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Backend
     - Behavior
   * - ``uni``
     - One worker in the caller's process. No resource reservation or CUDA-mask changes.
   * - ``mp``
     - Python ``spawn`` processes with ordered RPC and bounded shutdown.
       Classes and arguments must be pickleable; declared GPU needs receive disjoint rank masks.
   * - ``ray``
     - Ray actors with resource and placement options. Automatic task retries are disabled.
       ``from_workers`` borrows actor handles unless ``owned=True``.
   * - ``auto``
     - ``uni`` for one local worker, ``mp`` for several. Cluster options or an existing
       multi-worker Ray placement context select ``ray``.

Custom executors use a class or ``package.module:ExecutorSubclass`` reference.
Dot-form class references are unsupported. CPU execution imports no GPU runtime.

.. code:: python

   from reef.execution.executor import Executor, ExecutorLaunch, WorkerSpec, resolve


   class Counter:
       def __init__(self, rank):
           self.rank = rank
           self.value = 0

       def increment(self, amount):
           self.value += amount
           return self.rank, self.value


   if __name__ == "__main__":
       executor = Executor.create(ExecutorLaunch(
           workers=tuple(WorkerSpec(Counter, args=(rank,)) for rank in range(2)),
       ))
       try:
           pending = executor.collective_rpc("increment", args=(3,), non_block=True)
           print(resolve(pending, timeout=10))  # [(0, 3), (1, 3)]
       finally:
           executor.shutdown()

The main guard is required for spawn.

RPC and lifetime contracts
--------------------------

``collective_rpc`` submits to every rank before waiting and returns results
in rank order. ``rpc(rank, ...)`` addresses one worker.
``non_block=True`` returns ``ExecutorFuture``; ``resolve`` handles futures
and nested lists or tuples under one deadline.
Backend object references must not escape as futures.

Timeouts bound waiting. They do not cancel work or retry operations.
Job recovery must inspect durable state before resuming.
Shutdown terminates owned workers only. Failed construction must clean up
all partial allocations.

Terminal worker failure sets ``failure`` and fails outstanding waits with
``ExecutorFailedError``. New submissions are rejected and owned workers retire.
Reef does not rebuild groups or replay requests.
Method errors, ordinary timeouts and normal shutdown are not terminal failures.

Failure listeners implement ``on_executor_failure(failure)`` and receive one
notification, including when registered after failure.
They must not block or call shutdown. Listener exceptions are logged.
Multiprocessing monitors process sentinels; Ray monitors actor readiness.
A busy actor's queued probe does not establish failure.
Borrowed Ray actors need a registered listener for process-local monitoring;
listeners are not serialized.

Subclass ``Executor`` and implement ``_init_executor``, ``rpc``,
``collective_rpc``, ``check_health`` and ``shutdown``.
Alternatively, extend ``SubmittingExecutor`` with ``_submit`` and
``_submit_all``. Report terminal failures through ``_fail``.
Keep launch-specific settings in ``ExecutorLaunch.options``.
Model collectives, tensors and checkpoint reshaping remain backend responsibilities.

Custom service and inference executors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Service executors receive a ``ProcessWorker`` specification, or
``RayProcessWorker`` for Ray. Preserve its control methods:
``describe``, ``prepare``, ``start``, ``probe``, ``read_log``,
``request_stop``, ``tree_alive``, ``status`` and ``shutdown``.
``describe`` returns host, PIDs and run directory.
``status`` maps services to ``None`` while running or an exit code afterward;
a missing service is not alive.

Inference executors expose one control rank with URL, load-ID, engine update,
pause/resume, recovery and health operations.
Follow the control methods in ``reef.inference.engine_control``. Executor
shutdown owns serving teardown; callers must not reach into backend actor groups.

Slime worker executors
~~~~~~~~~~~~~~~~~~~~~~

Slime uses ``SlimeRayExecutor`` for placement and shared ``RayExecutor`` RPC.
``training.options.reef-executor-backend`` can select a custom worker executor.
``uni`` and ``mp`` are refused for Slime GPU workers.

Preserve rank order and worker methods. Launch options carry parsed Slime
arguments, node/GPU layout, placement group, role, reference/teacher flags
and actor class. Profile options cannot override these reserved values.

Harness evolution: worker executor selection
--------------------------------------------

Configure evaluation workers separately from rollout isolation:

.. code:: yaml

   execution:
     evolution:
       workers: 2
       resources:
         cpus_per_worker: 2
         gpus_per_worker: 1
   evolution:
     executor: sandbox

Omit GPU resources for endpoint-only scoring.
``workers`` fixes pool size; it is not a rollout limit.
Scorers declare requirements through ``execution_requirements()`` and load
models lazily inside workers. Explicit resources cannot lower declared needs.

Local GPU allocations require whole GPUs. ``mp`` masks ranks within its group;
``uni`` preserves caller visibility. Neither reserves GPUs against other programs.
Select Ray for cluster or fractional-GPU scheduling.
Installed GPUs and ``RAY_ADDRESS`` alone do not select it.
Insufficient local GPU capacity fails rather than switching backends.

Each scenario lazily creates and reuses one pool. Workers keep scorer state,
but each rollout gets a fresh subprocess and temporary directory.
Multiprocess scorer changes are not merged into the parent.
Method logic and publication remain in the parent process.

Scorer errors drain submitted work; worker death fails the pool permanently.
Close direct Python backends in ``finally``. GPU reservations remain until close.

Remote workers return compressed run records without requiring shared storage.
Failed RPCs can lose their trajectories. Local rollout process groups use
owner leases, but detached commands escape that cleanup and are unsupported.

Whole-stack deployment configuration
------------------------------------

``execution`` selectors accept backend names, import references, inline
options or profiles from ``executors``. Explicit selections override ``auto``.
Services with CUDA pins run locally; resource options or Ray placement
contexts select Ray. Other service controllers default to ``uni``.
Startup logs the selected backend and reason.

Role constraints:

- SGLang and vLLM rollout engines require their Ray executors.
- Training and rollout selectors cannot set ``workers`` or ``resources``;
  model configuration controls topology.
- Generator execution accepts one worker.
- Slime training accepts ``auto`` or Ray selections.
  In-process backends such as Tinker reject ``execution.training``.
- Ray execution cannot combine with explicit CUDA visibility overrides.

Ray starts only when selected. Reef can start a local cluster or connect
through ``reef.ray_address`` or ``RAY_ADDRESS``.
Connection failures do not fall back locally. Borrowed connections remain
open; Reef stops only clusters it owns.
Nodes must already contain the required software, files and models.

Readiness runs on each execution node. Independent services start concurrently;
dependents wait for their prerequisites. Remote consumers need routable
advertised addresses. Endpoint publication does not rewrite arbitrary localhost URLs.

Logs are tailed into ``run_dir/SERVICE.log``; worker metadata goes to
``SERVICE.worker.json``. Never signal remote PIDs locally.
Protect resolved configuration files because they can contain credentials.

Startup failure cancels readiness waits and cleans up all launched workers.
Shutdown stops dependents before dependencies and attempts cleanup after errors.
Commands must remain foreground processes. Reef does not automatically restart
services or replay their operations.

Verification
------------

CPU contracts cover RPC, placement, readiness, failures and cleanup.
Run the optional Ray service tests with:

.. code:: bash

   REEF_TEST_RAY=1 python -m pytest tests/deploy/test_service_executor_ray.py

These checks do not qualify GPU training or multi-machine networking.
