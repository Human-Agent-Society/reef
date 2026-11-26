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

Failure listeners receive one ``on_executor_failure(failure)`` notification.
They must not block or call shutdown.
Borrowed Ray actors require a listener for process-local monitoring.
See the `failure contract <../../reef/execution/executor/failure.py>`__.

Subclass ``Executor`` and implement ``_init_executor``, ``rpc``,
``collective_rpc``, ``check_health`` and ``shutdown``.
Alternatively, extend ``SubmittingExecutor`` with ``_submit`` and
``_submit_all``. Report terminal failures through ``_fail``.
Keep launch-specific settings in ``ExecutorLaunch.options``.
Model collectives, tensors and checkpoint reshaping remain backend responsibilities.

Custom service and inference executors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Service executors receive ``ProcessWorker`` specifications (``RayProcessWorker``
for Ray). Preserve their lifecycle and status methods.
Inference executors expose the control methods in
`engine_control <../../reef/inference/engine_control.py>`__.
Executor shutdown owns serving teardown; callers cannot access backend actor groups.

Slime worker executors
~~~~~~~~~~~~~~~~~~~~~~

Slime uses ``SlimeRayExecutor`` for placement and shared ``RayExecutor`` RPC.
``training.options.reef-executor-backend`` can select a custom worker executor.
``uni`` and ``mp`` are refused for Slime GPU workers.

Preserve rank order, worker methods and reserved launch options.
Profile options cannot override Slime's topology or worker configuration.

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
   recipe:
     implementation: reef.recipe.harness:HarnessEvolutionRecipe
     config:
       evolution:
         executor: sandbox

Omit GPUs for endpoint-only scoring. ``workers`` fixes pool size, not rollout count.
Scorers declare ``execution_requirements()`` and load models inside workers.
Explicit resources cannot reduce declared needs.

Local allocations need whole GPUs. ``mp`` masks ranks; ``uni`` preserves
caller visibility. Neither reserves GPUs against other programs.
Use Ray for cluster or fractional-GPU scheduling.
Insufficient local capacity fails without switching backends.

Each scenario reuses its pool and scorer state; each rollout gets a fresh
subprocess and directory. Worker changes do not update the parent scorer.
Scorer errors drain work; worker death permanently fails the pool.
Close direct Python backends in ``finally`` to release reservations.
Remote RPC failures can lose trajectories; detached commands are unsupported.

Whole-stack deployment configuration
------------------------------------

``execution`` selects backends or profiles from ``executors``.
Use the `settings reference <../reference/configuration-settings.rst#executors>`__
for role-specific options and topology restrictions.

Ray starts only when selected. Connection failures never fall back locally.
Nodes need the software, files, models and routable endpoints before launch.
Reef closes only connections and clusters it owns.

Custom service executors must support readiness, log collection and bounded
shutdown. On startup failure, clean up every partial launch.
Stop dependents before dependencies and continue cleanup after errors.
Commands must remain foreground processes; Reef does not restart them automatically.
Never signal remote PIDs locally. Protect resolved config files, which can
contain credentials.

Verification
------------

CPU contracts cover RPC, placement, readiness, failures and cleanup.
Run the optional Ray service tests with:

.. code:: bash

   REEF_TEST_RAY=1 python -m pytest tests/deploy/test_service_executor_ray.py

These checks do not qualify GPU training or multi-machine networking.
