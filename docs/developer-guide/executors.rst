Executors
=========

``Executor`` launches workers and provides ordered RPC, health checks and shutdown.
Use `sandbox environments <sandboxes.rst>`__ for rollout isolation.

Launch workers
--------------

- ``uni`` runs in the caller's process without resource reservation.
- ``mp`` uses spawn processes; workers and arguments must be pickleable.
- ``ray`` uses actors; automatic task retries are disabled.
- ``auto`` chooses local execution unless cluster options or placement require Ray.

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
GPU requests cannot exceed local capacity; use Ray for cluster or fractional GPUs.
Local masks do not reserve GPUs against other programs.

RPC and lifetime contracts
--------------------------

``collective_rpc`` submits every rank before waiting and returns rank-ordered results.
``rpc`` addresses one rank. ``non_block=True`` returns ``ExecutorFuture``;
``resolve`` handles nested futures under one deadline.
Timeouts do not cancel or retry work.

Shutdown terminates owned workers only. Failed construction cleans partial allocations.
Terminal worker failure rejects new work, fails waits with ``ExecutorFailedError``
and retires owned workers. No group rebuild or request replay occurs.
Ordinary method errors and timeouts are not terminal failures.

Failure listeners receive one ``on_executor_failure`` notification and must not block
or call shutdown. Borrowed Ray actors need a listener for local monitoring.

Implement a backend
-------------------

Subclass ``Executor`` with ``_init_executor``, ``rpc``, ``collective_rpc``,
``check_health`` and ``shutdown``. Alternatively, extend ``SubmittingExecutor``
with ``_submit`` and ``_submit_all``. Report terminal failure through ``_fail``.
Select custom classes as ``package.module:ExecutorSubclass``.
See the `executor contract <../../reef/execution/executor/base.py>`__.

Service workers must preserve ``ProcessWorker`` lifecycle, readiness and status methods.
Inference workers preserve `engine-control methods <../../reef/inference/engine_control.py>`__;
executor shutdown owns serving teardown.
Never signal remote PIDs locally. Commands must remain foreground processes.

Slime worker executors
~~~~~~~~~~~~~~~~~~~~~~

Slime requires Ray GPU workers. Custom executors selected through
``training.options.reef-executor-backend`` must preserve rank order and reserved launch settings.
``uni`` and ``mp`` are unsupported.

Deployment configuration
------------------------

Use ``execution`` selectors and ``executors`` profiles for worker placement.
Model settings determine training/rollout topology; see the
`settings reference <../reference/configuration-settings.rst#executors>`__.
Nodes need software, models, files and routable endpoints before launch.
Connection failures never fall back locally.

Harness scorers declare ``execution_requirements()`` and load models inside workers.
Configured resources cannot reduce those requirements.
Close pools in ``finally``; worker death permanently fails a pool.
Run `executor tests <../../tests/execution>`__ and qualify remote/GPU behavior separately.
