Sandbox environments
====================

Use ``reef.execution.sandbox`` to run harness programs or Harbor tasks.
Choose the interface for the work:

- ``ProgramExecutor`` runs one program through ``local``, ``sandbox``,
  ``e2b`` or ``modal``.
- A Harbor task environment builds the task's Dockerfile and runs its
  agent and verifier through ``docker``, ``e2b`` or ``modal``.

``build_executor(config)`` selects the executor.
Use `Executors <executors.rst>`__ for worker placement.

Run a command
-------------

``ExecutorEnvironment`` adapts a program executor to Harbor's async interface
without building a task image:

.. code:: python

   import asyncio
   from pathlib import Path
   from tempfile import TemporaryDirectory

   from reef.execution.sandbox.environment import ExecutorEnvironment
   from reef.execution.sandbox.process import LocalExecutor


   async def example():
       with TemporaryDirectory() as directory:
           root = Path(directory)
           workspace = root / "workspace"
           workspace.mkdir()
           environment = ExecutorEnvironment(LocalExecutor(), root=root, workspace=workspace)
           try:
               await environment.start(force_build=False)
               result = await environment.exec("printf hello")
               print(result.stdout, result.return_code)  # hello 0
           finally:
               await environment.stop(delete=True)


   asyncio.run(example())

``exec`` returns ``stdout``, ``stderr`` and ``return_code``.
Nonzero exits are results; launch failures raise ``ProgramLaunchError``
and timeouts raise ``ProgramTimeoutError``.
Use async methods in event loops; ``execute`` and ``run_sync`` serve synchronous callers.

Files and lifecycle
~~~~~~~~~~~~~~~~~~~

Upload files before commands use them. Relative paths start at the workspace;
transfers outside the staged root are refused. Cloud uploads push staged
files, and downloads pull fresh files before reading them.

The caller owns the staged root. Stopping an environment leaves those files.
A session opened by ``start`` is owned and closed by ``stop(delete=True)``.
An explicitly supplied session is borrowed and normally remains open.

Cancellation terminates environment processes and closes even borrowed
sessions before returning. Transfer and cleanup failures propagate.
Keeping stopped owned sessions and forcing template rebuilds are unsupported.

Bind an executor to a program
-----------------------------

Describe the program with ``SandboxProgram``, then bind it:

.. code:: python

   executor, binary = build_executor(config).for_program(program)

``binary=`` can override resolution.
Host executors resolve ``program.host_binary()``; cloud executors use the
image's ``program.binary`` and reject self-isolating programs.
Harness adapters supply ``AdapterProgram``.

Use ``open_session(forward_ports=..., timeout_s=...)`` to share sandbox state
or host ports. Always close the session in ``finally``.
``runs_on_host`` identifies access to host paths and resources.

Isolation
~~~~~~~~~

``local`` provides no isolation. ``sandbox`` selects bubblewrap on Linux or
Seatbelt on macOS. Evaluation and coding-agent isolation are configured separately.
See `Running runs <../user-guide/evolve-your-harness.rst#running-runs>`__.

Seatbelt differs from bubblewrap:

- Profiles cannot nest; native tool calls retain the outer run profile.
- Isolated networking restricts loopback ports but leaves host LAN addresses reachable.
- ``limits.memory_bytes`` is unsupported and fails preflight.

Harbor task environments
------------------------

``environment_config(environment, task_path, agent)`` creates the Harbor
settings for ``docker``, ``e2b`` or ``modal``.
Provider probes check packages, credentials and daemons.
Use E2B settings for compatible services such as AgentENV.

Set ``REEF_E2B_DOCKERFILE_BUILD=aenv`` for AgentENV Dockerfile builds.
Cloud lifetime is finite and includes build, agent and verifier time.
The `environment contract <../../reef/execution/sandbox/task_environment.py>`__
defines provider limits and timeout overrides.

Tasks with Compose services on AgentENV
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set both ``REEF_E2B_DOCKERFILE_BUILD=aenv`` and ``REEF_E2B_COMPOSE=aenv``
to run ``environment/docker-compose.yaml`` in one sandbox.
Agent commands target ``main``; services communicate by name.
Without this setting, E2B starts only the task image.

Other services need prebuilt images; only ``main`` can use the task Dockerfile.
Services share CPU and memory. Host files and variable substitutions are unsupported.
``aenv`` and ``aenv-compose-plan`` must share a directory on ``PATH``.
See the `Compose contract <../../reef/execution/sandbox/agentenv/compose.py>`__
for supported fields and image preparation.

Build templates before the trials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Call ``reef.execution.sandbox.e2b.build_task_template(task_path)`` to prepare
the task and separate verifier templates before spending rollout time.
It returns the task template alias and uses trial build limits and resources.

E2B evaluations prepare templates before rollouts, without consuming rollout slots.
Set ``evaluation.require_prepared_templates`` to prohibit trial-time builds.

For custom preparation, subclass ``TaskTemplateBuilder`` and configure
``evaluation.task_template_builder``:

- ``async build(scenario, task_name, task_directory)`` returns when templates
  are ready. Raise ``TimeoutError`` for unavailable builder capacity;
  other errors fail the attempt.
- Override ``build_for_owner`` when owner identity selects templates.
  Reject missing or unsupported owners; never substitute another owner.
- Override ``build_for_version(..., task_digest=...)`` for pinned work.
  Use the admitted digest or supplied pinned directory.

``evaluation.pin-task-files`` rejects builders without version-aware support.

Keep the files of a trial
~~~~~~~~~~~~~~~~~~~~~~~~~

Configure ``evaluation.trial-observer`` with a ``TrialObserver`` subclass.
``observe(trial)`` runs before scoring and directory removal.
Copy needed files before returning; do not retain the directory reference.
``trial.directory`` can be ``None``. Observer exceptions are logged without
changing the rollout or its score.

Who applies the config
~~~~~~~~~~~~~~~~~~~~~~

Reef's task rollout runner supplies environment settings for generation,
evaluation and training. Harbor commands started outside Reef use Harbor's
own environment defaults. See
`Rollout job parameters <../reference/configuration-generated-tasks.rst#rollout-job-parameters>`__.

Shared execution control
------------------------

``Execution`` controls cancellation and closure independently of result types:

- ``request_stop()`` is thread-safe, idempotent and nonblocking.
- ``wait_closed(timeout_seconds)`` returns ``True`` only after work and
  owned cleanup finish. It returns ``False`` on timeout and raises cleanup failures.
  ``None`` waits indefinitely; zero polls.

Closure does not imply success. Do not block the event loop or thread needed
for cleanup.
``ProgramExecution`` owns one run through an unstarted ``ExecutorEnvironment``.
``run`` and ``run_async`` return ``ProcessOutcome`` after cleanup;
a stopped execution cannot restart.

Local process ownership
-----------------------

``ProcessScope.run(ProcessRequest(...))`` launches a child and returns output,
exit status and termination reason. Supply an absolute working directory and
a complete environment mapping. Launch errors propagate.

The first termination reason remains final, even if the process later exits 0.
``grace_seconds`` bounds SIGTERM cleanup before SIGKILL; zero kills immediately.
Stopping refuses new launches; closure waits for owned children and output.

Attaching child scopes transfers shutdown ownership.
Owners must separately clean detached descendants, containers and remote sandboxes.
See `ProcessScope <../../reef/execution/processes.py>`__ for ownership constraints.

Background job ownership
------------------------

``BackgroundJobWorker`` gives each accepted submission a ``JobContext``.
Its process scope follows async calls and ``asyncio.to_thread``.
Runners implement ``run(job)`` or ``run_in_context(job, context)``.

Each submission produces one result, even for duplicate receipts.
Preserve the receipt. Publish results only after cleanup;
failures and cancellation produce ``Failed``.
Runners must join their own threads, tasks and remote resources.
Coroutine cancellation alone cannot stop detached threads.

Shutdown rejects new work and cancels queued jobs.
``close()`` waits at most twenty seconds, then raises ``TimeoutError`` if
cleanup remains. Callers can wait again; a timeout does not prove cleanup finished.
