Sandbox environments
====================

Use ``reef.execution.sandbox`` to run harness programs or Harbor tasks.
Choose the interface for the work:

- ``ProgramExecutor`` runs one program through ``local``, ``sandbox``,
  ``e2b`` or ``modal``.
- A Harbor task environment builds the task's Dockerfile and runs its
  agent and verifier through ``docker``, ``e2b`` or ``modal``.

``build_executor(config)`` selects a program executor.
``remote_executor(kind, section, environ)`` builds a cloud executor from
provider options, forwarded ports and ``env_from``.
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

``exec`` accepts a shell command, working directory, environment and timeout.
It returns ``ExecResult`` with ``stdout``, ``stderr`` and ``return_code``.
Nonzero exit codes are results; launch failures raise ``ProgramLaunchError``
and timeouts raise ``ProgramTimeoutError``.
Changing the executor's configured user is unsupported.
Use async methods inside an event loop; ``execute`` and ``run_sync`` bridge
synchronous callers.

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
Host executors use ``program.host_binary()`` and may install it.
Local sandboxes expose ``host_install_dir()`` read-only.
Cloud executors use ``program.binary`` from their image, install nothing on
the host and reject self-isolating programs.
Harness adapters supply ``AdapterProgram``.

Open a session when multiple processes need shared sandbox state or forwarded
host ports:

.. code:: python

   session = executor.open_session(forward_ports=(gateway_port,), timeout_s=600)
   try:
       ...  # Run processes and pull their output files.
   finally:
       session.close()

Host-backed sessions can reuse the executor. Cloud sessions own one sandbox
and its tunnels. ``runs_on_host`` identifies access to host paths and resources.

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

``checked_task_environment`` accepts ``docker``, ``e2b`` and ``modal``.
``task_environment_probes`` checks each provider's packages, credentials or daemon.
``environment_config(environment, task_path, agent)`` builds the Harbor section.
Use E2B settings for compatible services such as AgentENV.

Cloud lifetime includes build, agent and verifier limits plus one hour,
capped at one day. Agent timeout overrides replace its task limit;
setup timeout overrides add to it.
Without an agent limit, Modal uses two hours and E2B uses Harbor's one-day default.

Reef's E2B environment requests deletion on timeout and preserves
``COPY --chmod`` modes. ``REEF_E2B_DOCKERFILE_BUILD=aenv`` uses
``aenv build`` for AgentENV Dockerfiles, including multi-stage builds.
Modal uses a VM runtime for Compose tasks without GPUs.

AgentENV builds reuse ready templates and wait for active builders.
Failed templates and templates stuck waiting without a builder for three
minutes are rebuilt. Retryable capacity, proxy and connection errors retry
until the build timeout; AgentENV's own HTTP 500 fails immediately.

Tasks with Compose services on AgentENV
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set both ``REEF_E2B_DOCKERFILE_BUILD=aenv`` and ``REEF_E2B_COMPOSE=aenv``
to run ``environment/docker-compose.yaml`` in one sandbox.
Agent commands target ``main``; services communicate by name.
Without this setting, E2B starts only the task image.

- ``main`` uses the task's Dockerfile or ``docker_image``.
  Other services require ``image`` and cannot declare ``build``.
- All services share the task's CPU and memory.
- Compose uses images, not templates. ``build_task_main_image`` can prepare
  only the main image. Missing prepared images raise ``TemplateNotPreparedError``.
- Trials delete their sandboxes. AgentENV itself pauses them on lifetime expiry.
- ``aenv`` and ``aenv-compose-plan`` must share a directory on ``PATH``.

AgentENV supports 1–24 services, bridge networks and named volumes.
It rejects host file inputs such as ``env_file`` and ``secrets``, and does
not supply Compose variable substitutions.
Repository cleanup tools must preserve images referenced by AgentENV's
image-tag and description records.

Build templates before the trials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Call ``reef.execution.sandbox.e2b.build_task_template(task_path)`` to prepare
the task and separate verifier templates before spending rollout time.
It returns the task template alias and uses trial build limits and resources.

E2B evaluations prepare templates before admitting their rollouts.
Template work consumes neither rollout concurrency nor sandbox slots.
Capacity waits can retry for up to six hours per attempt.
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

The first accepted termination request fixes the reason:
``exited``, ``timed_out`` or ``stopped``. A later zero exit does not change it.
``grace_seconds`` allows SIGTERM cleanup before SIGKILL; zero kills immediately.
Stopping refuses new launches. Closure waits for owned children and output.

Attaching a child scope transfers shutdown ownership. Reattachment to the
same owner is safe; other owners and cycles are refused.
POSIX process-group cleanup cannot reclaim detached descendants, containers
or remote sandboxes. Their owners must clean them up separately.

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
Running jobs receive five seconds before cancellation, then local processes
receive ten seconds of termination grace.
``close()`` waits at most twenty seconds and raises ``TimeoutError`` if
cleanup remains. Callers can wait again; timeout neither reopens the worker
nor proves that resources were released.
