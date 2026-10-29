Sandbox environments
====================

Use ``reef.execution.sandbox.environment.ExecutorEnvironment`` to run commands and move
files through Harbor's ``BaseEnvironment`` interface. It supports Reef's local
and bubblewrap executors, and the cloud sandboxes of E2B and Modal. Harness rendering and candidate
evaluation remain outside this package.

Callers do not need to know which sandbox runs their commands. These
interfaces of ``reef.execution.sandbox`` name the sandbox kinds and cloud providers for
them:

- ``build_executor`` reads the deployment's ``executor`` setting and returns an
  ``ProgramExecutor``.
- ``remote_executor(kind, section, environ)`` returns the cloud sandbox of
  provider ``kind`` (a key of ``REMOTE_EXECUTORS``: ``e2b`` or ``modal``) from
  that provider's options in ``section``. A caller with its own settings, such
  as ``evolution.agent``, passes its section; the factory reads only the
  provider's options, ``forward_ports`` and ``env_from``.
- ``reef.execution.sandbox.remote.REMOTE_TASK_ENVIRONMENTS`` lists the Harbor environment
  types that run a task's containers on a provider, each with the variables
  its credentials come from.
- ``reef.execution.sandbox.task_environment`` names every Harbor task environment type
  (``HARBOR_TASK_ENVIRONMENTS``) and checks what a host needs to run tasks in
  the ones it has probes for (``PROBED_TASK_ENVIRONMENTS``: ``docker`` and
  ``e2b``). ``task_environment_probes(environment)`` returns a readiness probe
  for each need, such as the Docker daemon, or the E2B key, the e2b package and
  dockerfile-parse; ``checked_task_environment`` refuses any other name.

Every other caller uses these interfaces.

Executors and Harbor task environments
--------------------------------------

The two run different things. A ``ProgramExecutor`` launches one program,
such as a harness's pinned binary, locally, in bubblewrap, or in an E2B or
Modal sandbox, and ``ExecutorEnvironment`` adapts one to Harbor's
``BaseEnvironment`` without building a task image. A Harbor task environment
is where Harbor builds a task's ``environment/Dockerfile``, starts it, runs
the task's agent and verifier in it and stops it.

The generator service's checks (``harbor run -e`` children) and rollouts
(in-process Harbor trials) use Harbor's task environment lifecycle. They select
an environment by name and read its needs from ``reef.execution.sandbox.task_environment``.
Checks share local process management with the executors through
``reef.execution.processes.ProcessScope``; their command arguments and scoring
remain in the generator. The Terminus harness layers the two: an executor
launches its runner, which then drives a Harbor task environment.

Shared execution control
------------------------

``reef.execution.lifecycle.Execution`` controls a run or an owner of several
runs without specifying its result type. ``ProcessScope``, ``JobContext``,
``BackgroundJobWorker`` and ``ProgramExecution`` implement this interface.

``request_stop()`` is thread-safe and idempotent. It requests cancellation but
does not wait for cleanup. ``wait_closed(timeout_seconds)`` returns true only
after work and owned cleanup finish. It returns false when the timeout expires,
and raises a reported cleanup failure. ``None`` waits without a deadline;
zero polls. Do not block the thread or event loop needed to finish the execution.
Closure does not imply that the work succeeded.

``reef.execution.sandbox.environment.ProgramExecution(environment, argv)`` owns
one run through an unstarted ``ExecutorEnvironment``. Use ``run()`` from synchronous
code or await ``run_async()``. Both return ``ProcessOutcome`` after cleanup.
Another thread can request cancellation through ``Execution``. A stopped run
cannot start again. ``ProgramExecutor`` remains the executor or session resource;
it is not an execution lifetime. Existing ``execute(...)`` callers use this run object.

Results stay on concrete interfaces: ``ProcessScope.run`` returns ``ProcessResult``,
and the worker's ``poll`` consumes job results. The shared interface has no
result field, generic parameter or untyped result container.

Local process ownership
-----------------------

Use ``ProcessScope`` when a local command needs a timeout and coordinated
shutdown. ``run(ProcessRequest(...))`` starts a child, waits for it, and returns
its captured output, exit status and termination reason. The request supplies
an absolute working directory and the complete environment mapping. Missing
executables and other launch errors propagate without registering a child.

A result's reason is ``exited``, ``timed_out`` or ``stopped``. The first accepted
termination request determines the reason, even if the child handles termination
and exits with code zero. ``grace_seconds`` gives a child time to clean up after
SIGTERM; SIGKILL follows when the grace expires. Zero grace requests immediate
SIGKILL. The local executor retains its immediate timeout termination. Harbor
checks retain a cleanup grace before forced termination.

``request_stop()`` rejects new launches and signals current children. It does
not mean they have exited. ``wait_closed(timeout_seconds)`` returns true only
when the scope and its attached scopes have reaped their direct children and
finished handling their output. False means cleanup is still pending. A child
scope has one owner; attaching it transfers shutdown responsibility. Repeated
attachment to the same owner is allowed, but sharing between owners and cycles
are refused.

Scopes signal POSIX process groups by default. They cannot reclaim descendants
that leave the group, Docker containers or remote sandboxes. Harbor remains
responsible for its task environments. Giving Harbor a termination grace does
not prove that remote cleanup succeeded.

Background job ownership
------------------------

``BackgroundJobWorker`` gives each running submission a ``JobContext`` and binds
its process scope across async calls and ``asyncio.to_thread``. Local executor
launches attach their scopes to this owner; generator checks use it directly.
A runner can override ``run_in_context(job, context)`` to read cancellation or
use the scope explicitly. Existing ``run(job)`` implementations remain supported.

Every accepted submission produces one result, including separate submissions
with the same receipt. The runner must preserve the receipt. Results become
available after runner cleanup and owned-process reaping. Failure and cancellation
produce ``Failed``. Runners must await cleanup of their own threads, async tasks
and remote resources; cancelling a coroutine does not stop a detached thread.

``request_stop()`` stops admission and cancels queued work. Running jobs have
five seconds to finish before cancellation; their local processes then receive
a ten-second termination grace. ``wait_closed(timeout_seconds)`` returns false
while the worker thread remains alive, including when a runner delays cancellation.
``close()`` requests shutdown and waits up to twenty seconds, then raises
``TimeoutError`` if closure is incomplete. A caller can wait again; a timeout
does not reopen the worker or establish that its resources were released.


Run a command
-------------

Install Reef and create the root and workspace before constructing an
environment. This local example returns ``hello`` and exit code zero::

    import asyncio
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from harbor.environments.base import BaseEnvironment
    from reef.execution.sandbox.environment import ExecutorEnvironment
    from reef.execution.sandbox.process import LocalExecutor

    async def example():
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            environment: BaseEnvironment = ExecutorEnvironment(
                LocalExecutor(), root=root, workspace=workspace
            )
            try:
                await environment.start(force_build=False)
                result = await environment.exec("printf hello")
                print(result.stdout, result.return_code)
            finally:
                await environment.stop(delete=True)

    asyncio.run(example())

``exec`` accepts a shell command, a working directory inside the root,
environment variables and a timeout in seconds. It returns Harbor's
``ExecResult`` with ``stdout``, ``stderr`` and ``return_code``. A nonzero command
exit is a result. Launch failures raise ``RolloutLaunchError``; timeouts raise
``RolloutTimeout``. Commands run as the executor's configured user; selecting
another ``user`` is unsupported.


Bind an executor to a program
-----------------------------

An executor runs one program, such as a harness binary. Describe the program
with a ``reef.execution.sandbox.process.SandboxProgram`` subclass, then bind it::

    executor, binary = build_executor(config).for_program(program)

``for_program`` returns the bound executor and the binary its processes
launch. Pass ``binary=`` to override the resolved binary. An executor on the
Reef host resolves ``program.host_binary()``, which can install the binary.
Bubblewrap also binds ``program.host_install_dir()`` read-only. A cloud sandbox
resolves ``program.binary`` in its image (an E2B template, a Modal image) and
never installs on the Reef host; it refuses a ``self_isolating`` program. The harness layer provides
``reef.harness.rollouts.vendor_install.AdapterProgram`` for an adapter.

A run of several processes that must reach ports on the Reef host opens a
session::

    session = executor.open_session(forward_ports=(gateway_port,), timeout_s=600)
    try:
        ...  # launch processes; session.pull(root, "workspace") refreshes their files
    finally:
        session.close()

Local and default-network bubblewrap executors already reach the host, so the
session is the executor itself. An isolated bubblewrap network forwards the
ports. A cloud sandbox opens one sandbox with a tunnel for each port. ``push``, ``pull`` and
``close`` do nothing for an executor whose processes use the host's files.
``runs_on_host`` tells a caller whether processes see the host's paths,
environment and container runtime.


Files and lifecycle
-------------------

Upload and download files or directories with Harbor's corresponding methods.
Environment paths refer to the local staged root, including in a cloud sandbox. Relative
paths start at the workspace. Transfers reject paths outside the root.
Uploads call the session's ``push`` and downloads call its ``pull``, so a cloud
sandbox's download refreshes the staged copy before copying it to the destination.
Upload files before starting commands that use them.

The caller owns the staged root. ``stop`` terminates the environment's local
processes but does not delete caller-owned files. ``start`` opens the
executor's session. When that opens a new session, such as a cloud sandbox, the
environment owns it and ``stop(delete=True)`` closes it. Passing an open
session borrows it for a trial. Stopping the trial's environment leaves the
shared session running; its owner must close it after the agent finishes.

Cancelling a command terminates its environment's processes before returning.
Cancellation also closes the session, including a borrowed one. Ordinary
timeouts preserve the executor's file-copy behavior. Transfer and cleanup
failures propagate. Retaining a stopped owned session and forcing template
rebuilds are unsupported.

Local execution provides no sandbox isolation. Bubblewrap and the cloud sandboxes retain their
existing read-only inputs, writable state and network behavior. Configuration
remains independent for the evaluation rollouts and for an agent that a method
runs; see
`Running rollouts <../user-guide/evolve-your-harness.rst#running-rollouts>`__.

``execute`` and ``run_sync`` bridge existing synchronous callers to this async
interface. Use the async methods directly inside a running event loop.
