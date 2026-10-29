Sandbox environments
====================

Use ``reef.sandbox.environment.ExecutorEnvironment`` to run commands and move
files through Harbor's ``BaseEnvironment`` interface. It supports Reef's local
and bubblewrap executors, and the cloud sandboxes of E2B and Modal. Harness rendering and candidate
evaluation remain outside this package.

Callers do not need to know which sandbox runs their commands. These
interfaces of ``reef.sandbox`` name the sandbox kinds and cloud providers for
them:

- ``build_executor`` reads the deployment's ``executor`` setting and returns an
  ``ProgramExecutor``.
- ``remote_executor(kind, section, environ)`` returns the cloud sandbox of
  provider ``kind`` (a key of ``REMOTE_EXECUTORS``: ``e2b`` or ``modal``) from
  that provider's options in ``section``. A caller with its own settings, such
  as ``evolution.proposer_agent``, passes its section; the factory reads only
  the provider's options, ``forward_ports`` and ``env_from``.
- ``reef.sandbox.remote.REMOTE_TASK_ENVIRONMENTS`` lists the Harbor environment
  types that run a task's containers on a provider, each with the variables
  its credentials come from.
- ``reef.sandbox.task_environment`` names every Harbor task environment type
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
(in-process Harbor trials) use only Harbor's task environment lifecycle: they
select an environment by name, read its needs from
``reef.sandbox.task_environment``, and reuse nothing from the executors. The
Terminus harness layers the two: an executor launches its runner, and the
runner then drives a Harbor task environment, so that harness also checks that
the jail passes the environment's credentials.


Run a command
-------------

Install Reef and create the root and workspace before constructing an
environment. This local example returns ``hello`` and exit code zero::

    import asyncio
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from harbor.environments.base import BaseEnvironment
    from reef.sandbox.environment import ExecutorEnvironment
    from reef.sandbox.process import LocalExecutor

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
with a ``reef.sandbox.process.SandboxProgram`` subclass, then bind it::

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
shared session running; its owner must close it after the proposer finishes.

Cancelling a command terminates its environment's processes before returning.
Cancellation also closes the session, including a borrowed one. Ordinary
timeouts preserve the executor's file-copy behavior. Transfer and cleanup
failures propagate. Retaining a stopped owned session and forcing template
rebuilds are unsupported.

Local execution provides no sandbox isolation. Bubblewrap and the cloud sandboxes retain their
existing read-only inputs, writable state and network behavior. Configuration
remains independent for evaluation and proposer execution; see
`Running rollouts <../user-guide/evolve-your-harness.rst#running-rollouts>`__.

``execute`` and ``run_sync`` bridge existing synchronous callers to this async
interface. Use the async methods directly inside a running event loop.
