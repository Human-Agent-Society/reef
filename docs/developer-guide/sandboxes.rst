Sandbox environments
====================

Use ``reef.sandbox.environment.ExecutorEnvironment`` to run commands and move
files through Harbor's ``BaseEnvironment`` interface. It supports Reef's local,
bubblewrap and E2B execution implementations. Harness rendering and candidate
evaluation remain outside this package.

Callers do not need to know which sandbox runs their commands. Only
configuration code names a sandbox kind. For example,
``reef.sandbox.config.build_executor`` reads the deployment's ``executor``
setting and returns an ``EpisodeExecutor``. Every other caller uses that
interface.


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
exit is a result. Launch failures raise ``EpisodeLaunchError``; timeouts raise
``EpisodeTimeout``. Commands run as the executor's configured user; selecting
another ``user`` is unsupported.


Bind an executor to a program
-----------------------------

An executor runs one program, such as a harness binary. Describe the program
with a ``reef.sandbox.process.SandboxProgram`` subclass, then bind it::

    executor, binary = build_executor(config).for_program(program)

``for_program`` returns the bound executor and the binary its processes
launch. Pass ``binary=`` to override the resolved binary. An executor on the
Reef host resolves ``program.host_binary()``, which can install the binary.
Bubblewrap also binds ``program.host_install_dir()`` read-only. E2B resolves
``program.binary`` in a template and never installs on the Reef host; it
refuses a ``self_isolating`` program. The harness layer provides
``reef.harness.episodes.vendor_install.AdapterProgram`` for an adapter.

A run of several processes that must reach ports on the Reef host opens a
session::

    session = executor.open_session(forward_ports=(gateway_port,), timeout_s=600)
    try:
        ...  # launch processes; session.pull(root, "workspace") refreshes their files
    finally:
        session.close()

Local and default-network bubblewrap executors already reach the host, so the
session is the executor itself. An isolated bubblewrap network forwards the
ports. E2B opens one VM with a tunnel for each port. ``push``, ``pull`` and
``close`` do nothing for an executor whose processes use the host's files.
``runs_on_host`` tells a caller whether processes see the host's paths,
environment and container runtime.


Files and lifecycle
-------------------

Upload and download files or directories with Harbor's corresponding methods.
Environment paths refer to the local staged root, including with E2B. Relative
paths start at the workspace. Transfers reject paths outside the root.
Uploads call the session's ``push`` and downloads call its ``pull``, so an E2B
download refreshes the staged copy before copying it to the destination.
Upload files before starting commands that use them.

The caller owns the staged root. ``stop`` terminates the environment's local
processes but does not delete caller-owned files. ``start`` opens the
executor's session. When that opens a new session, such as an E2B VM, the
environment owns it and ``stop(delete=True)`` closes it. Passing an open
session borrows it for a trial. Stopping the trial's environment leaves the
shared session running; its owner must close it after the proposer finishes.

Cancelling a command terminates its environment's processes before returning.
Cancellation also closes the session, including a borrowed one. Ordinary
timeouts preserve the executor's file-copy behavior. Transfer and cleanup
failures propagate. Retaining a stopped owned session and forcing template
rebuilds are unsupported.

Local execution provides no sandbox isolation. Bubblewrap and E2B retain their
existing read-only inputs, writable state and network behavior. Configuration
remains independent for evaluation and proposer execution; see
`Running episodes <../user-guide/evolve-your-harness.rst#running-episodes>`__.

``execute`` and ``run_sync`` bridge existing synchronous callers to this async
interface. Use the async methods directly inside a running event loop.
