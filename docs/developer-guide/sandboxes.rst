Sandbox environments
====================

Use ``ProgramExecutor`` to run programs and Harbor to run task trials.
Program backends are ``local``, ``sandbox``, ``e2b``, ``modal`` and ``boxd``.

Run a command
-------------

``ExecutorEnvironment`` provides Harbor's async interface without building an image:

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

``exec`` returns stdout, stderr and exit status.
Launch failures raise ``ProgramLaunchError``; timeouts raise ``ProgramTimeoutError``.
Use async methods inside event loops.

The caller owns staged files. Transfers cannot escape the staged root.
Owned sessions close at stop; supplied sessions are borrowed.
Cancellation closes even borrowed sessions. Cleanup failures propagate.
For shared state or forwarded ports, use ``open_session`` and close it in ``finally``.
See the `environment contract <../../reef/execution/sandbox/environment.py>`__.

Isolation
---------

``local`` has no isolation. ``sandbox`` uses bubblewrap on Linux or Seatbelt on macOS.
Seatbelt profiles cannot nest; LAN addresses remain reachable, and memory limits are unsupported.
Cloud executors use image binaries and reject self-isolating programs.
Configure evaluation and coding-agent isolation separately.

Harbor task environments
------------------------

Choose a provider and configure the host that runs the trial.

.. list-table::
   :header-rows: 1

   * - Provider
     - Harbor environment
     - Trial host setup
   * - Docker
     - ``docker``
     - A running Docker daemon
   * - E2B
     - ``e2b``
     - ``reef-infra[e2b]`` and ``E2B_API_KEY``
   * - AgentENV
     - ``e2b``
     - ``reef-infra[e2b]`` with your deployment's endpoints and API key
       (`Dockerfile setup <../../reef/execution/sandbox/agentenv/build.py>`__)
   * - Modal
     - ``modal``
     - ``reef-infra[modal]``, ``MODAL_TOKEN_ID`` and ``MODAL_TOKEN_SECRET``

Harbor reads CPU, memory and time limits from the task's ``task.toml``. To build a trial's ``environment`` section,
pass the Harbor environment value from the table, the task directory and the agent settings to
`environment_config <../../reef/execution/sandbox/task_environment.py>`__.
For template preparation, see the `evaluation settings <../reference/configuration-settings.rst#evaluation>`__.

Keep the files of a trial
~~~~~~~~~~~~~~~~~~~~~~~~~

To save agent or verifier logs from an evaluation, configure ``evaluation.trial-observer``.
If ``trial.directory`` is available, copy the files you need in ``TrialObserver.observe(trial)``
before returning. The worker removes the directory afterward.

Cancellation and background work
--------------------------------

``Execution.request_stop`` is thread-safe, idempotent and nonblocking.
``wait_closed`` succeeds only after work and owned cleanup finish;
timeouts do not prove cleanup completed.
Do not block the thread/event loop needed for cleanup.

``ProcessScope`` owns local children; detached processes and remote resources need separate cleanup.
``BackgroundJobWorker`` propagates its scope through async calls and ``asyncio.to_thread``.
Each submission produces one result after cleanup, including failure and cancellation.
Runners must join their own threads and tasks.
``close`` raises ``TimeoutError`` if cleanup exceeds twenty seconds.
