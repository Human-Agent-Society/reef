Sandbox environments
====================

Use ``ProgramExecutor`` for programs and Harbor environments for task Dockerfiles.
Program backends are ``local``, ``sandbox``, ``e2b`` and ``modal``.
Harbor supports ``docker``, ``e2b`` and ``modal``.

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

``environment_config(environment, task_path, agent)`` supplies Harbor settings.
Provider checks require packages, credentials and any local daemon.
Cloud lifetime includes build, agent and verifier time; see
`provider limits <../../reef/execution/sandbox/task_environment.py>`__.

Tasks with Compose services on AgentENV
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set ``REEF_E2B_DOCKERFILE_BUILD=aenv`` and ``REEF_E2B_COMPOSE=aenv``.
Agent commands target ``main``; services share resources and communicate by name.
Other services require prebuilt images. Host files and variable substitutions are unsupported.
``aenv`` and ``aenv-compose-plan`` must share a directory on ``PATH``.
See the `Compose contract <../../reef/execution/sandbox/agentenv/compose.py>`__.

Build templates before the trials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``reef.execution.sandbox.e2b.build_task_template(task_path)`` prepares task/verifier templates before rollouts.
Set ``evaluation.require_prepared_templates`` to refuse trial-time builds.

Custom ``TaskTemplateBuilder.build(scenario, task_name, task_directory)`` returns when ready.
Raise ``TimeoutError`` for unavailable capacity; other failures end the attempt.
Owner-aware builders must reject unsupported owners.
Pinned work requires ``build_for_version(..., task_digest=...)`` using admitted files.
See `template contracts <../../reef/evaluation/task_templates.py>`__.

Keep the files of a trial
~~~~~~~~~~~~~~~~~~~~~~~~~

A configured ``TrialObserver.observe(trial)`` runs before scoring and cleanup.
Copy needed files before returning; ``trial.directory`` can be ``None``.
Observer errors log without changing scores.

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
