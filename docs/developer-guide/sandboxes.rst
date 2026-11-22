Sandbox environments
====================

``reef.execution.sandbox`` isolates what Reef runs for a harness or a task.
This page is for developers who run a program through a sandbox, or who run
Harbor tasks on a cloud provider. It covers two things that callers must not
confuse:

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - What
     - Runs
     - Kinds
   * - ``ProgramExecutor``
     - one program, such as a harness's pinned binary
     - ``local``, ``sandbox`` (bubblewrap on Linux, Seatbelt on macOS),
       ``e2b``, ``modal``
   * - Harbor task environment
     - a task's ``environment/Dockerfile``, with the task's agent and
       verifier inside it
     - ``docker``, ``e2b``, ``modal``

`Executors <executors.rst>`__ describes the executor settings of a
deployment. Harness rendering and candidate evaluation are outside this
package.

Callers do not name a sandbox kind or a provider. They use these interfaces
of ``reef.execution.sandbox``:

- ``build_executor(config)`` returns the ``ProgramExecutor`` of the
  deployment's ``executor`` setting.
- ``remote_executor(kind, section, environ)`` returns the cloud sandbox of
  provider ``kind`` (a key of ``REMOTE_EXECUTORS``: ``e2b`` or ``modal``). A
  caller with its own settings, such as ``evolution.agent``, passes its
  section. The factory reads only the provider's options, ``forward_ports``
  and ``env_from``.
- ``task_environment`` names the Harbor task environment types and makes
  their trial config (`Harbor task environments`_).

Run a command
-------------

``reef.execution.sandbox.environment.ExecutorEnvironment`` adapts an executor
to Harbor's ``BaseEnvironment`` without building a task image. Create the root
and the workspace first. This local example prints ``hello`` and exit code
zero::

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
``ExecResult`` with ``stdout``, ``stderr`` and ``return_code``.

- A nonzero command exit is a result.
- A launch failure raises ``ProgramLaunchError``.
- A timeout raises ``ProgramTimeoutError``.
- Commands run as the executor's configured user. Selecting another ``user``
  is unsupported.

``execute`` and ``run_sync`` bridge synchronous callers to this async
interface. Inside a running event loop, use the async methods directly.

Files and lifecycle
~~~~~~~~~~~~~~~~~~~

**Files.** Upload and download files or directories with Harbor's methods.
Environment paths refer to the local staged root, including in a cloud
sandbox. Relative paths start at the workspace, and a transfer refuses a path
outside the root. Uploads call the session's ``push`` and downloads call its
``pull``, so a cloud sandbox's download refreshes the staged copy first.
Upload files before you start the commands that use them.

**Ownership.** The caller owns the staged root. ``stop`` terminates the
environment's local processes but does not delete the caller's files.
``start`` opens the executor's session:

- When ``start`` opens a new session, such as a cloud sandbox, the environment
  owns it and ``stop(delete=True)`` closes it.
- When the caller passes an open session, the environment borrows it.
  ``stop`` leaves it running, and its owner must close it after the agent
  finishes.

**Cancellation.** Cancelling a command terminates its environment's processes
before it returns, and closes the session, including a borrowed one. Transfer
and cleanup failures propagate. Retaining a stopped owned session and forcing
template rebuilds are unsupported.

Bind an executor to a program
-----------------------------

An executor runs one program, such as a harness binary. Describe the program
with a ``reef.execution.sandbox.process.SandboxProgram`` subclass, then bind
it::

    executor, binary = build_executor(config).for_program(program)

``for_program`` returns the bound executor and the binary its processes
launch. Pass ``binary=`` to override the resolved binary. The harness package
provides ``reef.harness.tasks.vendor_install.AdapterProgram`` for an adapter.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Executor
     - How it resolves the binary
   * - On the Reef host
     - ``program.host_binary()``, which can install the binary. Bubblewrap
       binds ``program.host_install_dir()`` read-only, and Seatbelt allows
       reads of it.
   * - Cloud sandbox
     - ``program.binary`` in its image (an E2B template, a Modal image). It
       never installs on the Reef host, and it refuses a ``self_isolating``
       program.

A run of several processes that must reach ports on the Reef host opens a
session::

    session = executor.open_session(forward_ports=(gateway_port,), timeout_s=600)
    try:
        ...  # launch processes; session.pull(root, "workspace") refreshes their files
    finally:
        session.close()

- The local executor and a local sandbox on its default network already reach
  the host, so the session is the executor itself.
- An isolated bubblewrap network forwards the ports. An isolated Seatbelt
  profile allows them.
- A cloud sandbox opens one sandbox with a tunnel for each port.

``push``, ``pull`` and ``close`` do nothing for an executor whose processes
use the host's files. ``runs_on_host`` tells a caller whether processes see
the host's paths, environment and container runtime.

Isolation
~~~~~~~~~

The local executor provides no isolation. Bubblewrap, Seatbelt and the cloud
sandboxes each keep their read-only inputs, writable state and network rules;
`Running runs <../user-guide/evolve-your-harness.rst#running-runs>`__ lists
them, and the evaluation runs and a method's own agent are configured
separately.

On macOS, ``executor: sandbox`` selects ``SeatbeltExecutor``. It launches the
program with ``sandbox-exec`` under a profile with the same read, write and
credential rules as bubblewrap. It differs in three ways:

- Seatbelt profiles do not nest. A native loop in this sandbox runs its tool
  calls under the run's profile, and its trajectory reports enforcement
  ``none`` instead of a narrower profile per call.
- An isolated network denies every loopback port except the forwarded ones,
  but the host's LAN addresses stay reachable.
- macOS cannot limit address space, so preflight refuses
  ``limits.memory_bytes``. The other limits apply.

Harbor task environments
------------------------

Select a Harbor environment by name: ``docker``, ``e2b`` or ``modal``
(``HARBOR_TASK_ENVIRONMENTS``); ``checked_task_environment`` refuses any
other name. Use ``e2b`` for a self-hosted service that implements E2B's API,
such as AgentENV. Configure its endpoints in the rollout process, as
described under ``rollout.environment`` in the `configuration reference
<../reference/configuration.rst>`__.

``task_environment_probes(environment)`` returns a readiness probe for each
thing a host needs to run tasks in the type: the Docker daemon; the E2B key,
the e2b package and dockerfile-parse; or the two Modal token variables and
the modal package. ``remote.REMOTE_TASK_ENVIRONMENTS`` lists the variables
that each provider's credentials come from.

``environment_config(environment, task_path, agent)`` makes the
``environment`` section of a Harbor trial config. ``agent`` is the trial's
Harbor agent config, or ``None``. For ``docker``, the section holds the type
alone. For ``e2b`` and ``modal``, it also holds the sandbox's lifetime. For
``modal``, it selects Modal's VM runtime when the task has
``environment/docker-compose.yaml`` and asks for no GPU. For ``e2b``, it
names the class ``reef.execution.sandbox.e2b.KillOnTimeoutE2BEnvironment``
in Harbor's ``import_path`` field.

**Lifetime.** ``sandbox_timeout_secs`` is the sum of the task's build, agent
and verifier time limits, plus one hour, at most one day. A nonzero
``override_timeout_sec`` in ``agent`` replaces the task's agent time limit,
and an ``override_setup_timeout_sec`` adds to the sum. When the agent has no
time limit, the sum does not apply: ``modal`` gets a lifetime of two hours,
and ``e2b`` gets none, so Harbor's default of one day applies.

**The E2B class.** ``KillOnTimeoutE2BEnvironment`` is Harbor's
``E2BEnvironment`` with three differences:

- It creates the sandbox with ``lifecycle={"on_timeout": "kill"}``, which
  asks the service to delete the sandbox at the end of its lifetime. Without
  the argument, a self-hosted service can pause the sandbox instead.
- E2B's Dockerfile parser drops ``COPY --chmod``. The class adds a ``chmod``
  after each such copy, so the files keep their modes.
- AgentENV's E2B template builds refuse every ``COPY`` step. With
  ``REEF_E2B_DOCKERFILE_BUILD=aenv``, the class builds the Dockerfile with
  ``aenv build`` (``reef.execution.sandbox.agentenv_build``), which has
  Docker's semantics, multi-stage builds included.

With ``aenv``, the class uses a ready template and waits for a build that
another trial runs. It deletes and rebuilds a failed template, and a template
that stays ``waiting`` for three minutes with no build behind it. A failed
``aenv build`` deletes only the template that it made itself. Reef retries
HTTP 429, 502, 503 and 504 responses from the AgentENV API, the HTML page of
a proxy's HTTP 500, and connection timeouts until the build timeout. The
AgentENV API's own HTTP 500, for example for a full disk, fails the build at
once. An AgentENV worker answers 429 when it has no free build slot.

Build templates before the trials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, a trial builds a missing template inside its environment start
limit, and a wait for a free AgentENV builder counts against that limit. A
trial builds a verifier template only after its agent ends, so a late build
there loses the agent's work.

``reef.execution.sandbox.e2b.build_task_template(task_path)`` builds the
templates that a trial of the task would build, with the same aliases,
resources and build timeouts: the template of the task's environment, and the
template of each separate verifier environment of the task or of its steps.
It returns the alias of the task environment's template. A trial after it
finds the templates ready.

An evaluation on ``e2b`` uses it as follows. A deployment can also call the
function for each task before an evaluation.

- The evaluation makes the templates in one template item for each task,
  before it adds the task's rollouts. The item calls the deployment's
  ``evaluation.task_template_builder``, or ``build_task_template`` without
  one.
- It keeps template items for the first 32 tasks that still have rollouts to
  add, so their templates build while earlier rollouts run. The rollouts of a
  task whose template is ready start before those of an earlier task whose
  template still builds.
- A template item holds no rollout slot of ``max_concurrent_rollouts`` and no
  sandbox slot of a worker. When the build finds no free builder within the
  build timeout, the item waits again, for at most six hours in each attempt.
- ``evaluation.require_prepared_templates`` keeps template builds out of
  trials.

Who applies the config
~~~~~~~~~~~~~~~~~~~~~~

- ``reef.harness.harbor.rollouts.TaskRolloutRunner`` gives the config to each
  in-process trial and each environment build: the generator's runs, the
  evaluation rollouts and the training rollouts. The caller decides this, not
  the agent.
- A generator check on ``e2b`` runs ``harbor run -e <import path>``, with
  ``--ek sandbox_timeout_secs=<seconds>`` when the section holds a lifetime.
  A check passes no agent config, so only the task's time limits count. A
  check on ``docker`` or ``modal`` runs ``harbor run -e <type>`` without the
  config.
- Other Harbor runs get Harbor's own ``E2BEnvironment``: a ``harbor run -e
  e2b`` that Reef does not start, and the task environment that the
  ``terminus`` adapter of the Meta-Harness Terminal-Bench campaign drives
  from its own runner.

Shared execution control
------------------------

``reef.execution.lifecycle.Execution`` controls a run, or an owner of several
runs, without a result type. ``ProcessScope``, ``JobContext``,
``BackgroundJobWorker`` and ``ProgramExecution`` implement it. Results stay
on the concrete interfaces.

- ``request_stop()`` is thread-safe and idempotent. It requests cancellation
  and does not wait for cleanup.
- ``wait_closed(timeout_seconds)`` returns true only after the work and the
  owned cleanup finish, and false when the timeout expires. It raises a
  reported cleanup failure. ``None`` waits without a deadline; zero polls.
  Closure does not imply that the work succeeded.

Do not block the thread or event loop that the execution needs to finish.

``reef.execution.sandbox.environment.ProgramExecution(environment, argv)``
owns one run through an unstarted ``ExecutorEnvironment``. Call ``run()``
from synchronous code or await ``run_async()``; both return
``ProcessOutcome`` after cleanup. Another thread can cancel the run through
``Execution``. A stopped run cannot start again. A ``ProgramExecutor`` is the
executor or session resource, not the lifetime of a run.

Local process ownership
-----------------------

Use ``ProcessScope`` when a local command needs a timeout and coordinated
shutdown. ``run(ProcessRequest(...))`` starts a child, waits for it, and
returns a ``ProcessResult``: the captured output, the exit status and the
termination reason. The request supplies an absolute working directory and
the complete environment mapping. A missing executable or another launch
error propagates without registering a child.

- **Reason.** A result's reason is ``exited``, ``timed_out`` or ``stopped``.
  The first accepted termination request decides it, even if the child
  handles the signal and exits with code zero.
- **Grace.** ``grace_seconds`` gives a child time to clean up after SIGTERM;
  SIGKILL follows. Zero requests SIGKILL at once. The local executor
  terminates at once on a timeout. Harbor checks keep a cleanup grace.
- **Stop.** ``request_stop()`` refuses new launches and signals the current
  children. ``wait_closed`` returns true only when the scope and its attached
  scopes have reaped their direct children and handled their output.
- **Attachment.** A child scope has one owner, and attaching it transfers
  shutdown responsibility. Attaching again to the same owner is allowed. A
  second owner and a cycle are refused.

Scopes signal POSIX process groups by default. They cannot reclaim
descendants that leave the group, Docker containers or remote sandboxes.
Harbor remains responsible for its task environments, and a termination grace
for Harbor does not prove that remote cleanup succeeded.

Background job ownership
------------------------

``BackgroundJobWorker`` gives each running submission a ``JobContext`` and
binds its process scope across async calls and ``asyncio.to_thread``. Local
executor launches attach their scopes to this owner; generator checks use it
directly. A runner implements ``run(job)``, or overrides
``run_in_context(job, context)`` to read cancellation or use the scope.

**Results.** Every accepted submission produces one result, including
separate submissions with the same receipt. The runner must preserve the
receipt. A result becomes available after the runner's cleanup and after its
processes are reaped. Failure and cancellation produce ``Failed``. A runner
must await the cleanup of its own threads, async tasks and remote resources:
cancelling a coroutine does not stop a detached thread.

**Shutdown.**

- ``request_stop()`` stops admission and cancels queued work. Running jobs
  have five seconds to finish before cancellation. Their local processes then
  get a ten-second termination grace.
- ``wait_closed(timeout_seconds)`` returns false while the worker thread is
  alive, including when a runner delays cancellation.
- ``close()`` requests shutdown and waits up to twenty seconds, then raises
  ``TimeoutError`` if closure is incomplete. A caller can wait again. A
  timeout does not reopen the worker, and it does not show that the worker
  released its resources.
