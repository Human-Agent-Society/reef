"""Shared process execution and isolation through Harbor's environment contract.

``process`` owns the ``ProgramExecutor`` interface, local execution, resource
limits and execution adapters. The shared ``reef.execution.processes`` module owns local
child processes, termination grace and reaping. Callers bind an executor to a ``SandboxProgram``
and open sessions through that interface, without knowing the sandbox kind.
``config`` builds the executor a deployment's ``executor`` setting selects.
``task_environment`` names the Harbor task environments, checks what each
needs, and makes the environment config of a Harbor trial. For the ``e2b``
environment type, that config names a class in ``e2b``: Harbor's E2B
task environment, which also asks the service to delete each sandbox at its
timeout. ``remote`` owns cloud sessions, file transfer and gateway
tunnels for any provider, and the credentials of the remote Harbor task
environments; ``e2b`` and ``modal`` add each provider's sandbox. ``environment`` adapts any executor to
``harbor.environments.base.BaseEnvironment``. Harness rendering and evaluation
remain with their callers. This package depends on core values and Harbor,
never on harness, recipes or training.
"""
