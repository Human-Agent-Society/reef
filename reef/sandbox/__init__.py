"""Shared process execution and isolation through Harbor's environment contract.

``process`` owns the ``EpisodeExecutor`` interface, local execution, resource
limits and process cleanup. Callers bind an executor to a ``SandboxProgram``
and open sessions through that interface, without knowing the sandbox kind.
``config`` builds the executor a deployment's ``executor`` setting selects.
``e2b`` owns cloud sessions, file transfer and
gateway tunnels. ``environment`` adapts any executor to
``harbor.environments.base.BaseEnvironment``. Harness rendering and evaluation
remain with their callers. This package depends on core values and Harbor,
never on harness, recipes or training.
"""
