"""Shared job scheduling, process ownership and sandbox execution.

``lifecycle`` defines the non-generic Execution control interface using only
the standard library. ``processes`` uses it for local child ownership. ``jobs`` schedules asynchronous work and composes process
scopes for cleanup. ``sandbox`` implements program execution, isolation,
remote sessions and file transfer. Jobs and sandbox adapters depend on
processes; processes depends on neither. Business callers construct commands
and interpret results outside this package.
"""
