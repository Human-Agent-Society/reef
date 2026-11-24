"""The coordination store of a deployment: what connects the services.

No service calls another. Each one reads and writes this store.

- ``store`` holds the coordination store as one process opens it (``Coordination``).
- ``work_queue`` holds the work queues, through which control gives work to the other services.
- ``served_models`` holds the serving requests and the routes that completed requests publish.
- ``control_lease`` holds the control lease: which control instance is active.

These modules import ``reef.core`` and each other. ``reef.storage.sql.coordination`` implements the tables.

Import each module directly; this package exports nothing.
"""
