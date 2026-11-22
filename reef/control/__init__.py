"""The loaded scenarios of a process: their state and commits, and the workers that train them.

``scenario`` holds one scenario: its state, its commit ordering and its recovery. The modules at the package root
hold the dispatcher, the process-wide owner of every scenario:

- ``dispatcher`` holds :class:`Dispatcher`: the public scenario API, scenario
  deletion and idle unload, inline consumption, and the lifecycle of the
  background threads. It delegates scenario table management (creation, resolution,
  per-scenario locking) to :class:`reef.control.scenario.ScenarioRegistry`, then layers
  record acceptance and the training workers on top.
- ``ingest`` validates records and appends them to one loaded scenario.
- ``status`` builds the training status and runs its health checks.
- ``training_errors`` keeps the training errors and failure counts that the status
  and the operational metrics report.

``workers`` holds the background workers that the dispatcher runs.

Request handling lives in ``reef.api``; this package is transport-free.

Lock order: cycle locks, then a scenario's registry lock, then the
coordination lock of ``workers.training``, then the training errors lock.

Import each module directly; this package exports nothing.
"""
