"""The process-wide coordinator that owns every scenario and its training threads.

The dispatcher delegates scenario table management (creation, resolution,
per-scenario locking) to :class:`reef.scenario.ScenarioRegistry`, then layers
record acceptance, background training drains, and publication on top.
Request handling lives in ``reef.service``; the dispatcher is transport-free.

- ``dispatcher`` holds :class:`Dispatcher`: the public scenario API, scenario
  deletion and idle unload, inline consumption, and the lifecycle of the
  background threads.
- ``ingest`` validates records and appends them to one loaded scenario.
- ``local_workers`` runs one worker thread per local backend, shares the
  cycle locks with dispatched turns, and closes replaced instances.
- ``training_loop`` runs the training thread for dispatched backends.
- ``status`` builds the training status and runs its health checks.
- ``errors`` keeps the training errors and failure counts that the status
  and the operational metrics report.

Lock order: cycle locks, then a scenario's registry lock, then the
coordination lock of ``local_workers``, then the training errors lock.
"""

from reef.dispatcher.dispatcher import Dispatcher, build_default_dispatcher

__all__ = ["Dispatcher", "build_default_dispatcher"]
