"""The background workers that the dispatcher runs.

- ``local_workers`` runs one worker thread per trainer: it trains harness
  backends and submits and polls the jobs of weight trainers, and closes
  replaced instances. ``manager_worker`` runs the training manager.
- ``samples`` makes the samples of the scenarios of this process for the samples service.
- ``store_changes`` watches the shared store for what other processes add, and ``data_sync_worker`` syncs a
  loaded scenario with the items of its owner.
- ``served`` queues artifact startup and renews its lease in control. ``serving`` loads the selected artifacts in
  the inference service and publishes their routes.
- ``scenario_deletes`` holds the cleanup of deleted scenarios, and
  ``work_retention`` the sweep of control that runs it.

Import each module directly; this package exports nothing.
"""
