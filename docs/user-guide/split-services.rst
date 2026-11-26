Run the services in separate processes
======================================

Split services to scale them independently or keep inference available during
control restarts. Start from a working deployment; default ``reef serve`` runs
all seven services in one process.

.. figure:: ../assets/services-light.svg
   :alt: Control queues work in the shared store. Inference, data, harness, samples, training and evaluation read or write the store.

Start the services
------------------

Use the same configuration and token. Run each command in its own process:

.. code:: bash

   export REEF_TOKEN="$(openssl rand -hex 16)"
   reef serve -c deploy.yaml --service control --reef.port 8901
   reef serve -c deploy.yaml --service inference --reef.port 8902
   reef serve -c deploy.yaml --service samples --reef.port 8903
   reef serve -c deploy.yaml --service training --reef.port 8904
   reef serve -c deploy.yaml --service evaluation --reef.port 8905
   reef serve -c deploy.yaml --service data --reef.port 8906
   reef serve -c deploy.yaml --service harness --reef.port 8907

Each process starts only its named service. Start required external engines
and training backends separately. Install backend dependencies and credentials
on the training host.

All services must share:

- Scenario and coordination stores. Share SQLite files on one host; use
  ``storage.record-backend: postgres`` across hosts.
- The data store and its task and segment stores.
- Artifact storage, with checkpoint write access for training.

See `storage setup <storage.rst>`__. Give each process separate
``storage.artifact-work-dir`` and ``storage.artifact-cache-dir`` paths to avoid
concurrent working-copy corruption. Training services need no shared training
directory; keep harness ``evolution.state_dir`` on the training host.

Send each request to its service
--------------------------------

.. list-table::
   :header-rows: 1

   * - Service
     - Routes
   * - ``inference``
     - ``/v1/`` inference routes, evaluation inference, served-artifact routes and scenario ``inference/status``.
   * - ``data``
     - Namespace ``data`` routes, artifact file reads and scenario ``jobs``. Scenario reads also work on control.
   * - ``harness``
     - ``/reef/harness``, ``/reef/harness/artifacts``, ``/reef/harness/import`` and ``/reef/harness/adapters``.
   * - ``control``
     - Other routes, including scenario writes, training cancellation, ``/reef/status`` and harness step pages.
   * - ``samples``, ``training``, ``evaluation``
     - ``/healthz`` only; work arrives through queues.

Data and harness services need no control lease and support replicas behind
one address. Point harness clients at the harness service; set
``harness.inference_url`` to their reachable inference address.

Data writes wait 30 seconds for control synchronization. ``synced: false`` means
the write remains pending, not lost. Harness imports return 503 after that wait
but remain queued; resending the same files waits for the original request.

Inference
---------

Poll scenario ``inference/status`` after changing served artifacts.
Requests return 503 during startup and 404 for unknown or nonserving scenarios.
Control retries failed startup after 60 seconds, doubling to one hour maximum.
Resending ``served`` retries immediately.

One worker owns the runtime under a lease. On takeover, it removes routes the
new runtime cannot serve before opening it. Local addresses usually change;
requests return 503 until selections reload. Native engines restart and interrupt
requests. Remote routes survive only when the runtime confirms them.
Startup status is not a continuous health check.

Keep all services on the same Reef version. Keep engine addresses private.
After resetting coordination storage, restart the inference component.

Training and evaluation
-----------------------

Unavailable training services leave new jobs waiting without consuming samples.
Tinker/Spindle imports return 503 after 60 seconds without a training response.
Evaluation needs inference; candidate loads do not change production routes,
and evaluation calls do not become training records.

Set ``evaluation.on-stop`` and ``evaluation.finish-timeout-s`` for graceful shutdown.
Keep the timeout below the supervisor's kill deadline. Graceful requeue consumes
no attempt; a forced kill consumes one and waits for lease expiry.

Control restarts
----------------

Inference, queued work and data writes continue while control is down.
Control must return to submit new work and synchronize data.

Run one active control instance and optional standbys on the same store.
A standby loads no scenarios and answers no requests, including ``/healthz``.
Use that probe to route traffic to the active instance.

Normal shutdown releases the lease; takeover occurs within 10 seconds.
After a crash, takeover waits for lease expiry (30 seconds).
A process that loses its lease stops its training work and exits.
For upgrades, start a standby, then stop the active instance.
