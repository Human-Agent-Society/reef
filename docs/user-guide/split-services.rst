Run the services in separate processes
======================================

``reef serve`` runs all seven services together by default. Split them to scale workers or keep inference up during control restarts.

.. figure:: ../assets/services-light.svg
   :alt: The seven services exchange work through the shared store. Control queues work and the inference owner starts engines.

Start the services
------------------

Run one process per service with the same Reef version, configuration and ``REEF_TOKEN``.
Give each process on a host a distinct port.

.. code:: bash

   reef serve -c deploy.yaml --service inference --reef.port 8902

- Share the `stores <storage.rst>`__ and artifact repository. Training needs checkpoint write access.
- Give each process its own ``storage.artifact-work-dir`` and ``storage.artifact-cache-dir``.
- Start external engines and backends separately. Put training credentials, dependencies and harness ``evolution.state_dir`` on the training host.

Send each request to its service
--------------------------------

.. list-table::
   :header-rows: 1

   * - Service
     - Routes
   * - ``inference``
     - Model requests, evaluation inference and scenario ``inference/status``.
   * - ``data``
     - Namespace data routes, artifact files and scenario jobs. Control also accepts scenario reads.
   * - ``harness``
     - ``/reef/harness`` and its ``artifacts``, ``import`` and ``adapters`` routes.
   * - ``control``
     - Other routes, including scenario writes and harness step pages.
   * - ``samples``, ``training``, ``evaluation``
     - ``/healthz`` only.

Point harness clients at ``harness``. Set ``harness.inference_url`` to the inference address they can reach.
After 30 seconds without control, a data write returns ``synced: false`` and a harness import returns 503.
Both stay queued. Resending identical files waits for the original import.

Inference
---------

On takeover, local engines restart and may interrupt requests. Remote routes survive if the new owner confirms them.
Poll `startup status <../reference/http-api-inference.rst#startup-status>`__ before sending traffic.
Keep engine addresses private. Restart inference after resetting coordination storage.

Training and evaluation
-----------------------

While training is down, jobs wait without consuming samples. Tinker and Spindle imports time out after 60 seconds (503).
For graceful evaluation shutdown, set ``evaluation.finish-timeout-s`` below the supervisor's kill deadline when ``evaluation.on-stop`` is ``finish``.
A forced kill consumes an attempt and delays retry until lease expiry.

Control restarts
----------------

Inference and queued work continue during a control outage. New work and data synchronization need active control.

Route traffic to the instance that answers ``/healthz``. A standby takes over within 10 seconds of normal shutdown,
or after the 30-second lease expires on a crash. For upgrades, start the standby before stopping active control.
