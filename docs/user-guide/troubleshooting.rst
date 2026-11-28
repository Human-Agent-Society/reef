Troubleshooting
===============

Start with status, the response body and service logs under ``reef.run-dir``
(default ``.reef/run/``):

.. code:: bash

   curl -sS -H "Authorization: Bearer $REEF_TOKEN" "$REEF_URL/reef/status"

Starting Reef
-------------

Run ``reef serve -c deploy.yaml --print-config`` to check effective settings.
Flags and ``REEF_*`` variables can override the file. Config paths are relative
to the working directory. For readiness failures, check downloads, GPUs and
``inference.log``. See `configuration <../reference/configuration.rst>`__.

Requests
--------

Check the Bearer token for 401 and the scenario and served selection for 404.
During startup, poll ``/reef/scenarios/{scenario}/inference/status`` for 503 errors.
Read other `inference errors <../reference/http-api-inference.rst#request>`__
before retrying; provider 4xx responses keep their status.

Reports and training
--------------------

- **Report accepted, no job:** check ``job_data_ready``, ``pipeline``,
  ``job_samples`` and group completeness. Consumed or stale samples do not retrain.
- **Report rejected:** 400 means invalid recipe fields. For 409, resend unchanged
  content or use another ``agent_record_id``.
- **Failed job:** read its traceback and check the recipe's loss flags.
- **Storage full:** check `checkpoint retention
  <operate.rst#set-the-checkpoint-cadence>`__. Move unrelated checkpoint files elsewhere.
- **Repeated work after restart:** only saved checkpoints preserve progress.
  See `recovery <operate.rst#restart-and-recovery>`__.
- **Slime rejects train-from:** use a `new deployment
  <operate.rst#train-from-an-artifact>`__ from the desired checkpoint.

Harness evolution
-----------------

Read ``GET /reef/harness/artifacts`` and its step page. Ties and unchanged trees
publish nothing. Check model tool-call support and evaluation timeouts.

For missing receipts, upgrade ``reef-client``. For missing endpoint bindings,
install with ``reef-client harness connect``. Resolve changed files before retrying
sync; ``sync --discard`` drops local edits. See the `harness tutorial
<../getting-started/harness-tutorial.rst>`__ for setup and conflict recovery.
