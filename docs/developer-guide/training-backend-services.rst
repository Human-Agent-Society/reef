Training backend services
=========================

A training backend runs as a service that deploys alone (RFC #1146). Reef
reaches it over one HTTP contract, for weights and harnesses alike. The
backend manages its own resources, such as a Ray cluster, Modal Sandboxes or a
remote training API, and keeps no state between jobs: a job reads everything
that it continues from in its artifact.

Configure Reef
--------------

.. code:: yaml

   training:
     backend: reef.train.remote:RemoteDeployment
     options:
       url: http://slime-backend:8000
       base: hf://Qwen/Qwen3-8B@<commit>
       state-dir: /var/lib/reef/remote-training
       token-env: REEF_TRAINING_BACKEND_TOKEN

- ``url`` is the base URL of the backend service.
- ``base`` is the artifact that a new scenario starts from: ``hf://`` or a
  ``file://`` directory on a volume that Reef and the backend share.
- ``state-dir`` is a directory that control and the training service share.
  Reef keeps the release of each checkpoint there.
- ``token-env`` names the variable that holds the bearer token of the
  service.

The training service runs no training: it submits each job to the service
and reads its status.

The contract
------------

Each route is under ``/v1`` and needs ``Authorization: Bearer <token>``.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Route
     - Body and answer
   * - ``GET /v1/backend``
     - ``{name, release_format, shares_scenarios}``.
   * - ``POST /v1/jobs/check``
     - A job; 200, or 422 with ``{"error": {"message"}}``.
   * - ``POST /v1/jobs``
     - A job; 202. The same ``job_id`` again does nothing.
   * - ``GET /v1/jobs/{job_id}``
     - ``{job_id, state, checkpoints, metrics, error, progress}``; 404 for
       a job that the service does not know.
   * - ``POST /v1/jobs/{job_id}/cancel``
     - 202. The job then ends ``failed``.
   * - ``POST /v1/releases/check``
     - ``{paths, adapter_config}`` of an imported release; 200, or 422.

A job is ``{job_id, scenario, artifact: {uri}, data: {uri, data_id,
sample_count, kind}, parameters}``. ``artifact.uri`` is the artifact that the
job starts from, and ``data.uri`` is the job data directory (`The job data
directory <processors.rst#the-job-data-directory>`__). A checkpoint is
``{step, artifact_uri, metrics}``. ``state`` is ``running``, ``finished`` or
``failed``; ``progress`` is the backend's own JSON object, or null.

An artifact directory holds ``release/``, what serves, and ``state/``, what
only training reads, such as an optimizer state or a harness history. A job
from an artifact without ``state/`` starts a new training from its release.

Write a backend
---------------

A backend is a ``reef.train.remote.server.TrainingBackend``:

- ``from_config(config)`` builds it from the server's configuration.
- ``info`` is its ``BackendInfo``.
- ``check(job)`` raises ``ValueError`` for a job that it cannot train.
- ``run(job, artifact, data, run)`` trains the job. ``artifact`` and
  ``data`` are local directories; ``artifact`` is ``None`` for a URI that the
  server cannot open, such as ``hf://``, which the backend reads itself. The
  backend calls ``run.save(step, directory, metrics)`` for each checkpoint,
  with ``release/`` and ``state/`` in ``directory``, and ``run.report(progress)``
  while it runs. It returns the job's metrics. ``run.save`` raises
  ``JobCancelled`` after a cancel.

Run a training component
------------------------

``reef.train.remote.components:ComponentBackend`` runs a ``TrainingComponent``,
such as Tinker, as a backend:

.. code:: yaml

   backend:
     module: reef.train.remote.components:ComponentBackend
     config:
       component: {type: reef.train.tinker_backend.launch:component_factory, ...}

A job's artifact becomes the component's start: ``state/`` is the start that a
checkpoint of the backend wrote; a release alone goes through
``import_start``; and the URI ``base:`` is the component's ``base_start``.
Each checkpoint becomes an artifact: its ``path`` as ``state/`` and its
``release_path`` as ``release/``. Set Reef's ``base`` to ``base:`` to start new
scenarios from the component's own default start.

Run the server
--------------

.. code:: bash

   export REEF_TRAINING_BACKEND_TOKEN=...
   python -m reef.train.remote.server --config backend.yaml

.. code:: yaml

   backend:
     module: my_backend:MyBackend
     config: {}
   files: /var/lib/reef/backend-files
   port: 8000

``files`` is where the server reads the artifact and data URIs and keeps the
checkpoints: a local directory, which Reef must share for ``file://`` URIs,
or a segment store plugin reference, such as an S3 store. The server keeps its
jobs in memory: after a restart, a job that it does not know is a 404.
