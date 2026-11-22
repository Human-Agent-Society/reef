Service and storage settings
============================

The keys of the ``reef``, ``harness``, ``storage``, ``data`` and ``evaluation`` sections: the HTTP service and its scenario limits, the stores of a deployment, and the limits of the data and evaluation services.

The ``reef`` and ``harness`` sections
-------------------------------------

.. container:: config

   ``reef.host`` : 127.0.0.1
       bind address

   ``reef.port`` : 8900
       bind port

   ``reef.run-dir`` : .reef/run
       the directory of the service logs

   ``reef.ready-timeout`` : 30
       HTTP startup deadline in seconds. When the HTTP service loads the model into training workers or starts serving engines before it answers, as with Slime and whenever Reef's engines serve, Reef raises it to at least ``training.ready-timeout``

   ``reef.served-url``
       the URL at which a composite recipe's own evaluation calls (runs and the method's calls to the served model) reach this service; the default is loopback on the bind port, so set it when tasks run on another host. A recipe of one component calls its runtime's endpoint directly and ignores it

   ``reef.console-origins`` : []
       exact browser console origins allowed to access the HTTP service; disabled by default

   ``reef.token``
       the bearer token the service accepts. Use ``tokens: [...]`` to accept several while rotating.

   ``reef.allow-unauthenticated`` : false
       when true, serve without a token on a ``reef.host`` that other machines can reach. When false, Reef does not start on such an address without a token. A loopback address, such as ``127.0.0.1``, ``::1`` or ``localhost``, needs no token

   ``reef.allow-implicit-scenario-creation`` : true
       when false, a report, a record import or a harness install for an unknown scenario is HTTP 404. An inference request never creates a scenario, whatever this setting says

   ``reef.scenario-idle-unload-s`` : unset
       unload a scenario that is idle for this many seconds; the next request loads it again. This key and the three after it are described in `Limit the scenarios a service holds <../user-guide/operate.rst#limit-the-scenarios-a-service-holds>`__

   ``reef.max-loaded-scenarios`` : unset
       the most loaded scenarios; at the limit, the idle scenario with the oldest use unloads

   ``reef.max-scenarios`` : unset
       the most registered scenarios

   ``reef.scenario-admission`` : unset
       a ``ScenarioAdmission`` plugin reference that replaces the built-in limits

   ``reef.routing_manifest`` : unset
       the manifest of routing among external targets (`Scenario routing between students and teachers <configuration-routing.rst#scenario-routing-between-students-and-teachers>`__)

   ``reef.routing_deployment``
       the settings of the routing deployment (`Managed routing lifecycle <configuration-routing.rst#managed-routing-lifecycle>`__)

   ``harness.inference-url``
       the URL at which clients reach the inference service. A client's harness layer (``GET /reef/harness?form=client``) sends its model calls to ``<url>/reef/scenarios/<scenario>/v1``. The default is the ``service_url`` that the client gave, which is right when one process runs every service

`Operate a deployment <../user-guide/operate.rst>`__ and `Run the services in separate processes <../user-guide/split-services.rst>`__ describe how to run the service.

The ``storage`` section
-----------------------

Storage paths default under ``.reef/`` in the launch directory. Point them somewhere persistent. `Choose the storage backend <../user-guide/storage.rst>`__ describes how to choose each store.

.. container:: config

   ``storage.artifact-backend`` : git_lfs
       where snapshots are stored: ``git_lfs``, or a ``ConfigurableRepositoryBackendFactory`` plugin reference

   ``storage.artifact-repository`` : .reef/artifacts.git
       the Git LFS repository of the snapshot chain; only for ``git_lfs``. A Hugging Face repository needs the factory's ``refs`` and ``token_env`` options (`Choose the artifact backend <../user-guide/storage.rst#choose-the-artifact-backend>`__)

   ``storage.artifact-work-dir`` : .reef/artifact-work
       materialization scratch

   ``storage.artifact-cache-dir`` : .reef/artifact-cache
       fetched artifact cache; artifact imports run under ``imports/`` in it

   ``storage.artifact-sources`` : []
       ``ArtifactSourceReader`` plugin references that read artifact import URIs, beside the built-in ``hf``, ``file`` and ``s3`` readers

   ``storage.agent-record-dir`` : .reef/agent-record
       the SQLite scenario storage: records, commits and settings

   ``storage.record-backend`` : sqlite
       where scenario records, commits and settings live: ``sqlite``, ``postgres``, or a ``ScenarioStorage`` plugin reference

   ``storage.record-database-url`` : unset
       the PostgreSQL connection URL, for ``postgres``

   ``storage.record-database-schema`` : reef_records
       the PostgreSQL schema of the scenario storage

   ``storage.coordination-backend`` : follows ``record-backend``
       where the services share their work queues, served models and control lease: ``sqlite`` (processes on one host) or ``postgres`` (``record-database-url``); required when ``record-backend`` is a plugin

   ``storage.work-max-attempts`` : 3
       how many times a worker takes one work item (a training job, a record selection, an evaluation rollout, a data task) before the item fails. An attempt that ends because its service stops does not count

   ``storage.work-retention-s`` : 604800
       seconds that control keeps an ended work item, so that its producer reads the result; then control deletes it

   ``storage.task-file-backend`` : local
       where the files of generated tasks live: ``local``, or a ``TaskFileStore`` plugin reference

   ``storage.task-file-dir`` : .reef/task-files
       the files of generated tasks for ``local``, or the local cache of a plugin store

   ``storage.data-store`` : follows ``record-backend``
       where the deployment's data items live: ``sqlite``, ``postgres`` (``record-database-url``, schema ``reef_data``), or a ``DataStore`` plugin reference

   ``storage.segment-backend`` : unset
       where a SQL data store keeps large content as JSON Lines segments: unset (in the database), ``local`` (``<agent-record-dir>/segments``), or a ``SegmentStore`` plugin reference

   ``storage.job-directory-ttl-s`` : 604800
       seconds that the data store keeps a job data directory that no training job used; a job of the same samples uses the directory again (see `The job data directory <../developer-guide/training-backends.rst#the-job-data-directory>`__)

   ``storage.agent-record-retention-max-bytes`` : 21474836480
       20 GiB shared across all record items of the data store, including unconsumed records

.. warning::

   On ephemeral storage, a restart loses the scenario storage (with SQLite) and every version. See `Back up a deployment <../user-guide/storage.rst#back-up-a-deployment>`__.

**Record retention.** ``storage.agent-record-retention-max-bytes`` is a budget for the stored record bodies. Training does not delete the records that it consumed. The HTTP service runs a cleanup at startup and every 60 seconds.

- **What counts.** The budget measures the stored UTF-8 JSON of payloads, references and artifact references, across active and archived scenarios. SQLite shares the budget across the record directory, and PostgreSQL across the deployment schema. The history and the tools that an agent resends in every model call count once. The model calls of evaluation rollouts are outside the budget and are never evicted.
- **What is evicted.** Over the budget, storage evicts the oldest records by creation time, consumed or not. It keeps the model calls that a stored training sample holds. Records do not expire by age. Each eviction logs a warning with the scenario, record count, sequence range and freed bytes.
- **After an eviction.** The pipeline status shows ``record_data_incomplete``, ``evicted_record_count`` and ``evicted_body_bytes``, and the metrics show ``records/evicted_count`` and ``records/data_incomplete``; the totals survive a restart. Reef skips, with a warning, a pending report whose inputs are no longer stored. Retry hashes and commit metadata survive, so a retried upload neither restores evicted records nor trains them again. A consumer that reads records several times must tolerate missing records.
- **Limits.** The budget is not a filesystem quota. Concurrent writes can exceed it between cleanups, and indexes, retry hashes, commit histories and the WAL need more disk. SQLite reuses freed pages without shrinking its file. A failed cleanup is logged and retried. A standalone store needs an explicit maintenance call; see `Python API <python-api.rst>`__.

The ``data`` section
--------------------

Settings of the data service, which answers the data routes (`Data items <http-api-data-items.rst#http-api-data-items>`__).

.. container:: config

   ``data.max-task-archive-bytes`` : 536870912
       the most bytes of the body of a task archive (512 MiB), for ``POST`` and ``PUT`` on ``/reef/namespaces/{owner}/data/tasks``. A larger body gets HTTP 413. The data service writes the body to a local file before it unpacks it, so the setting limits the disk that one upload uses. A larger task uses the reference form, which sends no files

The ``evaluation`` section
--------------------------

These keys configure the evaluation service.

.. container:: config

   ``evaluation.agent_host`` : host.docker.internal
       the address at which a task's container reaches this host, for an evaluation on ``docker`` whose harness is an artifact item. In a remote environment, the agent bridges the container to this host through Harbor's exec and needs no address. See `HTTP API <http-api.rst>`__.
   ``evaluation.max_sandboxes`` : 4
       the task sandboxes that one process of the evaluation service holds at once, across all evaluations. Each running rollout holds one. Set it to the quota of the sandbox provider, divided by the evaluation processes.
   ``evaluation.on_stop`` : requeue
       how an evaluation worker handles running rollouts when shutdown begins (for example, on SIGTERM).

       - ``requeue`` cancels running rollouts and stops their sandboxes. Another worker can restart them from the beginning. The interruption does not count toward the retry limit.
       - ``finish`` stops taking new rollouts and lets running ones finish.
   ``evaluation.finish_timeout_s`` : unset
       how many seconds a worker using ``finish`` waits after shutdown begins. If unset, it waits indefinitely. Completed rollouts keep their results. When time runs out, the worker requeues only unfinished rollouts. Allow time for sandbox cleanup. If the worker is killed before it can requeue a rollout, that attempt can expire and count toward the retry limit.
   ``evaluation.trial_observer`` : unset
       a ``TrialObserver`` plugin (``reef.evaluation.trials``), as ``package.module:ClassName`` or ``module`` and ``config``. The evaluation worker gives it each trial: the rollout, the attempt, the trial directory and the error. It runs before the worker removes the directory, so it can keep the logs of the agent and the verifier. Its failure is logged and does not change the rollout.
   ``evaluation.task_template_builder`` : unset
       a ``TaskTemplateBuilder`` plugin (``reef.evaluation.task_templates``), as ``package.module:ClassName`` or ``module`` and ``config``. An evaluation on ``e2b`` calls it once for each task, before it adds the task's rollouts, to make the task's sandbox templates ready. Unset, ``DockerfileTemplateBuilder`` builds them from the task's Dockerfiles, as a trial would. A deployment that keeps task images in a registry names a builder that makes the templates from those images.
   ``evaluation.require_prepared_templates`` : false
       Set to ``true`` to require prepared task and verifier templates for new ``e2b`` evaluations. If preparation fails, the affected rollouts report the error and skip the trial.
