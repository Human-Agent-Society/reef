Scenario routing between students and teachers
==============================================

The routing manifest and the ``reef.routing_deployment`` keys. ``reef.routing_manifest`` (CLI: ``--reef.routing-manifest``) routes each Scenario's ``/v1/chat/completions`` requests, ordinary and SSE, among external targets. It is unset by default. Each configured Scenario serves its default target (usually the student). A request that matches a keyword or domain rule goes to another target (usually the teacher). A request can also pin one of the Scenario's targets. The `Semantic Router tutorial <../../tutorials/semantic-router/README.md>`__ shows a complete deployment, and `Routed external-model requests <http-api-routing.rst#routed-external-model-requests>`__ gives the request and record contract.

Requirements:

- Select ``recipe.implementation: recipe``. Routing records inference and feedback; it does not train the routed models.
- Set ``reef.tokens`` and ``reef.host: 127.0.0.1``.
- Clients call Reef with a service token and ``x-reef-scenario``. They may not send Reef's or the router's selection headers.

For ``model: auto``, Reef asks an unmodified Semantic Router (SR) for one preview and checks the selected rule and target against the manifest. A pin is checked locally and never calls SR. If a provider aborts a buffered request, Reef retries the same target:

.. container:: config

   ``inference.retry-initial-s`` : 0.05
       the first retry delay of a routed request, in seconds

   ``inference.retry-max-s`` : 1.0
       the longest retry delay, in seconds

   ``inference.retry-timeout-s`` : ``inference.timeout-s``
       the deadline of the retries, in seconds

The manifest (``schema_version: 2``) has three parts:

* ``targets`` maps target IDs to provider model IDs, endpoints and immutable external deployment revisions. An optional ``api_key_env`` names the environment variable of the provider credential.
* ``scenarios`` lists, for each Scenario, its ``targets``, its ``default_target`` and optional ``rules``. Each rule has a ``name``, a ``target`` and exactly one of ``keywords`` or ``domains``. If several rules match, the first one listed is used.

  - A keyword rule matches a request whose text contains any of its keywords (``match: any``, the default) or all of them (``match: all``). It ignores case unless ``case_sensitive: true``. A keyword of letters, digits or underscores matches only as a whole word. SR checks the latest user message that has text, and never tool results (`How SR checks keywords <../../tutorials/semantic-router/README.md#how-sr-checks-keywords>`__).
  - A domain rule matches when the deployment's domain classifier predicts one of its labels. It needs ``deployment.domain_classifier``. Within one Scenario, a label may appear in only one rule, ignoring case.

* ``deployment`` holds two optional keys.

  - ``semantic_router_image`` names the SR image by repository digest (``repository@sha256:<hex>``) or local image ID (``sha256:<hex>``). Without it, a managed deployment builds the pinned image (`Semantic Router image`_).
  - ``domain_classifier`` names a local pretrained domain classifier: ``model_directory`` (an absolute path without symlinks), ``threshold`` (a number from 0 to 1) and ``files``, the ``sha256:`` digest of each of exactly ``config.json``, ``tokenizer.json``, ``model.safetensors`` and ``category_mapping.json``. Reef checks the files and labels and mounts only these four files, read-only, into SR. See the `domain classifier section <../../tutorials/semantic-router/README.md#route-by-a-pretrained-domain-classifier>`__.

Target IDs and ``automatic_model`` must not start with ``reef-entrypoint/``, in any letter case: that prefix names the private router entrypoints.

``reef.routing_deployment.lifecycle`` selects who runs SR: ``managed`` (the default, `Managed routing lifecycle`_) or ``external`` (`External router lifecycle`_).

Managed routing lifecycle
-------------------------

With ``reef.routing_manifest`` set, ``reef serve`` runs the SR container and a routed Reef child. ``reef serve --component`` refuses a managed manifest. The lifecycle needs Linux, a local Docker daemon with Compose, and the local ``uni`` service executor. You run the model services yourself.

.. code-block:: yaml

   reef:
     host: 127.0.0.1
     routing_manifest: /path/to/input-manifest.yaml
     routing_deployment:
       deployment_id: code-opd-local
       restart_policy: fail_stop

These keys go under ``reef.routing_deployment``. An unknown key is an error. YAML accepts underscores or hyphens. The CLI takes the complete object through ``--reef.routing-deployment``.

.. list-table::
   :header-rows: 1

   * - Setting
     - Default
     - Meaning
   * - lifecycle
     - managed
     - ``managed``: this section. ``external``: see `External router lifecycle`_.
   * - deployment_id
     - required with routing_manifest
     - 1-48 lowercase letters, digits or hyphens; must start with a letter or digit.
   * - restart_policy
     - fail_stop
     - ``bounded`` permits a limited number of SR container recoveries.
   * - max_restarts
     - 2
     - Attempts per explicit deployment; failed startup attempts count and the budget never resets automatically.
   * - restart_backoff_seconds
     - [1.0, 2.0]
     - Delay before each recovery; later attempts reuse the last delay.
   * - ready_timeout_seconds
     - 120
     - Complete startup or recovery readiness deadline.
   * - drain_timeout_seconds
     - 30
     - Deadline for accepted requests, including uploads, and local response sends to finish.
   * - finalization_timeout_seconds
     - 5
     - Additional deadline for record writes after cancellation.
   * - container_stop_timeout_seconds
     - 10
     - Container stop operation timeout.
   * - image_build_timeout_seconds
     - 3600
     - Longest startup build of the SR image, when the manifest names none and no built image is cached.
   * - sr_management_port
     - 50053
     - SR's authenticated HTTP API (previews and configuration), published on host loopback.
   * - selection_timeout_seconds
     - 10
     - Whole seconds SR may spend on one preview; Reef waits five seconds longer for SR's reply.
   * - preview_max_concurrency
     - 32
     - Concurrent SR previews; more are refused with 429. Changes need a restart.
   * - sr_cpus / sr_memory_bytes
     - 4 / 4294967296
     - SR CPU and memory limits.

**Startup.** The container publishes only SR's HTTP API, on ``127.0.0.1:<sr_management_port>``. Reef generates SR's credentials at each start, and they never reach clients. Reef opens admission after SR reports the generated runtime as active and ready, the providers answer, and one preview per Scenario passes. ``/healthz`` stays a liveness check; it does not express routing readiness.

**Request limits.** Reef accepts request bodies up to 64 MiB, and a body must finish uploading within 30 seconds (408 otherwise). The preview body of an automatic request must fit SR's 10 MiB limit (413 otherwise); the same body as a pin is served. The tutorial lists every `error <../../tutorials/semantic-router/README.md#errors>`__.

**Policy updates.** Edit the configured ``reef.routing_manifest`` and send ``SIGHUP`` to the ``reef serve`` process. Reef and SR do not restart. `Change the routing policy <../../tutorials/semantic-router/README.md#change-the-routing-policy>`__ gives the procedure. The rules:

- An update can change targets (add, remove, enable, disable, or another provider model, revision or endpoint), rules, a Scenario's default target and its target membership. It cannot remove or disable a target that a default or a rule still names.
- The automatic model, the Scenario names, the domain classifier, the image, the deployment settings and the preview settings need a restart. An update never builds an image.
- A target's ``api_key_env`` can name only a variable that the Reef child had when it started.
- Reef first checks the new manifest's providers. A rejected update is logged as ``policy_rejected``, and the current policy keeps serving.
- During the update, new inference requests receive HTTP 503. Requests that already selected a provider finish on their original endpoint, so keep replaced model endpoints available until they finish.
- If an admitted request is still selecting when ``drain_timeout_seconds`` expires, or SR rejects the change, Reef reopens the old policy. Once SR runs the new configuration, a failed final check keeps admission closed; Reef does not return to the old policy.
- If ``reef serve`` stops while an update is pending, the next start renders the last committed manifest, not the edited source file, and records ``pending_policy_recovered``. Send ``SIGHUP`` again to apply the source.
- A ``SIGHUP`` during startup or recovery is applied once the deployment is ready. One that arrives after a stop began is ignored.

The generation's ``lifecycle-events.jsonl`` records the results (``policy_committed``, ``policy_rolled_back``, ``policy_rejected``).

**Shutdown and recovery.** Ctrl-C or SIGTERM closes acceptance, drains accepted requests and pending record writes within the configured deadlines, and then removes the container and its anonymous volumes. New inference requests receive 503 during the drain. ``bounded`` recovery recreates the SR container without replaying requests or restarting Reef or the model services. A Reef failure, unresolved record writes or exhausted recovery attempts stop the deployment.

**Ownership.** Reef records ownership under ``${XDG_STATE_HOME:-~/.local/state}/reef/routing/`` and locks it per deployment ID, so a different run directory does not permit a second owner. Reef removes only the containers and networks that it owns, and keeps records, model caches and provider containers. A killed parent may leave the SR container behind until the next start or an explicit cleanup. An interrupted Docker operation blocks the next start until the operator acknowledges it. See `Stop, inspect and clean up <../../tutorials/semantic-router/README.md#stop-inspect-and-clean-up>`__.

Semantic Router image
---------------------

``deployment.semantic_router_image`` is optional for a managed deployment.

* **The manifest names no image.** Startup builds the image from the SR commit pinned by this release, with ``docker buildx build``, and tags it ``reef-semantic-router:<commit prefix>-<recipe>``. Later starts reuse it. ``image_build_timeout_seconds`` bounds the build, and the Reef HTTP port is not open until the image is ready.
* **The manifest names an image.** Reef runs that image and builds none.

In both cases, the image's ``org.opencontainers.image.revision`` label must match the pinned commit, or Reef creates no container. Reef removes no image and no build cache. The tutorial's `Semantic Router image <../../tutorials/semantic-router/README.md#1-the-semantic-router-image>`__ section gives the build's requirements and limits.

Startup reads the manifest once and fixes the image for the generation. Policy updates keep the image, and recovery restarts it without a build. The generation's ``snapshot_version`` covers the image. Without a named image, the same manifest therefore gets another version on another host, or after a rebuild, and ``python -m reef.deploy.routing.cli validate`` prints ``"snapshot_version": null``. Name an image when hosts with the same Reef release and settings must get the same version.

External router lifecycle
-------------------------

Use ``reef.routing_deployment.lifecycle: external`` when each Reef service runs in its own process and another system, such as Kubernetes, runs SR. Reef then needs no Docker. Each replica of the ``inference`` service uses one SR on its own loopback. Every service can read the same configuration, because the other services ignore the manifest. ``reef serve`` without ``--component`` refuses this lifecycle.

.. code-block:: yaml

   reef:
     routing_manifest: /etc/reef/manifest.yaml
     routing_deployment:
       lifecycle: external
       sr_management_port: 50053
   recipe:
     implementation: recipe
   inference:
     # The launcher needs a reachable default; routed requests use the manifest's targets.
     upstream_url: http://127.0.0.1:8101
     upstream_model: code-student

To start a replica:

1. Set ``REEF_SR_MANAGEMENT_TOKEN`` and ``REEF_SR_SELECTION_TOKEN`` to two different secrets, for both SR and the replica.
2. Write SR's configuration, before SR starts:

   .. code-block:: bash

      python -m reef.deploy.routing.cli render \
        --config /etc/reef/reef.yaml --output /app/config/config.yaml

3. Run SR with ``/app/config/config.yaml`` as its argument, in the replica's network namespace. SR listens on ``127.0.0.1:<sr_management_port>``. The render step writes the file with mode 0600, so run SR as the user that wrote it.
4. Run ``reef serve --component inference``. Clients reach it on ``reef.host``.

Use ``GET /readyz`` as the replica's readiness check. It needs no token. It answers 200 after SR reports its generated runtime as active and ready, every enabled target lists its model, and one preview per Scenario passes. It answers 503 otherwise, including for requests of Scenarios outside the manifest. The replica reads SR every five seconds and checks a new runtime hash in the same way before requests use it.

This lifecycle differs from ``managed``:

* **Policy updates.** The replica reads the manifest once and has no ``SIGHUP`` update. To change the policy, replace the replicas. During a rollout, two policies can serve at the same time. Each record's ``snapshot_version`` shows which one served it.
* **Scope.** A manifest Scenario's ``POST /v1/chat/completions`` follows the routing rules. Every other Scenario and route keeps the ordinary ``inference`` behavior. Clients must call the ``inference`` service, because the other services do not route.
* **Stopping.** On SIGTERM the replica stops accepting connections and waits up to 60 seconds for requests. It then cancels the rest and waits ``finalization_timeout_seconds`` for their records. Give the process more time than this before it is killed.
* **Your responsibilities.** Reef does not build, start, restart or stop SR, and it does not check the image's revision label. ``render`` and the replica both require ``deployment.semantic_router_image``. Name an image built from this Reef release's SR commit; one that a managed start built qualifies.
* **Settings.** Only ``sr_management_port``, ``selection_timeout_seconds``, ``preview_max_concurrency`` and ``finalization_timeout_seconds`` apply. Set SR's CPU and memory limits where you run it.
