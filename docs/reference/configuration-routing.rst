Scenario routing between students and teachers
==============================================

Point ``reef.routing-manifest`` to a manifest that selects a default model and routing rules for each scenario.
Clients can also `pin a target <http-api-routing.rst#routed-external-model-requests>`__.

.. code:: yaml

   schema_version: 2
   automatic_model: auto
   deployment: {}
   targets:
     code-student:
       provider_model: code-student
       revision: student-r1
       url: http://127.0.0.1:8101
     code-teacher:
       provider_model: code-teacher
       revision: teacher-r1
       url: http://127.0.0.1:8102
   scenarios:
     code-opd:
       targets: [code-student, code-teacher]
       default_target: code-student
       rules:
         - name: teacher_requested
           keywords: [reef_teacher_required]
           target: code-teacher

Both lifecycles require ``recipe.implementation: recipe`` and ``reef.tokens``.
Reef retries aborted non-streaming requests against the same target using ``inference.retry-*``.
The retry deadline defaults to ``inference.timeout-s``.

Manifest keys
-------------

Use schema version 2. All keys use underscores. Reef rejects unknown keys.

- ``automatic_model`` names the request model that enables automatic routing. It must differ from every target ID.
  Neither it nor a target ID can start with ``reef-entrypoint/``, ignoring case.
- Each target requires ``provider_model``, an immutable ``revision`` and an HTTP ``url`` without credentials, query or fragment.
  Optional fields are ``role``, ``api_key_env`` and ``enabled`` (default true).
- Each scenario declares ``targets`` and a ``default_target``. The first matching rule wins. Default and rule targets must be enabled.
- Each rule requires ``name``, ``target`` and exactly one of ``keywords`` or ``domains``.
  Keyword rules use ``match: any`` or ``all`` and default to ``case_sensitive: false``.
  Domain labels must be unique across rules within a scenario, ignoring case.
- Include ``deployment``, even if empty. Its optional ``semantic_router_image`` accepts a repository digest or a local SHA-256 image ID.
- Domain rules require ``deployment.domain_classifier`` with an absolute, symlink-free ``model_directory`` and a ``threshold`` from 0 to 1.
  Its ``files`` mapping requires exactly the SHA-256 digests of ``config.json``, ``tokenizer.json``, ``model.safetensors`` and ``category_mapping.json``, each prefixed with ``sha256:``.

Managed routing lifecycle
-------------------------

Managed routing requires Linux, local Docker with Compose, and the local ``uni`` service executor.
Start the model services yourself. ``reef serve`` starts SR and Reef; ``--service`` is unsupported with this lifecycle.

.. code:: yaml

   reef:
     host: 127.0.0.1
     tokens: ["change-me"]
     routing-manifest: /path/to/manifest.yaml
     routing-deployment:
       deployment-id: code-opd-local
   recipe:
     implementation: recipe
   inference:
     upstream-url: http://127.0.0.1:8101
     upstream-model: code-student

Bind Reef to loopback. Each deployment ID must be unique per user and host.
Validate the config with ``python -m reef.deploy.routing.cli validate --config <file>``.

To apply rule or target edits, send ``SIGHUP`` to ``reef serve``. New requests receive 503 while admitted requests drain on the old endpoints.
Keep those endpoints available. Referenced targets must remain enabled, and credential variables must exist at startup.
Changes to ``automatic_model``, scenario names, classifier, image or deployment settings require a restart.

If provider checks, SR updates or draining fail, Reef keeps the old policy.
If the final check fails after SR changes policies, admission stays closed until Reef stops at ``ready-timeout-seconds``.
After an interrupted update, restart from the last committed policy and send ``SIGHUP`` again.
For state paths, events and recovery, follow the `operator procedure <../../tutorials/semantic-router/README.md#change-the-routing-policy>`__.

Ctrl-C and SIGTERM drain requests and record writes before removing SR.
Bounded restarts recreate failed SR containers without replaying requests. A Reef failure, unfinished writes or exhausted retries stop the deployment.

Semantic Router image
---------------------

Without a pinned image, Reef builds or reuses one from this release's SR commit before opening HTTP.
With an explicit image, Reef verifies its ``org.opencontainers.image.revision`` label against that commit.
Policy updates and recovery reuse the startup image.

The image contributes to each record's ``snapshot_version``. Without a pinned image, ``validate`` returns a null version and different builds can produce different versions.
See the `build requirements <../../tutorials/semantic-router/README.md#1-the-semantic-router-image>`__.

External router lifecycle
-------------------------

Set ``reef.routing-deployment.lifecycle: external`` and run each service with ``--service``.
Supply ``deployment.semantic_router_image`` built from this release's SR commit.
Each inference replica needs SR on its own loopback. Other services can share the config.
The host can bind any address. A deployment ID is optional.

To start a replica:

1. Set ``REEF_SR_MANAGEMENT_TOKEN`` and ``REEF_SR_SELECTION_TOKEN`` to secret values, for SR and for the replica.
2. Write the SR configuration before SR starts:

   .. code:: bash

      python -m reef.deploy.routing.cli render \
        --config /etc/reef/reef.yaml --output /app/config/config.yaml

3. Run SR with ``/app/config/config.yaml`` in the replica's network namespace.
   SR listens on ``127.0.0.1:<sr-management-port>``.
   The file has mode 0600. Run SR as the user that wrote it.
4. Run ``reef serve -c /etc/reef/reef.yaml --service inference``.

``GET /readyz`` returns 200 when SR is ready, enabled targets list their models and each scenario passes a preview. Otherwise it returns 503.

You own SR startup, recovery and image verification. Replace replicas to change policy.
During replacement, ``snapshot_version`` identifies the policy that served a request.
On SIGTERM a replica drains for up to 60 seconds, cancels remaining requests, then waits for record finalization.

Only ``sr-management-port``, ``selection-timeout-seconds``, ``preview-max-concurrency`` and ``finalization-timeout-seconds`` apply in external mode.
