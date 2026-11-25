Scenario routing between students and teachers
==============================================

This page gives the routing manifest and the two ways for operators to run the Semantic Router (SR). The `Semantic Router tutorial <../../tutorials/semantic-router/README.md>`__ shows a complete deployment; `Routed external-model requests <http-api-routing.rst#routed-external-model-requests>`__ gives the request contract.

In each Scenario of the manifest that ``reef.routing-manifest`` names, a request goes to the default target (usually the student), or to the target of a rule that it matches (usually the teacher). A request can also pin one of the Scenario's targets.

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

Both lifecycles need ``recipe.implementation: recipe`` and ``reef.tokens``.

``reef.routing-manifest`` and the keys of ``reef.routing-deployment`` are in the `Settings reference <configuration-settings.rst#service>`__. If a provider aborts a non-streaming request, Reef retries the same target, with the ``inference.retry-*`` delays and deadline (`Settings reference <configuration-settings.rst#inference>`__). An unset ``inference.retry-timeout-s`` takes the value of ``inference.timeout-s``.

Manifest keys
-------------

The manifest accepts only these keys, with underscores. An unknown key is an error.

.. container:: config

   ``schema_version``
       Required. Must be 2.

   ``automatic_model``
       Required. The ``model`` value of a request for which SR selects the target. It must differ from every target ID.

   ``targets.<id>.provider_model``
       Required. The model ID that the provider serves.

   ``targets.<id>.url``
       Required. The provider's HTTP endpoint, without credentials, query or fragment.

   ``targets.<id>.revision``
       Required. The immutable revision of the external deployment.

   ``targets.<id>.role``
       A label, for example ``student``.

   ``targets.<id>.enabled`` : true
       With false, no request goes to the target. Each default target and rule target must be enabled.

   ``targets.<id>.api_key_env``
       The environment variable that holds the provider credential.

   ``scenarios.<name>.targets``
       Required. The IDs of the Scenario's targets.

   ``scenarios.<name>.default_target``
       Required. The target of a request that matches no rule.

   ``scenarios.<name>.rules``
       A list. The first matching rule is used.

   ``rules[].name``
       Required.

   ``rules[].target``
       Required. The target of a matching request.

   ``rules[].keywords``
       A list of keywords. A rule has exactly one of ``keywords`` and ``domains``.

   ``rules[].domains``
       A list of classifier labels. In one Scenario, a label can occur in only one rule, ignoring case.

   ``rules[].match`` : any
       ``any`` matches a request with one of the keywords, ``all`` a request with all of them.

   ``rules[].case_sensitive`` : false
       With true, keywords match only in the same letter case.

   ``deployment``
       Required, also when empty (``deployment: {}``).

   ``deployment.semantic_router_image``
       The SR image, as ``repository@sha256:<hex>`` or a local image ID ``sha256:<hex>``. See `Semantic Router image`_.

   ``deployment.domain_classifier.model_directory``
       The absolute path, without symlinks, of the local classifier. Domain rules need ``domain_classifier`` with its three keys.

   ``deployment.domain_classifier.threshold``
       A number from 0 to 1.

   ``deployment.domain_classifier.files``
       The ``sha256:`` digest of each of ``config.json``, ``tokenizer.json``, ``model.safetensors`` and ``category_mapping.json``, and of no other file.

Target IDs and ``automatic_model`` must not start with ``reef-entrypoint/``, in any letter case.

Managed routing lifecycle
-------------------------

``reef serve`` without ``--service`` runs the SR container and a routed Reef child. ``reef serve --service`` refuses a managed manifest. You need Linux, a local Docker daemon with Compose, and the local ``uni`` service executor. You run the model services yourself.

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

- Set ``reef.host: 127.0.0.1`` and ``reef.routing-deployment.deployment-id``.
- Reef locks ``${XDG_STATE_HOME:-~/.local/state}/reef/routing/<deployment-id>/``, so a user runs one deployment per ID on a host.
- ``python -m reef.deploy.routing.cli validate --config <file>`` checks a configuration without Docker.

**Policy updates.** Edit the manifest and send ``SIGHUP`` to ``reef serve`` (`procedure <../../tutorials/semantic-router/README.md#change-the-routing-policy>`__). Reef and SR do not restart.

- An update cannot remove or disable a target that a default or a rule names.
- A change to ``automatic_model``, the Scenario names, the domain classifier, the image or ``reef.routing-deployment`` needs a restart.
- ``api_key_env`` can name only a variable that was set when ``reef serve`` started.
- New requests get 503 during the update. Admitted requests finish on the old endpoints; keep them available.
- Reef keeps the old policy if the provider check fails, if SR refuses the change, or if a request still selects after ``drain-timeout-seconds``.
- If the final check fails after SR runs the new policy, admission stays closed. After ``ready-timeout-seconds``, ``reef serve`` stops.
- If ``reef serve`` stops during an update, the next start uses the last committed manifest. Send ``SIGHUP`` again to apply the edited file.

``generations/<generation>/lifecycle-events.jsonl`` in that directory records the results, for example ``policy_committed``, ``policy_rolled_back`` and ``policy_rejected``.

**Shutdown and recovery.** Ctrl-C or SIGTERM drains requests and record writes, then removes the SR container. With ``restart-policy: bounded``, Reef recreates a failed SR container without replaying requests. A Reef failure, unfinished record writes or no remaining recovery attempts stop the deployment. See `Stop, inspect and clean up <../../tutorials/semantic-router/README.md#stop-inspect-and-clean-up>`__.

Semantic Router image
---------------------

- If the manifest names no image, startup builds it from the SR commit of this Reef release, or reuses an earlier build. The Reef HTTP port opens only after the image is ready.
- If the manifest names an image, Reef builds none.
- The image's ``org.opencontainers.image.revision`` label must match the pinned commit, or Reef starts no container.
- Updates and recovery keep the image of the start.

The ``snapshot_version`` of the records includes the image. Without a named image, the same manifest gets another version on another host or after a new build, and ``validate`` prints ``"snapshot_version": null``. The tutorial gives the `build requirements <../../tutorials/semantic-router/README.md#1-the-semantic-router-image>`__.

External router lifecycle
-------------------------

Use ``lifecycle: external`` when each Reef service runs in its own process and another system, such as Kubernetes, runs SR. Each ``inference`` replica uses one SR on its own loopback. The other services ignore the manifest, so all can read one configuration. ``reef serve`` without ``--service`` refuses this lifecycle.

.. code:: yaml

   reef:
     host: 0.0.0.0
     tokens: ["change-me"]
     routing-manifest: /etc/reef/manifest.yaml
     routing-deployment:
       lifecycle: external
   recipe:
     implementation: recipe
   inference:
     upstream-url: http://127.0.0.1:8101
     upstream-model: code-student

This lifecycle needs no ``deployment-id``, and ``reef.host`` can have any value.

To start a replica:

1. Set ``REEF_SR_MANAGEMENT_TOKEN`` and ``REEF_SR_SELECTION_TOKEN`` to secret values, for SR and for the replica.
2. Write the SR configuration before SR starts:

   .. code:: bash

      python -m reef.deploy.routing.cli render \
        --config /etc/reef/reef.yaml --output /app/config/config.yaml

3. Run SR with ``/app/config/config.yaml`` as its argument, in the replica's network namespace. SR listens on ``127.0.0.1:<sr-management-port>``. The file has mode 0600, so run SR as the user that wrote it.
4. Run ``reef serve --service inference``.

Use ``GET /readyz`` as the readiness check (`HTTP API: status <http-api-status.rst>`__). It answers 200 when SR reports its runtime as active and ready, every enabled target lists its model, and one preview per Scenario passes. Otherwise it answers 503.

Differences from ``managed``:

- **Policy updates.** The replica reads the manifest once and has no ``SIGHUP`` update. To change the policy, replace the replicas. During a rollout, two policies can serve; the ``snapshot_version`` of a record shows which one.
- **Stopping.** On SIGTERM the replica stops accepting connections, waits up to 60 seconds for requests, cancels the rest, and waits ``finalization-timeout-seconds`` for their records.
- **Your responsibilities.** Reef does not build, start, restart or stop SR, and does not check the image's revision label. ``render`` and the replica require ``deployment.semantic_router_image``. Name an image built from the SR commit of this Reef release.
- **Settings.** Only ``sr-management-port``, ``selection-timeout-seconds``, ``preview-max-concurrency`` and ``finalization-timeout-seconds`` apply.
