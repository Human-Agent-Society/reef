Scenario routing between students and teachers
==============================================

Use Semantic Router (SR) to route a scenario's requests between external models.
For a complete deployment, follow the `Semantic Router tutorial <../../tutorials/semantic-router/README.md>`__.
For client behavior, see `Routed external-model requests <http-api-routing.rst#routed-external-model-requests>`__.

Set ``reef.routing-manifest`` to a manifest file.
Each scenario has a default target and rules that select another target. Clients can also pin a target explicitly.

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

Choose ``managed`` to let Reef run SR, or ``external`` to run SR yourself.
Both require ``recipe.implementation: recipe`` and ``reef.tokens``.

See `Settings reference <configuration-settings.rst#service>`__ for deployment keys.
If a provider aborts a non-streaming request, Reef retries the same target using ``inference.retry-*`` settings.
``inference.retry-timeout-s`` defaults to ``inference.timeout-s``.

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

- Set ``reef.host: 127.0.0.1`` and ``reef.routing-deployment.deployment-id``.
- State is stored under ``${XDG_STATE_HOME:-~/.local/state}/reef/routing/<deployment-id>/``. A lock permits one deployment per user, host and ID.
- ``python -m reef.deploy.routing.cli validate --config <file>`` checks a configuration without Docker.

**Policy updates.** Edit the manifest and send ``SIGHUP`` to ``reef serve``.
This updates Reef and SR without restarting them. Follow the `update procedure <../../tutorials/semantic-router/README.md#change-the-routing-policy>`__.

- An update cannot remove or disable a target that a default or a rule names.
- A change to ``automatic_model``, the Scenario names, the domain classifier, the image or ``reef.routing-deployment`` needs a restart.
- ``api_key_env`` can name only a variable that was set when ``reef serve`` started.
- New requests get 503 during the update. Admitted requests finish on the old endpoints; keep them available.
- Reef keeps the old policy if provider checks fail, SR rejects the change, or request selection exceeds ``drain-timeout-seconds``.
- If the final check fails after SR runs the new policy, admission stays closed. After ``ready-timeout-seconds``, ``reef serve`` stops.
- If ``reef serve`` stops during an update, the next start uses the last committed manifest. Send ``SIGHUP`` again to apply the edited file.

``generations/<generation>/lifecycle-events.jsonl`` in that directory records the results, for example ``policy_committed``, ``policy_rolled_back`` and ``policy_rejected``.

**Shutdown and recovery.** Ctrl-C or SIGTERM drains requests and record writes, then removes SR.
``restart-policy: bounded`` recreates a failed SR container without replaying requests.
Reef failures, unfinished record writes or exhausted recovery attempts stop the deployment.
See `Stop, inspect and clean up <../../tutorials/semantic-router/README.md#stop-inspect-and-clean-up>`__.

Semantic Router image
---------------------

- If the manifest names no image, startup builds it from the SR commit of this Reef release, or reuses an earlier build. The Reef HTTP port opens only after the image is ready.
- If the manifest names an image, Reef builds none.
- The image's ``org.opencontainers.image.revision`` label must match the pinned commit, or Reef starts no container.
- Updates and recovery keep the image of the start.

Records include the image in ``snapshot_version``.
Without a named image, another host or build can produce a different version. In this case, ``validate`` prints ``"snapshot_version": null``.
See the `build requirements <../../tutorials/semantic-router/README.md#1-the-semantic-router-image>`__.

External router lifecycle
-------------------------

Set ``lifecycle: external`` when another system, such as Kubernetes, manages SR.
Run each Reef service separately. Each inference replica needs an SR instance on its own loopback.
Other services ignore the manifest and can share the config.
This lifecycle requires ``--service``.

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

3. Run SR with ``/app/config/config.yaml`` in the replica's network namespace.
   SR listens on ``127.0.0.1:<sr-management-port>``.
   The file has mode 0600. Run SR as the user that wrote it.
4. Run ``reef serve -c /etc/reef/reef.yaml --service inference``.

Use ``GET /readyz`` for readiness (`HTTP API: status <http-api-status.rst>`__).
It returns 200 when SR is active and ready, all enabled targets list their models, and each scenario passes a preview.
Otherwise, it returns 503.

Differences from ``managed``:

- **Policy updates.** Replace replicas to load a new manifest; ``SIGHUP`` is unsupported. During replacement, records identify their policy through ``snapshot_version``.
- **Stopping.** On SIGTERM the replica stops accepting connections, waits up to 60 seconds for requests, cancels the rest, and waits ``finalization-timeout-seconds`` for their records.
- **SR management.** You build, start, restart and stop SR. Reef does not check the image's revision label. ``render`` and the replica require ``deployment.semantic_router_image``. Use an image built from this release's SR commit.
- **Settings.** Only ``sr-management-port``, ``selection-timeout-seconds``, ``preview-max-concurrency`` and ``finalization-timeout-seconds`` apply.
