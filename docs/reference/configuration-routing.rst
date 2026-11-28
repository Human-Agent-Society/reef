Scenario routing between students and teachers
==============================================

Set ``reef.routing-manifest`` to a version 2 `routing manifest <../../tutorials/semantic-router/manifest.example.yaml>`__.
Use the record-only recipe (``recipe.implementation: recipe``) and set ``reef.tokens``.
Start the model endpoints used for primary requests before starting Reef.

In the manifest:

- Give each target a model name, immutable revision and HTTP URL. Use ``api_key_env`` for credentials.
- List each scenario's allowed targets and default target. The first matching keyword or domain rule wins.
  Defaults, rule targets and optional shadow targets must be enabled.
- Set ``automatic_model`` to a name distinct from all targets. A client can request it or `pin a target <http-api-routing.rst>`__.
- Include ``deployment`` even if empty. For domain rules, follow the `classifier setup <../../tutorials/semantic-router/README.md#route-by-a-pretrained-domain-classifier>`__.

Shadow targets
--------------

Set ``shadow_target`` on a scenario to send the request to a second model in the background.
It can name a catalog target outside the scenario's list of primary targets. If it matches the selected
primary, Reef calls the model once. The client receives only the primary response and receipt.

Reef sends at most one nonstreaming shadow request, even when the primary streams or retries.
A shadow timeout or failure leaves the primary response intact. Reef skips the shadow when all slots
are busy. Each slot stays occupied through the record write. The per-process concurrency, timeout
and response-size limits are in the `settings reference <configuration-settings.rst#service>`__.

Readiness skips providers used only for shadows. Their configured credential variables must still be set.
Changing only ``shadow_target`` updates Reef's policy. Adding a new catalog target also updates the
Semantic Router configuration. See `routing records <http-api-routing.rst#routed-external-model-requests>`__
for how to pair the responses.

Managed routing lifecycle
-------------------------

Use Linux, Docker Compose and the local ``uni`` service executor. Bind Reef to loopback.
Give each deployment a unique ``reef.routing-deployment.deployment-id`` per user and host.

.. code:: bash

   python -m reef.deploy.routing.cli validate --config reef.yaml
   reef serve -c reef.yaml

Reef starts Semantic Router. To change targets or rules, keep the old endpoints available and send ``SIGHUP`` to Reef.
New requests receive 503 while Reef waits for admitted requests to finish selecting their targets and publishes the new policy.
Primary and shadow calls that already selected their targets can finish on the old endpoints.
A full deployment drain also waits for running shadows and their records. A forced drain cancels provider calls and gives pending writes the finalization timeout.
Image, classifier and deployment changes require a restart.
See the `operator guide <../../tutorials/semantic-router/README.md#change-the-routing-policy>`__ for recovery and image requirements.

External router lifecycle
-------------------------

Set ``reef.routing-deployment.lifecycle: external``. Supply a Semantic Router image built from this Reef release's pinned commit.
Each inference replica needs its own router on loopback.

#. Set ``REEF_SR_MANAGEMENT_TOKEN`` and ``REEF_SR_SELECTION_TOKEN``.
#. Generate the config and start the router in the replica's network namespace. Run it as the user who generated the file.

   .. code:: bash

      python -m reef.deploy.routing.cli render --config reef.yaml --output /app/config/config.yaml

#. Start ``reef serve -c reef.yaml --service inference``. Wait for ``GET /readyz`` to return 200 before sending requests.

Replace replicas to change policy, including ``shadow_target``. Each record's ``snapshot_version`` identifies the policy that served it.
Stop and drain a scenario's replicas before archiving or deleting it. On shutdown a replica gives primary requests up to 60 seconds to finish,
then cancels remaining primary and shadow calls and allows their records to finish within ``finalization-timeout-seconds``.
