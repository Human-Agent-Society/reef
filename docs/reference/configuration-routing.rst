Scenario routing between students and teachers
==============================================

Set ``reef.routing-manifest`` to a version 2 `routing manifest <../../tutorials/semantic-router/manifest.example.yaml>`__.
Use the record-only recipe (``recipe.implementation: recipe``) and set ``reef.tokens``.
Start each model endpoint before starting Reef.

In the manifest:

- Give each target a model name, immutable revision and HTTP URL. Use ``api_key_env`` for credentials.
- List each scenario's allowed targets and default target. The first matching keyword or domain rule wins.
  Only enabled targets can serve as defaults or rule targets.
- Set ``automatic_model`` to a name distinct from all targets. A client can request it or `pin a target <http-api-routing.rst>`__.
- Include ``deployment`` even if empty. For domain rules, follow the `classifier setup <../../tutorials/semantic-router/README.md#route-by-a-pretrained-domain-classifier>`__.

Managed routing lifecycle
-------------------------

Use Linux, Docker Compose and the local ``uni`` service executor. Bind Reef to loopback.
Give each deployment a unique ``reef.routing-deployment.deployment-id`` per user and host.

.. code:: bash

   python -m reef.deploy.routing.cli validate --config reef.yaml
   reef serve -c reef.yaml

Reef starts Semantic Router. To change targets or rules, keep the old endpoints available and send ``SIGHUP`` to Reef.
New requests receive 503 while admitted requests drain. Image, classifier and deployment changes require a restart.
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

Replace replicas to change policy. Each record's ``snapshot_version`` identifies the policy that served it.
