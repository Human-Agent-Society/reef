Scenario model configuration
============================

A scenario may replace the deployment's inference model with an endpoint,
model name and protocol. The engine uses the same model clients for
both deployment defaults and scenario settings. Reef calls the scenario's
endpoint with the deployment's provider key (``inference.upstream-api-key``),
the only credential that reaches an engine. A scenario setting holds no key.
No provider discovery endpoint or resolver service is required.

Creating and updating
---------------------

Use the existing authenticated scenario API::

    POST /reef/scenarios
    Authorization: Bearer <service-token>
    Content-Type: application/json

    {
      "name": "my-harness",
      "model": {
        "url": "https://provider.example",
        "model": "my-model",
        "api": "openai"
      }
    }

``api`` accepts ``openai`` (the default), ``responses`` or ``anthropic``,
using the existing inference proxy protocols. ``url`` and ``model`` are
required. Other fields, such as ``api_key``, are refused with HTTP 400. The
endpoint URL cannot include user information, a query or a
fragment. Settings replace the full model configuration, rather than merging
individual fields.

Creation is idempotent: creating an existing scenario leaves its model
settings intact. Change them through
``POST /reef/scenarios/{scenario}/update`` with ``{"model": {...}}``.
``{"model": null}`` explicitly restores deployment defaults. The response
acknowledges the applied URL, model and protocol. An invalid update leaves
the current settings unchanged.
Model overrides require an inference-only runtime, since a training runtime
owns the weights it serves.

The storage backend keeps the settings with the records and commits of the
scenario. SQLite keeps them in the scenario database file, which only its
owner can read. PostgreSQL keeps them in the deployment schema. Without a durable record directory, settings last for the process
lifetime. Invalid saved settings fail scenario recovery. They do not silently
select the deployment model. Scenario deletion archives the settings with the
records.
See `Choose the storage backend <operate.rst#choose-the-storage-backend>`__.

Harness evolution
-----------------

A harness evolution step captures one model configuration before proposing.
It replaces ``models.served`` and each named binding (including teacher and
judge) with that endpoint. The proposer, baseline and candidate rollouts,
gates and model-based scoring share the captured bindings, including across
worker processes. A configuration change affects new inference requests and
new steps. An active step continues with its captured settings; a revoked
provider key fails without falling back to another provider.

Use the supplied bindings for every model call. A model-based judge is an
``RolloutScorer`` that overrides ``score_with_models(task, result, models)``::

    class Judge(RolloutScorer):
        def score(self, task, result):
            raise ValueError("the judge needs the step's model bindings")

        def score_with_models(self, task, result, models):
            verdict = models["judge"].chat([
                {"role": "user", "content": "Evaluate the rollout: " + str(result.trajectory)}
            ])
            return float(verdict.strip() == "pass")

A deterministic scorer implements ``score(task, result)`` only. Methods that construct clients from environment credentials must use
these bindings to participate in scenario model selection.

Install scripts use the current scenario protocol and model while pointing
clients through Reef. After changing protocol or model name, reinstall the
harness to refresh its client configuration.

Platform deployments
--------------------

Reef API Platform supplies a scoped proxy endpoint as ordinary model
settings. The vendor key remains encrypted in the platform. Standalone Reef
operators can supply their model endpoint directly; Reef calls it with the
deployment's provider key.

When upgrading from the platform resolver integration, pause new work,
upgrade both services and re-save every configured scenario before resuming
or recovering queued steps. Existing resolver configuration is not imported.
The old provider capability route, resolver environment variables and provider
version header have been removed.
