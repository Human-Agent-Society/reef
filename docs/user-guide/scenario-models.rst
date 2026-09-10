Scenario model configuration
============================

A scenario may replace the deployment's inference model with an endpoint,
API key, model name and protocol. The engine uses the same model clients for
both deployment defaults and scenario settings; billing stays with the caller.
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
        "api_key": "<model-key>",
        "model": "my-model",
        "api": "openai"
      }
    }

``api`` accepts ``openai`` (the default), ``responses`` or ``anthropic``,
using the existing inference proxy protocols. ``api_key`` is optional for
endpoints that do not require authentication. ``url`` and ``model`` are
required. The endpoint URL cannot include user information, a query or a
fragment. Settings replace the full model configuration, rather than merging
individual fields.

Creation is idempotent: creating an existing scenario leaves its model
settings intact. Change them through
``POST /reef/scenarios/{scenario}/update`` with ``{"model": {...}}``.
``{"model": null}`` explicitly restores deployment defaults. The response
acknowledges the applied URL, model, protocol and ``has_api_key``; it never
returns the key. An invalid update leaves the current settings unchanged.
Model overrides require an inference-only runtime, since a training runtime
owns the weights it serves.

Settings are atomically saved beside the scenario record store in a private
file with mode ``0600``. Keep that directory when restarting a deployment.
Without a durable record directory, settings last for the process lifetime.
Malformed saved settings fail scenario recovery. They do not silently select
the deployment model. Scenario deletion archives this file with its records.
Keys never enter published artifacts, algorithm state or API receipts.

Harness evolution
-----------------

A harness evolution step captures one model configuration before proposing.
It replaces ``models.served`` and each named binding (including teacher and
judge) with that endpoint. The proposer, baseline and candidate episodes,
gates and model-based scoring share the captured bindings, including across
worker processes. A configuration change affects new inference requests and
new steps. An active step continues with its captured settings; a revoked
key fails without falling back to another provider.

Use the supplied bindings for every model call. A callable judge can declare
the ``models`` keyword::

    def evaluate(task, result, *, models):
        verdict = models["judge"].chat([
            {"role": "user", "content": "Evaluate the episode: " + str(result.trajectory)}
        ])
        return float(verdict.strip() == "pass")

An ``EpisodeScorer`` subclass overrides ``score_with_models(task, result,
models)``. Deterministic scorers with ``evaluate(task, result)`` continue to
work. Methods that construct clients from environment credentials must use
these bindings to participate in scenario model selection.

Install scripts use the current scenario protocol and model while pointing
clients through Reef. After changing protocol or model name, reinstall the
harness to refresh its client configuration.

Platform deployments
--------------------

Reef API Platform supplies a scoped proxy endpoint and key as ordinary model
settings. The vendor key remains encrypted in the platform. Standalone Reef
operators can supply their model endpoint and key directly.

When upgrading from the platform resolver integration, pause new work,
upgrade both services and re-save every configured scenario before resuming
or recovering queued steps. Existing resolver configuration is not imported.
The old provider capability route, resolver environment variables and provider
version header have been removed.
