Harness evolve with a custom model provider
==========================================

A ``harness_evolve`` deployment can resolve each scenario's model provider
through Reef API Platform. BYOK covers inference, proposer and named auxiliary
models, model-based judges, evaluation episodes, and gate calls. The platform
holds the encrypted provider credential and forwards model calls with the
user's key. Reef receives a revocable scenario/version-scoped proxy credential.

Operator setup
--------------

Deploy the platform's BYOK API and this runtime together. Configure the Reef
service with the platform deployment ID and its registered service token::

    REEF_BYOK_RESOLVER_URL=https://api.reefinfra.ai/api/internal/deployments/<id>/provider
    REEF_BYOK_RESOLVER_TOKEN=<registered Reef service token>

The recipe must be a ``CordisRecipe`` named ``harness_evolve``. Without these
environment settings, existing deployment-wide model behavior is unchanged.
The authenticated ``GET /reef/providers/capabilities`` endpoint advertises
``harness_evolve_byok_v1`` and the configured resolver URL. The platform verifies
both before accepting user settings.

Allow the service and episode sandbox to reach the platform origin. User
providers are contacted by the platform, which validates the URL and DNS
addresses, pins the connection address and refuses redirects. Provider secrets
stay out of the published harness and algorithm state. Both episode executors
use the existing minimal inherited environment, without platform model keys.

Method integration
------------------

Proposers already receive ``models``. BYOK replaces ``models.served`` and every
named binding with the same user-selected endpoint, protocol and default model.
Use these bindings for every model call; do not construct a separate client
from environment credentials.

A callable model-based judge can declare the ``models`` keyword::

    def evaluate(task, result, *, models):
        verdict = models["judge"].chat([
            {"role": "user", "content": "Evaluate the episode: " + str(result.trajectory)}
        ])
        return float(verdict.strip() == "pass")

An ``EpisodeScorer`` subclass overrides ``score_with_models(task, result,
models)``. Existing deterministic scorers with ``evaluate(task, result)``
continue to work. The ``models`` argument is the same configuration used for
that step's proposer and episodes. Methods that make model calls outside this
contract must be migrated before deploying BYOK.

Configuration lifetime
----------------------

Each inference request and evolution step resolves a configuration snapshot.
All stages of a step use that snapshot. Platform inference requests include
``x-reef-provider-version``; the resolver rejects a mismatch before any model
call. Configuration failures never fall back to the managed runtime.

Replacing or clearing a binding revokes its proxy credential. An HTTP call
already in progress can complete; later calls from a step with the old binding
fail instead of changing credentials mid-step. New steps, including queued or
recovered work, resolve the user's current settings when they start. Explicitly
switching back to managed models therefore applies to those new steps.

Supported BYOK protocols are OpenAI Chat Completions and Anthropic Messages,
including token counting. The platform supplies matching native bindings to
backend model clients and evaluation episodes. Reinstall the client harness
after changing protocol or default model so its local model settings match.

Verification
------------

``tests/reef_service/test_scenario_provider.py`` uses real HTTP requests and a
subprocess episode runner to cover both protocols, proposer and judge calls,
baseline/candidate evaluation, tenant-independent scenario bindings, key
rotation, fail-closed behavior and credential-free publications. It also
checks the service capability, request-version and installation contracts.
