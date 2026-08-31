RFC: Reef Router
================

:Status: Deprecated
:Authors: @hanfeiyu
:Created: 2026-08-25
:Updated: 2026-08-30
:Discussion: https://github.com/Human-Agent-Society/reef/issues/415
:Implementation: Not started

.. warning::

   This historical RFC predates the issue-based RFC process and will be removed
   in a future cleanup.

Summary
-------

Give each Reef scenario its own model-routing policy. The gateway
authenticates the caller and fixes the scenario first;
`vLLM Semantic Router <https://github.com/vllm-project/semantic-router>`__
then selects a stable scenario role, such as ``student`` or ``teacher``, and
resolves it to a ready provider model. Reef validates and records that result.
Routing cannot change the bound scenario.

.. code:: text

   authenticated tenant
           │
           ▼
   Reef scenario A ──▶ SR entrypoint A ──▶ SR recipe A ──▶ roles {student, teacher}

   Reef scenario B ──▶ SR entrypoint B ──▶ SR recipe B ──▶ roles {stable, candidate}
           ▲
           └──────── route, fallback, and replay must stay in this scenario

The gateway binds each authenticated tenant to one Reef scenario. The scenario
remains Reef's isolated workload: its records, training state, and version
chain. The routing decision is ``(scenario, role)``; the executed model is
resolved and recorded with it. Routing is optional. Static Semantic Router
configuration comes first; learnable router artifacts remain follow-on work.

Motivation
----------

A scenario already isolates records, trainer state, and releases.
Routing one scenario into another would break that ownership model and could
mix tenant traffic or training data. Semantic Router owns model-choice policy;
Reef enforces scenario isolation and records the resulting execution.
Canary, shadow, and student/teacher policies should be reusable rather than
copied into every client or harness.

Goals and non-goals
-------------------

Goals
~~~~~

- Resolve and authorize the scenario before model routing.
- Select only authorized roles whose provider models are ready for that
  scenario.
- Freeze the scenario, role, resolved model, and routing version for the
  request lifetime.
- Preserve Reef admission, pause/resume, response validation, and serving
  version records.
- Record the role, resolved model, routing version, admitted artifact, and
  verified serving versions through existing Reef records and Semantic Router
  Replay.

Non-goals
~~~~~~~~~

- Route, fail over, shadow, or replay traffic across scenarios.
- Cold-load a model or historical artifact on the request path.
- Replace provider-gateway replica scheduling or KV-cache load balancing.
- Add a second routing policy engine or a new persistent record type.
- Implement learnable routing in this RFC.

Proposal
--------

Scenario isolation
~~~~~~~~~~~~~~~~~~

Each routed ingress is bound to one scenario. The reference deployment runs
one shared Envoy and one shared Semantic Router process. Its canonical
configuration declares one Semantic Router entrypoint and routing recipe per
Reef scenario. The gateway selects the entrypoint; the routing recipe owns
that scenario's policy, authorized roles, and role-to-provider-model
resolutions. These are Semantic Router recipes, not Reef recipes: one Reef
deployment still serves one Reef recipe, shared by all scenarios in that
deployment.

The gateway derives the scenario from authenticated tenant context. It does
not accept a client-selected scenario on routed ingress. Semantic Router
receives traffic only after that binding and has no operation that can alter
it.

.. code:: mermaid

   flowchart LR
       C[Client] --> G[Gateway authentication]
       G --> S{Bound scenario}
       subgraph SR[Shared Semantic Router process]
         RA[Scenario A entrypoint and router recipe]
         RB[Scenario B entrypoint and router recipe]
       end
       S -->|tenant A| RA
       S -->|tenant B| RB
       RA --> MA[Scenario A role map]
       RB --> MB[Scenario B role map]
       MA --> ReefA[Reef scenario A]
       MB --> ReefB[Reef scenario B]
       MA -.-x ReefB
       MB -.-x ReefA

One Semantic Router process per scenario is not required. Routing recipes
scope model eligibility and policy, while the process, canonical
configuration, providers, management API, replay backend, and metrics endpoint
remain shared and operator-only. This is logical isolation, not a process
security guarantee. Deployments requiring hard tenant isolation use separate
Envoy, Semantic Router, Reef, and runtime stacks. Semantic Router aliases are
process-wide, so the bridge names each one ``<scenario>/<role>``, for example
``tenant-a/student``. Reef validates the scenario prefix, role, and resolved
model together.

Architecture
~~~~~~~~~~~~

.. important::

   Envoy is required by the supported Semantic Router inference path.
   Semantic Router implements the Envoy ``ext_proc`` interface. An existing
   Envoy Gateway or Istio deployment may provide that data plane.

.. code:: mermaid

   flowchart LR
       C[Client] --> E[Envoy tenant route]
       E <-->|ext_proc gRPC| SR[Scenario entrypoint and router recipe]
       SR -->|selected role, resolved model, and routing version| E
       E --> R[Reef routed ingress]
       R -->|validated provider model| G[Provider gateway]
       G --> W[Ready worker]
       W --> R
       R -->|persist AgentRecord, then return response| C

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Component
     - Responsibility
   * - Tenant gateway
     - Authenticate the caller, bind one scenario, remove client copies of
       routing headers, and select the matching scenario entrypoint.
   * - Semantic Router
     - Evaluate signals, select one scenario role, resolve its provider model,
       and rewrite the request body.
   * - Reef
     - Enforce the scenario binding, validate the role and unchanged body model
       against one routing snapshot, admit the request, validate serving
       identity, and persist records.
   * - Provider gateway
     - Select a healthy replica for the resolved provider model.

Reef adds no competing policy engine. Semantic Router owns policy; Reef owns
scenario isolation and the auditable execution contract.

Routing decision
~~~~~~~~~~~~~~~~

Within a scenario recipe, the policy selects a stable role such as ``student``
or ``teacher``. Semantic Router has no role field; its shared provider registry
uses a logical model alias. The bridge encodes the role as
``<scenario>/<role>``. ``providers.models[].name``, ``modelCards[].name``,
and ``modelRefs[].model`` use that qualified alias, and
``x-selected-model`` carries it on the backend-bound request. Reef validates
the prefix and records the role component.

Semantic Router resolves the alias to the configured backend model identifier
and writes that value into the request-body ``model``. Reef does not replace
it. The policy decision is ``(scenario, role)``; its exact routed target is
``(scenario, role, model, routing.version)``. In v1, every qualified alias
maps to one final body model; nested model or LoRA sub-selection is out of
scope.

The bridge defines ``routing.version`` as an immutable identifier for the exact
active routing document. The initial adapter computes a SHA-256 hash of the
loaded canonical bytes, called ``DocumentHash`` in this RFC, and exposes it as
``x-vsr-config-version``. This is a Reef integration contract, not a model
release or an upstream tenant identifier.

Before admission, Reef pins one reconciled snapshot:

.. code:: text

   routing version
       └── <scenario>/<role> alias ──▶ body model + readiness

The snapshot must authorize the complete routed target. Unknown versions,
role/model mismatches, unavailable models, and unverifiable responses fail
closed.

.. code:: mermaid

   flowchart LR
       T[Authenticated tenant] --> S[Scenario: tenant-a]
       S --> R[Router configuration for tenant-a]
       Q[Request] --> R
       R --> M[Selected alias: tenant-a/student]
       M --> P[Resolved model: tenant-a/student-v7]
       P --> A[Validate snapshot and admit]

The resolved model must already be ready at the provider gateway. Reef never
cold-loads a model or searches another scenario for a similarly named role.

Request contract
~~~~~~~~~~~~~~~~

The Semantic Router bridge must expose request-side decision identity before
Reef admits the request. Current Semantic Router already adds
``x-selected-model`` and rewrites the body model on the backend-bound request.
A small adapter adds ``x-vsr-config-version`` and ``x-vsr-replay-id`` in the
same decision context before forwarding. Reef then pins the reconciled
snapshot keyed by that version. The client-facing response diagnostic
``x-vsr-selected-model`` is a different interface.

.. list-table::
   :header-rows: 1
   :widths: 32 25 43

   * - Field received by Reef
     - Set by
     - Purpose
   * - ``x-reef-scenario``
     - Authenticated gateway route
     - Fixed tenant scenario; never router output.
   * - ``x-selected-model``
     - Semantic Router
     - Selected ``<scenario>/<role>`` alias.
   * - ``x-vsr-config-version``
     - Semantic Router adapter
     - Active ``DocumentHash``.
   * - ``x-vsr-replay-id``
     - Semantic Router adapter
     - Detailed routing-data lookup.
   * - ``x-request-id``
     - Gateway
     - End-to-end correlation.
   * - Request-body ``model``
     - Semantic Router
     - Resolved backend model identifier to execute.

The routing headers and body model must validate against one pinned snapshot.
Neither ``DocumentHash`` nor a replay ID supplies scenario authorization by
itself; Replay lookup is scoped by scenario and routing version.

.. code:: mermaid

   sequenceDiagram
       participant C as Client
       participant G as Gateway
       participant SR as Scenario router
       participant R as Reef scenario
       participant B as Backend

       C->>G: inference request and tenant credential
       G->>G: authenticate; bind scenario
       G->>SR: request in scenario namespace
       SR-->>G: role alias, body model, version, and replay ID
       G->>R: fixed scenario and routed request
       R->>R: validate alias, body model, and snapshot
       R->>R: retain routed target; acquire serving state
       R->>B: unchanged resolved-model request
       Note over R,B: A pause or update may resume this request
       B-->>R: response and serving identity
       R->>R: validate and append AgentRecord
       R-->>C: response through gateway

Conceptual Reef code:

.. code:: python

   def routed_inference(request, binding):
       parsed = parse_request_headers(request.headers)
       require(parsed.scenario == binding.scenario)

       routing = parse_router_headers(request.headers)
       payload = dict(request.payload)
       model = require_nonempty(payload.get("model"))

       snapshot = registry.require_router_snapshot(routing.version)
       target = snapshot.require_target(
           scenario=binding.scenario,
           alias=routing.alias,
           model=model,
       )
       target.require_ready()

       scenario = registry.require_loaded(binding.scenario)
       admitted = scenario.admit(payload)
       response = admitted.backend.infer(payload)
       validate_serving_identity(response, admitted)

       return accept_inference(
           scenario=scenario.name,
           payload={
               **payload,
               "response": response,
               "metadata": {
                   "routing": routing.for_record(role=target.role),
               },
           },
           artifact_ref=admitted.artifact_ref,
       )

The tuple ``(scenario, role, model, routing.version)`` stays fixed across
admission waits, streaming, and the existing defensive retry of an explicit
backend ``abort``. A retry may reacquire the scenario's current admission
state, but it may not re-route. A normal Reef pause/update does not abort or
re-route the request; token-level serving version spans remain authoritative
when generation crosses a weight update.

Trust model
~~~~~~~~~~~

.. code:: python

   # Before tenant binding and routing
   strip_client_headers(
       "x-reef-scenario",
       "x-selected-model",
       "x-vsr-config-version",
       "x-vsr-replay-id",
   )

   scenario = authorize_tenant(request.credentials)
   bind_header("x-reef-scenario", scenario)
   set_request_model(entrypoint_for(scenario))

   # After the selected Semantic Router recipe runs
   require_header("x-selected-model")
   require_header("x-vsr-config-version")
   require_header("x-vsr-replay-id")
   require_body_field("model")
   preserve_or_create_header("x-request-id")

   # Reef routed ingress
   require_gateway_peer()
   require_header("x-reef-scenario", configured_scenario)
   require_loaded_scenario(configured_scenario)
   require_snapshot_target(
       version=header("x-vsr-config-version"),
       scenario=configured_scenario,
       alias=header("x-selected-model"),
       model=body("model"),
   )

A gateway credential for scenario A must not reach the scenario B routed
listener. Operators provision direct Reef ingress as a separate listener or
trust policy; routed mode does not create it.

Routing data in AgentRecord
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The base path executes one model and produces one existing ``INFERENCE``
record. The top-level ``scenario`` and ``metadata.routing.role`` form the
routing decision. The resolved model remains in the existing request
``payload.model``; routing metadata adds the routing version and correlation
IDs.

.. code:: text

   AgentRecord(INFERENCE)
   ├── agent_record_id: <id>
   ├── scenario: tenant-a                    # tenant-bound workload
   ├── request_type: inference
   ├── references: []
   ├── artifact_ref: <admitted scenario artifact>
   └── payload
       ├── model: tenant-a/student-v7        # resolved provider model
       ├── <existing request fields>
       ├── runtime_load_id[_spans]: <verified serving versions>
       ├── response: <validated response>
       └── metadata
           ├── tags: optional
           └── routing
               ├── router: vllm-sr
               ├── role: student
               ├── version: <DocumentHash>
               ├── request_id: <x-request-id>
               └── replay_id: <opaque ID>

Detailed signals, decision reasons, and captured bodies remain in Semantic
Router Replay. The bridge must scope Replay lookup and authorization by
``(scenario, routing.version, replay_id)``; the replay ID alone is not
authorization. Reef does not add a routing table or SQLite migration.

A follow-on policy that executes multiple roles must produce an inference
record containing the role and resolved model for every execution, link the
related attempts, and identify the client-visible result. #434 and #435 own
the exact shadow and escalation shapes.

A later evaluator sends an existing ``REPORT`` record in the same scenario
and references the inference:

.. code:: text

   AgentRecord(REPORT)
   ├── scenario: tenant-a
   ├── request_type: report
   ├── references: [<inference id>]
   └── payload
       ├── score: 0.95 | null
       ├── feedback: <existing feedback> | null
       └── metadata.routing
           └── outcome: <bounded evaluator data>

The referenced inference record contains the immutable scenario, role, model,
routing version, and replay ID. Downstream evaluation and learning join that
context through the reference instead of copying a partial route.

Router learning from inference and report records, plus versioned router
artifacts, requires a separate RFC.

This remains compatible with recipe-declared report contracts: those schemas
validate their declared fields while preserving additional report metadata.

Failure handling
~~~~~~~~~~~~~~~~

.. code:: mermaid

   flowchart TD
       Q[Request] --> A{Tenant authorized?}
       A -->|no| U[Reject]
       A -->|yes| S[Bind scenario]
       S --> H{Scenario router healthy?}
       H -->|no| F[Fail closed]
       H -->|yes| V{Routing version known?}
       V -->|no| X[Reject; no fallback]
       V -->|yes| L{Role authorized?}
       L -->|no| X
       L -->|yes| M{Role resolves to body model?}
       M -->|no| X
       M -->|yes| R{Resolved model ready?}
       R -->|no| X
       R -->|yes| E[Admit and execute]
       E --> I{Serving identity valid?}
       I -->|no| X
       I -->|yes| W[Record]

Router, gateway, or model failure never falls back to another scenario. An
in-scenario fallback is allowed only when that scenario policy explicitly
names it, records every executed role and model, and identifies the
client-visible result.

What the contract unlocks
~~~~~~~~~~~~~~~~~~~~~~~~~

All follow-on policies remain inside one scenario:

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Capability
     - Scenario-local roles
     - Follow-on
   * - Stable/default
     - One default role bound to a ready model.
     - Static Semantic Router policy.
   * - Canary
     - Stable and candidate roles in the same scenario.
     - `#432 <https://github.com/Human-Agent-Society/reef/issues/432>`__.
   * - Shadow
     - Primary and shadow roles in the same scenario.
     - `#434 <https://github.com/Human-Agent-Society/reef/issues/434>`__.
   * - Student/teacher
     - Student and teacher roles in the same scenario.
     - `#435 <https://github.com/Human-Agent-Society/reef/issues/435>`__.
   * - Retained version
     - A retained role bound to an already-ready model.
     - Lifecycle work aligned with #254.

Continual OPD is the clearest student/teacher example. Its current v1 uses a
harness-side router and separate student and teacher scenario/deployment pairs
because Reef has no first-class model router and one Reef deployment serves
one Reef recipe. This RFC is the follow-on: student and teacher become stable
roles inside one tenant-bound scenario.

.. code:: mermaid

   flowchart LR
       Q[Task] --> S[Student role]
       S --> V{Verifier passes?}
       V -->|yes| O[Return student answer]
       V -->|no| T[Escalate to teacher role]
       T --> O2[Return teacher answer]
       T -. distill teacher output .-> S2[Next student version]

Both roles, their resolved models, and every resulting record stay in one
scenario. The target is teacher-comparable task quality at lower average
inference cost because the student handles most requests. As distillation
improves the student, the escalation share and cost per solved task should
fall.

This is a measured objective, not a guarantee. Evaluation compares
student-only, teacher-only, and routed runs on task success, escalation rate,
latency, and cost per solved task. Post-response verifier escalation is the
scenario-local execution policy tracked by #435; Semantic Router alone does
not implement that second call.

A provider model can identify a full model or adapter already served by the
scenario's provider gateway. A role may change bindings on a later routing
version, but Reef never cold-loads it on this path.

Configuration
-------------

A routed Reef ingress is explicit and bound to one scenario:

.. code:: yaml

   reef:
     recipe: my_project.continual_opd:ContinualOPDRecipe  # supplied by #435
     router: vllm-sr
     router_scenario: tenant-a
     token: ${REEF_GATEWAY_TOKEN_TENANT_A}

The shared Semantic Router document binds ``tenant-a`` to its own entrypoint
and routing recipe. Semantic Router calls each role a logical model alias and
maps it to the provider model that Reef receives in the request body. This
conceptual fragment shows only the fields relevant to that contract. It uses
the target multi-entrypoint shape from `upstream #2331
<https://github.com/vllm-project/semantic-router/issues/2331>`__; implementation
must pin a Semantic Router build and validate the exact complete document.

.. code:: yaml

   providers:
     models:
       - name: tenant-a/student         # qualified alias; role = student
         provider_model_id: tenant-a/student-v7
         backend_refs:
           - name: reef-tenant-a
             endpoint: reef:8900
             protocol: http
             weight: 1
       - name: tenant-a/teacher         # qualified alias; role = teacher
         provider_model_id: tenant-a/teacher-v3
         backend_refs:
           - name: reef-tenant-a
             endpoint: reef:8900
             protocol: http
             weight: 1

   entrypoints:
     - model_names: [reef/tenant-a]
       recipe: tenant-a

   recipes:
     - name: tenant-a
       routing:
         modelCards:
           - name: tenant-a/student
           - name: tenant-a/teacher

         decisions:
           - name: mathematical-reasoning
             priority: 100
             rules:
               operator: OR
               conditions:
                 - type: domain
                   name: math
             modelRefs:
               - model: tenant-a/student
               - model: tenant-a/teacher
             algorithm:
               type: static

           - name: default
             priority: 0
             rules:
               operator: AND
               conditions: []
             modelRefs:
               - model: tenant-a/student
             algorithm:
               type: static

Each additional scenario adds another entrypoint, recipe, and qualified
aliases to this document. One routing version identifies the complete shared
document. The gateway replaces the client's body model with the scenario
entrypoint; Semantic Router then writes the resolved backend model. Reef
validates and forwards that value unchanged.

Reference example and reproducibility
-------------------------------------

The student/teacher implementation in #435 must ship a runnable, recipe-owned
``recipes/<method>/examples/opd_router/`` example. #435 chooses the owning
method; this router RFC does not create a new bundled Reef recipe merely to
host the demonstration.

.. code:: text

   recipes/<method>/examples/opd_router/
   ├── README.md                 # prerequisites, commands, expected outputs
   ├── pyproject.toml
   ├── run.py
   ├── run.sh
   ├── serve.yaml                # one Reef tenant scenario
   ├── semantic-router.yaml      # roles + provider-model mappings
   ├── envoy.yaml
   └── results/
       ├── manifest.json         # commits, models, data, seed, hardware
       └── summary.json          # comparable metrics for every run

The reproduction runs three configurations on the same task set:

1. student only;
2. teacher only;
3. student first with verifier-triggered teacher escalation.

``summary.json`` reports task success, escalation rate, input/output tokens,
normalized inference cost, cost per solved task, and p50/p95 latency. The
``README`` renders those values as one comparison table and gives the exact
command that produced them. Raw model weights, large traces, and secrets are
not committed.

The routed result passes only when it demonstrates the intended tradeoff:
task quality is measured against teacher-only, and cost per solved task is
lower than teacher-only. The RFC does not predeclare a quality tolerance;
the example PR must state and justify its threshold before reporting success.

The example receives an entrypoint contract test in
``tests/test_example_entrypoints.py`` and a deterministic CPU/container test
with fake student, teacher, and verifier services. A GPU reproduction records
the real model result but is not required in ordinary CPU CI.

If the experiment later needs a genuinely reusable record-to-training binding
that existing recipes cannot express, that method follows the normal recipe
RFC and review process. Until then, any experiment-specific recipe remains a
dotted reference owned by the example.

Compatibility and migration
---------------------------

Routing is off by default. Direct clients, records, and reports are unchanged.
Routed deployments add Envoy and Semantic Router and expose a scenario-bound
listener.

.. code:: text

   rollout:  load scenario models → start shared router → validate entrypoints
             → open routed listener → inspect records → move tenant traffic

   rollback: return that tenant to direct Reef; migrate no stored data

Security and privacy
--------------------

The routed listener accepts only its gateway credential. Envoy removes client
copies of scenario and router-output headers and replaces the client model
with the scenario entrypoint before routing. ``failure_mode_allow`` is false.
The ``ext_proc`` and management endpoints are private.

Semantic Router sees request content and inherits the scenario retention,
redaction, TLS, and data-residency requirements. Replay body capture is off by
default. Enabling it requires a scenario-specific retention review.

Operations and observability
----------------------------

Measure router health and latency, config reloads, role selection, resolved
model, readiness rejections, Reef admission, and backend failures per scenario.
Request IDs join traces. Scenario, role, model, config version, and replay IDs
must not create unbounded metric labels.

Testing
-------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Layer
     - Checks
   * - Contract
     - Client header stripping, fixed scenario, missing routing fields,
       unchanged body-model forwarding, role/model/version validation,
       cross-scenario rejection, and config reload snapshots.
   * - CPU
     - Buffered and streaming records, admission waits, pause/resume with
       mixed serving versions, explicit backend-``abort`` retry with the same
       scenario, role, model, and routing version, reports, and direct-client
       compatibility.
   * - Container
     - Two scenario entrypoint/Semantic Router recipe pairs in one router
       process; attempts to mix scenario, role, model, routing version, replay
       ID, or fallback are rejected.
   * - GPU smoke
     - Route between two roles bound to ready models in one scenario while a
       second scenario remains isolated. #435 also reproduces
       ``recipes/<method>/examples/opd_router/``.

The RFC itself requires no GPU.

Alternatives considered
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Alternative
     - Why not
   * - Router selects a Reef scenario
     - Breaks tenant isolation and can mix records, policy, or training
       data.
   * - One Semantic Router recipe for all scenarios
     - Allows policy, state, and fallback behavior to escape their scenarios.
   * - Build another Reef policy engine
     - Duplicates Semantic Router and still needs an Envoy bridge.
   * - Route directly to workers
     - Loses Reef admission, artifact, serving-version, and record contracts.
   * - Client-selected scenario, role, or model
     - Treats untrusted request fields as routing authority.

Risks and unresolved questions
------------------------------

- The request-metadata adapter is an upstream Semantic Router contribution or
  a small pinned patch until accepted upstream.
- Immutable referenced assets and a pinned router build are required because
  the routing version identifies canonical bytes, not later asset or binary
  changes.
- Current Reef binds one runtime/backend and one served artifact chain to a
  scenario. An authoritative, version-aware role-to-provider-model snapshot
  and readiness reconciliation API must land before heterogeneous targets
  receive production traffic.
- Semantic Router management, replay storage, provider credentials, and the
  metrics endpoint are process-wide. They remain operator-only; deployments
  requiring tenant-separated access use separate stacks.

Implementation plan
-------------------

1. Land or pin the Semantic Router request-metadata adapter.
2. Add Envoy tenant binding, header stripping, and one Semantic Router
   entrypoint/routing-recipe pair per scenario in the shared configuration.
3. Add Reef scenario-bound ingress, routed-target validation, unchanged
   body-model forwarding, and routing metadata in existing records.
4. Add versioned role-to-provider-model reconciliation and fail-closed
   readiness.
5. Add the two-scenario container isolation test and one small GPU smoke.
6. Implement canary and shadow separately; #435 adds student/teacher and the
   OPD reproduction. Learnable routing requires another RFC.

Until steps 1--4 are complete, routed ingress receives no production traffic.
RFC acceptance satisfies only the RFC milestone in #415; the implementation
issue remains open.

Decision record
---------------

Pending maintainer decision.
