.. _method-integration--one-method-one-trust-domain:

Method integration — one method, one trust domain
=================================================

   **Status (2026-08): resolved, beyond M3.** The judging half moved all the
   way into the recipe's processor (``recipes/openclawrl/processor.py``):
   sessions are reconstructed from recorded traffic by trace matching (no
   tags, no headers), PRM judging runs on a processor-private worker
   sanctioned by the ``DataProcessor`` contract, and verdicts become batch
   candidates directly — no report round-trip at all. The grader proxy and
   its ``x-reef-return-training`` opt-in are retired; agents point at Reef
   directly. File references below describe the pre-move layout and are kept
   as the record of why the move happened.

..

   A bundled method's correctness-critical logic — grading orchestration,
   session correlation, exclusion semantics — must live in **one** trust
   domain, versioned and tested with the processor that consumes its reports.
   Reef's core capabilities (records, inference, dispatch) are **served to**
   method components through APIs, never **re-implemented by** them. The
   request plane stays method-agnostic; what it gains is an egress surface, so
   a method can *observe* traffic without *carrying* it.

.. _1-what-openclaw-rl-revealed:

1. What OpenClaw-RL revealed
----------------------------

The ``openclawrl`` integration is our first method whose grading is itself part
of the algorithm (a PRM judging next-state feedback). Making it work under the
current rules — *no method-specific code in the request plane*, agent records have
a write-only HTTP surface — produced a topology with four structural defects:

1. **Correctness split across trust domains.** The method's exclusion
   semantics (next-state presence, neutral drops, the at-least-one promotion)
   live in ``recipes/openclawrl/examples/openclawrl/sessions.py`` (since removed),
   while `OpenClawRLProcessor <../../recipes/openclawrl/processor.py>`__
   deliberately accepts *any* numeric score. The processor's docstring says
   "the external grader owns the method's exclusion semantics" — load-bearing
   training logic labeled as an example, outside CI's method tests, outside
   the version chain.
2. **Core capabilities rebuilt outside, degraded.** The grader is a
   forwarding proxy that re-implements a miniature request plane
   (``grader.py`` ≈ ``service/`` +
   ``inference.py`` forwarding) and a miniature record store
   (``sessions.py`` ≈ ``records.py``,
   but in-memory, TTL-swept, lost on crash). ``sessions.py`` says it plainly:
   *"Adapted from the reef-internal coordinator on the openclaw branch."*
   The coordinator was not designed outside — it was evicted.
3. **Two unsynchronized failure domains.** The grader holds turns Reef has no
   verdict for; Reef holds records the grader may never judge (crash, TTL).
   Neither side can recover the other's state: the grader cannot re-read what
   Reef stored, and Reef cannot know a session ended.
4. **A serving regression.** Because the grader must see complete responses,
   it buffers upstream and synthesizes SSE all at once — every agent behind
   this method loses true streaming, which Reef itself supports.

None of these are bugs in the grader. They are forced moves given one missing
capability, which is the actual root cause:

**Reef's agent-record surface is write-only.**
`service/routes/records.py <../../reef/service/routes/records.py>`__
registers ``POST /reef/report`` — nothing reads,
queries, or subscribes. An external component that needs to see exchanges has
exactly one option: sit on the traffic path and keep its own copy. Meanwhile
the storage primitive for egress already exists —
`RecordStore.replay_page <../../reef/records.py>`__ is keyset-paginated and
documented "for internal streaming consumers" — it simply has no HTTP surface.

The same forces will apply to every future method whose grading is
algorithmic (ACE's Reflector, ReasoningBank's distiller, OpenClaw-RL's
missing OPD half). Without a rule change, each one rebuilds the same degraded
proxy-plus-shadow-store.

.. _2-principles:

2. Principles
-------------

These are the standards new method integrations follow. Each traces to a
defect above.

- **P1 · One trust domain per method.** Everything that decides *what
  trains* — grading orchestration, correlation, exclusion, promotion — ships
  in one package with the processor that consumes its output, tested
  together. A processor may never delegate its acceptance semantics to an
  unversioned external component.
- **P2 · Serve capabilities, don't copy them.** If a method component needs
  storage, forwarding, or correlation, Reef exposes the capability through an
  API. Re-implementing a core module outside ``reef/`` is a design smell that
  blocks review.
- **P3 · Observe, don't carry.** Method components consume traffic from the
  egress surface (§3.1); they do not proxy the request path. The request
  plane stays method-agnostic — that contract is kept, not weakened.
- **P4 · Training-relevant state is durable.** Any state whose loss changes
  what trains (session tables, pending verdicts, retry queues) lives in
  Reef's store or is reconstructible from it by replay. Process memory is a
  cache, never the source of truth.
- **P5 · No serving regressions.** A method integration may not degrade the
  request path it observes — streaming stays end-to-end native. Buffered
  capture with synthesized SSE is disallowed.
- **P6 · Reports carry the whole signal.** ``POST /reef/report`` already
  accepts structured ``feedback``; a method that produces more than a scalar
  (hints, teacher log-probs, rubric output) puts it in the report payload
  under a method-namespaced key, and its processor declares what it reads.
  No side channels.
- **P7 · ``examples/`` teaches; it never load-bears.** Anything a bundled
  method requires for correctness graduates out of ``examples/`` into the
  method's package. What remains in ``examples/`` is the harness a user would
  replace anyway (workload, agent plugin, launch config).

.. _3-design:

3. Design
---------

.. _31-agent-record-egress:

3.1 Agent-record egress
~~~~~~~~~~~~~~~~~~~~~~~

Add a read surface over the existing store, mirroring ``replay_page``:

+-------------------------------------------------------------+----------------------------------+
| Route                                                       | What it does                     |
+=============================================================+==================================+
| ``GET /reef/agent-record?scenario=&after_sequence=&limit=`` | keyset page of records           |
|                                                             | (INFERENCE and REPORT), in       |
|                                                             | append order                     |
+-------------------------------------------------------------+----------------------------------+
| ``GET /reef/agent-record/stream?scenario=&after_sequence=`` | the same page shape,             |
|                                                             | long-poll/SSE, for low-latency   |
|                                                             | consumers                        |
+-------------------------------------------------------------+----------------------------------+

Egress is the inverse of ingress and reuses its auth. A consumer's position
is its own cursor (``after_sequence``); Reef keeps no per-consumer state, so a
crashed grader recovers by replaying from its last durable cursor. Compaction
is safe unchanged: records are only compacted after training consumed them,
which requires the method's own report — a record cannot be compacted before
its grader has seen it, because the grader is what makes it trainable.

.. _32-request-tags:

3.2 Request tags
~~~~~~~~~~~~~~~~

Methods need side-channel context stamped by the agent harness (today:
``x-openclawrl-*`` headers, parsed by the proxy, held in memory). Generalize:
the request plane accepts ``x-reef-tag-<name>: <value>`` headers and stores
them on the INFERENCE record's payload metadata. Tags are opaque to the
request plane (method-agnostic, P3 preserved) but visible to egress
consumers and processors. Session correlation state then becomes *derived*
state, reconstructible by replay (P4).

.. _33-method-packages:

3.3 Method packages
~~~~~~~~~~~~~~~~~~~

A bundled method is one package under ``reef/<name>/`` (landed 2026-08 as
``recipe``/``processor``/``preparer``/``slime``) containing
every correctness-critical part: the recipe, the processor, and any grading
service (judge orchestration, PRM client, correlation). Grading services run
as service entries in the deployment config — as the grader already does in
``training-openclawrl.yaml`` — but from inside the package, on the egress API.
There are no re-exports: a method's names are imported from its package.

The considered alternative — in-core observer hooks on the request path
(a method plugin invoked post-serve) — is rejected for now: it breaks the
method-agnostic request plane for a latency win no current method needs.
Egress-with-replay gives the same visibility with a crash story the hook
model lacks. Revisit only if a method demonstrably cannot tolerate
observe-lag.

.. _34-report-payload-conventions:

3.4 Report payload conventions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The report schema is already open (``score``, ``feedback``, ``references``,
``metadata``). This RFC fixes conventions rather than adding fields:

- ``score``: scalar consumed by scored-rollout processors, as today.
- ``metadata.training.eligible: false``: the existing framework-neutral
  opt-out (`reported.py <../../reef/train/processors/reported.py>`__,
  ``_report_is_trainable``) — how a method submits a record-keeping report
  that must not train.
- ``feedback.<method>.*``: structured method signal. For OpenClaw-RL's OPD
  half: ``feedback.openclawrl.hint`` (the selected directive hint) and
  ``feedback.openclawrl.teacher`` (a reference to teacher log-probs captured
  via a non-trainable inference call). A processor that consumes structured
  feedback documents the keys it reads in its docstring.

.. _4-applying-this-to-openclaw-rl:

4. Applying this to OpenClaw-RL
-------------------------------

Milestones, each independently shippable:

1. **M1 — egress + tags.** Add §3.1 and §3.2. No behavior change for
   existing methods.
2. **M2 — grader off the data path.** Agents point at Reef directly (true
   streaming restored, P5). The grader becomes an egress consumer: replay →
   correlate by tags → judge with the PRM → ``POST /reef/report``. Its session
   table becomes derived state; the retry queue survives because report ids
   stay deterministic (``openclawrl:<receipt>``) and
   `records.py <../../reef/records.py>`__ dedup is unchanged.
3. **M3 — graduate the package.** Move grader, sessions, and PRM client into
   ``reef/methods/openclawrl/`` beside the recipe and processor; CI runs the
   exclusion-semantics tests against the processor contract (P1, P7).
   ``recipes/openclawrl/examples/openclawrl/`` keeps the opencode plugin, workload, and launch
   config.
4. **M4 — the OPD half.** With P6 conventions in place, the missing half of
   the paper lands as data, not new infrastructure: the grader extracts
   hints, runs the hint-augmented teacher forward as a non-trainable
   inference call, and reports ``score`` + ``feedback.openclawrl.*``; a
   combine processor consumes both signals. This is where the current
   architecture would have forced a third out-of-tree component; under this
   RFC it is one package growing one processor.

.. _5-what-does-not-change:

5. What does not change
-----------------------

- **The client contract.** Two headers, one report route, receipts. Agents
  integrated today keep working unmodified; tags are optional.
- **The evolution boundary.** This RFC moves *method* code, not *run state*.
  A grader's session table is derived bookkeeping over Reef-served traffic,
  which is exactly why it belongs on Reef's store — the
  `evolution boundary <evolution-boundary.rst>`__ is about harness-owned search
  state, and is untouched.
- **Recipe authorship.** Third-party methods still need only a recipe class
  and a config entry. Method packages are how *bundled* methods are held to
  a higher standard, not a new requirement on users.

.. _6-compliance-checklist-for-a-new-bundled-method:

6. Compliance checklist for a new bundled method
------------------------------------------------

Before a method merges, its review answers yes to all of:

- ☐ All accept/reject/promotion semantics live in the method package,
  covered by tests against the processor (P1).
- ☐ No component proxies the request path or re-implements a core module
  (P2, P3).
- ☐ Killing any method component and restarting it loses no
  training-relevant state — everything derives from replay (P4).
- ☐ An agent behind this method gets native streaming (P5).
- ☐ Every training signal arrives through ``POST /reef/report`` under
  documented keys (P6).
- ☐ ``examples/`` for this method contains only harness the user would
  replace (P7).
