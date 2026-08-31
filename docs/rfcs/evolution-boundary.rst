.. _the-evolution-boundary--what-reef-updates-and-what-it-doesnt:

The evolution boundary — what Reef updates, and what it doesn't
===============================================================

   Reef versions and updates **served artifacts** — model weights, skills,
   harness trees, context playbooks. **Run state** — the bookkeeping a harness
   produces while executing a fixed algorithm — stays harness-side. Whether an
   update is gated before publishing is a separate, per-backend risk policy, not
   what decides the boundary.

.. _1-the-question:

1. The question
---------------

The same design question keeps arriving in different costumes:

- Should TTT-Discover's PUCT archive (visit counts, pruning, selection) move
  from `recipes/tttd/examples/tttd <../../recipes/tttd/examples/tttd/harness/search.py>`__ into a training backend?
- ACE-style playbooks are updated every batch by incremental merges, with no
  win/lose gate — is that Reef's job or the harness's?
- GEPA evolves prompts by keeping a candidate frontier — Reef-side or external?

One early intuition — *"Reef owns the updates that might make things worse,
because Reef has the gate"* — does not survive contact with the codebase:
**no update is guaranteed an improvement**. A PPO step can regress; the
``openclawrl`` and ``sao`` algorithms
publish every batch with no gate at all, and their safety net is not a gate but
the version chain — every publish is a commit, receipts tie each outcome to the
exact version that produced it, and rollback restores any earlier one. So
gating cannot be the boundary criterion. Two orthogonal axes are.

.. _2-axis-1--what-changes-this-decides-where-the-code-lives:

2. Axis 1 — *what* changes. This decides where the code lives.
--------------------------------------------------------------

**Served artifacts → Reef.** Anything that answers *"which version produced
this response?"*: model weights, ``SKILL.md``, the harness tree's
configuration, rules, prompts, skills, extension code. They are long-lived,
shared by every subsequent request, and changes to them are exactly what needs
version identity, receipts, stale-publish fencing, and rollback.
Updating them is a training backend's job, and every update lands on the version chain.

**Run state → harness.** State produced by *executing* a fixed rule: a PUCT
archive's visit counts and prunes, the working memory of one search, one
conversation's context. These updates are dense, order-dependent bookkeeping
defined by the algorithm — not hypotheses to adjudicate. There is no
accept/reject decision to make about a visit-count increment, and gating one
would break the algorithm's semantics. Their scope is one run or one problem
instance. The harness owns them; Reef may at most persist snapshots (§5).

The same rule holds from the artifact side: *learned state is runtime
output, never committed to the evolved artifact*. The genome — the skill or
policy text — evolves through Reef; the memory that text accumulates while
running does not.

Worked example: TTT-Discover
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The TTTD harness is the cleanest demonstration of the boundary: during a run its
*code never changes* — only archive state does. So the search loop stays
external (`recipes/tttd/examples/tttd <../../recipes/tttd/examples/tttd/README.md>`__), and the Reef side
is exactly the part that needs the training backend: grouped advantage
computation and the ``tttd`` loss
(`recipes/tttd/slime/ <../../recipes/tttd/slime>`__).
A litmus test that the archive is state rather than genome: its per-step
updates cannot be replayed against each other as competing candidates — a
gate has no place to stand.

Conversely, the TTTD harness *does* contain genome that could evolve through
Reef one day: the prompt template, the exploration constant, the
``search_value`` design, the two-phase prefill text. Evolving those from
cross-run scores would be the same gated text-artifact loop
``HarnessEvolveBackend`` runs today. The archive still would not move.

.. _3-axis-2--how-changes-are-accepted-a-per-backend-policy:

3. Axis 2 — *how* changes are accepted. A per-backend policy.
-------------------------------------------------------------

Acceptance policies form a spectrum. All of them are Reef-side, and all land
on the same version chain:

+----------------------+--------------------------+----------------------+
| Policy               | Examples                 | Safety net           |
+======================+==========================+======================+
| Ungated continual —  | ``openclawrl`` · ``sao`` | version chain +      |
| publish every batch  | (weights); ACE-style     | rollback + receipts  |
|                      | playbook merges          |                      |
|                      | (artifacts, §5)          |                      |
+----------------------+--------------------------+----------------------+
| Compare with current | ``HarnessEvolveBackend`` | the decision, then   |
| — select when wins   |                          | the version chain    |
| exceed losses        |                          |                      |
+----------------------+--------------------------+----------------------+
| Population / Pareto  | GEPA-style prompt        | the frontier, then   |
| — keep a candidate   | evolution (§5)           | the version chain    |
| frontier             |                          |                      |
+----------------------+--------------------------+----------------------+

The gate is a risk-management choice available to any served artifact and
required by none. That weights may update ungated while harness artifacts
currently cannot is an implementation gap, not a principle (§5).

.. _4-rule-of-thumb:

4. Rule of thumb
----------------

**Reef evolves the rules; harnesses run them.** State produced by running the
rules gets persistence at most; changes to the rules themselves get versioned
publishes — gated or not is the backend's call.

+----------------+-----------------+----------------+----------------+
| The thing      | Axis 1          | Axis 2         | Lives          |
+================+=================+================+================+
| a PPO/SPO/SAO  | served artifact | ungated        | Reef backend   |
| weight step    |                 | continual      |                |
+----------------+-----------------+----------------+----------------+
| a ``SKILL.md`` | served artifact | score          | Reef backend   |
| / harness-tree |                 |                |                |
| edit           |                 | comparison     |                |
+----------------+-----------------+----------------+----------------+
| an ACE         | served artifact | ungated        | Reef backend   |
| playbook delta |                 | continual      | (§5.1)         |
+----------------+-----------------+----------------+----------------+
| a GEPA prompt  | served artifact | Pareto         | Reef backend   |
| candidate      |                 | frontier       | (§5.2)         |
+----------------+-----------------+----------------+----------------+
| a PUCT archive | run state       | — (algorithm   | harness        |
| update         |                 | bookkeeping)   |                |
+----------------+-----------------+----------------+----------------+
| one run's      | run state       | —              | harness        |
| working memory |                 |                |                |
+----------------+-----------------+----------------+----------------+

The placement litmus is the direction of data flow, not how agent-like the
code is:

- Code whose input is a request or the environment and whose output is the
  **next request** — task orchestration, grading, search loops such as TTTD's
  PUCT selection — is a harness. It runs outside, however sophisticated.
- Code whose input is the **record store** and whose output is a **publish on
  the version chain** — a weight step, a skill edit, an ACE reflector/curator
  pass — is a recipe backend. It runs inside Reef even when it calls an LLM
  to do its thinking: it never answers a user request and never touches the
  environment.

That is why TTTD's harness stays external while skill evolution (and
ACE on top of it) is built in, and the two placements are one rule, not an
inconsistency: the TTTD search loop has both ends on the environment side;
the evolution loop has both ends on Reef's side. One method can have both —
TTTD's search harness outside, its grouped training step inside — and the
seam between them is always the same pair: receipts out, reports in.

.. _5-what-the-boundary-unlocks:

5. What the boundary unlocks
----------------------------

1. **Ungated continual publishing for harness artifacts** (unlocks ACE,
   `arXiv:2510.04618 <https://arxiv.org/abs/2510.04618>`__). Structurally
   identical to online weight training with the artifact in the weights'
   place: reflector/curator merge per batch → publish → the version chain as
   the safety net. ACE's own failure mode — context collapse — is answered by
   rollback;
   a gate is optional hardening, not a prerequisite.
2. **Population/Pareto acceptance** (unlocks GEPA,
   `arXiv:2507.19457 <https://arxiv.org/abs/2507.19457>`__). Generalize the gate
   from a single incumbent to a candidate frontier carried in backend state.
   Reef's record store already holds the execution traces and scores a
   reflective mutator needs; an external implementation would have to rebuild
   that capture and the versioning both.
3. **Run-state snapshots on the version chain** (TTTD durability, optional).
   ``PUCTArchive.state_dict()`` is already JSON-ready; letting snapshots ride
   the version chain as attachments would keep a crashed harness's resume aligned
   with the weight version it trained. Persistence only — selection logic
   stays harness-side.
