.. _the-evolution-boundary--what-reef-updates-and-what-it-doesnt:

The evolution boundary: what Reef updates, and what it doesn't
===============================================================

:Status: Deprecated

.. warning::

   This historical RFC predates the issue-based RFC process and will be removed
   in a future cleanup.

   Reef versions and updates **served artifacts** such as model weights, skills,
   harness trees, and context playbooks. **Run state** is the bookkeeping a
   harness produces while executing a fixed algorithm, and it stays on the
   harness side. Whether an
   update is gated before publishing is a separate, per-backend risk policy, not
   what decides the boundary.

.. _1-the-question:

1. The question
---------------

Several methods raise the same design question:

- Should TTT-Discover's PUCT archive (visit counts, pruning, selection) move
  from `recipes/tttd/examples/tttd <../../recipes/tttd/examples/tttd/harness/search.py>`__ into a training backend?
- ACE-style playbooks are updated every batch by incremental merges, with no
  win/lose gate. Is that Reef's job or the harness's?
- GEPA evolves prompts by keeping a candidate frontier. Should that live in
  Reef or outside it?

The presence of a gate cannot determine this boundary because **no update is
guaranteed to be an improvement**. A PPO step can regress. The
``openclawrl`` and ``sao`` algorithms
publish every batch with no gate at all, and their safety net is not a gate but
the version chain: every publish is a commit, receipts tie each outcome to the
exact version that produced it, and rollback restores any earlier one. Gating
therefore cannot be the boundary criterion. The boundary depends on two
orthogonal axes.

.. _2-axis-1--what-changes-this-decides-where-the-code-lives:

2. Axis 1: *what* changes
-------------------------

**Served artifacts → Reef.** Anything that answers *"which version produced
this response?"*: model weights, ``SKILL.md``, the harness tree's
configuration, rules, prompts, skills, extension code. They are long-lived,
shared by every subsequent request, and changes to them are exactly what needs
version identity, receipts, stale-publish fencing, and rollback.
Updating them is a training backend's job, and every update lands on the version chain.

**Run state → harness.** State produced by *executing* a fixed rule: a PUCT
archive's visit counts and prunes, the working memory of one search, or one
conversation's context. These updates are dense, order-dependent bookkeeping
defined by the algorithm, not hypotheses to adjudicate. There is no
accept/reject decision to make about a visit-count increment, and gating one
would break the algorithm's semantics. Their scope is one run or one problem
instance. The harness owns them; Reef may at most persist snapshots (§5).

The same rule holds from the artifact side: *learned state is runtime output,
never committed to the evolved artifact*. The skill or policy text evolves
through Reef; the memory it accumulates while running does not.

Worked example: TTT-Discover
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

During a TTTD run, the harness code does not change; only the archive state
does. The search loop therefore stays
external (`recipes/tttd/examples/tttd <../../recipes/tttd/examples/tttd/README.md>`__), and the Reef side
is exactly the part that needs the training backend: grouped advantage
computation and the ``tttd`` loss
(`recipes/tttd/slime/ <../../recipes/tttd/slime>`__).
The archive is state rather than genome because its per-step updates cannot be
replayed against each other as competing candidates. An accept-or-reject gate
therefore does not apply.

Conversely, the TTTD harness *does* contain genome that could evolve through
Reef one day: the prompt template, the exploration constant, the
``search_value`` design, the two-phase prefill text. Evolving those from
cross-run scores would be the same gated text-artifact loop
``HarnessEvolveBackend`` runs today. The archive still would not move.

.. _3-axis-2--how-changes-are-accepted-a-per-backend-policy:

3. Axis 2: *how* changes are accepted
-------------------------------------

Acceptance policies form a spectrum. All of them are Reef-side, and all land
on the same version chain:

+----------------------+--------------------------+----------------------+
| Policy               | Examples                 | Safety net           |
+======================+==========================+======================+
| Ungated continual:   | ``openclawrl``, ``sao``  | version chain,       |
| publish every batch  | (weights); ACE-style     | rollback, receipts   |
|                      | playbook merges          |                      |
|                      | (artifacts, §5)          |                      |
+----------------------+--------------------------+----------------------+
| Compare with current | ``HarnessEvolveBackend`` | the decision, then   |
| select when wins     |                          | the version chain    |
| exceed losses        |                          |                      |
+----------------------+--------------------------+----------------------+
| Population / Pareto  | GEPA-style prompt        | the frontier, then   |
| keep a candidate     | evolution (§5)           | the version chain    |
| frontier             |                          |                      |
+----------------------+--------------------------+----------------------+

The gate is a risk-management choice available to any served artifact and
required by none. That weights may update ungated while harness artifacts
currently cannot is an implementation gap, not a principle (§5).

.. _4-rule-of-thumb:

4. Rule of thumb
----------------

Reef versions changes to the rules that a served artifact applies. The harness
executes those rules and owns the resulting run state. Reef may persist that
state, while changes to the rules are published as new versions. The backend
decides whether to gate those publishes.

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
| a PUCT archive | run state       | algorithm      | harness        |
| update         |                 | bookkeeping    |                |
+----------------+-----------------+----------------+----------------+
| one run's      | run state       | not applicable | harness        |
| working memory |                 |                |                |
+----------------+-----------------+----------------+----------------+

The placement litmus is the direction of data flow, not how agent-like the
code is:

- Code whose input is a request or the environment and whose output is the
  **next request** is a harness. This includes task orchestration, grading, and
  search loops such as TTTD's PUCT selection. It runs outside Reef.
- Code whose input is the **record store** and whose output is a **publish on
  the version chain** is a recipe backend. Examples include a weight step, a
  skill edit, and an ACE reflector/curator pass. It runs inside Reef even when
  it calls an LLM: it never answers a user request and never touches the
  environment.

TTTD's harness stays external because its search loop reads and writes
environment-side state. Skill evolution, including ACE on top of it, stays in
Reef because that loop reads records and publishes artifacts. One method can
have both: TTTD's search harness runs outside while its grouped training step
runs inside. They exchange receipts and reports.

.. _5-what-the-boundary-unlocks:

5. What the boundary unlocks
----------------------------

1. **Ungated continual publishing for harness artifacts** (unlocks ACE,
   `arXiv:2510.04618 <https://arxiv.org/abs/2510.04618>`__). Structurally
   identical to online weight training with the artifact in the weights'
   place: reflector/curator merge per batch → publish → the version chain as
   the safety net. Rollback addresses ACE's own failure mode of context
   collapse;
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
   with the weight version it trained. This adds persistence only; selection
   logic stays on the harness side.
