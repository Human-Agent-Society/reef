harness_evolve
==============

The cookbook recipe that updates text instead of weights. It proposes one
edit to the agent's harness tree, runs the agent both ways on your tasks, and
publishes the edit only if it wins.

`Evolve your harness <../evolve-your-harness.rst>`__ is the guide and the
runnable example.

+-------------+------------------------------------------------------------+
| Evolves     | the harness tree: config, rules, prompts, skills,          |
|             | extensions                                                 |
+-------------+------------------------------------------------------------+
| Signal      | one report with a finite ``score`` and exactly one         |
|             | reference                                                  |
+-------------+------------------------------------------------------------+
| Package     | ``reef/harness_evolve/``                                   |
+-------------+------------------------------------------------------------+
| Loss family | none (no weight training)                                  |
+-------------+------------------------------------------------------------+
| Processor   | reported feedback, producing a ``TraceBatch``              |
+-------------+------------------------------------------------------------+
| Needs       | one Reef process and one harness binary. No GPU.           |
+-------------+------------------------------------------------------------+
| Example     | ``tutorials/harness_evolve/``,                             |
|             | ``recipes/skillclaw/``                                     |
+-------------+------------------------------------------------------------+

How Reef implements it
----------------------

Reef snapshots the tree, applies one proposed mutation, renders both trees
through the adapter descriptor, runs one episode per task per side, and keeps
the winner. Any outcome other than a win restores the snapshot. A winning tree
publishes as a new version, pulled through ``GET /reef/harness``.

Configuration
-------------

This recipe never reads the flat ``reef.*`` section. Its configuration lives in
``<name>.yaml`` under ``REEF_RECIPE_CONFIG_DIR``. The deployment config names
that preset with ``reef.recipe`` (`Recipe configuration
<../../reference/configuration.rst#recipe-configuration>`__). ``batch_size`` and
``max_score`` go under ``data:``; the rest goes under ``evolution:``.

.. config::

   data.batch_size | 1 | traces per mutation attempt
   data.max_score | 0.0 | upper bound of the score window that batches

The window has no lower bound, so the default keeps only traces at or below
zero.

.. config::

   evolution.propose | a ``Proposer``, a plain callable, or a dotted ``module:attribute``
   evolution.evaluate | an ``EpisodeScorer``, likewise
   evolution.selection | score_comparison | ``always``, or a dotted reference to an object with ``decide``
   evolution.tasks | non-empty list of episode prompts, scored once per tree per step
   evolution.adapter | pi | ``opencode``, or an entry-point adapter
   evolution.binary | overrides the adapter's binary name
   evolution.seed | entry options loaded into the tree on first boot; recovered state takes precedence
   evolution.models | auxiliary models for the method; each key read via its ``api_key_env``
   evolution.version_check | appends the adapter's update notice, so a pulled tree reports when it is behind

The served model's binding is appended at render time; it never enters the
published files. The seed defines the baseline the first mutation is measured
against.

Run the example
---------------

`Evolve your harness <../evolve-your-harness.rst#run-the-example>`__ has the
prerequisites and the walkthrough;
``tutorials/harness_evolve/run.sh`` is the loop already
wired.

Results
-------

`examples/harness_evolve
<../../../tutorials/harness_evolve/README.md>`__ measured one
end-to-end run in 63 s on Qwen3-8B: one failing task entered the window, the
served model proposed a new skill beside the starter, the gate scored the
candidate 3.0 against 2.0 (1 win, 0 losses, 2 ties), and the tree published.

SkillClaw
~~~~~~~~~

`examples/skillclaw
<../../../recipes/skillclaw/README.md>`__ is the
larger worked instance: a full method package on the same mechanism. Each night,
``propose`` makes one decision per skill group plus the no-skill bucket and maps
them to a single composite mutation sequence. ``selection: always`` publishes
every night that produces a proposal.

It also shows the one extension point beyond the three slots. Its skill catalog
must reach the model on every live request, so its recipe overrides
``build_surface``, the hook controlling how a published artifact reaches the
request path (`skillclaw_recipe.py
<../../../recipes/skillclaw/harness/skillclaw_recipe.py>`__).

See also
--------

- `HTTP API <../../reference/http-api.rst#harness-artifacts>`__: pulling, pinning, and installing a published tree.
