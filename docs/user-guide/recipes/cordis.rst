cordis
======

The harness half of Reef, built in: cordis is the general engine that evolves
any kind of harness, carried by ``reef/train/cordis_backend/`` the way
``slime_backend`` carries weights. The demo deployment selects it through a
recipe config named ``harness_evolve``. It proposes one edit to the agent's harness
tree, runs the agent both ways on your tasks, and publishes the edit only if
it wins. The demo lives in ``tutorials/harness_evolve/``; the paper-backed
reproduction on the same engine is ``recipes/skillclaw/``.

`Evolve your harness <../evolve-your-harness.rst>`__ is the guide and the
runnable example.

+-------------+------------------------------------------------------------+
| Evolves     | the harness tree: config, rules, prompts, skills,          |
|             | extensions                                                 |
+-------------+------------------------------------------------------------+
| Signal      | one report with a finite ``score`` and exactly one         |
|             | reference                                                  |
+-------------+------------------------------------------------------------+
| Package     | ``reef/train/cordis_backend/``                             |
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
   evolution.version_check | appends the adapter's update notice; an interactive pulled tree offers to run the update or skip when behind

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

`tutorials/harness_evolve
<../../../tutorials/harness_evolve/README.md>`__ measured one
end-to-end run in 63 s on Qwen3-8B: one failing task entered the window, the
served model proposed a new skill beside the starter, the gate scored the
candidate 3.0 against 2.0 (1 win, 0 losses, 2 ties), and the tree published.

The committed notebook run repeats the arc with no GPU at all: ollama
``qwen2.5:7b`` on a Mac mini failed one task, the self proposer updated the
seed skill, and the gate published on 1 win, 0 losses, 2 ties.

SkillClaw
~~~~~~~~~

`recipes/skillclaw
<../../../recipes/skillclaw/README.md>`__ is the
paper reproduction: a full method package on the same engine. Each night,
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
