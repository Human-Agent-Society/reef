skillclaw
=========

SkillClaw (`arXiv:2604.08377 <https://arxiv.org/abs/2604.08377>`__) grows an
agent's skill pool from the agent's own traffic. By day the agent drains a
frozen task list against the current pool. By night one decision per
observation changes the pool, and the next day measures what changed.

+-------------+--------------------------------------------------------------+
| Evolves     | the harness tree: the agent's skill pool                     |
+-------------+--------------------------------------------------------------+
| Signal      | one report per task, with the day's last report closing      |
|             | the batch                                                    |
+-------------+--------------------------------------------------------------+
| Loss family | none (no weight training)                                    |
+-------------+--------------------------------------------------------------+
| Package     | ``recipes/skillclaw/``                                       |
+-------------+--------------------------------------------------------------+
| Processor   | reported feedback, producing a ``TraceBatch``                |
+-------------+--------------------------------------------------------------+
| Needs       | a Reef process, the ``pi`` binary, and Docker for the        |
|             | Harbor tasks. Reef itself needs no GPU.                      |
+-------------+--------------------------------------------------------------+
| Example     | ``recipes/skillclaw/``                                       |
+-------------+--------------------------------------------------------------+

What it does
------------

A day is the frozen 60 task list run once against the pool the night before
left behind. A night reads that day's recorded traffic and makes one decision
per skill group plus the no-skill bucket: create, improve, merge, or skip.
Every non-skip decision applies, so the pool the next day measures is always
the one the night produced.

The method needs no separate judge of its own edits. The next day is the
measurement, which is why the control run matters: it replays the same rounds
and reports nothing, so its pool never changes and its scores are the baseline
the method run is read against.

How Reef implements it
----------------------

skillclaw is a method package on the harness evolution engine
(``reef/train/cordis_backend/``); `Evolve your harness
<../evolve-your-harness.rst>`__ describes the mechanism it runs on. The method
supplies ``propose`` and ``evaluate``; the engine owns render, episodes,
ledger, and revert. ``propose`` runs the night flow and maps its decisions to
one composite mutation sequence, so a whole night applies under one snapshot
and settles under one verdict.

Two choices distinguish it from the tutorial method. ``selection: always``
publishes every night that produces a proposal, because the next day is the
gate. And the skill catalog must reach the model on every live request, so the
recipe overrides ``build_surface``, the hook controlling how a published
artifact reaches the request path (`recipe.py
<../../../recipes/skillclaw/recipe.py>`__). That override is
the one extension point beyond the method's own callables.

Configuration
-------------

``recipes/skillclaw/skillclaw.yaml`` is the recipe config the driver boots. It
names ``recipes.skillclaw.recipe:SkillClawRecipe`` as its ``implementation``
and sets ``batch_size: 60`` with ``max_score: .inf``, so every report of the
day batches and the day's last one triggers the night. The engine's keys are
in `Recipe configuration
<../../reference/configuration.rst#recipe-configuration>`__.

Run the example
---------------

`recipes/skillclaw <../../../recipes/skillclaw/README.md>`__ has the
prerequisites and the walkthrough. ``REEF_SC_RUN`` selects the run:
``skillclaw`` is the method run, ``frozen`` the control.

Results
-------

The gain criterion is fixed before any data is read: a category's gain counts
as real only when the method run's final day beats the control mean by more
than two control standard deviations, one sided, calibrated to a 5.6 percent
false positive rate per category.

On GLM-5.3-Flash served locally on four GPUs, six nights applied 13 skill
improvements and 8 creations, and the pool grew from 9 to 17 skills. In the
Productivity category the method run's final day beats the control mean by
+12.05 points (2.29 sd); on the subset of tasks scored on every day in both
runs the margin grows to +15.72 (2.50 sd). ``python run.py report`` recomputes
the numbers from the stored run data.

See also
--------

- `HTTP API <../../reference/http-api.rst#harness-artifacts>`__: pulling, pinning, and installing a published tree.
