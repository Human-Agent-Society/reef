Choosing a recipe
=================

A recipe is picked along two axes: **what it evolves**, and **how it learns**.

+-------------------------+--------------------------------------------------+------------------------------------------+
|                         | **Reactive** — learns from the traffic           | **Proactive** — generates its own        |
|                         | it already serves                                | attempts                                 |
+=========================+==================================================+==========================================+
| **Model weights**       | ``sao`` — feedback on each attempt over a stream | ``tttd`` — repeated attempts at one      |
|                         | of tasks                                         | problem, at test time                    |
|                         |                                                  |                                          |
|                         | ``openclawrl`` — multi-turn traffic,             |                                          |
|                         | reward read from the next state                  |                                          |
+-------------------------+--------------------------------------------------+------------------------------------------+
| **Harness** — prompts,  | ``harness_evolve`` — proposes edits from         | —                                        |
| rules, skills, config   | the failures in its own served traffic           |                                          |
+-------------------------+--------------------------------------------------+------------------------------------------+

Pick by the signal your workload can produce.

+-----------------------------------------------+-------------------------------------------------+---------------+------------+
| The signal you have                           | Recipe                                          | Evolves       | Needs GPUs |
+===============================================+=================================================+===============+============+
| Feedback on each attempt, over a stream of    | `sao <recipes/sao.rst>`__                       | model weights | yes        |
| tasks                                         |                                                 |               |            |
+-----------------------------------------------+-------------------------------------------------+---------------+------------+
| A fixed grid of sibling attempts at one       | `tttd <recipes/tttd.rst>`__                     | model weights | yes        |
| problem                                       |                                                 |               |            |
+-----------------------------------------------+-------------------------------------------------+---------------+------------+
| Nothing on the wire — just agent              | `openclawrl <recipes/openclawrl.rst>`__         | model weights | yes        |
| conversations                                 |                                                 |               |            |
+-----------------------------------------------+-------------------------------------------------+---------------+------------+
| Feedback on individual requests, and failures | `harness_evolve <recipes/harness-evolve.rst>`__ | harness tree  | no         |
| worth learning from                           |                                                 |               |            |
+-----------------------------------------------+-------------------------------------------------+---------------+------------+
| None yet — you only want the records          | ``recipe``, the base kind                       | nothing       | no         |
+-----------------------------------------------+-------------------------------------------------+---------------+------------+

How a recipe is selected
------------------------

A deployment serves exactly one recipe, named by ``reef.recipe`` in its config.
Every scenario it creates binds to that recipe, permanently. Requests never name
a recipe — the scenario header is the only routing a caller provides.

.. code:: yaml

   reef:
     recipe: sao          # a bundled kind
     batch_size: 1

Naming — bundled kind, dotted class, or preset — and the per-kind config shapes
are in `Configuration <../reference/configuration.rst#recipe-configuration>`__.

Every recipe has a checkpoint strategy, defaulting to ``EveryNVersions(1)``.
``checkpoint_every_n_versions`` is the shorter spelling in deployment YAML.

See also
--------

- `tttd <recipes/tttd.rst>`__ includes the bundled TTT-Discover example,
  formal circle-packing results, and recovery details.
- `Write a recipe <../developer-guide/write-a-recipe.rst>`__ — when none of them fits.
