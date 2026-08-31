Choosing a recipe
=================

A recipe is picked along two axes: **what it evolves**, and **how it learns**.

+-------------------------+--------------------------------------------------+------------------------------------------+
|                         | **Reactive:** learns from the traffic            | **Proactive:** generates its own         |
|                         | it already serves                                | attempts                                 |
+=========================+==================================================+==========================================+
| **Model weights**       | ``sao``: feedback on each attempt over a stream  | ``tttd``: repeated attempts at one       |
|                         | of tasks                                         | problem, at test time                    |
|                         |                                                  |                                          |
|                         | ``openclawrl``: multi-turn traffic,              |                                          |
|                         | reward read from the next state                  |                                          |
+-------------------------+--------------------------------------------------+------------------------------------------+
| **Harness:** prompts,   | ``skillclaw``: grows a skill pool                  | not available                            |
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
| Agent conversations without reports           | `openclawrl <recipes/openclawrl.rst>`__         | model weights | yes        |
+-----------------------------------------------+-------------------------------------------------+---------------+------------+
| Feedback on individual requests, and failures | `skillclaw <recipes/skillclaw.rst>`__           | harness tree  | no         |
| worth learning from                           | (built into Reef)                               |               |            |
+-----------------------------------------------+-------------------------------------------------+---------------+------------+

How a recipe is selected
------------------------

A deployment serves exactly one recipe, named by ``reef.recipe`` in its config.
Every scenario it creates binds to that recipe, permanently. Requests never name
a recipe. The scenario header is the only routing a caller provides.

.. code:: yaml

   reef:
     recipe: recipes.sao.recipe:SAORecipe
     batch_size: 1

``reef.recipe`` accepts the core value ``recipe``, a dotted class, or a preset.
Reef does not register or import learning methods. The ``recipes/`` tree in
this repository is a cookbook; installed method packages work the same way.
`Configuration <../reference/configuration.rst#recipe-configuration>`__
describes each spelling.

Every recipe has a checkpoint strategy, defaulting to ``EveryNVersions(1)``.
``checkpoint_every_n_versions`` is the shorter spelling in deployment YAML.
