Choose a recipe for agent learning
==================================

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
| **Harness:** prompts,   | ``skillclaw``: grows a skill pool from           | not available                            |
| rules, skills, config   | the failures in its own served traffic           |                                          |
|                         |                                                  |                                          |
|                         | ``gepa``: rewrites the tree by reflecting on     |                                          |
|                         | the transcripts it already served                |                                          |
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
| A score per request, and a stronger model to  | `gepa <recipes/gepa.rst>`__                     | harness tree  | no         |
| reflect with                                  | (built into Reef)                               |               |            |
+-----------------------------------------------+-------------------------------------------------+---------------+------------+

How a recipe is selected
------------------------

A deployment serves exactly one recipe, named by ``reef.recipe`` in its config.
Every scenario it creates uses that recipe. Requests never name a recipe, and
scenario snapshots do not store one. The scenario header is the only routing a
caller provides. The artifact repository is therefore deployment-owned: do not
point deployments configured with different recipes at the same repository.

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

Run the recipe you chose
------------------------

Start with the `inference and feedback quickstart
<../getting-started/quickstart.rst>`__ if you have not sent traffic through
Reef yet. For a harness recipe, follow `Evolve agent prompts, rules, and skills
<evolve-your-harness.rst>`__. For a weight recipe, follow `Train model weights
from agent feedback <evolve-your-model.rst>`__. Each recipe page above provides
its own configuration and example.
