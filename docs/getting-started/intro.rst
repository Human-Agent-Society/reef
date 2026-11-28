Reef: continual learning for AI agents
======================================

Reef records model calls and feedback. A recipe uses that feedback to train weights or update an agent's harness.

Why Reef
--------

Use your existing tests, graders or user reports to decide what the agent should learn.
When Reef serves an artifact, the inference record identifies the version that answered.

.. figure:: ../assets/reef-overview-light.svg
   :alt: The harness sends requests and feedback through Reef. Reef returns model responses and publishes new versions.

What Reef evolves
-----------------

- `Model weights <../user-guide/evolve-your-model.rst>`__ learn from training samples.
- `Harnesses <../user-guide/evolve-your-harness.rst>`__ change rules, prompts, skills or code. Reef compares candidates on your tasks.

Start using Reef
----------------

`Record your first request and report <quickstart.rst>`__, then
`choose a recipe <../user-guide/recipes.rst>`__ for your workload.
