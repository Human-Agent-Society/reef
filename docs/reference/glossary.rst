Glossary
========

A recipe uses feedback from a scenario to update model weights or a harness.

Artifact
--------

An immutable version of a component, identified by ``artifact_id``. Serving and training can select different artifacts.

Component
---------

A part of a scenario, such as model weights or a harness, with its own artifact selection.

Feedback
--------

A score, text or structured data that describes a model exchange. The recipe defines its meaning.

Harness
-------

The code, instructions and tools an agent uses to call a model and complete tasks.

Job
---

One training run with starting weights or harness files, job data and recipe parameters.

Loss family
-----------

The implementation that computes a tensor loss from model outputs and the objective's training signal. See `Loss families <../developer-guide/loss-families.rst>`__.

OpenClaw-RL
-----------

A `recipe <../user-guide/recipes/openclawrl.rst>`__ that learns from later conversation turns.

Receipt
-------

The ID of a recorded inference exchange. A report uses receipts to identify the responses it grades.

Recipe
------

The choice of learning stages, objective and components for a scenario. The default recipe only records exchanges.

Recipe reference
----------------

The value of ``recipe.implementation``, such as ``reef.recipe.sft:SFTRecipe``. See `Recipe configuration <configuration-recipe.rst>`__.

Report
------

A record that submits feedback and identifies related inference records in ``references``.

Rollout
-------

One run of a task with a model and harness. A verifier can score the result.

Runtime
-------

The implementation that serves inference requests and, when supported, loads published weights.

Sample
------

One training input: a recorded trajectory or a task to run. ``job_samples`` counts these inputs.

SAO
---

`Single-Rollout Asynchronous Optimization <../user-guide/recipes/sao.rst>`__ trains on one graded rollout at a time.

Scenario
--------

An isolated workload with its own records, recipe, training state and artifact history.

Skill
-----

A set of agent instructions in a ``SKILL.md`` file under the harness tree's ``skills/`` directory.

Snapshot chain
--------------

The repository history that Reef uses to publish and recover snapshots. Selecting an existing artifact adds no snapshot.

Training objective
------------------

The part of a recipe that prepares the training signal and selects a loss family.

TTT-Discover
------------

A `test-time training recipe <../user-guide/recipes/tttd.rst>`__ that specializes a model as it searches for a solution to one problem.
