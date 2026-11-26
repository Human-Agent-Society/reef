GEPA: reflective agent harness optimization
===========================================

Use `GEPA <https://arxiv.org/abs/2507.19457>`__ to improve rules, skills,
or commands without training model weights. A reflection model proposes
changes. Task scores select the candidates and the tree that serves.

Run the example
---------------

Use the `AIME example <../../../recipes/gepa/examples/aime/README.md>`__
with Pi 0.84.2 and a reflection-model API key. Reef needs no GPU.
From the repository checkout:

.. code:: bash

   cd recipes/gepa/examples/aime
   uv pip install -e .
   REEF_PI_BINARY=/path/to/pi ./run.sh --dry-run
   export OPENAI_API_KEY=...
   REEF_PI_BINARY=/path/to/pi REEF_GEPA_WORKERS=4 ./run.sh

The dry run checks setup without model calls. The live run evolves the
tree and evaluates its first and final versions on one held-out split.
Use ``--seed`` and ``--budget`` to change the search.
``REEF_GEPA_WORKERS`` defaults to 128; reduce it for memory or rate limits.

How it works
------------

GEPA selects an archived parent, evaluates a training minibatch, and asks
reflection to rewrite one file. A child enters validation only after beating
its parent on that minibatch. Publication requires beating the served tree.

Successful evaluations are cached. Incomplete evaluations can run again;
``max_metric_calls`` still controls the budget. Each commit saves the archive.
A failed step does not advance it; restart resumes from the last commit.

Configuration
-------------

Start from the example's ``gepa.yaml``. Match ``job_samples`` to
``minibatch_size`` and select ``mean`` for GEPA's publication rule:

.. code:: yaml

   evolution:
     method:
       module: recipes.gepa:GEPA
       config:
         minibatch_size: 3
         seed: 0
         max_metric_calls: 150
         components: [rules, skill]
     evaluation:
       policy: compare
       statistic: mean
   pipeline:
     sample2job:
       config:
         job_samples: 3

Under ``evolution.method.config``:

- ``components`` defaults to ``[rules, skill]``. Only ``text``, ``skill``,
  and ``command`` formats can change; other parts stay fixed.
- ``feedback`` optionally names a ``Feedback`` class. Without it, feedback
  restates the score.
- ``trainset_size`` enables seeded minibatch ordering. The driver must follow
  the committed archive's problem plan.
- ``reflection_model`` defaults to ``reflection`` under ``evolution.models``.
  If absent, reflection uses the model under test.

``evolution.evaluation.statistic`` defaults to ``wins``; ``mean`` requires a
strictly higher mean. See `Recipe configuration
<../../reference/configuration-recipe.rst#recipe-configuration>`__ for the
engine's settings.

Results
-------

The `example results <../../../recipes/gepa/examples/aime/README.md>`__
contain two Reef runs, two upstream runs, and a synthetic fidelity check.
They do not establish equal performance on every task.

Related guides
--------------

- `Evolve your harness <../evolve-your-harness.rst>`__: configure the deployment.
- `Harness artifacts <../../reference/http-api-harness.rst#harness-artifacts>`__: pull or pin a tree.
