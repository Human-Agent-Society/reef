GEPA: reflective agent harness optimization
===========================================

Use GEPA to improve an agent's rules, skills, or commands without training
model weights. A reflection model proposes changes, and task scores decide
which candidates enter the archive and which tree serves.

Reef implements the search from `GEPA <https://arxiv.org/abs/2507.19457>`__
over a harness tree. It needs task feedback and a reflection model.

Run the example
---------------

Use the `AIME example <../../../recipes/gepa/examples/aime/README.md>`__
with Pi 0.84.2 and a reflection-model API key. Reef itself needs no GPU.
From the repository checkout:

.. code:: bash

   cd recipes/gepa/examples/aime
   uv pip install -e .
   REEF_PI_BINARY=/path/to/pi ./run.sh --dry-run
   export OPENAI_API_KEY=...
   REEF_PI_BINARY=/path/to/pi REEF_GEPA_WORKERS=4 ./run.sh

The dry run checks the binary and pinned datasets, loads the recipe, and
prints the plan without model calls. The live run evolves the tree, then
evaluates the first and final trees on the same held-out split.

Use ``--seed`` to select the search seed and ``--budget`` to reduce the
search budget. ``REEF_GEPA_WORKERS`` defaults to 128 persistent workers.
Reduce it to fit host memory and provider rate limits.

How it works
------------

A deployment selects ``recipes.gepa:GEPA`` as its harness evolution method.
Each step performs one reflection round:

1. Choose a parent from the archive. Selection favors candidates that lead
   on at least one validation problem.
2. Run the parent on a training minibatch. Reuse recorded traffic when the
   parent is the served tree.
3. Ask the reflection model to rewrite one evolvable file using the
   transcripts and feedback.
4. Evaluate the child on the same problems. Keep it only when its score
   exceeds the parent's score.
5. Validate the candidate. With ``policy: compare`` and ``statistic: mean``,
   publish only a mean score above the served tree's.

Successful cached evaluations are reused; incomplete runs can run again.
``max_metric_calls`` still counts evaluations according to GEPA's budget.
Reef commits the archive with each step. An interrupted search resumes from
the last commit; a failed step does not advance the archive.

Configuration
-------------

The example's ``gepa.yaml`` uses
``reef.recipe.harness:HarnessEvolutionRecipe``. Match ``job_samples`` to the
method's minibatch size:

.. code:: yaml

   evolution:
     seed: seed                          # the first tree, a directory
     models:
       reflection: {url: ..., model: ..., api_key_env: OPENAI_API_KEY}
     method:
       module: recipes.gepa:GEPA
       config:
         minibatch_size: 3
         seed: 0                         # the search seed
         skip_perfect_score: true
         perfect_score: 1.0
         max_metric_calls: 150
         components: [rules, skill]
         trainset_size: 45               # optional; plans GEPA's minibatch order
         feedback: harness.aime:AIMEFeedback  # optional; the score restated when absent
     evaluation:
       tasks: [...]                      # the validation set
       scorer: harness.aime:AIMEAnswerScorer
       policy: compare
       statistic: mean                   # GEPA's rule: a strictly better mean
   pipeline:
     sample2job:
       config:
         job_samples: 3                  # the third report closes the minibatch

.. container:: config

   ``evolution.method.config.components`` : [rules, skill]
       Parts the search may rewrite. Their formats must be ``text``, ``skill``
       or ``command``. Config, code, native files, and other parts stay fixed.

   ``evolution.method.config.feedback`` : the score restated
       A ``Feedback`` class that adds one sentence per example. Use it to
       explain a wrong answer when the benchmark provides that information.

   ``evolution.method.config.trainset_size`` : optional
       Enables GEPA's seeded minibatch order. The driver must send the
       problems selected by the committed archive's plan.

   ``evolution.method.config.reflection_model`` : reflection
       Model name under ``evolution.models``. If absent, reflection uses
       the model under test.

   ``evolution.evaluation.statistic`` : wins
       Set ``mean`` for GEPA's rule. The default counts task pairings.

`Recipe configuration <../../reference/configuration-recipe.rst#recipe-configuration>`__
documents the engine's own keys.

Results
-------

The `example README <../../../recipes/gepa/examples/aime/README.md>`__
records two Reef runs and two upstream GEPA runs, with saved results.
A synthetic fidelity test checks matching candidates, parents, minibatches,
and validation means. It does not establish equal performance on every task.

Related guides
--------------

- `Evolve your harness <../evolve-your-harness.rst>`__: the mechanism this method plugs into.
- `HTTP API <../../reference/http-api-harness.rst#harness-artifacts>`__: pulling and pinning a published tree.
