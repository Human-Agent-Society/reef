GEPA: reflective agent harness optimization
===========================================

GEPA (`arXiv:2507.19457 <https://arxiv.org/abs/2507.19457>`__) is reflective
prompt evolution. It trains no weights; it improves the text that an agent is
given. It keeps an archive of candidates. In each round it picks one candidate
as the parent, reads how the parent did on a few problems, and asks a stronger
model to write a better version. It keeps the new candidate only if it scores
higher than its parent on the same problems. Reef runs that search over a
whole harness tree, so the tree that the deployment serves improves over time.

+-------------+--------------------------------------------------------------+
| Evolves     | the text files of the harness tree: rules, skills, commands  |
+-------------+--------------------------------------------------------------+
| Signal      | one report per served request, batched into a minibatch      |
+-------------+--------------------------------------------------------------+
| Loss family | none (no weight training)                                    |
+-------------+--------------------------------------------------------------+
| Package     | ``recipes/gepa/``                                            |
+-------------+--------------------------------------------------------------+
| Pipeline    | reported feedback, producing trace items                     |
+-------------+--------------------------------------------------------------+
| Needs       | a Reef process, the ``pi`` binary, and a reflection model.   |
|             | Reef itself needs no GPU.                                    |
+-------------+--------------------------------------------------------------+
| Example     | ``recipes/gepa/examples/aime/``                              |
+-------------+--------------------------------------------------------------+

How it works
------------

GEPA is a method on the harness evolution engine: ``recipes.gepa:GEPA`` is a
``Method``, and a deployment runs it with
``reef.recipe.harness:HarnessEvolutionRecipe``. `Evolve your harness
<../evolve-your-harness.rst>`__ describes the engine. One step is one
reflection round:

1. **Pick a parent.** Each candidate is weighted by the number of validation
   problems on which it is the best. A candidate that is mediocre overall but
   best at one kind of problem still becomes a parent, which keeps the search
   exploring.
2. **Run a minibatch.** The parent runs on a small batch of training
   problems. When the parent is the served tree, the minibatch is free: the
   engine already holds that tree's real traffic and scores. Any other parent
   runs on the evaluation's worker pool.
3. **Reflect.** The reflection model reads the transcripts, with one sentence
   of feedback about each, and rewrites one component of the tree: one
   evolvable file.
4. **Test the child.** The child runs on the same problems as its parent.
   Only a child that scores strictly higher goes into the step's harness
   tree. Otherwise the tree is unchanged and the step publishes nothing, but
   the method's state keeps the iteration.
5. **Validate.** The step's evaluation scores the candidate on every problem
   in ``evolution.evaluation.tasks`` with the ``compare`` policy on the mean.
   Reef publishes the candidate only when its mean is strictly higher than
   the served tree's. Thus the served tree is the best candidate in the
   archive.

Every run goes through the score cache. A tree that already scored on a
problem does not run on it again; a run that could not complete runs again.
The method still counts each evaluation as GEPA does, so ``max_metric_calls``
means what it means upstream.

The archive is a file in the method's state directory. It holds each
candidate with its parent, its score on every validation problem, the
problems on which it is the best, the component to rewrite next, and the
metric calls spent. Reef commits the state with each step. An interrupted
search resumes from the last commit, and a step that fails cannot advance the
archive. The next step reads the per-problem scores and the publication
decision from the evolution history before it samples the next parent.

Reef does not import the upstream package at runtime. The reflection prompt
is copied under its MIT licence into ``recipes/gepa/reflection_prompt.md``.
The parent sampling rule and the per-problem front update are reimplemented
and tested against upstream (`Results`_).

Configuration
-------------

``recipes/gepa/examples/aime/gepa.yaml`` is the recipe config the driver boots.
It names ``reef.recipe.harness:HarnessEvolutionRecipe`` as its
``implementation``, GEPA as ``evolution.method`` with GEPA's own settings, and
GEPA's validation pass as ``evolution.evaluation``. Its ``pipeline`` section
sets the step's traces to GEPA's minibatch:

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
       the parts of the adapter's tree whose files the search may rewrite. Each must be a part whose format is ``text``, ``skill`` or ``command``. GEPA never rewrites config, code or native files, and the files of every other part stay fixed.

   ``evolution.method.config.feedback`` : the score restated
       a ``Feedback`` class. Its sentence about each example is the only signal that the reflection model reads. If the benchmark can explain why an answer was wrong, say so here.

   ``evolution.method.config.trainset_size`` : optional
       makes each iteration draw its parent from the seeded generator that also orders the training problems, in GEPA's own order. A driver that draws the same plan from the committed archive sends the problems that the plan names, and the method reads them as the step's traffic.

   ``evolution.method.config.reflection_model`` : reflection
       the name, under ``evolution.models``, of the model that reflects. When no model has this name, the method reflects with the model under test.

   ``evolution.evaluation.statistic`` : wins
       set ``mean`` for GEPA's rule. The default, ``wins``, counts task pairings.

`Recipe configuration <../../reference/configuration.rst#recipe-configuration>`__
documents the engine's own keys.

Run the example
---------------

`recipes/gepa/examples/aime <../../../recipes/gepa/examples/aime/README.md>`__
has the prerequisites and the full walkthrough. The driver embeds a Reef
service and runs every minibatch problem as a real run against the served
tree. At the end it runs two passes over a held-out split, one on the first
tree and one on the last, so that the two are comparable. ``--seed`` picks
the search seed. ``--dry-run`` boots the recipe and prints the plan without a
model call.

Results
-------

The example's README records four runs, two of this method and two of
upstream GEPA. The records are kept beside it.

``tests/recipes/gepa/test_gepa_fidelity.py`` checks the two exact rules. It
runs upstream ``gepa.optimize`` and this method on a synthetic task from the
same seed, the method through whole steps of the engine. It requires the same
candidates, parents, minibatches and validation means at every iteration. The
test runs where the upstream package is installed.

Related guides
--------------

- `Evolve your harness <../evolve-your-harness.rst>`__: the mechanism this method plugs into.
- `HTTP API <../../reference/http-api.rst#harness-artifacts>`__: pulling and pinning a published tree.
