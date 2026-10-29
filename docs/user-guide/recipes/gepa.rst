GEPA: reflective agent harness optimization
===========================================

GEPA (`arXiv:2507.19457 <https://arxiv.org/abs/2507.19457>`__) is reflective prompt
evolution. Instead of training weights, it improves the words an agent is given.
It keeps an archive of candidate prompts. On each round it picks one of them as
a parent, looks at how that parent actually did on a handful of problems, and
asks a stronger model to read those transcripts and write a better version. The
new candidate is kept only if it does better than its parent on the same
problems. Reef runs that search over a whole harness tree rather than over a
single string, so what improves over time is the tree the deployment serves.

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

What it does
------------

One round of the search is one reflection.

It begins by choosing which candidate to improve. The choice is not simply the
best one so far. Each candidate is weighted by how many individual problems it
is the best at, so a candidate that is mediocre overall but unusually good at
one kind of problem still gets its turn as a parent. This is what keeps the
search exploring instead of hill-climbing into the first thing that worked.

The parent is then run on a small batch of training problems, and those
transcripts, together with a sentence about what went wrong in each one, are
handed to the reflection model. That model rewrites one component of the
tree, one evolvable file, and hands back a new version. The child is run on
exactly the same problems as its parent, and only a child that scores
strictly better survives. A child that fails costs only those few rollouts,
which is what makes the search affordable; a child that wins is then measured
properly, on the full validation set.

The archive is the method's memory, and it holds rather more than the prompts
themselves. Alongside each candidate it keeps the parent it came from, the
score it got on every individual validation problem, which problems it is
currently the best at, which component should be rewritten next, and how much
of the metric-call budget has been spent. The archive is a file in the
method's state directory, which Reef commits with each step, so a search that
is interrupted picks up from the last commit rather than starting again, and a
step that fails or is never committed cannot advance it.

How Reef implements it
----------------------

gepa is a method on the harness evolution engine: ``recipes.gepa:GEPA`` is a
``Method``, and a deployment runs it with the harness evolution recipe
``reef.recipe.harness:HarnessEvolutionRecipe``. `Evolve your harness
<../evolve-your-harness.rst>`__ describes the mechanism it runs on.

Each step runs the reflection round described above: pick a parent, rewrite
one component, test the child on the same problems the parent saw, and write
the child into the step's harness tree only when the child actually won. A
round that ends without a winner leaves the tree unchanged, so the step
publishes nothing, but the method's state keeps the iteration it spent.

The step's evaluation is the validation pass. It scores the candidate on every
problem in ``evolution.evaluation.tasks`` with the ``compare`` policy on the
mean, and publishes the candidate only when its mean is strictly better than
the served tree's. The served tree's per-problem scores come from the score
cache, so they are the scores it was validated with, which keeps the tree being
served equal to the best candidate the archive holds. A rollout that could not
run is the exception: the cache does not keep it, so the next evaluation runs
that problem again. The next step reads the per-problem scores and the
decision from the evolution history and folds them into the archive before it
samples the next parent.

Nothing in Reef imports the upstream package at runtime. The reflection prompt
is copied word for word under its MIT licence into
``recipes/gepa/reflection_prompt.md``, and ``recipes/gepa/reflection.py`` names
its source. The two rules that have to match exactly, which are how a parent is
sampled and how the per-problem fronts are updated, are reimplemented here and
checked against the real upstream code by a test described under Results.

One detail is worth knowing because it affects cost. When the parent happens to
be the tree currently being served, its minibatch is free: the engine has
already collected that tree's real traffic, scores included, so there is nothing
to re-run. Any other parent, and every child, runs as rollouts on the
evaluation's worker pool, because no traffic was ever served from it. Every
rollout goes through the score cache, so a tree that already scored on a
problem is not run on it again; a rollout that could not run runs again. The
method still counts each evaluation as GEPA does, so the metric-call budget
means what it means upstream.

Configuration
-------------

``recipes/gepa/examples/aime/gepa.yaml`` is the recipe config the driver boots.
It names ``reef.recipe.harness:HarnessEvolutionRecipe`` as its
``implementation``, GEPA as ``evolution.method`` with GEPA's own settings, and
GEPA's validation pass as ``evolution.evaluation``:

.. code:: yaml

   evolution:
     seed: seed                          # the first tree, a directory
     batch_size: 3                       # the third report closes the minibatch
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

Four of these deserve a note. ``feedback`` names a ``Feedback`` class whose
sentence the reflection model reads about each example, and it is the entire
signal the reflection gets, so a benchmark that can explain why an answer was
wrong should say so here rather than leaving the default, which only restates
the score. ``components`` names the parts of the adapter's tree whose files
the search may rewrite, ``[rules, skill]`` by default; the files of every
other part stay fixed. Each name must be a part whose format is ``text``,
``skill`` or ``command``, such as ``rules``, ``skill`` and ``command``: GEPA
never rewrites config, code or native files. ``trainset_size`` makes each
iteration draw its parent from the seeded generator that also orders the
training problems, in GEPA's own order: a driver that draws the same plan
from the committed archive sends the problems the plan names, and the method
reads them as the step's traffic.
``statistic: mean`` is what makes the evaluation GEPA's: the default,
``wins``, counts task pairings instead. If no reflection model is declared,
the method reflects using the model under test; ``reflection_model`` names
another one. The engine's own configuration keys are documented in `Recipe
configuration <../../reference/configuration.rst#recipe-configuration>`__.

Run the example
---------------

`recipes/gepa/examples/aime <../../../recipes/gepa/examples/aime/README.md>`__
has the prerequisites and the full walkthrough. The driver embeds a Reef
service and runs every minibatch problem as a real rollout against the served
tree, which means the method reflects on traffic the tree genuinely produced
rather than on a simulation of it. At the end it seals two passes over a
held-out split, one on the tree it started with and one on the tree it
finished with, so the two are directly comparable. ``--seed``
picks the search seed, and ``--dry-run`` boots the recipe and prints the plan
without making a single model call.

Results
-------

The example's README records four runs against upstream GEPA, two of each,
and the records are kept beside it.

``tests/recipes/gepa/test_gepa_fidelity.py`` checks the two exact rules. It
runs upstream ``gepa.optimize`` and this method side by side on a synthetic
task from the same seed, the method through whole steps of the engine, and
requires the same candidates, parents, minibatches and validation means at
every iteration. It runs where the upstream package is installed.

See also
--------

- `Evolve your harness <../evolve-your-harness.rst>`__: the mechanism this method plugs into.
- `HTTP API <../../reference/http-api.rst#harness-artifacts>`__: pulling and pinning a published tree.
