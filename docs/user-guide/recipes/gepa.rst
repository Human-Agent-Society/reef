gepa
====

GEPA (`arXiv:2507.19457 <https://arxiv.org/abs/2507.19457>`__) is reflective prompt
evolution. Instead of training weights, it improves the words an agent is given.
It keeps an archive of candidate prompts. On each round it picks one of them as
a parent, looks at how that parent actually did on a handful of problems, and
asks a stronger model to read those transcripts and write a better version. The
new candidate is kept only if it does better than its parent on the same
problems. Reef runs that search over a whole harness tree rather than over a
single string, so what improves over time is the composition the deployment
serves.

+-------------+--------------------------------------------------------------+
| Evolves     | the harness tree: rules, skills, agent commands              |
+-------------+--------------------------------------------------------------+
| Signal      | one report per served request, batched into a minibatch      |
+-------------+--------------------------------------------------------------+
| Loss family | none (no weight training)                                    |
+-------------+--------------------------------------------------------------+
| Package     | ``recipes/gepa/``                                            |
+-------------+--------------------------------------------------------------+
| Processor   | reported feedback, producing a ``TraceBatch``                |
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
composition and hands back a new version. The child is run on exactly the same
problems as its parent, and only a child that scores strictly better survives.
A child that fails costs only those few episodes, which is what makes the
search affordable; a child that wins is then measured properly, on the full
validation set.

The archive is the method's memory, and it holds rather more than the prompts
themselves. Alongside each candidate it keeps the parent it came from, the
score it got on every individual validation problem, which problems it is
currently the best at, which component should be rewritten next, and how much
of the metric-call budget has been spent. All of this lives in a single JSON
file per scenario, rewritten after every change, so a search that is
interrupted picks up where it left off rather than starting again.

How Reef implements it
----------------------

gepa is a method package on the harness evolution engine
(``reef/train/cordis_backend/``); `Evolve your harness
<../evolve-your-harness.rst>`__ describes the mechanism it runs on. The method
fills in both of the seams the engine offers.

``propose`` is the reflection round described above: pick a parent, rewrite one
component, test the child on the same problems the parent saw, and hand the
engine a mutation only when the child actually won.

``selection`` is the validation pass. This works out neatly, because by the
time the engine asks whether to publish a candidate, it has already scored both
the candidate and the incumbent on every problem in ``evolution.tasks``. Those
per-problem scores are exactly what the search needs. Recording them updates
the archive, and the tree is published only when the candidate's mean is
strictly better, which keeps the composition being served equal to the best
candidate the archive holds.

Nothing in Reef imports the upstream package at runtime. The reflection prompt
is copied word for word under its MIT licence and attributed in
``recipes/gepa/reflection.py``. The two rules that have to match exactly, which
are how a parent is sampled and how the per-problem fronts are updated, are
reimplemented here and checked against the real upstream code by a test
described under Results.

One detail is worth knowing because it affects cost. When the parent happens to
be the composition currently being served, its minibatch is free: the engine
has already batched that composition's real traffic, scores included, so there
is nothing to re-run. Any other parent has to be run from scratch, because no
traffic was ever served from it.

Configuration
-------------

``recipes/gepa/examples/aime/gepa.yaml`` is the recipe config the driver boots.
It names ``recipes.gepa.recipe:GEPARecipe`` as its ``implementation`` and adds
one block of its own under ``evolution``:

.. code:: yaml

   evolution:
     evaluate: harness.aime:evaluate
     feedback: harness.aime:feedback     # optional; the score restated when absent
     tasks: [...]                        # the validation set
     models:
       reflection: {url: ..., model: ..., api_key_env: OPENAI_API_KEY}
     gepa:
       archive: ${REEF_WORK}/gepa
       minibatch_size: 3
       seed: 0
       skip_perfect_score: true
       perfect_score: 1.0
       max_metric_calls: 150
       components: [rules, skill]

Two of these deserve a note. ``feedback`` is the sentence the reflection model
reads about each example, and it is the entire signal the reflection gets, so a
benchmark that can explain why an answer was wrong should say so here rather
than leaving the default, which only restates the score. ``components`` lists
the kinds of node the search is allowed to rewrite, which is how you keep it
away from parts of the tree that should stay fixed. If no reflection model is
declared, the method reflects using the model under test. The engine's own
configuration keys are documented in `Recipe configuration
<../../reference/configuration.rst#recipe-configuration>`__.

Run the example
---------------

`recipes/gepa/examples/aime <../../../recipes/gepa/examples/aime/README.md>`__
has the prerequisites and the full walkthrough. The driver embeds a Reef
service and runs every minibatch problem as a real episode against the served
tree, which means the method reflects on traffic the composition genuinely
produced rather than on a simulation of it. At the end it seals two passes over
a held-out split, one on the composition it started with and one on the
composition it finished with, so the two are directly comparable. ``--seed``
picks the search seed, and ``--dry-run`` boots the recipe and prints the plan
without making a single model call.

Results
-------

The example reproduces the upstream AIME quickstart, using the same 45 training
problems, 45 validation problems, and sealed 150-problem AIME-2025 split, the
same 150-call budget, and dated snapshots of the same task and reflection
models.

At seed 0 the method produced four candidates within 198 metric calls and
improved the sealed split from 26.67 percent to 46.67 percent. The upstream run
at the same seed went from 26.67 to 38.67. More telling than the final numbers
is that the search took the same route: at every round it reflected from the
same parent on the same training problems as the upstream run, and the two runs
scored the starting composition identically.

The selected scores differ because two of the model calls remain stochastic
even once the search path is pinned. The reflection model writes a different
prompt each time it is asked, and the task model scores the same prompt
differently from one run to the next, so a single seed of this benchmark has a
wide spread. Seed 1 of the method landed on the upstream run's twelve-point
gain exactly. Both records are kept under
``recipes/gepa/examples/aime/results/``.

Exact agreement is checked in the one place it can be. The test at
``tests/reef_service/test_gepa_fidelity.py`` drives the real upstream optimizer
and this method side by side on a synthetic task from a single seed, with the
scoring and the reflection made deterministic, and requires that they produce
the same candidates from the same parents on the same minibatches with the same
validation means, round after round.

See also
--------

- `Evolve your harness <../evolve-your-harness.rst>`__: the mechanism this method plugs into.
- `HTTP API <../../reference/http-api.rst#harness-artifacts>`__: pulling, pinning, and installing a published tree.
