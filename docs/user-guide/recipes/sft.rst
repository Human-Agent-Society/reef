SFT: supervised fine-tuning on demonstrations
=============================================

Supervised fine-tuning learns from demonstrations off policy: the
demonstration becomes the response the model is trained to produce for the
recorded request, token by token. In Reef it is the control arm of the
self-distillation comparisons (`#502
<https://github.com/Human-Agent-Society/reef/issues/502>`__): it takes the
report the self-distillation recipes take, a rollout's receipt and the
demonstration as ``context``, so a harness drives both methods with the
same feedback and the difference between them is the loss alone.

+-------------+------------------------------------------------------------+
| Evolves     | model weights                                              |
+-------------+------------------------------------------------------------+
| Signal      | one report with the demonstration as ``context`` per       |
|             | rollout                                                    |
+-------------+------------------------------------------------------------+
| Loss family | ``sft``                                                    |
+-------------+------------------------------------------------------------+
| Package     | ``recipes/sft/``                                           |
+-------------+------------------------------------------------------------+
| Processor   | reported feedback, singleton                               |
+-------------+------------------------------------------------------------+
| Needs       | GPUs                                                       |
+-------------+------------------------------------------------------------+
| Example     | `SFT on a skill stream                                     |
|             | <../../../recipes/sft/examples/skill_stream/README.md>`__  |
+-------------+------------------------------------------------------------+

What it does
------------

The harness sends a request through Reef, obtains a demonstration of the
response from somewhere else (a dataset, a stronger model), and reports the
demonstration against the request's receipt. The student's own response is
recorded and ignored. With the default ``batch_size`` of 1, each report is
one training step.

.. flow::
   :loop: the next request is served by the updated weights

   Rollout :: the student answers a request
   Demonstration :: a reference response for the same request
   Report :: the demonstration as ``context`` against the rollout's receipt
   Step* :: train the demonstration's tokens as the assistant turn of the request
   Version :: publish the updated weights to the engine

How Reef implements it
----------------------

The processor turns every ``TeacherContextReport`` into one
``TrajectoryItem`` whose tokens are the recorded request rendered with the
served model's chat template (``tokenizer_path``) followed by the
demonstration as the assistant message; the loss mask covers the
demonstration's tokens (the content, the end-of-turn token and the
template's turn separator), so the trained sequence is exactly the model's
own rendering of that answer. The student's engine log-probs and load spans
are dropped with its response.

The ``sft`` loss family is Slime's stock ``sft_loss`` over that row: every
masked token, unweighted. Advantages are refused rather than silently
discarded; a reward-weighted objective belongs to a family whose loss
consumes them.

The report contract
-------------------

A ``TeacherContextReport``: a report references one inference record and
carries the demonstration as ``metadata.context``. A ``score`` is
optional metadata; the recipe never trains on it.

.. code:: json

   {
     "references": ["<receipt of the student's request>"],
     "metadata": {"context": "<the demonstration>"}
   }

Configuration
-------------

.. config::

   batch_size | 1 | reports per optimizer step. Must equal the driver's ``--global-batch-size`` because each sample is its own data-parallel unit.
   tokenizer_path | required | the served model's tokenizer directory; it renders the request and the demonstration with the chat template the engine applied.
   max_staleness | 0 | accepted lag between the producing and serving version.

The Slime driver takes ``--loss-type sft_loss`` and
``--disable-compute-advantages-and-returns``; the optimizer is the driver's
(``training.options``). The family has no flags of its own.

Related guides
--------------

- `SFT on a skill stream <../../../recipes/sft/examples/skill_stream/README.md>`__:
  the sequential experiment this recipe is the control arm of, and what
  it forgets.
- `Train model weights from agent feedback <../evolve-your-model.rst>`__:
  set up the GPU stack and inspect published updates.
- `Loss families <../../developer-guide/loss-families.rst>`__: how a family
  such as ``sft`` plugs into the Slime backend.
