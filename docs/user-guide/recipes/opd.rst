OPD: learn from a separate teacher on your own responses
=======================================================

On-policy distillation lets a frozen teacher score the tokens the student
actually generated. Reef trains the student's weights on that signal and
publishes them before the next batch is sampled. The method follows
`Thinking Machines Lab's write-up
<https://thinkingmachines.ai/blog/on-policy-distillation/>`__.

The recipe lives in ``recipes/opd/`` and uses Reef's Slime distillation
backend. Inference runs on SGLang; training runs on Megatron. No hosted
training API is required.

The learning loop
-----------------

.. flow::
   :loop: wait for publication before generating the next batch

   Sample :: the student answers prompts through Reef
   Report :: submit one empty teacher-context report per inference receipt
   Teacher :: score the exact recorded prompt and response IDs with frozen weights
   Train* :: minimize sampled-token reverse KL with one optimizer update
   Publish :: serve the updated student weights

Teacher and student see identical token sequences. The processor uses the
captured inference tokens, preserving thinking switches and assistant
prefixes. It rejects non-empty teacher context. Correctness scores may be
recorded as metadata, but do not supply the training objective.

Report contract
---------------

Each report references exactly one inference receipt:

.. code:: json

   {
     "references": ["<inference receipt>"],
     "metadata": {"teacher_context": ""}
   }

Configuration
-------------

Select ``recipes.opd.recipe:OPDRecipe``. The recipe's ``batch_size`` must
match the trainer's global batch size. ``max_staleness`` defaults to zero;
wait for each published update before sampling the next batch.

.. config::

   batch_size | 1 | number of responses in one optimizer step.
   tokenizer_path | required | local student tokenizer directory, shared by the teacher.
   max_teacher_tokens | 0 | maximum recorded sequence length; overflow reports are skipped and counted. 0 disables the check.

The Slime options belong in ``training.options``:

.. config::

   opd-teacher | separate | teacher weights loaded from the configured checkpoint.
   opd-teacher-checkpoint | required | compatible local Hugging Face or Megatron checkpoint.
   opd-divergence | reverse | sampled-token KL(student || teacher).
   opd-top-k | 1 | selects the sampled-token reverse-KL path; the sampled token probability is normalized over the full vocabulary.
   opd-teacher-update-rate | 0 | the teacher stays frozen.
   opd-importance-sampling-cap | 0 | no additional truncated importance-sampling correction for one on-policy update.

Also enable ``loss-type: custom_loss``, ``use-rollout-logprobs: true`` and
``disable-compute-advantages-and-returns: true``. The family rejects multiple
optimizer updates per rollout. The shared backend supports alternative
divergences and representations through the inherited ``opd-*`` flags;
changing these changes the experiment.

The separate checkpoint is loaded into the actor's architecture. Both models
must have compatible architecture, vocabulary and token IDs. A larger model
with only the same tokenizer is not supported by this loader. The teacher
copy consumes additional host memory; its forward pass reuses the actor's
GPUs without an optimizer update.

Math reproduction
-----------------

The `math example <../../../recipes/opd/examples/math/README.md>`__ targets
Qwen3.5-9B-Base after OpenThoughts3 SFT, with Qwen3.5-9B as teacher. This is
the model pair in the updated cookbook. The original blog's mathematical
reasoning curve used Qwen3-8B, so an experiment report must distinguish the
model change from its method and measured result.

The P1 acceptance criterion is an AIME'24 improvement over the frozen SFT
baseline of roughly the blog's ten percentage points. Startup tests alone
do not meet it. Record the initialization, command, configuration, per-version
predictions and scores, and learning curve before claiming reproduction.
