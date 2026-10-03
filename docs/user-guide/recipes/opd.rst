OPD: learn from a separate teacher on your own responses
========================================================

On-policy distillation lets a frozen teacher score the tokens the student
actually generated. Reef trains the student's weights on that signal and
publishes them before the next batch is sampled. The method follows
`Thinking Machines Lab's write-up
<https://thinkingmachines.ai/blog/on-policy-distillation/>`__.

The recipe lives in ``recipes/opd/`` and uses Reef's Slime distillation
backend. Inference runs on SGLang; training runs on Megatron.

The learning loop
-----------------

.. flow::
   :loop: wait for publication before generating the next batch

   Sample :: the student answers prompts through Reef
   Report :: submit one empty teacher-context report per inference receipt
   Teacher :: score the exact recorded prompt and response ids with frozen weights
   Train* :: minimize sampled-token reverse KL with one optimizer update
   Publish :: serve the updated student weights

Teacher and student see identical token sequences. The processor uses the
captured inference tokens, preserving thinking switches and assistant
prefixes, and loads no tokenizer. It rejects a non-empty teacher context. A
reported score is kept as metadata and does not enter the loss.

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

Select ``recipes.opd.recipe:OPDRecipe``. ``batch_size`` responses form one
optimizer step. ``max_staleness`` defaults to zero: wait for each published
update before sampling the next batch.

.. config::

   batch_size | 1 | responses in one optimizer step.
   max_teacher_tokens | 0 | maximum recorded sequence length; a longer report is released and counted in ``teacher_overflow_reports``. 0 disables the check.

The Slime options belong in ``training.options``:

.. config::

   opd-teacher | separate | the teacher's weights come from ``opd-teacher-checkpoint``.
   opd-teacher-checkpoint | required | the teacher checkpoint, loaded into the actor's layout for the teacher pass.
   opd-divergence | reverse | sampled-token KL(student || teacher).
   opd-top-k | 1 | selects the sampled-token reverse-KL estimator; the sampled token's probability is normalized over the full vocabulary.
   opd-teacher-update-rate | 0 | the teacher stays frozen.
   opd-importance-sampling-cap | 0 | no truncated importance-sampling correction: one on-policy update per batch.

Also set ``loss-type: custom_loss``, ``use-rollout-logprobs: true`` and
``disable-compute-advantages-and-returns: true``. The family rejects multiple
optimizer updates per batch. The shared backend's other divergences and
representations are reachable through the inherited ``opd-*`` flags; changing
them changes the method.

The separate checkpoint is swapped into the actor's architecture for the
teacher pass, so teacher and student must share the architecture, parallel
layout and token ids; a larger model with only the same tokenizer is not
supported by this loader. The teacher copy takes additional host memory and
reuses the actor's GPUs without an optimizer update. In LoRA mode the
student's adapter is cleared while the teacher's weights are loaded.

Math example
------------

The `math example <../../../recipes/opd/examples/math/README.md>`__ distils
Qwen3.5-9B into Qwen3.5-9B-Base after OpenThoughts3 SFT and evaluates on
AIME'24, the model pair of the updated cookbook; the original write-up used
Qwen3-8B. The page documents the protocol, the deployment configurations and
the measured results.
