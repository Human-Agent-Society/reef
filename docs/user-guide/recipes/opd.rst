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

Teacher and student see identical token sequences. The record-to-task stage
(``OPDTasks``) uses the captured inference tokens, preserving thinking switches and assistant
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

Select ``recipes.opd.recipe:OPDRecipe``. ``max_staleness`` in
``recipe.config`` defaults to zero: wait for each published update before
sampling the next batch. The stage settings go in the deployment's top-level
``pipeline`` section:

.. code:: yaml

   pipeline:
     record2task:
       config:
         max_teacher_tokens: 8192
     task2batch:
       config:
         batch_size: 32

.. config::

   pipeline.task2batch.config.batch_size | 1 | responses in one optimizer step.
   pipeline.record2task.config.max_teacher_tokens | 0 | maximum recorded sequence length; a longer report is skipped and counted in ``teacher_overflow_reports``. 0 disables the check.

The Slime options belong in ``training.options``:

.. config::

   opd-teacher | separate | the teacher's weights come from ``opd-teacher-checkpoint``.
   opd-teacher-checkpoint | required | the teacher checkpoint, loaded into the actor's layout for the teacher pass.
   opd-divergence | reverse | sampled-token KL(student || teacher).
   opd-top-k | 1 | selects the sampled-token reverse-KL estimator; the sampled token's probability is normalized over the full vocabulary.
   opd-teacher-update-rate | 0 | the teacher stays frozen.
   opd-importance-sampling-cap | 0 | no truncated importance-sampling correction: one on-policy update per batch.
   opd-advantage-clip | 0 | symmetric bound on the teacher/student log-prob difference used in the sampled reverse-KL gradient. 0 disables clipping.

Also set ``loss-type: custom_loss``, ``use-rollout-logprobs: true`` and
``disable-compute-advantages-and-returns: true``. The family rejects multiple
optimizer updates per batch. The shared backend's other divergences and
representations are reachable through the inherited ``opd-*`` flags; changing
them changes the method.

Set ``opd-advantage-clip: 1.0`` to bound each token's advantage to
``[-1, 1]`` before importance-sampling weights and sample weights are applied.
Score centering uses the same bounded advantage. This option changes the
gradient; the reported divergence remains the unclipped estimate. It requires
``opd-divergence: reverse``, positive ``opd-top-k``, and
``opd-top-k-distribution: renormalized``. The bound is fixed, rather than
derived from running statistics.

The separate checkpoint is swapped into the actor's architecture for the
teacher pass, so teacher and student must share the architecture, parallel
layout and token ids; a larger model with only the same tokenizer runs on a
teacher engine instead. The teacher copy takes additional host memory and
reuses the actor's GPUs without an optimizer update. In LoRA mode the
student's adapter is cleared while the teacher's weights are loaded.

Teacher engine
--------------

A teacher that shares only the student's token ids, such as Qwen3-32B for a
Qwen3-8B student, is served as a frozen SGLang engine on its own GPUs. The
deployment's ``teacher`` section declares it and replaces
``opd-teacher-checkpoint``:

.. code-block:: yaml

   teacher:
     model-path: Qwen/Qwen3-32B
     num-gpus: 2
     port: 30001
     options:
       context-length: 32768
   training:
     options:
       opd-teacher: separate
       opd-top-k: 1
       rollout-temperature: 1.0

``num-gpus`` reserves the teacher's GPUs beyond the student's and sets its
tensor parallel size, ``port`` is its internal endpoint, ``timeout`` bounds one
scoring request, and ``options`` are native SGLang options. Before each
optimizer step the bridge sends every recorded prompt and response to the
engine and stores the teacher's log-probability of each sampled token, so the
loss stays the sampled-token reverse KL. The engine path requires rollout
temperature 1, because SGLang's input log-probs are not tempered, and the
teacher-selected top-K representation; the teacher's tokenizer must map tokens
to the student's ids, and its context limit must cover the longest recorded
sequence. A scoring failure fails the step before it starts, so no partially
scored batch trains. ``perf/distill_teacher_time`` reports the scoring time of
each step in seconds.

Math example
------------

The `math example <../../../recipes/opd/examples/math/README.md>`__ distils
Qwen3-32B into a Qwen3-8B-Base student fine-tuned on OpenThoughts3 math
through the teacher engine and evaluates on AIME'24, the model pair of the
original write-up. The page documents the protocol, the eight-GPU
configuration and the results.
