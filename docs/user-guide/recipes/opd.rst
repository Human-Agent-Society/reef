OPD: learn from a separate teacher on your own responses
========================================================

Use on-policy distillation (OPD) to train a student on its own responses,
scored by a frozen teacher. It requires GPUs and the Slime backend.
The method follows `Thinking Machines Lab's write-up
<https://thinkingmachines.ai/blog/on-policy-distillation/>`__.

Run the example
---------------

The `math example <../../../recipes/opd/examples/math/README.md>`__ trains
an OpenThoughts3-fine-tuned Qwen3-8B-Base student with a Qwen3-32B teacher.
It requires the Slime GPU image and eight 80 GB GPUs.

Follow the example's setup commands to download the pinned models, copy the
teacher's chat template, and prepare the datasets. Then run:

.. code:: bash

   export OPD_MODEL_PATH=/work/models/Qwen3-8B-Base-SFT
   export OPD_TEACHER_PATH=/work/models/Qwen3-32B
   export OPD_RUN_DIR=/work/opd
   recipes/opd/examples/math/run.sh

The driver collects one job's responses from a single snapshot, submits their
reports, and waits for one committed update. Results go to
``$OPD_RUN_DIR/results``. The README gives the AIME'24 protocol and results.

How it works
------------

1. Sample a job's responses through Reef.
2. Report each inference receipt with an empty ``teacher_context``.
3. Score the student's tokens with the frozen teacher.
4. Train the student, publish its weights, and then sample the next job.

Teacher and student receive the same token sequence. Match the training
component's `chat template arguments
<../../reference/configuration-inference.rst#chat-template-arguments>`__
to serving, including its thinking mode.

OPD is on-policy. A response from an artifact that no longer serves does
not train. Wait for publication before generating the next job's responses.

Report contract
---------------

Each report references exactly one inference receipt. The stage rejects a
non-empty ``teacher_context``. A reported score is metadata only and does not
enter the loss.

.. code:: json

   {
     "references": ["<inference receipt>"],
     "metadata": {"teacher_context": ""}
   }

Configuration
-------------

Set ``job_samples`` to the optimizer batch size for one update per job:

.. code:: yaml

   recipe:
     implementation: recipes.opd.recipe:OPDRecipe
   pipeline:
     record2sample:
       config:
         max_teacher_tokens: 8192
     sample2job:
       config:
         job_samples: ${training.config.global_batch_size}
   training:
     config:
       global_batch_size: 32

.. container:: config

   ``pipeline.sample2job.config.job_samples`` : required
       Responses per job. Match ``training.config.global_batch_size``.

   ``training.config.global_batch_size``
       Responses per optimizer step. Collect this many before waiting for publication.

   ``pipeline.record2sample.config.max_teacher_tokens`` : 0
       Maximum rendered sequence length. Longer samples are excluded and counted
       in ``unrenderable_samples``. 0 disables the limit.

The Slime options belong in ``training.options``:

.. container:: config

   ``opd-teacher`` : separate
       the teacher's weights come from ``opd-teacher-checkpoint``.

   ``opd-teacher-checkpoint`` : required
       the teacher checkpoint, loaded into the actor's layout for the teacher pass.

   ``opd-divergence`` : reverse
       sampled-token KL(student \|\| teacher).

   ``opd-top-k`` : 1
       selects the sampled-token reverse-KL estimator; the sampled token's probability is normalized over the full vocabulary.

   ``opd-teacher-update-rate`` : 0
       the teacher stays frozen.

   ``opd-importance-sampling-cap`` : 0
       no truncated importance-sampling correction: one on-policy update per optimizer batch.

   ``opd-advantage-clip`` : 0
       symmetric bound on the teacher/student log-prob difference used in the sampled reverse-KL gradient. 0 disables clipping.

Also set ``loss-type: custom_loss`` and
``disable-compute-advantages-and-returns: true``. Slime refuses
``use-rollout-logprobs``. The inherited ``opd-*`` flags reach the shared
backend's other divergences and representations; changing them changes the
method.

Set ``opd-advantage-clip: 1.0`` to bound each token's advantage to
``[-1, 1]`` before importance-sampling weights and sample weights are applied.
Score centering uses the same bounded advantage. The option changes the
gradient; the reported divergence remains the unclipped estimate. It requires
``opd-divergence: reverse``, a positive ``opd-top-k``, and
``opd-top-k-distribution: renormalized``.

A teacher checkpoint must share the student's architecture, parallel layout,
and token IDs. It uses extra host memory and reuses the actor's GPUs.
In LoRA mode, the teacher pass clears the student's adapter.
Use a separate teacher engine when only the token IDs match.

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

.. container:: config

   ``teacher.num-gpus``
       reserves the teacher's GPUs beyond the student's and sets its tensor parallel size.

   ``teacher.port``
       the teacher's internal endpoint.

   ``teacher.timeout``
       bounds one scoring request.

   ``teacher.options``
       native SGLang options.

Before training, Slime scores every rendered prompt and response on the
teacher engine. This keeps the sampled-token reverse-KL loss. Requirements:

- Rollout temperature 1, because SGLang's input log-probs are not tempered.
- The teacher-selected top-K representation.
- A teacher tokenizer that maps tokens to the student's ids.
- A teacher context limit that covers the longest rendered sequence.

A scoring failure fails the step before it starts, so no partially scored job
trains. ``perf/distill_teacher_time`` reports the scoring time of each step in
seconds.
