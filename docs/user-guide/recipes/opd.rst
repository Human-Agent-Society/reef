OPD: learn from a separate teacher on your own responses
========================================================

On-policy distillation lets a frozen teacher score the tokens the student
actually generated. Reef trains the student's weights on that signal and
publishes them before the next job's responses are sampled. The method follows
`Thinking Machines Lab's write-up
<https://thinkingmachines.ai/blog/on-policy-distillation/>`__.

+-------------+------------------------------------------------------------+
| Evolves     | model weights                                              |
+-------------+------------------------------------------------------------+
| Signal      | one report with an empty ``teacher_context`` per response  |
+-------------+------------------------------------------------------------+
| Loss family | ``opd``                                                    |
+-------------+------------------------------------------------------------+
| Package     | ``recipes/opd/``                                           |
+-------------+------------------------------------------------------------+
| Pipeline    | reported feedback, singleton                               |
+-------------+------------------------------------------------------------+
| Needs       | GPUs and the Slime training backend                        |
+-------------+------------------------------------------------------------+
| Example     | ``recipes/opd/examples/math/``                             |
+-------------+------------------------------------------------------------+

How it works
------------

.. container:: flow

   Sample
       the student answers prompts through Reef

   Report
       submit one empty teacher-context report per inference receipt

   Teacher
       score the student's rendered prompt and response ids with frozen weights

   **Train**
       minimize sampled-token reverse KL with one optimizer update

   Publish
       serve the updated student weights

   :Loop: wait for publication before generating the next job's responses

Teacher and student see identical token sequences. The record-to-sample stage,
``OPDSamples``, makes one sample from each report. The training component
renders the student's sequence with the model's chat template and gives the
teacher that same sequence. Give the component the serving engine's chat
template arguments (`Chat template arguments
<../../reference/configuration-inference.rst#chat-template-arguments>`__), so that the
rendered sequence keeps the thinking switches of serving.

The stage is on-policy: a response of an artifact that no longer serves does
not train. Wait for each published update before you sample the next job's
responses.

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

The stage settings go in the deployment's top-level ``pipeline`` section.
``job_samples`` is the responses of one job, and Slime's
``--global-batch-size`` is the responses of one optimizer step:

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
       responses in one job. Set it to ``training.config.global_batch_size`` so that one job is one step.

   ``training.config.global_batch_size``
       responses in one optimizer step. Sample this many responses before you wait for publication, so that one job is one step.

   ``pipeline.record2sample.config.max_teacher_tokens`` : 0
       maximum rendered sequence length; the training component leaves out a longer sample and counts it in ``unrenderable_samples``. 0 disables the check.

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

.. container:: config

   ``teacher.num-gpus``
       reserves the teacher's GPUs beyond the student's and sets its tensor parallel size.

   ``teacher.port``
       the teacher's internal endpoint.

   ``teacher.timeout``
       bounds one scoring request.

   ``teacher.options``
       native SGLang options.

Before each optimizer step, Slime sends every rendered prompt and response to
the engine and stores the teacher's log-probability of each sampled token, so
the loss stays the sampled-token reverse KL. The engine path has these
requirements:

- Rollout temperature 1, because SGLang's input log-probs are not tempered.
- The teacher-selected top-K representation.
- A teacher tokenizer that maps tokens to the student's ids.
- A teacher context limit that covers the longest rendered sequence.

A scoring failure fails the step before it starts, so no partially scored job
trains. ``perf/distill_teacher_time`` reports the scoring time of each step in
seconds.

Run the example
---------------

The `math example <../../../recipes/opd/examples/math/README.md>`__ distils
Qwen3-32B into a Qwen3-8B-Base student fine-tuned on OpenThoughts3 math
through the teacher engine and evaluates on AIME'24, the model pair of the
original write-up. The page documents the protocol, the eight-GPU
configuration and the results.
