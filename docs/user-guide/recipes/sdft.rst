SDFT: learn from demonstrations without forgetting
==================================================

Use Self-Distillation Fine-Tuning
(`SDFT <https://arxiv.org/abs/2601.19897>`__) to learn from demonstrations
on the student's own responses. The teacher reads a demonstration; the
student reads the original request. Reef's implementation requires Slime and GPUs.

Run the example
---------------

The `skill-stream example
<../../../recipes/sdft/examples/skill_stream/README.md>`__ trains Tool Use,
then Science Q&A. Use the Slime training environment with eight GPUs:
four for training and four for serving.

.. code:: bash

   cd recipes/sdft/examples/skill_stream
   hf download Qwen/Qwen2.5-7B-Instruct --local-dir ~/models/Qwen2.5-7B-Instruct
   ./run.sh

The example evaluates both skills every ten steps. Its README specifies the
protocol and the frozen-teacher settings.

How it works
------------

1. Sample a response through Reef.
2. Report a demonstration for the same request against its inference receipt.
3. Train on the student's tokens, using the demonstration-conditioned
   teacher's distributions.
4. Publish the updated weights before sampling the next job.

The teacher receives the demonstration in the final user message, or in a
new user message after a tool result. ``context_template`` controls this text.
The default loss is full-vocabulary forward KL. The teacher starts from the
initial model and moves 1% toward the policy after each step by default.

The skill-stream example freezes the teacher with update rate 0. Its moving
teacher runs drifted and collapsed. The recipe is on-policy; responses from
an artifact that no longer serves do not train.

Reef excludes samples whose teacher sequences exceed ``max_teacher_tokens``
or cannot render, and counts them in ``unrenderable_samples``.

The report contract
-------------------

A report references one inference record and carries the demonstration as
``metadata.teacher_context``. A ``score`` is optional metadata and the
recipe never trains on it.

.. code:: json

   {
     "references": ["<receipt of the student's request>"],
     "metadata": {"teacher_context": "<the demonstration>"}
   }

Configuration
-------------

Match ``job_samples`` to the optimizer batch size for one update per job:

.. code:: yaml

   recipe:
     implementation: recipes.sdft.recipe:SDFTRecipe
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

   ``pipeline.record2sample.config.max_teacher_tokens`` : 0
       Maximum teacher sequence length. Longer sequences are excluded.
       0 disables the limit. Match the trainer's context window.

   ``pipeline.record2sample.config.context_template`` : the reference's block
       the text added to the final user message. ``{context}`` marks where the demonstration goes.

   ``pipeline.sample2job.config.job_samples`` : required
       Rollouts per job. Match ``training.config.global_batch_size``.

   ``training.config.global_batch_size``
       Rollouts per optimizer step, passed to Slime as ``--global-batch-size``.

The record stage also accepts ``accept_multi_turn_policy_samples``,
``realign_threshold``, and ``scaffold_tolerance``. An invalid
``context_template``, unknown key, or stage setting under ``recipe.config``
stops startup.

Slime takes ``--loss-type custom_loss`` and
``--disable-compute-advantages-and-returns``, and refuses
``--use-rollout-logprobs``. Set ``--apply-chat-template-kwargs`` to the
serving engine's chat template arguments (`Chat template arguments
<../../reference/configuration-inference.rst#chat-template-arguments>`__). The family
adds its own flags:

.. container:: config

   ``--sdft-teacher`` : self
       ``self`` uses the model with the demonstration. ``separate`` uses
       ``--sdft-teacher-checkpoint``, which must have the same architecture.

   ``--sdft-divergence`` : forward
       ``forward`` is KL(teacher \|\| student); ``reverse`` is
       KL(student \|\| teacher). ``jsd`` uses generalized Jensen-Shannon
       divergence. ``--sdft-jsd-beta`` sets its teacher weight; default 0.5.

   ``--sdft-top-k`` : 0
       keep the teacher's top-K log-probs and its log-prob at the sampled token. 0 keeps the whole distribution.

   ``--sdft-teacher-update-rate`` : 0.01
       Fraction of policy weights mixed into the teacher after each step.
       1 copies the policy; 0 freezes the initial weights.

   ``--sdft-importance-sampling-cap`` : 2.0
       cap of the truncated importance-sampling weight. 0 disables the correction.

   ``--sdft-skip-response-tokens`` : 0
       response tokens at the start of every sample left out of the loss. The paper's runs used 3.

The default full-vocabulary forward KL does not support `score centering
<../../developer-guide/loss-families.rst#score-centering>`__; only the
sampled reverse-KL mode does.

Full-vocabulary scoring needs substantial host memory: about 2 GB per
16k-token response with a 248k vocabulary at tensor parallel size 4.
Limit ``max_teacher_tokens`` accordingly. A moving teacher also keeps another
copy of the actor's weights on the host per rank.

Results
-------

The example records one SDFT run and one SFT control on Qwen2.5-7B-Instruct.

.. image:: ../../assets/sdft/learning-curve.png
   :alt: Tool Use and Science Q&A accuracy during SDFT and SFT training

Both methods learn Tool Use. During that stage, Science Q&A drops below the
base model with SFT but not with SDFT. Both then learn Science Q&A;
SDFT retains more Tool Use performance. These runs do not estimate variance
across seeds.

Related guides
--------------

- `Inference and feedback quickstart <../../getting-started/quickstart.rst>`__:
  learn the request, receipt, and report workflow.
- `Train model weights from agent feedback <../evolve-your-model.rst>`__:
  set up the GPU stack and inspect published updates.
- `Loss families <../../developer-guide/loss-families.rst>`__: how a family
  such as ``sdft`` plugs into the Slime backend, and the distillation base
  it is built on.
