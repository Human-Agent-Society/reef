SDFT: learn from demonstrations without forgetting
==================================================

Self-Distillation Fine-Tuning (`arXiv:2601.19897
<https://arxiv.org/abs/2601.19897>`__) learns from demonstrations on policy.
The served model, shown a demonstration in its prompt, is the teacher; the
same model without it is the student; the student samples the response and
the loss pulls its next-token distributions toward the teacher's at every
position of that sample. Compared with supervised fine-tuning on the same
demonstrations, the paper reports higher new-task accuracy and far less
forgetting.

+-------------+------------------------------------------------------------+
| Evolves     | model weights                                              |
+-------------+------------------------------------------------------------+
| Signal      | one report with the demonstration as ``teacher_context``   |
|             | per rollout                                                |
+-------------+------------------------------------------------------------+
| Loss family | ``sdft``                                                   |
+-------------+------------------------------------------------------------+
| Package     | ``recipes/sdft/``                                          |
+-------------+------------------------------------------------------------+
| Processor   | reported feedback, singleton                               |
+-------------+------------------------------------------------------------+
| Needs       | GPUs, and a backend that captures tokens and log-probs     |
+-------------+------------------------------------------------------------+
| Example     | `SDFT on a skill stream                                    |
|             | <../../../recipes/sdft/examples/skill_stream/README.md>`__ |
+-------------+------------------------------------------------------------+

What it does
------------

The harness sends a request through Reef, obtains a demonstration of the
response from somewhere else (a reference solution, a stronger model), and
reports the demonstration against the request's receipt. Nothing is executed
for the demonstrator; the student's response is the one the environment saw.
With the default ``batch_size`` of 1, each report is one training step.

.. flow::
   :loop: the next request is served by the updated weights

   Rollout :: the student answers a request
   Demonstration :: a reference response for the same request
   Report :: the demonstration as ``teacher_context`` against the rollout's receipt
   Step* :: distil the demonstration-conditioned teacher on the student's own tokens
   Version :: publish the updated weights to the engine

How Reef implements it
----------------------

The processor is the shared ``DistillProcessor``
(`Processors <../../developer-guide/processors.rst>`__): it turns every
``TeacherContextReport`` into one ``TrajectoryItem`` carrying the student's
recorded tokens plus ``teacher_tokens``, the teacher's request rendered with
the served model's chat template (``tokenizer_path``) followed by the
student's response ids verbatim. What differs per recipe is how the
teacher's request is composed; SDFT's processor adds the demonstration
block (``context_template``, whose default is the reference
implementation's wording) to the request's final user message, or as a new
user message when the request ends in a tool result. A harness whose
demonstrations need another layout, such as a native tool-call turn or a
block in the system prompt, subclasses the processor with its own
``teacher_request`` and names that recipe.

The ``sdft`` loss family is a thin family on the Slime backend's
distillation base (``reef/train/slime_backend/distill/``, described in
`Loss families <../../developer-guide/loss-families.rst>`__): it names
itself, sets the reference's defaults, and runs as a ``custom_loss``.
Before each step, the base's pre-train hook runs one forward pass over
every sample's teacher sequence and keeps the teacher's next-token
distribution at each response position. The loss then puts the student's
distribution at the same positions, from the training forward over the
plain request, against it.

The teacher is the model itself (``--sdft-teacher self``) and its weights
follow the reference implementation: a copy of the initial weights, kept
on the host beside the actor's own backup, that moves toward the policy by
``--sdft-teacher-update-rate`` after every step (the reference's
``ref_model_mixup_alpha``, 0.01 in its runs). The paper's
description, the model as its own teacher, is the rate of 1: no copy, the
pass runs on the actor's weights, and only the prompt differs. In practice
that setting collapses within a few steps on Science Q&A (responses grow to
the window and the KL falls to zero on degenerate text), because the teacher
drifts with the student and the demonstration stops changing its
distribution; the slow-moving copy is what keeps the signal.

Per token, the loss is the divergence over the full vocabulary. Forward KL
(teacher toward student, GKD-style) is the default, since the authors report
that the paper's results used it; the reverse KL and the generalized JSD
are switches, as is a top-K representation of the teacher. Each sample
contributes the mean over its trained response tokens, weighted by the
reference's truncated importance-sampling ratio between the policy and the
rollout engine's log-probs, so SDFT requires an inference backend that
attaches engine-native tensors.

The report contract
-------------------

A report references one inference record and carries the demonstration as
``metadata.teacher_context``. A ``score`` is optional metadata; the recipe never
trains on it. The same contract serves SDPO, where the teacher context is the
environment feedback instead of a demonstration.

.. code:: json

   {
     "references": ["<receipt of the student's request>"],
     "metadata": {"teacher_context": "<the demonstration>"}
   }

Configuration
-------------

.. config::

   batch_size | 1 | rollouts per optimizer step. Must equal the driver's ``--global-batch-size`` because each sample is its own data-parallel unit.
   tokenizer_path | required | the served model's tokenizer directory; it renders the teacher prompt with the chat template the engine applied.
   max_teacher_tokens | 0 | a report whose teacher sequence is longer is skipped and counted (``teacher_overflow_reports``); 0 disables the check. Set it to the trainer's window.
   context_template | the reference's block | the text added to the final user message, with ``{context}`` as the demonstration's placeholder.
   max_staleness | 0 | accepted lag between the producing and serving version.

The Slime driver takes ``--loss-type custom_loss``,
``--use-rollout-logprobs`` and ``--disable-compute-advantages-and-returns``,
plus the family's own flags:

.. config::

   --sdft-teacher | self | who scores the samples: ``self`` is the model itself reading the demonstration; ``separate`` another checkpoint of the same architecture (``--sdft-teacher-checkpoint``).
   --sdft-divergence | forward | ``forward`` is KL(teacher || student), ``reverse`` is KL(student || teacher), ``jsd`` the generalized Jensen-Shannon divergence (``--sdft-jsd-beta``, default 0.5, is the teacher's mixture weight).
   --sdft-top-k | 0 | keep the teacher's top-K log-probs and its log-prob at the sampled token instead of its whole distribution; 0 keeps the whole distribution.
   --sdft-teacher-update-rate | 0.01 | fraction of the current policy mixed into the teacher's weights after every step; 1 makes the current policy the teacher, 0 freezes the initial weights.
   --sdft-importance-sampling-cap | 2.0 | cap of the truncated importance-sampling weight; 0 disables the correction.
   --sdft-skip-response-tokens | 0 | response tokens at the start of every sample left out of the loss; the paper's runs used 3.

The teacher pass keeps one ``[response tokens, vocabulary / tensor parallel]``
float16 block per sample on the host between the pass and the step, about
2 GB for a 16k-token response of a 248k-vocabulary model at tensor parallel
4, so long-context deployments size ``max_teacher_tokens`` with that in mind.
A teacher copy costs the actor's weights once more on the host per rank,
in bfloat16 plus a float32 accumulator.

Related guides
--------------

- `Inference and feedback quickstart <../../getting-started/quickstart.rst>`__:
  learn the request, receipt, and report workflow.
- `Train model weights from agent feedback <../evolve-your-model.rst>`__:
  set up the GPU stack and inspect published updates.
- `Loss families <../../developer-guide/loss-families.rst>`__: how a family
  such as ``sdft`` plugs into the Slime backend, and the distillation base
  it is built on.
