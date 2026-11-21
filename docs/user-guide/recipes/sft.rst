SFT: train on target responses
==============================

Supervised fine-tuning (SFT) trains the served model to give a target
response to a recorded request. The loss is the cross-entropy of the target
tokens. Use SFT to train on demonstrations that a harness reports. You can
also use it to train on a dataset that you import. SFT ships with
``reef-infra`` as ``reef.recipe.sft:SFTRecipe``.

+-------------+------------------------------------------------------------+
| Evolves     | model weights                                              |
+-------------+------------------------------------------------------------+
| Signal      | one report for each recorded request; the target is the    |
|             | report's ``teacher_context`` or the recorded response      |
+-------------+------------------------------------------------------------+
| Loss family | ``sft`` (Slime's ``sft_loss``, Tinker's ``cross_entropy``) |
+-------------+------------------------------------------------------------+
| Package     | ``reef/recipe/sft/``                                       |
+-------------+------------------------------------------------------------+
| Pipeline    | reported feedback, singleton                               |
+-------------+------------------------------------------------------------+
| Needs       | a training backend: Slime (on GPUs) or Tinker              |
+-------------+------------------------------------------------------------+

How it works
------------

Each report references one recorded request. The ``target`` setting selects
the response that the sample trains:

- ``demonstration`` (default): the report's ``teacher_context``. The harness
  sends a request through Reef, gets a reference response from a different
  source, and reports that response against the request's receipt. The recipe
  ignores the model's own response. SDFT uses the same reports, so the two
  methods can train on the same feedback.
- ``response``: the recorded response message, with its tool calls. Use this
  target for imported examples, where the recorded response is the dataset's
  answer.

SFT has two pipeline stages
(`Training pipeline <../../developer-guide/write-a-recipe.rst>`__):

- ``SFTSamples`` makes one sample from each report: the recorded request with
  the target as the final assistant turn. Only that turn trains. All samples
  have the same weight, and a reported ``score`` is only metadata. The stage
  never finds a sample stale, because an SFT target is fixed text.
- ``SFTJobData`` puts the samples into job data in arrival order, reports and
  uploaded trajectories interleaved (`Train on whole conversations`_). With
  the default ``buffer_size``, one job's data holds ``job_samples`` samples.

The training component renders each sample with the model's chat template
(`Chat template arguments
<../../reference/configuration.rst#chat-template-arguments>`__). It leaves out
a sample that cannot render or that is longer than ``max_tokens``, and counts
it in the job metric ``unrenderable_samples``.

The ``sft`` objective is built in on Slime (``sft_loss``) and on Tinker
(``cross_entropy``, with ``training.backend: tinker``). Neither loss reads
old-policy log probs, so the objective accepts more than one epoch on a job's
data.

The report contract
-------------------

The report type is ``TeacherContextReport``. With ``target: demonstration``,
the report contains the demonstration:

.. code:: json

   {
     "references": ["<receipt of the request>"],
     "metadata": {"teacher_context": "<the demonstration>"}
   }

With ``target: response``, the report only marks the example as ready to
train: ``{"references": ["<receipt of the request>"]}``.

Train on a dataset
------------------

Write each example as an inference record. Its ``response`` is the dataset's
answer. After each inference record, write a report that references it. Put
one envelope on each line of a JSONL file:

.. code:: json

   {"agent_record_id": "sft-0001", "request_type": "inference", "payload": {"model": "served-model", "messages": [{"role": "user", "content": "What is 2 + 2?"}], "response": {"choices": [{"message": {"role": "assistant", "content": "4"}}]}}}
   {"agent_record_id": "sft-0001-report", "request_type": "report", "payload": {"references": ["sft-0001"]}}

Import the file with ``reef-client`` (`Import records for cold-start learning
<../../reference/http-api.rst#import-records-for-cold-start-learning>`__):

.. code:: bash

   reef-client import records.jsonl --url http://127.0.0.1:8900 --scenario my-agent

A job trains when Reef holds ``job_samples`` reports, so the last reports of
a dataset wait until that number is available. Reef reads the imported reports
only as fast as jobs train, so a large import waits in the record store, not
as samples. Reef does not repeat a dataset. To train on the data again, import
it again with new record IDs.

Train on whole conversations
----------------------------

A complete conversation, such as an agent rollout with tool calls, can train
every agent turn at once. Create it as a trajectory item that the scenario uses
(`Create trajectory items <../../reference/http-api.rst#create-trajectory-items>`__).
Each uploaded trajectory is one sample. The training component renders it:

- The steps become chat messages: an ``agent`` step is an assistant turn with
  its tool calls, and each observation result is a tool message.
- An agent step with ``is_copied_context: true``, ATIF's mark for a step
  copied from earlier context, is context only: it renders, and it trains
  nothing. Every other agent turn trains.
- The tokens are the conversation, rendered with the chat template and the
  agent's ``tool_definitions`` as the tools, up to the last trained agent
  turn. Steps after it are left out.
- The loss mask selects the tokens of every trained agent turn. System, user,
  tool and copied context tokens get no loss.

For example, to keep an earlier answer as context and train only the last one:

.. code:: json

   {"steps": [
     {"step_id": 1, "source": "user", "message": "What is 37 + 45?"},
     {"step_id": 2, "source": "agent", "message": "82", "is_copied_context": true},
     {"step_id": 3, "source": "user", "message": "Show the sum."},
     {"step_id": 4, "source": "agent", "message": "37 + 45 = 82."}
   ]}

Reef checks the structure of each trajectory when it accepts the upload. It
refuses the whole upload with 400, naming the trajectory, when one cannot
train:

- it has no agent step that trains (all its agent steps are copied context),
  or its first trained agent step has no step before it;
- ``is_copied_context`` is not a boolean;
- a step's ``observation`` is not empty and is not an object; a content part
  of a message or an observation result is not an object, a text part has a
  ``text`` that is not a string, or an image part has no ``source.path``.

The training component leaves a trajectory out of its job when:

- the chat template does not render each trained turn as a prefix of the
  whole conversation. A template that removes the reasoning of earlier
  turns, for example, does not. Set the component's chat template arguments
  so that it keeps them, or upload each turn as its own trajectory;
- the sample is longer than ``max_tokens``.

Configuration
-------------

The top-level ``pipeline`` section of the deployment configures the stages.
Each stage reads only its own ``config``.

.. code:: yaml

   recipe:
     implementation: reef.recipe.sft:SFTRecipe
   pipeline:
     record2sample:
       config:
         target: response
         max_tokens: 8192
     sample2job:
       config:
         job_samples: ${training.config.global_batch_size}
   training:
     backend: slime
     config:
       global_batch_size: 64
     options:
       loss-type: "sft_loss"
       global-batch-size: "${training.config.global_batch_size}"
       disable-compute-advantages-and-returns: true

.. container:: config

   ``pipeline.record2sample.config.target`` : demonstration
       ``demonstration`` trains the report's ``teacher_context``. ``response`` trains the recorded response message.

   ``pipeline.record2sample.config.max_tokens`` : 0
       the maximum tokens in a sample, prompt and target together. The training component leaves a longer sample out and counts it in ``unrenderable_samples``. 0 disables the limit. Set it to the trainer's sequence length.

   ``pipeline.sample2job.config.job_samples`` : required
       the samples of one job. The example sets it to ``training.config.global_batch_size``, so one job is one optimizer step.

   ``pipeline.sample2job.config.buffer_size`` : ``job_samples``
       the most samples that one job takes. Later samples wait for the next jobs.

   ``training.config.global_batch_size`` : none
       the samples in one optimizer step. The example passes it to Slime as ``--global-batch-size``.

Targets without reasoning belong in non-thinking mode. A hybrid-thinking
template, such as Qwen3.5's or Qwen3.8's, renders thinking mode unless
``enable_thinking`` is false. Give serving and the training component the same
chat template arguments:

.. code:: yaml

   inference:
     handler-config:
       chat_template_kwargs:
         enable_thinking: false
   training:
     options:
       apply-chat-template-kwargs: '{"enable_thinking": false}'   # Slime

On Tinker, set ``training.options.chat_template_kwargs`` to the mapping
instead.

Further rules:

- Configure the optimizer, its learning rate, and its schedule in
  ``training.options``. The recipe refuses an ``optimization`` section.
- Slime requires ``--loss-type sft_loss`` and refuses
  ``--use-rollout-logprobs``. If SFT is the only method of the run, also set
  ``--disable-compute-advantages-and-returns``; it removes a forward pass
  that ``sft_loss`` does not need.
- With ``target: demonstration``, a report without ``teacher_context`` raises
  an error. With ``target: response``, a recorded inference without a
  response message raises an error.

Pack samples into steps
-----------------------

By default, ``--global-batch-size`` counts samples. The tokens in one
optimizer step then change with the sample lengths. To count packs of tokens
instead, as ms-swift's ``--packing`` does, set ``packing_length`` in the
jobs' ``scheduling`` parameter. ``recipe.config.job_parameters`` sends it
with every job:

.. code:: yaml

   recipe:
     implementation: reef.recipe.sft:SFTRecipe
     config:
       job_parameters:
         scheduling:
           packing_length: 8192
         save_interval: 50
   pipeline:
     sample2job:
       config:
         job_samples: 4096

- Slime packs the samples of each job into packs of at most
  ``packing_length`` tokens, the prompt and the target together. It fills
  the packs best-fit decreasing and never splits a sample. A sample longer
  than ``packing_length`` is a pack of its own.
- One optimizer step trains ``--global-batch-size`` packs: about
  ``global_batch_size * packing_length`` tokens.
- Packing works within one job. A job's packs seldom fill whole steps, so
  the last step of each job is partial. Set ``job_samples`` to the samples of
  many steps, so that the partial step is a small part of the job. Or set
  ``scheduling.remainder`` to ``drop`` to leave it out.
- Packing decides only which samples share a step. Slime computes the loss
  of a step as it does without packing. It still splits a step into
  micro-batches by ``--use-dynamic-batch-size`` and ``--max-tokens-per-gpu``.
  Set ``--max-tokens-per-gpu`` to at least ``packing_length``.

``save_interval`` saves a checkpoint after each ``save_interval`` optimizer
steps of a job, and at its end. Without it, a job saves one checkpoint at its
end. Each checkpoint is a candidate that Reef can publish.

SFT, then RL on uploaded tasks
------------------------------

One scenario can run SFT first and then RL on uploaded train tasks. The
recipe gives parameters to trajectory jobs only, so a task job trains in
``manual`` mode, and its request must give the objective and the rollouts in
``parameters``. Reef refuses a task request without an objective.

1. Set the scenario to ``manual``. Create the trajectories and the tasks as
   items that it uses.
2. Send a request that names the trajectories in ``samples``. The job trains
   them with SFT.
3. Send a request that names the tasks in ``samples``, with
   ``{"objective": "my_methods.objectives:MyObjective", "rollout": {"rollouts_per_task": 4}}``
   in ``parameters``. The deployment must provide the method. The job rolls
   the tasks out and trains on the rollouts.

The training component must run both kinds of job, as Tinker does. One Slime
deployment cannot: a task job needs the ``slime`` loss family, which refuses
trajectory jobs (`Task jobs on Slime
<../../reference/configuration.rst#task-jobs-on-slime>`__). See
`Manual training <../../reference/http-api.rst#manual-training>`__ for the
request and `Choose the samples that train
<../../reference/http-api.rst#choose-the-samples-that-train>`__ for ``samples``.

Limitations
-----------

- A report must reference exactly one recorded request. A report that
  references more than one request raises an error. To train each turn of an
  agent rollout, report each request separately, or upload the rollout as an
  ATIF trajectory (`Train on whole conversations`_).
- The recipe does not filter reports by score. To train only on accepted
  responses, report only those responses.

Related guides
--------------

- `Train model weights from agent feedback <../evolve-your-model.rst>`__:
  set up the GPU stack and examine published updates.
- `SDFT <sdft.rst>`__: learn on policy from the same demonstrations.
- `Loss families <../../developer-guide/loss-families.rst>`__: how a loss
  family connects to the Slime backend.
- `Train each kind of sample with its own method
  <../../developer-guide/recipe-training-service.rst#job-parameters>`__:
  let another recipe train some of its jobs with the ``sft`` objective.
