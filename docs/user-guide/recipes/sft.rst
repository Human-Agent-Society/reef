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
| Needs       | GPUs and the Slime training backend                        |
+-------------+------------------------------------------------------------+

What it does
------------

Each report references one recorded request. The ``target`` setting selects
the response that the sample trains:

- ``demonstration`` (default): the report's ``teacher_context``. The harness
  sends a request through Reef. It gets a reference response from a different
  source, and reports that response against the request's receipt. The recipe
  ignores the model's own response. SDFT uses the same reports, so the two
  methods can train on the same feedback.
- ``response``: the recorded response message, with its tool calls. Use this
  target for imported examples, where the recorded response is the dataset's
  answer.

The sample is the recorded request with the target as the final assistant
turn. The training component renders it with the model's chat template, and
the loss mask selects the target tokens. All samples have the same weight. A
reported ``score`` is only metadata.

How Reef implements it
----------------------

SFT has two pipeline stages
(`Training pipeline <../../developer-guide/processors.rst>`__):

- ``SFTSamples`` makes each report its own record set and makes one
  ``TrajectorySample`` from it: the report's last call as an ATIF trajectory,
  with the target as the final agent turn, the only turn that trains. The
  sample holds no tokens. ``max_tokens`` travels with it as
  ``extra.reef.max_tokens``. The stage never finds a sample stale. An SFT target
  is fixed text, not tokens that an older model version sampled. Thus the
  stage takes no ``max_staleness`` and refuses the key.
- ``SFTJobData`` puts the samples into job data in store order: reports in
  arrival order, and uploaded ATIF trajectories in upload order, interleaved
  by arrival (see `Train on whole conversations`_). With the default
  ``buffer_size``, one job's data holds
  ``pipeline.sample2job.config.job_samples`` samples.

The training component renders each sample with the model's chat template
(`Chat template arguments
<../../reference/configuration.rst#chat-template-arguments>`__). It leaves out
a sample that cannot render or that is longer than ``max_tokens``, and counts
it in the job metric ``unrenderable_samples``.

The ``sft`` objective (``SftObjective``) sends no advantages. On Slime, the
``sft`` loss family in ``reef/train/slime_backend/sft.py`` selects Slime's
``sft_loss``. On Tinker, the built-in ``sft`` family trains Tinker's
``cross_entropy`` on the target tokens (``training.backend: tinker``). Both
losses read no old-policy log probs. Thus the objective accepts more than
one epoch on a job's data.

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

A job trains when Reef holds ``job_samples`` reports. Thus the last
reports of a dataset wait until that number of reports is available. Reef
reads the imported reports only as fast as jobs train, so a large import
waits in the record store, not in the buffer. Reef does not repeat a dataset. To train on the data again, import it again
with new record IDs.

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

.. config::

   pipeline.record2sample.config.target | demonstration | ``demonstration`` trains the report's ``teacher_context``. ``response`` trains the recorded response message.
   pipeline.record2sample.config.max_tokens | 0 | the maximum tokens in a sample, prompt and target together. The training component leaves a longer sample out and counts it in ``unrenderable_samples``. 0 disables the limit. Set it to the trainer's sequence length.
   pipeline.sample2job.config.job_samples | required | the samples of one job. The example sets it to ``training.config.global_batch_size``, so one job is one optimizer step.
   pipeline.sample2job.config.buffer_size | ``job_samples`` | the most samples that one job takes. Later samples wait in the sample store for the next jobs.
   training.config.global_batch_size | none | the samples in one optimizer step. The example passes it to Slime as ``--global-batch-size``.

A hybrid-thinking template such as Qwen3.5's or Qwen3.8's renders thinking
mode unless ``enable_thinking`` is false: Qwen3.8 opens the system prompt with
a reasoning-effort instruction, and the generation prompt opens a think block.
Targets without reasoning belong in non-thinking mode. Give serving and
the training component the same chat template arguments:

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

Configure the optimizer, its learning rate, and its schedule in
``training.options``. The recipe refuses an ``optimization`` section.

Slime requires ``--loss-type sft_loss`` and refuses
``--use-rollout-logprobs``. If SFT is the only method of the run, also set
``--disable-compute-advantages-and-returns``. ``sft_loss`` reads no log probs
and no advantages. The flag removes the forward pass that calculates them.

These reports raise an error:

- With ``target: demonstration``, a report without ``teacher_context``.
- With ``target: response``, a recorded inference without a response message.

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

The recipe gives parameters to trajectory jobs only. An uploaded train task
(a task item in the ``train`` split, ``POST /reef/namespaces/{owner}/data/tasks``) trains in ``manual``
mode, for an instruction whose ``parameters`` give its job's method and
rollout. Thus one scenario runs SFT first and then RL:

1. Set the scenario to ``manual``, and create the trajectories and the tasks as items that it uses.
2. Put the trajectories in the manual buffer and send an instruction. The job
   trains them with SFT.
3. Put the tasks in the manual buffer and send an instruction with
   ``{"method": {"objective": "task_group"}, "rollout": {"rollouts_per_task": 4}}``
   in ``parameters``. The job rolls the tasks out and trains on the rollouts.

The training component must run task jobs: Tinker does, and Slime does with
its own rollout engines (`Task jobs on Slime
<../../reference/configuration.rst#task-jobs-on-slime>`__). See
`Manual training <../../reference/http-api.rst#manual-training>`__ for the
instruction and `Choose the samples that train
<../../reference/http-api.rst#choose-the-samples-that-train>`__ for the buffer.

Train SFT for some jobs of another recipe
-----------------------------------------

A recipe can train some jobs with SFT, for example its trajectory jobs, and its
task jobs with a policy objective. Its ``parameters_by_kind()``
puts ``TrainingMethod("reef.recipe.sft:SftObjective")`` in the ``method``
parameter of the jobs of that kind (`Train each kind of sample with its own
method
<../../developer-guide/write-a-recipe.rst#train-each-kind-of-sample-with-its-own-method>`__).
The pipeline of that recipe must make samples that the ``sft`` family can
train.

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
