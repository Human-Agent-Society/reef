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

The recipe renders the recorded request with the served model's chat
template. It then adds the target as the assistant turn. The loss mask
selects the target tokens. All samples have the same weight. A reported
``score`` is only metadata.

How Reef implements it
----------------------

SFT has three pipeline stages
(`Training pipeline <../../developer-guide/processors.rst>`__):

- ``SFTRecord2Record`` makes each report its own record set. It never drops a
  record set unit as stale. An SFT target is fixed text, not tokens that an older model
  version sampled. Thus the stage takes no ``max_staleness`` and refuses the key.
- ``SFTTasks`` makes one ``TrajectoryItem`` from each report. The item has the
  rendered tokens and an empty ``rollout_log_probs``.
- ``SFTJobData`` puts the tasks into job data in arrival order. One job's
  data holds ``pipeline.task2job.config.job_samples`` samples. It also takes
  uploaded ATIF trajectories (see `Train on whole conversations`_).

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

A complete conversation, such as an agent episode with tool calls, can train
every agent turn at once. Upload it as an ATIF trajectory
(`Upload trajectories <../../reference/http-api.rst#upload-trajectories>`__).
``SFTTasks`` renders each uploaded trajectory into one sample:

- The steps become chat messages: an ``agent`` step is an assistant turn with
  its tool calls, and each observation result is a tool message.
- An agent step with ``is_copied_context: true``, ATIF's mark for a step
  copied from earlier context, is context only: it renders, and it trains
  nothing. Every other agent turn trains.
- The tokens are the conversation, rendered with the chat template, up to the
  last trained agent turn. Steps after it are left out.
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

Reef renders each trajectory when it accepts the upload. It refuses the whole
upload with 400, naming the trajectory, when one cannot train:

- it has no agent step that trains (all its agent steps are copied context),
  or its first trained agent step has no step before it;
- ``is_copied_context`` is not a boolean;
- the chat template does not render each turn as a prefix of the whole
  conversation. A template that removes the reasoning of earlier turns, for
  example, does not; set ``chat_template_kwargs`` so that it keeps them, or
  upload each turn as its own report;
- the sample is longer than ``max_tokens``.

Configuration
-------------

The top-level ``pipeline`` section of the deployment configures the stages.
Each stage reads only its own ``config``. ``SFTRecord2Record`` takes no
config.

.. code:: yaml

   recipe:
     implementation: reef.recipe.sft:SFTRecipe
   pipeline:
     record2task:
       config:
         tokenizer_path: /models/Qwen3-8B
         target: response
         max_tokens: 8192
     task2job:
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

   pipeline.record2task.config.tokenizer_path | required | the served model's tokenizer directory. It renders the request and the target with the engine's chat template.
   pipeline.record2task.config.target | demonstration | ``demonstration`` trains the report's ``teacher_context``. ``response`` trains the recorded response message.
   pipeline.record2task.config.max_tokens | 0 | the maximum tokens in a sample, prompt and target together. The recipe skips a longer report and counts it in ``sft_overflow_reports``. 0 disables the limit. Set it to the trainer's sequence length.
   pipeline.record2task.config.chat_template_kwargs | {} | the chat template arguments of every render, for example ``{enable_thinking: false}``. Set ``inference.handler-config.chat_template_kwargs`` to the same value, so the samples render as the engine serves.
   pipeline.task2job.config.job_samples | required | the samples of one job. The example sets it to ``training.config.global_batch_size``, so one job is one optimizer step.
   pipeline.task2job.config.buffer_size | four times ``job_samples`` | the most samples that wait for a job. Reef releases the oldest reports beyond it.
   training.config.global_batch_size | none | the samples in one optimizer step. The example passes it to Slime as ``--global-batch-size``.

A hybrid-thinking template such as Qwen3.5's or Qwen3.8's renders thinking
mode unless ``enable_thinking`` is false: Qwen3.8 opens the system prompt with
a reasoning-effort instruction, and the generation prompt opens a think block.
Targets without reasoning belong in non-thinking mode. Set the same
``chat_template_kwargs`` in both places. A ``${...}`` reference resolves to a
string, so write the mapping in each:

.. code:: yaml

   inference:
     handler-config:
       chat_template_kwargs:
         enable_thinking: false
   pipeline:
     record2task:
       config:
         chat_template_kwargs:
           enable_thinking: false

Configure the optimizer, its learning rate, and its schedule in
``training.options``. The recipe refuses an ``optimization`` section.

Slime requires ``--loss-type sft_loss``. It does not require
``--use-rollout-logprobs``. If SFT is the only method of the run, also set
``--disable-compute-advantages-and-returns``. ``sft_loss`` reads no log probs
and no advantages. The flag removes the forward pass that calculates them.

These reports raise an error:

- With ``target: demonstration``, a report without ``teacher_context``.
- With ``target: response``, a recorded inference without a response message.
- With either target, a chat template that does not render the generation
  prompt as the start of the assistant turn.

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
     task2job:
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

Train SFT for some jobs of another recipe
-----------------------------------------

A recipe can train some jobs with SFT, for example an SFT warmup before RL.
Its ``JobPlan`` puts
``TrainingMethod("reef.recipe.sft:SftObjective")`` in the ``method``
parameter of those jobs (`Switch
methods within a run
<../../developer-guide/write-a-recipe.rst#switch-methods-within-a-run>`__).
The pipeline of that recipe must make samples that the ``sft`` family can
train. The first job after the SFT phase can restart training
(``JobRequest(..., restart=True)``): it keeps the weights and starts a new
optimizer state and learning-rate schedule.

Limitations
-----------

- A report must reference exactly one recorded request. A report that
  references more than one request raises an error. To train each turn of an
  agent episode, report each request separately, or upload the episode as an
  ATIF trajectory (`Train on whole conversations`_).
- An uploaded trajectory renders no tool definitions; only its messages and
  tool calls reach the chat template.
- The recipe does not filter reports by score. To train only on accepted
  responses, report only those responses.

Related guides
--------------

- `Train model weights from agent feedback <../evolve-your-model.rst>`__:
  set up the GPU stack and examine published updates.
- `SDFT <sdft.rst>`__: learn on policy from the same demonstrations.
- `Loss families <../../developer-guide/loss-families.rst>`__: how a loss
  family connects to the Slime backend.
