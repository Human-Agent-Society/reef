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
| Loss family | ``sft`` (Slime's ``sft_loss``)                             |
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
  set as stale. An SFT target is fixed text, not tokens that an older model
  version sampled. Thus ``max-staleness`` has no effect.
- ``SFTTasks`` makes one ``TrajectoryItem`` from each report. The item has the
  rendered tokens and an empty ``rollout_log_probs``.
- ``DistillBatches`` puts ``batch_size`` tasks into each batch.

The ``sft`` objective (``SftObjective``) sends no advantages. The ``sft`` loss
family in ``reef/train/slime_backend/sft.py`` selects Slime's ``sft_loss``.
This loss reads no old-policy log probs. Thus the objective accepts more than
one epoch on a batch.

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

A batch trains when Reef holds ``batch_size`` reports. Thus the last reports
of a dataset wait until that number of reports is available. Reef does not
repeat a dataset. To train on the data again, import it again with new
record IDs.

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
     task2batch:
       config:
         batch_size: 256
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
   pipeline.task2batch.config.batch_size | 1 | the number of samples in one batch. ``--global-batch-size`` is the number of samples in one optimizer step. Thus a batch of 256 trains four steps of 64. A last step with fewer samples also trains.

Configure the optimizer, its learning rate, and its schedule in
``training.options``. The recipe refuses an ``optimization`` section.

The Slime driver requires ``--loss-type sft_loss``. It does not require
``--use-rollout-logprobs``. If SFT is the only method of the run, also set
``--disable-compute-advantages-and-returns``. ``sft_loss`` reads no log probs
and no advantages. The flag removes the forward pass that calculates them.

These reports raise an error:

- With ``target: demonstration``, a report without ``teacher_context``.
- With ``target: response``, a recorded inference without a response message.
- With either target, a chat template that does not render the generation
  prompt as the start of the assistant turn.

Train SFT for some jobs of another recipe
-----------------------------------------

A recipe can train some jobs with SFT, for example an SFT warmup before RL.
Its ``TrainingMethodSelector`` returns
``TrainingMethod("reef.recipe.sft:SftObjective")`` for those jobs (`Switch
methods within a run
<../../developer-guide/write-a-recipe.rst#switch-methods-within-a-run>`__).
The pipeline of that recipe must make samples that the ``sft`` family can
train.

Limitations
-----------

- Only the Slime backend implements the ``sft`` family. The Tinker backend
  refuses it before training starts (`Train with Tinker <../tinker.rst>`__).
- A report must reference exactly one recorded request. A report that
  references more than one request raises an error. To train each turn of an
  agent episode, report each request separately.
- The recipe does not filter reports by score. To train only on accepted
  responses, report only those responses.

Related guides
--------------

- `Train model weights from agent feedback <../evolve-your-model.rst>`__:
  set up the GPU stack and examine published updates.
- `SDFT <sdft.rst>`__: learn on policy from the same demonstrations.
- `Loss families <../../developer-guide/loss-families.rst>`__: how a loss
  family connects to the Slime backend.
