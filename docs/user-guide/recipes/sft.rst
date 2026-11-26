SFT: train on target responses
==============================

Use supervised fine-tuning (SFT) for demonstrations, imported responses,
or uploaded conversations. Target tokens receive cross-entropy loss.
``reef.recipe.sft:SFTRecipe`` ships with Reef and supports Slime or Tinker.

Configuration
-------------

Configure a runtime with `Train model weights <../evolve-your-model.rst>`__
or `Train with Tinker <../tinker.rst>`__. For the dataset workflow below,
use ``target: response``:

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
         job_samples: 64
   training:
     backend: slime
     options:
       loss-type: sft_loss
       global-batch-size: 64
       disable-compute-advantages-and-returns: true

- ``target`` defaults to ``demonstration``: train ``metadata.teacher_context``
  instead of the recorded response.
- ``max_tokens`` limits each rendered segment, including prompt and target.
  Its default, 0, disables the limit. Excluded samples count in
  ``unrenderable_samples``.
- Required ``job_samples`` counts stored samples; ``buffer_size`` defaults
  to this value. A trajectory can produce several optimizer samples.
- Slime's ``global-batch-size`` counts samples per optimizer step.
  ``use-rollout-logprobs`` and the ``optimization`` section are refused.

Match serving and training `chat template arguments
<../../reference/configuration-inference.rst#chat-template-arguments>`__.
For targets without reasoning, set ``enable_thinking: false`` on a hybrid
model. Slime uses ``apply-chat-template-kwargs``; Tinker uses
``training.options.chat_template_kwargs``.

Train on a dataset
------------------

Save one inference record and one referencing report per example in
``records.jsonl``:

.. code:: json

   {"agent_record_id": "sft-0001", "request_type": "inference", "payload": {"model": "served-model", "messages": [{"role": "user", "content": "What is 2 + 2?"}], "response": {"choices": [{"message": {"role": "assistant", "content": "4"}}]}}}
   {"agent_record_id": "sft-0001-report", "request_type": "report", "payload": {"references": ["sft-0001"]}}

Import it into the configured scenario:

.. code:: bash

   reef-client import records.jsonl --url http://127.0.0.1:8900 --scenario my-agent

Training starts at ``job_samples`` ready reports. An incomplete batch waits.
To repeat a dataset, import it again with new record IDs. See
`Record imports
<../../reference/http-api-records.rst#import-records-for-cold-start-learning>`__
for the complete envelope format.

The report contract
-------------------

Each report references exactly one inference request. For demonstrations:

.. code:: json

   {"references": ["<inference receipt>"], "metadata": {"teacher_context": "<demonstration>"}}

With ``target: response``, omit ``teacher_context``. Reef trains the recorded
response, including tool calls. Missing target content or multiple references
cause an error.

How it works
------------

Only target tokens receive loss. Targets do not become stale after publication.
Reef consumes reports and
trajectories in arrival order. Slime uses ``sft_loss`` and Tinker uses
``cross_entropy``; both permit multiple epochs per job.

Train on whole conversations
----------------------------

`Upload an ATIF trajectory
<../../reference/http-api-data-upload.rst#create-trajectory-items>`__ as an
item used by the scenario. It counts as one stored sample.
Every agent turn trains except those with ``is_copied_context: true``.
Other roles receive no loss. Rendering includes ``tool_definitions`` and
omits steps after the last trained turn.

The upload API validates ATIF and rejects images for weight training.
SFT skips stored trajectories it cannot train and logs a warning.
A trajectory needs a trained agent turn with a preceding step;
``is_copied_context`` must be boolean.

Templates that remove earlier context can produce several segments.
Each segment trains only its new agent tokens. Reef excludes the whole
trajectory if any segment exceeds ``max_tokens`` or a trained turn cannot
extend its own generation prompt. ``unrenderable_samples`` counts excluded
original samples.

Pack samples into steps
-----------------------

Set ``recipe.config.job_parameters.scheduling.packing_length`` to batch packs
of tokens instead of samples:

.. code:: yaml

   recipe:
     config:
       job_parameters:
         scheduling:
           packing_length: 8192
         save_interval: 50

Slime never splits a sample; an oversized sample gets its own pack.
One optimizer step consumes ``global-batch-size`` packs. Set
``max-tokens-per-gpu`` to at least ``packing_length``.
Packing stays within a job, whose last step can be partial.
``scheduling.remainder: drop`` discards that remainder.

``save_interval`` creates candidates at that optimizer-step interval and
at job end. Without it, the job saves only at its end.

SFT, then RL on uploaded tasks
------------------------------

Use `manual training
<../../reference/http-api-training.rst#manual-training>`__ to select trajectories
for SFT, then tasks for RL. Task requests must supply an objective and
rollout settings in ``parameters``, for example:

.. code:: json

   {"objective": "my_methods.objectives:MyObjective", "rollout": {"rollouts_per_task": 4}}

The deployment must provide that objective. Tinker supports both job kinds.
A single Slime deployment cannot: its task-loss family refuses trajectory
jobs. See `Task jobs on Slime
<../../reference/configuration-training.rst#task-jobs-on-slime>`__.

Limitations
-----------

Scores affect neither sample weights nor eligibility.
Report only accepted targets if that is your training policy.

Related guides
--------------

- `SDFT <sdft.rst>`__: learn on policy from the same demonstrations.
- `Job parameters <../../developer-guide/recipe-training-service.rst#job-parameters>`__:
  choose methods for different sample kinds.
