SFT: train on target responses
==============================

Use supervised fine-tuning (SFT) to train on reported demonstrations,
imported request-response pairs, or uploaded conversations. The target tokens
receive cross-entropy loss. ``reef.recipe.sft:SFTRecipe`` ships with Reef and
supports Slime on GPUs or Tinker.

Configuration
-------------

Use ``target: response`` for the dataset example below. Set the model and
runtime using `Train model weights <../evolve-your-model.rst>`__ or
`Train with Tinker <../tinker.rst>`__. This Slime configuration trains batches
of 64 samples:

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
       Train ``teacher_context`` by default, or select ``response`` to train
       the recorded assistant response.

   ``pipeline.record2sample.config.max_tokens`` : 0
       Maximum tokens per rendered segment, including prompt and target.
       A sample with an over-limit segment is excluded and counted in
       ``unrenderable_samples``. 0 disables the limit.

   ``pipeline.sample2job.config.job_samples`` : required
       Stored samples per job. A trajectory can render as several optimizer
       samples, so a job can exceed one optimizer step.

   ``pipeline.sample2job.config.buffer_size`` : ``job_samples``
       the most samples that one job takes. Later samples wait for the next jobs.

   ``training.config.global_batch_size`` : none
       the samples in one optimizer step. The example passes it to Slime as ``--global-batch-size``.

For targets without reasoning, set ``enable_thinking: false`` on a
hybrid-thinking model. Match serving and training chat template arguments:

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

Set optimizer settings in ``training.options``; ``optimization`` is refused.
Slime requires ``sft_loss`` and refuses ``use-rollout-logprobs``.
For an SFT-only run, ``disable-compute-advantages-and-returns`` avoids an
unneeded forward pass.

Train on a dataset
------------------

Write the dataset answer as the inference record's ``response``.
Follow it with a report referencing that record. Save one envelope per JSONL
line in ``records.jsonl``:

.. code:: json

   {"agent_record_id": "sft-0001", "request_type": "inference", "payload": {"model": "served-model", "messages": [{"role": "user", "content": "What is 2 + 2?"}], "response": {"choices": [{"message": {"role": "assistant", "content": "4"}}]}}}
   {"agent_record_id": "sft-0001-report", "request_type": "report", "payload": {"references": ["sft-0001"]}}

Import the file with ``reef-client`` (`Import records for cold-start learning
<../../reference/http-api-records.rst#import-records-for-cold-start-learning>`__):

.. code:: bash

   reef-client import records.jsonl --url http://127.0.0.1:8900 --scenario my-agent

Training starts when ``job_samples`` reports are ready. A final incomplete
batch waits for more reports. Large imports remain in the record store until
training consumes them. Reef does not automatically repeat the dataset;
import it again with new record IDs to repeat it.

The report contract
-------------------

With ``target: demonstration``, reference one request and supply its target:

.. code:: json

   {
     "references": ["<receipt of the request>"],
     "metadata": {"teacher_context": "<the demonstration>"}
   }

With ``target: response``, send only
``{"references": ["<receipt of the request>"]}``.
The report marks the recorded response as ready to train.
A missing demonstration or recorded response raises an error for its target
mode. Reports that reference more than one request are also refused.

How it works
------------

Each report references one inference request. ``target`` selects what trains:

- ``demonstration`` (default): train the report's ``teacher_context`` and
  ignore the recorded response. These are the same reports that SDFT accepts.
- ``response``: train the recorded response, including its tool calls.
  Use this for imported examples with known answers.

Only the target assistant turn receives loss. Scores do not affect sample
weights or eligibility. SFT targets are fixed text, so they do not become
stale when a new artifact serves.

Reef consumes reports and uploaded trajectories in arrival order.
Slime uses ``sft_loss``; Tinker uses ``cross_entropy``. Neither requires
old-policy log probabilities, and both permit multiple epochs per job.

Train on whole conversations
----------------------------

Upload an ATIF trajectory as an item that the scenario uses
(`Create trajectory items
<../../reference/http-api-data-upload.rst#create-trajectory-items>`__).
Each trajectory counts as one stored sample.

Every agent turn trains unless it has ``is_copied_context: true``.
System, user, tool, and copied-context tokens receive no loss.
The renderer includes ``tool_definitions`` and omits steps after the last
trained turn. For example, this trajectory trains only the final answer:

.. code:: json

   {"steps": [
     {"step_id": 1, "source": "user", "message": "What is 37 + 45?"},
     {"step_id": 2, "source": "agent", "message": "82", "is_copied_context": true},
     {"step_id": 3, "source": "user", "message": "Show the sum."},
     {"step_id": 4, "source": "agent", "message": "37 + 45 = 82."}
   ]}

A trajectory can render as several segments. A new segment starts when a
turn's prompt no longer extends the previous segment's tokens. This supports
templates that remove earlier reasoning. Each segment trains only its own
new agent tokens.

Reef rejects the whole upload with HTTP 400, naming the trajectory, when:

- it has no trained agent turn, or the first trained turn has no preceding step;
- ``is_copied_context`` is not a boolean;
- a nonempty ``observation`` is not an object;
- a message or observation content part is not an object;
- a text part's ``text`` is not a string, or an image lacks ``source.path``.

During training, Reef excludes a trajectory when any segment exceeds
``max_tokens``. It also excludes one when a trained turn does not extend its
own generation prompt under the selected chat template.
``unrenderable_samples`` counts rejected original samples.

Pack samples into steps
-----------------------

By default, ``--global-batch-size`` counts samples. For batches based on
packs of tokens, set ``scheduling.packing_length`` in the job parameters:

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

- Slime groups each job's samples into packs of up to ``packing_length``
  tokens, including prompts. It never splits a sample. An oversized sample
  gets its own pack.
- One optimizer step trains ``--global-batch-size`` packs: about
  ``global_batch_size * packing_length`` tokens.
- Packing stays within each job. Its last step can be partial.
  Use larger jobs to reduce that share, or set ``scheduling.remainder: drop``.
- Packing does not change the loss. Dynamic micro-batching still uses
  ``--max-tokens-per-gpu``. Set it to at least ``packing_length``.

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
<../../reference/configuration-training.rst#task-jobs-on-slime>`__). See
`Manual training <../../reference/http-api-training.rst#manual-training>`__ for the
request and `Choose the samples that train
<../../reference/http-api-samples.rst#choose-the-samples-that-train>`__ for ``samples``.

Limitations
-----------

- The recipe does not filter by score. Report only accepted responses if
  only those responses should train.
- Reports train one recorded request each. Report separate turns individually,
  or upload an ATIF trajectory to train a complete conversation.

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
