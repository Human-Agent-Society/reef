SFT: train on target responses
==============================

Use supervised fine-tuning (SFT) for demonstrations, imported responses,
or uploaded conversations. Target tokens receive cross-entropy loss.
``reef.recipe.sft:SFTRecipe`` ships with Reef and supports Slime or Tinker.

Train on a dataset
------------------

Configure a runtime with `Train model weights <../evolve-your-model.rst>`__
or `Train with Tinker <../tinker.rst>`__. For recorded target responses:

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

Save one inference record and referencing report per example in
``records.jsonl``:

.. code:: jsonl

   {"agent_record_id": "sft-0001", "request_type": "inference", "payload": {"model": "served-model", "messages": [{"role": "user", "content": "What is 2 + 2?"}], "response": {"choices": [{"message": {"role": "assistant", "content": "4"}}]}}}
   {"agent_record_id": "sft-0001-report", "request_type": "report", "payload": {"references": ["sft-0001"]}}

.. code:: bash

   reef-client import records.jsonl --url http://127.0.0.1:8900 --scenario my-agent

Training starts at ``job_samples`` ready reports. An incomplete batch waits.
To repeat the dataset, import it with new record IDs.
See `Record imports
<../../reference/http-api-records.rst#import-records-for-cold-start-learning>`__
for the envelope format.

Training requirements
---------------------

``target`` defaults to ``demonstration``, which trains
``metadata.teacher_context`` instead of the recorded response.
Reports must reference exactly one request. Its selected target must exist.
Scores do not filter or weight targets. Report only responses you want to train.

Match serving and training chat templates, including thinking mode.
``max_tokens`` limits each rendered segment, including its prompt;
0 disables the limit. Over-limit samples do not train.
Targets remain eligible after new weights serve.

Train on whole conversations
----------------------------

`Upload ATIF trajectories
<../../reference/http-api-data-upload.rst#create-trajectory-items>`__ to train
multiple turns. Each trajectory counts as one stored sample but can produce
several optimizer samples. Agent turns train unless marked
``is_copied_context: true``; other roles receive no loss.

The upload API validates ATIF and rejects images for weight training.
SFT skips stored trajectories it cannot train and logs a warning.
Rendering can split a conversation into segments; if any segment exceeds
``max_tokens``, the whole trajectory is excluded.

See `Job parameters
<../../developer-guide/recipe-training-service.rst#job-parameters>`__ for
packing, checkpoints, and training different sample kinds.
