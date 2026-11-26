OpenClaw-RL: learn from agent conversations
===========================================

Use `OpenClaw-RL <https://arxiv.org/abs/2603.10165>`__ to train from
conversations. A judge treats the user's next message as feedback on the
previous reply. The recipe requires Slime, GPUs, and a separately served judge.
The agent sends no training reports.

Run the example
---------------

The `homework example
<../../../recipes/openclawrl/examples/openclawrl/README.md>`__ runs 72 GSM8K
sessions with Hermes and a simulated student. It needs Docker and seven
GPUs: four for training, one each for serving, judging, and the student.

.. code:: bash

   docker build -f docker/Dockerfile.reef -t reef-openclawrl .
   hf download Qwen/Qwen3-4B-Thinking-2507 --local-dir ~/models/Qwen3-4B-Thinking-2507
   hf download Qwen/Qwen3-32B --local-dir ~/models/Qwen3-32B
   bash recipes/openclawrl/examples/openclawrl/run.sh

The script starts the stack and runs sessions in order.
The README includes one recorded run and replay; it does not estimate seed variance.

Training requirements
---------------------

Send a stable ``x-reef-tag-session`` header, unique within the scenario.
Reef orders turns by arrival. Concurrent conversations must not share a tag.
Without tags, identical histories or context compaction can break matching.
Unmatched turns never train.

Set ``prm_url`` and ``prm_tokenizer_path`` under
``pipeline.record2sample.config``. An empty URL disables judging and training.
Reef does not manage the judge process; the example's Compose file starts it.
Match ``pipeline.sample2job.config.job_samples`` to Slime's optimizer batch size.

The objective combines judged rewards and hint-conditioned distillation.
Match serving and training chat templates. Unrenderable turns or turns
without a renderable hint candidate do not train.
Use the example's configuration for judge limits and session expiry.
