OpenClaw-RL: learn from agent conversations
===========================================

Use `OpenClaw-RL <https://arxiv.org/abs/2603.10165>`__ to train from
conversations. A judge treats the user's next message as feedback on the
previous reply. The recipe requires Slime, GPUs, and a separately served judge.

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
The agent sends no training reports.

How it works
------------

Correlate sessions
~~~~~~~~~~~~~~~~~

Send a stable ``x-reef-tag-session`` header, unique within the scenario.
Reef orders turns by arrival. Concurrent conversations must not share a tag.

Without tags, Reef matches extending transcripts. Identical histories,
compaction, or sliding windows can break matching. Unmatched turns never
train. Reef warns about untagged traffic and expired unmatched sessions;
``prm_record_file`` includes ``correlation`` counters.

Judge and train
~~~~~~~~~~~~~~~

The next turn triggers a process reward model (PRM) judgment of acceptance
and possible hindsight hints. The objective combines PPO and distillation
from a hint-conditioned frozen base model. ``training.options`` sets
``openclawrl-w-rl: 1.0``, ``openclawrl-w-opd: 1.0``, and
``openclawrl-topk: 4`` by default.

Match serving and training `chat template arguments
<../../reference/configuration-inference.rst#chat-template-arguments>`__.
Unrenderable turns or turns without a renderable hint candidate do not train;
``unrenderable_samples`` counts them.

Configuration
-------------

Set the PRM address and tokenizer, then the job size:

.. code:: yaml

   recipe:
     implementation: recipes.openclawrl.recipe:OpenClawRLRecipe
   pipeline:
     record2sample:
       config:
         prm_url: http://127.0.0.1:23001
         prm_tokenizer_path: /models/prm
     sample2job:
       config:
         job_samples: 16

Match required ``job_samples`` to Slime's ``global-batch-size`` for one
update per job. Under ``pipeline.record2sample.config``:

.. list-table::
   :header-rows: 1

   * - Setting
     - Default
     - Meaning
   * - ``prm_url`` / ``prm_tokenizer_path``
     - empty
     - Empty URL disables judging and training; a URL requires the tokenizer.
   * - ``session_ttl_s`` / ``max_open_sessions``
     - 900 / 4096
     - Idle expiry in seconds / session limit; oldest activity is evicted first.
   * - ``prm_m`` / ``prm_temperature``
     - 3 / 0.6
     - Votes per turn / sampling temperature.
   * - ``prm_max_tokens`` / ``prm_max_hint_candidates``
     - 8192 / 3
     - Judge token budget / hint candidates per turn.
   * - ``prm_timeout_s`` / ``prm_concurrency``
     - 120 / 8
     - Timeout seconds / turns judged concurrently.

Invalid values, unknown keys, or stage settings under ``recipe.config``
stop startup. Optional ``pipeline.sample2job.config.prm_record_file`` writes
one JSON line per job; its empty default disables recording.

Reef does not manage the PRM process. The example's Compose file starts it,
sets a 3600-second timeout, and records judge counters.

Results
-------

The `recorded run and replay
<../../../recipes/openclawrl/examples/openclawrl/README.md>`__ cover 36 sessions.
A pass requires a correct first answer with the requested reasoning and formatting.
The run reaches three consecutive passes at session 14.
This is one run, not a seed-variance estimate.

Related guides
--------------

- `Train model weights <../evolve-your-model.rst>`__: configure the GPU stack.
- `Inference and feedback <../../getting-started/quickstart.rst>`__: connect an agent.
