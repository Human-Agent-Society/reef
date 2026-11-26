GEPA: reflective agent harness optimization
===========================================

Use `GEPA <https://arxiv.org/abs/2507.19457>`__ to improve rules, skills,
or commands without training model weights. A reflection model proposes
changes; task scores select a candidate for publication. Reef needs no GPU.

Run the example
---------------

The `AIME example <../../../recipes/gepa/examples/aime/README.md>`__ requires
Pi 0.84.2 and a reflection-model API key. From the repository checkout:

.. code:: bash

   cd recipes/gepa/examples/aime
   uv pip install -e .
   REEF_PI_BINARY=/path/to/pi ./run.sh --dry-run
   export OPENAI_API_KEY=...
   REEF_PI_BINARY=/path/to/pi REEF_GEPA_WORKERS=4 ./run.sh

The dry run checks setup without model calls. The live run evolves the
tree and evaluates its first and final versions on one held-out split.
Use ``--seed`` and ``--budget`` to change the search.
``REEF_GEPA_WORKERS`` defaults to 128; reduce it for memory or rate limits.

Configure the search
--------------------

Start from the example's ``gepa.yaml``:

- Match ``pipeline.sample2job.config.job_samples`` to the method's ``minibatch_size``.
- Set ``evolution.evaluation.policy: compare`` and ``statistic: mean``.
  Publication requires a strictly higher mean; the default statistic is ``wins``.
- ``evolution.method.config.components`` defaults to ``[rules, skill]``.
  Only ``text``, ``skill``, and ``command`` formats can change.

See `Harness evolution <../evolve-your-harness.rst>`__ for deployment and
the `example README <../../../recipes/gepa/examples/aime/README.md>`__ for
recorded results.
