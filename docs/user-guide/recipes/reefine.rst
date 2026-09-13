Reefine
========

Reefine is the built-in recipe for refining a pi coding harness from plain
language instructions. Its implementation is
``reef.recipe.reefine:ReefineRecipe``, and its proposer and evaluator ship in
the Reef wheel, so the service needs no tutorial checkout or training GPUs.

Start the bundled profile with an OpenAI-compatible endpoint:

.. code:: bash

   reef serve --recipe reefine \
     --inference.upstream-url http://127.0.0.1:11434 \
     --inference.upstream-model gemma4:26b \
     --inference.upstream-api-key dummy

The profile listens on ``127.0.0.1:8901``, uses token ``reef-local``, and keeps
state under ``.reef/reefine/``. For custom deployments, copy
``reef/service/profiles/reefine.yaml`` and pass it with ``-c``. The
`Reefine tutorial <https://github.com/Human-Agent-Society/reef/tree/main/tutorials/reefine>`__
includes installation, bug-fix and research demos, and recorded measurements.

Behavior and configuration
--------------------------

* ``training-mode: manual`` runs one step for each accepted instruction on
  ``POST /reef/train``. Use ``hybrid`` to also learn from failing reports.
* The served model proposes skills, rules, agent commands, or pi extensions.
  Requests and update notices are enabled in the seed by default.
* ``evolution.review_kinds: [code_extension]`` holds code changes pending
  human promotion. Client requirements must pass setup before installation.
* ``evolution.selection: always`` records evaluation scores and publishes
  admitted candidates whose evaluation ran, even without score improvement.
  Code-extension review still applies.

The bundled evaluator recognizes only the profile's sieve, Fibonacci, and CSV
tasks. These measure arithmetic regressions, not whether a requested workflow
works. Set both ``evolution.tasks`` and ``evolution.evaluate`` for a different
workload, and use ``evolution.selection: score_comparison`` to require
improvement.
All ``CordisRecipe`` evolution settings remain available, including custom
proposers, seeds, execution settings, and publication policies.

``REEF_PROPOSER_TIMEOUT_S`` and ``REEF_PROPOSER_MAX_TOKENS`` override the model
call budgets. Defaults are 120 seconds and 4096 reply tokens for instructions,
60 seconds and 2048 tokens for failure-driven proposals. The tutorial's
``run.sh`` raises these to 900 seconds and 16384 tokens for its local model.

Migration
---------

The former ``tutorials/harness-requests/`` directory is now
``tutorials/reefine/``. Existing runs can retain their state by moving their
``work/`` directory and keeping the old scenario name in the driver. The
``evolve-your-harness`` tutorial's proposer and evaluator entrypoints delegate
to Reefine, so its existing configurations continue to work.
