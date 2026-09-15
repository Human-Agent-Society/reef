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

How it works
------------

1. Ask. In a ``reef-pi`` session, ``/reef-harness <what it should do>`` has
   the model think the change through before anything is filed: when it
   triggers, what state the harness must know and how it learns it, what
   you must provide. When an open point would change what gets built, it
   asks you up to three questions, each with concrete options, then files
   your original words with the answers as clarifications (``--direct`` as
   the first word files at once). From the shell, ``reef-pi harness
   "<text>"`` posts the same instruction to ``POST /reef/train``; add
   ``--wait`` to stay until the step settles. Either ask prints the
   request's page link (``GET /reef/harness/requests/<id>/page``), which
   reloads every five seconds, naming the step's state, until the result
   is on it.
2. Step. In ``training-mode: manual`` the service runs one evolve step for
   each accepted instruction. The served model writes a design first (the
   request in one sentence, what triggers the behavior, what state the
   harness must know and where it comes from, what only you can provide as
   ``requires`` items with a ``prompt`` each), then the entries, and a
   second call reviews them against the request. The evaluation runs the
   candidate on the health task: it publishes when the tree still works,
   and the step's page carries the design and the review either way.
3. Result. The session that asked reports it in the chat when the step
   settles, and ``reef-pi harness ... --wait`` prints the same line:
   published as a release, ready but waiting for your review because it
   changes an extension, rejected by the evaluation, or skipped with the reason,
   followed by the points the review left uncovered.
4. Promote. A release that touches a ``code_extension`` waits as pending.
   The session links its page (``/reef-versions <step>``, ``reef-pi page
   <step>``). The result is a non-blocking notice: keep chatting, then run
   ``/reef-versions <step> promote`` when you are ready to review it.
5. Install and set up. Run ``/reef-versions <step> install`` for a published
   release (``reef-pi update`` from the shell); an explicit promote also
   offers to install its new release. After confirmation, the session
   collects what the change needs: for each unmet ``requires`` item it shows the item's
   ``prompt`` and asks for the value of an ``env`` item, kept in
   ``.reef-harness-env`` beside the install, or for a confirmation before a
   ``permission`` or ``service`` check runs (``reef-pi setup`` asks the
   same from the shell). A release with an unmet item is never installed.
6. Reload. Type ``/reload`` in the session, or restart ``reef-pi``, and the
   new release is the harness you are talking to.

Behavior and configuration
--------------------------

* ``training-mode: manual`` runs one step for each accepted instruction on
  ``POST /reef/train``. Use ``hybrid`` to also learn from failing reports.
* The served model proposes skills, rules, agent commands, or pi extensions.
  Requests and update notices are enabled in the seed by default.
* ``evolution.review_kinds: [code_extension]`` holds code changes pending
  human promotion. Client requirements must pass setup before installation.
* ``evolution.selection: floor`` is the default: the evaluation runs the candidate
  alone and publishes it when every task scores at least
  ``evolution.floor_score`` (``1.0``). The current release is not run, and an
  episode that could not run misses the floor.
* What only you can provide (a phone number, a credential, a permission, an
  account) is a ``requires`` item, ``{name, kind, check?, prompt?}``, whose
  ``prompt`` is one sentence of at most 200 characters that setup shows when
  it asks for the item; the Setup table of the step's page has a prompt
  column. An ``env`` item's value is read at run time from
  ``process.env.NAME``: the proposer is told that an extension never asks
  you for it in the session, never stores it in a file of its own and never
  hardcodes it, and its review lists a value the extension asks for or
  stores itself as uncovered.

The health floor
----------------

The profile's one evaluation task is a health check:

.. code:: yaml

   tasks:
     - '[health] Run the shell command `echo reef-ok` with your shell tool and reply with its exact output
       as a plain word alone on the last line.'

The bundled evaluator grades the reply's last line, ``reef-ok`` exactly. The
floor answers one question: does the tree still work after the change? The
model binding answers, the shell tool runs, every extension loads. It says
nothing about whether the change does what was asked; the step's design and
review notes and the person judge that. Set both ``evolution.tasks`` and
``evolution.evaluate`` for a workload of your own, and
``evolution.selection: score_comparison`` to require the candidate to beat
the current release on them instead.

What the step records
---------------------

Every step's catalog row carries the request under
``metrics.training_request`` and the proposer's notes under
``metrics.proposal_notes``; the step page
(``GET /reef/harness/releases/<step>/page``, ``reef-pi page <step>``) renders
them:

* ``design``: the proposer's plan for the request, a few sentences, as the
  page's Design section.
* ``review``: the second call's result, ``complete`` or ``partial``, with
  the points of the request the entries cover and the ones they leave
  uncovered, as the Review section; the result line in the session and
  from ``--wait`` names the uncovered points. Absent when the review call
  failed, which never blocks the step.
* ``refused_requires``: the ``requires`` items the proposer wrote that could
  not be honored, each with the reason, under "refused by the step" in the
  Setup section. An ``env`` item whose check is a shell test is brought to
  the one variable it names first, so ``test -n "$TOKEN"`` becomes the
  variable ``TOKEN`` rather than a refusal.
* ``undeclared_env``: the variables a written extension reads through
  ``process.env`` that no ``requires`` item names. Nothing adds them; the
  page shows them so you can set them or ask for the item.
* ``failure``: why a request step produced no change: the model call
  failed (how long it took, the reply budget and the endpoint's error; a
  reply without text adds that a thinking model may have spent the budget
  on its reasoning and names ``REEF_PROPOSER_MAX_TOKENS``) or the reply
  held no usable entry. The result line in the session and from ``--wait``
  quotes it, and the page shows it as ``proposer failure`` in the Result
  section.

All ``CordisRecipe`` evolution settings remain available, including custom
proposers, seeds, execution settings, and publication policies.

``REEF_PROPOSER_TIMEOUT_S`` and ``REEF_PROPOSER_MAX_TOKENS`` override the model
call budgets. Defaults are 600 seconds and 65536 reply tokens for the call that
answers an instruction (a thinking model reasons for tens of thousands of tokens
before it writes an extension), 60 seconds and 4096 tokens for the short plan
call before it, 120 seconds and 16384 tokens for the review after it, and 60
seconds and 8192 tokens for failure-driven proposals; the reply budgets are
sized for a thinking model, which spends part of the budget on its reasoning
before the JSON (a review budget of 8192 came back empty on one). The
tutorial's ``run.sh`` sets 900 seconds and 16384 tokens for its local model.

Migration
---------

The former ``tutorials/harness-requests/`` directory is now
``tutorials/reefine/``. Existing runs can retain their state by moving their
``work/`` directory and keeping the old scenario name in the driver. The
``evolve-your-harness`` tutorial's proposer and evaluator entrypoints delegate
to Reefine, so its existing configurations continue to work.
