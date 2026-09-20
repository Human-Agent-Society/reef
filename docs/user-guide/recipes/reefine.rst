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
     --inference.upstream-api-key dummy \
     --recipe.config.evolution.multimodal.api_key sk-or-...

For an OpenAI Responses or Anthropic endpoint, add
``--inference.upstream-api responses`` or ``--inference.upstream-api anthropic``.
The last line, an OpenRouter key (or ``REEF_MULTIMODAL_API_KEY``), gives the
harness image, speech, embedding and decision models (see
`Images, speech, embeddings and decisions`_); leave it out to go without them,
or when the upstream is OpenRouter.

The profile listens on ``127.0.0.1:8901``, requires no token unless ``REEF_TOKEN`` is set, and keeps
state under ``.reef/reefine/``. For custom deployments, copy
``reef/service/profiles/reefine.yaml`` and pass it with ``-c``. The
`Reefine tutorial <https://github.com/Human-Agent-Society/reef/tree/main/tutorials/reefine>`__
includes installation, bug-fix and research demos, and recorded measurements.

How it works
------------

1. Ask. In a ``reef-pi`` session, ``/evolve <what it should do>`` has
   the model think the change through in the background before anything is
   filed, while the chat shows one line (``ctrl+o`` expands the whole
   clarification) and the session keeps its input and context: when it
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
   each accepted instruction. Where the host can isolate it, a coding agent
   answers the instruction (see `The agent proposer`_): it edits the tree,
   runs the changed harness for real and hands the entries back, and a
   review call reads them against the request. Otherwise the served model
   answers it in a few calls: it writes a design first (the
   request in one sentence, what triggers the behavior, what state the
   harness must know and where it comes from, what only you can provide as
   ``requires`` items with a ``prompt`` each), then the entries, and a
   second call reviews them against the request. The review also says
   whether the entries deliver the requested behavior or only put a
   substitute in its place, such as a rule describing it. When the review
   is partial or finds a substitute, the model writes the answer again with
   the review's findings, up to three answers in all. The step keeps the
   delivering answer with the fewest uncovered points; when no answer
   delivers, it is skipped with the reason. The evaluation runs the
   candidate on the health task: it publishes when the tree still works,
   and the step's page carries the design and the review either way.
3. Result. The session that asked reports it in the chat when the step
   settles, and ``reef-pi harness ... --wait`` prints the same line:
   published as a release, ready but waiting for your review because it
   changes an extension, rejected by the evaluation, or skipped with the reason,
   followed by the points the review left uncovered.
4. Read. A release that touches a ``code_extension`` is held back from the
   served head, so it reaches no session on its own. ``/versions
   <step>`` opens its page (``reef-pi page <version>`` from the shell), which
   carries the design, the review and the numbers.
5. Install and set up. A step that settles while you are between turns offers
   its install there; otherwise run ``/versions <version> install``
   (``reef-pi update`` from the shell). A release held back from the head is
   served as part of installing it, so installing is the one decision. After
   confirmation, the session collects what the change needs: for each unmet ``requires`` item it shows the item's
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
* The agent proposer, or the served model where the host cannot isolate the
  agent, proposes skills, rules, agent commands, or pi extensions.
  Requests and update notices are enabled in the seed by default.
* The proposer is instructed to integrate new slash commands into pi's native
  ``/`` autocomplete dropdown alongside built-in commands, with descriptions,
  using prompt templates or extension command registration. The review checks
  that integration and the path from invocation to a visible result, including
  state and an off command for modes. Headless trials cannot verify the dropdown;
  the agent records any unverified interactive checks in its design.
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

The agent proposer
------------------

The text proposer writes an extension it never runs, so a model name it
guessed or a parameter a provider refuses only shows once you use the
change. The agent proposer runs the served model as a pi coding agent
instead. Its working directory holds the tree as one file per entry
(``harness/skills``, ``rules``, ``commands`` and ``extensions``, plus
``requires.json`` and ``design.md``), with Reef's own entries, the
extension API reference among them, read-only beside it. It may read
documentation on the network, and two tools of its own:

* ``harness_check`` runs the working directory through Reef's admission, as
  the step will.
* ``harness_trial`` runs the changed harness for real, online, on a task the
  agent writes, and returns the session's final text, the tools it called,
  every image, speech, embedding or decision call it made with the
  provider's error when one failed, and the end of its stderr.

While it runs, the request page's Activity lists each tool the agent calls,
each check and trial with its result, and each image or speech call with the
provider's status, so a long run shows what it is doing; opening the
``/evolve`` spinner in pi lists the latest few.

When the agent stops, its files are read back into the step's mutations,
``requires`` items and design, and the review runs as for the text
proposer. The agent's session log lands in the step record as
``agent-session.jsonl``.

The agent holds no credential. It and its trials reach models through a
loopback gateway whose address carries a random token: the served model
with the served key (always the served model, whatever a request names),
and ``/v1/images``, ``/v1/embeddings``, ``/v1/audio/speech`` and
``/v1/decisions`` on the recipe's multimodal gateway (``evolution.multimodal``,
see below), and the provider's
model list (``GET /models?modality=``, fetched with its key), so the agent picks
a model that exists without holding a key. Every call spends from
``evolution.max_model_calls_per_step`` and is recorded in the step's
``proposer.json``. Nothing else is reachable through it, Reef's own routes
included.

Isolation (``evolution.proposer_agent.sandbox``, or ``REEF_PROPOSER_SANDBOX``):

* ``bwrap`` (the default where the host can): the agent and its trials run in
  a bubblewrap jail with an empty environment, a read-only system and only
  the working directory writable, in a network namespace pasta connects to
  the internet with no host address reachable but the gateway's port. It
  needs ``bwrap`` and ``pasta`` (the ``passt`` package) and a service that
  runs as a non-root user with user namespaces allowed.
* ``e2b``: the agent and its trials run in an `E2B <https://e2b.dev>`__
  cloud sandbox, a microVM with the internet and no route to the Reef host.
  The gateway's port answers at the same loopback address inside it through
  a tunnel Reef opens from its side (a relay in the sandbox that Reef polls
  over the sandbox's public address, with a per-run secret), so it works
  from a laptop as from a server, and no other host port is reachable. The
  agent's files are copied in, refreshed for each check and trial, and copied
  back when it stops. It needs ``pip install 'reef-infra[e2b]'`` and an E2B
  key (``e2b_api_key``, else ``E2B_API_KEY``); ``e2b_template`` names the
  sandbox image, else Reef builds ``reef-pi-<version>`` (the pinned pi on
  Node 22) on first use, in about a minute. The sandbox's own user can reach
  root in it; nothing there holds a key.

  .. code:: bash

     E2B_API_KEY=e2b_... reef serve --recipe reefine \
       --inference.upstream-url https://openrouter.ai/api \
       --inference.upstream-model z-ai/glm-5.3 \
       --recipe.config.evolution.proposer_agent.sandbox e2b

* ``none``: no isolation. The agent runs with the service's user and full
  network access, fed your clients' text. Choose it only where you trust
  every client, such as your own machine.
* Left unset on a host that cannot isolate, the agent is off and the text
  proposer answers; the service logs why.

``timeout_s`` (1800) bounds the whole agent run and ``trial_timeout_s`` (300)
each trial. A run past its limit hands back no change. Set
``evolution.max_model_calls_per_step`` to bound what one request may spend:
an agent run makes a model call per turn and may probe several provider
models before it settles on one.

Images, speech, embeddings and decisions
----------------------------------------

``evolution.multimodal`` names one gateway that serves these modalities behind
a single key. Reef relays a scenario's ``/v1/images``, ``/v1/embeddings``,
``/v1/audio/speech`` and ``/v1/decisions`` to it with the key, the way the
upstream serves chat: an extension calls them at ``REEF_SERVICE_URL`` with the
scenario and token headers, in the provider's own format, and holds no provider
key. Nothing is recorded, so these calls are not learning signal. The agent
proposer's trials reach the same gateway, so an extension it proves in a trial
calls what the harness will call.

.. code:: yaml

   evolution:
     multimodal:
       preset: openrouter          # or openai-compatible (OrcaRouter, LiteLLM, ...)
       url: https://openrouter.ai/api   # the preset's address unless set; required for openai-compatible
       api_key: ${REEF_MULTIMODAL_API_KEY}   # or api_key_env: NAME

``api_key`` takes the key the way ``inference.upstream_api_key`` takes the chat
key; the profile sets it to ``${REEF_MULTIMODAL_API_KEY}``, and
``--recipe.config.evolution.multimodal.api_key`` sets it on the command line.
Empty, the upstream key serves when the upstream is the same address, so a
deployment that chats through OpenRouter needs nothing more. Without a key,
or for a route the preset does not serve (``openai-compatible`` has no
decisions), those routes answer 501. Recipes other than reefine offer none.

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
(``GET /reef/harness/releases/<step>/page``, ``reef-pi page <version>``) renders
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
