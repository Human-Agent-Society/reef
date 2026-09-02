Evolve your harness
===================

A harness is everything around the model: the control loop, rules, prompt
templates, skills, tools, config, and extension code. Together, the model and
harness form an agent. Harness evolution improves the harness tree while the
model weights stay fixed. Reef needs no GPU for this. The model stays a fixed
endpoint, hosted or local, and the agent stays online throughout.

Reef supplies the mechanism: it snapshots the tree, applies a mutation, runs
the paired episodes, and publishes or reverts. You supply two Python
callables, ``propose`` (which edit to try) and ``evaluate`` (how an episode
scored). `Write a harness method <../developer-guide/write-a-harness-method.rst>`__ documents the
contract.

The harness tree
----------------

Reef stores the mutable, versioned files of one harness in a single object
called the tree. A tree is a flat list of entries, and each entry has three
fields: ``id`` is unique within the tree, ``name`` selects one of the node
kinds below, and ``config`` holds that kind's own fields. For the named kinds
(``agent_command``, ``skill``, ``code_extension``), ``config.name`` is the
file name the entry renders to. Five kinds are registered in
`reef/harness/nodes.py <../../reef/harness/nodes.py>`__:

+--------------------+----------------------------------------------------------+
| ``name``           | Renders as                                               |
+====================+==========================================================+
| ``config``         | a JSON object deep-merged into one of the agent's config |
|                    | files                                                    |
+--------------------+----------------------------------------------------------+
| ``rules``          | text appended to the agent's rules file                  |
+--------------------+----------------------------------------------------------+
| ``agent_command``  | a named prompt template                                  |
+--------------------+----------------------------------------------------------+
| ``skill``          | a named ``SKILL.md``                                     |
+--------------------+----------------------------------------------------------+
| ``code_extension`` | a named code file the harness loads in process           |
+--------------------+----------------------------------------------------------+

The table describes what each kind contains. Where each kind is written is
decided by an adapter, which maps every kind to a concrete file for one agent.
Reef bundles two adapters, ``pi`` and ``opencode``, each for a third-party
coding agent CLI. With the ``pi`` adapter, ``GET /reef/harness`` serves:

.. code:: text

   pi-agent/
     settings.json             <- config, target "primary"
     models.json               <- config, target "models"
     AGENTS.md                 <- rules
     prompts/<name>.md         <- agent_command
     skills/<name>/SKILL.md    <- skill
     extensions/<name>.ts      <- code_extension

The loop
--------

.. flow::
   :loop: publish the winner, or restore the snapshot

   Batch :: scored reports retained by the score window
   ``propose`` :: one proposal, a mutation or a sequence applied as one, or ``None``
   Episodes* :: run the candidate and current tree on the same tasks
   Verdict :: publish the candidate or restore the snapshot

No evolution runs while traffic flows. A report enters the *window* when it
references at least one receipt and its score is at or below ``max_score``;
the default for harness evolution keeps only failures. A report over one
receipt batches as that exchange; a report over several batches as one
trajectory sample carrying every referenced exchange in order, which is what
``reef-pi report`` sends for a whole run (``--per-receipt`` fans the score
across the receipts as separate reports instead). When ``batch_size``
window entries have accumulated, one step runs the loop once. With
``evolution.promote_failures: true`` a failing trace's prompt is added to the
gate as a permanent task, so the seed tasks are the floor of a suite that
grows from real failures and no later candidate can win while bringing one
back (the method's ``evaluate`` must score an arbitrary prompt). A prompt is
real traffic, so it meets the tree's own credential tripwire first: a prompt
carrying a key-shaped literal is never promoted, never persisted, and never
re-run as a task, and the step goes on without it. Which prompts are
promoted is the method's call: an optional ``evolution.promote`` callable
receives the step's trace samples (and the failure manifest when its
signature names ``manifest``) and returns the prompts to promote; without it
every failing trace's user prompt is promoted. Reef still dedupes, screens,
and caps whatever it returns. ``batch_size``
and ``max_score`` live under ``data:`` in the recipe config, and
``data.batch_policy: records`` drops the report requirement entirely:
recorded traffic alone batches, unscored, for methods that judge for
themselves.

A publish passes the gate as the suite stood at the time, so a suite that
keeps growing can later expose a published tree as a regression on a task the
gate had not seen yet. ``evolution.recheck_every: N`` (0, off, by default)
closes that gap: every ``N`` steps, and at once when the served model or the
adapter version has changed since the publish, the loop re-gates the last
published tree against the tree it replaced on the current suite instead of
proposing. If
the older tree now wins, the loop publishes it, which rolls the deployment
back; if the published tree still wins, nothing changes. Only the tree from
the most recent publish is kept as a rollback target, and a rollback consumes
it, so the recheck reverts one bad publish rather than walking the whole
history back.

Most of a step's cost is the evaluation. Every task runs on both trees,
``episode_repeats`` times each (once by default), which makes
``2 x len(tasks) x episode_repeats`` headless episodes, interleaved so both
sides of a pairing see the same upstream conditions. Each episode renders one
side into a throwaway root, runs the agent binary with the task as its prompt
under the ``episode_timeout_s`` limit (600 s by default), reads the
trajectory back, and deletes the root.

Each episode runs through an executor. The default ``local`` executor runs the
binary as a plain subprocess, which is right for development and the tests. A
hosted service that evaluates model-proposed trees sets ``evolution.executor:
sandbox`` so each episode runs in a bubblewrap jail (a fresh non-root
namespace, a read-only base filesystem, no host credentials, resource limits,
and no network unless a model endpoint is allowlisted); a deployment that
requires it refuses to start without the sandbox runtime.

The throwaway root contains nothing except the rendered tree: a fresh working
directory and a fresh ``HOME``, with no repository and no files from your
machine. A task must therefore state the whole problem in its prompt. A task
that refers to files the episode cannot see fails on both sides, which ties
the comparison and publishes nothing.

The edge cases resolve conservatively. A ``None`` proposal skips the step. An
episode that could not run ranks below every real score, so a candidate
cannot win on a crash, and when both sides fail the step is a tie. When the
verdict is a rejection, Reef restores the snapshot it took before the
mutation. Every verdict is recorded in the scenario's commit log together
with its mutation and both score vectors.

When it fits
------------

Harness evolution fits when the bottleneck is in the text, for example a
prompt that mishandles a task family, a missing skill, or a config default
that is wrong for the deployment. It also fits when there is no weight access
because the model is a closed endpoint, and when iteration speed matters,
since a step needs only one service and one harness binary. It does not fit
when the model itself cannot do the task.

Before you start
----------------

- ``pip install reef-client``: the loop driver imports it.
- An OpenAI-compatible endpoint serving the model under test, hosted or local.
  ``REEF_UPSTREAM_URL`` takes no ``/v1`` suffix.
- The ``pi`` binary on ``PATH``
  (``npm i -g @earendil-works/pi-coding-agent@0.84.2``); ``serve.yaml`` names
  it under ``evolution.binary``.

Run the example
---------------

From a Reef checkout:

.. code:: bash

   export REEF_UPSTREAM_API_KEY=sk-...    # only if your endpoint needs one
   cd tutorials/harness_evolve
   ./run.sh

``serve.yaml`` holds the endpoint (``http://127.0.0.1:8000``, no ``/v1``
suffix), the model (``qwen3-8b``), and the service token as literals; edit
them there to point at your own. The provider key is the one value it does
not hold.

`1_evolve_your_harness.ipynb
<../../tutorials/harness_evolve/1_evolve_your_harness.ipynb>`__ is the same
pass as a notebook, cell by cell, with the service managed as a subprocess;
its committed outputs are a full local run on ollama with no GPU.

``run.sh`` copies the recipe config out of ``serve.yaml``, starts the service, and runs
``run.py``: three exact-answer coding tasks go through Reef, each reply is
graded, and every result is reported against its receipt. Only failures enter
the window, so the first failing report triggers one evolve step. In this
example the served model is its own proposer, and it answers with one skill
mutation.

The example's scenario is ``harness-evolve-demo``. ``run.sh`` keeps the
service up only while ``run.py`` runs. When the loop finishes, it prints the
published release, the gate metrics, and the evolved ``SKILL.md``,
then stops the service.

Watch it learn
--------------

To follow the same step live, from a second terminal while ``run.sh`` is
still running:


.. code:: bash

   curl -sS -H "Authorization: Bearer reef-local" \
     -H "x-reef-scenario: harness-evolve-demo" \
     http://127.0.0.1:8900/reef/harness            # 404 until a step publishes
   curl -sS -H "Authorization: Bearer reef-local" \
     -H "x-reef-scenario: harness-evolve-demo" \
     http://127.0.0.1:8900/reef/harness/releases

One step is six episodes, three tasks on each of the two trees, and the
reference run finished in 63 s on Qwen3-8B: one failing task entered the
window, the served model proposed a new skill beside the starter, and the gate
scored the candidate 3.0 against 2.0 (1 win, 0 losses, 2 ties). The committed
notebook run repeats the arc with no GPU at all, on ollama ``qwen2.5:7b``. The run has succeeded when one
task fails, the failing report opens the window, one evolve step runs, and
``GET /reef/harness`` stops returning 404. ``/reef/harness/releases`` then
shows a published version.

If ``/reef/harness`` still returns 404 after a few minutes, the run has
failed. A missing ``pi`` binary or a server without tool calling does not
fail at config time: every episode fails, both sides tie, no candidate ever
wins, and the route stays 404. Confirm that ``pi --version`` runs and that
the server accepts tool calls before suspecting the recipe; vLLM needs
``--enable-auto-tool-choice --tool-call-parser hermes``, and without those
flags it rejects pi's ``tool_choice: "auto"`` requests with a 400 while
still answering plain requests. A missing model server does not produce
this symptom: the record phase raises on its first call and ``run.py``
exits with the upstream error before any evolve step runs.

A model that answers all three tasks correctly also leaves the route at 404,
because nothing fails, so nothing batches and no step runs. ``run.py`` prints
``every task passed: nothing batched, no evolve step runs`` when that
happens.

Install the published tree
--------------------------

Clients pull an evolved harness the way they install any coding agent:

.. code:: bash

   curl -fsS -H "Authorization: Bearer reef-local" \
     -H "x-reef-scenario: harness-evolve-demo" \
     'http://127.0.0.1:8900/reef/harness/install?adapter=pi' | bash

   reef-pi -p "fix the failing test in auth.py"
   reef-pi report --score 0 --feedback "missed the empty-token case"

The script installs the pinned agent, writes the tree, and puts a
``reef-<adapter>`` wrapper (here ``reef-pi``) on your PATH. The wrapper keeps
the receipts from a run, so ``report`` only needs the result. Pinning,
rollback, and the raw manifest routes are in `HTTP API
<../reference/http-api.rst#harness-artifacts>`__.

Write a method
--------------

Reef ships no proposer and no episode scorer. You supply ``propose``,
``evaluate``, and optionally a selection policy; `Write a harness method
<../developer-guide/write-a-harness-method.rst>`__ documents the contract, with worked examples.

Connect a different agent
-------------------------

`Harness adapters <../developer-guide/harness-adapters.rst>`__ is the descriptor reference and
how to connect an agent that has no adapter yet.
