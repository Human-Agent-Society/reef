Tutorial: evolve your agent's harness
=====================================

.. page::
   :for: anyone who wants Reef to improve the coding agent they already use
   :needs: a Reef checkout (`Laptop, no GPU <installation.rst#laptop-no-gpu>`__), Node.js with ``npm``, and an API key for a model that can call tools
   :outcome: Reef running, your agent connected to it, and a training step that Reef ran from your feedback

A coding agent's harness is everything around the model: its rules, skills,
commands and settings. Reef learns a better harness from how the agent does
its work, and gives each new version back to you. Your own configuration
stays as it is.

In this tutorial you start Reef, connect the pi agent to it, use the agent,
tell Reef how a session went, and see what Reef does with it.
The last two sections point to the guides for writing your own method and
for adding another agent.

.. flow::
   :loop: your next session starts on the new version

   Use :: you run the agent; its model calls go through Reef
   Record :: Reef keeps each call and the feedback on it
   Evolve* :: a method edits a copy of the harness
   Evaluate :: the copy and the current version run the same tasks
   Publish :: a better copy becomes the current version

What runs where
---------------

Two programs share the work. ``reef-client`` runs on your machine, next to
the agent. Reef runs as a service, on the same machine or on a server.

.. diagram::
   :caption: The client installs Reef's version beside your own configuration and sends back what a session changed. Reef records the calls, evolves the harness, and publishes versions.

   <div class="fig-lane">
     <div class="fig-group">
       <span class="fig-group-label">Your machine</span>
       <div class="fig-node">Your agent<span class="fig-node-caption">with your own settings, rules and skills</span></div>
       <div class="fig-node">reef-client<span class="fig-node-caption">installs and syncs Reef's version</span></div>
     </div>
     <div class="fig-edge">
       <span>model calls, changed files</span>
       <svg class="fig-arrow fig-arrow-next" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 12h13"/><path d="m13 7 5 5-5 5"/></svg>
       <svg class="fig-arrow fig-arrow-back" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M19 12H6"/><path d="m11 7-5 5 5 5"/></svg>
       <span>answers, new versions</span>
     </div>
     <div class="fig-group">
       <span class="fig-group-label">Reef</span>
       <div class="fig-node fig-emphasis">Method<span class="fig-node-caption">proposes a new version</span></div>
       <div class="fig-node">Releases<span class="fig-node-caption">every version, in Git</span></div>
     </div>
   </div>

Each version of the harness is a *harness tree*: a directory of text files
such as ``rules.md`` and ``skills/<name>/SKILL.md``. Reef stores every
version as a *release*. The newest one is the *head*.

Start Reef
----------

From the repository root, point Reef at your model and start the example
deployment. It evolves the rules of the pi agent.

.. code:: bash

   export REEF_UPSTREAM_URL=https://api.moonshot.ai   # without /v1
   export REEF_UPSTREAM_MODEL=kimi-k2.6
   export REEF_UPSTREAM_API_KEY=sk-...
   reef serve -c tutorials/evolve-your-harness/configs/deployment.yaml

When ``curl http://127.0.0.1:8901/healthz`` answers ``{"ok": true}``, Reef
is ready. The deployment file names the agent, the method that changes the
harness, and the tasks that every new version must pass. `Evolve your harness
<../user-guide/evolve-your-harness.rst>`__ explains each of its settings.

Connect your machine
--------------------

Pick a name for your harness. Everyone who connects to the same name shares
one harness.

.. code:: bash

   pip install reef-client
   export SCENARIO=my-agent
   reef-client harness connect $SCENARIO --reef http://127.0.0.1:8901

``connect`` prints one line when it is done:

.. code:: text

   Done. Open a new terminal and run pi.

It did three things:

- It installed the newest version of the harness in
  ``~/.local/share/reef-client/harness/my-agent``.
- It installed pi, at the version Reef tests with, if your machine did not
  have it.
- It made your ``pi`` command run through Reef. It added a ``pi`` command in
  ``~/.local/share/reef-client/bin`` and put that directory first on
  ``PATH`` in your shell's startup file.

It did not read or write your own pi configuration. ``reef-client harness
disconnect`` undoes all three. When the deployment needs a token, ``connect``
asks for it, or reads it from ``REEF_TOKEN``.

Use the agent
-------------

Open a new terminal and run ``pi`` as you always do, then give it a task.
pi loads your own settings, rules and skills, and Reef's version on top of
them. Its model calls go to Reef, which forwards them to your model and
records each one. Before pi starts, the client installs the newest version.
After pi exits, the client sends back what the session changed in Reef's
files, such as a skill that pi wrote.

Tell Reef how it went
---------------------

Reef learns from calls that have feedback. Score the session with one report
that names its model calls: ``0`` for a failure, ``1`` for a success. The
``feedback`` text says what went wrong. The commands use ``jq``.

.. code:: bash

   CALLS=$(curl -s "http://127.0.0.1:8901/reef/scenarios/$SCENARIO/records?request_type=inference&limit=100" \
     | jq -c '[.records[].agent_record_id]')
   curl -s -X POST http://127.0.0.1:8901/reef/report \
     -H "x-reef-scenario: $SCENARIO" -H "Content-Type: application/json" \
     -d "{\"references\": $CALLS, \"score\": 0, \"feedback\": \"it said the test passed without running it\"}"

The calls of one report form one trace, in order. With a deployment token,
add ``-H "Authorization: Bearer $REEF_TOKEN"`` to each ``curl``.

A program that calls the model itself reads the record id from the
``x-reef-agent-record-id`` header of each response. The fields of a report
are the ones the deployment's method reads; `Use a published tree
<../user-guide/evolve-your-harness.rst#use-a-published-tree>`__ lists other
ways to give feedback, such as uploaded tasks.

What Reef does next
-------------------

This deployment starts a training step for each report. The step does not
change the head directly. It makes a candidate and tests it first.

.. flow::

   Copy :: the method gets a copy of the head and your session
   Edit* :: it writes new rules from what went wrong
   Check :: Reef refuses files the agent cannot read
   Evaluate :: candidate and head run the same tasks
   Decide :: the better one is the head

When the candidate wins, it becomes the new head, and your next ``pi``
session starts on it. When it loses or ties, the head stays. List the steps
and their results:

.. code:: bash

   curl -s -H "x-reef-scenario: $SCENARIO" http://127.0.0.1:8901/reef/harness/releases

Each step also has a page, ``/reef/harness/releases/<step>/page``, that shows
what changed, why, and the scores. A change to code, such as a pi extension,
waits for you to promote it, because code runs in pi's process with your
permissions.

When you and Reef change the same file
--------------------------------------

You can edit Reef's files too, for example the rules in
``~/.local/share/reef-client/harness/my-agent/pi-agent/AGENTS.md``. The client
sends your edit with the next sync, and Reef merges it onto the head. Text
files merge line by line:

+---------------------------------+--------------------------------------+
| You changed                     | Result                               |
+=================================+======================================+
| nothing                         | Reef's version                       |
+---------------------------------+--------------------------------------+
| a line that Reef did not change | both changes                         |
+---------------------------------+--------------------------------------+
| the same line that Reef changed | a conflict; your file stays as it is |
+---------------------------------+--------------------------------------+

A conflict names the file. Make the line match one side and run
``reef-client harness sync``, or run ``reef-client harness sync --discard``
to take Reef's version. Code files do not merge by line: a code file that
both sides changed is a conflict.

Write your own method
---------------------

The method decides how the harness changes: a class with one operation,
``step``, which Reef calls once per training step. This deployment names the
tutorial's method, ``RulesMethod`` in
``tutorials/evolve-your-harness/harness/plugins.py``, as
``evolution.method``. It rewrites ``rules.md``, the tree path where every
adapter keeps the agent's rules; pi's adapter renders it to
``pi-agent/AGENTS.md``. The first head takes its ``rules.md`` from the seed,
``tutorials/evolve-your-harness/seeds/pi/rules.md``. `Write a harness method
<../developer-guide/write-a-harness-method.rst>`__ shows how to write your
own.

Add another agent
-----------------

Reef connects pi, Claude Code, opencode and codex. Each has an adapter: one
``adapter.yaml`` that says which of the agent's files Reef may evolve, where
the agent keeps them, and how it reaches a model. `Harness adapters
<../developer-guide/harness-adapters.rst>`__ shows how to write one, with a
checklist of what to test.

Next steps
----------

- `Evolve your harness <../user-guide/evolve-your-harness.rst>`__: the
  deployment settings, offline runs on uploaded tasks, and reviews.
- `Recipes <../user-guide/recipes.rst>`__: methods that come with Reef, such as
  GEPA and Meta-Harness.
- `HTTP API <../reference/http-api.rst>`__: every route the client and the
  service use.
