Quickstart
==========

Reef adds four things to an ordinary inference endpoint: a scenario, a receipt,
a report, and a release chain.

Two more terms matter once you want it to learn: the **recipe** and the
**artifact**, shown below.

.. flow::

   Request :: serve under a scenario and return a receipt
   Report :: feedback that quotes receipts
   Recipe* :: the method that turns reports into the next artifact
   Artifact* :: what gets versioned: weights or a harness tree
   Release chain :: the history of served artifacts

Run the loop
------------

This runs the serving half of the loop against a hosted provider, on a laptop,
with no GPU.

.. steps::

   #. **Install.** Follow the laptop path in `Installation
      <installation.rst>`__.

   #. **Serve.** ``external-provider.yaml`` is one process that
      proxies an OpenAI-compatible provider and records what it serves.

      .. code:: bash

         export REEF_TOKEN=reef-local
         export REEF_UPSTREAM_API_KEY=sk-...

         reef serve -c recipes/basic/external-provider.yaml

      The config supplies the provider, the model, and a ``.reef/`` state
      directory beside the checkout. Only the two secrets stay in the
      environment: the upstream key and the Reef token. The file is a template
      to copy, so it does not include a token.

      ``reef serve`` runs in the foreground and holds the terminal until
      Ctrl-C. Leave it running and open a second terminal for everything below.

      .. code:: bash

         curl -f http://127.0.0.1:8900/healthz     # {"ok": true}

   #. **Send a request and report on it.** The body is the provider's;
      ``x-reef-scenario`` is the only thing Reef adds. The response header
      ``x-reef-agent-record-id`` carries the **receipt**, which names the
      stored exchange. Tests, a verifier, a rubric, a thumbs-down, or any other
      grader you already have can report feedback against it.

      .. code:: python

         import httpx

         reef = httpx.Client(
             base_url="http://127.0.0.1:8900",
             headers={"Authorization": "Bearer reef-local", "x-reef-scenario": "hello-reef"},
         )

         response = reef.post(
             "/v1/chat/completions",
             json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Return exactly: reef is ready"}]},
         )
         response.raise_for_status()
         receipt = response.headers["x-reef-agent-record-id"]  # e.g. ee5aa401634b4567bf9dae21816abde4
         matched = response.json()["choices"][0]["message"]["content"].strip() == "reef is ready"

         report = reef.post(
             "/reef/report",
             json={"score": float(matched), "feedback": "matched" if matched else "wrong answer", "references": [receipt]},
         )
         print(report.json())
         # {"agent_record_id": "cce17dd7...", "scenario": "hello-reef", "request_type": "report"}

      Reports are records too, so the response carries the report's own record
      id. Only inference-record ids are receipts, and they appear only
      in ``references``.

   #. **Or use the wire client.** Install the stdlib-only ``reef-client`` with
      ``pip install reef-client``. It keeps the receipt for you.

      .. code:: python

         from reef_client import ReefClient

         client = ReefClient("http://127.0.0.1:8900", token="reef-local")

         body, receipt = client.inference_with_record(
             "hello-reef",                                   # the scenario
             "/v1/chat/completions",
             {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
         )
         answer = body["choices"][0]["message"]["content"]

         # ... whatever already judges your agent decides this score ...
         client.report("hello-reef", {"score": 1.0}, references=[receipt])

      With an OpenAI SDK instead, point ``base_url`` at
      ``http://127.0.0.1:8900/v1``, send ``x-reef-scenario`` as a default
      header, and read ``x-reef-agent-record-id`` off the raw response.

      On the wire it is an ordinary provider request with one added header, and
      the receipt comes back in a response header:

      .. code:: bash

         curl -sS -D - -o /dev/null \
           http://127.0.0.1:8900/v1/chat/completions \
           -H "Authorization: Bearer reef-local" \
           -H "x-reef-scenario: hello-reef" \
           -H "Content-Type: application/json" \
           -d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}'
         # x-reef-agent-record-id: ee5aa401634b4567bf9dae21816abde4

   #. **Read the release chain.**

      .. code:: bash

         curl -sS -H "Authorization: Bearer reef-local" \
           http://127.0.0.1:8900/reef/scenarios/hello-reef/releases
         # {"scenario": "hello-reef", "releases": [{"release_id": "...", "operation": "creation", "current": true, ...}]}

      One release, and it will stay at one: this deployment's recipe is the
      core ``recipe``, which records and trains nothing.

To make the chain advance, bind a recipe that learns. To use a weight recipe,
copy ``recipes/basic/external-provider.yaml``, set
``reef.recipe: recipes.sao.recipe:SAORecipe``, and serve the new config.
Weight recipes need GPUs (`Evolve your model
<../user-guide/evolve-your-model.rst>`__).

The one that runs on a laptop is ``harness_evolve``, and it takes a second file:
a **preset** naming your ``propose`` and ``evaluate`` callables and the tasks to
evaluate on. ``tutorials/harness_evolve/run.sh`` wires the
whole loop together. Run it, then read `Evolve your harness
<../user-guide/evolve-your-harness.rst>`__ for an explanation of each piece.



Scenario
--------

A scenario is one workload: its records, its training state, its release chain.
Scenarios never share data or updates.

The first request carrying a new ``x-reef-scenario`` creates it and binds it to
the deployment's recipe, permanently. Later requests just name it.

.. code:: bash

   curl -sS -H "Authorization: Bearer reef-local" http://127.0.0.1:8900/reef/scenarios

One deployment serves one recipe, so a request never names a method.

Receipt
-------

Every recorded exchange gets an id, and that id is the receipt. It identifies
the request, the response, and the release that produced it.

The receipt arrives in the ``x-reef-agent-record-id`` response header or in the
terminal SSE metadata for a stream. `HTTP API
<../reference/http-api.rst#inference>`__ lists the exact field for each dialect.
A stream carries it only after the record is stored.

Report
------

A report is feedback that quotes receipts. It carries ``score``, ``feedback``,
``references``, and ``metadata``; `HTTP API <../reference/http-api.rst#report>`__ gives the
types and the rules.

Whatever already decides whether your agent did well stays in your harness.
That may be a test suite, a verifier, or a human. Reports are consumed at most
once, so a retry or late arrival is never counted twice.

Release chain
-------------

Every accepted update creates a release with a parent, so a scenario's history
is a chain rather than a mutable pointer. A receipt names the release that
served it.

.. flow::
   :loop: each accepted update extends the chain

   r0 :: the starting artifact
   r1 :: first accepted update
   r2* :: current release serving requests

Durable releases are Git-backed and can be pinned or rolled back. Between
checkpoints, runtime load IDs live in engine memory; their bytes are not
restorable after a restart. `Architecture
<architecture.rst#the-release-chain>`__ has the details.

Recipe and artifact
-------------------

The **artifact** is the thing that gets versioned: model weights, or a harness
tree of rules, prompts, skills, and config.

The **recipe** is the method that produces the next one. It decides which
records are eligible, how they become a batch, what signal that batch carries,
and whether a candidate is good enough to publish. `Write a recipe
<../developer-guide/write-a-recipe.rst>`__ covers each of those decisions.

.. flow::

   Records :: requests, responses, and reports for the scenario
   Batch :: eligible records selected by the recipe
   Candidate :: a new unpublished artifact
   Version :: the published artifact now serving

Pick a cookbook recipe from `Choosing a recipe <../user-guide/recipes.rst>`__, or write one in
`Write a recipe <../developer-guide/write-a-recipe.rst>`__.
