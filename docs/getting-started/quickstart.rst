Quickstart
==========

A typical Reef workflow includes the following steps:

1. **Request.** Harness sends a request with a *scenario*.
2. **Response.** Reef forwards the response from the model to the harness, with
   a *receipt* attached to it.
3. **Report.** An entity provides a *report* containing feedback for the
   response associated with the *receipt* to Reef.
4. **Learn.** Reef runs the training / learning process against the reports in
   batches according to its *recipe*.
5. **Evolve.** Reef potentially creates a new version of the *artifact*, which
   may become immediately effective (e.g. model evolution) or requires a pull
   from the harness to take effect (e.g. harness evolution).
6. **Update.** Harness can view the history of served artifacts (*release
   chain*) and potentially pull a new harness tree.

This workflow touches a few key concepts:

- **scenario** - A specific type of workload. For example, a code reviewer, or a
  math problem solver. Reef manages the training of each scenario separately,
  i.e. records, training state and release chain.
- **receipt** - The id of a recorded inference exchange. It identifies the
  request, the response, and the release that produced it.
- **report** - A feedback quoting one or more receipts.
- **recipe** - How Reef use the report to evolve new versions of artifacts. A
  deployment serves one recipe, and every scenario on it evolves under that
  recipe.
- **artifact** - The content a release selects: model weights, or a harness
  tree of rules, prompts, skills, and config. Versioned.
- **release** - One accepted publication in a scenario. The chain of them is
  the scenario's history.

We will explore these concepts in greater depth later. For now, it's time to
get our hands dirty and experience these steps ourselves.

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

The one that runs on a laptop is ``harness_evolve``, and it takes a second
file: a **preset** naming your ``propose`` and ``evaluate`` callables and the
tasks to evaluate on. ``tutorials/harness_evolve/run.sh`` wires the whole loop
together. Run it, then read `Evolve your harness
<../user-guide/evolve-your-harness.rst>`__ for an explanation of each piece.
