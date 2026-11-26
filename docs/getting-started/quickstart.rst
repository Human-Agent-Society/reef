Quickstart: serve inference and report feedback
===============================================

Start a local Reef service, send a model request and report feedback.
You need a hosted model API key. The default recipe records exchanges and feedback.
You can run it on a laptop without a GPU.

A **scenario** groups the records and state of one workload.
Each recorded model exchange has a **receipt**, which is its record ID.
A feedback **report** refers to one or more receipts.

Run the loop
------------

.. container:: steps

   #. **Install.** Follow `Laptop, no GPU <installation.rst#laptop-no-gpu>`__.
      Install ``httpx`` for the Python example below:

      .. code:: bash

         pip install httpx

   #. **Serve.** Set the provider credentials and start Reef:

      .. code:: bash

         export REEF_TOKEN=reef-local
         export REEF_UPSTREAM_API_KEY=sk-...

         reef serve \
           --inference.upstream-url https://api.openai.com \
           --inference.upstream-model gpt-4o

      Reef listens on ``127.0.0.1:8900`` and stores state in ``.reef/`` under
      the current directory. Without ``-c``, Reef reads no configuration file.
      To use another provider, change the URL and model.

      Leave the service running. Open a second terminal and check its health:

      .. code:: bash

         curl -f http://127.0.0.1:8900/healthz     # {"ok": true}

   #. **Create a scenario.** Inference requests for an unknown scenario return HTTP 404.

      .. code:: bash

         curl -sS -X POST http://127.0.0.1:8900/reef/scenarios \
           -H "Authorization: Bearer reef-local" \
           -H "Content-Type: application/json" \
           -d '{"name": "hello-reef"}'

      Check the scenario's inference status. Continue when ``state`` is ``ready``.
      If it is ``pending`` or ``starting``, repeat the check. If it is ``failed``, read ``error``.

      .. code:: bash

         curl -fsS http://127.0.0.1:8900/reef/scenarios/hello-reef/inference/status \
           -H "Authorization: Bearer reef-local"

   #. **Send a request and report feedback.** Use the provider's request body
      and add ``x-reef-scenario``. Read the receipt from
      ``x-reef-agent-record-id`` in the response headers.

      .. code:: python

         import httpx

         reef = httpx.Client(
             base_url="http://127.0.0.1:8900",
             headers={"Authorization": "Bearer reef-local", "x-reef-scenario": "hello-reef"},
             timeout=120,
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
         report.raise_for_status()
         print(report.json())
         # {"agent_record_id": "cce17dd7...", "scenario": "hello-reef", "request_type": "report"}

      The report response contains the report's record ID.
      Use inference receipts in ``references``; do not substitute report IDs.

   #. **Optional: use the client.** ``reef-client`` returns the receipt with
      the response body. Install it with ``pip install reef-client``.

      .. code:: python

         from reef_client import ReefClient

         client = ReefClient("http://127.0.0.1:8900", token="reef-local")

         body, receipt = client.inference_with_record(
             "hello-reef",                                   # the scenario
             "/v1/chat/completions",
             {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
         )
         answer = body["choices"][0]["message"]["content"]

         # Replace this score with your grader's result.
         client.report("hello-reef", {"score": 1.0}, references=[receipt])

      With the OpenAI SDK, set ``base_url`` to
      ``http://127.0.0.1:8900/v1``, send ``x-reef-scenario`` as a default
      header, and read the receipt from the raw response.

   #. **Read the artifacts.**

      .. code:: bash

         curl -sS -H "Authorization: Bearer reef-local" \
           http://127.0.0.1:8900/reef/scenarios/hello-reef/artifacts
         # {"scenario": "hello-reef", "artifacts": [{"artifact_id": "...", "operation": "creation", "active": true, ...}]}

      The list contains the initial artifact. The default ``recipe`` records
      requests and reports but does not train. A learning recipe can publish
      new artifacts.

Choose your next step
---------------------

- `Evolve a harness <harness-tutorial.rst>`__ with a hosted model.
- `Train model weights <../user-guide/evolve-your-model.rst>`__ with local GPUs or Tinker.
- `Choose a recipe <../user-guide/recipes.rst>`__ for your feedback.
- Use the `HTTP API reference <../reference/http-api.rst>`__ to connect your own agent.
