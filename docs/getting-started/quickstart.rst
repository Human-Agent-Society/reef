Quickstart: serve inference and report feedback
===============================================

Record a model response and grade it with a hosted model and a local Reef service.
Install Reef with the `laptop instructions <installation.rst#laptop-no-gpu>`__ and set your provider API key.

Run the loop
------------

.. container:: steps

   #. **Start Reef.** In the first terminal, run:

      .. code:: bash

         export REEF_TOKEN=reef-local
         export REEF_UPSTREAM_API_KEY=sk-...
         reef serve --inference.upstream-url https://api.openai.com --inference.upstream-model gpt-4o

   #. **Create a scenario** to hold this workload's records. In a second terminal, run:

      .. code:: bash

         REEF=http://127.0.0.1:8900
         AUTH='Authorization: Bearer reef-local'
         curl -fsS "$REEF/reef/scenarios" -H "$AUTH" -H 'Content-Type: application/json' \
           -d '{"name":"hello-reef"}'
         curl -fsS "$REEF/reef/scenarios/hello-reef/inference/status" -H "$AUTH"

      Repeat the status request until ``state`` is ``ready``. If it is ``failed``, read ``error``.

   #. **Send a request and grade the answer.** Run this Python example in the Reef environment.
      The receipt identifies the exchange you are grading.

      .. code:: python

         from reef_client import ReefClient

         client = ReefClient("http://127.0.0.1:8900", token="reef-local")
         body, receipt = client.inference_with_record(
             "hello-reef", "/v1/chat/completions",
             {"model": "gpt-4o", "messages": [{"role": "user", "content": "Reply with exactly: reef is ready"}]},
         )
         answer = body["choices"][0]["message"]["content"]
         score = float(answer.strip() == "reef is ready")
         report = client.report("hello-reef", {"score": score}, references=[receipt])
         print(report["agent_record_id"])

The printed ID identifies the feedback report. The default recipe stores both records without training.
For raw HTTP requests, see `Inference and feedback <../reference/http-api-inference.rst>`__.

Choose your next step
---------------------

- `Update a coding agent's rules <harness-tutorial.rst>`__.
- `Train model weights <../user-guide/evolve-your-model.rst>`__.
