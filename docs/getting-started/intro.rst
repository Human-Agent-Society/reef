Introduction
============

Reef is a continual learning infrastructure. It serves an inference endpoint in
front of the model your agent already calls, records what it served, accepts
feedback about it, and uses that feedback to publish a better version of the
model weights or of the agent's harness. The model and harness together form
the agent.

Nothing about the agent has to change except the base URL it sends requests to.

.. diagram::
   :caption: An agent combines a harness and a model. Reef serves their connection and can improve either component.

   <div class="fig-lane">
     <div class="fig-node">The harness<span class="fig-node-caption">runs the control loop, prompts, skills, tools, and config</span></div>
     <div class="fig-edge">
       <span>requests and feedback</span>
       <svg class="fig-arrow fig-arrow-next" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 12h13"/><path d="m13 7 5 5-5 5"/></svg>
       <svg class="fig-arrow fig-arrow-back" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M19 12H6"/><path d="m11 7-5 5 5 5"/></svg>
       <span>answers and a <strong>new harness tree</strong></span>
     </div>
     <div class="fig-node fig-emphasis">Reef<span class="fig-node-caption">serves requests, records results, trains weights, and evolves the harness</span></div>
     <div class="fig-edge">
       <span>inference and <strong>new weights</strong></span>
       <svg class="fig-arrow fig-arrow-next" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 12h13"/><path d="m13 7 5 5-5 5"/></svg>
     </div>
     <div class="fig-node">The model<span class="fig-node-caption">runs the weights: local engine or hosted API</span></div>
   </div>

Why Reef
--------

Agents accumulate feedback that nothing consumes: a tests-passed signal, a
thumbs-down, a rubric score. Turning that into a better agent normally means an
offline pipeline: export logs, build a dataset, train, evaluate, redeploy.

Reef closes that loop in the serving lifecycle for continual learning.

.. flow::
   :loop: the next request is served by the new version

   Serve :: forward the request and keep the exchange
   Record :: store the artifact version that produced the response
   Learn* :: the recipe uses records to create a candidate version
   Publish :: an accepted candidate becomes the current version

Existing inference engines and RL frameworks cover parts of that loop, but not
the whole thing:

+-------------------------------------+------------------------+----------------------------+----------+
| Ability                             | Inference engine       | RL training framework      | **Reef** |
|                                     | (vLLM, SGLang, …)      | (slime, veRL, AReaL, …)    |          |
+=====================================+========================+============================+==========+
| Serves live traffic                 | ✓                      | ✗                          | ✓        |
+-------------------------------------+------------------------+----------------------------+----------+
| Trains weights                      | ✗                      | ✓                          | ✓        |
+-------------------------------------+------------------------+----------------------------+----------+
| Versions what it served             | ✗                      | ✗                          | ✓        |
+-------------------------------------+------------------------+----------------------------+----------+
| Stays live through updates          | ✗                      | ✗                          | ✓        |
+-------------------------------------+------------------------+----------------------------+----------+
| Evolves beyond weights              | ✗                      | ✗                          | ✓        |
| (skills, harness)                   |                        |                            |          |
+-------------------------------------+------------------------+----------------------------+----------+

What Reef can evolve
--------------------

Model weights and the harness tree are the two *artifacts* Reef versions. Each
updates one of the agent's two components. The recipe you configure picks one
(composite version is a feature in the roadmap).

**Model weights.** Reef runs the training step and hot-swaps the result into the
serving engine. See `Evolve your model <../user-guide/evolve-your-model.rst>`__.

**The harness tree** is the versioned representation of the harness's mutable
rules, prompts, skills, config, and extension code. Reef proposes an edit, runs
the current and proposed versions on your tasks, and keeps the winner. See `Evolve your harness
<../user-guide/evolve-your-harness.rst>`__.

What a call looks like
----------------------

A deployment names one recipe in its config. Every scenario it creates binds
to that recipe:

.. code:: yaml

   reef:
     recipe: recipe          # default recipe; stores records only;
     upstream_url: https://api.openai.com  # redirect to the OpenAI API

`Quickstart <quickstart.rst#run-the-loop>`__ starts one on a laptop. Reef's
inference endpoint is OpenAI- and Anthropic-compatible, so a request to a served
deployment is the one you would send to the provider, plus an
``x-reef-scenario`` header:

.. code:: bash

   curl -sS -i http://127.0.0.1:8900/v1/chat/completions \
     -H "Authorization: Bearer $REEF_TOKEN" \
     -H "x-reef-scenario: hello-reef" \
     -H "Content-Type: application/json" \
     -d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}'

The response carries ``x-reef-agent-record-id``: the **receipt** naming the
stored exchange.
