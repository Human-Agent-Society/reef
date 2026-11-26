Tutorial: evolve your agent's harness
=====================================

Use feedback from a pi session to improve its rules. You need a
`Reef checkout <installation.rst#laptop-no-gpu>`__, Node.js with npm,
``jq`` and a model API key with tool-call support.

Start Reef
----------

From the repository root, set your model credentials and start the example:

.. code:: bash

   export REEF_UPSTREAM_URL=https://api.moonshot.ai   # without /v1
   export REEF_UPSTREAM_MODEL=kimi-k2.6
   export REEF_UPSTREAM_API_KEY=sk-...
   reef serve -c tutorials/evolve-your-harness/configs/deployment.yaml

Check readiness with ``curl http://127.0.0.1:8901/healthz``.
The expected response is ``{"ok": true}``.

Connect your machine
--------------------

Choose a scenario name. Clients that connect to the same scenario share its harness.

.. code:: bash

   export SCENARIO=my-agent
   reef-client harness connect $SCENARIO --reef http://127.0.0.1:8901

``connect`` installs pi if needed and configures its launcher. For authentication,
set ``REEF_TOKEN`` or enter the token when prompted.

Open a new terminal and run ``pi``. Give it a task, then exit. The launcher sends model calls to Reef and syncs your file changes.

Tell Reef how it went
---------------------

Return to the terminal where you set ``SCENARIO``. Grade the session using its last inference receipt.
Use 0 for failure or 1 for success. This example finds the latest receipt in the scenario.
Run it immediately after your session, with no other clients active.

.. code:: bash

   (
     set -e
     RECORDS="http://127.0.0.1:8901/reef/scenarios/$SCENARIO/records?request_type=inference&limit=100"
     AFTER=0
     while [ "$AFTER" != null ]; do
       PAGE=$(curl -fsS "$RECORDS&after_sequence=$AFTER")
       AFTER=$(jq '.next_after_sequence' <<< "$PAGE")
     done
     LAST=$(jq -er '.records[-1].agent_record_id' <<< "$PAGE")
     curl -fsS -X POST http://127.0.0.1:8901/reef/report \
       -H "x-reef-scenario: $SCENARIO" -H "Content-Type: application/json" \
       -d "$(jq -n --arg id "$LAST" '{references: [$id], score: 0, feedback: "the test did not run"}')"
   )

For an authenticated deployment, add
``-H "Authorization: Bearer $REEF_TOKEN"`` to each ``curl`` command.
An API client can read the receipt directly from ``x-reef-agent-record-id``.

What Reef does next
-------------------

A score below 1 prompts the method to revise ``rules.md``. Reef tests the revision against the current rules.
It keeps the revision if it wins more comparisons than it loses. A score of 1 skips the revision.

Inspect the results:

.. code:: bash

   curl -s -H "x-reef-scenario: $SCENARIO" http://127.0.0.1:8901/reef/harness/artifacts

Each step has a page at ``/reef/harness/artifacts/<step>/page``.
A code change, such as a pi extension, waits until you `activate it
<../user-guide/evolve-your-harness.rst#activate-an-inactive-artifact>`__.

When you and Reef change the same file
--------------------------------------

Edit ``~/.local/share/reef-client/harness/my-agent/plugin/rules.md`` locally.
Sync merges edits to different lines. Conflicting lines or concurrent code-file changes require manual resolution.

Resolve the conflict and run ``reef-client harness sync``, or use ``--discard`` to discard local edits.
Run ``reef-client harness disconnect`` to remove Reef's package.
