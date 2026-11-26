Connect your own agent
======================

:For: developers who have an agent and want to run it through Reef
:Before you start: a Reef checkout, Python 3.12 or newer, uv, and Docker for Harbor tasks
:You will have: a custom agent that uses Reef's model binding, records its calls, and completes a scored task

Your agent owns its loop, tools, and task state. Reef supplies the model
connection and runs the agent against tasks. A Harbor verifier scores the
work. You choose whether those scores become training feedback.

Start with the `LangGraph example <../../tutorials/custom-agent/README.md>`__.
It uses Reef's existing Python adapter. It needs no new adapter class or
change to Reef's registry.

Choose an entry point
---------------------

.. list-table::
   :header-rows: 1
   :widths: 28 37 35

   * - What you have
     - What to provide
     - What Reef manages
   * - A Python agent, such as a LangGraph graph
     - A ``program/main.py`` entry point, pinned dependencies, and the
       model and log integration below
     - A harness tree that can run, be evaluated, and be versioned
   * - A Harbor agent class
     - Its ``module:Class`` import path and model connection options
     - Task runs and captured model calls; the external source remains
       part of your deployment
   * - An agent with its own CLI, configuration, or plugin format
     - An adapter package registered through ``reef.harness_adapters``
     - Rendering, running, and versioning the parts that adapter declares

Use the Python adapter for the first example. The adapter reference covers
`external adapter packages <../developer-guide/harness-adapters.rst#connect-a-new-agent>`__.
Installing such a package makes it available to the Reef processes that run
the agent; users do not have to add its name to Reef's source.

Prepare a Python harness
------------------------

Put the entry point and its modules in a source tree:

.. code:: text

   tree/
     program/
       main.py
       tools.py
     config/
       primary.json

Only ``program/main.py`` is required. Modules use one filename under
``program/``; nested packages and arbitrary resource files are not accepted
by this adapter. Optional ``config/primary.json`` holds your program's
settings. Reef renders it as ``program/config.json``.

Declare dependencies and the Python version in ``main.py`` with PEP 723
inline script metadata. Pin the packages your agent needs. Reef runs:

.. code:: bash

   uv run --script /reef-run/program/main.py '<task instruction>'

In a Harbor task, the process starts in the task's working directory.
Resolve configuration files relative to ``__file__``, not that working
directory. Write task outputs where the instruction requires them.

Use the supplied model
~~~~~~~~~~~~~~~~~~~~~~

Reef writes ``program/model.json`` for each run. This runtime file contains
``api``, ``base_url``, ``api_key``, and ``model``. Do not store it in the
source tree. The Python adapter currently supports ``api: openai``:
OpenAI Chat Completions, with ``base_url`` already ending in ``/v1``.

Pass those values to your model client. For LangChain's ``ChatOpenAI``, use
``use_responses_api=False`` to keep the client on Chat Completions. Bind
every model node to this connection. A node that uses an unrelated endpoint
bypasses Reef's records and cannot be included in its feedback.

The program keeps its own tools and graph. It does not need to import
``reef``. The example's host-side driver uses Reef to render and launch it;
the program inside the task container imports only its declared dependencies.

Keep a readable trajectory
~~~~~~~~~~~~~~~~~~~~~~~~~

Reef's proxy records model calls independently of the agent's session files.
To give the Harbor verifier an ATIF trajectory, also write one JSONL line
per completed model call under ``REEF_PROGRAM_SESSION_DIR``. Each line is
the request body with its response body in a ``response`` field:

.. code:: json

   {"model":"example-model","messages":[{"role":"user","content":"Hello"}],"response":{"choices":[{"message":{"role":"assistant","content":"Hello"}}]}}

Preserve tool calls, tool results, and their IDs in those bodies. Do not
write authentication headers. A graph's debug events alone do not satisfy
this format. The example captures non-streaming HTTP bodies before
LangChain converts them, then Reef converts the log to ATIF.

Let model errors and an exhausted step budget fail the program. Do not write
a success answer after an exception. Bound the loop and model timeout;
Harbor also applies the task's agent timeout. Inspect the trial's
``agent/command.json`` for the exit code and output.

Run the example
---------------

Install Reef as described in `development <../contributing/development.rst>`__.
From the repository root, with the environment active and Docker running:

.. code:: bash

   python tutorials/custom-agent/run.py --smoke

The smoke run uses scripted model replies. It still starts Reef, runs the
real LangGraph program inside a Harbor container, records two model calls,
executes a tool, and runs the verifier. It checks integration, not model
quality. Success ends with:

.. code:: text

   Passed: model -> multiply tool -> model -> answer.txt -> verifier reward 1.0

For a real model, set ``REEF_UPSTREAM_API_KEY`` and pass ``--upstream-url``
and ``--model`` instead. The example's `README
<../../tutorials/custom-agent/README.md>`__ gives the complete commands,
Linux networking setting, output paths, and failure checks.

Connect an existing Harbor agent
--------------------------------

If you already have a Harbor adapter, install its package in the environment
where Reef and Harbor load it. Supply its configuration to
``TaskRolloutRunner`` or the evaluation request's ``harness`` field:

.. code:: json

   {
     "import_path": "my_agent.harbor:MyAgent",
     "model_name": "{model}",
     "kwargs": {
       "base_url": "{base_url}/v1",
       "api_key": "{api_key}"
     }
   }

The constructor options above are an example: use the names your class
accepts. Reef replaces the three placeholders. ``{base_url}`` here does
not include ``/v1``. An installed agent also needs an address reachable from
its task container; a host-side agent can reach the local capture proxy.

Harbor's `custom agent guide
<https://docs.harborframework.com/agents/custom-agents>`__ defines the
``BaseAgent`` and ``BaseInstalledAgent`` lifecycle. Passing an import path
does not put the class's source into a Reef harness artifact. Use a Python
tree or an external Reef adapter when Reef must version the changing files.

Choose feedback and evolution separately
---------------------------------------

The example runs a record-only Reef deployment and sets
``TaskRolloutRunner.is_reporting=False``. It keeps calls and the verifier
score without submitting a training report. It does not publish or activate
a harness version.

For a recipe that consumes verifier rewards, the runner can submit reports
with ``is_reporting=True``. It handles the receipts and report linkage;
the agent does not need its own reporter. Reef's candidate evaluation path
keeps evaluation calls separately and does not submit training reports.
See `evaluation requests <../reference/http-api.rst#evaluate-an-artifact>`__
for that interface.

To evolve the program, configure the existing harness recipe with
``evolution.adapter: python``, a seed tree, a method, and evaluation tasks.
The ``program`` part requires human activation after an evolved change.
Follow `Evolve your harness <evolve-your-harness.rst>`__ for that workflow.

This first integration does not add task-provided MCP servers or skills to
the Python program. Its tools are defined by the agent. Persistent graph
checkpoints, human interrupts, and installation as a local client plugin
also need their own integration; the Python adapter has no ``run.layer``.
