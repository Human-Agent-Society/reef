Write a harness method
======================

A harness method edits an agent's source tree once per training job.
Implement ``Method.run(job)`` and select it in ``evolution.method``.
Use the built-in ``HarnessEvolutionRecipe``; no custom recipe is required.

Reef supplies a workspace and services, validates the resulting tree and
applies the job's evaluation policy. A selected tree becomes a snapshot.
The deployment's activation rule then decides whether it serves.

An example: add a tool
----------------------

This method asks the served model to create a pi extension from failed traces.
The configured evaluation policy compares the resulting tree with the head:

.. code:: python

   from reef.core.prompts import untrusted_text
   from reef.harness.tree import TreePatch
   from reef.train.harness_backend import Method, MethodJob


   class AddTool(Method):
       def run(self, job: MethodJob) -> None:
           workspace = job.workspace
           failing = [trace for trace in workspace.traces() if trace.score is not None and trace.score < 1.0]
           if not failing:
               workspace.write_output("No failing traces.")
               return
           traces = "\n\n".join(f"Task: {trace.task}\nReply: {trace.reply}" for trace in failing)
           prompt = (
               "Write one pi extension that uses pi.registerTool to help with these failures. "
               "Return only the TypeScript source.\n\n" + untrusted_text(traces)
           )
           code = job.services.models.served.chat([{"role": "user", "content": prompt}])
           name = f"extensions/tool-{workspace.step}.ts"
           workspace.write_tree(workspace.tree().apply(TreePatch({name: code.strip() + "\n"})))
           workspace.write_output(f"Proposed {name}.")

Other adapters support different source formats.
Read ``workspace.host()["parts"]`` before generating files.
See `Configure a deployment`_ for the scorer and activation settings.

What Reef does with the job
----------------------------

- An unchanged tree skips the job.
- An invalid tree skips the job, records validation problems and is not retried.
- A valid changed tree enters evaluation.
- An exception fails the job. No tree or method state publishes;
  model calls remain in the job record.

The workspace
-------------

Reef keeps each job under
``<evolution.state_dir>/<scenario>/<component>/steps/<step>``:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Directory
     - Access
     - Purpose
   * - ``harness/``
     - Read/write
     - Candidate tree; use ``tree()`` and ``write_tree(tree)``.
   * - ``material/``
     - Read-only
     - ``traces()``, ``tasks()``, ``evaluation_repeats()``, ``request()`` and ``host()``.
   * - ``state/``
     - Read/write
     - Method files carried between successful jobs.
   * - ``scratch/``
     - Read/write
     - Temporary files removed when the job ends.
   * - ``output/``
     - Read/write
     - Summary and structured notes.

A ``Trace`` describes traffic or an offline run. It provides optional
``task``, ``reply``, ``score``, ``feedback`` and ``failure`` values,
plus record and client identifiers. ``trajectory()`` returns ATIF for traffic
traces, or ``None``. Untagged clients use ``untagged``.
Files outside the adapter's declared tree parts fail validation.

The services
------------

- ``models.served`` selects the model under test;
  ``models["name"]`` selects an ``evolution.models`` binding.
  ``chat(messages)`` returns text; ``complete(body)`` returns the raw response.
- ``check(tree)`` returns ``ValidationResult`` with ``passed``, ``problems``
  and ``message()``.
- ``evaluate(tree, tasks, repeats=1)`` returns scored ``RunRecord`` values.
  ``score.value`` is 0 for no answer, or ``None`` when execution failed.
  ``failure`` describes that failure.
- ``run_agent(prompt, tree=, model=)`` runs the configured coding agent.
- ``agent`` exposes its ``AgentExecution``, or ``None``.
- ``calls`` tracks the job's budget and call records.

Model calls count against ``max_model_calls_per_step`` and appear in
``calls.json``. Evaluation requires a scorer and remaining
``max_runs_per_step`` capacity; reused runs are free.
Invalid trees raise ``ValueError``. Passing one string as ``tasks`` raises
``TypeError``. Failed evaluation admission runs no tasks.

Run a coding agent
~~~~~~~~~~~~~~~~~~

Set ``runs_agent=True`` to require ``evolution.agent``.
The agent runs in the workspace and uses Reef's budgeted gateway at
``$REEF_AGENT_URL`` instead of holding model credentials.
The gateway also exposes ``GET /tools``, ``POST /check`` and ``POST /evaluate``.
For example, checking uses ``{"tree": "harness"}``.

``run_agent`` returns ``AgentOutcome``: exit code, timeout, error and duration.
For custom orchestration, ``services.agent`` exposes the descriptor, executor,
binary and per-run ``timeout_s``.
Set ``runs_without_agent=True`` only if the method also supports
``services.agent is None``.

Constructors receive settings only. Access models, agents and credentials
through ``job.services`` at run time.

A method made of prompts
------------------------

``AgentMethod`` runs a coding agent once using a prompt directory:

- ``START.md`` is required. Reef fills ``{step}`` and ``{history}``.
- ``RESUME.md`` optionally replaces it when earlier state exists.
- ``rules.md`` and ``skills/`` form the coding agent's own harness.
- ``README.md`` is for people.

.. code:: yaml

   evolution:
     method:
       module: reef.train.harness_backend:AgentMethod
       config: {prompts: methods/rules-editor, model: proposer}
     agent: {adapter: claude, sandbox: bwrap}

What the method says
--------------------

``workspace.write_output(summary, notes)`` writes ``output/summary.md`` and
``output/notes.json``. Job pages expose them as ``method_summary`` and
``method_notes``. These values do not affect selection.

Keep state between jobs
------------------------

Files in ``state/`` must be UTF-8 text totaling at most 16 MiB.
Reef commits them with the job and restores them for the next job,
including after restart. Failed jobs commit no state.

Read the history
----------------

``workspace.history()`` exposes read-only candidates, runs and decisions.
Use ``tree(checksum)`` for files, ``run_dir(run)`` for trajectories,
``scores(checksum, tasks, repeats=1)`` for latest scores and
``decision(step)`` for a job's selection result.
Missing or failed scores are ``None``.

``workspace.step`` is one plus the number of ended jobs, including failures.
A retried unsettled job reuses its step. Ignore history entries with a
step greater than or equal to the current step.

Evaluation
----------

``evolution.evaluation`` selects tasks, scorer and publication policy.
It reuses runs from ``services.evaluate`` only when their context matches:
models, adapter and binary, scorer, timeouts, residue checks and execution
settings. Changing that context evaluates the head again.
See `Publication policy
<../user-guide/evolve-your-harness.rst#evaluation-the-publication-policy>`__.

Score a run
~~~~~~~~~~~

A ``TaskScorer`` returns a finite score; higher is better:

.. code:: python

   from reef.evaluation.harness import TaskScorer
   from reef.harness.tasks.trajectory import final_assistant_text


   class ExactAnswer(TaskScorer):
       def score(self, task, result):
           text = final_assistant_text(result.trajectory) or ""
           return 1.0 if text.strip().endswith("2880067194370816120") else 0.0

This scorer is specific to the Fibonacci task below.
Use ``score_with_models(task, result, models)`` for model-based judging,
while also implementing abstract ``score``.
Declare ``execution_requirements()`` and load GPU models lazily in workers.
``VerifierRewardScorer`` reads Harbor rewards. Nonfinite scores fail.

Collect traffic prompts
~~~~~~~~~~~~~~~~~~~~~~~

With ``collect_failures=true``, Reef adds trace prompts to evaluation.
A configured ``TaskCollector.collect(samples)`` selects which prompts to keep.
Without a collector, Reef keeps every trace's prompt.

Configure a deployment
----------------------

.. code:: yaml

   recipe:
     implementation: reef.recipe.harness:HarnessEvolutionRecipe
     config:
       evolution:
         adapter: pi
         seed: methods/seed
         method: methods.add_tool:AddTool
         evaluation:
           tasks:
             - "Compute fib(90) exactly. Reply with the integer alone on the last line."
           scorer: methods.scoring:ExactAnswer
           policy: compare
   activation:
     module: always
     config:
       review_kinds: [extension]
   inference:
     upstream-url: ${REEF_UPSTREAM_URL}
     upstream-model: qwen3-8b

Run ``reef serve -c`` with the configuration path where ``methods`` is importable.
The seed must pass adapter validation. Selected extension changes wait for
manual activation because ``review_kinds`` includes ``extension``.
Use ``tutorials/evolve-your-harness/configs/`` for complete examples and
`Harness evolution keys <../reference/configuration-harness.rst#harness-evolution-keys>`__
for configuration details.

Method settings
~~~~~~~~~~~~~~~

Implement ``from_config(config)`` and parse a ``PluginSettings`` dataclass
for method-specific settings. ``Settings.parse`` rejects unknown keys.
Select it with:

.. code:: yaml

   method:
     module: methods.add_tool:AddTool
     config:
       model: proposer

The method must declare and use ``model`` for this example.
Define the binding in ``evolution.models.proposer`` with ``url``, ``model``
and ``api_key_env``.

Package a method
----------------

Keep the method, settings and prompts in ``recipes/<name>/``.
Add a README and runnable ``examples/``.

Configuration failures
----------------------

Deployment loading raises ``RecipeConfigError`` for missing evolution settings,
unknown adapters, invalid seeds or unknown review kinds.
Training refuses a missing method, a scorerless policy other than ``none``,
or a required coding agent without configuration.
Invalid method settings raise ``PluginConfigError``.

Untrusted input
---------------

Wrap trace text with ``untrusted_text`` before inserting it into prompts.
Trust client tags as identities only when a gateway supplies them.
Require review for generated executable parts such as extensions, native
tools and hooks. Native-loop changes always require manual activation.

Keep credentials out of tree files; validation rejects them.
Model credentials come from render-time bindings or ``api_key_env``.
