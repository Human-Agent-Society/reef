Write a harness method
======================

Implement ``Method.run(job)`` to edit an agent's source tree.
Select ``HarnessEvolutionRecipe`` and ``recipe.config.evolution.method``.
Reef validates and evaluates the tree; activation decides whether it serves.

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

- ``harness/``: candidate files; use ``tree()`` and ``write_tree(tree)``.
- ``material/``: read-only traces, tasks, request and host capabilities.
- ``state/``: files retained between successful jobs.
- ``scratch/``: temporary files removed after the job.
- ``output/``: summary and notes.

Traces provide optional task, reply, score and feedback values.
``trajectory()`` returns ATIF for traffic traces, otherwise ``None``.
Files outside declared tree parts fail validation.
See the `workspace contract <../../reef/train/harness_backend/workspace.py>`__.

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

Model calls count against ``max_model_calls_per_step``.
Evaluation requires a scorer and remaining
``max_runs_per_step`` capacity; reused runs are free.
Invalid trees raise ``ValueError``. Passing one string as ``tasks`` raises
``TypeError``. Failed evaluation admission runs no tasks.

Run a coding agent
~~~~~~~~~~~~~~~~~~

Set ``runs_agent=True`` to require ``evolution.agent``.
``run_agent`` runs in the workspace through Reef's budgeted gateway and
returns ``AgentOutcome`` with exit status, timeout and error information.
Set ``runs_without_agent=True`` only when ``services.agent=None`` is supported.

Constructors receive settings only. Access models, agents and credentials
through ``job.services`` at run time.
See the `agent protocol <../../reef/train/harness_backend/agent/protocol.md>`__
for custom orchestration.

A method made of prompts
------------------------

``AgentMethod`` runs a coding agent from a prompt directory.
Provide ``START.md`` with optional ``{step}`` and ``{history}`` placeholders.
``RESUME.md`` can replace it after earlier state exists;
``rules.md`` and ``skills/`` define the coding agent's harness.
Select ``reef.train.harness_backend:AgentMethod`` and set its
``config.prompts`` directory and optional ``config.model`` binding.

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

``workspace.history()`` reads candidates, runs, scores and selection decisions.
Missing or failed scores are ``None``.
``workspace.step`` is one plus the number of ended jobs, including failures.
A retried unsettled job reuses its step. Ignore history entries with a
step greater than or equal to the current step.

Evaluation
----------

``evolution.evaluation`` selects tasks, scorer and publication policy.
Runs from ``services.evaluate`` are reused only with matching evaluation settings.
Changing those settings evaluates the head again.
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

This scorer matches the Fibonacci task below. Nonfinite scores fail.
For model judging, implement both ``score`` and ``score_with_models``.
Declare GPU requirements and load models lazily in workers.

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

Run ``reef serve -c CONFIG`` where ``methods`` is importable.
The seed must pass adapter validation. Extension changes wait for manual activation.
See `Harness evolution keys <../reference/configuration-harness.rst>`__ for configuration.

Method settings
~~~~~~~~~~~~~~~

Implement ``from_config(config)`` with a ``PluginSettings`` dataclass;
``Settings.parse`` rejects unknown keys.
Configure ``method.module`` and ``method.config``.
Named models require bindings under ``evolution.models``.

Package a method
----------------

Keep the method, settings and prompts in ``recipes/<name>/``.
Add a README and runnable ``examples/``.

Configuration failures
----------------------

Invalid deployment settings raise ``RecipeConfigError``;
invalid method settings raise ``PluginConfigError``.
Training requires a method, a scorer unless policy is ``none``, and any required coding agent.

Untrusted input
---------------

Wrap trace text with ``untrusted_text`` before inserting it into prompts.
Trust client tags as identities only when a gateway supplies them.
Require review for generated executable parts such as extensions, native
tools and hooks. Native-loop changes always require manual activation.

Keep credentials out of tree files; validation rejects them.
Model credentials come from render-time bindings or ``api_key_env``.
